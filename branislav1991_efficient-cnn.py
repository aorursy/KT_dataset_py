import cv2
from itertools import zip_longest
import json
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
RUN_EVALUATION = False
DO_PLOT = True
NUM_ITERS = 20
N_EPOCHS = 300
EMBEDDING_DIM = 128
LR = 0.003
EPSILON = 0.01
ROT_AUG = False
FLIP_AUG = False
IO_CONSISTENCY_CHECK = True
TESTTIME_FLIP_AUG = False
BACKGROUND = 0

BASIC_PATH = Path('../input/abstraction-and-reasoning-challenge/')
SUBMISSION_PATH = Path('../input/abstraction-and-reasoning-challenge/')

TRAIN_PATH = BASIC_PATH / 'training'
EVAL_PATH = BASIC_PATH / 'evaluation'
TEST_PATH = BASIC_PATH / 'test'
SAMPLE_SUBMISSION_PATH = SUBMISSION_PATH / 'sample_submission.csv'
SUBMISSION_PATH = 'submission.csv'

if RUN_EVALUATION:
    train_task_files = sorted(os.listdir(TRAIN_PATH))
    train_tasks = []
    for task_file in train_task_files:
        with open(str(TRAIN_PATH / task_file), 'r') as f:
            task = json.load(f)
            train_tasks.append(task)

    eval_task_files = sorted(os.listdir(EVAL_PATH))
    eval_tasks = []
    for task_file in eval_task_files:
        with open(str(EVAL_PATH / task_file), 'r') as f:
            task = json.load(f)
            eval_tasks.append(task)
    eval_tasks = eval_tasks[:100]

test_task_files = sorted(os.listdir(TEST_PATH))
test_tasks = []
for task_file in test_task_files:
    with open(str(TEST_PATH / task_file), 'r') as f:
        task = json.load(f)
        test_tasks.append(task)
cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
cnorm = colors.Normalize(vmin=0, vmax=9)

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

def make_one_hot(labels, C=2):
    one_hot = torch.Tensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_().float().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def img2tensor(img):
    correct_img = img.copy() # do this because of the occasional neagtive strides in the numpy array
    return torch.tensor(correct_img, dtype=torch.long)[None,None,:,:]

def resize(images, size):
    if images.shape[2:] == size:
        return images

    new_images = []
    for i in range(images.shape[0]):
        image = images[i,0,:,:].cpu().numpy()
        image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_NEAREST)
        image = img2tensor(image)
        new_images.append(image)
    return torch.cat(new_images)

def collate(batch):
    tensors = list(zip(*batch))
    batch = (torch.cat(t) for t in tensors)
    return batch

def rot_aug(task):
    rotated_datasets = []
    for tt in task['train']:
        for k in range(1,4):
            it = np.rot90(np.array(tt['input']), k).tolist()
            ot = np.rot90(np.array(tt['output']), k).tolist()
            rotated_datasets.append({'input': it, 'output': ot})
    
    task['train'].extend(rotated_datasets)
    return task

def check_consistency(task):
    cons_colors = [True] * 10
    for tt in task['train']:
        inp = np.array(tt['input'])
        out = np.array(tt['output'])
        if inp.shape[0] != out.shape[0] or inp.shape[1] != out.shape[1]:
            return False, False
        for i in range(10):
            if np.any(out[inp==i] != i):
                cons_colors[i] = False
    return cons_colors

def copy_bg_fg(pred, input, colors):
    for i in range(len(colors)):
        if colors[i]:
            pred[input==i] = i
    
    return pred

def visualize_results_transformer(states, in_states, labels=[]):
    if DO_PLOT:
        out_states = states.detach()
        if len(states.shape) == 4:
            out_states = out_states[:,:10,:,:]
            out_states = torch.argmax(out_states, dim=1)
        in_states = in_states[:,:10,:,:]

        n_rows = 2 if labels == [] else 3
        fig, axes = plt.subplots(n_rows, len(states), squeeze=False)
        for c, zi in enumerate(zip_longest(out_states, in_states, labels)):
            o, i, l = zi
            viz_in_sample = torch.argmax(i, dim=0).cpu()
            axes[0,c].imshow(o.cpu(), cmap=cmap, norm=cnorm)
            axes[1,c].imshow(viz_in_sample, cmap=cmap, norm=cnorm)
            if n_rows > 2:
                axes[2,c].imshow(l.cpu(), cmap=cmap, norm=cnorm)
        plt.show()
output_size = None # this is used to store the most likely output size of the test dataset
class ARCDataset(Dataset):
    def __init__(self, task, mode='train'):
        '''We use GA predictions also to predict the shape of the output'''
        self.task = task
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.task['train'])
        else:
            return len(self.task['test'])
    
    def __getitem__(self, idx):
        global output_size
        if self.mode == 'train':
            in_out = [(self.task['train'][idx]['input'], self.task['train'][idx]['output'])]
        elif self.mode == 'eval':
            in_out = [(self.task['test'][idx]['input'], self.task['test'][idx]['output'])]
        else:
            in_out = [(self.task['test'][idx]['input'], [])]

        image = torch.cat([img2tensor(img[0]) for img in in_out])

        if self.mode == 'train' or self.mode == 'eval':
            label = torch.cat([img2tensor(img[1]) for img in in_out])

            # save size to use it for test set
            if self.mode == 'train':
                output_size = label.shape[2:]

            if FLIP_AUG: # flip augmentation
                label_fh = label.clone().flip(2)
                label_fv = label.clone().flip(3)
                label = torch.cat([label, label_fh, label_fv], dim=0)

        else:
            n_labels = 3*image.shape[0] if FLIP_AUG else image.shape[0]
            label = torch.tensor([]).view(1,1,1,-1) # no label for testing
            label = label.expand((n_labels,-1,-1,-1))

        image = resize(image, size=output_size)

        if FLIP_AUG: # flip augmentation
            image_fh = image.clone().flip(2)
            image_fv = image.clone().flip(3)
            image = torch.cat([image, image_fh, image_fv], dim=0)

        image = make_one_hot(image, C=10).float()

        label = label.squeeze(1)
        return image.cuda(), label.cuda()


class CAModel(nn.Module):
    def __init__(self):
        super().__init__()

        # embedding calculated from input
        self.embed_in = nn.ModuleList([nn.Conv2d(10, EMBEDDING_DIM, 3, padding=1) for _ in range(3)])

        self.embed_out = nn.ModuleList([nn.Conv2d(EMBEDDING_DIM, 10, 1) for _ in range(3)])
        for i in range(3):
            nn.init.constant_(self.embed_out[i].weight, 0.0)
            nn.init.constant_(self.embed_out[i].bias, 0.0)

        self.dropout = nn.Dropout2d(p=0.1)
        self.norm1 = nn.InstanceNorm2d(EMBEDDING_DIM)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Conv2d(EMBEDDING_DIM, EMBEDDING_DIM, 1)

    def forward(self, state_grid, n_iters):
        color_grid = state_grid[:,:10,:,:]
        for it in range(n_iters): # iterate for random number of iterations
            update_grid = self.embed_in[0](color_grid)
            update_grid = F.relu(update_grid)
            if update_grid.shape[2] > 1 or update_grid.shape[3] > 1:
                update_grid = self.norm1(update_grid)
            update_grid = self.dropout(update_grid)

            # SENet
            squeezed = self.squeeze(update_grid)
            squeezed = self.excite(squeezed)
            squeezed = torch.sigmoid(squeezed)

            update_grid = update_grid * squeezed

            update_grid = self.embed_out[0](update_grid)
            update_grid = self.dropout(update_grid)

            color_grid = color_grid + update_grid

        return color_grid
def train_task_st(task, test_if_solved=False):
    if IO_CONSISTENCY_CHECK: # we check if the background or foreground stays the same in input and target
        cons_colors = check_consistency(task)

    if ROT_AUG: # perform rotation augmentation; for each training dataset, rotate it by 90 degrees
        task = rot_aug(task)

    train_set = ARCDataset(task, mode='train')
    train_loader = DataLoader(train_set, batch_size=1, num_workers=0, collate_fn=collate)

    test_mode = 'eval' if test_if_solved else 'test'
    test_set = ARCDataset(task, mode=test_mode)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, collate_fn=collate)

    model = CAModel().cuda()

    print('Training...')
    model.train()
    optimizer = Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(N_EPOCHS):
        for i, train_batch in enumerate(train_loader):
            in_states, labels = train_batch

            optimizer.zero_grad()

            states = in_states.clone().detach()
            states = model(states, NUM_ITERS)
            total_loss = loss_fn(states, labels)

            # predict output from output to improve stability
            labels_oh = make_one_hot(labels.unsqueeze(1), C=10)
            labels_oh = model(labels_oh, NUM_ITERS)
            stability_loss = loss_fn(labels_oh, labels)
            total_loss += 2 * stability_loss

            total_loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print("Epoch: {0}: Loss: {1:3f}".format(epoch, total_loss.cpu().item()))

        if (epoch+1) % N_EPOCHS == 0:
            visualize_results_transformer(states[:10], in_states[:10], labels[:10])


    print('Testing...')
    is_solved = True
    output_samples = []

    model.eval()
    for test_batch in test_loader:
        in_states, labels = test_batch
        label = labels[0]

        states = in_states.clone()
        states = model(states, NUM_ITERS)

        if TESTTIME_FLIP_AUG: # revert augmentations
            states[1] = states[1].flip(1)
            states[2] = states[2].flip(2)
            out_state = torch.mean(states[:,:10,:,:], dim=0)
        else:
            out_state = states[0,:10,:,:]
        
        in_state = in_states[0,:10,:,:]
        out_state_am = torch.argmax(out_state, dim=0)

        if IO_CONSISTENCY_CHECK:
            out_state_am = copy_bg_fg(out_state_am, torch.argmax(in_state, dim=0), cons_colors)

        visualize_results_transformer(out_state_am.unsqueeze(0), in_state.unsqueeze(0))
        if test_if_solved and not torch.equal(label, out_state_am):
            is_solved = False
        
        output_samples.append(out_state_am.cpu().tolist())

    return output_samples, is_solved
if RUN_EVALUATION:
    # make predictions on training set
    #n_solved = 0
    #for idx, task in enumerate(train_tasks):
    #    print("TRAIN TASK " + str(idx + 1))
    #    is_solved = False
    #    _, is_solved = train_task_st(task, test_if_solved=True)

    #    if is_solved:
    #        print('Solved training task {0}'.format(idx+1))
    #    n_solved += int(is_solved)

    #print('Solved {0} training tasks'.format(n_solved))

    # make predictions on evaluation set
    n_solved = 0
    for idx, task in enumerate(eval_tasks):
        print("EVAL TASK " + str(idx + 1))
        is_solved = False
        _, is_solved = train_task_st(task, test_if_solved=True)

        if is_solved:
            print('Solved evaluation task {0}'.format(idx+1))
        n_solved += int(is_solved)

    print('Solved {0} evaluation tasks'.format(n_solved))
if not RUN_EVALUATION:
    # make predictions on test set
    test_predictions = []
    for idx, task in enumerate(test_tasks):
        print("TASK " + str(idx + 1))
        test_predictions.extend(train_task_st(task)[0])
        
    # Make submission
    str_test_predictions = []
    for idx, pred in enumerate(test_predictions):
        pred = flattener(pred)
        str_test_predictions.append(pred)
        
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH, index_col='output_id')
    sample_sub["output"] = str_test_predictions

    sample_sub.to_csv('submission.csv')