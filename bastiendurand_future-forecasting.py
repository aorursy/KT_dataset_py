import tarfile

tar = tarfile.open("../input/train-seq/train_seq.tgz", "r:gz")

tar.extractall()

tar.close()



from shutil import copyfile

copyfile(src = "../input/dataloader-seq/dataset_seq.py", dst = "../working/dataset_seq.py")
import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import copy



import dataset_seq as d_loader

import torch.utils.data as torch_d



#!pip install torchsummary

#from torchsummary import summary
BATCHSIZE=50



dataset = d_loader.Balls_CF_Seq ("mini_balls_seq", 7000)

train_dataset, test_dataset = torch_d.random_split(dataset, [int(7000*0.9), int(7000*0.1)])



dataloaders = {}

dataloaders['train'] = torch.utils.data.DataLoader(train_dataset,batch_size=BATCHSIZE, shuffle=True)

dataloaders['val'] = torch.utils.data.DataLoader(test_dataset,batch_size=BATCHSIZE, shuffle=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']

first_batch = next(iter(dataloaders["train"]))

_, bb = first_batch

bb = bb.to(device)

bb_0 = bb[0]

for i in range(bb.size(1)):

    plt.xlim(0, 100)

    plt.ylim(100, 0)

    plt.axes()

    for j in range(bb.size(2)):

        rectangle = plt.Rectangle((bb_0[i][j][0].item(),bb_0[i][j][1].item()), 

                                  bb_0[i][j][2].item()-bb_0[i][j][0].item(), 

                                  bb_0[i][j][3].item()-bb_0[i][j][1].item(),

                                  fc=COLORS[j],ec="black")

        plt.gca().add_patch(rectangle)

    plt.gca().set_aspect('equal')

    print("TIME = ", i+1)

    plt.show()
class myLSTM(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, lstm_layers, batch_first, output_size):

        super(myLSTM, self).__init__()

        self.output_size = output_size

        self.lstm_layers = lstm_layers

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first = batch_first)

        #self.fc = nn.Linear(hidden_dim, output_size)

        self.fc = nn.Sequential(

          nn.Linear(hidden_dim, 500),

          nn.ReLU(),

          nn.Linear(500, output_size),

        )



    def forward(self, x, hidden):

        out, hidden = self.lstm(x, hidden)

        fc_input = out[-1]

        output = self.fc(fc_input)

        return output

    

    def init_hidden(self, batch_size):

        hidden = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(device)

        cell = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(device)

        return (hidden, cell)
def intersect(box_a, box_b):

    """ We resize both tensors to [A,B,2] without new malloc:

    [A,2] -> [A,1,2] -> [A,B,2]

    [B,2] -> [1,B,2] -> [A,B,2]

    Then we compute the area of intersect between box_a and box_b.

    Args:

      box_a: (tensor) bounding boxes, Shape: [A,4].

      box_b: (tensor) bounding boxes, Shape: [B,4].

    Return:

      (tensor) intersection area, Shape: [A,B].

    """

    A = box_a.size(0)

    B = box_b.size(0)

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),

                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),

                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)

    return inter[:, :, 0] * inter[:, :, 1]
def IOU(box_a, box_b):

    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap

    is simply the intersection over union of two boxes.  Here we operate on

    ground truth boxes and default boxes.

    E.g.:

        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    Args:

        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]

        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]

    Return:

        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]

    """

    inter = intersect(box_a, box_b)

    area_a = ((box_a[:, 2]-box_a[:, 0]) *

              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]

    area_b = ((box_b[:, 2]-box_b[:, 0]) *

              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter

    return inter / union  # [A,B]
def customLoss(outputs, groundtruth_bb):

    bb = groundtruth_bb.view(-1, 9*4)

    iou = 0

    bb = bb.view(-1, 9, 4)

    outputs = outputs.view(-1, 9, 4)

    for i in range(outputs.size(0)):

        iou += IOU(outputs[i], bb[i])

    iou = iou / outputs.size(0)

    return (1 - iou.mean())
def calcLoss (net, dataloader, customLoss, mse):

    correct = 0

    iou = 0

    with torch.no_grad():

        for data in dataloader:

            _, bb= data

            hidden_0 = net.init_hidden(bb.size(0))

            bb = bb.view(bb.size(0), bb.size(1), 9*4)

            bb = bb.permute(1, 0, 2)

            bb_input = bb[:-1]

            #bb_input = bb_input.permute(1, 0, 2) #if batch_first = true

            bb_input = bb_input.to(device)

            bb_output = bb[-1]

            bb_output = bb_output.to(device)

            outputs = net(bb_input, hidden_0)

            correct += (customLoss(outputs, bb_output) + mse(outputs, bb_output)) * outputs.size(0)

    return (correct / len(dataloader.dataset))
def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):

    train_accuracies = []

    valid_accuracies = []

    

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = calcLoss(model, dataloaders["val"], customLoss, criterion)

    print('Initial val loss: {:.4f}'.format(best_loss))



    for epoch in range(1, num_epochs+1):

        print('Epoch {}/{}'.format(epoch, num_epochs))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for _ , bb in dataloaders[phase]:

                hidden_0 = model.init_hidden(bb.size(0))

                bb = bb.view(bb.size(0), bb.size(1), 9*4)

                bb = bb.permute(1, 0, 2)

                bb_input = bb[:-1]

                bb_input = bb_input.to(device)

                bb_output = bb[-1]

                bb_output = bb_output.to(device)

                

                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(bb_input, hidden_0)

                    loss = customLoss(outputs, bb_output) + criterion(outputs, bb_output)

                    #loss = criterion(outputs, bb_output)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                        optimizer.step()



                # statistics

                running_loss += loss.item() * bb_output.size(0)

            if phase == 'train' and scheduler != None:

                scheduler.step()



            epoch_loss = running_loss / (len(dataloaders[phase].dataset))

            

            if phase == 'train':

                train_accuracies.append(epoch_loss)

            else:

                valid_accuracies.append(epoch_loss)



            print('{} Loss: {:.4f} '.format(

                phase, epoch_loss))



            # deep copy the model

            if phase == 'val' and epoch_loss < best_loss:

                best_loss = epoch_loss

                best_model_wts = copy.deepcopy(model.state_dict())



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model, train_accuracies, valid_accuracies
input_dim = 9*4

hidden_dim = 1000

lstm_layers = 1 

batch_first = False



model = myLSTM(input_dim, hidden_dim, lstm_layers, batch_first, input_dim).to(device)



print(model)
criterion = torch.nn.MSELoss()



optimizer = optim.Adam(model.parameters(), 0.0001)
EPOCH_NUMBER = 30



model, train_accs, val_accs = train_model(dataloaders, model, criterion, optimizer, None,

                       num_epochs=EPOCH_NUMBER)



torch.save(model.state_dict(), "./model")
f = plt.figure(figsize=(10, 8))

plt.plot(train_accs, label='training loss')

plt.plot(val_accs, label='validation loss')

plt.legend()

plt.show()
first_batch = next(iter(dataloaders["val"]))

_, bb = first_batch

bb = bb.view(bb.size(0), bb.size(1), 9*4)

bb = bb.permute(1, 0, 2)

bb_input = bb[:-1]

bb_input = bb_input.to(device)



bb_pred = model(bb_input, model.init_hidden(bb.size(1)))

bb_pred = bb_pred.view(-1, 9, 4)

bb_groundtruth = bb[-1].view(-1, 9, 4)



for i in range(5):

    plt.xlim(0, 100)

    plt.ylim(100, 0)

    plt.axes()

    for j in range(bb_groundtruth.size(1)):

        rectangle_groundtruth = plt.Rectangle((bb_groundtruth[i][j][0].item(),bb_groundtruth[i][j][1].item()), 

                                  bb_groundtruth[i][j][2].item()-bb_groundtruth[i][j][0].item(), 

                                  bb_groundtruth[i][j][3].item()-bb_groundtruth[i][j][1].item(),

                                  fc=COLORS[j],ec="black")

        rectangle_pred = plt.Rectangle((bb_pred[i][j][0].item(),bb_pred[i][j][1].item()), 

                                  bb_pred[i][j][2].item()-bb_pred[i][j][0].item(), 

                                  bb_pred[i][j][3].item()-bb_pred[i][j][1].item(),

                                  fc=COLORS[j],ec="red")

        

        plt.gca().add_patch(rectangle_groundtruth)

        plt.gca().add_patch(rectangle_pred)

    plt.gca().set_aspect('equal')

    print("TIME = 20")

    plt.show()
!rm -rf mini_balls_seq
#SOME TESTS

"""

input_dim = 9*4

hidden_dim = 10

lstm_layers = 1 

batch_first = False



#lstm_layer = nn.LSTM(input_dim, hidden_dim, lstm_layers)#, batch_first=True)

lstm_layer = myLSTM(input_dim, hidden_dim, lstm_layers, batch_first, input_dim)



seq_len = 20



_, bb  = next(iter(dataloaders["val"]))

bb = bb.view(bb.size(0), bb.size(1), 9*4)#bb.view(BATCHSIZE, seq_len, input_dim)

bb = bb.permute(1, 0, 2)

bb = bb[:-1]

print(bb.shape)



hidden_state_0 = torch.randn(lstm_layers, BATCHSIZE, hidden_dim)

cell_state_0 = torch.randn(lstm_layers, BATCHSIZE, hidden_dim)

hidden = (hidden_state_0, cell_state_0)



print(bb.shape, hidden[0].shape)



out = lstm_layer(bb, hidden)

#out = out.permute(1,0,2)

#fc_input = out[-1]

#print(fc_input.shape)



print("Output shape: ", out.shape)

print("Output : ", out[:5])

print(calcLoss(lstm_layer, dataloaders["val"], customLoss, torch.nn.MSELoss()))

#print("Hidden: ", len(hidden))

"""