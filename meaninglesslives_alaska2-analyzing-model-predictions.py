! git clone https://github.com/dwgoon/jpegio

# Once downloaded install the package

!pip install jpegio/.

import jpegio as jio

!rm -rf jpegio

!pip install -q efficientnet_pytorch

from efficientnet_pytorch import EfficientNet
import cv2

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset

from albumentations import ToFloat, Compose

from albumentations.pytorch import ToTensor

from efficientnet_pytorch import EfficientNet

import matplotlib.pyplot as plt

import os

import pandas as pd

import random

import jpegio as jio

from tqdm.notebook import tqdm





# # Seed everything

seed = 42

print(f'setting everything to seed {seed}')

random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
data_dir = '../input/alaska2-image-steganalysis'

folder_names = ['JMiPOD/', 'JUNIWARD/', 'UERD/']

class_names = ['Cover', 'JMiPOD_75', 'JMiPOD_90', 'JMiPOD_95', 

               'JUNIWARD_75', 'JUNIWARD_90', 'JUNIWARD_95',

                'UERD_75', 'UERD_90', 'UERD_95']

class_labels = { name: i for i, name in enumerate(class_names) }

print(class_labels)

print('75, 90, 95 refers to JPEG image quality')
class Net(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        # 1280 is the number of neurons in last layer. is diff for diff. architecture

        self.dense_output = nn.Linear(1280, num_classes)



    def forward(self, x):

        feat = self.model.extract_features(x)

        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)

        return self.dense_output(feat)

    

device = 'cuda'

# device = 'cpu'

model = Net(num_classes=len(class_labels)).to(device)

fn = '../input/alaska2trainvalsplit/val_loss_6.08_auc_0.875.pth'

# pretrained model in my pc. now i will train on all images for 2 epochs

model.load_state_dict(torch.load(fn, map_location=device))
class Alaska2Dataset(Dataset):



    def __init__(self, df, augmentations=None):



        self.data = df

        self.augment = augmentations



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        fn, label = self.data.loc[idx]

        im = cv2.imread(fn)[:, :, ::-1]

        if self.augment:

            # Apply transformations

            im = self.augment(image=im)

        return im, label, fn.split('/')[-1]

    

val_df = pd.read_csv('../input/alaska2trainvalsplit/alaska2_val_df.csv')

val_df.sample(5)
batch_size = 16

num_workers = 4

AUGMENTATIONS_TEST = Compose([

    #     Resize(img_size, img_size, p=1), # does nothing if it's alread 512.

    ToFloat(max_value=255),

    ToTensor()

], p=1)

valid_dataset = Alaska2Dataset(val_df.sample(1000).reset_index(drop=True),

                               augmentations=AUGMENTATIONS_TEST)



valid_loader = torch.utils.data.DataLoader(valid_dataset,

                                           batch_size=batch_size,

                                           num_workers=num_workers,

                                           shuffle=False)
# https://www.kaggle.com/anokas/weighted-auc-metric-updated

from sklearn import metrics

def alaska_weighted_auc(y_true, y_valid):

    tpr_thresholds = [0.0, 0.4, 1.0]

    weights = [2,   1]



    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)



    # size of subsets

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])



    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

    normalization = np.dot(areas, weights)



    competition_metric = 0

    for idx, weight in enumerate(weights):

        y_min = tpr_thresholds[idx]

        y_max = tpr_thresholds[idx + 1]

        mask = (y_min < tpr) & (tpr < y_max)

        # pdb.set_trace()



        x_padding = np.linspace(fpr[mask][-1], 1, 100)



        x = np.concatenate([fpr[mask], x_padding])

        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

        y = y - y_min  # normalize such that curve starts at y=0

        score = metrics.auc(x, y)

        submetric = score * weight

        best_subscore = (y_max - y_min) * weight

        competition_metric += submetric



    return competition_metric / normalization
tk1 = tqdm(valid_loader, total=int(len(valid_loader)))

model.eval()

running_loss = 0

y, preds = [], []

val_loss = []

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():

    for (im, labels, _) in tk1:

        inputs = im["image"].to(device, dtype=torch.float)

        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        y.extend(labels.cpu().numpy().astype(int))

        preds.extend(F.softmax(outputs, 1).cpu().numpy())

        running_loss += loss.item()

        tk1.set_postfix(loss=(loss.item()))



    epoch_loss = running_loss / (len(valid_loader)/batch_size)

    val_loss.append(epoch_loss)

    preds = np.array(preds)

    # convert multiclass labels to binary class

    y = np.array(y)

    labels = preds.argmax(1)



    for class_label in np.unique(y):

        idx = y == class_label

        acc = (labels[idx] == y[idx]).astype(np.float).mean()*100

        print('accuracy for class', class_names[class_label], 'is', acc)



    acc = (labels == y).mean()*100

    new_preds = np.zeros((len(preds),))

    temp = preds[labels != 0, 1:]

    new_preds[labels != 0] = temp.sum(1)

    new_preds[labels == 0] = 1-preds[labels == 0, 0]

    y = np.array(y)

    y_all = y.copy()

    y[y != 0] = 1

    auc_score = alaska_weighted_auc(y, new_preds)

    print(

        f'Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}')
import itertools

import sklearn

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.linspace(-0.5, len(classes)-0.5, len(classes))

    plt.xticks(tick_marks, classes, rotation=15)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j+0.05, i+0.05, f'{cm[i, j]:.3}%',

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Compute confusion matrix

cnf_matrix = sklearn.metrics.confusion_matrix(y_all, preds.argmax(1))



# Plot non-normalized confusion matrix

plt.figure(figsize=(30,9))

foo = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Confusion matrix')
# code inspired from https://github.com/jacobgil/pytorch-grad-cam

class FeatureExtractor():

    """ Class for extracting activations and 

    registering gradients from targetted intermediate layers """



    def __init__(self, model):

        self.model = model

        self.gradients = []



    def save_gradient(self, grad):

        self.gradients.append(grad)



    def __call__(self, x):

        outputs = []

        self.gradients = []

        x = self.model(x)

        x.register_hook(self.save_gradient)

        outputs += [x]

        return outputs, x





class ModelOutputs():

    """ Class for making a forward pass, and getting:

    1. The network output.

    2. Activations from intermeddiate targetted layers.

    3. Gradients from intermeddiate targetted layers. """



    def __init__(self, model, feature_module, final_layer):

        self.model = model

        self.feature_module = feature_module

        self.feature_extractor = FeatureExtractor(

            self.feature_module)

        self.final_layer = final_layer



    def get_gradients(self):

        return self.feature_extractor.gradients



    def __call__(self, x):

        target_activations = []

        for name, module in self.model._modules.items():

            if name == '_avg_pooling':

                break

            if module == self.feature_module:

                target_activations, x = self.feature_extractor(x)

            else:

                if '_blocks' in name:

                    for m in module:

                        x = m(x)

                elif name == '_bn0':

                    x = module(x)

                    x = self.model._modules['_swish'](x)

                else:

                    x = module(x)



        x = self.model._modules['_swish'](x)

        x = F.avg_pool2d(x, x.size()[2:]).reshape(-1, 1280)

        x = self.final_layer(x)

        return target_activations, x

    

class GradCam:

    def __init__(self, model, feature_module, final_layer, device=device):

        self.model = model

        self.feature_module = feature_module

        self.model.eval()

        self.device = device

        self.extractor = ModelOutputs(

            self.model, self.feature_module, final_layer)



    def forward(self, input):

        return self.model(input)



    def __call__(self, input, index=None):

        features, output = self.extractor(input)



        if index == None:

            index = np.argmax(output.cpu().data.numpy(), 1)

            pred_classes = output.argmax(1)



        # print('predicted class', class_names[index], 'probability:', F.softmax(

        #     output, 1)[0, index].item())



        # one_hot = np.zeros((index.shape[0], output.size()[-1]), dtype=np.float32)

        # one_hot[0][index] = 1

        # one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        one_hot = torch.zeros(

            (index.shape[0], output.size()[-1]), dtype=torch.float).to(self.device)

        one_hot.scatter_(1, output.argmax(

            1, keepdim=True), 1).requires_grad_(True)

        one_hot = torch.sum(one_hot * output)



        self.feature_module.zero_grad()

        self.model.zero_grad()

        one_hot.backward(retain_graph=True)



        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()



        target = features[-1]

        target = target.cpu().data.numpy()



        weights = np.mean(grads_val, axis=(2, 3))

        cam = np.zeros(

            (target.shape[0], target.shape[2], target.shape[3]), dtype=np.float32)



        for img_num, w1 in enumerate(weights):

            for filt_num, w2 in enumerate(w1):

                cam[img_num] += w2 * target[img_num, filt_num, :, :]



        cam = np.maximum(cam, 0)

        cam = np.array(

            [(cv2.resize(c, input.shape[2:]) - c.min())/c.max() for c in cam])

        # cam = np.array([cv2.resize(c, input.shape[2:]) for c in cam])



        # cam = cam - np.min(cam)

        # cam = cam / np.max(cam)

        return cam, pred_classes
def show_cam_on_image(img, mask, fn, p, l):

    dct_diff = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_VIRIDIS)

    dct_diff = np.float32(dct_diff) / 255

    cam = dct_diff + np.float32(img)

    cam = cam / np.max(cam)

    cam = np.uint8(255 * cam)

    cv2.putText(cam, 'Actual:'+class_names[l.item()], (5, img.shape[0] - 5),

                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)

    cv2.putText(cam, 'Pred:'+class_names[p.item()], (5, img.shape[0] - 30),

                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)



    orig_im = cv2.imread(f'{data_dir}/Cover/{fn}')

    cv2.putText(orig_im, 'Orig Im.', (5, orig_im.shape[0] - 5),

                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)

    folder_name = class_names[l.item()].split('_')[0]

    im = cv2.imread(f"{data_dir}/{folder_name}/{fn}")

#     pdb.set_trace()

    diff_rgb = im - orig_im

    cv2.putText(diff_rgb, 'RGB Diff.', (5, diff_rgb.shape[0] - 5),

                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)



    jio_im = jio.read(f"{data_dir}/Cover/{fn}")

    im = jio.read(f"{data_dir}/{folder_name}/{fn}")

    dct_diff = np.zeros_like(orig_im)



    if folder_name != 'Cover':

        diff_dct = np.array(im.coef_arrays) - np.array(jio_im.coef_arrays)

        diff_dct = (diff_dct-diff_dct.min())/(diff_dct.max()-diff_dct.min())

        diff_dct = np.uint8(diff_dct.transpose(1, 2, 0)*255)

        # dct coeff diff is -1

        idx_x, idx_y, _ = np.where(diff_dct == 0)

        dct_diff[idx_x, idx_y, 2] = 255

        # dct coeff diff is +1

        idx_x, idx_y, _ = np.where(diff_dct == 255)

        dct_diff[idx_x, idx_y, 1] = 255



#     pdb.set_trace()

    cv2.putText(dct_diff, 'DCT Diff.', (5, img.shape[0] - 5),

                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)

    im = np.concatenate([orig_im, cam, diff_rgb, dct_diff], 1)

#     cv2.imwrite(f"{fn}", im)  # uncomment if you want to write images

    return im
AUGMENTATIONS_TEST = Compose([

    ToFloat(max_value=255),

    ToTensor()

], p=1)





grad_cam = GradCam(model=model.model, feature_module=model.model._conv_head,

                   final_layer=model.dense_output, device=device)
def process_batch(loader, device=device):

    for (im, labels, all_fn) in loader:

        inputs = im["image"].to(device, dtype=torch.float).requires_grad_(True)

        masks, pred_classes = grad_cam(inputs, None)

        inputs = inputs.permute(0, 2, 3, 1).detach().cpu().numpy()

#         plt.figure(figsize=(20, 10))

        plt.figure(figsize=(len(inputs)*5, len(inputs)*5))

        for i, (img, mask, fn, p, l) in enumerate(zip(inputs, masks, all_fn, pred_classes, labels)):

            viz = show_cam_on_image(img, mask, fn, p, l)

            plt.subplot(len(inputs), 1, i+1)

            plt.imshow(viz)

            plt.axis('off')

        break # remove this to process all images

    plt.subplots_adjust(hspace=0.02, wspace=0)

#     plt.tight_layout()
temp_df = val_df[val_df.Label==0].sample(100).reset_index(drop=True)

valid_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)







valid_loader = torch.utils.data.DataLoader(valid_dataset,

                                           batch_size=16,

                                           num_workers=0,

                                           shuffle=False)



process_batch(valid_loader)
temp_df = val_df[val_df.Label.isin([1,2,3])].reset_index(drop=True)

valid_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)







valid_loader = torch.utils.data.DataLoader(valid_dataset,

                                           batch_size=16,

                                           num_workers=0,

                                           shuffle=False)



process_batch(valid_loader, device=device)
temp_df = val_df[val_df.Label.isin([4,5,6])].reset_index(drop=True)

valid_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)







valid_loader = torch.utils.data.DataLoader(valid_dataset,

                                           batch_size=16,

                                           num_workers=0,

                                           shuffle=False)



process_batch(valid_loader, device=device)
temp_df = val_df[val_df.Label.isin([7,8,9])].reset_index(drop=True)

valid_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)







valid_loader = torch.utils.data.DataLoader(valid_dataset,

                                           batch_size=16,

                                           num_workers=0,

                                           shuffle=False)



process_batch(valid_loader, device=device)