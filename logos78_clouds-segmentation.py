import numpy as np

import pandas as pd

import os

import math



import cv2 as cv

from PIL import Image



import torch

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split



import torch.nn as nn

from torchvision import models

import torch.nn.functional as F



from torch.optim.optimizer import Optimizer, required

from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm.auto import tqdm as tq



import matplotlib.pyplot as plt

import seaborn as sns
path = "../input/understanding_cloud_organization/"

shape = (1400, 2100)

resized = (350, 525)

resized_inv = (525, 350)

train_on_gpu = torch.cuda.is_available()
def prepare_dataset(df) :

    # Extract masks

    fish = df[df['Image_Label'].str.contains('Fish')].EncodedPixels.to_numpy()

    flower = df[df['Image_Label'].str.contains('Flower')].EncodedPixels.to_numpy()

    gravel = df[df['Image_Label'].str.contains('Gravel')].EncodedPixels.to_numpy()

    sugar = df[df['Image_Label'].str.contains('Sugar')].EncodedPixels.to_numpy()

    

    # Extract files name

    df.Image_Label = df.Image_Label.str.replace('_Fish', '')

    df.Image_Label = df.Image_Label.str.replace('_Flower', '')

    df.Image_Label = df.Image_Label.str.replace('_Gravel', '')

    df.Image_Label = df.Image_Label.str.replace('_Sugar', '')

    images = df.Image_Label.unique()

    

    

    fish = np.reshape(fish, (fish.shape[0],1))

    flower = np.reshape(flower, (flower.shape[0],1))

    gravel = np.reshape(gravel, (gravel.shape[0],1))

    sugar = np.reshape(sugar, (sugar.shape[0],1))

    images = np.reshape(images, (images.shape[0],1))

    

    # Create a new dataset where each row represents an image and all its masks

    new_df = np.concatenate((images, fish, flower, gravel, sugar), axis=1)

    new_df = pd.DataFrame(data=new_df, columns=['Image', 'Fish', 'Flower', 'Gravel', 'Sugar'])

    

    return new_df
dataset = prepare_dataset(pd.read_csv(path+'train.csv'))

dataset.head()
### Transform inputs data (1D) into 2D masks

def rle_decode(mask, shape=(1400,2100)) :

    m = str(mask)

    if m == 'nan' :

        m = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    else :

        m = m.split()

        starts = np.asarray(m[0:][::2], dtype=int) - 1

        lengths = np.asarray(m[1:][::2], dtype=int)

        ends = starts + lengths

        

        m = np.zeros(shape[0]*shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):

            m[lo:hi] = 1

        

    m = m.reshape(shape, order='F')

    m = cv.resize(m, resized_inv)

    return m





def display_img_with_masks(img, masks) :

    plt.figure()

    plt.imshow(to_img(img))

    for mask in masks :

        if np.sum(mask) > 0 :

            plt.imshow(mask, alpha=0.2, cmap='gray')

    plt.show()
class Dataset(Dataset) :

    # Constructor

    def __init__(self, df=None, transform=None, train=True) :

        self.directory = "/kaggle/input/understanding_cloud_organization/"

        if train == True :

            self.directory = self.directory + "train_images/"

        else :

            self.directory = self.directory + "test_images/"

        self.all_files = [self.directory + img for img in df.Image]

        self.masks = df.drop(columns=['Image']).to_numpy()

        self.transform = transform

        self.len = len(self.all_files)

        

    # Getter

    def __getitem__(self, idx):

        image = Image.open(self.all_files[idx])

        y = self.masks[idx]

        Y = np.zeros((4,350,525))

        for i in range(4) :

            Y[i,:,:] = rle_decode(y[i])

        if self.transform:

            image = self.transform(image)

        return image, Y

    

    def __len__(self):

        return self.len
transform = transforms.Compose([transforms.Resize(resized), transforms.ToTensor()])

to_img = transforms.ToPILImage()



train, validation = train_test_split(dataset, test_size=0.2, random_state=4)

train_set = Dataset(train, transform)

validation_set = Dataset(validation, transform)
img, masks = train_set[0]

display_img_with_masks(img, masks)
class Conv(nn.Module) :

    def __init__(self, in_ch, out_ch) :

        super(Conv, self).__init__()

        self.conv = nn.Sequential(

                nn.Conv2d(in_ch, out_ch, 3, padding=1),

                nn.BatchNorm2d(out_ch),

                nn.ReLU(inplace=True),

                nn.Conv2d(out_ch, out_ch, 3, padding=1),

                nn.BatchNorm2d(out_ch),

                nn.ReLU(inplace=True))

        

    def forward(self, x) :

        x = self.conv(x)

        return x





class Down(nn.Module) :

    def __init__(self, in_ch, out_ch) :

        super(Down, self).__init__()

        self.layer = nn.Sequential(nn.MaxPool2d(2), Conv(in_ch, out_ch))

        

    def forward(self, x) :

        x = self.layer(x)

        return x





class Up(nn.Module) :

    def __init__(self, in_ch, out_ch) :

        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, 2)

        self.conv = Conv(in_ch, out_ch)

        

    def forward(self, x, x_prev) :

        x = self.up(x)

        

        diffY = x_prev.size()[2] - x.size()[2]

        diffX = x_prev.size()[3] - x.size()[3]

        x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x_prev, x], dim=1)

        

        return self.conv(x)





class UNet(nn.Module) :

    def __init__(self, n_chan, n_classes) :

        super(UNet, self).__init__()

        self.init = Conv(n_chan,64)

        self.down1 = Down(64,128)

        self.down2 = Down(128,256)

        self.down3 = Down(256,512)

        self.down4 = Down(512,512)

        self.up1 = Up(1024,256)

        self.up2 = Up(512,128)

        self.up3 = Up(256,64)

        self.up4 = Up(128,64)

        self.out = nn.Conv2d(64, n_classes, 1)

        

    def forward(self, x):

        x1 = self.init(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        x = self.out(x)

        return torch.sigmoid(x)
model = UNet(3, 4).float()

if train_on_gpu:

    model.cuda()
class DiceLoss(nn.Module) :

    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid') :

        super().__init__()

        self.activation = activation

        self.eps = eps

    

    def forward(self, pred, truth) :

        tp = torch.sum(pred*truth)

        fp = torch.sum(pred) - tp

        fn = torch.sum(truth) - tp

        score = (2*tp + self.eps) / (2*tp + fn + fp + self.eps)

        return 1-score

        

        

class BCEDiceLoss(DiceLoss):

    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):

        super().__init__(eps, activation)

        self.bce = nn.BCELoss(reduction='mean')



    def forward(self, pred, truth):

        dice = super().forward(pred, truth)

        bce = self.bce(pred, truth)

        return dice + bce



    

def calc_dice_score(outputs, targets, threshold=None, min_size=None, eps=1e-7) :

    if threshold is not None :

        outputs = (outputs > threshold).float()

    

    if min_size is not None :

        if torch.sum(outputs) < min_size :

            outputs = torch.zeros(outputs.shape[0], outputs.shape[1])

    

    intersection = torch.sum(targets * outputs)

    union = torch.sum(targets) + torch.sum(outputs)

    dice = (2*intersection + eps) / (union + eps)



    return dice
class RAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        if not 0.0 <= lr:

            raise ValueError("Invalid learning rate: {}".format(lr))

        if not 0.0 <= eps:

            raise ValueError("Invalid epsilon value: {}".format(eps))

        if not 0.0 <= betas[0] < 1.0:

            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        if not 0.0 <= betas[1] < 1.0:

            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

            

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.buffer = [[None, None, None] for ind in range(10)]

        super(RAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):

        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:

                    N_sma, step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    else:

                        step_size = 1.0 / (1 - beta1 ** state['step'])

                    buffered[2] = step_size



                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)



                # more conservative since it's an approximated value

                if N_sma >= 5:            

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)

                else:

                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)



                p.data.copy_(p_data_fp32)



        return loss
criterion = BCEDiceLoss(eps=1.0, activation=None)

optimizer = RAdam(model.parameters(), lr = 0.005)
def train_model(model, n_epochs,

                train_set, validation_set, batch_size,

                criterion, optimizer) :

    

    train_loss_list = []

    valid_loss_list = []

    dice_score_list = []

    lr_rate_list = []

    valid_loss_min = np.Inf # track change in validation loss

    

    current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]

    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)

    train_loader = DataLoader(train_set, batch_size = batch_size)

    validation_loader = DataLoader(validation_set, batch_size = batch_size)



    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss

        train_loss = 0.0

        valid_loss = 0.0

        dice_score = 0.0



        ###################

        # train the model #

        ###################

        model.train()

        bar = tq(train_loader, postfix={"train_loss":0.0})

        for data, target in bar:

            if train_on_gpu:

                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            output = model(data).double()

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()*data.size(0)

            bar.set_postfix(ordered_dict={"train_loss":loss.item()})



        ######################    

        # validate the model #

        ######################

        model.eval()

        del data, target

        with torch.no_grad():

            bar = tq(validation_loader, postfix={"valid_loss":0.0, "dice_score":0.0})

            for data, target in bar:

                if train_on_gpu:

                    data, target = data.cuda(), target.cuda()

                output = model(data).double()

                loss = criterion(output, target)

                valid_loss += loss.item()*data.size(0)

                dice_cof = calc_dice_score(output.cpu(), target.cpu()).item()

                dice_score +=  dice_cof * data.size(0)

                bar.set_postfix(ordered_dict={"valid_loss":loss.item(), "dice_score":dice_cof})



        # calculate average losses

        train_loss = train_loss/len(train_loader.dataset)

        valid_loss = valid_loss/len(validation_loader.dataset)

        dice_score = dice_score/len(validation_loader.dataset)

        train_loss_list.append(train_loss)

        valid_loss_list.append(valid_loss)

        dice_score_list.append(dice_score)

        lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])



        # print training/validation statistics 

        print('Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f} Dice Score: {:.6f}'.format(

            epoch, train_loss, valid_loss, dice_score))



        # save model if validation loss has decreased

        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

            valid_loss_min,

            valid_loss))

            torch.save(model.state_dict(), 'clouds_segmentation_unet.pt')

            valid_loss_min = valid_loss



        scheduler.step(valid_loss)

    

    return model, [train_loss_list, valid_loss_list, dice_score_list, lr_rate_list]
n_epochs = 20

batch_size = 4

#model, history = train_model(model, n_epochs, train_set, validation_set, batch_size, criterion, optimizer)

#train_loss_list, valid_loss_list, dice_score_list, lr_rate_list = history
'''

plt.figure(figsize=(10,10))

plt.plot([i[0] for i in lr_rate_list])

plt.ylabel('learing rate during training', fontsize=22)

plt.show()



plt.figure(figsize=(10,10))

plt.plot(train_loss_list,  marker='o', label="Training Loss")

plt.plot(valid_loss_list,  marker='o', label="Validation Loss")

plt.ylabel('loss', fontsize=22)

plt.legend()

plt.show()



plt.figure(figsize=(10,10))

plt.plot(dice_score_list)

plt.ylabel('Dice score')

plt.show()

'''
checkpoint = torch.load("../input/clouds-output/clouds_segmentation_unet.pt")

model.load_state_dict(checkpoint)
def calc_score(outputs, targets, threshold=None, min_size=None, eps=1e-7) :

    if threshold is not None :

        outputs = (outputs > threshold).float()

    

    if min_size is not None :

        if torch.sum(outputs) < min_size :

            outputs = torch.zeros(outputs.shape[0], outputs.shape[1])

    

    inter = torch.sum(targets * outputs)

    pred = torch.sum(outputs)

    truth = torch.sum(targets)

    

    dice = (2*inter + eps) / (pred + truth + eps)

    recall = (inter + eps) / (truth + eps)

    precision = (inter + eps) / (pred + eps) 

    

    return dice, precision, recall





def post_processing(model, validation_set) :

    threshold_list = [i/100 for i in range(10,100,5)]

    size_list = [500, 1000, 2500, 5000, 7500, 10000]

    

    dice_table = np.zeros((len(threshold_list), len(size_list)))

    precision_table = np.zeros((len(threshold_list), len(size_list)))

    recall_table = np.zeros((len(threshold_list), len(size_list)))

    

    model.eval()

    with torch.no_grad():

        bar = tq(DataLoader(validation_set, batch_size=1))

        for data, target in bar:

            if train_on_gpu:

                data, target = data.cuda(), target.cuda()

            output = model(data).cpu()

            target  = target.cpu()

            i = 0

            for t in threshold_list :

                j = 0

                for s in size_list :

                    for mask, tar in zip(output[0], target[0]) :

                        score = calc_score(mask, tar, t, s)

                        dice_table[i,j] += score[0]

                        precision_table[i,j] += score[1]

                        recall_table[i,j] += score[2]

                    j += 1

                i += 1



    dice_table /= (len(validation_set)*4)

    precision_table /= (len(validation_set)*4)

    recall_table /= (len(validation_set)*4)

    

    # Merge the tables into a dataframe

    dice_df = []

    i = 0

    for t in threshold_list :

        j = 0

        for s in size_list :

            dice_df.append((t, s, dice_table[i,j], precision_table[i,j], recall_table[i,j]))

            j += 1

        i += 1



    dice_df = pd.DataFrame(dice_df, columns=['threshold', 'size', 'dice', 'precision', 'recall'])

    dice_df.to_csv('dice_scores.csv', index=False)

    

    return dice_df
#dice_df = post_processing(model, validation_set)
dice_df = pd.read_csv('../input/clouds-output/dice_scores.csv')
print(dice_df.groupby(['threshold'])['dice'].max(),'\n')

print(dice_df.groupby(['size'])['dice'].max(),'\n')

print(dice_df.sort_values('dice', ascending=False))
print(dice_df.groupby(['threshold'])['precision'].max(),'\n')

print(dice_df.groupby(['threshold'])['recall'].max(),'\n')
sns.lineplot(x='threshold', y='dice', hue='size', data=dice_df);

plt.title('Threshold and min size vs dice');
params = dice_df.sort_values('dice', ascending=False)[0:1].to_numpy()

params = list(params[0][:2])

print('Best parameters')

print('threshold =', params[0], '   size =', params[1])
def process_mask(outputs, threshold, min_size) :

    outputs = (outputs > threshold).float()

    

    if torch.sum(outputs) < min_size :

        outputs = torch.zeros(outputs.shape[0], outputs.shape[1])

    return outputs





def show_results(orig_img, output, target, params) :

    threshold, min_size = params

    fontsize = 14

    class_list = ['Fish', 'Flower', 'Sugar', 'Gravel']

    

    f, ax = plt.subplots(3, 5, figsize=(24, 12))

    ax[0,0].imshow(orig_img)

    ax[0,0].set_title("Original image", fontsize=fontsize)



    for i in range(4):

        ax[0,i+1].imshow(target[i])

        ax[0,i+1].set_title(f"Original mask {class_list[i]}", fontsize=fontsize)



    ax[1,0].imshow(orig_img)

    ax[1,0].set_title("Original image", fontsize=fontsize)



    for i in range(4):

        dice_score = calc_dice_score(output[i], target[i]).item()

        ax[1,i+1].imshow(output[i])

        ax[1,i+1].set_title(f"Raw predicted mask {class_list[i]}\nDice score = {round(dice_score,2)}", fontsize=fontsize)



    ax[2,0].imshow(orig_img)

    ax[2,0].set_title("Transformed image", fontsize=fontsize)



    for i in range(4):

        dice_score = calc_dice_score(output[i], target[i], threshold, min_size).item()

        ax[2,i+1].imshow(process_mask(output[i], threshold, min_size))

        ax[2,i+1].set_title( f"Predicted mask with processing {class_list[i]}\nDice score = {round(dice_score,2)}", fontsize=fontsize)

### Choose few images to display with their original, predicted and processed masks

n = 50

num_of_img_to_display = 4

display_set = Dataset(validation[n:n+num_of_img_to_display], transform)



model.eval()

with torch.no_grad():

    for data, target in DataLoader(display_set, batch_size=1):

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        output = model(data).cpu()[0]

        img = to_img(data.cpu()[0])

        show_results(img, output, target.cpu()[0], params)
test_dataset = prepare_dataset(pd.read_csv(path + 'sample_submission.csv'))
test_set = Dataset(test_dataset, transform, train=False)

test_loader = DataLoader(test_set, batch_size=1)
def mask2rle(img):

    """

    Convert mask to rle.

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    """

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)
def prepare_sub(model, test_loader) :

    encoded_pixels = []

    model.eval()

    with torch.no_grad():

        bar = tq(test_loader)

        for data, target in bar:

            if train_on_gpu:

                data = data.cuda()

            output = model(data).cpu()[0]

            for mask in output :

                mask = process_mask(mask, params[0], params[1])

                mask = mask2rle(mask.detach().numpy())

                encoded_pixels.append(mask)

    return encoded_pixels
#submission = pd.read_csv(path + 'sample_submission.csv')

#submission['EncodedPixels'] = prepare_sub(model, test_loader)

submission = pd.read_csv('../input/clouds-output/submission.csv')

submission.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)