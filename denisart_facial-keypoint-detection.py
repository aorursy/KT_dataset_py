!unzip ../input/facial-keypoints-detection/test.zip

!unzip ../input/facial-keypoints-detection/training.zip
!ls
import copy

from collections import namedtuple, Counter



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error



%matplotlib inline
import torch

import torch.nn as nn

import torch.optim as optim

import torch.functional as F

from torchvision import models, transforms

from torch.utils.data import Dataset, SubsetRandomSampler



device = torch.device('cuda:0')
class FacialKeypoiuntsTrainDataset(Dataset):

    '''Facial Keypoint Detection Train Dataset.



    Url:

        https://www.kaggle.com/c/facial-keypoints-detection/overview



    Arguments:

        data_file (str): csv file with train dataset.

        transform (callable, optional):

            Optional transform to be applied on a sample.

    '''



    def __init__(self, data_file, transform=None):

        self.transform = transform

 

        dataset = pd.read_csv(data_file)

        dataset.fillna(method='ffill', inplace=True)



        self.images = dataset['Image']

        dataset = dataset.drop(['Image'], axis=1)



        self.positions_name = list(dataset.columns)

        self.positions = dataset.to_numpy()



    def __len__(self):

        return len(self.positions)



    def __getitem__(self, index):

        '''Get sample by index. The sample it is dict:



                {'img': img, 'landmarks': landmarks},



        where `img` in is transformed image in torch Tensor format and

        `landmarks` it is Tensor with facial keypoints.

        

        Also, we convert images from grayscale to RGB using original image for

        all channels.

        '''

        x = self.images[index]

        x = np.array([float(x) for x in x.split(' ')])

        x = x.reshape((96, 96))

        img = np.stack((x, x, x), axis=-1) / 255.



        y = self.positions[index]

        sample = {'img': img, 'landmarks': y}



        if self.transform:

            sample = self.transform(sample)



        return sample



    def show_samples(self, indices, title=None, count=10):

        '''Show image with landmarks by indices.

        

        Arguments:

            indices (array_like): the array with image indeces.

            title (str, optional): the title of figure.

            count (int, optional): the number of images for plot.

        '''

        plt.figure(figsize=(count*3, 3))

        display_indices = indices[:count]



        if title:

            plt.suptitle(title)



        for i, index in enumerate(display_indices):

            sample = self.__getitem__(index)



            x, y = sample['img'], sample['landmarks']

            x = np.asarray(x)

            y = np.asarray(y)



            y = y.reshape((y.size // 2, 2))



            plt.subplot(1, count, i + 1)

            if self.transform:

                plt.imshow(np.transpose(x, (1, 2, 0)))

            else:    

                plt.imshow(x)

            plt.scatter(y[:, 0], y[:, 1], s=15, marker='.', c='r')



            plt.grid(False)

            plt.axis('off')
class FacialKeypoiuntsTestDataset(Dataset):

    '''Facial Keypoint Detection Test Dataset.



    Url:

        https://www.kaggle.com/c/facial-keypoints-detection/overview



    Arguments:

        data_file (str): csv file with test dataset.

    '''



    def __init__(self, data_file):

        self.dataset = pd.read_csv(data_file)



    def __len__(self):

        return len(self.dataset)



    def __getitem__(self, index):

        '''Get sample by index. The sample it is tuple:



                (img_t, img_id),



        where `img_t` in is image in torch Tensor format and

        `img_id` it is image id from test dataset.



        Also, we convert images from grayscale to RGB using original image for

        all channels.

        '''

        data_sr = self.dataset.iloc[index]

        img_id = int(data_sr['ImageId'])

        x = data_sr['Image']



        x = np.array([float(x) for x in x.split(' ')])

        x = x.reshape((96, 96))

        img = np.stack((x, x, x), axis=-1) / 255.

        img = np.transpose(img, (2, 0, 1)).copy()



        img_t = torch.from_numpy(img).type(torch.FloatTensor)



        return img_t, img_id



    def show_samples(self, indices, title=None, count=10):

        '''Show image by indices.

        

        Arguments:

            indices (array_like): the array with image indeces.

            title (str, optional): the title of figure.

            count (int, optional): the number of images for plot.

        '''

        plt.figure(figsize=(count*3, 3))

        display_indices = indices[:count]



        if title:

            plt.suptitle(title)



        for i, index in enumerate(display_indices):

            x, y = self.__getitem__(index)

            x = np.asarray(x)



            plt.subplot(1, count, i + 1)

            plt.imshow(np.transpose(x, (1, 2, 0)))

            plt.title(f'Image Id {y}')



            plt.grid(False)

            plt.axis('off')
train_dataset = FacialKeypoiuntsTrainDataset('./training.csv')

test_dataset = FacialKeypoiuntsTestDataset('./test.csv')



# indices for plot

train_plot_indices = np.random.choice(len(train_dataset), 10)

test_plot_indices = np.random.choice(len(test_dataset), 10)



train_dataset.show_samples(

    train_plot_indices, title='Samples from Facial Keypoints Train Dataset', count=7)



test_dataset.show_samples(

    test_plot_indices, title='Samples from Facial Keypoints Test Dataset', count=7)
class RandomVerticalFlip(object):

    '''Random Vertical Flip for sample.



    Arguments:

        p (float, optional):

            Probability of the image being flipped. Default value is 0.5

    '''



    def __init__(self, p=0.5):

        assert isinstance(p, float), 'p must be of type float.'

        assert (p >= 0.) and (p <= 1.), 'p must be from [0, 1].'

        self.p = p



    def __call__(self, sample):

        img, landmarks = sample['img'], sample['landmarks']

        flip_prob = np.random.binomial(1, self.p)



        if flip_prob == 0:

            new_img = img

            new_landmarks = landmarks

        else:

            n = img.shape[0]



            new_img = np.flip(img, axis=1)

            new_landmarks = np.zeros(landmarks.size)



            # left eye center

            (new_landmarks[0], new_landmarks[1]) = (n - landmarks[2], landmarks[3])

            # right eye center

            (new_landmarks[2], new_landmarks[3]) = (n - landmarks[0], landmarks[1])

            # left_eye_inner_corner

            (new_landmarks[4], new_landmarks[5]) = (n - landmarks[8], landmarks[9])

            # left_eye_outer_corner

            (new_landmarks[6], new_landmarks[7]) = (n - landmarks[10], landmarks[11])

            # right_eye_inner_corner

            (new_landmarks[8], new_landmarks[9]) = (n - landmarks[4], landmarks[5])

            # right_eye_outer_corner

            (new_landmarks[10], new_landmarks[11]) = (n - landmarks[6], landmarks[7])

            # left_eyebrow_inner_end

            (new_landmarks[12], new_landmarks[13]) = (n - landmarks[16], landmarks[17])

            # left_eyebrow_outer_end

            (new_landmarks[14], new_landmarks[15]) = (n - landmarks[18], landmarks[19])

            # right_eyebrow_inner_end

            (new_landmarks[16], new_landmarks[17]) = (n - landmarks[12], landmarks[13])

            # right_eyebrow_outer_end

            (new_landmarks[18], new_landmarks[19]) = (n - landmarks[14], landmarks[15])

            # nose_tip

            (new_landmarks[20], new_landmarks[21]) = (n - landmarks[20], landmarks[21])

            # mouth_left_corner

            (new_landmarks[22], new_landmarks[23]) = (n - landmarks[24], landmarks[25])

            # mouth_right_corner

            (new_landmarks[24], new_landmarks[25]) = (n - landmarks[22], landmarks[23])

            # mouth_center_top_lip_x

            (new_landmarks[26], new_landmarks[27]) = (n - landmarks[26], landmarks[27])

            # mouth_center_bottom_lip_x

            (new_landmarks[28], new_landmarks[29]) = (n - landmarks[28], landmarks[29])



        return {'img': new_img, 'landmarks': new_landmarks}





class RandomTranslation(object):

    '''Random Translation for sample.



    Arguments:

        translate (tuple, optional): 

            Tuple of maximum absolute fraction for horizontal and vertical

            translations. For example translate=(a, b), then horizontal shift

            is randomly sampled in the range

                -img_width * a < dx < img_width * a

            and vertical shift is randomly sampled in the range

                -img_height * b < dy < img_height * b.

            Will not translate by default.

    '''



    def __init__(self, translate=None):

        self.translate = translate



        if self.translate is not None:

            assert1_text = 'translate must be (float, float).'

            assert2_text = 'translate[0] must be from [0, 1].'

            assert3_text = 'translate[1] must be from [0, 1].'



            assert (isinstance(self.translate, tuple)) and (len(self.translate) == 2), assert1_text

            assert (self.translate[0] >= 0.) and (self.translate[0] <= 1.), assert2_text

            assert (self.translate[1] >= 0.) and (self.translate[1] <= 1.), assert2_text



    def __call__(self, sample):

        img, landmarks = sample['img'], sample['landmarks']

        n_img = img.shape[0]

        n_landmarks = landmarks.size



        max_dx = self.translate[0] * n_img

        max_dy = self.translate[1] * n_img



        dx = np.random.randint(-max_dx, max_dx)

        dy = np.random.randint(-max_dy, max_dy)



        new_img = np.zeros(img.shape)

        new_landmarks = np.zeros(n_landmarks)



        if dx >= 0:

            li_x, ri_x = dx, n_img

        else:

            li_x, ri_x = 0, n_img + dx 



        if dy >= 0:

            li_y, ri_y = dy, n_img

        else:

            li_y, ri_y = 0, n_img + dy



        new_img[li_x:ri_x, li_y:ri_y, 0] = img[li_x-dx:ri_x-dx, li_y-dy:ri_y-dy, 0]

        new_img[li_x:ri_x, li_y:ri_y, 1] = img[li_x-dx:ri_x-dx, li_y-dy:ri_y-dy, 1]

        new_img[li_x:ri_x, li_y:ri_y, 2] = img[li_x-dx:ri_x-dx, li_y-dy:ri_y-dy, 2]



        for j in range(n_landmarks // 2):

            new_landmarks[2*j + 1] = landmarks[2*j + 1] + dx

            new_landmarks[2*j] = landmarks[2*j] + dy



        return {'img': new_img, 'landmarks': new_landmarks}





class ToTensor(object):

    '''Convert ndarrays in sample to Tensors.'''



    def __call__(self, sample):

        img, landmarks = sample['img'], sample['landmarks']

        img = np.transpose(img, (2, 0, 1)).copy()



        return {'img': torch.from_numpy(img).type(torch.FloatTensor),

                'landmarks': torch.from_numpy(landmarks).type(torch.FloatTensor)}





class Normalization(object):

    '''Normalize a tensor image with mean and standard deviation.'''



    def __init__(self, mean, std, inplace=False):

        self.normalization = transforms.Normalize(mean, std, inplace)



    def __call__(self, sample):

        img = sample['img']

        img_n = self.normalization(img)



        return {'img': img_n, 'landmarks': sample['landmarks']}
train_dataset = FacialKeypoiuntsTrainDataset(

    './training.csv',

    transform=transforms.Compose([

        RandomVerticalFlip(0.5),

        RandomTranslation((0.2, 0.2)),

        ToTensor(),

    ])

)



train_dataset.show_samples(

    train_plot_indices, title='Samples from Facial Keypoints Train Dataset', count=7)
batch_size = 32



data_size = len(train_dataset)

validation_fraction = .2



val_split_size = int(np.floor(validation_fraction * data_size))

indeces = list(range(data_size))

np.random.shuffle(indeces)



val_indeces, train_indeces = indeces[:val_split_size], indeces[val_split_size:]



test_indeces = list(range(len(test_dataset)))

print(f'train size = {len(train_indeces)}, validation size = {len(val_indeces)}')

print(f'test size = {len(test_indeces)}')



train_sampler = SubsetRandomSampler(train_indeces)

val_sampler = SubsetRandomSampler(val_indeces)

test_sampler = SubsetRandomSampler(test_indeces)



train_loader = torch.utils.data.DataLoader(train_dataset,

                                           batch_size=batch_size,

                                           sampler=train_sampler)



val_loader = torch.utils.data.DataLoader(train_dataset,

                                         batch_size=batch_size,

                                         sampler=val_sampler)



test_loader = torch.utils.data.DataLoader(test_dataset,

                                          batch_size=batch_size,

                                          sampler=test_sampler)
nn_model = models.resnet18(pretrained=True)



# add new output layer

# also, recall that landmarks has size 30

nn_model.fc = nn.Linear(nn_model.fc.in_features, 30)



nn_model = nn_model.type(torch.FloatTensor)

nn_model = nn_model.to(device)



*old_params, new_params = nn_model.parameters()
def train_model(model, train_loader, val_loader, loss, optimizer, scheduler, num_epoch, plot_epoch):

    '''The train model.

    

    Arguments:

        model (torch.nn.Module): the model for training.

        train_loader (torch.utils.data.dataloader.DataLoader):

            the loader of train dataset.

        val_loader (torch.utils.data.dataloader.DataLoader):

            the loader of validation dataset.

        loss (torch.nn.modules.loss): the loss for optimization.

        optimizer (torch.optim): the optimizer.

        scheduler (torch.optim.lr_scheduler): the scheduler.

        num_epoch (int): the number of epochs.

        plot_epoch (int): the length of the interval for showing intermediate results.

    '''

    loss_history = []

    train_rmse_history = []

    val_rmse_history = []



    best_model = None

    best_val_rmse = None



    indices = np.random.choice(batch_size, 10)



    for epoch in range(num_epoch):

        print(f'Epoch {epoch + 1:2d} / {num_epoch:2d}', end=' ')



        model.train()



        average_loss = 0

        average_rmse = 0



        for i_step, sample in enumerate(train_loader):

            x, y = sample['img'], sample['landmarks']

            x_gpu = x.to(device)

            y_gpu = y.to(device)



            prediction = model(x_gpu)



            loss_value = loss(prediction, y_gpu)

            optimizer.zero_grad()

            loss_value.backward()

            optimizer.step()



            average_loss += loss_value.item()



            rmse = mean_squared_error(

                y,

                prediction.cpu().detach().numpy(),

                squared=False)

            average_rmse += rmse



        average_loss = average_loss / i_step

        train_rmse = average_rmse / i_step

        val_rmse = compute_rmse(model, val_loader)



        loss_history.append(average_loss)

        train_rmse_history.append(train_rmse)

        val_rmse_history.append(val_rmse)



        print(f'Loss = {average_loss:.4f}, Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}')



        if best_val_rmse is None:

            best_val_rmse = val_rmse

            best_model = copy.deepcopy(model)



        if val_rmse < best_val_rmse:

            best_val_rmse = val_rmse

            best_model = copy.deepcopy(model)



        scheduler.step()



        if ((epoch + 1) % plot_epoch) == 0:

            plot_results(model, val_loader, indices, title=f'Examples for epoch {epoch + 1}')



    print(f'   Best val RMSE = {best_val_rmse:.4f}')



    return loss_history, train_rmse_history, val_rmse_history, best_val_rmse, best_model





def compute_rmse(model, val_loader):

    '''Compute Root Mean Square Error for validation data.'''

    average_rmse = 0



    model.eval()

    for i_step, sample in enumerate(val_loader):

        x, y = sample['img'], sample['landmarks']

        x_gpu = x.to(device)



        with torch.no_grad():

            prediction = model(x_gpu)



        rmse = mean_squared_error(

            y,

            prediction.cpu().detach().numpy(),

            squared=False)



        average_rmse += rmse



    val_rmse = float(average_rmse) / i_step



    return val_rmse





def plot_results(model, val_loader, indices, title=None, count=10):

    '''

    '''

    plt.figure(figsize=(count*3, 3))

    display_indices = indices[:count]



    if title:

        plt.suptitle(title)



    model.eval()



    for i_step, samples in enumerate(val_loader):

        x_gpu, _ = samples['img'], samples['landmarks']

        x_gpu = x_gpu.to(device)



        with torch.no_grad():

            prediction = model(x_gpu)



        ys = prediction.cpu().detach().numpy()

        xs = x_gpu.cpu().detach().numpy()



        for i, index in enumerate(display_indices):

            x = xs[index]



            y = ys[index]

            y = y.reshape((y.size // 2, 2))



            plt.subplot(1, count, i + 1)

            if x.shape[0] == 3:

                plt.imshow(np.transpose(x, (1, 2, 0)))

            else:    

                plt.imshow(x)

            plt.scatter(y[:, 0], y[:, 1], s=15, marker='.', c='r')



            plt.grid(False)

            plt.axis('off')

        plt.show()



        break
loss = nn.MSELoss().to(device)



optimizer = optim.Adam([

    {'params': old_params, 'lr': 0.0001},

    {'params': new_params}    

], lr=0.001, weight_decay=0.01)



scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.)



(loss_history, train_rmse_history, val_rmse_history,

 best_val_rmse, best_model) = train_model(nn_model, train_loader, val_loader, loss,

                                          optimizer, scheduler, num_epoch=50, plot_epoch=3)
def test_dataset_prediction(test_loader, submissions_data_file,

                            lookup_table_data_file, positions_name, model):

    '''Prediction of test dataset samples.



    Arguments:

        test_loader (): Path to csv file with test dataset.

        submission_data_file (str): Path to csv file with submission template.

        lookup_table_data_file (str): Path to csv file with lookup table.

        position_names (list): Facial positions name.

        model (nn.Module): The model for prediction.

    '''

    submission = pd.read_csv(submissions_data_file)

    lookup_table = pd.read_csv(lookup_table_data_file)



    model.eval()

    for i_step, (x, img_id) in enumerate(test_loader):

        x_gpu = x.to(device)



        with torch.no_grad():

            prediction = model(x_gpu)



        prediction = np.asarray(prediction.cpu())



        for b_id in range(prediction.shape[0]):

            for p_id in range(len(positions_name)):

                p_name = positions_name[p_id]

                location = prediction[b_id, p_id]



                row_id = lookup_table.loc[(

                    (lookup_table['FeatureName'] == p_name) &

                    (lookup_table['ImageId'] == img_id[b_id].item())

                )]['RowId'].to_numpy()



                if row_id.size == 1:

                    submission['Location'][int(row_id[0]) - 1] = location



    return submission
test_submissions = test_dataset_prediction(

    test_loader, '../input/facial-keypoints-detection/SampleSubmission.csv',

    '../input/facial-keypoints-detection/IdLookupTable.csv',

    train_dataset.positions_name, best_model)



test_submissions.to_csv('./submission.csv', index=False)