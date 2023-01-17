from typing import List

import logging

from typing import Optional

from functools import partial

from typing import Tuple

from typing import Union





import torch.nn as nn

import numpy as np

import os

import pandas as pd

import torch

from torch.optim import Adam

from torchvision.models.resnet import BasicBlock

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from PIL import Image

from matplotlib import pyplot as plt

from torchvision.models.resnet import ResNet

from sklearn.metrics import roc_auc_score

from torch import Tensor

from torchvision import transforms

from torch.autograd import Variable

import albumentations as A
DATA_FOLDER = '../input/histopathologic-cancer-detection'

LABELS = f'{DATA_FOLDER}/train_labels.csv'

TRAIN_IMAGES_FOLDER = f'{DATA_FOLDER}/train'

SAMPLE_SUBMISSION = f'{DATA_FOLDER}/sample_submission.csv'

USE_GPU = torch.cuda.is_available()
USE_GPU
logging.basicConfig(level='INFO')

logger = logging.getLogger()
labels = pd.read_csv(LABELS)
labels
# Метод для конвертации дата фрейма ответов в нумпай массив 

def format_labels_for_data_set(labels):

    return (labels['label'].values.reshape(-1,1))



# Сплит train на тренировочную и валидационную выборки

def train_valid_split(df, split_percent, limit_df= 10000 ):

#     limit_d -  count of images

    df = df.sample(n = df.shape[0])

    df = df.iloc[:limit_df]

    split = round(limit_df * split_percent / 100)

    train = df.iloc[:split]

    valid = df.iloc[split:]

    return (train, valid)



# возвращает полный путь к картинкам из labels sample

def format_path_to_images_for_dataset(labels, path):

    return [os.path.join(path, f'{f}.tif') for f in labels['id'].values]
# класс, в котором определяется исходный DataSet и делается масштабирование исходных данных

class MainDataset(Dataset):

    def __init__(self, x_dataset, y_dataset, x_tfms):

        self.x_dataset = x_dataset

        self.y_dataset = y_dataset

        self.x_tfms = x_tfms

        

    def __len__(self):

        return self.x_dataset.__len__() 

        

    def __getitem__(self, index):

        x = self.x_dataset[index]

        y = self.y_dataset[index]

        if x_tfms is not None:

            x = self.x_tfms(x)

        return x, y



# возвращает картинку (с учетом ее полного пути) по индексу

class ImageDataset(Dataset):

    def __init__(self, path_to_image):

        self.path_to_image = path_to_image

    

    def __len__(self):

        return len(self.path_to_image)

    

    def __getitem__(self, index):

        img = Image.open(self.path_to_image[index])

        

        # Compose a complex augmentation pipeline

        augmentation_pipeline = A.Compose([

            A.HorizontalFlip(p = 0.5), # apply horizontal flip to 50% of images

            A.OneOf(

                [

                    # apply one of transforms to 50% of images

                    A.RandomContrast(), # apply random contrast

                    A.RandomGamma(), # apply random gamma

                    A.RandomBrightness(limit = -0.1), # apply random brightness

                ],

                p = 1

            ),

            A.ShiftScaleRotate(p = 0.5)

        ],

        p = 1)

        

        image_aug = augmentation_pipeline(image = np.array(img))['image']

        image = Image.fromarray(image_aug, 'RGB')

        return image





# возвращает label по индексу

class LabelDataset(Dataset):

    def __init__(self, labels):

        self.labels = labels

    

    def __len__(self):

        return len(labels)

    

    def __getitem__(self, index):

        return self.labels[index]

labels = pd.read_csv(LABELS)

sample_submission = pd.read_csv(SAMPLE_SUBMISSION)



train, valid = train_valid_split(labels, 70)



train_labels = format_labels_for_data_set(train)

valid_labels = format_labels_for_data_set(valid)



train_images = format_path_to_images_for_dataset(train, TRAIN_IMAGES_FOLDER)

valid_images = format_path_to_images_for_dataset(valid, TRAIN_IMAGES_FOLDER)



train_images_dataset = ImageDataset(train_images)

valid_images_dataset = ImageDataset(valid_images)

train_labels_dataset = LabelDataset(train_labels)

valid_labels_dataset = LabelDataset(valid_labels)

# Посмотрим на картинки с аугментацией

def implot(dataset, w=2, h=2, cols=12, max_charts = 24 ):

    rows = (max_charts) / cols + 1

    images = [dataset[3] for i in range(max_charts)]

    plt.figure(figsize = (cols * w, rows * h))

    plt.tight_layout()

    for chart, img in enumerate(images, 1):

        ax = plt.subplot(rows, cols, chart)

        ax.imshow(np.array(img))

        ax.axis('off')
implot(train_images_dataset)
# зададим форматирование и перевод в тензорный вид наших тренировочных данных 

x_tfms = transforms.Compose([transforms.ToTensor(), 

                             transforms.Normalize(

                                 mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225]

                             )

                            ])
# определим объеккты из данных и ответов для загрузки в data loader

train_dataset = MainDataset(train_images_dataset, train_labels_dataset, x_tfms)

valid_dataset = MainDataset(valid_images_dataset, valid_labels_dataset, x_tfms)
# загружаем данные в data loader, определяем число батчей

shuffle = True

batch_size = 512

num_workers = 0



train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)


def to_gpu(tensor):

    return tensor.cuda() if USE_GPU else tensor



def create_resnet9_model(output_dim: int = 1) -> nn.Module:

    model = ResNet(BasicBlock, [1, 1, 1, 1])

    # размер входящей картинки

    in_features = model.fc.in_features

    # output size = 1X1

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    model.fc = nn.Linear(in_features, output_dim)

    model = to_gpu(model)

    return model



resnet9 = create_resnet9_model()

resnet9


lr = 1e-3

optimizer = Adam(resnet9.parameters(), lr)


loss = nn.BCEWithLogitsLoss()


def auc_writer(y_true, y_predicted, iteration):

    try:

        score = roc_auc_score(np.vstack(y_true), np.vstack(y_predicted))

    except:

        score = -1

    print(f'iteration: {iteration}, roc_auc: {score}')

    logger.info(f'iteration: {iteration}, roc_auc: {score}')    

    

loss_writer_train = auc_writer

loss_writer_valid = auc_writer


def predict(model, dataloader):

    model.eval()

    y_true, y_hat = [], []

    

    for x, y in dataloader:

        x = Variable(T(x))

        y = Variable(T(y))

        output = model(x)

        

        y_true.append(to_numpy(y))

        y_hat.append(to_numpy(output))

    

    return y_true, y_hat


def T(tensor):

    if not torch.is_tensor(tensor):

        tensor = torch.FloatTensor(tensor)

    else:

        tensor = tensor.type(torch.FloatTensor)

    if USE_GPU:

        tensor = to_gpu(tensor)

    return tensor





def to_numpy(tensor):

    if type(tensor) == np.array or type(tensor) == np.ndarray:

        return np.array(tensor)

    elif type(tensor) == Image.Image:

        return np.array(tensor)

    elif type(tensor) == Tensor:

        return tensor.cpu().detach().numpy()

    else:

        raise ValueError(msg)




def iteration_trigger(iteration, every_x_iteration):

    if every_x_iteration == 1:

        return True

    elif iteration > 0 and iteration % every_x_iteration == 0:

        return True

    else:

        return False

    





def init_triggers(step = 1, train = 10, valid = 10):

    do_step_trigger = partial(iteration_trigger, every_x_iteration = step)

    train_loss_trigger = partial(iteration_trigger, every_x_iteration = train)

    valid_loss_trigger = partial(iteration_trigger, every_x_iteration = valid)

    

    return do_step_trigger, train_loss_trigger, valid_loss_trigger





do_step_trigger, train_loss_trigger, valid_loss_trigger = init_triggers(1, 10, 20)




def train_one_epoch(model, 

                    train_data_loader, 

                    valid_data_loader, 

                    loss, 

                    optimizer, 

                    loss_writer_train, 

                    loss_writer_valid,

                    do_step_trigger,

                    train_loss_trigger,

                    valid_loss_trigger):

    

    y_true_train, y_hat_train = [], []

    for iteration, (x, y) in enumerate(train_data_loader):

        x_train = Variable(T(x), requires_grad = True)

        y_train = Variable(T(y), requires_grad = True)

        

        output = model(x_train)

        y_true_train.append(to_numpy(y_train))

        y_hat_train.append(to_numpy(output))

        loss_values = loss(output, y_train)

        loss_values.backward()

        

        #делаем шаг на каждой итерации и сбрасываем градиент

        if do_step_trigger(iteration):

            optimizer.step()

            optimizer.zero_grad()

        

        # проверяем, если итерация кратна train_step = 10, то тогда записываем в лог значение roc_auc

        if train_loss_trigger(iteration):

            print('train_loss_trigger: ')

            loss_writer_train(y_true_train, y_hat_train, iteration)

            y_true_train, y_hat_train = [], []

        

        # проверяем, если итерация кратна valid_step = 20, то тогда записываем в лог значение roc_auc

        if valid_loss_trigger(iteration):

            print('valid_loss_trigger:')

            y_true_valid, y_hat_valid = predict(model, valid_data_loader)

            loss_writer_valid(y_true_valid, y_hat_valid, iteration)

        

    return model


resnet9 = train_one_epoch(resnet9, 

                    train_dataloader, 

                    valid_dataloader, 

                    loss, 

                    optimizer, 

                    loss_writer_train, 

                    loss_writer_valid,

                    do_step_trigger,

                    train_loss_trigger,

                    valid_loss_trigger)
TEST_IMAGES_FOLDER = f'{DATA_FOLDER}/test/'



# преобразуем исходные данные сначала в Image, затем в Tensor



#сделаем функцию, которая возвращает список названий картинок в папке test

def test_image_collection(directory: str) -> List:

    images_name = []

    for filename in os.listdir(directory):

        images_name.append(TEST_IMAGES_FOLDER + filename)

    return(images_name)



test_image = test_image_collection(TEST_IMAGES_FOLDER)

test_images_dataset = ImageDataset(test_image)    



# зададим форматирование и перевод в тензорный вид наших тестовых данных 

class TestDataset(Dataset):

    def __init__(self, x_dataset: Dataset, x_tfms: Optional = None):

        self.x_dataset = x_dataset

        self.x_tfms = x_tfms

        

    def __len__(self) -> int:

        return self.x_dataset.__len__() 

        

    def __getitem__(self, index: int) -> Tuple:

        x = self.x_dataset[index]

        if x_tfms is not None:

            x = self.x_tfms(x)

        return x

    

test_dataset = TestDataset(test_images_dataset, x_tfms)    



# загружаем данные в data loader, определяем число батчей

batch_size = 512

num_workers = 0

shuffle = False



test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
# определим функцию предсказания для тестовой выборки

def predict_test(model, dataloader):

    model.eval()

    y_hat = []

    

    for x in dataloader:

        x = Variable(T(x))

        output = model(x)

        

        y_hat.append(to_numpy(output))

    return y_hat
# сделаем предсказания для тестовой выборки

y_hat_test = predict_test(resnet9, test_dataloader)
# запишем ответы в DataFrame

predictions = pd.DataFrame(

    list(

        zip(

            test_image,

            np.vstack(y_hat_test).reshape(-1)

        )

    ), 

     columns=['id', 'label'])

predictions['id'] = predictions['id'].apply(lambda x: x.split('/')[-1].split('.')[0]) 
predictions.to_csv('submission.csv', index=False)
predictions
#определим метод для отображения картинок

max_charts = 60

def implot_errors(files, w=2, h=2, cols=12):

    rows = len(files) / cols + 1

    images = [Image.open(f) for f in files]

    plt.figure(figsize = (cols * w, rows * h))

    plt.tight_layout()

    for chart, img in enumerate(images, 1):

        ax = plt.subplot(rows, cols, chart)

        ax.imshow(np.array(img))

        ax.axis('off')
# сделаем таблицу с предсказаниями и реальными значениями на валидационной выборке

y_true, y_hat = predict(resnet9, valid_dataloader)



predictions_comparison = pd.DataFrame(

    list(

        zip(

            valid_labels.reshape(-1), 

            np.vstack(y_hat).reshape(-1),

            valid_images

        )

    ), 

     columns=['true', 'pred', 'files'])



predictions_comparison.head(3)

files = predictions_comparison[predictions_comparison['true']==1].sort_values('pred')['files'].values[:max_charts]

implot_errors(files)
files = predictions_comparison[predictions_comparison['true']==0].sort_values('pred', ascending=False)['files'].values[:max_charts]

implot_errors(files)
files = predictions_comparison[predictions_comparison['true']==1].sort_values('pred', ascending=False)['files'].values[:max_charts]

implot_errors(files)
files = predictions_comparison[predictions_comparison['true']==0].sort_values('pred', ascending=True)['files'].values[:max_charts]

implot_errors(files)