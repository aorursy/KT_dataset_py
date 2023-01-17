from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('hXYd0WRhzN4',width=800, height=500)
#plotly

!pip install --upgrade pip --quiet

!pip install chart_studio --quiet



import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



import seaborn as sns

sns.set(style="whitegrid")



#pydicom

import pydicom

# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')





# Settings for pretty nice plots

plt.style.use('fivethirtyeight')

plt.show()
os.listdir('../input/siim-isic-melanoma-classification/')

BASE_PATH = '../input/siim-isic-melanoma-classification'







print('Reading data...')

train = pd.read_csv(f'{BASE_PATH}/train.csv')

test = pd.read_csv(f'{BASE_PATH}/test.csv')

submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

print('Reading data completed')
display(train.head())

print("Shape of train :", train.shape)
display(test.head())

print("Shape of test :", test.shape)
# checking missing data

total = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)

missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head()
# checking missing data

total = test.isnull().sum().sort_values(ascending = False)

percent = (test.isnull().sum()/test.isnull().count()*100).sort_values(ascending = False)

missing_test_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_test_data.head()
def plot_count(df, feature, title='', size=2.5):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    total = float(len(df))

    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set2')

    plt.title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count(train, 'benign_malignant')
plot_count(train, 'sex')


plot_count(train, 'anatom_site_general_challenge')



train['diagnosis'].value_counts(normalize=True).sort_values().iplot(kind='barh',

                                                      xTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.7,

                                                      color='blue',

                                                      theme='pearl',

                                                      bargap=0.2,

                                                      gridcolor='white',

                                                      title='Distribution in the training set'

                                                    )
def plot_relative_distribution(df, feature, hue, title='', size=2):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    total = float(len(df))

    sns.countplot(x=feature, hue=hue, data=df, palette='Set2')

    plt.title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_relative_distribution(

    df=train,

    feature='sex',

    hue='benign_malignant',

    title = 'relative count plot of sex with benign_malignant',

    size=2.8

)
plot_relative_distribution(

    df=train,

    feature='anatom_site_general_challenge',

    hue='benign_malignant',

    title = 'relative count plot of anatom_site_general_challenge with benign_malignant',

    size=3

)
train['age_approx'].iplot(

    kind='hist',

    bins=30,

    color='blue',

    xTitle='Age',

    yTitle='Count',

    title='Age Distribution'

)
import PIL

from PIL import Image, ImageDraw





def display_images(images, title=None): 

    f, ax = plt.subplots(5,3, figsize=(18,22))

    if title:

        f.suptitle(title, fontsize = 30)



    for i, image_id in enumerate(images):

        image_path = os.path.join(BASE_PATH, f'jpeg/train/{image_id}.jpg')

        image = Image.open(image_path)

        

        ax[i//3, i%3].imshow(image) 

        image.close()       

        ax[i//3, i%3].axis('off')



        benign_malignant = train[train['image_name'] == image_id]['benign_malignant'].values[0]

        ax[i//3, i%3].set_title(f"image_name: {image_id}\nSource: {benign_malignant}", fontsize="15")



    plt.show() 
benign = train[train.benign_malignant == 'benign'].sample(n=15, random_state=42)

display_images(benign.image_name.values, title = 'benign images')
malignant = train[train.benign_malignant == 'malignant'].sample(n=15, random_state=42)

display_images(malignant.image_name.values, title='malignant images')
female_patients = train[train.sex == 'female']

benign = female_patients[female_patients.benign_malignant == 'benign'].sample(n=15, random_state=42)

display_images(benign.image_name.values, title='benign images for female patients')
malignant = female_patients[female_patients.benign_malignant == 'malignant'].sample(n=15, random_state=42)

display_images(malignant.image_name.values, title='malignant images for female patients')
male_patients = train[train.sex == 'male']

benign = male_patients[male_patients.benign_malignant == 'benign'].sample(n=15, random_state=42)

display_images(benign.image_name.values, title='benign images for male patients')
malignant = male_patients[male_patients.benign_malignant == 'malignant'].sample(n=15, random_state=42)

display_images(malignant.image_name.values, title='malignant images for male patients')
anatom_sites = [ site for site in list(train.anatom_site_general_challenge.unique()) if type(site) != float ]
for site in anatom_sites[:2]:

    site_df = train[train.anatom_site_general_challenge == site].sample(n=15, random_state=42)

    display_images(site_df.image_name.values, title = f'patient images for anatom_site == {site}')
!pip install --upgrade pip --quiet

!pip install wtfml --quiet

!pip install pretrainedmodels --quiet
import os

import torch

import albumentations



import torch.nn as nn

from sklearn import metrics

from sklearn import model_selection

from torch.nn import functional as F



from wtfml.utils import EarlyStopping

from wtfml.engine import Engine

from wtfml.data_loaders.image import ClassificationLoader



import pretrainedmodels
BASE_PATH = "../input/siim-isic-melanoma-classification"

DATA_PATH = "../input/siic-isic-224x224-images/test/"

MODEL_PATH = "../input/melanoma-resnext50"



df_test = pd.read_csv(f"{BASE_PATH}/test.csv")



device = torch.device("cuda")

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)

aug = albumentations.Compose([

    albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)

])
class SEResnext50_32x4d(nn.Module):

    def __init__(self, pretrained='imagenet'):

        super(SEResnext50_32x4d, self).__init__()

        

        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=None)

        self.out = nn.Linear(2048, 1)

    

    def forward(self, image, targets):

        bs, _, _, _ = image.shape

        

        x = self.model.features(image)

        x = F.adaptive_avg_pool2d(x, 1)

        x = x.reshape(bs, -1)

        

        out = self.out(x)

        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))



        return out, loss
ENSEMBLES = [

    {'model': SEResnext50_32x4d(pretrained=None), 'state_dict': f'{MODEL_PATH}/model_0.bin', 'weight': 1},

    {'model': SEResnext50_32x4d(pretrained=None), 'state_dict': f'{MODEL_PATH}/model_1.bin', 'weight': 1},

    {'model': SEResnext50_32x4d(pretrained=None), 'state_dict': f'{MODEL_PATH}/model_2.bin', 'weight': 1},

    {'model': SEResnext50_32x4d(pretrained=None), 'state_dict': f'{MODEL_PATH}/model_3.bin', 'weight': 1},

    {'model': SEResnext50_32x4d(pretrained=None), 'state_dict': f'{MODEL_PATH}/model_4.bin', 'weight': 1},

]
models = []



for ensemble in ENSEMBLES:

    model = ensemble['model']

    model_path = ensemble['state_dict']

    model.load_state_dict(torch.load(model_path))

    model.to(device)

    models.append(model)
images = df_test.image_name.values.tolist()

images = [os.path.join(DATA_PATH, i + ".png") for i in images]

targets = np.zeros(len(images))



test_dataset = ClassificationLoader(

    image_paths=images,

    targets=targets,

    resize=None,

    augmentations=aug,

)



test_loader = torch.utils.data.DataLoader(

    test_dataset, batch_size=48, shuffle=False, num_workers=4

)
predictions = []



for model in models:

    prediction = Engine.predict(test_loader, model, device=device)

    prediction = np.vstack((prediction)).ravel()

    predictions.append(prediction)
predictions = np.mean(predictions, axis=0)

sample = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

sample.loc[:, "target"] = predictions

sample.to_csv("submission.csv", index=False)
sample.head()