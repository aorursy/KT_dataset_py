# Regular Imports

import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2



import pydicom # for DICOM images

from skimage.transform import resize



# SKLearn

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



import warnings

warnings.filterwarnings("ignore")



# Set Color Palettes for the notebook

colors_nude = ['#e0798c','#65365a','#da8886','#cfc4c4','#dfd7ca']

sns.palplot(sns.color_palette(colors_nude))



# Set Style

sns.set_style("whitegrid")

sns.despine(left=True, bottom=True)
list(os.listdir('../input/siim-isic-melanoma-classification'))
# Directory

directory = '../input/siim-isic-melanoma-classification'



# Import the 2 csv s

train_df = pd.read_csv(directory + '/train.csv')

test_df = pd.read_csv(directory + '/test.csv')



print('Train has {:,} rows and Test has {:,} rows.'.format(len(train_df), len(test_df)))



# Change columns names

new_names = ['dcm_name', 'ID', 'sex', 'age', 'anatomy', 'diagnosis', 'benign_malignant', 'target']

train_df.columns = new_names

test_df.columns = new_names[:5]
df1_styler = train_df.head().style.set_table_attributes("style='display:inline'").set_caption('Head Train Data')

df2_styler = test_df.head().style.set_table_attributes("style='display:inline'").set_caption('Head Test Data')



display_html(df1_styler._repr_html_() + df2_styler._repr_html_(), raw=True)
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



msno.matrix(train_df, ax = ax1, color=(207/255, 196/255, 171/255), fontsize=10)

msno.matrix(test_df, ax = ax2, color=(218/255, 136/255, 130/255), fontsize=10)



ax1.set_title('Train Missing Values Map', fontsize = 16)

ax2.set_title('Test Missing Values Map', fontsize = 16);
# Data

nan_sex = train_df[train_df['sex'].isna() == True]

is_sex = train_df[train_df['sex'].isna() == False]



# Figure

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



a = sns.countplot(nan_sex['anatomy'], ax = ax1, palette=colors_nude)

b = sns.countplot(is_sex['anatomy'], ax = ax2, palette=colors_nude)

ax1.set_title('NAN Gender: Anatomy', fontsize=16)

ax2.set_title('Rest Gender: Anatomy', fontsize=16)



a.set_xticklabels(a.get_xticklabels(), rotation=35, ha="right")

b.set_xticklabels(b.get_xticklabels(), rotation=35, ha="right")

sns.despine(left=True, bottom=True);



# Benign/ Malignant check

print('Out of 65 NAN values, {} are benign and 0 malignant.'.format(nan_sex['benign_malignant'].value_counts()[0]))
# Check how many are males and how many females

anatomy = ['lower extremity', 'upper extremity', 'torso']

train_df[(train_df['anatomy'].isin(anatomy)) & (train_df['target'] == 0)]['sex'].value_counts()



# Impute the missing values with male

train_df['sex'].fillna("male", inplace = True) 
# Data

nan_age = train_df[train_df['age'].isna() == True]

is_age = train_df[train_df['age'].isna() == False]



# Figure

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



a = sns.countplot(nan_age['anatomy'], ax = ax1, palette=colors_nude)

b = sns.countplot(is_age['anatomy'], ax = ax2, palette=colors_nude)

ax1.set_title('NAN age: Anatomy', fontsize=16)

ax2.set_title('Rest age: Anatomy', fontsize=16)



a.set_xticklabels(a.get_xticklabels(), rotation=35, ha="right")

b.set_xticklabels(b.get_xticklabels(), rotation=35, ha="right")

sns.despine(left=True, bottom=True);



# Benign/ Malignant check

print('Out of 68 NAN values, {} are benign and 0 malignant.'.format(nan_age['benign_malignant'].value_counts()[0]))
# Check the mean age

anatomy = ['lower extremity', 'upper extremity', 'torso']

median = train_df[(train_df['anatomy'].isin(anatomy)) & (train_df['target'] == 0) & (train_df['sex'] == 'male')]['age'].median()

print('Median is:', median)



# Impute the missing values with male

train_df['age'].fillna(median, inplace = True) 
anatomy = train_df.copy()

anatomy['flag'] = np.where(train_df['anatomy'].isna()==True, 'missing', 'not_missing')



# Figure

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



sns.countplot(anatomy['flag'], hue=anatomy['sex'], ax=ax1, palette=colors_nude)



sns.distplot(anatomy[anatomy['flag'] == 'missing']['age'], 

             hist=False, rug=True, label='Missing', ax=ax2, 

             color=colors_nude[2], kde_kws=dict(linewidth=4))

sns.distplot(anatomy[anatomy['flag'] == 'not_missing']['age'], 

             hist=False, rug=True, label='Not Missing', ax=ax2, 

             color=colors_nude[3], kde_kws=dict(linewidth=4))



ax1.set_title('Gender for Anatomy', fontsize=16)

ax2.set_title('Age Distribution for Anatomy', fontsize=16)

sns.despine(left=True, bottom=True);



# Benign - malignant

ben_mal = anatomy[anatomy['flag'] == 'missing']['benign_malignant'].value_counts()

print('From all missing values, {} are benign and {} malignant.'.format(ben_mal[0], ben_mal[1]))
# Impute for anatomy

train_df['anatomy'].fillna('torso', inplace = True) 
anatomy = test_df.copy()

anatomy['flag'] = np.where(test_df['anatomy'].isna()==True, 'missing', 'not_missing')



# Figure

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



sns.countplot(anatomy['flag'], hue=anatomy['sex'], ax=ax1, palette=colors_nude)



sns.distplot(anatomy[anatomy['flag'] == 'missing']['age'],

             hist=False, rug=True, label='Missing', ax=ax2, 

             color=colors_nude[2], kde_kws=dict(linewidth=4, bw=0.1))



sns.distplot(anatomy[anatomy['flag'] == 'not_missing']['age'], 

             hist=False, rug=True, label='Not Missing', ax=ax2, 

             color=colors_nude[3], kde_kws=dict(linewidth=4, bw=0.1))



ax1.set_title('Gender for Anatomy', fontsize=16)

ax2.set_title('Age Distribution for Anatomy', fontsize=16)

sns.despine(left=True, bottom=True);
# Select most frequent anatomy for age 70

value = test_df[test_df['age'] == 70]['anatomy'].value_counts().reset_index()['index'][0]



# Impute the value

test_df['anatomy'].fillna(value, inplace = True) 
# Save the files

train_df.to_csv('train_clean.csv', index=False)

test_df.to_csv('test_clean.csv', index=False)
# Figure

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



a = sns.countplot(data = train_df, x = 'benign_malignant', palette=colors_nude[2:4],

                 ax=ax1)

b = sns.distplot(a = train_df[train_df['target']==0]['age'], ax=ax2, color=colors_nude[2], 

                 hist=False, rug=True, kde_kws=dict(linewidth=4), label='Benign')

c = sns.distplot(a = train_df[train_df['target']==1]['age'], ax=ax2, color=colors_nude[3], 

                 hist=False, rug=True, kde_kws=dict(linewidth=4), label='Malignant')



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')

    

ax1.set_title('Frequency for Target Variable', fontsize=16)

ax2.set_title('Age Distribution the Target types', fontsize=16)

sns.despine(left=True, bottom=True);
plt.figure(figsize=(16, 6))

a = sns.countplot(data=train_df, x='benign_malignant', hue='sex', palette=colors_nude)



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')



plt.title('Gender split by Target Variable', fontsize=16)

sns.despine(left=True, bottom=True);
# Delete 'atypical melanocytic proliferation','cafe-au-lait macule'

# train_df = train_df[~train_df['diagnosis'].isin(['atypical melanocytic proliferation','cafe-au-lait macule'])]



# Figure

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



a = sns.countplot(train_df['anatomy'], ax=ax1, palette = colors_nude)

b = sns.countplot(train_df['diagnosis'], ax=ax2, palette = colors_nude)



a.set_xticklabels(a.get_xticklabels(), rotation=35, ha="right")

b.set_xticklabels(b.get_xticklabels(), rotation=35, ha="right")



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')

    

for p in b.patches:

    b.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')

    

ax1.set_title('Anatomy Frequencies', fontsize=16)

ax2.set_title('Diagnosis Frequencies', fontsize=16)

sns.despine(left=True, bottom=True);
plt.figure(figsize=(16, 6))

a = sns.countplot(data=train_df, x='benign_malignant', hue='anatomy', palette=colors_nude)



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')



plt.title('Anatomy split by Target Variable', fontsize=16)

sns.despine(left=True, bottom=True);
# Figure

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



a = sns.countplot(train_df[train_df['target']==0]['diagnosis'], ax=ax1, palette = colors_nude)

b = sns.countplot(train_df[train_df['target']==1]['diagnosis'], ax=ax2, palette = colors_nude)



a.set_xticklabels(a.get_xticklabels(), rotation=35, ha="right")

b.set_xticklabels(b.get_xticklabels(), rotation=35, ha="right")



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')

    

for p in b.patches:

    b.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')

    

ax1.set_title('Benign cases: Diagnosis view', fontsize=16)

ax2.set_title('Malignant cases: Diagnosis view', fontsize=16)

sns.despine(left=True, bottom=True);
# Figure

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16, 6))



a = sns.countplot(test_df['sex'], palette=colors_nude, ax=ax1)

b = sns.countplot(test_df['anatomy'], ax=ax2, palette = colors_nude)

c = sns.distplot(a = test_df['age'], ax=ax3, color=colors_nude[3], 

                 hist=False, rug=True, kde_kws=dict(linewidth=4))



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')

    

for p in b.patches:

    b.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')

    

b.set_xticklabels(b.get_xticklabels(), rotation=35, ha="right")



ax1.set_title('Test: Gender Frequencies', fontsize=16)

ax2.set_title('Test: Anatomy Frequencies', fontsize=16)

ax3.set_title('Test: Age Distribution', fontsize=16)

sns.despine(left=True, bottom=True);
# Count the number of images per ID

patients_count_train = train_df.groupby(by='ID')['dcm_name'].count().reset_index()

patients_count_test = test_df.groupby(by='ID')['dcm_name'].count().reset_index()



# Figure

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



a = sns.distplot(patients_count_train['dcm_name'], kde=False, bins=50, 

                 ax=ax1, color=colors_nude[0], hist_kws={'alpha': 1})

b = sns.distplot(patients_count_test['dcm_name'], kde=False, bins=50, 

                 ax=ax2, color=colors_nude[1], hist_kws={'alpha': 1})

    

ax1.set_title('Train: Images per Patient Distribution', fontsize=16)

ax2.set_title('Test: Images per Patient Distribution', fontsize=16)

sns.despine(left=True, bottom=True);
# Save the files

train_df.to_csv('train_clean.csv', index=False)

test_df.to_csv('test_clean.csv', index=False)
# === DICOM ===

# Create the paths

path_train = directory + '/train/' + train_df['dcm_name'] + '.dcm'

path_test = directory + '/test/' + test_df['dcm_name'] + '.dcm'



# Append to the original dataframes

train_df['path_dicom'] = path_train

test_df['path_dicom'] = path_test



# === JPEG ===

# Create the paths

path_train = directory + '/jpeg/train/' + train_df['dcm_name'] + '.jpg'

path_test = directory + '/jpeg/test/' + test_df['dcm_name'] + '.jpg'



# Append to the original dataframes

train_df['path_jpeg'] = path_train

test_df['path_jpeg'] = path_test
# === TRAIN ===

to_encode = ['sex', 'anatomy', 'diagnosis']

encoded_all = []



label_encoder = LabelEncoder()



for column in to_encode:

    encoded = label_encoder.fit_transform(train_df[column])

    encoded_all.append(encoded)

    

train_df['sex'] = encoded_all[0]

train_df['anatomy'] = encoded_all[1]

train_df['diagnosis'] = encoded_all[2]



if 'benign_malignant' in train_df.columns : train_df.drop(['benign_malignant'], axis=1, inplace=True)
# === TEST ===

to_encode = ['sex', 'anatomy']

encoded_all = []



label_encoder = LabelEncoder()



for column in to_encode:

    encoded = label_encoder.fit_transform(test_df[column])

    encoded_all.append(encoded)

    

test_df['sex'] = encoded_all[0]

test_df['anatomy'] = encoded_all[1]
# Save the files

train_df.to_csv('train_clean.csv', index=False)

test_df.to_csv('test_clean.csv', index=False)
print('Train .dcm number of images:', len(list(os.listdir('../input/siim-isic-melanoma-classification/train'))), '\n' +

      'Test .dcm number of images:', len(list(os.listdir('../input/siim-isic-melanoma-classification/test'))), '\n' +

      'Train .jpeg number of images:', len(list(os.listdir('../input/siim-isic-melanoma-classification/jpeg/train'))), '\n' +

      'Test .jpeg number of images:', len(list(os.listdir('../input/siim-isic-melanoma-classification/jpeg/test'))), '\n' +

      '-----------------------', '\n' +

      'There is the same number of images as in train/ test .csv datasets')
shapes_train = []



for k, path in enumerate(train_df['path_jpeg']):

    image = Image.open(path)

    shapes_train.append(image.size)

    

    if k >= 100: break

        

shapes_train = pd.DataFrame(data = shapes_train, columns = ['H', 'W'], dtype='object')

shapes_train['Size'] = '[' + shapes_train['H'].astype(str) + ', ' + shapes_train['W'].astype(str) + ']'
plt.figure(figsize = (16, 6))



a = sns.countplot(shapes_train['Size'], palette=colors_nude)



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')

    

plt.title('100 Images Shapes', fontsize=16)

sns.despine(left=True, bottom=True);
def show_images(data, n = 5, rows=1, cols=5, title='Default'):

    plt.figure(figsize=(16,4))



    for k, path in enumerate(data['path_dicom'][:n]):

        image = pydicom.read_file(path)

        image = image.pixel_array

        

        # image = resize(image, (200, 200), anti_aliasing=True)



        plt.suptitle(title, fontsize = 16)

        plt.subplot(rows, cols, k+1)

        plt.imshow(image)

        plt.axis('off')
# Show Benign Samples

show_images(train_df[train_df['target'] == 0], n=10, rows=2, cols=5, title='Benign Sample')
# Show Malignant Samples

show_images(train_df[train_df['target'] == 1], n=10, rows=2, cols=5, title='Malignant Sample')
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16,6))

plt.suptitle("B&W", fontsize = 16)



for i in range(0, 2*6):

    data = pydicom.read_file(train_df['path_dicom'][i])

    image = data.pixel_array

    

    # Transform to B&W

    # The function converts an input image from one color space to another.

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (200,200))

    

    x = i // 6

    y = i % 6

    axes[x, y].imshow(image, cmap=plt.cm.bone) 

    axes[x, y].axis('off')
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16,6))

plt.suptitle("Without Gaussian Blur", fontsize = 16)



for i in range(0, 2*6):

    data = pydicom.read_file(train_df['path_dicom'][i])

    image = data.pixel_array

    

    # Transform to B&W

    # The function converts an input image from one color space to another.

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image = cv2.resize(image, (200,200))

    

    x = i // 6

    y = i % 6

    axes[x, y].imshow(image, cmap=plt.cm.bone) 

    axes[x, y].axis('off')
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16,6))

plt.suptitle("With Gaussian Blur", fontsize = 16)



for i in range(0, 2*6):

    data = pydicom.read_file(train_df['path_dicom'][i])

    image = data.pixel_array

    

    # Transform to B&W

    # The function converts an input image from one color space to another.

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image = cv2.resize(image, (200,200))

    image=cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0) ,256/10), -4, 128)

    

    x = i // 6

    y = i % 6

    axes[x, y].imshow(image, cmap=plt.cm.bone) 

    axes[x, y].axis('off')
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16,6))

plt.suptitle("Hue, Saturation, Brightness", fontsize = 16)



for i in range(0, 2*6):

    data = pydicom.read_file(train_df['path_dicom'][i])

    image = data.pixel_array

    

    # Transform to B&W

    # The function converts an input image from one color space to another.

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    image = cv2.resize(image, (200,200))

    

    x = i // 6

    y = i % 6

    axes[x, y].imshow(image, cmap=plt.cm.bone) 

    axes[x, y].axis('off')
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16,6))

plt.suptitle("LUV Color Space", fontsize = 16)



for i in range(0, 2*6):

    data = pydicom.read_file(train_df['path_dicom'][i])

    image = data.pixel_array

    

    # Transform to B&W

    # The function converts an input image from one color space to another.

    image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

    image = cv2.resize(image, (200,200))

    

    x = i // 6

    y = i % 6

    axes[x, y].imshow(image, cmap=plt.cm.bone) 

    axes[x, y].axis('off')
# Necessary Imports

import torch

from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms

import torchvision
# Select a small sample of the .jpeg image paths

image_list = train_df.sample(12)['path_jpeg']

image_list = image_list.reset_index()['path_jpeg']



# Show the sample

plt.figure(figsize=(16,6))

plt.suptitle("Original View", fontsize = 16)

    

for k, path in enumerate(image_list):

    image = mpimg.imread(path)

        

    plt.subplot(2, 6, k+1)

    plt.imshow(image)

    plt.axis('off')
# Create PyTorch Dataset Object

class DatasetExample(Dataset):

    def __init__(self, image_list, transforms=None):

        self.image_list = image_list

        self.transforms = transforms

    

    # To get item's length

    def __len__(self):

        return (len(self.image_list))

    

    # For indexing

    def __getitem__(self, i):

        # Read in image

        image = plt.imread(self.image_list[i])

        image = Image.fromarray(image).convert('RGB')        

        image = np.asarray(image).astype(np.uint8)

        if self.transforms is not None:

            image = self.transforms(image)

            

        return torch.tensor(image, dtype=torch.float)
# Predefined Show Images Function

def show_transform(image, title="Default"):

    plt.figure(figsize=(16,6))

    plt.suptitle(title, fontsize = 16)

    

    # Unnormalize

    image = image / 2 + 0.5  

    npimg = image.numpy()

    npimg = np.clip(npimg, 0., 1.)

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()
# Transform

transform = transforms.Compose([

     transforms.ToPILImage(),

     transforms.Resize((300, 300)),

     transforms.CenterCrop((100, 100)),

     transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

     ])



# Create the dataset

pytorch_dataset = DatasetExample(image_list=image_list, transforms=transform)

pytorch_dataloader = DataLoader(dataset=pytorch_dataset, batch_size=12, shuffle=True)



# Select the data

images = next(iter(pytorch_dataloader))

 

# show images

show_transform(torchvision.utils.make_grid(images, nrow=6), title="Crop")
# Transform

transform = transforms.Compose([

     transforms.ToPILImage(),

     transforms.Resize((300, 300)),

     transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),

     transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

     ])



# Create the dataset

pytorch_dataset = DatasetExample(image_list=image_list, transforms=transform)

pytorch_dataloader = DataLoader(dataset=pytorch_dataset, batch_size=12, shuffle=True)



# Select the data

images = next(iter(pytorch_dataloader))

 

# show images

show_transform(torchvision.utils.make_grid(images, nrow=6), title="Color Jitter")
# Transform

transform = transforms.Compose([

     transforms.ToPILImage(),

     transforms.Resize((300, 300)),

     transforms.RandomGrayscale(p=0.7),

     transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

     ])



# Create the dataset

pytorch_dataset = DatasetExample(image_list=image_list, transforms=transform)

pytorch_dataloader = DataLoader(dataset=pytorch_dataset, batch_size=12, shuffle=True)



# Select the data

images = next(iter(pytorch_dataloader))

 

# show images

show_transform(torchvision.utils.make_grid(images, nrow=6), title="Random Greyscale")
# Transform

transform = transforms.Compose([

     transforms.ToPILImage(),

     transforms.Resize((300, 300)),

     transforms.RandomVerticalFlip(p=0.7),

     transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

     ])



# Create the dataset

pytorch_dataset = DatasetExample(image_list=image_list, transforms=transform)

pytorch_dataloader = DataLoader(dataset=pytorch_dataset, batch_size=12, shuffle=True)



# Select the data

images = next(iter(pytorch_dataloader))

 

# show images

show_transform(torchvision.utils.make_grid(images, nrow=6), title="Random Vertical Flip")
def hair_remove(image):

    # convert image to grayScale

    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)



    # kernel for morphologyEx

    kernel = cv2.getStructuringElement(1,(17,17))



    # apply MORPH_BLACKHAT to grayScale image

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)



    # apply thresholding to blackhat

    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)



    # inpaint with original image and threshold image

    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)



    return final_image
# Select a small sample of the .jpeg image paths

# We select some hairy photos on purpose

hairy_photos = train_df[train_df["sex"] == 1].reset_index().iloc[[12, 14, 17, 22, 33, 34]]

image_list = hairy_photos['path_jpeg']

image_list = image_list.reset_index()['path_jpeg']
# Show the Augmented Images

plt.figure(figsize=(16,3))

plt.suptitle("Original Hairy Images", fontsize = 16)

    

for k, path in enumerate(image_list):

    image = mpimg.imread(path)

    image = cv2.resize(image,(300, 300))

        

    plt.subplot(1, 6, k+1)

    plt.imshow(image)

    plt.axis('off')
# Show the sample

plt.figure(figsize=(16,3))

plt.suptitle("Non Hairy Images", fontsize = 16)

    

for k, path in enumerate(image_list):

    image = mpimg.imread(path)

    image = cv2.resize(image,(300, 300))

    image = hair_remove(image)

        

    plt.subplot(1, 6, k+1)

    plt.imshow(image)

    plt.axis('off')