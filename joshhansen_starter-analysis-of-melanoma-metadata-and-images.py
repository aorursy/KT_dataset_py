# loading packages

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
import plotly.express as px

import os

from tqdm import tqdm
# Setting color palette.
orange_black = [
    '#fdc029', '#df861d', '#FF6347', '#aa3d01', '#a30e15', '#800000', '#171820'
]

# Detting plot styling.
plt.style.use('ggplot')
# setting file paths

base_path = '/mnt/data/kaggle/competitions/siim-isic-melanoma-classification'
train_img_path = f"{base_path}/jpeg/train/"
test_img_path = f"{base_path}/jpeg/test"
img_stats_path = f"{base_path}/melanoma2020imgtabular"
# Loading train and test data.

train = pd.read_csv(os.path.join(base_path, 'train.csv'))
test = pd.read_csv(os.path.join(base_path, 'test.csv'))
# Checking columns.

print(
    f'Train data has {train.shape[1]} features, {train.shape[0]} observations and Test data {test.shape[1]} features, {test.shape[0]} observations.\nTrain features are:\n{train.columns.tolist()}\nTest features are:\n{test.columns.tolist()}'
)
# Renaming columns.

train.columns = [
    'img_name', 'id', 'sex', 'age', 'location', 'diagnosis',
    'benign_malignant', 'target'
]
test.columns = ['img_name', 'id', 'sex', 'age', 'location']
# taking samples from train data
train.sample(5)
# Taking samples from test data:

test.sample(5)
# Checking missing values:

def missing_percentage(df):

    total = df.isnull().sum().sort_values(
        ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = (df.isnull().sum().sort_values(ascending=False) / len(df) *
               100)[(df.isnull().sum().sort_values(ascending=False) / len(df) *
                     100) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


missing_train = missing_percentage(train)
missing_test = missing_percentage(test)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(x=missing_train.index,
            y='Percent',
            data=missing_train,
            palette=orange_black,
            ax=ax[0])
sns.barplot(x=missing_test.index,
            y='Percent',
            data=missing_test,
            palette=orange_black,
            ax=ax[1])

ax[0].set_title('Train Data Missing Values')
ax[1].set_title('Test Data Missing Values')
# Creating a customized chart and giving in figsize etc.

fig = plt.figure(constrained_layout=True, figsize=(20, 9))

# Creating a grid

grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

ax1 = fig.add_subplot(grid[0, :2])

# Set the title.

ax1.set_title('Gender Distribution')

sns.countplot(train.sex.sort_values(ignore_index=True),
              alpha=0.9,
              ax=ax1,
              color='#fdc029',
              label='Train')
sns.countplot(test.sex.sort_values(ignore_index=True),
              alpha=0.7,
              ax=ax1,
              color='#171820',
              label='Test')
ax1.legend()

# Customizing the second grid.

ax2 = fig.add_subplot(grid[0, 2:])

# Plot the countplot.

sns.countplot(train.location,
              alpha=0.9,
              ax=ax2,
              color='#fdc029',
              label='Train',
              order=train['location'].value_counts().index)
sns.countplot(test.location,
              alpha=0.7,
              ax=ax2,
              color='#171820',
              label='Test',
              order=test['location'].value_counts().index), ax2.set_title(
                  'Anatom Site Distribution')

ax2.legend()

# Customizing the third grid.

ax3 = fig.add_subplot(grid[1, :])

# Set the title.

ax3.set_title('Age Distribution')

# Plot the histogram.

sns.distplot(train.age, ax=ax3, label='Train', color='#fdc029')
sns.distplot(test.age, ax=ax3, label='Test', color='#171820')

ax3.legend()

plt.show()
# Filling anatom site.

for df in [train, test]:
    df['location'].fillna('unknown', inplace=True)
# Double checking:

ids_train = train.location.values
ids_test = test.location.values
ids_train_set = set(ids_train)
ids_test_set = set(ids_test)

location_not_overlap = list(ids_train_set.symmetric_difference(ids_test_set))
n_overlap = len(location_not_overlap)
if n_overlap == 0:
    print(
        f'There are no different body parts occuring between train and test set...'
    )
else:
    print('There are some not overlapping values between train and test set!')
# Filling age and sex.

train['sex'].fillna(train['sex'].mode()[0], inplace=True)

train['age'].fillna(train['age'].median(), inplace=True)
# Checking missing value counts:

print(
    f'Train missing value count: {train.isnull().sum().sum()}\nTest missing value count: {train.isnull().sum().sum()}'
)
# Train data:

cntstr = train.location.value_counts().rename_axis('location').reset_index(
    name='count')

fig = px.treemap(cntstr,
                 path=['location'],
                 values='count',
                 color='count',
                 color_continuous_scale=orange_black,
                 title='Scans by Anatom Site General Challenge - Train Data')

fig.update_traces(textinfo='label+percent entry')
fig.show()
# Test data:

cntste = test.location.value_counts().rename_axis('location').reset_index(
    name='count')

fig = px.treemap(cntste,
                 path=['location'],
                 values='count',
                 color='count',
                 color_continuous_scale=orange_black,
                 title='Scans by Anatom Site General Challenge - Test Data')

fig.update_traces(textinfo='label+percent entry')
fig.show()
# Creating a customized chart and giving in figsize etc.

fig = plt.figure(constrained_layout=True, figsize=(20, 9))
# Creating a grid
grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

# Customizing the first grid.

ax1 = fig.add_subplot(grid[1, :2])
# Set the title.
ax1.set_title('Scanned Body Parts - Female')

# Plot:

sns.countplot(
    train[train['sex'] == 'female'].location.sort_values(ignore_index=True),
    alpha=0.9,
    ax=ax1,
    color='#fdc029',
    label='Female',
    order=train['location'].value_counts().index)
ax1.legend()

# Customizing the second grid.

ax2 = fig.add_subplot(grid[1, 2:])

# Set the title.

ax2.set_title('Scanned Body Parts - Male')

# Plot.

sns.countplot(
    train[train['sex'] == 'male'].location.sort_values(ignore_index=True),
    alpha=0.9,
    ax=ax2,
    color='#171820',
    label='Male',
    order=train['location'].value_counts().index)

ax2.legend()

# Customizing the third grid.

ax3 = fig.add_subplot(grid[0, :])

# Set the title.

ax3.set_title('Malignant Ratio Per Body Part')

# Plot.

loc_freq = train.groupby('location')['target'].mean().sort_values(
    ascending=False)
sns.barplot(x=loc_freq.index, y=loc_freq, palette=orange_black, ax=ax3)

ax3.legend()

plt.show()
# Plotting interactive sunburst:

fig = px.sunburst(data_frame=train,
                  path=['benign_malignant', 'sex', 'location'],
                  color='sex',
                  color_discrete_sequence=orange_black,
                  maxdepth=-1,
                  title='Sunburst Chart Benign/Malignant > Sex > Location')

fig.update_traces(textinfo='label+percent parent')
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
fig.show()
# Plotting age vs sex vs target:

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.lineplot(x='age',
             y='target',
             data=train,
             ax=ax[0],
             hue='sex',
             palette=orange_black[:2],
             ci=None)
sns.boxplot(x='benign_malignant',
            y='age',
            data=train,
            ax=ax[1],
            hue='sex',
            palette=orange_black)

plt.legend(loc='lower right')

ax[0].set_title('Malignant Scan Frequency by Age')
ax[1].set_title('Scan Results by Age and Sex')

plt.show()
# Creating a customized chart and giving in figsize etc.

# Plotting age dist vs target and age dist vs datasets

fig = plt.figure(constrained_layout=True, figsize=(20, 12))

# Creating a grid

grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

# Customizing the first grid.

ax1 = fig.add_subplot(grid[0, :2])

# Set the title.

ax1.set_title('Age Distribution by Scan Outcome')

# Plot

ax1.legend()

sns.kdeplot(train[train['target'] == 0]['age'],
            shade=True,
            ax=ax1,
            color='#171820',
            label='Benign')
sns.kdeplot(train[train['target'] == 1]['age'],
            shade=True,
            ax=ax1,
            color='#fdc029',
            label='Malignant')

# Customizing second grid.

ax2 = fig.add_subplot(grid[0, 2:])

# Set the title.

ax2.set_title('Age Distribution by Train/Test Observations')

# Plot.

sns.kdeplot(train.age, label='Train', shade=True, ax=ax2, color='#171820')
sns.kdeplot(test.age, label='Test', shade=True, ax=ax2, color='#fdc029')

ax2.legend()

# Customizing third grid.

ax3 = fig.add_subplot(grid[1, :])

# Set the title.

ax3.set_title('Age Distribution by Gender')

# Plot

sns.distplot(train[train.sex == 'female'].age,
             ax=ax3,
             label='Female',
             color='#fdc029')
sns.distplot(train[train.sex == 'male'].age,
             ax=ax3,
             label='Male',
             color='#171820')
ax3.legend()

plt.show()
print(
    f'Number of unique Patient ID\'s in train set: {train.id.nunique()}, Total: {train.id.count()}\nNumber of unique Patient ID\'s in test set: {test.id.nunique()}, Total: {test.id.count()}'
)
train['age_min'] = train['id'].map(train.groupby(['id']).age.min())
train['age_max'] = train['id'].map(train.groupby(['id']).age.max())

test['age_min'] = test['id'].map(test.groupby(['id']).age.min())
test['age_max'] = test['id'].map(test.groupby(['id']).age.max())
train['n_images'] = train.id.map(train.groupby(['id']).img_name.count())
test['n_images'] = test.id.map(test.groupby(['id']).img_name.count())
# Creating a customized chart and giving in figsize etc.

fig = plt.figure(constrained_layout=True, figsize=(20, 12))

# Creating a grid

grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

# Customizing the first grid.

ax1 = fig.add_subplot(grid[0, :2])

# Set the title.

ax1.set_title('Number of Scans Distribution by Scan Outcome')

# Plot

sns.kdeplot(train[train['target'] == 0]['n_images'],
            shade=True,
            ax=ax1,
            color='#171820',
            label='Benign')
sns.kdeplot(train[train['target'] == 1]['n_images'],
            shade=True,
            ax=ax1,
            color='#fdc029',
            label='Malignant')

ax1.legend()

# Customizing the second grid.

ax2 = fig.add_subplot(grid[0, 2:])

# Set the title.

ax2.set_title('Number of Scans Distribution by Train/Test Observations')

# Plot

sns.kdeplot(train.n_images, label='Train', shade=True, ax=ax2, color='#171820')
sns.kdeplot(test.n_images, label='Test', shade=True, ax=ax2, color='#fdc029')
ax2.legend()

# Customizing the third grid.

ax3 = fig.add_subplot(grid[1, :])

# Set the title.

ax3.set_title('Malignant Scan Result Frequency by Number of Scans')

# Plot

z = train.groupby('n_images')['target'].mean()
sns.lineplot(x=z.index, y=z, color='#171820', ax=ax3)
ax3.legend()

plt.show()
diag = train.diagnosis.value_counts()
fig = px.pie(diag,
             values='diagnosis',
             names=diag.index,
             color_discrete_sequence=orange_black,
             hole=.4)
fig.update_traces(textinfo='percent+label', pull=0.05)
fig.show()
# Getting image sizes by using os:

for data, location in zip([train, test], [train_img_path, test_img_path]):
    images = data['img_name'].values
    sizes = np.zeros(images.shape[0])
    for i, path in enumerate(tqdm(images)):
        sizes[i] = os.path.getsize(os.path.join(location, f'{path}.jpg'))

    data['image_size'] = sizes
# Plotting image sizes:

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.kdeplot(train[train['target'] == 0]['image_size'],
            shade=True,
            ax=ax[0],
            color='#171820',
            label='Benign')
sns.kdeplot(train[train['target'] == 1]['image_size'],
            shade=True,
            ax=ax[0],
            color='#fdc029',
            label='Malignant')

sns.kdeplot(train.image_size,
            label='Train',
            shade=True,
            ax=ax[1],
            color='#171820')
sns.kdeplot(test.image_size,
            label='Test',
            shade=True,
            ax=ax[1],
            color='#fdc029')

ax[0].set_title('Scan Image Size Distribution by Scan Outcome')
ax[1].set_title('Scan Image Size Distribution by Train/Test Observations')

plt.show()
from keras.preprocessing import image

for data, location in zip([train, test],[train_img_path, test_img_path]):
    images = data['img_name'].values
    reds = np.zeros(images.shape[0])
    greens = np.zeros(images.shape[0])
    blues = np.zeros(images.shape[0])
    mean = np.zeros(images.shape[0])
    x = np.zeros(images.shape[0], dtype=int)
    y = np.zeros(images.shape[0], dtype=int)
    for i, path in enumerate(tqdm(images)):
        img = np.array(image.load_img(os.path.join(location, f'{path}.jpg')))

        reds[i] = np.mean(img[:,:,0].ravel())
        greens[i] = np.mean(img[:,:,1].ravel())
        blues[i] = np.mean(img[:,:,2].ravel())
        mean[i] = np.mean(img)
        x[i] = img.shape[1]
        y[i] = img.shape[0]

    data['reds'] = reds
    data['greens'] = greens
    data['blues'] = blues
    data['mean_colors'] = mean
    data['width'] = x
    data['height'] = y

train['total_pixels']= train['width']*train['height']
#test['total_pixels']= test['width'].astype(str)*test['height']
test['total_pixels']= test['width']*test['height']
# Loading color data:

# train_attr = pd.read_csv(
#     os.path.join(img_stats_path, 'train_mean_colorres.csv'))
# test_attr = pd.read_csv(os.path.join(img_stats_path, 'test_mean_colorres.csv'))
# train_attr.head()
# train = pd.concat([train, train_attr], axis=1)
# test = pd.concat([test, test_attr], axis=1)

train['res'] = train['width'].astype(str) + 'x' + train['height'].astype(str)
test['res'] = test['width'].astype(str) + 'x' + test['height'].astype(str)

# Creating a customized chart and giving in figsize etc.

fig = plt.figure(constrained_layout=True, figsize=(20, 12))

# Creating a grid

grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

# Customizing the first grid.

ax1 = fig.add_subplot(grid[0, :2])

# Set the title.

ax1.set_title('RGB Channels of Benign Images')

# Plot.

sns.distplot(train[train['target'] == 0].reds,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='red',
             kde=True,
             ax=ax1,
             label='Reds')
sns.distplot(train[train['target'] == 0].greens,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='green',
             kde=True,
             ax=ax1,
             label='Greens')
sns.distplot(train[train['target'] == 0].blues,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='blue',
             kde=True,
             ax=ax1,
             label='Blues')

ax1.legend()

# Customizing the second grid.

ax2 = fig.add_subplot(grid[1, :2])

# Set the title.

ax2.set_title('RGB Channels of Malignant Images')

# Plot

sns.distplot(train[train['target'] == 1].reds,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='red',
             kde=True,
             ax=ax2,
             label='Reds')
sns.distplot(train[train['target'] == 1].greens,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='green',
             kde=True,
             ax=ax2,
             label='Greens')
sns.distplot(train[train['target'] == 1].blues,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='blue',
             kde=True,
             ax=ax2,
             label='Blues')
ax2.legend()

# Customizing the third grid.

ax3 = fig.add_subplot(grid[:, 2])

# Set the title.

ax3.set_title('Mean Colors by Train/Test Images')

# Plot

sns.kdeplot(train.mean_colors,
            shade=True,
            label='Train',
            ax=ax3,
            color='#171820',
            vertical=True)
sns.kdeplot(test.mean_colors,
            shade=True,
            label='Test',
            ax=ax3,
            color='#fdc029',
            vertical=True)
ax3.legend()

plt.show()
# Creating a customized chart and giving in figsize etc.

fig = plt.figure(constrained_layout=True, figsize=(20, 12))

# Creating a grid

grid = gridspec.GridSpec(ncols=4, nrows=3, figure=fig)

# Customizing the first grid.

ax1 = fig.add_subplot(grid[0, :2])

# Set the title.

ax1.set_title('Scan Image Resolutions of Train Set')

# Plot.

tres = train.res.value_counts().rename_axis('res').reset_index(name='count')
tres = tres[tres['count'] > 10]
sns.barplot(x='res', y='count', data=tres, palette=orange_black, ax=ax1)
plt.xticks(rotation=20)

ax1.legend()

# Customizing the second grid.

ax2 = fig.add_subplot(grid[0, 2:])

# Set the title.

ax2.set_title('Scan Image Resolutions of Test Set')

# Plot

teres = test.res.value_counts().rename_axis('res').reset_index(name='count')
teres = teres[teres['count'] > 10]
sns.barplot(x='res', y='count', data=teres, palette=orange_black, ax=ax2)
plt.xticks(rotation=20)
ax2.legend()

# Customizing the third grid.

ax3 = fig.add_subplot(grid[1, :])

# Set the title.

ax3.set_title('Scan Image Resolutions by Target')

# Plot.

sns.countplot(x='res',
              hue='benign_malignant',
              data=train,
              order=train.res.value_counts().iloc[:12].index,
              palette=orange_black,
              ax=ax3)
ax3.legend()

# Customizing the last grid.

ax4 = fig.add_subplot(grid[2, :])

# Set the title.

ax4.set_title('Malignant Scan Result Frequency by Image Resolution')

# Plot.

res_freq = train.groupby('res')['target'].mean()
res_freq = res_freq[(res_freq > 0) & (res_freq < 1)]
sns.lineplot(x=res_freq.index, y=res_freq, palette=orange_black, ax=ax4)
ax4.legend()

plt.show()
# Creating a customized chart and giving in figsize etc.

fig = plt.figure(constrained_layout=True, figsize=(20, 14))

# Creating a grid

grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

# Customizing the first grid.

ax1 = fig.add_subplot(grid[0, :2])

# Set the title.

ax1.set_title('RGB Channels of Train Images With "Mysterious" Set')

# Plot.

sns.distplot(train.reds,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='red',
             kde=True,
             ax=ax1,
             label='Reds')
sns.distplot(train.greens,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='green',
             kde=True,
             ax=ax1,
             label='Greens')
sns.distplot(train.blues,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='blue',
             kde=True,
             ax=ax1,
             label='Blues')

ax1.legend()

# Customizing the second grid.

ax2 = fig.add_subplot(grid[1, :2])

# Set the title.

ax2.set_title('RGB Channels of Test Images Without "Mysterious" Set')

# Plot

sns.distplot(test[test['res'] != '1920x1080'].reds,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='red',
             kde=True,
             ax=ax2,
             label='Reds')
sns.distplot(test[test['res'] != '1920x1080'].greens,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='green',
             kde=True,
             ax=ax2,
             label='Greens')
sns.distplot(test[test['res'] != '1920x1080'].blues,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='blue',
             kde=True,
             ax=ax2,
             label='Blues')
ax2.legend()

# Customizing the third grid.

ax3 = fig.add_subplot(grid[:, 2])

# Set the title.

ax3.set_title('Mean Colors by Train/Test Images Without "Mysterious" Set')

# Plot

sns.kdeplot(train.mean_colors,
            shade=True,
            label='Train',
            ax=ax3,
            color='#171820',
            vertical=True)
sns.kdeplot(test[test['res'] != '1920x1080'].mean_colors,
            shade=True,
            label='Test',
            ax=ax3,
            color='#fdc029',
            vertical=True)
ax3.legend()

# Customizing the last grid.

ax2 = fig.add_subplot(grid[2, :2])

# Set the title.

ax2.set_title('RGB Channels of "Mysterious" Set')

# Plot

sns.distplot(test[test['res'] == '1920x1080'].reds,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='red',
             kde=True,
             ax=ax2,
             label='Reds')
sns.distplot(test[test['res'] == '1920x1080'].greens,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='green',
             kde=True,
             ax=ax2,
             label='Greens')
sns.distplot(test[test['res'] == '1920x1080'].blues,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.3
             },
             color='blue',
             kde=True,
             ax=ax2,
             label='Blues')
ax2.legend()

plt.show()
# Creating a customized chart and giving in figsize etc.

# Plotting age dist vs target and age dist vs datasets

fig = plt.figure(constrained_layout=True, figsize=(20, 12))

# Creating a grid

grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

# Customizing the first grid.

ax1 = fig.add_subplot(grid[0, :2])

# Set the title.

ax1.set_title('Scan Image Size Distribution by Train/Test Observations')

# Plot

ax1.legend()

sns.kdeplot(train['image_size'],
            shade=True,
            ax=ax1,
            color='#171820',
            label='Train')
sns.kdeplot(test['image_size'],
            shade=True,
            ax=ax1,
            color='#fdc029',
            label='Test')

# Customizing second grid.

ax2 = fig.add_subplot(grid[0, 2:])

# Set the title.

ax2.set_title('Scan Image Size Distribution Without "Mysterious Set"')

# Plot.

sns.kdeplot(train.image_size,
            label='Train',
            shade=True,
            ax=ax2,
            color='#171820')
sns.kdeplot(test[test['res'] != '1920x1080'].image_size,
            label='Test',
            shade=True,
            ax=ax2,
            color='#fdc029')
ax2.legend()

# Customizing third grid.

ax3 = fig.add_subplot(grid[1, :])

# Set the title.

ax3.set_title('Image Size Distribution of Mysterious Images')

# # Plot

sns.distplot(test[test['res'] == '1920x1080'].image_size,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.9
             },
             color='#FF6347',
             kde=True,
             ax=ax3,
             label='Mysterious Images')
ax3.legend()

plt.show()
# Creating a customized chart and giving in figsize etc.

# Plotting age dist vs target and age dist vs datasets

fig = plt.figure(constrained_layout=True, figsize=(20, 12))

# Creating a grid

grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

# Customizing the first grid.

ax1 = fig.add_subplot(grid[0, :2])

# Set the title.

ax1.set_title('Number of Images Distribution by Train/Test Observations')

# Plot

ax1.legend()

sns.kdeplot(train['n_images'],
            shade=True,
            ax=ax1,
            color='#171820',
            label='Train')
sns.kdeplot(test['n_images'],
            shade=True,
            ax=ax1,
            color='#fdc029',
            label='Test')

# Customizing second grid.

ax2 = fig.add_subplot(grid[0, 2:])

# Set the title.

ax2.set_title('Scan Image Size Distribution Without "Mysterious Set"')

# Plot.

sns.kdeplot(train.n_images,
            label='Train',
            shade=True,
            ax=ax2,
            color='#171820')
sns.kdeplot(test[test['res'] != '1920x1080'].n_images,
            label='Test',
            shade=True,
            ax=ax2,
            color='#fdc029')
ax2.legend()

# Customizing third grid.

ax3 = fig.add_subplot(grid[1, :])

# Set the title.

ax3.set_title('Number of Images Distribution of Mysterious Images')

# Plot

sns.distplot(test[test['res'] == '1920x1080'].n_images,
             hist_kws={
                 "rwidth": 0.75,
                 'edgecolor': 'black',
                 'alpha': 0.9
             },
             color='#FF6347',
             kde=True,
             ax=ax3,
             label='Mysterious Images')
ax3.legend()

plt.show()
fig, ax = plt.subplots(figsize=(20, 6))

sns.kdeplot(test[test['res'] != '1920x1080'].age,
            shade=True,
            label='Without Mystery Set',
            color='#171820',
            )
sns.kdeplot(test[test['res'] == '1920x1080'].age,
            shade=True,
            label='With Mystery Set',
            color='#fdc029',
            )

plt.legend(loc='upper right')

ax.set_title('Age Distribution With/Without Mysterious Set')


plt.show()
mystery = test[test['res'] == '1920x1080']
mystimages = mystery['img_name'].values

nonmystery = test[test['res'] != '1920x1080']
nonmystimages = nonmystery['img_name'].values

random_myst_images = [np.random.choice(mystimages+'.jpg') for i in range(12)]
random_nmyst_images = [np.random.choice(nonmystimages+'.jpg') for i in range(12)]

# Location of test images
img_dir = f'{base_path}/jpeg/test'
plt.figure(figsize=(12,6))
for i in range(12):
    
    plt.subplot(3, 4, i + 1)
    img = plt.imread(os.path.join(img_dir, random_myst_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
plt.suptitle('Sample Images From Mysterious Test Set', fontsize=14)
plt.tight_layout()   
  
plt.figure(figsize=(12,6))
for i in range(12):
    
    plt.subplot(3, 4, i + 1)
    img = plt.imread(os.path.join(img_dir, random_nmyst_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off') 
    
plt.suptitle('Sample Images From Rest of the Test Set', fontsize=14, y=1.05)
plt.tight_layout()   
# Display numerical correlations between features on heatmap.

sns.set(font_scale=1.1)
correlation_train = train[['target','age','age_min',
 'age_max',
 'n_images',
 'image_size',
 'reds',
 'greens',
 'blues', 
 'width',
 'height',
 ]].corr()
mask = np.triu(correlation_train.corr())
plt.figure(figsize=(16, 6))
sns.heatmap(correlation_train,
            annot=True,
            fmt='.1f',
            cmap='coolwarm',            
            mask=mask,
            linewidths=1,
            cbar=False)

plt.show()


# Loading lanscape data

train40 = pd.read_csv(f'{base_path}/landscape/train40Features.csv')
test40 = pd.read_csv(f'{base_path}/landscape/test40Features.csv')

trainmet = pd.read_csv(f'{base_path}/landscape/trainMetrics.csv')
testmet = pd.read_csv(f'{base_path}/landscape/testMetrics.csv')
# dropping duplicate data from lanscape dataset

train40.drop(['sex', 'age_approx', 'anatom_site_general_challenge'],
             axis=1,
             inplace=True)

test40.drop(['sex', 'age_approx', 'anatom_site_general_challenge'],
            axis=1,
            inplace=True)
# merging both datasets

train = pd.concat([train, train40, trainmet], axis=1)
test = pd.concat([test, test40, testmet], axis=1)
# checking out new dataset

train.head()
# getting dummy variables for gender on train set

sex_dummies = pd.get_dummies(train['sex'], prefix='sex')
train = pd.concat([train, sex_dummies], axis=1)

# getting dummy variables for gender on test set

sex_dummies = pd.get_dummies(test['sex'], prefix='sex')
test = pd.concat([test, sex_dummies], axis=1)

# dropping not useful columns

train.drop(['sex','res','img_name','id','diagnosis','benign_malignant'], axis=1, inplace=True)
test.drop(['sex','res','img_name','id'], axis=1, inplace=True)
# getting dummy variables for location on train set

anatom_dummies = pd.get_dummies(train['location'], prefix='anatom')
train = pd.concat([train, anatom_dummies], axis=1)

# getting dummy variables for location on test set

anatom_dummies = pd.get_dummies(test['location'], prefix='anatom')
test = pd.concat([test, anatom_dummies], axis=1)

# dropping not useful columns

train.drop('location', axis=1, inplace=True)
test.drop('location', axis=1, inplace=True)
# loading modelling libraries

import xgboost as xgb

import sklearn

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score
# dividing train set and labels for modelling

X = train.drop('target', axis=1)
y = train.target
# taking holdout set for validating with stratified y

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

# 5 fold stratify for cv

cv = StratifiedKFold(5, shuffle=True, random_state=42)
#from importlib import reload
#reload(sklearn)
#reload(xgb)
# setting model hyperparameters, didn't include fine tuning here because of timing reasons...

xg = xgb.XGBClassifier(
    n_estimators=750,
    min_child_weight=0.81,
    learning_rate=0.025,
    max_depth=2,
    subsample=0.80,
    colsample_bytree=0.42,
    gamma=0.10,
    random_state=42,
    n_jobs=-1,
)
estimators = [xg]
# cross validation scheme

def model_check(X_train, y_train, estimators, cv):
    model_table = pd.DataFrame()

    row_index = 0
    for est in estimators:

        MLA_name = est.__class__.__name__
        model_table.loc[row_index, 'Model Name'] = MLA_name

        cv_results = cross_validate(est,
                                    X_train,
                                    y_train,
                                    cv=cv,
                                    scoring='roc_auc',
                                    return_train_score=True,
                                    n_jobs=-1)

        model_table.loc[row_index,
                        'Train roc Mean'] = cv_results['train_score'].mean()
        model_table.loc[row_index,
                        'Test roc Mean'] = cv_results['test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1

    model_table.sort_values(by=['Test roc Mean'],
                            ascending=False,
                            inplace=True)

    return model_table
# display cv results

raw_models = model_check(X_train, y_train, estimators, cv)
display(raw_models)
# fitting train data

xg.fit(X_train, y_train)

# predicting on holdout set
validation = xg.predict_proba(X_test)[:, 1]

# checking results on validation set
roc_auc_score(y_test, validation)
# finding feature importances and creating new dataframe basen on them

feature_importance = xg.get_booster().get_score(importance_type='weight')

keys = list(feature_importance.keys())
values = list(feature_importance.values())

importance = pd.DataFrame(data=values, index=keys,
                          columns=["score"]).sort_values(by="score",
                                                         ascending=False)
plt.figure(figsize=(16, 10))
sns.barplot(x=importance.score.iloc[:20],
            y=importance.index[:20],
            orient='h',
            palette='Reds_r')

plt.show()
# predicting on test set

#test['total_pixels']= test['width']*test['height']

# test['total_pixels'].dtype

#type(test['total_pixels'].iloc[0])


predictions = xg.predict_proba(test)[:, 1]
# loading sample submission
sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))

meta_df = pd.DataFrame(columns=['image_name', 'target'])

# assigning predictions on submission df

meta_df['image_name'] = sample['image_name']
meta_df['target'] = predictions
# creating submission csv file

meta_df.to_csv('metav.csv', header=True, index=False)

