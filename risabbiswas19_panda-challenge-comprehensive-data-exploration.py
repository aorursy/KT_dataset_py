import pandas as pd 

import numpy as np

import os

import matplotlib

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from matplotlib.patches import Rectangle

import seaborn as sns

import openslide

import PIL

%matplotlib inline 

PATH = "/kaggle/input/prostate-cancer-grade-assessment/"
sample_submission_df = pd.read_csv(os.path.join(PATH,'sample_submission.csv'))

train_df = pd.read_csv(os.path.join(PATH,'train.csv'))

test_df = pd.read_csv(os.path.join(PATH,'test.csv'))
print(f"sample submission shape: {sample_submission_df.shape}")

print(f"train shape: {train_df.shape}")

print(f"test shape: {test_df.shape}")
sample_submission_df.head()
train_df.head()
test_df.head()
train_image_list = os.listdir(os.path.join(PATH, 'train_images'))

train_label_masks_list = os.listdir(os.path.join(PATH, 'train_label_masks'))
print(f"train image_id list: {train_df.image_id.nunique()}")

print(f"train image list: {len(train_image_list)}")

print(f"train label masks list: {len(train_label_masks_list)}")
print(f"sample of image_id list: {train_df.image_id.values[0:3]}")

print(f"sample of image list: {train_image_list[0:3]}")

print(f"sample of label masks list: {train_label_masks_list[0:3]}")
trimmed_image_list = []

for img in train_image_list:

    trimmed_image_list.append(img.split('.tiff')[0])
trimmed_label_masks_list = []

for img in train_label_masks_list:

    trimmed_label_masks_list.append(img.split('_mask.tiff')[0])
intersect_i_m = (set(trimmed_image_list) & set(trimmed_label_masks_list))

intersect_id_m = (set(train_df.image_id.unique()) & set(trimmed_label_masks_list))

print(f"image (tiff) & label masks: {len(intersect_i_m)}")

print(f"image_id (train) & label masks: {len(intersect_id_m)}")
!pip install bokeh
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show

from bokeh.palettes import Spectral8

output_notebook()
provider = train_df['data_provider'].unique().tolist()

count_provider = train_df['data_provider'].value_counts().tolist()



tools = "hover,pan,wheel_zoom,box_zoom,reset,lasso_select"

source = ColumnDataSource(data=dict(provider=provider, counts=count_provider, color=Spectral8))

p = figure(x_range=provider, y_range=(0,6000), plot_width =900, plot_height = 400, title="Data Provider - Data Count",

           toolbar_location="right", tools = tools)

p.xaxis.axis_label = "Data Provider"

p.yaxis.axis_label = "Value Count"



# Render and show the vbar plot

p.vbar(x='provider', top='counts', width=0.3, color='color', source=source)

show(p)



isup_grade = train_df['isup_grade'].unique().tolist()

count_isup_grade = train_df['isup_grade'].value_counts().tolist()



source2 = ColumnDataSource(data=dict(isup_grade=isup_grade, counts=count_isup_grade, color=Spectral8))

p2 = figure(y_range=(0,6000), plot_width =900, plot_height = 400, title="ISUP grade - Data Count",

           toolbar_location="right", tools = tools)



p2.xaxis.axis_label = "isup_grade"

p2.yaxis.axis_label = "Value Count"

p2.vbar(x='isup_grade', top='counts', width=0.3, color='color', source=source2)

show(p2)
gleason_score = train_df['gleason_score'].unique().tolist()

count_gleason_score= train_df['gleason_score'].value_counts().tolist()



source_3 = ColumnDataSource(data=dict(gleason_score = gleason_score, counts = count_gleason_score, color=Spectral8))

p3 = figure(x_range = gleason_score,y_range=(0,4000), plot_width =900, plot_height = 400, title="Gleason score - Data Count",

           toolbar_location="right", tools = tools)



p3.xaxis.axis_label = "isup_grade"

p3.yaxis.axis_label = "Value Count"

p3.vbar(x='gleason_score', top='counts', width=0.4, color='color', source = source_3)

show(p3)
sns.set(style="whitegrid")

fig, ax = plt.subplots(nrows=1,figsize=(15,6))

temp = train_df.groupby('isup_grade')['gleason_score'].value_counts()

df = pd.DataFrame(data={'Exams': temp.values}, index=temp.index).reset_index()

sns.barplot(ax=ax,x = 'isup_grade', y='Exams',hue='gleason_score',data=df,palette = 'magma',dodge = True)

plt.title("Number of Examinations Grouped on ISUP Grade and Gleason Score")

plt.show()
sns.set(style="whitegrid")

fig, ax = plt.subplots(nrows=1,figsize=(15,6))

heatmap_data = pd.pivot_table(df, values='Exams', index=['isup_grade'], columns='gleason_score')

sns.heatmap(heatmap_data,linewidth=0.5,linecolor='skyblue')

plt.title('Number of examinations grouped on ISUP grade and Gleason score')

plt.show()
sns.set(style="whitegrid")

fig, ax = plt.subplots(nrows=1,figsize=(18,10)) 

tmp = train_df.groupby('data_provider')['gleason_score'].value_counts() 

df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 

sns.barplot(ax=ax,x = 'data_provider', y='Exams',hue='gleason_score',data=df, palette='Set2',errcolor='black',saturation = 1) 

plt.title("Number of examinations grouped on Data provider and Gleason score") 

plt.show()
sns.set(style="whitegrid")

fig, ax = plt.subplots(nrows=1,figsize=(18,10)) 

tmp = train_df.groupby('data_provider')['isup_grade'].value_counts() 

df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 

sns.barplot(ax=ax,x = 'data_provider', y='Exams',hue='isup_grade',data=df, palette='Set3',errcolor='black',saturation = 5)

plt.title("Number of examinations grouped on Data provider and Gleason score") 

plt.show()
def show_images(df, read_region=(1780,1950)):

    

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join(PATH,"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        ax[i//3, i%3].imshow(patch) 

        image.close()       

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title('ID: {}\nSource: {} ISUP: {} Gleason: {}'.format(

                data_row[1][0], data_row[1][1], data_row[1][2], data_row[1][3]))



    plt.show()
images = [

    '07a7ef0ba3bb0d6564a73f4f3e1c2293',

    '037504061b9fba71ef6e24c48c6df44d',

    '035b1edd3d1aeeffc77ce5d248a01a53',

    '059cbf902c5e42972587c8d17d49efed',

    '06a0cbd8fd6320ef1aa6f19342af2e68',

    '06eda4a6faca84e84a781fee2d5f47e1',

    '0a4b7a7499ed55c71033cefb0765e93d',

    '0838c82917cd9af681df249264d2769c',

    '046b35ae95374bfb48cdca8d7c83233f'

]

data_sample = train_df.loc[train_df.image_id.isin(images)]

show_images(data_sample)
data_sample
image_path = os.path.join(PATH,"train_images")

image = openslide.OpenSlide(os.path.join(image_path, '00a76bfbec239fd9f465d6581806ff42.tiff'))

patch = image.read_region((1780,1950), 0, (256, 256))



display(patch)



image.close()
def print_slide_details(slide, show_thumbnail=True, max_size=(600,400)):

  

    if show_thumbnail:

        display(slide.get_thumbnail(size=max_size))

        

    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)

    

    print(f"File id: {slide}")

    print(f"Dimensions: {slide.dimensions}")

    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")

    print(f"Number of levels in the image: {slide.level_count}")

    print(f"Downsample factor per level: {slide.level_downsamples}")

    print(f"Dimensions of levels: {slide.level_dimensions}")
train_labels = train_df.set_index('image_id')

example_slides = [

    '028098c36eb49a8c6aa6e76e365dd055',

    '0280f8b612771801229e2dde52371141',

    '028dc05d52d1dd336952a437f2852a0a',

    '02a2dcd6ad8bc1d9ad7fdc04ffb6dff3',

    '049031b0ea0dede1ca1e5ca470c1332d',

    '05f4e9415af9fdabc19109c980daf5ad',

    '07fd8d4f02f9b95d86da4bc89563e077'

    

]



for case_id in example_slides:

    biopsy = openslide.OpenSlide(os.path.join(image_path, f'{case_id}.tiff'))

    print_slide_details(biopsy)

    biopsy.close()

    



    print(f"ISUP grade: {train_labels.loc[case_id, 'isup_grade']}")

    print(f"Gleason score: {train_labels.loc[case_id, 'gleason_score']}\n\n")
def show_masks(slides): 

    f, ax = plt.subplots(5,3, figsize=(18,22))

    for i, slide in enumerate(slides):

        

        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{slide}_mask.tiff'))

        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])



        ax[i//3, i%3].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) 

        mask.close()       

        ax[i//3, i%3].axis('off')

        

        image_id = slide

        data_provider = data_sample_mask.loc[slide, 'data_provider']

        isup_grade = data_sample_mask.loc[slide, 'isup_grade']

        gleason_score = data_sample_mask.loc[slide, 'gleason_score']

        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")

        f.tight_layout()

        

    plt.show()
images_mask  = [

    '07a7ef0ba3bb0d6564a73f4f3e1c2293',

    '037504061b9fba71ef6e24c48c6df44d',

    '035b1edd3d1aeeffc77ce5d248a01a53',

    '059cbf902c5e42972587c8d17d49efed',

    '06a0cbd8fd6320ef1aa6f19342af2e68',

    '06eda4a6faca84e84a781fee2d5f47e1',

    '0a4b7a7499ed55c71033cefb0765e93d',

    '0838c82917cd9af681df249264d2769c',

    '028098c36eb49a8c6aa6e76e365dd055',

    '0280f8b612771801229e2dde52371141',

    '028dc05d52d1dd336952a437f2852a0a',

    '02a2dcd6ad8bc1d9ad7fdc04ffb6dff3',

    '049031b0ea0dede1ca1e5ca470c1332d',

    '05f4e9415af9fdabc19109c980daf5ad',

    '07fd8d4f02f9b95d86da4bc89563e077'

]



mask_dir = os.path.join(PATH,"train_label_masks")

data_sample_mask = train_df.set_index('image_id')

show_masks(images_mask)
def overlay_mask_on_slide(images, center='radboud', alpha=0.8, max_size=(800, 800)):

    

    f, ax = plt.subplots(5,3, figsize=(18,22))

    

    

    for i, image_id in enumerate(images):

        

        slide = openslide.OpenSlide(os.path.join(data_dir, f'{image_id}.tiff'))

        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))

        slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])

        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

        mask_data = mask_data.split()[0]

        

        

        # Create alpha mask

        alpha_int = int(round(255*alpha))

        if center == 'radboud':

            alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)

        elif center == 'karolinska':

            alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)



        alpha_content = PIL.Image.fromarray(alpha_content)

        preview_palette = np.zeros(shape=768, dtype=int)



        if center == 'radboud':

            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}

            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

        elif center == 'karolinska':

            # Mapping: {0: background, 1: benign, 2: cancer}

            preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)



        mask_data.putpalette(data=preview_palette.tolist())

        mask_rgb = mask_data.convert(mode='RGB')

        overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)

        overlayed_image.thumbnail(size=max_size, resample=0)



        

        ax[i//3, i%3].imshow(overlayed_image) 

        slide.close()

        mask.close()       

        ax[i//3, i%3].axis('off')

        

        data_provider = train.loc[image_id, 'data_provider']

        isup_grade = train.loc[image_id, 'isup_grade']

        gleason_score = train.loc[image_id, 'gleason_score']

        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
BASE_PATH = '../input/prostate-cancer-grade-assessment'

data_dir = f'{BASE_PATH}/train_images'

train = train_df.set_index('image_id')
images_com = [

    '07a7ef0ba3bb0d6564a73f4f3e1c2293',

    '037504061b9fba71ef6e24c48c6df44d',

    '035b1edd3d1aeeffc77ce5d248a01a53',

    '059cbf902c5e42972587c8d17d49efed',

    '06a0cbd8fd6320ef1aa6f19342af2e68',

    '06eda4a6faca84e84a781fee2d5f47e1',

    '0a4b7a7499ed55c71033cefb0765e93d',

    '0838c82917cd9af681df249264d2769c',

    '028098c36eb49a8c6aa6e76e365dd055',

    '0280f8b612771801229e2dde52371141',

    '028dc05d52d1dd336952a437f2852a0a',

    '02a2dcd6ad8bc1d9ad7fdc04ffb6dff3',

    '049031b0ea0dede1ca1e5ca470c1332d',

    '05f4e9415af9fdabc19109c980daf5ad',

    '07fd8d4f02f9b95d86da4bc89563e077'

]



overlay_mask_on_slide(images_com)
files = os.listdir(PATH+"train_images/")

print(f"there are {len(files)} tiff files in train_images folder")

for i in train_df.image_id:

    assert i+".tiff" in files

print("all training image_ids have their files in train_images folder")
slide = openslide.OpenSlide(PATH+"train_images/"+files[105])

spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)

print(f"File id: {slide}")

print(f"Dimensions: {slide.dimensions}")

print(f"Microns per pixel / pixel spacing: {spacing:.3f}")

print(f"Number of levels in the image: {slide.level_count}")

print(f"Downsample factor per level: {slide.level_downsamples}")

print(f"Dimensions of levels: {slide.level_dimensions}")

patch = slide.read_region((1780,1950), 0, (256, 256))

display(patch) 

slide.close()
import time

start_time = time.time()

slide_dimensions, spacings, level_counts = [], [], []

down_levels, level_dims = [], []



for image_id in train_df.image_id:

    image = str(image_id)+'.tiff'

    image_path = os.path.join(PATH,"train_images",image)

    slide = openslide.OpenSlide(image_path)

    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)

    slide_dimensions.append(slide.dimensions)

    spacings.append(spacing)

    level_counts.append(slide.level_count)

    down_levels.append(slide.level_downsamples)

    level_dims.append(slide.level_dimensions)

    slide.close()

    del slide





end_time = time.time()

print(f"Total processing time: {round(end_time - start_time,2)} sec.")
train_df['width']  = [i[0] for i in slide_dimensions]

train_df['height'] = [i[1] for i in slide_dimensions]

train_df['spacing'] = spacings

train_df['level_count'] = level_counts
train_df
fig = plt.figure(figsize=(18,10))

ax = sns.scatterplot(x='width', y='height', data=train_df, alpha=0.3)

plt.title("height(y) width(x) scatter plot")

plt.show()
fig = plt.figure(figsize=(18,10))

ax = sns.scatterplot(x='width', y='height', hue='isup_grade', data=train_df, alpha=0.6)

plt.title("height(y) width(x) scatter plot with target")

plt.show()
fig, ax = plt.subplots(nrows=1,figsize=(18,10)) 

sns.distplot(train_df['width'], kde=True, label='width')

sns.distplot(train_df['height'], kde=True, label='height')

plt.xlabel('dimension')

plt.title('Images Width and Height distribution')

plt.legend()

plt.show()
def plot_distribution_grouped(feature, feature_group, hist_flag=True):

    fig, ax = plt.subplots(nrows=1,figsize=(18,10)) 

    for f in train_df[feature_group].unique():

        df = train_df.loc[train_df[feature_group] == f]

        sns.distplot(df[feature], hist=hist_flag, label=f)

    plt.title(f'Images {feature} distribution, grouped by {feature_group}')

    plt.legend()

    plt.show()
plot_distribution_grouped('width', 'data_provider')
plot_distribution_grouped('height', 'data_provider')
plot_distribution_grouped('width', 'isup_grade', False)
plot_distribution_grouped('height', 'isup_grade', False)
plot_distribution_grouped('width', 'gleason_score', False)
plot_distribution_grouped('height', 'gleason_score', False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,10))

sns.distplot(ax=ax1, a=train_df['width'])

ax1.set_title("width distribution")

sns.distplot(ax=ax2, a=train_df['height'])

ax2.set_title("height distribution")

plt.show()
shapes = [j for i in level_dims for j in i]

level  = np.array([j for i in level_dims for j in range(len(i))])

widths  = np.array([i[0] for i in shapes])

heights = np.array([i[1] for i in shapes])

fig, axes = plt.subplots(1, 3 ,figsize=(18,10))

for i in range(3):

    ax = sns.scatterplot(ax=axes[i], x=widths[level==i], y=heights[level==i], alpha=0.9)

    axes[i].set_title(f"Level {i}")

plt.tight_layout()

plt.show()



fig = plt.figure(figsize=(18,10))

sns.scatterplot(x=widths, y=heights,hue=level, alpha=0.9)

plt.show()