import pandas as pd

import numpy as np

from pathlib import Path

from skimage.io import MultiImage

import openslide

import matplotlib.pyplot as plt



input_path = Path("../input/prostate-cancer-grade-assessment")

train_images_path = input_path.joinpath("train_images")

train_label_masks_path = input_path.joinpath("train_label_masks")
train_dataset = pd.read_csv(input_path.joinpath("train.csv"), index_col="image_id")



# Use the same mark for 'negative'/'0+0' Gleason score

train_dataset.loc[train_dataset['gleason_score'] == 'negative', 'gleason_score'] = '0+0'



train_dataset.info()
train_images_names = [path.stem for path in train_images_path.glob("*.tiff")]

train_images = pd.Series(train_images_names, index=train_images_names, name="image_id")

merge = train_dataset.merge(train_images, how='outer', left_index=True, right_index=True, indicator=True)



train_dataset_only = merge[merge['_merge'] == 'left_only']

print(f"There are {len(train_dataset_only)} records in train.csv which don't match files in train_images...")

print(train_dataset_only.head())

print()

train_images_only = merge[merge['_merge'] == 'right_only']

print(f"There are {len(train_images_only)} files in train_images missing in train.csv...")

print(train_images_only.head())
def image_shape(train_dataset_row: pd.Series) -> tuple:

    try:

        with openslide.OpenSlide(train_images_path.joinpath(f"{train_dataset_row.name}.tiff").as_posix()) as open_slide:

            return open_slide.level_dimensions

    except openslide.OpenSlideError:

        return np.nan

    

train_dataset['level_dimensions'] = train_dataset.apply(image_shape, axis=1)
train_dataset[train_dataset['level_dimensions'].isna()]
train_dataset['level_dimensions'].apply(len).unique()
print(f"Total number of unique level dimensions of train images is {len(train_dataset['level_dimensions'].unique())}")
radboud_level_dimensions = train_dataset[train_dataset['data_provider']=='radboud']['level_dimensions'].unique()

karolinska_level_dimensions = train_dataset[train_dataset['data_provider']=='karolinska']['level_dimensions'].unique()

print(f"Number of unique level dimensions of radboud train images is {len(radboud_level_dimensions)}")

print(f"radboud level dimensions varies from {radboud_level_dimensions.min()} to {radboud_level_dimensions.max()}")

print(f"Number of unique level dimensions of karolinska train images is {len(karolinska_level_dimensions)}")

print(f"karolinska level dimensions varies from {karolinska_level_dimensions.min()} to {karolinska_level_dimensions.max()}")
distrib_by_data_provider = train_dataset.groupby(['data_provider', 'isup_grade', 'gleason_score']).size()

data_provider_count = distrib_by_data_provider.sum(level='data_provider')

data_provider_isup_grade_count = distrib_by_data_provider.sum(level=['data_provider', 'isup_grade'])

isup_grade_count = distrib_by_data_provider.sum(level=['isup_grade'])



# ISUP grade visualization

fig_isup, ax_isup = plt.subplots(1, 2, figsize=(16,8))



# On the left axes we visualize 

# ISUP data distribution by data provider through donut graph 

size = 0.3

outer_colors = ['tab:blue', 'tab:orange']

cmap_blues = plt.get_cmap("Blues")

cmap_oranges = plt.get_cmap("Oranges")

inner_colors = np.concatenate([cmap_blues(np.linspace(0, 1, 6)), cmap_oranges(np.linspace(0, 1, 6))])



# outer ring

ax_isup[0].pie(

    data_provider_count,

    radius=1,

    autopct='%.2f%%',

    pctdistance=0.85,

    textprops={'size': 'larger'},

    wedgeprops=dict(width=size, edgecolor='w'),

    colors=outer_colors

)



# inner ring

ax_isup[0].pie(

    data_provider_isup_grade_count,

    radius=1-size,

    labels=data_provider_isup_grade_count.index.get_level_values('isup_grade'),

    labeldistance=0.5,

    autopct='%.2f%%',

    pctdistance=0.8,

    textprops={'size': 'smaller'},

    wedgeprops=dict(width=size, edgecolor='w'),

    colors=inner_colors

)



ax_isup[0].text(

    0.5, 0.5,

    'ISUP grade',

    horizontalalignment='center',

    verticalalignment='center',

    transform = ax_isup[0].transAxes

)



ax_isup[0].axis('equal')

ax_isup[0].set_title("ISUP grade distribution by data provider")

ax_isup[0].legend(data_provider_count.index)



# On the right axes we visualize 

# total ISUP data distribution through pie graph 

cmap_red = plt.get_cmap('Reds')

colors = cmap_red(np.linspace(0, 1, 6))

ax_isup[1].pie(

    isup_grade_count,

    labels = isup_grade_count.index,

    autopct='%.2f%%',

    textprops={'size': 'larger'},

    wedgeprops=dict(edgecolor='w'),

    colors=colors

)



ax_isup[1].axis('equal')

ax_isup[1].set_title("Total ISUP grade distribution")



# Gleason score visualization

fig_gleason, ax_gleason = plt.subplots(figsize=(16,8))

data_provider_gleason_score_count = distrib_by_data_provider.sum(level=['data_provider', 'gleason_score'])



radboud = ax_gleason.bar(data_provider_gleason_score_count['radboud'].index.get_level_values('gleason_score'), data_provider_gleason_score_count['radboud'], color='tab:orange')

karolinska = ax_gleason.bar(data_provider_gleason_score_count['karolinska'].index.get_level_values('gleason_score'), data_provider_gleason_score_count['karolinska'], bottom=data_provider_gleason_score_count['radboud'], color='tab:blue')



# Attach a text label above each bar, displaying its share in percentage

total = data_provider_gleason_score_count.sum()

for rect_bottom, rect_upper in zip(radboud, karolinska):

    height = rect_bottom.get_height() + rect_upper.get_height()

    ax_gleason.annotate(

        '%.2f%%' % (height * 100 / total),

        xy=(rect_bottom.get_x() + rect_bottom.get_width() / 2, height),

        xytext=(0, 3),  # 3 points vertical offset

        textcoords="offset points",

        ha='center', va='bottom'

    )



ax_gleason.set_title("Gleason score distribution by data provider")

ax_gleason.legend((karolinska, radboud), ('karolinska', 'radboud'))

ax_gleason.grid(axis='y')



plt.show()