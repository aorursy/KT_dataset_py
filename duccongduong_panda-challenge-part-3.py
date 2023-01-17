# Imports

import os



import openslide

from IPython.display import Image, display

#     Allows viewing of images



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Setting dataset directories to variables

train_dir = "/kaggle/input/prostate-cancer-grade-assessment/train_images/"

#     train_images

mask_dir = "/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/"

#     train_masks



train_csv = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/train.csv")

#     train_csv
train_csv.drop(index=7273, inplace=True)

#     Dropping 7273



train_csv['gleason_score'] = train_csv['gleason_score'].apply(lambda x: '0+0' if x == 'negative' else x)

#     Converting gleason_score = negative to gleason_score = 0+0
# Confirming

try:

    train_csv.loc[7273]

except:

    print('index 7273 not found')

    

print(train_csv[train_csv['isup_grade']==0]['gleason_score'].value_counts())
train_csv.head()
train_csv.dtypes
train_csv = train_csv.astype({'data_provider': 'category',

                 'isup_grade': 'category',

                 'gleason_score': 'category'})
train_csv.dtypes
sns.countplot(data=train_csv, x='data_provider', palette='viridis')



plt.title("Samples by Provider", fontdict={'fontsize': 12})

plt.ylabel("No. of Samples", labelpad=6.5, fontdict={'fontsize': 12})

plt.xlabel("Institution", labelpad=7, fontdict={'fontsize': 13})

plt.yticks()

sns.despine(left=True)
# There are similar sample amounts coming from karolinska and radboud 

print(train_csv.data_provider.value_counts())

print(train_csv.data_provider.value_counts(normalize=True))
sns.countplot(data=train_csv, x='isup_grade', palette="cividis")



plt.title("Samples by ISUP Grade", fontdict={'fontsize': 12})

plt.ylabel("No. of Samples", labelpad=6.5, fontdict={'fontsize': 12})

plt.xlabel("ISUP Grade", labelpad=7, fontdict={'fontsize': 13})

plt.yticks()

sns.despine(left=True)
# There are similar sample counts for samples ISUP Grade >= 2

# 0 ISUP Grade Samples (Negative for Cancer) is the largest category, 1 is the second highest. Both almost double the other categories

print(train_csv.isup_grade.value_counts())

print(train_csv.isup_grade.value_counts(normalize=True))

print("\nNegative Samples Take Up More Than 27% of the Dataset. Potentially be a source of class imbalance")
plt.figure(figsize=(7,5))

sns.countplot(data=train_csv, x='gleason_score', palette="hot")



plt.title("Samples by Gleason Score", fontdict={'fontsize': 14})

plt.ylabel("No. of Samples", labelpad=6.5, fontdict={'fontsize': 12})

plt.xlabel("Gleason Score", labelpad=7, fontdict={'fontsize': 13})

plt.yticks()

sns.despine(left=True)
# The Same Class Inbalance is observed because Gleason Score is tied to ISUP Grades 

# Although there are very few samples for 3+5 and 5+3 samples

# Lack of 3+5 samples impacts predicting ISUP Grade 4 samples and predicting Gleason-Score 3+5 samples 

# Lack of 5+3 samples impacts predicting ISUP Grade 5 samples and predicting Gleason-Score 5+3 samples 



# With a Gleason-Score predictor, what changes needs to be made?

#     This predictor would predict poorly serious cancers (Grade 5), so there is a potential case of misclassifing serious cancers as not so serioius.

#     In a medical domain, false negative for a serious diagnosis isn't premissable

# With a ISUP Grade predictor, what changes needs to be made?

#     This predictor would have prediction inconsistencies for both Grade 4 and Grade 5 cancers as it will 'miss' a category of samples at

#     that grade.



# The model has to be designed to account for these flaws in the dataset.

#     IF:

#         A ISUP Model predicts : 5

#         A Glea Model predicts : 5+3, 4+4, 3+5 (A 4)

#     What do I conclude?

#         If the G-Model predicts a 3+5: make the final model predict 5. It's a measure to lower false negatives

#         If the G-Model preidcts a 5+3 or 4+4: Check prediction weightings, if high and ISUP Model has a low weighting for 5 probs let it be a 4..
plt.figure(figsize=(8,5))

sns.countplot(data=train_csv, x='isup_grade', hue='gleason_score', palette="icefire")



plt.title("Samples by ISUP Grade grouped by Gleason Score", fontdict={'fontsize': 14})

plt.ylabel("No. of Samples", labelpad=6.5, fontdict={'fontsize': 12})

plt.xlabel("ISUP Grade", labelpad=7, fontdict={'fontsize': 13})

plt.yticks()

plt.legend(loc='upper left', bbox_to_anchor=(1,1))

sns.despine(left=True)
# Oh No, the problem doesn't extend to 3+5 and and 5+3. It extends to 5+4, 5+5

# But the point still stands, I may need the intelligence of both a G-score and I-grade model to make my final predictions.



# Graph proves that data munging cleaned up the dataset (No rows with mismathed ISUP Grades and Gleason Scores)
plt.figure(figsize=(8,5))

sns.countplot(data=train_csv, x='isup_grade', hue='data_provider', palette="icefire")



plt.title("Samples by ISUP Grade grouped by Provider", fontdict={'fontsize': 14})

plt.ylabel("No. of Samples", labelpad=6.5, fontdict={'fontsize': 12})

plt.xlabel("ISUP Grade", labelpad=7, fontdict={'fontsize': 13})

plt.yticks()

plt.legend(loc='upper left', bbox_to_anchor=(1,1))

sns.despine(left=True)
# It seems radboud provides more serious biospies

# While karolinska provides more beigin biospies



# I hope my models don't fit for variations due to the provider (image size, color due to staining, microscopes), because of provider inbalance
plt.figure(figsize=(8,5))

sns.countplot(data=train_csv, x='gleason_score', hue='data_provider', palette="icefire")



plt.title("Samples by Gleason Score grouped by Provider", fontdict={'fontsize': 14})

plt.ylabel("No. of Samples", labelpad=6.5, fontdict={'fontsize': 12})

plt.xlabel("Gleason Score", labelpad=7, fontdict={'fontsize': 13})

plt.yticks()

plt.legend(loc='upper left', bbox_to_anchor=(1,1))

sns.despine(left=True)
# Shows the same, but by Gleason Score.



# I'll need to find what's different between karolinska and radboud images to conclude possible issues with the data.
sns.boxplot(data=train_csv.astype({'isup_grade': 'int8'}), x='data_provider', y='isup_grade', palette='cool')



plt.title("ISUP Grade Distrubtion by Institute", fontdict={'fontsize': 14})

plt.ylabel("ISUP Grade", labelpad=6.5, fontdict={'fontsize': 12})

plt.xlabel("Institute", labelpad=7, fontdict={'fontsize': 13})

plt.yticks()

plt.tick_params(axis='x', length=0)

sns.despine(left=True, bottom=True)
# Boxplot further shows the inbalance
pen_marked_images = [

    'fd6fe1a3985b17d067f2cb4d5bc1e6e1',

    'ebb6a080d72e09f6481721ef9f88c472',

    'ebb6d5ca45942536f78beb451ee43cc4',

    'ea9d52d65500acc9b9d89eb6b82cdcdf',

    'e726a8eac36c3d91c3c4f9edba8ba713',

    'e90abe191f61b6fed6d6781c8305fe4b',

    'fd0bb45eba479a7f7d953f41d574bf9f',

    'ff10f937c3d52eff6ad4dd733f2bc3ac',

    'feee2e895355a921f2b75b54debad328',

    'feac91652a1c5accff08217d19116f1c',

    'fb01a0a69517bb47d7f4699b6217f69d',

    'f00ec753b5618cfb30519db0947fe724',

    'e9a4f528b33479412ee019e155e1a197',

    'f062f6c1128e0e9d51a76747d9018849',

    'f39bf22d9a2f313425ee201932bac91a',

]
# Display One of Them

image_name = pen_marked_images[0]+".tiff"

slide = openslide.OpenSlide(os.path.join(train_dir,image_name))

display(slide.get_thumbnail(size=(400,500)))

slide.close()