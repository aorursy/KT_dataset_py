# Importing useful libraries. 



import numpy as np 

import pandas as pd 

import os

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

#set directory

MAIN_DIR = '../input/prostate-cancer-grade-assessment'

# load data

train = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv')).set_index('image_id')

test = pd.read_csv(os.path.join(MAIN_DIR, 'test.csv')).set_index('image_id')

display(train.head())

print("Shape of training data :", train.shape)

print("unique data provider :", len(train.data_provider.unique()))

print("unique isup_grades (Target Variable) :", len(train.isup_grade.unique()))

print("unique gleason_scores :", len(train.gleason_score.unique()))
display(test.head())

print('Shape of test data: ', test.shape)

print('unique data provider: ', len(test.data_provider.unique()))
# We now define a function that will be useful to us several times. Thanks to Rohit Singh's notebook.

def plot_count(df, feature, title='', size=2):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    total = float(len(df))

    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='deep')

    plt.title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 9,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count(df=train, feature='data_provider', title = 'Data provider - count and percentage share')
plot_count(df=train, feature='isup_grade', title = 'ISUP grade - count and percentage share')
plot_count(df=train, feature='gleason_score', title = 'Gleason Score - count and percentage share', size=3)
print(len(train[train['gleason_score'] == '4+5'])/(len(train[train['gleason_score'] == '4+5'])+len(train[train['gleason_score'] == '5+4'])+len(train[train['gleason_score'] == '5+5'])+len(train[train['gleason_score'] == '3+5'])+len(train[train['gleason_score'] == '5+3'])))
print(len(train[(train.data_provider == 'radboud') & (train.gleason_score == '0+0')]))

print(len(train[(train.data_provider == 'radboud') & (train.gleason_score == 'negative')]))

print(len(train[(train.data_provider == 'karolinska') & (train.gleason_score == '0+0')]))

print(len(train[(train.data_provider == 'karolinska') & (train.gleason_score == 'negative')]))
def plot_relative_distribution(df, feature, hue, title='', size=2):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    total = float(len(df))

    sns.countplot(x=feature, hue=hue, data=df, palette='deep')

    plt.title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_relative_distribution(df=train, feature='gleason_score', hue='data_provider', title = 'relative distribution of Gleason score by Data provider', size=3.5)
plot_relative_distribution(df=train, feature='isup_grade', hue='data_provider', title = 'relative distribution of ISUP grade by data_provider', size=2)
plot_relative_distribution(df=train, feature='gleason_score', hue='isup_grade', title = 'relative distribution of Gleason score by ISUP grade', size=3)
print(train[(train.gleason_score == '0+0') & (train.isup_grade != 0)])

print(train[(train.gleason_score == '4+4') & (train.isup_grade != 4)])

print(train[(train.gleason_score == '3+3') & (train.isup_grade != 1)])

print(train[(train.gleason_score == '4+3') & (train.isup_grade != 3)])

print(train[(train.gleason_score == 'negative') & (train.isup_grade != 0)])

print(train[(train.gleason_score == '4+5') & (train.isup_grade != 5)])

print(train[(train.gleason_score == '3+4') & (train.isup_grade != 2)])

print(train[(train.gleason_score == '5+4') & (train.isup_grade != 5)])

print(train[(train.gleason_score == '5+5') & (train.isup_grade != 5)])

print(train[(train.gleason_score == '5+3') & (train.isup_grade != 4)])

print(train[(train.gleason_score == '3+5') & (train.isup_grade != 4)])