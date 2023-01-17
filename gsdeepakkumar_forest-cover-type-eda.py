import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
kaggle=1



if kaggle==0:

    train=pd.read_csv("train.csv")

    test=pd.read_csv("test.csv")

    sample_submission=pd.read_csv("sample_submission.csv")

else:

    train=pd.read_csv("../input/learn-together/train.csv")

    test=pd.read_csv("../input/learn-together/test.csv")

    sample_submission=pd.read_csv("../input/learn-together/sample_submission.csv")

print(f'Shape of train {train.shape} and Shape of test {test.shape}')
train.dtypes
test.dtypes
#Convert the datatypes to categorical,

cat_columns=train.columns[11:55]

for col in cat_columns:

    print(f'Converting {col} as categorical')

    train[col]=train[col].astype('category')

    test[col]=test[col].astype('category')
#Check if the data is balanced or unbalanced,

train['Cover_Type'].value_counts()
#Check if the dataset has any missing values

missing_train=train.isnull().sum()

missing_test=test.isnull().sum()
missing_train[missing_train>0].index
missing_test[missing_test>0].index
train['Elevation'].describe()
test['Elevation'].describe()
train.groupby('Cover_Type')['Elevation'].median()
plt.figure(figsize=(8,8))

sns.boxplot(train['Cover_Type'],train['Elevation'])

plt.title("Boxplot of Elevation with CoverType in train dataset")

plt.xlabel("Cover Type")
plt.figure(figsize=(12,8))

sns.distplot(train['Elevation'].values, bins=50, kde=False, color="red")

plt.title("Histogram of Elevation")

plt.xlabel('Elevation', fontsize=12)

plt.show()
(train['Elevation']>3750).sum()
plt.figure(figsize=(12,8))

sns.distplot(test['Elevation'].values, bins=50, kde=False, color="red")

plt.title("Histogram of Elevation")

plt.xlabel('Elevation', fontsize=12)

plt.show()
def plot_numerical(variable):

    plt.figure(figsize=(16,6))

    plt.subplot(121)

    sns.distplot(train[variable].values, bins=50, kde=False, color="red")

    plt.title(f'Histogram of {variable} in train')

    plt.xlabel(f'{variable}', fontsize=12)

    plt.subplot(122)

    sns.distplot(test[variable].values, bins=50, kde=False, color="red")

    plt.title(f'Histogram of {variable} in test')

    plt.xlabel(f'{variable}', fontsize=12)

    plt.show()

    

def plot_boxplot(variable):

    #print(f'Plotting boxplot for {variable}\n')

    plt.figure(figsize=(8,8))

    sns.boxplot(train['Cover_Type'],train[variable])

    plt.title(f'Boxplot of {variable} with CoverType in train dataset')

    plt.xlabel("Cover Type")

    
plot_numerical('Aspect')
plot_boxplot('Aspect')
plot_numerical('Slope')
train['Slope'].describe()
test['Slope'].describe()
plot_boxplot('Slope')
distance_cols=['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']
for col in distance_cols:

    print(f'Plotting Histogram for {col}\n')

    plot_numerical(col)

   
for col in distance_cols:

    plot_boxplot(col)

   
hillshade_cols=['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']
for col in hillshade_cols:

    print(f'Plotting for {col}\n')

    plot_numerical(col)
for col in hillshade_cols:

    plot_boxplot(col)
#Check the unique values for each categorical variables :

for col in cat_columns:

    print(f'{col} has {train[col].nunique()} unique values in train\n')

    print(f'{col} has {test[col].nunique()} unique values in test\n')