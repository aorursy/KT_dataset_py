import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#for dirname, _, filenames in os.walk('../input/siim-isic-melanoma-classification/jpeg/'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
BASE_DIR = '../input/siim-isic-melanoma-classification/'
train = pd.read_csv(BASE_DIR + 'train.csv')
test  = pd.read_csv(BASE_DIR + 'test.csv')
im_train_path = BASE_DIR + 'jpeg/train/'
im_test_path = BASE_DIR + 'jpeg/test/'

def dataPlot(train, train_malignant):
    train_malignant = train[train['benign_malignant']=='malignant']
    targets = ['sex', 'age_approx', 'benign_malignant','anatom_site_general_challenge','diagnosis']
    fig, ax = plt.subplots(3, 2, figsize=(20, 10))
    sns.countplot(x=targets[0], data=train, ax=ax[0, 0])
    sns.countplot(x=targets[1], data=train, ax=ax[0, 1])
    sns.countplot(x=targets[2], data=train, ax=ax[1, 0])
    sns.countplot(x=targets[3], data=train, ax=ax[1, 1])
    sns.countplot(x=targets[4], data=train_malignant, ax=ax[2, 0])
    plt.show()
    return

#Data review
print('Train shape and sample')
train.shape
print(train.sample(n=5, random_state=1))
print('Test shape and sample')
test.shape
print(train.sample(n=3, random_state=1))
print('Sex:\n',train['sex'].value_counts())
print('Age:\n',train['age_approx'].value_counts())
print('Localization:\n',train['anatom_site_general_challenge'].value_counts())
print(train['benign_malignant'].value_counts())
train_malignant = train[train['benign_malignant']=='malignant']
dataPlot(train, train_malignant)
def missing_values_table(df):
        print('Size of table is ',df.size)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns
    
print('Train data missing (null or NaN values)')
print(missing_values_table(train))
print('\nTest data missing (null or NaN values)')
print(missing_values_table(test))
train_fill = train.fillna('unknown')
test_fill = test.fillna('unknown')
#Remove rows  with unknown values based in previous work
train_fill = train_fill[train_fill.anatom_site_general_challenge != 'unknown']
train_fill = train_fill[train_fill.age_approx != 'unknown']
train_fill = train_fill[train_fill.sex != 'unknown']
test_fill = test_fill[test_fill.anatom_site_general_challenge != 'unknown']
from PIL import Image
def drawImages(sample):
    plt.figure(figsize = (18,18))
    for iterator, filename in enumerate(sample):
        image = Image.open(filename)
        plt.subplot(4,2,iterator+1)
        plt.imshow(image)   
    plt.tight_layout()
    return

print("Sample benign images")
sample_train = im_train_path + train_fill[train_fill['benign_malignant']=='benign'].sample(4).image_name + '.jpg'
drawImages(sample_train)
print("Sample malignant images")
sample_train = im_train_path + train_fill[train_fill['benign_malignant']=='malignant'].sample(4).image_name + '.jpg'
drawImages(sample_train)