#Import libraries
import os
from os import listdir
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#Read the files available 
print(os.listdir("../input/siim-isic-melanoma-classification"))
#map to image data path
image_path = "../input/siim-isic-melanoma-classification/"

#train data set path
df_train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

#test data set path
df_test =  pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
#print train data set shape
print('train data', df_train.shape)

#print test dataset shape
print('test data', df_test.shape)

# test set percentage 
df_train.shape[0] / df_test.shape[0]

#Train data set info
df_train.info()
#Test data set info
df_test.info()
# count of the total missing values of each variables in train data
df_train.isnull().sum()
# count of the total missing values of each variables in test data
df_test.isnull().sum()
#1st 5 observations in train data set 
df_train.head()
#1st 5 observations in test data set 
df_test.head()
# Variable benign_malignant counts in train set
df_train['benign_malignant'].value_counts()
benign_malignant = df_train['benign_malignant'].value_counts()
labels = 'benign', 'malignant'
plt.pie (benign_malignant, labels=labels, autopct='%1.1f%%' )
plt.title('benign vs malignant')
plt.axis('equal')
plt.show()
df_train.groupby("benign_malignant").target.value_counts()
# variable anatom_site_general_challenge in train data set
df_train['anatom_site_general_challenge'].value_counts()
# variable anatom_site_general_challenge in test data set
df_test['anatom_site_general_challenge'].value_counts()
fig, ax = plt.subplots(1,2, figsize=(16,8))

sns.countplot(df_train['anatom_site_general_challenge'].sort_values(ascending=False), ax=ax[0], palette="Greens_r")
ax[0].set_xlabel("")
labels = ax[0].get_xticklabels();
ax[0].set_xticklabels(labels, rotation=90);
ax[0].set_title("Location of imaged site in Train set");

sns.countplot(df_test['anatom_site_general_challenge'].sort_values(ascending=False), ax=ax[1], palette="Blues_r");
ax[1].set_xlabel("")
labels = ax[1].get_xticklabels();
ax[1].set_xticklabels(labels, rotation=90);
ax[1].set_title("Location of imaged site in Test set");
# count of male and female of variable sex in train data
df_train['sex'].value_counts()
# count of male and female of variable sex in test data
df_test['sex'].value_counts()
fig, ax = plt.subplots(1,2,figsize=(10,5))
sns.countplot(df_train['sex'], palette="Greens_r", ax=ax[0]);
ax[0].set_xlabel("")
ax[0].set_title("Train data set gender counts");

sns.countplot(df_test['sex'], palette="Blues_r", ax=ax[1]);
ax[1].set_xlabel("")
ax[1].set_title("Test data set gender counts");
#variable age_approx in train set
df_train['age_approx'].value_counts()
#variable age_approx in test set
df_test['age_approx'].value_counts()
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.countplot(df_train['age_approx'], color="lightgreen", ax=ax[0]);
labels = ax[0].get_xticklabels();
ax[0].set_xticklabels(labels, rotation=90);
ax[0].set_xlabel("");
ax[0].set_title("Age distribution in train set");

sns.countplot(df_test['age_approx'], color="blue", ax=ax[1]);
labels = ax[1].get_xticklabels();
ax[1].set_xticklabels(labels, rotation=90);
ax[1].set_xlabel("");
ax[1].set_title("Age distribution in test set");
# Male and female frequency per age group in train set
df_train.groupby("age_approx").sex.value_counts()
# Male and female frequency per age group in test set
df_test.groupby("age_approx").sex.value_counts()
fig, ax = plt.subplots(1,2,figsize=(24,7))
sns.countplot(df_train['age_approx'], hue=df_train['sex'], ax=ax[0], palette="Greens_r");
sns.countplot(df_test['age_approx'], hue=df_test['sex'], ax=ax[1], palette="Blues_r");
# diagnosis type in train set
df_train['diagnosis'].value_counts()
# Total percentage of unknown diagnosis
27124/33126*100
