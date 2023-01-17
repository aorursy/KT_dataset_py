import numpy as np #linear algebra

import pandas as pd  #data processing e.g. reading files (pd.read_csv)

import matplotlib.pyplot as plt #plotting and visualization

import seaborn as sns #visualization

%matplotlib inline
sns.set_style('darkgrid')

plt.figure(figsize = (15,15))
train=pd.read_csv("../input/train_u6lujuX_CVtuZ9i.csv")

test=pd.read_csv("../input/test_Y3wMUE5_7gLdaTN.csv")
train.head()
train.shape
#Check for missing values in the dataset

train.isnull().sum()
train.describe()
train.info()
sns.set_palette('GnBu_d')

sns.set_style('darkgrid')
#Explore Gender

fig,ax = plt.subplots(figsize=(10, 10))

sns.countplot(x = 'Gender', data = train, palette = 'inferno')
#Explore Marital ststus of applicants

fig,ax = plt.subplots(figsize=(10, 10))

sns.countplot(x = 'Married', data = train, palette = 'inferno')
#Explore Educational Level of applicants

fig,ax = plt.subplots(figsize=(10, 10))

sns.countplot(x = 'Education', data = train, palette = 'inferno')
#Explore Employment Status

fig,ax = plt.subplots(figsize=(10, 10))

sns.countplot(x = 'Self_Employed', data = train, palette = 'inferno')
#Explore Employment Status

fig,ax = plt.subplots(figsize=(10, 10))

sns.countplot(x = 'Property_Area', data = train, palette = 'inferno')
#Explore Loan Applicants Income

fig,ax = plt.subplots(figsize=(10, 10))

sns.distplot(train['ApplicantIncome'], kde = True, bins = 100)
#Explore Co-Applicants Income

fig,ax = plt.subplots(figsize=(10, 10))

sns.distplot(train['CoapplicantIncome'], kde = True, bins = 100)
#Explore Loan Status

fig,ax = plt.subplots(figsize=(10, 10))

sns.countplot(x = 'Loan_Status', data = train, palette = 'inferno')