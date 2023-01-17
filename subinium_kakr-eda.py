# 데이터 분석 라이브러리

import numpy as np

import pandas as pd



# 시각화 라이브러리

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

sns.set_style("whitegrid")
train_data = pd.read_csv('/kaggle/input/kakr-4th-competition/train.csv')

test_data = pd.read_csv('/kaggle/input/kakr-4th-competition/test.csv')
train_data.head()
train_data.info()
train_data.describe()
train_data.describe(include='O')
for col in train_data.columns:

    if train_data[col].dtype == 'object':

        categories = train_data[col].unique()

        print(f'[{col}] ({len(categories)})')

        print('\n'.join(categories))

        print()
for col in train_data.columns:

    if train_data[col].dtype == 'object':

        categories = train_data[col].unique()

        print(f'[{col}] ({len(categories)})')
train_data['sex'].value_counts()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.countplot(x='sex', data=train_data)

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)



sns.countplot(x='sex', data=train_data, ax=axes[0], palette="Set2", edgecolor='black') 

sns.countplot(x='income', data=train_data, ax=axes[1], color='gray', edgecolor='black') 



# Margin & Label 조정

for ax in axes : 

    ax.margins(0.12, 0.15)

    ax.xaxis.label.set_size(12)

    ax.xaxis.label.set_weight('bold')



    

# figure title    

plt.suptitle('Categorical Distribution', 

             fontsize=17, 

             fontweight='bold',

             x=0.05, y=1.06,

             ha='left' # horizontal alignment

            ) 



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)



sns.countplot(x='race', data=train_data, ax=axes[0], color="gray", edgecolor='black') 

sns.countplot(x='native_country', data=train_data, ax=axes[1], color='gray', edgecolor='black') 



# Margin & Label 조정

for ax in axes : 

    ax.margins(0.12, 0.15)

    ax.xaxis.label.set_size(12)

    ax.xaxis.label.set_weight('bold')



plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=90 )    

    

# figure title    

plt.suptitle('Categorical Distribution 2', 

             fontsize=17, 

             fontweight='bold',

             x=0.05, y=1.06,

             ha='left' # horizontal alignment

            ) 



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)



sns.countplot(x='relationship', data=train_data, ax=axes[0], palette="Set2", edgecolor='black') 

sns.countplot(x='marital_status', data=train_data, ax=axes[1], palette='Set2', edgecolor='black') 



# Margin & Label 조정

for ax in axes : 

    ax.margins(0.12, 0.15)

    ax.xaxis.label.set_size(12)

    ax.xaxis.label.set_weight('bold')



plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=50 )    

    

# figure title    

plt.suptitle('Categorical Distribution 2', 

             fontsize=17, 

             fontweight='bold',

             x=0.05, y=1.06,

             ha='left' # horizontal alignment

            ) 



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)



sns.countplot(x='workclass', data=train_data, ax=axes[0], palette="Set2", edgecolor='black') 

sns.countplot(x='occupation', data=train_data, ax=axes[1], palette='Set2', edgecolor='black') 

sns.countplot(x='education', data=train_data, ax=axes[2], palette='Set2', edgecolor='black') 



# Margin & Label 조정

for idx, ax in enumerate(axes) : 

    ax.margins(0.12, 0.15)

    ax.xaxis.label.set_size(12)

    ax.xaxis.label.set_weight('bold')

    plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=90 )    

    

# figure title    

plt.suptitle('Categorical Distribution 3', 

             fontsize=17, 

             fontweight='bold',

             x=0.05, y=1.06,

             ha='left' # horizontal alignment

            ) 



plt.tight_layout()

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.hist(train_data['age'], bins=10)

ax.set_ylim(0, 6000)

ax.set_title('Age Distribution')

plt.show()