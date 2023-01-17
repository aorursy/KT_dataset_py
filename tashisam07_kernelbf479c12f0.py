#import packages needed for the procedure

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set

%matplotlib inline



#read data as data

data = pd.read_csv('../input/titanic/train.csv')



data.head()
#check the dimensions of the table

print('Dimension of the table is: ',data.shape)
data.describe()
sns.boxplot(x='Pclass',y='Survived',data=data)
#setup the figure size 

plt.rcParams['figure.figsize']=(20,10)



#make subplots

fig, axes=plt.subplots(nrows=2,ncols=2)



#Specify the features of interest

num_features=['Age','SibSp','Parch','Fare']

xaxes=num_features

yaxes=['Counts','Counts','Counts','Counts']



#draw Histogram

axes=axes.ravel()

for idx, ax in enumerate(axes):

    ax.hist(data[num_features[idx]].dropna(), bins=40)

    ax.set_xlabel(xaxes[idx], fontsize=20)

    ax.set_ylabel(yaxes[idx], fontsize=20)

    ax.tick_params(axis='both', labelsize=15)
#set up the figure size 

%matplotlib inline

plt.rcParams['figure.figsize']= (20,10)



#make subplots

fig, axes=plt.subplots(nrows=2, ncols=2)



#make the data read to feed into the visualiser

X_Survived=data.replace({'Survived': {1: 'yes',0: 'no'}}).groupby('Survived').size().reset_index(name='Counts')['Survived']

Y_Survived=data.replace({'Survived': {1: 'yes',0: 'no'}}).groupby('Survived').size().reset_index(name='Counts')['Counts']

#make the barplot

axes[0, 0].bar(X_Survived,Y_Survived)

axes[0, 0].set_title('Survived', fontsize=25)

axes[0, 0].set_ylabel('Counts', fontsize=20)

axes[0, 0].tick_params(axis='both',labelsize=15)



#make the data read to feed into the visualiser

X_Pclass=data.replace({'Pclass': {1: '1st',2: '2nd',3: '3rd'}}).groupby('Pclass').size().reset_index(name='Counts')['Pclass']

Y_Pclass=data.replace({'Pclass': {1: '1st',2: '2nd',3: '3rd'}}).groupby('Pclass').size().reset_index(name='Counts')['Counts']

#make the barplot

axes[0, 1].bar(X_Pclass,Y_Pclass)

axes[0, 1].set_title('Pclass', fontsize=25)

axes[0, 1].set_ylabel('Counts', fontsize=20)

axes[0, 1].tick_params(axis='both',labelsize=15)



#make the data read to feed into the visualiser

X_Sex=data.groupby('Sex').size().reset_index(name='Counts')['Sex']

Y_Sex=data.groupby('Sex').size().reset_index(name='Counts')['Counts']

#make the barplot

axes[1, 0].bar(X_Sex,Y_Sex)

axes[1, 0].set_title('Sex', fontsize=25)

axes[1, 0].set_ylabel('Counts', fontsize=20)

axes[1, 0].tick_params(axis='both',labelsize=15)



#make the data read to feed into the visualiser

X_Embarked=data.groupby('Embarked').size().reset_index(name='Counts')['Embarked']

Y_Embarked=data.groupby('Embarked').size().reset_index(name='Counts')['Counts']

#make the barplot

axes[1, 1].bar(X_Embarked,Y_Embarked)

axes[1, 1].set_title('Embarked', fontsize=25)

axes[1, 1].set_ylabel('Counts', fontsize=20)

axes[1, 1].tick_params(axis='both',labelsize=15)
sns.set(style="ticks",color_codes=True)

sns.catplot(x='Age',y='Survived',data=data)
#Consider Pclass for model training. 

#grid = sns.FacetGrid(data, col='Pclass', hue='Survived')

grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(data, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()