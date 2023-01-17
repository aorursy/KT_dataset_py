from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))


sns.set(style='white', context='notebook', palette='deep')
pylab.rcParams['figure.figsize'] = 12,8
warnings.filterwarnings('ignore')
mpl.style.use('ggplot')
sns.set_style('white')
%matplotlib inline
# import train and test to play with it
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
type(train)
type(test)
# Modify the graph above by assigning each species an individual color.
g = sns.FacetGrid(train, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
train.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False)
plt.figure()
#This gives us a much clearer idea of the distribution of the input attributes:


# To plot the species data using a box plot:

sns.boxplot(x="Fare", y="Age", data=test )
plt.show()
# Use Seaborn's striplot to add data points on top of the box plot 
# Insert jitter=True so that the data points remain scattered and not piled into a verticle line.
# Assign ax to each axis, so that each plot is ontop of the previous axis. 

ax= sns.boxplot(x="Fare", y="Age", data=train)
ax= sns.stripplot(x="Fare", y="Age", data=train, jitter=True, edgecolor="gray")
plt.show()
# Tweek the plot above to change fill and border color color using ax.artists.
# Assing ax.artists a variable name, and insert the box number into the corresponding brackets

ax= sns.boxplot(x="Fare", y="Age", data=train)
ax= sns.stripplot(x="Fare", y="Age", data=train, jitter=True, edgecolor="gray")

boxtwo = ax.artists[2]
boxtwo.set_facecolor('red')
boxtwo.set_edgecolor('black')
boxthree=ax.artists[1]
boxthree.set_facecolor('yellow')
boxthree.set_edgecolor('black')

plt.show()
# histograms
train.hist(figsize=(15,20))
plt.figure()
train["Age"].hist();
f,ax=plt.subplots(1,2,figsize=(20,10))
train[train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train[train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()

# scatter plot matrix
pd.plotting.scatter_matrix(train,figsize=(10,10))
plt.figure()
# violinplots on petal-length for each species
sns.violinplot(data=train,x="Fare", y="Age")
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()
# Using seaborn pairplot to see the bivariate relation between each pair of features
sns.pairplot(train, hue="Age")
# updating the diagonal elements in a pairplot to show a kde
sns.pairplot(train, hue="Age",diag_kind="kde")
# seaborn's kdeplot, plots univariate or bivariate density estimates.
#Size can be changed by tweeking the value used
sns.FacetGrid(train, hue="Survived", size=5).map(sns.kdeplot, "Fare").add_legend()
plt.show()
# Use seaborn's jointplot to make a hexagonal bin plot
#Set desired size and ratio and choose a color.
sns.jointplot(x="Age", y="Survived", data=train, size=10,ratio=10, kind='hex',color='green')
plt.show()
# we will use seaborn jointplot shows bivariate scatterplots and univariate histograms with Kernel density 
# estimation in the same figure
sns.jointplot(x="Age", y="Fare", data=train, size=6, kind='kde', color='#800000', space=0)
plt.figure(figsize=(7,4)) 
sns.heatmap(train.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()
sns.heatmap(train.corr(),annot=False,cmap='RdYlGn',linewidths=0.2)  
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
train['Pclass'].value_counts().plot(kind="bar");
sns.factorplot('Pclass','Survived',hue='Sex',data=train)
plt.show()
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(train[train['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(train[train['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(train[train['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()
# shape
print(train.shape)
#columns*rows
train.size
train.isnull().sum()
# remove rows that have NA's
#train = train.dropna()
print(train.info())
train['Age'].unique()
train["Pclass"].value_counts()

train.head(5) 
train.tail() 
train.sample(5) 
train.describe() 
train.isnull().sum()
train.groupby('Pclass').count()
train.columns
train.where(train['Age']==30)
train[train['Age']>7.2]
# Seperating the data into dependent and independent variables
X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values
cols = train.columns
features = cols[0:12]
labels = cols[4]
print(features)
print(labels)
X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
