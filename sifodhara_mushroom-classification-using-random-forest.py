# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## import libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
mushroom_df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

## read our data
mushroom_df.head() ## check head of our dataset
mushroom_df.shape ## check shape of our dataset
mushroom_df.info() ## check info about our dataset
mushroom_df.isnull().sum()
mushroom_df.describe(include='all') ## check description of the dataset
mushroom_df.columns



## all variables of the dataset
len(mushroom_df.columns) ## total 23 variables are there in the dataset
# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# create a dictionary of classes and their totals

d = mushroom_df["class"].value_counts().to_dict()



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (18, 6))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

ax.pie(d.values(), # pass the values from our dictionary

       labels = d.keys(), # pass the labels from our dictonary

       autopct = '%1.1f%%', # specify the format to be plotted

       textprops = {'fontsize': 10, 'color' : "white"} # change the font size and the color of the numbers inside the pie

      )

# ----------------------------------------------------------------------------------------------------

# prettify the plot



# set the title

ax.set_title("Pie chart")



# set the legend and add a title to the legend

ax.legend(loc = "upper left", bbox_to_anchor = (1, 0, 0.5, 1), fontsize = 10, title = "mushroom Class")

plt.show()
cols = list(mushroom_df.columns)

plt.figure(figsize=(40,20))



for i in enumerate(cols):

    plt.subplot(5,5,i[0]+1)

    ax = sns.countplot(x=i[1],hue='class',data=mushroom_df)

    ax.set_xlabel(i[1],fontsize=20)

plt.tight_layout()

plt.show()

## let's check some variable with imbalance lebels

mushroom_df['veil-type'].value_counts()/mushroom_df.shape[0]

## hence remove this variable 

mushroom_df.drop('veil-type',axis=1,inplace=True)
mushroom_df.shape ## check final shape
## import libraries for data preproccessing

import sklearn 

from sklearn.preprocessing import OneHotEncoder

import category_encoders as ce

import sklearn

from sklearn.model_selection import train_test_split

df_train,df_test = train_test_split(mushroom_df,train_size=0.7,random_state=5) ## split data in train and test
y_train = df_train.pop('class') ## x and y split of train data

X_train = df_train
y_test = df_test.pop('class') ## x and y split of test data

X_test = df_test
encoder = ce.OneHotEncoder(cols=list(X_train.columns))



X_train = encoder.fit_transform(X_train) ## one hot encoding on all variables

X_train.head() ## check head of x 
X_test = encoder.transform(X_test) ## ebcoding done on x of test
X_test.head() ## check x test
y_train = y_train.apply(lambda x:0 if x=='e' else 1) ## convert target variable into 0 and 1
y_test = y_test.apply(lambda x:0 if x=='e' else 1) ## convert target variable into 0 and 1
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0)

sel.fit(X_train)  # fit finds the features with zero variance

# if we sum over get_support, we get the number of features that are not constant

sum(sel.get_support())
sel1 = VarianceThreshold(threshold=0.1)

sel1.fit(X_train)  # fit finds the features with 90% variance

# if we sum over get_support, we get the number of features that are not constant

sum(sel1.get_support())
X_train = X_train[X_train.columns[sel1.get_support()]] ## select variables with proper distribution of values
X_test = X_test[X_test.columns[sel1.get_support()]] ## select variables with proper distribution of values
from sklearn.ensemble import RandomForestClassifier ## import libraries for randomforest
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

def evaluate_model(rf):

    print("confusion matrix for training set: ",confusion_matrix(y_train,rf.predict(X_train)))

    print("accuracy score of training set: ",accuracy_score(y_train,rf.predict(X_train)))

    print("--"*50)

    print("confusion matrix for test set: ",confusion_matrix(y_test,rf.predict(X_test)))

    print("accuracy score of test set: ",accuracy_score(y_test,rf.predict(X_test)))

    print("**"*50)

rfc = RandomForestClassifier(random_state = 50)

rfc.fit(X_train,y_train) ## train our first model with default parameters
evaluate_model(rfc) ## evaluate the model
feature_score = pd.DataFrame({'features':X_train.columns,'feature score':rfc.feature_importances_}) 
feature_score.sort_values(by='feature score',ascending=False).head(10) ## check top 10 features
feature_score.sort_values(by='feature score',ascending=False).tail(10) ## check least top 10 features
X_train.drop('stalk-color-below-ring_3',axis=1,inplace=True) ## remove from train
X_test.drop('stalk-color-below-ring_3',axis=1,inplace=True) ## remove from test
rfc1 = RandomForestClassifier(random_state=10)

rfc1.fit(X_train,y_train) ## fit our secoend model
evaluate_model(rfc1) ## evaluate model
feature_score = pd.DataFrame({'features':X_train.columns,'feature score':rfc1.feature_importances_})
feature_score.sort_values(by='feature score',ascending=False).tail(10)
X_train.drop('cap-color_6',axis=1,inplace=True) ## checking by removing cap-color_6
X_test.drop('cap-color_6',axis=1,inplace=True)
rfc2 = RandomForestClassifier(random_state=20)

rfc2.fit(X_train,y_train) ## again fit the model
evaluate_model(rfc2) ## accuracy score still not changed
len(X_train.columns) ## final list of features
from sklearn import tree

plt.figure(figsize=(30,15))

tree.plot_tree(rfc2.estimators_[0],filled=True)

plt.show()



## plot one decision tree of the random forest