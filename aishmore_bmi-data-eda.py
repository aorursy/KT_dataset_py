# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#Importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing data

data = pd.read_csv('../input/500-person-gender-height-weight-bodymassindex/500_Person_Gender_Height_Weight_Index.csv')

data.head()
data.isnull().any()
#Seeing what columns we have, what the data types there are, 

#and what number of entries per column we have.

data.info()
# Set default plot grid

sns.set_style('whitegrid')
# Index Historgram: Frequency of values falling under each Index [0,1,2,3,4,5]

plt.rcParams['figure.figsize'] = (6, 6)

sns.countplot(data['Index'], palette='YlGnBu')

ax = plt.gca()

ax.set_title("Histogram of Index")
# Height Historgram: Frequency of values falling under certain height intervals

plt.rcParams['figure.figsize'] = (30, 10)

sns.countplot(data['Height'], palette='YlGnBu')

ax = plt.gca()

ax.set_title("Histogram of Height")
# Weight Historgram: Frequency of values falling under certain weight intervals

plt.rcParams['figure.figsize'] = (30, 10)

sns.countplot(data['Weight'], palette='YlGnBu')

ax = plt.gca()

ax.set_title("Histogram of Weight")
# Plot relation between weight and height

sns.jointplot(x='Weight', y='Height', data=data, kind='kde')
# Trend in Gender based on relationship between Height and Weight

sns.lmplot(x='Height', y='Weight', hue='Gender', data=data,

           fit_reg=True, height=7, aspect=1.25, palette = "Accent")

ax = plt.gca()

ax.set_title("Height Vs Weight Data Grouped by Gender")
# Trend in Index based on relationship between Height and Weight 

sns.lmplot(x='Height', y='Weight', hue='Index', data=data,

           fit_reg=True, height=7, aspect=1.25, palette='Accent')

ax = plt.gca()

ax.set_title("Height Vs Weight Data Grouped by Index")
# Segregate data based on whether the gender is Male or Female

male_data = data[data['Gender']=='Male']

female_data = data[data['Gender']=='Female']
# Trend in Index based on relationship between Height and Weight 

male_data = data[data['Gender']=='Male']

female_data = data[data['Gender']=='Female']

sns.lmplot(x='Height', y='Weight', hue='Index', data=male_data,

           fit_reg=True, height=7, aspect=1.25,palette='Accent')

ax = plt.gca()

ax.set_title("Male Height Vs Weight Data Grouped by Index")



sns.lmplot(x='Height', y='Weight', hue='Index', data=female_data,

           fit_reg=True, height=7, aspect=1.25,palette='Accent')

ax = plt.gca()

ax.set_title("Female Height Vs Weight Data Grouped by Index")
# Gives us basic correlation index for numerical variables

data.corr()
# Provides visual context for correlations via color scale

plt.rcParams['figure.figsize'] = (8, 7)

sns.heatmap(data.corr(), annot=True)
plt.rcParams['figure.figsize'] = (8, 7)

sns.heatmap(male_data.corr(), annot=True)
plt.rcParams['figure.figsize'] = (8, 7)

sns.heatmap(female_data.corr(), annot=True)
# Ordinal Encoding

data["Gender"] = data["Gender"].astype('category')

data["Gender_Enc"] = data["Gender"].cat.codes

data.head()

# One Hot Encoding

dummies = pd.get_dummies(data['Gender'])

data = data.join(dummies)

data.head()
# Dropping last two columns with dummy values from one-hot encoding as they are redundant

data = data.drop(columns=['Male', 'Female'], axis=1)

data.head()
# Select columns to add to X and y sets

features = list(data.columns.values)

features.remove('Gender')

features.remove('Index')

X = data[features]

y = data['Index']
# Import additional required libraries

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.model_selection import *

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

from sklearn.model_selection import cross_val_score
# Import required class from sklearn library

from sklearn.model_selection import train_test_split



# Split X and y into train and test

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)
# Import required class from sklearn library

from sklearn.neighbors import KNeighborsClassifier



# Fit k-nearest neighbors classifier with training sets for n = 3

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)
# Run a prediction

y_pred = knn.predict(X_test)
# Import remaining required classes from sklearn

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

from sklearn.model_selection import cross_val_score
#Get confusion matrix

print(confusion_matrix(y_test,y_pred))
# Get classification report

print(classification_report(y_test,y_pred))
# Get accuracy score

score = np.mean(y_pred == y_test)

print(score)
# Get error rate

error = np.mean(y_pred != y_test)

print(error)
sns.regplot(x=y_test, y=y_pred)
fig = sns.jointplot(x=y_test, y=y_pred, kind='hex')

x0, x1 = fig.ax_joint.get_xlim()

y0, y1 = fig.ax_joint.get_ylim()

lims = [max(x0, y0), min(x1, y1)]

fig.ax_joint.plot(lims, lims, ':k')    
df = pd.DataFrame({ 'ytest':y_test,'ypred':y_pred})

sns.residplot('ytest','ypred',data=df) 