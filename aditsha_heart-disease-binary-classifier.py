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
#Importing important libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Training Data

raw_data = pd.read_csv('../input/MLChallenge-2/final.csv')

raw_data.head()
#Shuffle the data in order to ensure that each data point creates an "independent" change on the model, without being biased by the same points before them.

from sklearn.utils import shuffle

raw_data = shuffle(raw_data)



raw_data.head()
#Checking for Missing values in the data

#We also check for Caregorical Data but that isn't present so no need to label encode data

raw_data.isnull().sum(axis=0)
#Re-ensuring missing value data

raw_data.isnull().values.any()
#Understanding Dataset

raw_data.describe()
#Comparing the two classes if outputs in order to ensure the training is done on both equally

raw_data['target'].value_counts()
#Visualizing the above

sns.countplot(raw_data['target'])
#Examining age and target relationship as age is trivially the most important factor

sns.countplot(x='age', hue='target', data=raw_data, palette='colorblind', edgecolor=sns.color_palette('dark', n_colors=1))
#Defining the correlation between all parameters

raw_data.corr()
#Ploting the relationship of all parameters

plt.figure(figsize = (7,7))

sns.heatmap(raw_data.corr(), annot = True, fmt='.0%')
colms = raw_data.shape[1]

print("Number of columns = {}".format(colms))
# Creating an output y and input x 

# Also dropping Id parameter as it is not relevent data

y_train = raw_data.iloc[:, colms-1:colms]

x_train = raw_data.iloc[:, 1:colms-1]



x_train = np.array(x_train)

y_train = np.array(y_train)



print(f"Shape of X is {x_train.shape} and Shape of Y is {y_train.shape}")
#Scaling the data

from sklearn.preprocessing import StandardScaler

x_scalar = StandardScaler()



x_train = x_scalar.fit_transform(x_train)



print(x_train.shape, y_train.shape)
#Model

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 2)

forest.fit(x_train, y_train)
#Checking Model Score

model = forest

model.score(x_train, y_train)
#Test Input Data & Making Predictions

pred_data = pd.read_csv('../input/MLChallenge-2/Test.csv')

pred_data.head()
# Dropping Id parameter as it is not relevent to data output

colms = raw_data.shape[1]

print("Number of columns = {}".format(colms))



pred_data = pred_data.iloc[:, 1:colms]
pred_data.head()
#Checking for Missing values in Test Input Data

#We also check for Caregorical Data but that isn't present so no need to label encode any part of data

pred_data.isnull().sum(axis=0)
#Re-ensuring missing value in Test Input Data

pred_data.isnull().values.any()
#Converting pred_data into a numpy array

pred_data = np.array(pred_data)



#Scaling the Test Input Data

from sklearn.preprocessing import StandardScaler

x_scalar = StandardScaler()



pred_data = x_scalar.fit_transform(pred_data)



print(pred_data.shape)
#Making Predictions

y_pred = model.predict(pred_data)

print(y_pred.shape)
#The Predictions

print(y_pred)