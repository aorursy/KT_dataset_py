# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Step 2 : Data import

import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Use pandas to read in csv file

train = pd.read_csv('/kaggle/input/glass/glass.csv')

train.head(5)
train.describe()
#Step 3: Clean up data

# Use the .isnull() method to locate missing data

missing_values = train.isnull()

missing_values.tail
#create new column for "Type" to "g_type" form 0 or 1.

train['g_type'] = train.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

train.head()
#split dataset in features and target variable

feature_cols = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'g_type']

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Glass Correlation Matrix',fontsize=25)

sns.heatmap(train[feature_cols].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            #"BuGn_r" to reverse 

            linecolor='b',annot=True,annot_kws={"size":8},mask=None,cbar_kws={"shrink": .9});

X = train.loc[:,['Ca','Al','Ba']]

y = train.g_type # Target variable

train.Type.value_counts().sort_index # Features

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression



# instantiate the model (using the default parameters)

logreg = LogisticRegression()



# fit the model with data

logreg.fit(X_train,y_train)



#

y_pred=logreg.predict(X_test)




pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(logreg, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(X_test)



# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
import matplotlib.pyplot as plt

# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test,y_pred)

cnf_matrix

%matplotlib inline

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')