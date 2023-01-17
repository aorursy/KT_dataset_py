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

#Import all required library

import pandas as pd

import numpy as np

import os 

# to save model

import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
data = pd.read_csv('/kaggle/input/glass/glass.csv')

data.describe()
data.head()
#create new column for "Type" to "g_type" form 0 or 1.

data['g_type'] = data.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

data.head()
# create "Glass correlation Marxix"

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'g_type']

mask = np.zeros_like(data[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Glass Correlation Matrix',fontsize=25)

sns.heatmap(data[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            #"BuGn_r" to reverse 

            linecolor='b',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
y = data.g_type

X = data.loc[:,['Na','Al','Ba']]

data.Type.value_counts().sort_index()
data.isnull().sum()
#apply model Logistic regression

model_logr = LogisticRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=10)

output_model = model_logr.fit(X_train, y_train)

output_model
pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model_logr, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(X_test)
model_logr.fit(X_train,y_train)

y_predict = model_logr.predict(X_test)

y_predict
print(classification_report(y_test,y_predict))
import matplotlib.pyplot as plt

# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test,y_predict)

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
plt.scatter(X,y, color='black')