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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn import metrics
glass_data = pd.read_csv('/kaggle/input/glass/glass.csv')

glass_data.head()
glass_data.describe()
sns.countplot(x='Type', data=glass_data)

pd.value_counts(glass_data['Type'].values.ravel())
missing_values = glass_data.isnull()

missing_values.head()
sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
glass_data['g_type'] = glass_data.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

glass_data.head()
# create "Glass correlation Marxix"

columns = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'g_type']

matrix = np.zeros_like(glass_data[columns].corr(), dtype=np.bool) 

matrix[np.triu_indices_from(matrix)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Glass Correlation Matrix',fontsize=25)

sns.heatmap(glass_data[columns].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            linecolor='b',annot=True,annot_kws={"size":8},mask=matrix,cbar_kws={"shrink": .9});
y = glass_data.g_type

X = glass_data.loc[:,['Al','Ba']]

glass_data.Type.value_counts().sort_index()
model_logr = LogisticRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=220)

output_model = model_logr.fit(X_train, y_train)

output_model
score = output_model.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = output_model.predict(X_test)
model_logr.fit(X_train,y_train)

y_predict = model_logr.predict(X_test)

y_predict
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict.flatten()})

df.head()
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