# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
print(train.shape)

print(test.shape)
train.head(10)
test.head(10)
plt.figure(figsize=(10,8))

sns.set(style="whitegrid")

sns.countplot(train['label'], linewidth=1.25, edgecolor=sns.color_palette('magma'), palette='Pastel2')

plt.xlabel('Count of target column');



train['label'].value_counts()
train.isnull().any().describe()
test.isnull().any().describe()
# Seperating the target column and features column

x = train.drop(['label'], axis=1).values

y = train.label.values
# Splitting the data in to test & train to make our model learn.

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.2, random_state=42)

train_x.shape, test_x.shape, train_y.shape, test_y.shape
feature_scale = StandardScaler()

train_x = feature_scale.fit_transform(train_x)

test_x = feature_scale.transform(test_x)
# Scaling test data set as well

scale_test_data = feature_scale.fit_transform(test)
# Finding out the image at a particular index

plt.figure(figsize=(8,6))

index=5

print("Label: " + str(train_y[index]))

plt.imshow(train_x[index].reshape((28,28)),cmap='gray')

plt.show()
rf = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=10, random_state=42)

rf.fit(train_x, train_y)
#predict value of label using classifier

prediction= rf.predict(test_x)
print('Random Forest training accuracy: ', rf.score(train_x, train_y))

print('Random Forest test accuracy: ' + str(accuracy_score(test_y,prediction)))
# Confusion Matrix for Random Forest



plt.figure(figsize=(15,10))

sns.set(font_scale=1.3)

sns.heatmap(confusion_matrix(test_y,prediction), annot = True, fmt = '.0f', cmap = 'Oranges_r')

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Random Forest Confusion Matrix\n\n')

plt.show()
#predict test data

prediction_test = rf.predict(scale_test_data)
submission['Label'] = prediction_test
submission.head(6)
submission.to_csv("Submission-khushboo.csv", index=False)