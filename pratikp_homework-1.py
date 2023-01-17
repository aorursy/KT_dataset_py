# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# df = pd.read_csv('../input/diabetes.csv')
df = pd.read_csv('../input/diabetes.csv')
dataset = pd.read_csv('../input/diabetes.csv', header=None)

# a = pd.DataFrame.from_csv('../input/diabetes.csv', sep=' ')



# Any results you write to the current directory are saved as output.
#x = df[np.logical_not(np.isnan(df))]
# drop the null values
df.dropna(thresh=1)
correlations = df.corr()
fig = plt.figure()
#ax = fig.gca(projection='3d')
# draw the graph
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()
# different view using seaborn library
sns.heatmap(correlations, annot = True)
df.hist(bins=50, figsize=(20, 15))
plt.show()
# Drop the SkinThickness column
df.drop('SkinThickness', 'columns')

# Show the true false ratio
# i = 0
# for column in df.columns[8:]:
#     a = df[column]
#     print(a)
#     if a == 0:
#         i = i + 1
#     print(df[column])
# print(i)
# pandas library is easier to use
true = (df['Outcome']==1).sum()
false = (df['Outcome']==0).sum()
print('True/False ratio =', round(true / (true + false), 5) * 100 , '%')
# remove rows where glucose is 0
df = df[(df.Glucose != 0)]
print(df.shape)



# split dataset
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[feature_names]
y = df.Outcome
# train the dataset
train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=42)
# train_set_labels = train_set[y].copy()
# train_set = train_set.drop("Outcome", axis=1)

# test_set_labels = test_set[y].copy()
# test_set = test_set.drop("Outcome", axis=1)

train, test = train_test_split(df, test_size = 0.3)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(train[['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']], train['Outcome'])
prediction = clf.predict(test[['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
accuracy_score(test['Outcome'],prediction)

dlf = tree.DecisionTreeClassifier(criterion="gini")
dlf = dlf.fit(train[['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']], train['Outcome'])
prediction2 = dlf.predict(test[['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
accuracy_score(test['Outcome'],prediction2)
