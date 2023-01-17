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
import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
# read in data file 

csv_file = '/kaggle/input/drug200.csv'



df = pd.read_csv(csv_file)

df.head()
df.info()
df.shape
df.dtypes
df.isnull().sum()
counts = df['Drug'].value_counts().sort_values().plot(kind='bar')

plt.title('Drugs to Administer')

plt.xlabel('Drug Type')

plt.ylabel('Total Administered')

plt.show()
# features/indepenedent variable

X = df.drop(columns='Drug').values

print(X[:5])



# target variable

Y = df['Drug'].values

print(Y[:5])
# 'Sex' feature

le_sex = LabelEncoder()

# fit object with values of features

le_sex.fit(['M', 'F'])

# transfrom values

X[:,1] = le_sex.transform(X[:,1])



# 'BP' features

le_bp = LabelEncoder()

# fit object with values of features

le_bp.fit(['HIGH', 'NORMAL', 'LOW'])

# transfrom values

X[:,2] = le_bp.transform(X[:,2])



# 'Cholesterol'

le_chol = LabelEncoder()

# fit object with values of features

le_chol.fit(['HIGH', 'NORMAL'])

# transfrom values

X[:,3] = le_chol.transform(X[:,3])



X[:5]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)
# tree object

Tree = DecisionTreeClassifier()

# fit Tree object with training values

Tree.fit(x_train, y_train)
# estimates of testing set

yhat = Tree.predict(x_test)

yhat[:5]
# create new dataframe showing differences in estimated and actual

pred_act = pd.DataFrame({'Actual': y_test, 'Estimated':yhat})

pred_act.head(10)
from sklearn.metrics import classification_report



report = classification_report(y_test, yhat)

print(report)

# Score of the model ie accuracy

accuracy = Tree.score(x_test, y_test)

print('Model has an accuracy score of: ', str(round(accuracy*100, 2)), '%')
from sklearn.tree import export_graphviz

import graphviz
# feature and target columns

features = df.columns[:5]

target = df['Drug'].unique().tolist()



data = export_graphviz(Tree, out_file=None,

                      feature_names=features,

                      class_names=target, filled=True,

                      rounded=True, special_characters=True)



# create and show graph

graph = graphviz.Source(data)

graph