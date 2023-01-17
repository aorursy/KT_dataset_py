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
df = pd.read_csv(r'/kaggle/input/mushroom-classification/mushrooms.csv')
df.head()
df.info()
df.shape
df['cap-shape'].value_counts()
mapping = {'x':'x','f':'f','k':'others','b':'others','s':'others','c':'others'}

df['cap-shape'].replace(mapping,inplace=True)
df['cap-shape'].value_counts()
df['cap-surface'].value_counts()
df.drop(df.loc[df['cap-surface']=='g'].index, inplace=True)
df['cap-surface'].value_counts()
df['cap-color'].value_counts()
mapping = {'n':'n','g':'g','e':'e','y':'y','w':'w','b':'others','u':'others','p':'others','c':'others','r':'others'}

df['cap-color'].replace(mapping,inplace=True)
df['cap-color'].value_counts()
df['bruises'].value_counts()
df['odor'].value_counts()
mapping = {'n':'n','f':'f','y':'others','s':'others','a':'others','l':'others','p':'others','c':'others','m':'others'}

df['odor'].replace(mapping,inplace=True)
df['odor'].value_counts()
df['gill-spacing'].value_counts()
df['gill-size'].value_counts()
df['gill-color'].value_counts()
mapping = {'b':'b','p':'p','w':'w','n':'n','g':'g','h':'h','u':'others','k':'others','e':'others','y':'others','o':'others','r':'others'}

df['gill-color'].replace(mapping,inplace=True)
df['gill-color'].value_counts()
df['stalk-shape'].value_counts()
df['stalk-root'].value_counts()
mapping = {'b':'b','?':'?','e':'e','c':'others','r':'others'}

df['stalk-root'].replace(mapping,inplace=True)
df['stalk-root'].value_counts()
df['stalk-surface-above-ring'].value_counts()
df.drop(df.loc[df['stalk-surface-above-ring']=='y'].index, inplace=True)
df['stalk-surface-above-ring'].value_counts()
df['stalk-surface-below-ring'].value_counts()
mapping = {'s':'s','k':'k','f':'others','y':'others'}

df['stalk-surface-below-ring'].replace(mapping,inplace=True)
df['stalk-color-above-ring'].value_counts()
mapping = {'w':'w','p':'p','g':'g','n':'others','b':'others','o':'others','e':'others','c':'others'}

df['stalk-color-above-ring'].replace(mapping,inplace=True)
df['stalk-color-above-ring'].value_counts()
df['stalk-color-below-ring'].value_counts()
mapping = {'w':'w','p':'p','g':'g','n':'others','b':'others','o':'others','e':'others','c':'others','y':'others'}

df['stalk-color-below-ring'].replace(mapping,inplace=True)
df['stalk-color-below-ring'].value_counts()
df['veil-type'].value_counts()
df.drop('veil-type',axis=1,inplace=True)
df['veil-color'].value_counts()
mapping = {'w':'w','n':'others','o':'others'}

df['veil-color'].replace(mapping,inplace=True)
df['ring-number'].value_counts()
mapping = {'o':'o','t':'others','n':'others'}

df['ring-number'].replace(mapping,inplace=True)
df['ring-type'].value_counts()
mapping = {'p':'p','e':'e','l':'l','f':'l','n':'l'}

df['ring-type'].replace(mapping,inplace=True)
df['spore-print-color'].value_counts()
mapping = {'w':'w','n':'n','k':'k','h':'h','r':'others','u':'others','y':'others','b':'others','o':'others'}

df['spore-print-color'].replace(mapping,inplace=True)
df['spore-print-color'].value_counts()
df['population'].value_counts()
mapping = {'v':'v','y':'y','s':'s','n':'others','a':'others','c':'others'}

df['population'].replace(mapping,inplace=True)
df['habitat'].value_counts()
mapping = {'d':'d','g':'g','p':'p','l':'l','u':'others','m':'others','w':'others'}

df['habitat'].replace(mapping,inplace=True)
df.columns
y = df['class']

X = df.drop('class',axis=1)
final_features = pd.get_dummies(X).reset_index(drop=True)

print('Features size:', final_features.shape)

final_features.head()
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(final_features,y,test_size=0.3,random_state=600)
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(train_x, train_y)
model.score(train_x,train_y)
model.score(test_x,test_y)
y_pred = model.predict(test_x)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_y,y_pred)

accuracy
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(test_y,y_pred)

conf_mat
true_positive = conf_mat[0][0]

false_positive = conf_mat[0][1]

false_negative = conf_mat[1][0]

true_negative = conf_mat[1][1]
Precision = true_positive/(true_positive+false_positive)

Precision
Recall = true_positive/(true_positive+false_negative)

Recall
F1_Score = 2*(Recall * Precision) / (Recall + Precision)

F1_Score