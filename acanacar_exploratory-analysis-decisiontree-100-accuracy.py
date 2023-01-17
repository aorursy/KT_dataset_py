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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint as pp
df = pd.read_csv('../input/drug-classification/drug200.csv',delimiter=',')

df.sample(10)
print(df.isnull().sum())

print(df.describe(include='all'))

print(df.shape)
types_lookup = {}
types_lookup['BP'] = list(df['BP'].unique())
types_lookup['Cholesterol'] = list(df['Cholesterol'].unique())
types_lookup['Drug'] = list(df['Drug'].unique())
types_lookup['Sex'] = list(df['Sex'].unique()) 
pp(types_lookup)
sns.countplot(x=df.Drug,data=df)
df.groupby('Drug')['BP'].value_counts().unstack().plot.bar()
df.groupby('Drug')['Cholesterol'].value_counts().unstack().plot.bar()
df.groupby('Drug')['Sex'].value_counts().unstack().plot.bar()
df.boxplot(column=['Age'], by=['Drug'], figsize=(12, 8))

# I just wanted to see threshold valus for drug-a and drug-b
gb = df.groupby('Drug')
print(f"For drug-B : {gb.get_group('drugB')['Age'].describe()}")  # drug b segment has minimum age of 51. 
print(f"For drug-A : {gb.get_group('drugA')['Age'].describe()}")  # drug a segment has max age of 50. 
plt.figure(figsize=(12, 8))
sns.swarmplot(x='Drug',
              y='Na_to_K',
              data=df)

plt.title('Drug and Sodium/Potassium Ratio')

df_before_preprocess = df.copy()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
cols_ = ['Sex', 'BP', 'Cholesterol', 'Drug']
for col in cols_:
    df[col] = label_encoder.fit_transform(df[col])
from sklearn.model_selection import train_test_split
X = df.drop('Drug', axis=1)
y = df['Drug']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, stratify = y) 
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)
from sklearn.metrics import accuracy_score,confusion_matrix
predictions = dt.predict(X_test)
print(accuracy_score(y_test,predictions))
print(df_before_preprocess['Drug'])
print(df['Drug'])
from sklearn import tree
feature_names = list(X_train.columns)
class_names = list(types_lookup['Drug'])

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dt,
               feature_names = feature_names, 
               class_names=class_names,
               filled = True);

