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
df = pd.read_csv('../input/vodafone-subset/vodafone-subset-6.csv')

df.head()
import seaborn as sns
sns.boxplot(df['viber_volume']);
df[ df['viber_count']>0 ].shape
df[ df['linkedin_count']>0 ].shape
sns.countplot(x='target', hue='car', data=df);
df.info()
df.dtypes.value_counts()
non_numeric = df.dtypes[ df.dtypes.values=='object' ]

non_numeric
df['SCORING'].value_counts()
df['device_type_rus'].value_counts()
# df['device_type_rus'] = df['device_type_rus'].map({'0': 0, 'phone':1, 'smartphone':2})

# df['SCORING'] = df['SCORING'].map({})
df_1 = pd.get_dummies(df, columns=['device_type_rus', 'SCORING'])

df_2 = df_1.drop(list(non_numeric.index[:-3]), axis=1)

df_3 = df_2.drop('user_hash', axis=1)
df_3.head()
df_3.dtypes.value_counts()
df_4 = df_3.dropna()

df_4.shape
from sklearn.model_selection import train_test_split

X = df_4.drop('target', axis=1)

y = df_4['target']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth=5, random_state=1)

tree.fit(X_train, y_train)
# Визуализация

from sklearn.tree import export_graphviz



export_graphviz(tree, out_file='tree.dot')

print(open('tree.dot').read()) 

# Далее скопировать полученный текст на сайт https://dreampuf.github.io/GraphvizOnline/ и сгенерировать граф

# Вставить картинку в блокнот: ![](ссылка)
X.columns[25]
y_pred = tree.predict(X_valid)

from sklearn.metrics import accuracy_score

accuracy_score(y_pred, y_valid)