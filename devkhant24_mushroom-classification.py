# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
df.drop('veil-type',axis=1,inplace=True)
a = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

        'stalk-surface-below-ring', 'stalk-color-above-ring',

        'stalk-color-below-ring', 'veil-color', 'ring-number',

        'ring-type', 'spore-print-color', 'population', 'habitat']
for i in a:

    print(i,':',len(df[i].unique()))
dum = pd.get_dummies(df[['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

                    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

                    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

                    'stalk-surface-below-ring', 'stalk-color-above-ring',

                    'stalk-color-below-ring', 'veil-color', 'ring-number',

                    'ring-type', 'spore-print-color', 'population', 'habitat']],drop_first=True)
df.drop(['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

        'stalk-surface-below-ring', 'stalk-color-above-ring',

        'stalk-color-below-ring', 'veil-color', 'ring-number',

        'ring-type', 'spore-print-color', 'population', 'habitat'],axis=1,inplace=True)
df = pd.concat([dum,df],axis=1)
x = df.drop('class_p',axis=1)

y = df['class_p']
best = SelectKBest(score_func=chi2,k='all')

fit = best.fit(x,y)
dfs = pd.DataFrame(fit.scores_)

dfc = pd.DataFrame(x.columns)

feat = pd.concat([dfc,dfs],axis=1)

feat.columns =['Specs','Score']

print(feat.nlargest(20,'Score'))
x = df[['stalk-surface-below-ring_s','gill-color_n','odor_y','odor_s','stalk-surface-above-ring_s','habitat_p','spore-print-color_w'

       ,'population_v','gill-spacing_w','spore-print-color_k','spore-print-color_n','bruises_t','ring-type_p','ring-type_l',

       'spore-print-color_h','gill-size_n','stalk-surface-below-ring_k','stalk-surface-above-ring_k','odor_f','odor_n']]
xg = xgboost.XGBClassifier()

lg = LogisticRegression()

rf = RandomForestClassifier(random_state=1,criterion='entropy')

dt = DecisionTreeClassifier(random_state=1,criterion='gini')

svm = SVC(kernel='linear')

knn = KNeighborsClassifier()

ss = StandardScaler()
score = cross_val_score(dt,x,y,cv=5,scoring='accuracy')

score.mean()