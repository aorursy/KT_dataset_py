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
df = pd.read_csv('../input/mobile-price-classification/train.csv')
df
df.columns
y = df.price_range
df.price_range.value_counts
X = df.drop(columns='price_range')
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report
model1 = DecisionTreeClassifier()

model2 = DecisionTreeClassifier(max_leaf_nodes=100)

model3 = DecisionTreeClassifier(max_leaf_nodes=120)

model4 = DecisionTreeClassifier(max_leaf_nodes=140)

model5 = DecisionTreeClassifier(max_leaf_nodes=150)
my_list = [model1,model2,model3,model4,model5]

for model in my_list:

    model.fit(X_train,y_train)

    preds = model.predict(X_valid)

    print(accuracy_score(y_valid, preds))

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
model1_ra = RandomForestClassifier()

model2_ra = RandomForestClassifier(n_estimators=50)

model3_ra= RandomForestClassifier(n_estimators=150)

model4_ra = RandomForestClassifier(n_estimators=200)

model5_ra = RandomForestClassifier(n_estimators=250)

model6_ra = RandomForestClassifier(n_estimators=300)
my_list = [model1_ra,model2_ra,model3_ra,model4_ra,model5_ra,model6_ra]

for model in my_list:

    model.fit(X_train,y_train)

    preds = model.predict(X_valid)

    print(accuracy_score(y_valid, preds))
from sklearn.naive_bayes import GaussianNB
model_na = GaussianNB()
model_na.fit(X_train,y_train)

preds = model_na.predict(X_valid)

print(accuracy_score(y_valid, preds))