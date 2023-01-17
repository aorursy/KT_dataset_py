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
import seaborn as sn
import numpy as np
import matplotlib.pyplot  as plt
df = pd.read_csv("/kaggle/input/income/adult.csv")
df.head()
df
df.shape
df.info()
df.describe()
df.isnull().sum()
df.workclass.value_counts()
df.workclass.value_counts().plot.barh()
df.education.value_counts().plot.barh()
df["marital-status"].value_counts()
df["marital-status"].value_counts().plot.barh()
df.occupation.value_counts()
df.occupation.value_counts().plot.barh()
df.relationship.value_counts().plot.barh()
df.race.value_counts().plot.barh()
df.sex.value_counts().plot.barh()
df.income.value_counts().plot.barh()
df["native-country"].value_counts()
sn.boxplot(x=df.income,y=df.age)
sn.boxplot(x="income",y="fnlwgt",data=df)
sn.boxplot(x="income",y="education-num",data=df)
sn.boxplot(x="hours-per-week",y='income',data=df)
sn.pairplot(df)
sn.pairplot(df,hue="income")
df_x = pd.get_dummies(df)
df_x
df_y=df_x.drop(['income_ <=50K','income_ >50K'], axis=1)
df_y.columns
df_y
Education_Income=pd.crosstab(df.education,df.income)
Education_Income
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_y,df.income,test_size=0.02,random_state=1000)
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
m = RandomForestClassifier(n_jobs=-1)
m.fit(X_train,y_train)
m.score(X_train,y_train)
m.score(X_test,y_test)
m = RandomForestClassifier(n_jobs=-1,oob_score=True)
m.fit(X_train,y_train)
m.score(X_train,y_train)
m.score(X_test,y_test)
m.oob_score_
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=5, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train,y_train)
m.score(X_train,y_train)
m.score(X_test,y_test)
m.oob_score_
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
def rf_feat_importance(m, df_x):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
fi = rf_feat_importance(m, df_y);
plot_fi(fi[:10]);
