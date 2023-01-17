# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#from sklearn.metrics import log_loss
#log_loss(y_test, model2.predict_proba(X_test))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import missingno as msno

#FOR CORELATION AND HEATMAP
import seaborn as sns
from sklearn.metrics import confusion_matrix

#COLORS
clr0= "darkseagreen"
clr1= "cadetblue"
clr2= 'dimgray'
clr3= 'darkgrey'
clr23= ['dimgray' ,'darkgrey']


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
print(df_train.shape)

df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
print(df_test.shape)
df_test['target'] = np.nan
df = pd.concat([df_train, df_test])
print(df.shape)
df= df.replace(" ?", np.NaN)
df.drop(columns=['education', 'relationship'],inplace=True)
df.dropna(inplace=True)
#Features Types
Numeric_features = [
    'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week', 'target'
]
Categorical_features = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country'
]
plt.figure( figsize=(12, 4))
sns.heatmap( df [Numeric_features].corr(), annot=True, cmap='Greys')
obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()
df[df.isnull().any(axis=1)]
df["race"] = df["race"].astype('category')
df["sex"] = df["sex"].astype('category')
df["marital-status"] = df["marital-status"].astype('category')
df["native-country"] = df["native-country"].astype('category')
df
df['native-country']= df['native-country'].replace([' Haiti', ' Jamaica', ' Mexico', ' Canada', ' Dominican-Republic', ' Guatemala', ' Nicaragua', ' Honduras',' Outlying-US(Guam-USVI-etc)'],value='NAmericas')
df['native-country']= df['native-country'].replace([' Columbia', ' Peru', ' Ecuador', ' Trinadad&Tobago', ' Cuba', ' El-Salvador', ' Puerto-Rico'],value='SAmericas')
df['native-country']= df['native-country'].replace([' Japan', ' Philippines', ' Thailand', ' Vietnam', ' Laos', ' China',
                                                   ' Hong', ' Taiwan', ' Cambodia'],value='Asia')
df['native-country']= df['native-country'].replace([' Iran', ' India'],value='ME')
df['native-country']= df['native-country'].replace([' Greece', ' Poland', ' Yugoslavia', ' France', ' Ireland', ' Germany',
                                                   ' Italy', ' Portugal', ' England', ' Scotland', ' Hungary', ' Holand-Netherlands'],value='EU')
df['native-country']= df['native-country'].replace(' South',value='Other')
df['workclass']= df['workclass'].replace([' State-gov', ' Federal-gov', ' Local-gov'],value='Gov')
df['workclass']= df['workclass'].replace([' Self-emp-not-inc', ' Self-emp-inc'],value='SelfEm')
df['marital-status']= df['marital-status'].replace([' Never-married', ' Divorced', ' Seperated', ' Widowed'],value='Unmarried')
df['marital-status']= df['marital-status'].replace([' Married-civ-spouse', ' Married-spouse-absent', ' Married-AF-spouse'],value='Married')
df= pd.get_dummies(df, columns=['sex', 'race', 'native-country', 'marital-status', 'workclass'])
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["occupation"] = lb_make.fit_transform(df["occupation"])
df
x_train = df.loc[df['target'].notna()].drop(columns=['target'])
y_train = df.loc[df['target'].notna()]['target']
x_test = df.loc[df['target'].isna()].drop(columns=['target'])
y_test = df.loc[df['target'].isna()]['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.3)
x_train, x_train, y_train, y_holdout = train_test_split(
    df[df['target'].notna()].values, y, test_size=0.3, random_state=20)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
tree= DecisionTreeClassifier(random_state=17)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
params = {'max_depth': np.arange(2,13), 'min_samples_leaf': np.arange(2,11)}
skf= StratifiedKFold(n_splits= 5, shuffle=True, random_state=17)
best_tree= GridSearchCV(estimator=tree, param_grid=params, cv=skf)
best_tree.fit(X_train, y_train)
best_tree.best_params_
best_tree.best_estimator_
best_tree.best_score_
pred_test_better = best_tree.predict(X_test)
accuracy_score(y_test, pred_test_better)
df_submit = pd.DataFrame({
    'uid': df.loc[df['target'].isna()]['uid'],
    'target': best_tree
})
d
df_submit.to_csv('/kaggle/working/submit.csv', index=False)
