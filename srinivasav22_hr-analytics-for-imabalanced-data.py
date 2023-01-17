# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from sklearn.datasets import make_classification

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import metrics

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train_LZdllcl.csv')

df.columns
df['department'].value_counts()
plt.rcParams['figure.figsize'] = [10, 5]

ct = pd.crosstab(df.department,df.is_promoted,normalize='index')

ct.plot.bar(stacked=True)

plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))
reg = pd.crosstab(df.region,df.is_promoted,normalize='index')

reg.plot.bar(stacked=True)

plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))
plt.rcParams['figure.figsize'] = [5, 5]

edu = pd.crosstab(df.education,df.is_promoted,normalize='index')

edu.plot.bar(stacked=True)

plt.rcParams['figure.figsize'] = [5, 5]

plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))
pd.crosstab(df.gender,df.is_promoted,normalize='index')
pd.crosstab(df.recruitment_channel,df.is_promoted,normalize='index')
pd.crosstab(df['KPIs_met >80%'],df.is_promoted,normalize='index')
sales = df[df['department']=='Sales & Marketing']

operations = df[df['department']=='Operations']

technology = df[df['department']=='Technology']

hr = df[df['department']=='HR']

fin = df[df['department']=='Finance']

legal = df[df['department']=='Legal']

RnD = df[df['department']=='R&D']

pd.crosstab(sales.gender,sales.is_promoted,normalize='index')
pd.crosstab(operations.gender,operations.is_promoted,normalize='index')
pd.crosstab(technology.gender,technology.is_promoted,normalize='index')
plt.rcParams['figure.figsize'] = [3, 5]

gender = pd.crosstab(RnD.gender,RnD.is_promoted,normalize='index')

gender.plot.bar(stacked=True)

plt.legend(title='is_promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
rating = pd.crosstab(df.previous_year_rating,df.is_promoted,normalize='index')

rating.plot.bar(stacked=True)

plt.legend(title='is_promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
bins = [30,40,50,60,70,80,90,100]

labels = ['30-40','40-50','50-60','60-70','70-80','80-90','90-100']

df['score_binned'] = pd.cut(df['avg_training_score'], bins=bins, labels=labels)

df['score_binned'].value_counts()
plt.rcParams['figure.figsize'] = [10, 5]

score_bin = pd.crosstab(df.score_binned,df.is_promoted,normalize='index')

score_bin.plot.bar(stacked=True)

plt.legend(title='is_promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
plt.rcParams['figure.figsize'] = [5, 5]

age_bins = [20,30,40,50,60]

age_labels = ['20-30','30-40','40-50','50-60']

df['age_binned'] = pd.cut(df['age'], bins=bins, labels=labels)

df['age_binned'].value_counts()

age_bin = pd.crosstab(df.age_binned,df.is_promoted,normalize='index')

age_bin.plot.bar(stacked=True)

plt.rcParams['figure.figsize'] = [5, 5]

plt.legend(title='is_promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
plt.rcParams['figure.figsize'] = [14, 5]

sns.scatterplot(x='age',y='avg_training_score',hue='is_promoted',data=df)
df.groupby(["education"])['avg_training_score'].mean()
df.isnull().any()
df['previous_year_rating'] = df.groupby(["KPIs_met >80%"])["previous_year_rating"].apply(lambda x: x.fillna(x.mean()))

df["education"] = df["education"].astype('object')

df['education'] = df.groupby(["department"])["education"].apply(lambda x: x.fillna(x.value_counts().index[0]))
scaled_features = df.copy()

col_names = ['no_of_trainings', 'age','previous_year_rating','length_of_service','awards_won?','avg_training_score']

label_names = ['department','gender','recruitment_channel','region']

features = scaled_features[col_names]

scaler = preprocessing.StandardScaler().fit(features.values)

features = scaler.transform(features.values)

scaled_features = pd.get_dummies(scaled_features, columns=label_names, drop_first=True)

scaled_features[col_names] = features

scaled_features.drop(columns=['employee_id','age','education','score_binned','age_binned'],inplace=True)
X_train, X_test, y_train, y_test = train_test_split(

    scaled_features.loc[:, scaled_features.columns != 'is_promoted'], scaled_features['is_promoted'], test_size=0.33, random_state=42)

#forest = RandomForestClassifier(n_jobs=-1, random_state=0,class_weight='balanced',n_estimators=100,bootstrap=True, max_depth=80)

forest = GradientBoostingClassifier(loss='exponential',max_features='auto')

param_grid = {

    'n_estimators': [200,500,800]

}

grid_search = GridSearchCV(estimator = forest, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data

grid_search.fit(X_train, y_train)

grid_search.best_params_
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importance = {}

for i in range(len(X_train.columns)):

    feature_importance[X_train.columns[i]] = feature_importances[i]

importance_df = pd.DataFrame(list(feature_importance.items()),columns=['feature','importance'])

importance_df = importance_df.sort_values('importance',ascending=False)

plt.xticks(rotation='vertical')

plt.rcParams['figure.figsize'] = [18, 10]

sns.barplot(x="feature",y="importance",data=importance_df)
pred = grid_search.predict(X_test)

accuracy = metrics.accuracy_score(y_test, pred)

'accuracy - '+str(accuracy)
f1 = metrics.f1_score(y_test, pred)

'f1 score - '+str(f1)
recall = metrics.recall_score(y_test,pred)

'recall - '+str(recall)
precision = metrics.precision_score(y_test,pred)

'precision - '+str(precision)