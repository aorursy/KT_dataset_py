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
df_sum=pd.read_csv('/kaggle/input/corona-virus-report/country_wise_latest.csv')
df_sum.info()
df_sum.head()
s = (df_sum.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
labeled_data= df_sum.copy()


# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()

for col in object_cols:
    labeled_data[col] = label_encoder.fit_transform(df_sum[col])
    
print(labeled_data)
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,4))
plt.xlim(labeled_data.Confirmed.min(), labeled_data.Confirmed.max()*1.1)
sns.boxplot(x=labeled_data.Confirmed )

plt.figure(figsize=(10,4))
plt.xlim(labeled_data.Deaths.min(), labeled_data.Deaths.max()*1.1)
sns.boxplot(x=labeled_data.Deaths )

plt.figure(figsize=(10,4))
plt.xlim(labeled_data. Recovered.min(), labeled_data. Recovered.max()*1.1)
sns.boxplot(x=labeled_data. Recovered)


plt.figure(figsize=(10,4))
plt.xlim(labeled_data.Active.min(), labeled_data.Active.max()*1.1)
sns.boxplot(x=labeled_data.Active)
print(labeled_data.Confirmed[labeled_data.Confirmed>2000000].count())
print(labeled_data.Deaths[labeled_data.Deaths>80000].count())
print(labeled_data.Recovered[labeled_data.Recovered>750000].count())
print(labeled_data.Active[labeled_data.Active>1000000].count())
#labeled_data=labeled_data[labeled_data.Confirmed<700000]
#labeled_data=labeled_data[labeled_data.Deaths<80000]
#labeled_data=labeled_data[labeled_data.Recovered<500000]
#labeled_data=labeled_data[labeled_data.Active<150000]
#labeled_data
#plt.figure(figsize=(10,4))
#plt.xlim(labeled_data.Confirmed.min(), labeled_data.Confirmed.max()*1.1)
#sns.boxplot(x=labeled_data.Confirmed )

#plt.figure(figsize=(10,4))
#plt.xlim(labeled_data.Deaths.min(), labeled_data.Deaths.max()*1.1)
#sns.boxplot(x=labeled_data.Deaths )

#plt.figure(figsize=(10,4))
#plt.xlim(labeled_data. Recovered.min(), labeled_data. Recovered.max()*1.1)
#sns.boxplot(x=labeled_data. Recovered)


#plt.figure(figsize=(10,4))
#plt.xlim(labeled_data.Active.min(), labeled_data.Active.max()*1.1)
#sns.boxplot(x=labeled_data.Active)
labeled_data = labeled_data.drop(columns=['Deaths / 100 Recovered'])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
labeled_data = scaler.fit_transform(labeled_data)

print(labeled_data)
labeled_data=pd.DataFrame(labeled_data)
labeled_data.columns = ['Country/Region',  'Confirmed',  'Deaths',  'Recovered',  'Active',  'New cases', 'New deaths',  'New recovered',  'Deaths / 100 Cases',  'Recovered / 100 Cases', 'Confirmed last week',  '1 week change', '1 week % increase','WHO Region']
from sklearn.model_selection import train_test_split

# Select predictors
cols_to_use = ['Country/Region', 'Confirmed', 'Recovered' , 'Active' , 'New cases', 'New deaths'  ,'New recovered' , 'Confirmed last week' , 'Recovered / 100 Cases','Deaths / 100 Cases','1 week change','1 week % increase','WHO Region']

X = labeled_data[cols_to_use]

# Select target
y = labeled_data.Deaths

# Separate data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle= True)
# Model validation.
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

param_grid = {
    "learning_rate": [0.01, 0.025, 0.05],
    "max_depth":[5,8],
    "criterion": ["explained_variance"],
    "subsample":[0.5, 0.8, 0.85],
    "n_estimators":[500,1000]
}

gridsearch = GridSearchCV(XGBRegressor(), param_grid=param_grid, cv=5,
                         scoring='explained_variance',n_jobs=-1).fit(X_train, y_train,
             early_stopping_rounds=15, 
             eval_set=[(X_test, y_test)], 
             verbose=False)
print(gridsearch.score(X_train, y_train))
print(gridsearch.best_params_)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

predictions = gridsearch.predict(X_test)

print("Mean Absolute Error: " + str(mean_absolute_error(y_test, predictions)))

explained_variance_score=explained_variance_score(y_test, predictions)
print('Explained_variance_score: '+ str(explained_variance_score))
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
parameters = {
    "max_depth": [3, 5, 7, 9, 11, 13],
}

model_desicion_tree = DecisionTreeRegressor()

model_desicion_tree = GridSearchCV(
    model_desicion_tree, 
    parameters, 
    cv=5,
    scoring='explained_variance',
)

model_desicion_tree.fit(X_train, y_train)

print(f'Best parameters {model_desicion_tree.best_params_}')
print(f'Mean cross-validated accuracy score of the best_estimator: {model_desicion_tree.best_score_:.3f}')
parameters = {
    "n_estimators": [5, 10, 15, 20, 25], 
    "max_depth": [3, 5, 7, 9, 11, 13],
}

model_random_forest = RandomForestRegressor()

model_random_forest = GridSearchCV(
    model_random_forest, 
    parameters, 
    cv=5,
    scoring='explained_variance',
)

model_random_forest.fit(X_train, y_train)

print(f'Best parameters {model_random_forest.best_params_}')
print(f'Mean cross-validated accuracy score of the best_estimator: {model_random_forest.best_score_:.3f}')
my_model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=8,subsample=0.5 )
my_model.fit(X_train, y_train, 
             early_stopping_rounds=10, 
             eval_set=[(X_test, y_test)], 
             verbose=False)

predictions = my_model.predict(X_test)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_test)))

from sklearn import model_selection
kfold = 10 #validation subsamples
result_val = {} #list for results of validation

scores = model_selection.cross_val_score(model_desicion_tree, X, y, cv = kfold)
result_val['Desicion Tree'] = scores.mean()
scores = model_selection.cross_val_score(model_random_forest, X, y, cv = kfold)
result_val['Random Forest'] = scores.mean()
scores = model_selection.cross_val_score(my_model, X, y, cv = kfold)
result_val['XGBRegressor'] = scores.mean()
scores = model_selection.cross_val_score(gridsearch,X, y, cv = kfold)
result_val['Gridsearch'] = scores.mean()
pd.DataFrame.from_dict(data = result_val, orient='index').plot(kind='bar', legend=False)
import matplotlib.pyplot as plt
f = plt.figure(figsize=(15, 15))
plt.matshow(X.corr(), fignum=f.number)
plt.xticks(range(X.shape[1]), X.columns, fontsize=14, rotation=45)
plt.yticks(range(X.shape[1]), X.columns, fontsize=14)
cb = plt.colorbar();
from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(my_model, (10,14))
import seaborn as sns
plt.figure(figsize = (10, 10), dpi = 80)
# plot the data using seaborn
ax = sns.boxplot(x = "WHO Region", y = "Deaths", data = df_sum)


# ----------------------------------------------------------------------------------------------------
# prettify the plot

# change the font of the x and y ticks (numbers on the axis)
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# set and x and y label
ax.set_xlabel("WHO Region", fontsize = 14)
ax.set_ylabel("Deaths", fontsize = 14)

# set a title
ax.set_title("Boxplot", fontsize = 14);
ax.margins(y=0.05)
ax.set_ylim([-1,60000])
plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(X, kind="scatter", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()
import squarify 

# Prepare Data
df = df_sum.groupby('WHO Region').size().reset_index(name='counts')
labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
sizes = df['counts'].values.tolist()
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

# Draw Plot
plt.figure(figsize=(12,8), dpi= 80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

# Decorate
plt.title('WHO Region')
plt.axis('off')
plt.show()
