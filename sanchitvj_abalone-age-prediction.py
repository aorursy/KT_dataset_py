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
import seaborn as sns

color = sns.color_palette()

import matplotlib.pyplot as plt



import cufflinks as cf

import plotly.offline as py

color = sns.color_palette()

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

import plotly.tools as tls



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
df = pd.read_csv('../input/abalone-dataset/abalone.csv')
df.head(3)
df.shape
df['age'] = df['Rings']+1.5

df.drop('Rings', axis = 1, inplace = True)
df.info()
df.describe()
df.isnull().sum()
df.var()
df_sex = df['Sex'].value_counts()

print(df_sex.head())

trace = go.Bar(x = df_sex.index[: : -1] ,y = df_sex.values[: : -1], marker = dict(color = 'lightseagreen'))

data = [trace]

layout = go.Layout(height = 400, width = 500, title='Sex Distribution')

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows= 2, ncols = 4, figsize = (24,10))

sns.boxplot(ax = ax1, y = 'Length', data = df, color = 'green')

sns.boxplot(ax = ax2, y = 'Diameter', data = df, color = 'red')

sns.boxplot(ax = ax3, y = 'Height', data = df, color = 'limegreen')

sns.boxplot(ax = ax4, y = 'Whole weight', data = df, color = 'cyan')

sns.boxplot(ax = ax5, y = 'Shucked weight', data = df, color = 'salmon')

sns.boxplot(ax = ax6, y = 'Viscera weight', data = df, color = 'mediumorchid')

sns.boxplot(ax = ax7, y = 'Shell weight', data = df, color = 'lime')

sns.boxplot(ax = ax8, y = 'age', data = df, color = 'plum')
df1 = df.copy()

df2 = df.copy()

df3 = df.copy()



df_m = df1[df1['Sex'] == 'M']

df_m.drop('Sex', axis = 1, inplace= True)

df_f = df2[df2['Sex'] == 'F']

df_f.drop('Sex', axis = 1, inplace= True)

df_i = df3[df3['Sex'] == 'I']

df_i.drop('Sex', axis = 1, inplace= True)

df_m.drop(['age'], axis=1, inplace = True)

df_f.drop(['age'], axis=1, inplace = True)

df_i.drop(['age'], axis=1, inplace = True)



df_m = df_m.mean()

df_f = df_f.mean()

df_i = df_i.mean()

trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'M', marker = dict(color = 'cyan'))

trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'F', marker = dict(color = 'violet'))

trace3 = go.Bar(x = df_i.index[::-1], y = df_i.values[::-1], name = 'I', marker = dict(color = 'lightsteelblue'))

data = [trace1, trace2, trace3]

layout = go.Layout(title = 'Feature Distribution', width = 800)

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
df4 = df.copy()

df5 = df.copy()

df6 = df.copy()

df_m1 = df4[df4['Sex'] == 'M']

df_m1.drop('Sex', axis = 1, inplace= True)

df_f1 = df5[df5['Sex'] == 'F']

df_f1.drop('Sex', axis = 1, inplace= True)

df_i1 = df6[df6['Sex'] == 'I']

df_i1.drop('Sex', axis = 1, inplace= True)

df_m1.drop(['Length','Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight'], axis=1, inplace = True)

df_f1.drop(['Length','Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight'], axis=1, inplace = True)

df_i1.drop(['Length','Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight'], axis=1, inplace = True)



df_m1 = df_m1.mean()

df_f1 = df_f1.mean()

df_i1 = df_i1.mean()

trace1 = go.Bar(x = df_m1.index[::-1], y = df_m1.values[::-1], name = 'M', marker = dict(color = 'limegreen'))

trace2 = go.Bar(x = df_f1.index[::-1], y = df_f1.values[::-1], name = 'F', marker = dict(color = 'olive'))

trace3 = go.Bar(x = df_i1.index[::-1], y = df_i1.values[::-1], name = 'I', marker = dict(color = 'seagreen'))

data = [trace1, trace2, trace3]

layout = go.Layout(title = 'Feature Distribution', width = 750)

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
sns.pairplot(data = df, hue = 'Sex', palette = 'Dark2')
sns.set_style('darkgrid')

sns.lmplot(x = 'Length', y = 'age', data = df, hue = 'Sex', palette = 'Set1', scatter_kws={'edgecolor':'white', 'alpha':0.7, 'linewidth':0.5})
df.drop(df[(df['Length']<0.1) & (df['age'] < 5)].index, inplace=True)

df.drop(df[(df['Length']<0.8) & (df['age'] > 25)].index, inplace=True)

df.drop(df[(df['Length']>=0.8) & (df['age']< 25)].index, inplace=True)
sns.lmplot(x = 'Diameter', y = 'age', data = df, hue = 'Sex', palette = 'Dark2', scatter_kws={'edgecolor':'white', 'alpha':0.7, 'linewidth':0.5})
df.drop(df[(df['Diameter']<0.1) & (df['age'] < 5)].index, inplace=True)

df.drop(df[(df['Diameter']<0.6) & (df['age'] > 25)].index, inplace=True)

df.drop(df[(df['Diameter']>=0.6) & (df['age']< 25)].index, inplace=True)
sns.lmplot(x = 'Height', y = 'age', data = df, hue = 'Sex', palette = 'viridis', scatter_kws={'edgecolor':'white', 'alpha':0.7, 'linewidth':0.5})
df.drop(df[(df['Height']>0.4) & (df['age'] < 15)].index, inplace=True)

df.drop(df[(df['Height']<0.4) & (df['age'] > 25)].index, inplace=True)
sns.lmplot(x = 'Whole weight', y = 'age', data = df, hue = 'Sex', palette = 'magma', scatter_kws={'edgecolor':'white', 'alpha':0.7, 'linewidth':0.5})
df.drop(df[(df['Whole weight']>= 2.5) & (df['age'] < 25)].index, inplace=True)

df.drop(df[(df['Whole weight']<2.5) & (df['age'] > 25)].index, inplace=True)
sns.lmplot(x = 'Shucked weight', y = 'age', data = df, hue = 'Sex', palette = 'gist_heat', scatter_kws={'edgecolor':'white', 'alpha':0.7, 'linewidth':0.5})
df.drop(df[(df['Shucked weight']>= 1) & (df['age'] < 20)].index, inplace=True)

df.drop(df[(df['Shucked weight']<1) & (df['age'] > 20)].index, inplace=True)
sns.lmplot(x = 'Viscera weight', y = 'age', data = df, hue = 'Sex', palette = 'gnuplot', scatter_kws={'edgecolor':'white', 'alpha':0.7, 'linewidth':0.5})
df.drop(df[(df['Viscera weight']> 0.5) & (df['age'] < 20)].index, inplace=True)

df.drop(df[(df['Viscera weight']<0.5) & (df['age'] > 25)].index, inplace=True)
sns.lmplot(x = 'Shell weight', y = 'age', data = df, hue = 'Sex', palette = 'twilight_r', scatter_kws={'edgecolor':'white', 'alpha':0.7, 'linewidth':0.5})
df.drop(df[(df['Shell weight']> 0.6) & (df['age'] < 25)].index, inplace=True)

df.drop(df[(df['Shell weight']<0.8) & (df['age'] > 25)].index, inplace=True)
plt.figure(figsize = (8,6))

corr = df.corr()

sns.heatmap(corr, annot = True)
df.drop('Sex', axis = 1, inplace = True)

df.head()
df['age'].value_counts()
df['age'].mean()
df_1 = df.copy()
Age = []

for i in df_1['age']:

    if i > 11.12:

        Age.append('1')

    else:

        Age.append('0')

df_1['Age'] = Age

df_1.drop('age', axis = 1, inplace = True)

df_1.head()
df_1['Age'].value_counts()
X = df_1.drop('Age', axis = 1).values

y = df_1['Age'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



print('Classification Report: \n', classification_report(y_test, y_pred))

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

lr_train_acc = lr.score(X_train, y_train)

print('Training Score: ', lr_train_acc)

lr_test_acc = lr.score(X_test, y_test)

print('Testing Score: ', lr_test_acc)
svc = SVC(C = 1, gamma= 1)

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)



print('Classification Report: \n', classification_report(y_test, y_pred))

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

svc_train_acc = svc.score(X_train, y_train) 

print('Training Score: ', svc_train_acc)

svc_test_acc = svc.score(X_test, y_test)

print('Testing Score: ', svc_test_acc)
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors= i)

    knn.fit(X_train, y_train)

    y_predi = knn.predict(X_test)

    error_rate.append(np.mean(y_test != y_predi))

    

plt.figure(figsize = (10,8))

plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
knn = KNeighborsClassifier(n_neighbors= 31)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)



print('Classification Report: \n', classification_report(y_test, y_pred))

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

knn_train_acc = knn.score(X_train, y_train) 

print('Training Score: ', knn_train_acc)

knn_test_acc = knn.score(X_test, y_test)

print('Testing Score: ', knn_test_acc)
dt = DecisionTreeClassifier(max_depth = 5)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)



print('Classification Report: \n', classification_report(y_test, y_pred))

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

dt_train_acc = dt.score(X_train, y_train) 

print('Training Score: ', dt_train_acc)

dt_test_acc = dt.score(X_test, y_test)

print('Testing Score: ', dt_test_acc)
rf = RandomForestClassifier(n_estimators= 150, max_depth= 5)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



print('Classification Report: \n', classification_report(y_test, y_pred))

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

rf_train_acc = rf.score(X_train, y_train) 

print('Training Score: ', rf_train_acc)

rf_test_acc = rf.score(X_test, y_test)

print('Testing Score: ', rf_test_acc)
adb = AdaBoostClassifier(n_estimators= 100)

adb.fit(X_train, y_train)

y_pred = adb.predict(X_test)



print('Classification Report: \n', classification_report(y_test, y_pred))

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

adb_train_acc = adb.score(X_train, y_train) 

print('Training Score: ', adb_train_acc)

adb_test_acc = adb.score(X_test, y_test)

print('Testing Score: ', adb_test_acc)
gdb = GradientBoostingClassifier(n_estimators= 200, max_depth = 2, min_samples_leaf= 2)

gdb.fit(X_train, y_train)

y_pred = gdb.predict(X_test)



print('Classification Report: \n', classification_report(y_test, y_pred))

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

gdb_train_acc = gdb.score(X_train, y_train) 

print('Training Score: ', gdb_train_acc)

gdb_test_acc = gdb.score(X_test, y_test)

print('Testing Score: ', gdb_test_acc)
xgb = XGBClassifier(objective = "binary:logistic", n_estimators = 100, max_depth = 3, subsample = 0.8, colsample_bytree = 0.6, learning_rate = 0.1)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)



print('Classification Report: \n', classification_report(y_test, y_pred))

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

xgb_train_acc = xgb.score(X_train, y_train) 

print('Training Score: ', xgb_train_acc)

xgb_test_acc = xgb.score(X_test, y_test)

print('Testing Score: ', xgb_test_acc)
x = ['Logistic Regression','SVC', 'KNN', 'Decision Tree','Random Forest','AdaBoost','Gradient Boosting','XGBoost']

y1 = [lr_train_acc, svc_train_acc, knn_train_acc, dt_train_acc, rf_train_acc, adb_train_acc, gdb_train_acc, xgb_train_acc]

y2 = [lr_test_acc, svc_test_acc, knn_test_acc, dt_test_acc, rf_test_acc, adb_test_acc, gdb_test_acc, xgb_test_acc]



trace1 = go.Bar(x = x, y = y1, name = 'Training Accuracy', marker = dict(color = 'cyan'))

trace2 = go.Bar(x = x, y = y2, name = 'Testing Accuracy', marker = dict(color = 'violet'))

data = [trace1,trace2]

layout = go.Layout(title = 'Accuracy Plot', width = 750)

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)