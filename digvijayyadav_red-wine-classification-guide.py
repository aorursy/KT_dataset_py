# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from pandas_profiling import ProfileReport

from sklearn import metrics

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler, LabelEncoder





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

print('The Dataset contains {} rows and {} columns '.format(df.shape[0], df.shape[1]))
df.head()
df.describe()
ProfileReport(df)
df['quality'].value_counts().index
plt.figure(figsize=(18,10))

sns.heatmap(df.corr(), annot=True, cmap=plt.cm.plasma)
df.isnull().sum().sum()
df.info()
df.hist(bins=40, figsize=(10,15))

plt.show()
df.plot(kind='density', subplots=True, layout=(4,3), sharex=False)

plt.show()
data = df.groupby(by="fixed acidity")[["fixed acidity", "density", "citric acid"]].first().reset_index(drop=True)



# Figure

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16, 6))



a = sns.distplot(data["fixed acidity"], ax=ax1, hist=False, kde_kws=dict(lw=6, ls="--"))

b = sns.distplot(data["density"], ax=ax2, hist=False, kde_kws=dict(lw=6, ls="--"))

c = sns.distplot(data["citric acid"], ax=ax3, hist=False, kde_kws=dict(lw=6, ls="--"))



a.set_title("Fixed Acidity Distribution", fontsize=16)

b.set_title("Density Distribution", fontsize=16)

c.set_title("Citric Acid distribution", fontsize=16)
from pandas.plotting import scatter_matrix



sm = scatter_matrix(df, figsize=(16, 10), diagonal='kde')



[s.xaxis.label.set_rotation(40) for s in sm.reshape(-1)]

[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]



#May need to offset label when rotating to prevent overlap of figure



[s.get_yaxis().set_label_coords(-0.6,0.5) for s in sm.reshape(-1)]



#Hide all ticks



[s.set_xticks(()) for s in sm.reshape(-1)]

[s.set_yticks(()) for s in sm.reshape(-1)]

plt.show()
# Dividing wine as good and bad by giving the limit for the quality



bins = (2, 6, 8)

group_names = ['bad', 'good']

df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

# Now lets assign a labels to our quality variable



label_quality = LabelEncoder()



# Bad becomes 0 and good becomes 1

df['quality'] = label_quality.fit_transform(df['quality'])

print(df['quality'].value_counts())

sns.countplot(df['quality'])

plt.show()
x = df.drop(['quality'], axis=1)

y = df['quality']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 50)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)





cols = ['fixed acidity',

'volatile acidity',

'citric acid',

'residual sugar',

'chlorides',

'free sulfur dioxide',

'total sulfur dioxide',

'density',

'pH',

'sulphates',

'alcohol'

       ]
dtc = DecisionTreeClassifier(max_depth=200)

dtc.fit(x_train, y_train)

preds = dtc.predict(x_test)

score = dtc.score(x_test, y_test)

score
preds[:5]
y_test[:5]
Ks = 100

mean_acc = np.zeros((Ks-1))

for n in range(1,Ks):

    

    #Train Model and Predict  

    dtc = DecisionTreeClassifier(max_depth = n).fit(x_train,y_train)

    yhat=dtc.predict(x_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)



mean_acc
print( "The best accuracy was with", mean_acc.max(), "with depth =", mean_acc.argmax()+1)
cf = metrics.classification_report(preds,y_test)

print(cf)
rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

preds = rfc.predict(x_test)

score = rfc.score(x_test,y_test)

score
preds[:5]
y_test[:5]
Ks = 100

mean_acc = np.zeros((Ks-1))

for n in range(1,Ks):

    

    #Train Model and Predict  

    rfc = RandomForestClassifier(n_estimators = n).fit(x_train,y_train)

    yhat=dtc.predict(x_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)



mean_acc
print( "The best accuracy was with", mean_acc.max(), "with n_estimator =", mean_acc.argmax()+1)
cf = metrics.classification_report(preds,y_test)

print(cf)
rfc_plot = metrics.plot_roc_curve(rfc, x_test,y_test)
dtc_plot = metrics.plot_roc_curve(dtc, x_test,y_test)
dtc_eval = cross_val_score(dtc, x_test, y_test, cv=10)

print('Cross Val Score accuracy is {:.2f}'.format(dtc_eval.mean()))
rfc_eval = cross_val_score(rfc, x_test, y_test, cv=10)

print('Cross Val Score accuracy is {:.2f}'.format(rfc_eval.mean()))
tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

dtc_cv = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=10)

dtc_cv.fit(x_test, y_test)
dtc_cv.best_params_
dtc_new = DecisionTreeClassifier(criterion='entropy', max_depth = 8)

dtc_new.fit(x_train,y_train)

new_score  = dtc_new.score(x_test, y_test)

new_score
param_grid = { 

    'n_estimators': [200, 500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}



rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

rfc_cv.fit(x_test, y_test)
rfc_cv.best_params_
rfc_new = RandomForestClassifier(criterion='gini', max_depth = 5, max_features='auto', n_estimators=500)

dtc_new.fit(x_train,y_train)

new_score  = dtc_new.score(x_test, y_test)

new_score