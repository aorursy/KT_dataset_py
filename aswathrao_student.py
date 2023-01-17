# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install fpdf

import pandas as pd

import matplotlib

from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig

from fpdf import FPDF

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn import datasets 

import pandas as pd

import numpy as np

import seaborn as sns

sns.set(style='white')

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/student-performance-in-class/iitstudentperformance.csv')
df.columns
sns.pairplot(df,hue='Class');

savefig('pairplot.png')



pdf = FPDF()

pdf.add_page()

pdf.set_xy(0, 0)

pdf.set_font('Arial', '', 14)

pdf.cell(200, 50, txt="Pair Plot",ln=1, align="C")

pdf.ln()

# Then put a blue underlined link

pdf.set_text_color(0, 0, 255)

pdf.set_font('', 'U')

pdf.image('pairplot.png',x = 40, y = 40, w = 150, h = 150)

df.hist(figsize=(15, 7))

savefig('Histogram.png')



#pdf = FPDF()

pdf.add_page()

pdf.set_xy(0, 0)

pdf.set_font('Arial', '', 14)

pdf.cell(200, 50, txt="Histogram",ln=1, align="C")

pdf.ln()

# Then put a blue underlined link

pdf.set_text_color(0, 0, 255)

pdf.set_font('', 'U')

pdf.image('Histogram.png',x = 40, y = 40, w = 150, h = 150)
g = sns.PairGrid(df, hue="Class",palette='Set1')

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();

savefig('Pairgrid.png')



#pdf = FPDF()

pdf.add_page()

pdf.set_xy(0, 0)

pdf.set_font('Arial', '', 14)

pdf.cell(200, 50, txt="Pairgrid.png",ln=1, align="C")

pdf.ln()

# Then put a blue underlined link

pdf.set_text_color(0, 0, 255)

pdf.set_font('', 'U')

pdf.image('Pairgrid.png',x = 40, y = 40, w = 150, h = 150)


melt = pd.melt(df,id_vars='Class',value_vars=['raisedhands','VisITedResources','AnnouncementsView','Discussion'])

plt.rcParams['figure.figsize']=(15,4)

sns.swarmplot(x='variable',y='value',hue='Class' , data=melt,palette='Set1')

plt.ylabel('Values from 0 to 100')

plt.xlabel('Attributes')

plt.title('High, Middle & Low level students');

savefig('Swarmplot.png')



#pdf = FPDF()

pdf.add_page()

pdf.set_xy(0, 0)

pdf.set_font('Arial', '', 14)

pdf.cell(200, 50, txt="Swarmplot.png",ln=1, align="C")

pdf.ln()

# Then put a blue underlined link

pdf.set_text_color(0, 0, 255)

pdf.set_font('', 'U')

pdf.image('Swarmplot.png',x = 40, y = 40, w = 150, h = 150)
sns.countplot(data=df,x='gender',hue='Class',palette='Set1');

savefig('CountGC.png')



#pdf = FPDF()

pdf.add_page()

pdf.set_xy(0, 0)

pdf.set_font('Arial', '', 14)

pdf.cell(200, 50, txt="Class Vs Gender.png",ln=1, align="C")

pdf.ln()

# Then put a blue underlined link

pdf.set_text_color(0, 0, 255)

pdf.set_font('', 'U')

pdf.image('CountGC.png',x = 40, y = 40, w = 150, h = 150)
sns.countplot(data=df,x='Relation',hue='Class',palette='Set1');

savefig('CountRC.png')



#pdf = FPDF()

pdf.add_page()

pdf.set_xy(0, 0)

pdf.set_font('Arial', '', 14)

pdf.cell(200, 50, txt="Class Vs Relation.png",ln=1, align="C")

pdf.ln()

# Then put a blue underlined link

pdf.set_text_color(0, 0, 255)

pdf.set_font('', 'U')

pdf.image('CountRC.png',x = 40, y = 40, w = 150, h = 150)
df = df.drop('PlaceofBirth',1)
ls = ['gender','Relation','Topic','SectionID','GradeID','NationalITy','Class','StageID','Semester','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays']



for i in ls:

    plt.figure(figsize=(10, 7))

    g = sns.factorplot(i,data=df,kind='count')
target = df.pop('Class')



X = pd.get_dummies(df)
X
le = LabelEncoder()

y = le.fit_transform(target)
y
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

ss = StandardScaler()

#print X_train

X_train_std = ss.fit_transform(X_train)

X_test_std = ss.fit_transform(X_test)

#print X_train_std
type(X_train_std)
feat_labels = X.columns[:58]

forest = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)

forest.fit(X_train,y_train)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

#for f in range(X_train.shape[1]):

#    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))

plt.figure(figsize=(20, 15))

h = sns.barplot(importances[indices],feat_labels[indices])

savefig('Barplot.png')



#pdf = FPDF()

pdf.add_page()

pdf.set_xy(0, 0)

pdf.set_font('Arial', '', 14)

pdf.cell(200, 50, txt="Feature Importance",ln=1, align="C")

pdf.ln()

# Then put a blue underlined link

pdf.set_text_color(0, 0, 255)

pdf.set_font('', 'U')

pdf.image('Barplot.png',x = 40, y = 40, w = 150, h = 150)
X_train_new = X_train

X_test_new = X_test

print('X_train in columns')

print(X_train.columns)

print('X_test in columns')

print(X_test.columns)  
ls = ['VisITedResources','raisedhands','AnnouncementsView','StudentAbsenceDays_Above-7','StudentAbsenceDays_Under-7','Discussion']



for i in X_train.columns:

    if i in ls:

        pass

    else:

        X_train_new.drop(i , axis=1, inplace=True)



for i in X_test.columns:

    if i in ls:

        pass

    else:

        X_test_new.drop(i , axis=1, inplace=True)
models = []

models.append(('LR', LinearRegression()))

models.append(('LASSO', Lasso()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))

models.append(('SVR', SVR()))
# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=7)

    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='neg_mean_squared_error')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Standardize the dataset

pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))

pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))

pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=10, random_state=7)

    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='neg_mean_squared_error')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)



fig = plt.figure()

fig.suptitle('Scaled Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

savefig('AlgoCompare.png')

ax.set_xticklabels(names)

plt.show()





#pdf = FPDF()

pdf.add_page()

pdf.set_xy(0, 0)

pdf.set_font('Arial', '', 14)

pdf.cell(200, 50, txt="Scaled Algorithm Comparison",ln=1, align="C")

pdf.ln()

# Then put a blue underlined link

pdf.set_text_color(0, 0, 255)

pdf.set_font('', 'U')

pdf.image('AlgoCompare.png',x = 40, y = 40, w = 150, h = 150)
#lasso algorithm tuning

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

k_values = np.array([.1,.11,.12,.13,.14,.15,.16,.09,.08,.07,.06,.05,.04])

param_grid = dict(alpha=k_values)

model = Lasso()

kfold = KFold(n_splits=10, random_state=7)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)

grid_result = grid.fit(rescaledX, y_train)





print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
#using ensembles

ensembles = []

ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',AdaBoostRegressor())])))

ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor())])))

ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',RandomForestRegressor())])))

ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',ExtraTreesRegressor())])))

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=10, random_state=7)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Compare Algorithms

fig = plt.figure()

fig.suptitle('Scaled Ensemble Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

savefig('EnsembleAlgoCompare.png')

ax.set_xticklabels(names)

plt.show()





#pdf = FPDF()

pdf.add_page()

pdf.set_xy(0, 0)

pdf.set_font('Arial', '', 14)

pdf.cell(200, 50, txt="Scaled Ensemble Algorithm Comparison",ln=1, align="C")

pdf.ln()

# Then put a blue underlined link

pdf.set_text_color(0, 0, 255)

pdf.set_font('', 'U')

pdf.image('EnsembleAlgoCompare.png',x = 40, y = 40, w = 150, h = 150)
# Tune scaled AdaboostRegressor

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

param_grid = dict(n_estimators=np.array([50,100,150,200,250,300,350,400]))

model = AdaBoostRegressor(random_state=7)

kfold = KFold(n_splits=10, random_state=7)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)

grid_result = grid.fit(rescaledX, y_train)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# prepare the model

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

model = GradientBoostingRegressor(random_state=7, n_estimators=400)

model.fit(rescaledX, y_train)
# transform the validation dataset

rescaledValidationX = scaler.transform(X_test)

predictions = model.predict(rescaledValidationX)

print(mean_squared_error(y_test, predictions))
pdf.output('Report.pdf')