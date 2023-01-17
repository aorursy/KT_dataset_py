import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

%matplotlib inline

plt.style.use('seaborn-whitegrid')

sns.set_context("poster")



dataset = pd.read_csv("../input/adult.csv")



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, KFold

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



from xgboost import XGBClassifier



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



print("Setup Complete")
dataset.head(5)
dataset.isnull().sum()

#checking for missing values
#Object in the case = Text

#Int64 = Numbers

dataset.dtypes
dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1})

dataset["workclass"] = dataset["workclass"].replace(["?"],'Unknown')

fig, ax = plt.subplots(figsize=(25,7))

sns.set_context("poster")

current_palette = sns.diverging_palette(255, 133, l=60, n=7, center="dark")



fig = sns.barplot(x='workclass',y='income',data=dataset,palette=current_palette)



fig.set_ylabel("Income >50K Probability [%]")

fig.set_xlabel("Workclass")
fig, axe = plt.subplots(2,1,figsize=(27,12))

current_palette =sns.cubehelix_palette(8)

plt.style.use('seaborn-whitegrid')

fig = sns.barplot(x='marital.status',y='income',data=dataset,ax=axe[0],order =['Never-married','Separated','Widowed','Divorced','Married-spouse-absent','Married-AF-spouse','Married-civ-spouse'],palette=current_palette)



fig.set_ylabel("Income >50K Probability [%]")

fig.set_xlabel("Marital Status")





datasetCopy= dataset.copy()

datasetCopy["marital.status"] = datasetCopy["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], '1')

datasetCopy["marital.status"] = datasetCopy["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], '0')

datasetCopy["marital.status"] = datasetCopy["marital.status"].astype(int)



sns.set_context("poster")

fig = sns.barplot(x='marital.status',y='income',data=datasetCopy,ax=axe[1], order=[1,0],palette=['indigo','silver'])



fig.set(xticklabels =['Married','Single'])

fig.set_ylabel("Income >50K Probability [%]")

fig.set_xlabel("Marital Status")
fig, ax = plt.subplots(figsize=(25,7))

sns.set_context("poster")

current_palette = sns.color_palette("Blues")



fig = sns.barplot(x='relationship',y='income',data=dataset, order=['Own-child','Other-relative','Unmarried','Not-in-family','Husband','Wife'], palette=current_palette)



fig.set_ylabel("Income >50K Probability [%]")

fig.set_xlabel("Relationship")
numeric_columns = ['marital.status','education.num','age','hours.per.week','capital.gain','capital.loss','income']



fig, axe = plt.subplots(figsize=(15,15))

sns.set_context("poster")

sns.set(font_scale=2)

map1 = sns.color_palette("RdBu_r", 7)

sns.heatmap(datasetCopy[numeric_columns].corr(),annot=True, fmt='.2f',linewidths=2,cmap = map1)
fig, axe = plt.subplots(figsize=(25,8))

sns.set_context("poster")



g=sns.violinplot(x='age',y='income',hue='sex',hue_order=["Male","Female"],data=dataset,orient="h",palette=["cornflowerblue","indianred",])



g.set_ylabel("Income")

g.set_xlabel("Age")

g.set(yticklabels =['<=50k','>50k'])

setThis = g.legend(loc='center right')

plt.xlim(0,100)

axe.xaxis.set_major_locator(ticker.MultipleLocator(5))
dataset = pd.read_csv("../input/adult.csv")

fig, axe = plt.subplots(figsize=(25,8))

sns.set_context("poster")



g=sns.violinplot(x='age',y='marital.status',hue='income',data=datasetCopy,orient="h",palette=["c",'seagreen'])



g.set_ylabel("Marital Status")

g.set_xlabel("Age")

setThis = g.legend(loc='center right')

setThis.get_texts()[0].set_text("<=50k")

setThis.get_texts()[1].set_text(">50k")

g.set(yticklabels =['Single','Married'])

plt.xlim(0,100)

axe.xaxis.set_major_locator(ticker.MultipleLocator(5))
fig, axe = plt.subplots(figsize=(31,9))

sns.set_context("poster")



g=sns.violinplot(x="education.num",y='age',hue='income',data=dataset,palette=["skyblue",'mediumseagreen'],ax = axe)

sns.lineplot(x="education.num",y='age',hue='income',data=dataset,palette=["c",'green'],ax=axe)





plt.ylim(0,100)

g.set_xlabel("Years of Continuous Education")

g.set_ylabel("Age")

setThis = g.legend(bbox_to_anchor=(.906, 1),loc=2)

setThis.get_texts()[0].set_text("Income")

axe.yaxis.set_major_locator(ticker.MultipleLocator(10))
dataset = pd.read_csv("../input/adult.csv")

sns.set_context("poster")

g=sns.lmplot(x="age",y='hours.per.week',hue='income',col="income",data=dataset,markers= ['x','o'],palette=["c",'seagreen'],height=15,line_kws={'color': 'darkslategray'})



g= (g.set_axis_labels("Age","Hours Per Week"))
#Before we can begin to model are dataset, we first have to drop any categorical data and convert the one's we want to keep into binary:: Yes (1) or No (0)

dataset["marital.status"] = dataset["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')

dataset["marital.status"] = dataset["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')

dataset["marital.status"] = dataset["marital.status"].map({"Married":0, "Single":1})

dataset["marital.status"] = dataset["marital.status"]

dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1})

dataset.drop(labels=["sex","workclass","education","occupation","relationship","race","native.country"], axis = 1, inplace = True)



dataset.head(5)
numeric_columns = ['marital.status','age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']

X=dataset[numeric_columns]

Y=dataset.income

train_X, val_X, train_y, val_y = train_test_split(X,Y,test_size=0.21,random_state=0)



outcome = []

Modelnames = []

models = []

models.append(('Random Forest Classifier', RandomForestClassifier(n_estimators=50, max_features=4)))

models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))

models.append(('Decision Tree Classifier', DecisionTreeClassifier()))

models.append(('Logistic Regression', LogisticRegression(solver='lbfgs')))



kfoldCV = StratifiedKFold(n_splits=5, random_state=0)

xgb_model = XGBClassifier(n_estimators=250)

results = cross_val_score(xgb_model, train_X, train_y, cv=kfoldCV)

print("XGBClassifier: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

outcome.append(results)

Modelnames.append("XGBClassifier")



for name, model in models:

    kfoldCV = KFold(n_splits=5, random_state=0)

    cv_r = cross_val_score(model, train_X, train_y, cv=kfoldCV, scoring='accuracy')

    outcome.append(cv_r)

    Modelnames.append(name)

    print("%s: %.2f%% (%.2f%%)" % (name, cv_r.mean()*100, cv_r.std()*100))
fig, axe = plt.subplots(figsize=(27,10))

data1 ={'Names': Modelnames,'Results': outcome}

fig.suptitle('Model Accuracy Comparison')

current_palette = sns.color_palette("RdBu_r", 5)

sns.set_context("poster")

sns.boxplot(x='Names',y='Results',data=data1,palette = current_palette)
eval_set=[(val_X,val_y)]

for i in [50,100,200,400,800,1600]:

    xgb_model = XGBClassifier(n_estimators=i,learning_rate=0.05).fit(train_X,train_y,early_stopping_rounds=i-i*.75,eval_set=eval_set, verbose=False)

    results = xgb_model.predict(val_X)

    predictions = [round(value) for value in results]

    accuracy = accuracy_score(val_y, predictions)

    print("Accuracy: %.2f%% --- %.2f N_estimators" % (accuracy * 100.0,i))
xgb_model = XGBClassifier(n_estimators=1600,learning_rate=0.05).fit(train_X,train_y,early_stopping_rounds=i-i*.75,eval_set=eval_set, verbose=False)

results = xgb_model.predict(val_X)

print("Accuracy: %s%%" % (100*accuracy_score(val_y, results)))

print(classification_report(val_y, results))