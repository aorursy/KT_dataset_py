# Basics

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Algorithms

from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network

import xgboost as xgb

from xgboost.sklearn import XGBClassifier



# Model

from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score



import os
print(os.listdir("../input"))
df = pd.read_csv('../input/heart.csv')

df.shape
df.head()
df.describe()
fig,ax=plt.subplots(4,3,figsize=(15,15))

for i in range(12):

    plt.subplot(4,3,i+1)

    sns.distplot(df.iloc[:,i], kde=True, color='blue')
df.target.value_counts().plot.pie(autopct='%1.1f%%')
sns.countplot(data=df, x='sex', hue='target')

plt.xlabel('sex (0: female, 1: male)')
sns.violinplot(df.age)

plt.xticks(rotation=90)

plt.title("Age Rates")
sns.factorplot(data=df, x='cp', y='trestbps', hue='target')

plt.title("Trestbps vs. Chest Pain")
import matplotlib.pyplot as plt



def correlation_heatmap(df, method):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(method=method),

        cmap = colormap,

        square=True, 

        annot=True, 

        annot_kws={'fontsize':9 }

    )

    

    plt.title('Correlation Matrix', y=1.05, size=15)
correlation_heatmap(df, 'pearson')
# Remove low correlations to the target

df = df.drop(['chol', 'fbs'], axis=1)
# Get variables for a model

x = df.drop(["target"], axis=1)

y = df["target"]



#Do train data splitting

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.22, random_state=101)
MLA = [

    ensemble.AdaBoostClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    gaussian_process.GaussianProcessClassifier(),

    linear_model.LogisticRegressionCV(),

    linear_model.RidgeClassifierCV(),

    linear_model.Perceptron(),

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    neighbors.KNeighborsClassifier(),

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    xgb.XGBClassifier()

    ]
#Do some preperation for the loop

col = []

algorithms = pd.DataFrame(columns = col)

idx = 0



#Train and score algorithms

for a in MLA:

    

    a.fit(X_train, y_train)

    pred = a.predict(X_test)

    acc = accuracy_score(y_test, pred) #Other way: a.score(X_test, y_test)

    f1 = f1_score(y_test, pred)

    cv = cross_val_score(a, X_train, y_train).mean()

    

    Alg = a.__class__.__name__

    

    algorithms.loc[idx, 'Algorithm'] = Alg

    algorithms.loc[idx, 'Accuracy'] = round(acc * 100, 2)

    algorithms.loc[idx, 'F1 Score'] = round(f1 * 100, 2)

    algorithms.loc[idx, 'CV Score'] = round(cv * 100, 2)



    idx+=1
#Compare invidual models

algorithms.sort_values(by = ['CV Score'], ascending = False, inplace = True)    

algorithms.head()
#Plot them

g = sns.barplot("CV Score", "Algorithm", data = algorithms)

g.set_xlabel("CV score")

g = g.set_title("Algorithm Scores")
gnb = naive_bayes.GaussianNB()

gnb.fit(X_train, y_train)
lr = linear_model.LogisticRegressionCV()

lr.fit(X_train, y_train)
y_scores = gnb.predict_proba(X_train)

y_scores = y_scores[:,1]



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)
auroc = roc_auc_score(y_train, y_scores)

print("ROC-AUC Score:", auroc)