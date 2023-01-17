# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from pprint import pprint

from sklearn import metrics

import shap

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve
df = pd.read_csv("../input/train.csv")

df.head()
df = df.drop(["PassengerId", "Ticket","Cabin", "Name"], axis=1)

df.head()
sexdict = {"male":0, "female":1}

df.Sex = df.Sex.replace(sexdict)



df=df.dropna()
df = df.query('Embarked == "C" |Embarked == "S"')
df=pd.get_dummies(df, ["Embarked"])

df.head()
sns.distplot(df.Age)

plt.title("Distribution of Passenger's Age")
sns.distplot(df.Fare)

plt.title("Distribution of ticket price")


sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")
pclass = df.groupby("Pclass").mean()

sns.barplot(pclass.index, pclass.Survived)

plt.title("Survival vs Pclass")
training_variables = df.columns.tolist() 

objective_variable = training_variables[0]

del training_variables[0]  
X = df[training_variables]

y = df[objective_variable]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size =0.5)



print(X_train.shape, y_train.shape)

print(X_val.shape, y_val.shape)

print(X_test.shape, y_test.shape)


roc_auc_scoreDict1 = {}

max_depth = [1, 3, 5]

learning_rate= [0.5, 0.1, 0.01]

subsample = [0.1, 0.5, 1]

for d in max_depth:

    

    roc_auc_scoreDict1[d] = {}

    for l in learning_rate:

        roc_auc_scoreDict1[d][l] = {}

        for s in subsample:

            print("%d, %f, %f " % (d, l, s))

            clf2 = XGBClassifier(max_depth= d, learning_rate=l,n_estimators=5000, subsample=s, n_jobs=-1)

            clf2.fit(X_train, y_train, verbose=100, early_stopping_rounds=50, eval_set=[(X_train,y_train),(X_val,y_val)])

            y_pred = clf2.predict_proba(X_val[:])[:, 1]

            

            roc_auc_scoreDict1[d][l][s] = roc_auc_score(y_val, y_pred)

print(roc_auc_scoreDict1)

    
from pprint import pprint

pprint(roc_auc_scoreDict1)
reform = {(level1_key, level2_key, level3_key): values

            for level1_key, level2_dict in roc_auc_scoreDict1.items()

             for level2_key, level3_dict in level2_dict.items()

              for level3_key, values      in level3_dict.items()}

reform
tipos = ['-', '-.', '--']

colors = ['r', 'b', 'g']

i = 0

j = 0

for d in max_depth:

    tipo = tipos[i]

    for l in learning_rate:

        aucs = [reform[(d, l ,s)] for s in subsample]

        plt.plot(subsample, aucs, linestyle=tipo, color = colors[j], label='d=%s, l=%s' % (d, l))

        j += 1

    i += 1

    j = 0

    

plt.xlabel('subsample')

plt.ylabel('auc')

plt.legend()
best_clf = XGBClassifier(max_depth=5, learning_rate=0.1, subsample=0.5, n_estimators=10000 ,n_jobs=-1)

best_clf.fit(X_train, y_train,verbose=100,early_stopping_rounds=100,eval_set=[(X_train,y_train),(X_val,y_val)])
ypred_test = best_clf.predict(X_test)





score = roc_auc_score(y_test, ypred_test)

score
fpr, tpr,_= roc_curve(y_test, ypred_test)

plt.figure()

plt.plot(fpr, tpr, color='red',lw=2, label='ROC curve')

plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC curve')

plt.show()
explainer = shap.TreeExplainer(best_clf)

shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)