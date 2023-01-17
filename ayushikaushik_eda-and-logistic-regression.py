# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

%matplotlib inline

import warnings      #to avoid warnings

warnings.filterwarnings('ignore')

# importing data

data = pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')

print(f"Let's see first 5 rows of the dataset.")

data.head()
import pandas_profiling

data.profile_report()
sns.set(style="whitegrid",palette='Set2')
print("Distribution of boolean variables")

print(' “1” means “Yes”, “0” means “No”')

fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(12,8))

sns.countplot(data.TenYearCHD,ax=axes[0,0])

sns.countplot(data.male,ax=axes[0,1])

axes[0,1].set_xlabel("0 is female and 1 is male")

sns.countplot(data.currentSmoker,ax=axes[0,2])

sns.countplot(data.BPMeds,ax=axes[1,0])

sns.countplot(data.prevalentStroke,ax=axes[1,1])

sns.countplot(data.prevalentHyp,ax=axes[1,2])

plt.tight_layout()
sns.set(style="darkgrid",palette='Set1')

print("Distribution of continuous variables")

fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(12,8))

sns.distplot(data.age,ax=axes[0,0])

sns.distplot(data.BMI,ax=axes[0,1])

sns.distplot(data.glucose,ax=axes[1,0])

sns.distplot(data.cigsPerDay,ax=axes[1,1])

sns.distplot(data.sysBP,ax=axes[2,0])

sns.distplot(data.diaBP,ax=axes[2,1])

sns.distplot(data.totChol,ax=axes[3,0])

sns.distplot(data.heartRate,ax=axes[3,1])

plt.tight_layout()
#Plotting a linegraph to check the relationship between age and cigsPerDay, totChol, glucose.

graph_3 = data.groupby("age").cigsPerDay.mean()

graph_4 = data.groupby("age").totChol.mean()

graph_5 = data.groupby("age").glucose.mean()



plt.figure(figsize=(10,6))

sns.lineplot(data=graph_3, label="cigsPerDay")

sns.lineplot(data=graph_4, label="totChol")

sns.lineplot(data=graph_5, label="glucose")

plt.title("Graph showing totChol and cigsPerDay in every age group.",{'fontsize':18})

plt.xlabel("age", size=20)

plt.ylabel("count", size=20)

plt.xticks(size=12)

plt.yticks(size=12);
graph = data.groupby("age",as_index=False).currentSmoker.sum()

plt.figure(figsize=(10,6))

sns.barplot(x=graph["age"], y=graph["currentSmoker"])

plt.title("Graph showing which age group has more smokers.",{'fontsize':18});
# Let's have a visual look at missing data

msno.matrix(data);
data.groupby('diabetes').mean()['glucose']
def impute_glucose(cols):

    dia=cols[0]

    glu=cols[1]

    if pd.isnull(glu):

        if dia == 0:

            return 79

        else:

            return 170

    else:

        return glu



data['glucose'] = data[['diabetes','glucose']].apply(impute_glucose,axis=1)
#Another way to visualize missing data

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='summer');
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')

imputer.fit(data)

imputed_data = imputer.transform(data)

imputed_data = pd.DataFrame(imputed_data,columns=data.columns)
print("just to cross-check all missing data is gone!")

msno.bar(imputed_data);
#Libraries needed for model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import cross_val_score
#First split the data

X_train, X_test, y_train, y_test = train_test_split(imputed_data.drop('TenYearCHD',axis=1), 

                                                    imputed_data['TenYearCHD'], test_size=0.30, 

                                                    random_state=101)

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, predictions)))
score=cross_val_score(LogisticRegression(),imputed_data.drop('TenYearCHD',axis=1),imputed_data['TenYearCHD'],cv=10)

print(f"After k-fold cross validation score is {score.mean()}")
from sklearn.model_selection import GridSearchCV

parameters = [{'penalty':['l1','l2']}, 

              {'C':[1, 10, 100, 1000]}]

grid_search = GridSearchCV(estimator = logmodel,  

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 5,

                           verbose=0)

grid_search.fit(X_train, y_train)

# best score achieved during the GridSearchCV

print('GridSearch CV best score : {:.4f}\n'.format(grid_search.best_score_))

# print parameters that give the best results

print(f'Parameters that give the best results : {grid_search.best_params_}')
score2=cross_val_score(grid_search,imputed_data.drop('TenYearCHD',axis=1),imputed_data['TenYearCHD'],cv=10)

print(f"After k-fold cross validation score is {score2.mean()}")