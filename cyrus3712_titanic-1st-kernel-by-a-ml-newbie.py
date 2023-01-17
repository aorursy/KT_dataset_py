import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Import libraries for Preprocessing and Features Selection

import statsmodels.api as sm

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score



# Import libraries for Machine Leanring (Classification)

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# Import libraries forGridSearch to find out the best Parameters

from sklearn.model_selection import GridSearchCV

 

sns.set(style="darkgrid")

%matplotlib inline



import os 

print(os.listdir("../input/"))
# Import the the testing and training sets

test_set = pd.read_csv('../input/test.csv')

train_set = pd.read_csv('../input/train.csv')
train_set.head()
train_set.info()

#Total instances: 891

#Missing data in THREE attributes (AGE, Cabin, Embarked
test_set.head()
test_set.info()

#Total instances: 418

#Missing data in 3 attributes (Age, Fare, Cabin)
train_set.describe()
# Correlation of each attributes

sns.pairplot(data=train_set.dropna())
# Correlation of each attributes

corr = train_set.corr()

sns.heatmap(corr, linewidths=0.3,cmap=sns.diverging_palette(220, 20, n=7),cbar_kws={"shrink": 1},annot=True)

plt.title('Correlation of each attributes')
# Survived v.s. Fare

sns.jointplot(data=train_set,x='Age',y='Fare',kind='reg',height=7,y_jitter=.03)
# Number of Survived (Male v.s. Female)

sns.countplot(data=train_set,x='Survived',hue='Sex')

plt.title('Number of Survived (Male v.s. Female)')
# Number of Survived based on Sex and Fare

plt.figure(figsize=(5,5))

sns.violinplot(data=train_set,y='Fare',x='Survived',hue='Sex',split=True,linewidth=0.5)

plt.title('Number of Survived')
# Sex v.s. Survived

sexNsur = train_set[['Sex','Survived']].groupby('Sex',as_index=False).mean()

sexNsur['%']=sexNsur['Survived']/(np.sum(sexNsur['Survived']))

plt.figure(figsize=(5,5))

plt.title('Probability of Survival based on Sex')

plt.pie(sexNsur['%'],labels=sexNsur['Sex'],startangle=140,autopct='%1.1f%%')
def fare_class (fare):

    if fare < np.percentile(train_set['Fare'], 25):

        return '0 - 25%'

    elif fare < np.percentile(train_set['Fare'], 50):

        return '26 - 50%'

    elif fare < np.percentile(train_set['Fare'], 75):

        return '51 - 75%'

    else:

        return '76 - 100%'



train_set['FareClass']=train_set['Fare'].apply(fare_class)



fareNsur=train_set[['FareClass','Survived']].groupby('FareClass',as_index=False).mean()

fareNsur['%']=fareNsur['Survived']/(np.sum(fareNsur['Survived']))

plt.figure(figsize=(5,5))

plt.title('Probability of Survival based on Fare Class')

plt.pie(fareNsur['%'],labels=fareNsur['FareClass'],startangle=140,autopct='%1.1f%%')
def agegroup (age):

    if age < 2:

        return 'Infant'

    elif age >= 2 and age < 18:

        return 'Youth'

    elif age >= 18 and age <= 35:

        return 'Young Adult'

    elif age > 36 and age <= 55:

        return 'Adult Adult'

    else:

        return 'Senior'

train_set['AgeGroup']=train_set['Age'].apply(agegroup)



ageNsur=train_set[['AgeGroup','Survived']].groupby('AgeGroup',as_index=False).mean()

ageNsur['%']=ageNsur['Survived']/(np.sum(ageNsur['Survived']))

plt.figure(figsize=(5,5))

plt.title('Probability of Survival based on Age Group')

plt.pie(ageNsur['%'],labels=ageNsur['AgeGroup'],startangle=140,autopct='%1.1f%%')
pclassNsur=train_set[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()

pclassNsur['%']=pclassNsur['Survived']/(np.sum(pclassNsur['Survived']))

plt.figure(figsize=(5,5))

plt.title('Probability of Survival based on Number of Pclass')

plt.pie(pclassNsur['%'],labels=pclassNsur['Pclass'],startangle=140,autopct='%1.1f%%')
# Drop the unqiue identifiers (PassengerId / Name)

# More than a half of the data in cabin is missing, it is pointless to fill them all

train_set.drop(['PassengerId','Name','Ticket','Cabin','AgeGroup','FareClass'],axis=1,inplace=True)

test_set.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
# Handle the missing data in AGE

age_imputer = Imputer(strategy='mean',axis=0)

train_set.iloc[:,3:4] = age_imputer.fit_transform(train_set.iloc[:,3:4]) 

test_set.iloc[:,2:3] = age_imputer.transform(test_set.iloc[:,2:3])
# Handle the missing FARE

fare_imputer = Imputer(strategy='most_frequent',axis=0)

test_set.iloc[:,5:6] = fare_imputer.fit_transform(test_set.iloc[:,5:6])
# Label Encoder (Sex)

from sklearn.preprocessing import LabelEncoder

sex_labelencoder = LabelEncoder()

train_set['Sex'] = sex_labelencoder.fit_transform(train_set['Sex'])

test_set['Sex'] = sex_labelencoder.transform(test_set['Sex'])
# Dummy Table (Pclass)

pclass_train_dummy = pd.get_dummies(train_set['Pclass'],prefix='Pclass_',drop_first=True,dtype=int)

pclass_test_dummy = pd.get_dummies(test_set['Pclass'],prefix='Pclass_',drop_first=True,dtype=int)



train_set = train_set.join(pclass_train_dummy)

test_set = test_set.join(pclass_test_dummy)
train_set['Embarked'].value_counts()
# Handle the missing data in EMBARKED by filling in the most-frequent value 

train_set.iloc[61,7] = 'S'

train_set.iloc[829,7] = 'S'



# Dummy Table (Embarked)

embarked_train_dummy = pd.get_dummies(train_set['Embarked'],prefix='embarked_',drop_first=True,dtype=int)

embarked_test_dummy = pd.get_dummies(test_set['Embarked'],prefix='embarked_',drop_first=True,dtype=int)



train_set = train_set.join(embarked_train_dummy)

test_set = test_set.join(embarked_test_dummy)
train_set.info()
test_set.info()
# Drop the duplicated attributes which have been handled via Label Encoder and Dummy Table (One Hot Encoder)

train_set.drop(['Pclass','Embarked'],axis=1,inplace=True)

test_set.drop(['Pclass','Embarked'],axis=1,inplace=True)
X_train = train_set.iloc[:,1:].values

y_train = train_set.iloc[:,0].values

X_test = test_set.iloc[:,:].values
# Feature Scaling (Standaradization)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# BACKWARD ELIMINATION 

def backwardElimination(x,sl):

    num = len(x[0])

    temp = np.zeros((891,9)).astype(int)

    for i in range(0,num):

        regressor_OLS = sm.OLS(y_train,x).fit()

        maxVar = max(regressor_OLS.pvalues).astype(float)

        adjR_before = regressor_OLS.rsquared_adj.astype(float)

        if maxVar > sl:

            for j in range(0,num - i):

                if (regressor_OLS.pvalues[j].astype(float)==maxVar):

                    temp[:,j]=x[:,j]

                    x = np.delete(x,j,1)

                    tmp_regressor = sm.OLS(y_train,x).fit()

                    adjR_after = tmp_regressor.rsquared_adj.astype(float)

                    print(regressor_OLS.summary())

                    if (adjR_before >= adjR_after):

                        x_rollback = np.hstack((x,temp[:,[0,j]]))

                        x_rollback = np.delete(x_rollback,j,1)

                        return x_rollback

                    else:

                        continue

    regressor_OLS.summary()

    return x
# Significant Value = 0.05

SL = 0.05

X_opt = X_train[:,[0,1,2,3,4,5,6,7,8]]

X_modeled_train = backwardElimination(X_opt,SL)
# Cross-Validation

def cross_val(classifier):

    result = cross_val_score(classifier, X_modeled_train, y_train, cv=10)

    return result
# Logistic Regression

logistic_classifier = LogisticRegression(random_state=0)

logistic_classifier.fit(X_modeled_train,y_train)

logistic_cross_val = cross_val(logistic_classifier)

logisitic_mean = logistic_cross_val.mean()
# K-Nearest Neighbor (KNN)

knn_classifier = KNeighborsClassifier(n_neighbors=10, metric='minkowski',p=2)

knn_classifier.fit(X_modeled_train,y_train)

knn_cross_val = cross_val(knn_classifier)

knn_mean = knn_cross_val.mean()
# Support Vector Classification (SVC)

svc_classifier = SVC(kernel='rbf',gamma=5,C=1)

svc_classifier.fit(X_modeled_train,y_train)

svc_cross_val = cross_val(svc_classifier)

svc_mean = svc_cross_val.mean()
# Naive Bayes

nb_classifier = GaussianNB()

nb_classifier.fit(X_modeled_train,y_train)

nb_cross_val = cross_val(nb_classifier)

nb_mean = nb_cross_val.mean()
# Decision Tree Clasification

dt_classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)

dt_classifier.fit(X_modeled_train,y_train)

dt_cross_val = cross_val(dt_classifier)

dt_mean = dt_cross_val.mean()
# Random Forest Classification

randomforest_classifier = RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)

randomforest_classifier.fit(X_modeled_train,y_train)

randomforest_cross_val = cross_val(randomforest_classifier)

randomforest_mean = randomforest_cross_val.mean()
# XGBoost

xgb_classifier = XGBClassifier()

xgb_classifier.fit(X_modeled_train,y_train)

xgb_cross_val = cross_val(xgb_classifier)

xgb_mean = xgb_cross_val.mean()
# Model Visulization and Comparison

plt.figure(figsize=(15,15))

plt.plot([1]*10,logistic_cross_val, '.')

plt.plot([2]*10,knn_cross_val, '.')

plt.plot([3]*10,svc_cross_val, '.')

plt.plot([4]*10,nb_cross_val, '.')

plt.plot([5]*10,dt_cross_val, '.')

plt.plot([6]*10,randomforest_cross_val, '.')

plt.plot([7]*10,xgb_cross_val, '.')

plt.boxplot([logistic_cross_val,knn_cross_val,svc_cross_val,nb_cross_val,dt_cross_val,randomforest_cross_val,xgb_cross_val], labels=('Logistic Classification','KNN','SVC','Naive Bayes','Decision Tree','Random Forest','XGBoost'))

plt.ylabel('Accuracy')

plt.xlabel('Models Comparison')

plt.ylim(0.6,0.9)

plt.title('Cross-Validation Results', fontsize=20)

plt.show()
# GridSearchCV

parameter = [{'C':[0.1,1, 10, 100, 1000], 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}]
# GridSearchCV (XGBoost)

xgb_grid_search = GridSearchCV(estimator = xgb_classifier,param_grid=parameter,scoring='accuracy',n_jobs=-1)

xgb_grid_search = xgb_grid_search.fit(X_train,y_train)

xgb_best_accuracy = xgb_grid_search.best_score_

xgb_best_parameters = xgb_grid_search.best_params_
# GridSearchCV (SVC)

svc_grid_search = GridSearchCV(estimator = svc_classifier,param_grid=parameter,scoring='accuracy',n_jobs=-1)

svc_grid_search = svc_grid_search.fit(X_train,y_train)

svc_best_accuracy = svc_grid_search.best_score_

svc_best_parameters = svc_grid_search.best_params_
index = ['XGBoost','SVC']

result = {'Best Accuracy':[xgb_best_accuracy,svc_best_accuracy], 

          'Best Parameters':[xgb_best_parameters,svc_best_parameters]}

result_table = pd.DataFrame(data=result,index=index)
result_table