import numpy as np

import pandas as pd

from collections import Counter

import matplotlib.pyplot as plt

import os

titanic = pd.read_csv('../input/titanic/train.csv')

titanic.describe()

titanic.head()
titanic.shape
titanic.columns
titanic.isnull().sum().sort_values(ascending=False)
#replacing null values with mean using simple imputer

from sklearn.impute import SimpleImputer

SI = SimpleImputer(strategy='mean')

titanic.Age = SI.fit_transform(np.array(titanic.Age).reshape(-1,1))
titanic.isnull().sum().sort_values(ascending=False)
# find out corrleation between class and fare

np.corrcoef(titanic.Fare,titanic.Pclass)
selcolumns = list(['Pclass','Sex','Age','Fare','Embarked'])
# apply anova f-test on Age and survived(continous-category)

import statsmodels.api as sm

from statsmodels.formula.api import ols

ftest = ols("Survived ~ Age", data=titanic).fit()

anova = sm.stats.anova_lm(ftest)

print(anova)
# apply anova f-test on fare and survived(continous-category)

import statsmodels.api as sm

from statsmodels.formula.api import ols

ftest = ols("Survived ~ Fare", data=titanic).fit()

anova = sm.stats.anova_lm(ftest)

print(anova)
from scipy.stats import chi2_contingency

from scipy.stats import chi2

table=pd.crosstab(titanic.Survived,titanic.Embarked)

print(table)

chisquare,p,dof,expected=chi2_contingency(table)

print('dof:',dof)

print('chisquare:',chisquare)

print('p:',p)
from scipy.stats import chi2_contingency

from scipy.stats import chi2

table=pd.crosstab(titanic.Survived,titanic.Sex)

print(table)

chisquare,p,dof,expected=chi2_contingency(table)

print('dof:',dof)

print('chisquare:',chisquare)

print('p:',p)
from scipy.stats import chi2_contingency

from scipy.stats import chi2

table=pd.crosstab(titanic.Survived,titanic.Pclass)

print(table)

chisquare,p,dof,expected=chi2_contingency(table)

print('dof:',dof)

print('chisquare:',chisquare)

print('p:',p)
#defining dataframe for independent variables taken

Independent = pd.DataFrame(titanic,columns=selcolumns)

Independent.describe()
# defining dataframe for dependent variable

Dependent = pd.DataFrame(titanic,columns=["Survived"])

Dependent.describe()
# split the data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(Independent,

                                                    Dependent,

                                                    test_size=0.20,

                                                   random_state=5)

print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)

print(Y_train[0:10])
# extract only continuous columns

#i have considered these three as corelation exists

ContColumns = list(['Age'])

X_train_cont = pd.DataFrame(X_train,columns=ContColumns)

X_test_cont = pd.DataFrame(X_test,columns=ContColumns)

print(X_train_cont.shape)

print(X_test_cont.shape)
# apply sclaing on continuous variables

from sklearn.preprocessing import StandardScaler

SS = StandardScaler(with_mean = True, with_std = True)

X_train_cont2 = SS.fit_transform(X_train_cont)

X_test_cont2 = SS.transform(X_test_cont)

print(type(X_train_cont2))

print(X_train_cont2.shape)

print(X_train_cont2[0:10])
print(SS.mean_)

print(np.mean(X_train.Age))
# convert X_train3 as dataframe

X_train_cont3 = pd.DataFrame(X_train_cont2)

X_test_cont3 = pd.DataFrame(X_test_cont2)

print(X_train_cont3.describe())
# extract only catgorical columns

CatColumns = list(['Pclass','Sex'])

X_train_cat = pd.DataFrame(X_train,columns=CatColumns)

X_test_cat = pd.DataFrame(X_test,columns=CatColumns)

print(X_train_cat.describe())
# apply labelencoder neighbourhood

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

X_train_cat.Neighborhood = LE.fit_transform(X_train_cat.Pclass)

X_test_cat.Neighborhood = LE.transform(X_test_cat.Pclass)
# apply labelencoder neighbourhood

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

X_train_cat.Neighborhood = LE.fit_transform(X_train_cat.Sex)

X_test_cat.Neighborhood = LE.transform(X_test_cat.Sex)
# apply one hot enconding on categorical

from sklearn.preprocessing import OneHotEncoder

OHE = OneHotEncoder(sparse=False,handle_unknown='error'

                    )

X_train_cat1 = OHE.fit_transform(X_train_cat)

X_test_cat1 = OHE.transform(X_test_cat)

print(X_train_cat1.shape)

print(X_test_cat1.shape)

print(X_train_cat1[0:5])
# convert array to data frame

X_train_cat2 = pd.DataFrame(X_train_cat1)

X_test_cat2 = pd.DataFrame(X_test_cat1)

print(X_train_cat2.head())
# merge scaled continuous and onehotencoded categorical data

X_train_final = pd.concat([X_train_cont3,X_train_cat2],axis=1,join='outer')

X_test_final = pd.concat([X_test_cont3,X_test_cat2],axis=1,join='outer')

print(X_train_final.shape)

print(X_test_final.shape)
X_train_final.head()
# DecisionTree Classifier

from sklearn.tree import DecisionTreeClassifier

CART = DecisionTreeClassifier(class_weight='balanced')

# build grid search parameters

##parms = {'min_samples_split': [10,8,5], 

#         'min_samples_leaf': [2,3,5],

#         'min_impurity_decrease': [0.005,0.01],

#         'max_depth':[2,3,4]} 

parms = {'max_depth':[2,3,4,5,6,7,8,9,10]} 



# grid search

from sklearn.model_selection import GridSearchCV

CV = GridSearchCV(estimator = CART,param_grid = parms, scoring = 'f1_macro',

                  cv=3,refit=True,

                  return_train_score = True,verbose=10,n_jobs=1)

CV.fit(X_train_cat1,Y_train.Survived)
CVResults = pd.DataFrame(CV.cv_results_)

CVResults.to_excel('CVResutls_CART2.xlsx')
print(CV.best_estimator_)
print(CV.best_estimator_.feature_importances_)
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(CV.best_estimator_, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
# Preidct on Test data and check the confusion matrix

predictTest = CV.predict(X_test_cat1)

print(predictTest[0:10])
predictProb = CV.predict_proba(X_test_cat1)

print(predictProb[0:10])

PProb = pd.DataFrame(predictProb)
# generate confusion matrix

# import confusion matrix and classification report

from sklearn.metrics import confusion_matrix, classification_report

print (confusion_matrix(Y_test,predictTest))

print (classification_report(Y_test,predictTest))
# generate ROC curve

from sklearn.metrics import roc_auc_score, roc_curve

# AUC for class 1

AUC = roc_auc_score(Y_test, PProb[1])

print(AUC)

# define variables for True Positive Rate and Falst Positive Rate & threshold value

TPR = dict()

FPR = dict()

THR = dict()

FPR, TPR, THR = roc_curve(Y_test, PProb[1])
print(THR)

print(TPR)

print(FPR)
# plot the FPR as X-axis and TPR as Y-axis

plt.plot(FPR, TPR)

# plot the minimum line 

plt.plot([0,1], [0,1], color='navy', linestyle = '--')

# set X and Y limits

plt.xlim([0.0, 1.05])

plt.ylim([0.0, 1.05])

# mention lables for X and Y

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()