''' The Notebook has been created for beginners(like me too) to learn how to EDA,

how to find features using feature selection techniques and predicting the same using Random forest Technique'''

import pandas as pd

import numpy as np

import seaborn as sb

pd.pandas.set_option('display.max_columns',None)

from sklearn.metrics import roc_auc_score,roc_curve

import matplotlib.pyplot as plt
data = pd.read_csv('https://raw.githubusercontent.com/liuy14/Kidney_Disease_Detection/master/Kidney%20Disease%20Study/datasets/chronic_kidney_disease.csv',\

                        names=['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot','hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class'],\

                       na_values=['?',np.nan,' '])

data.head()
#finding variance of each column can be done after filling null value

np.var(data)>0.01
data.info()
X=data.drop('class',axis=1)

y=data['class']
# finding numerical feature with datatypes

numerical_feature = [feat for feat in X.columns if X[feat].dtypes!='O']

numerical_feature
numerical_feature_nan = [feat for feat in numerical_feature if X[feat].isnull().sum()>0]

numerical_feature_nan
#Filling the null value in numerical colum using median of the column

for i in numerical_feature_nan:

    median_value = X[i].median()

    X[i] = X[i].fillna(median_value)

    
categorical_feature = [feat for feat in X.columns if X[feat].dtypes=='O']

X[categorical_feature]
categorical_feature_nan = [feat for feat in categorical_feature if X[feat].isnull().sum()>0]

categorical_feature_nan
#Finding distinct value in field including null values

X[categorical_feature_nan].nunique()
for i in categorical_feature_nan:

    print(i,X[i].unique())

        
#dm has a space before yes,to remove

X['dm'] = X['dm'].str.strip()
#Function to fill null categorical feature with most repeating value of the feature itself

def fil_cat(listt):

    for i in listt:

        highcount = (pd.Series(X[i]).value_counts().index)[0]

        X[i] = X[i].fillna(highcount)
fil_cat(categorical_feature_nan)
X.head()
#Onehotencoding without sklearn

df = pd.get_dummies(X[categorical_feature],drop_first=True)

df
# concating the dummies to original data

X = pd.concat([X,df],axis=1)
X.head()
X.drop(categorical_feature,axis=1,inplace=True)
# Replacing the ckd value from y(dependant feature) with 1 N O

y = pd.DataFrame(y,columns=['class'])

y['class']=y['class'].apply(lambda x:1 if x=='ckd' else 0)
y.head()
sb.countplot(y['class'])
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import classification_report,confusion_matrix,recall_score,precision_score
RT= RandomForestClassifier(n_estimators=200

                          )
#y is a series and python throws error,,hence using ravel to make it array,scoring is recall

print(np.mean(cross_val_score(RT,X,y.values.ravel(),scoring='recall',cv=5)))
# Startify is to balance variation in y field(It migth not have same no of 1 and 0's ryt..so using stratify)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,

                                                    random_state=21, stratify=y)
pd.Series(y_test['class']).value_counts()
RT.fit(X_train,y_train)
pred = RT.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(pred,y_test))
# No matter what diesease you are working on Recall score is the most important than accuracy

print(recall_score(pred,y_test))
print(precision_score(pred,y_test))
from sklearn.model_selection import RandomizedSearchCV
parameter = dict(n_estimators=[200,300,400,500],max_depth=[2,3,4,5,6,None],criterion=['gini','entropy'])
random = RandomizedSearchCV(estimator=RandomForestClassifier(),param_distributions=parameter,n_iter=5)
random.fit(X,y.values.ravel())
# Gives the best parameter

random.best_params_
#building a model based on the best params

RTactual = RandomForestClassifier(n_estimators= 300, max_depth= 3, criterion='gini')
RTactual.fit(X_train,y_train)
# predict_proba is used to probability of value getting 1 value

predicted = RTactual.predict(X_test)

predicted_prob = RTactual.predict_proba(X_test)[:,1]

predicted_prob
print(classification_report(y_test,predicted))
print(confusion_matrix(y_test,predicted))
print(recall_score(y_test,predicted))
print(precision_score(y_test,predicted))
# fpr and tpr are found here using probability

fpr,tpr,thre=roc_curve(y_test,predicted_prob)
print(fpr,tpr)
#plotting the fpr and tpr as a graph and more the are under the graph better it is

plt.figure(figsize=[10,10])

sb.lineplot(x=fpr,y=tpr,lw=3)
# roc score can be found using this method

print(np.round(roc_auc_score(y_test,predicted_prob)))
# feature selection is done below and found respective recall score 

from sklearn.feature_selection import VarianceThreshold
VTfeatures = X.columns[(VarianceThreshold(threshold=0.01).fit(X,y)).get_support()]

VTfeatures
X_train_VT = X_train[VTfeatures]

X_test_VT = X_test[VTfeatures]
X_train_VT.shape,X_test_VT.shape
RTactual.fit(X_train_VT,y_train)
print(recall_score(y_test,RTactual.predict(X_test_VT)))
confusion_matrix(y_test,RTactual.predict(X_test_VT))
fpr,tpr,thres = roc_curve(y_test,RTactual.predict_proba(X_test_VT)[:,1])
plt.plot(fpr,tpr)
coeff_matrix = X_train.corr()
sb.heatmap(coeff_matrix,cmap='rainbow')
corr_col = set()

for i in range(len(coeff_matrix.columns)):

    for j in range(i):

        if np.abs(coeff_matrix.iloc[i,j])>0.75:

            column = coeff_matrix.columns[i]

            corr_col.add(column)
X_train_coeff = X_train.drop(corr_col,axis=1)

X_test_coeff = X_test.drop(corr_col,axis=1)

len(X_test_coeff.columns)
RTactual.fit(X_train_coeff,y_train)
confusion_matrix(y_test,RTactual.predict(X_test_coeff))
Recall_coeff = recall_score(y_test,RTactual.predict(X_test_coeff))
print(precision_score(y_test,RTactual.predict(X_test_coeff)))
fpr,tpr,thre = roc_curve(y_test,RTactual.predict_proba(X_test_coeff)[:,1])
plt.plot(fpr,tpr)
from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import chi2

from sklearn.linear_model import Lasso
feature = SelectFromModel(Lasso(alpha=0.0005)).fit(X_train,y_train).get_support()

X_train_lasso =X_train[ X_train.columns[feature]]

X_test_lasso = X_test[X_test.columns[feature]]

feature
RTactual.fit(X_train_lasso,y_train)
confusion_matrix(y_test,RTactual.predict(X_test_lasso))
confusion_matrix(y_test,RTactual.predict(X_test_lasso))
Recall_lasso = recall_score(y_test,RTactual.predict(X_test_lasso))

Recall_lasso
from sklearn.feature_selection import SelectKBest
recall = []

precision = []

for i in range(len(X_train.columns)):

    if i!=0:

        feature = SelectKBest(chi2,k=i).fit(X_train,y_train).get_support()

        X_train_K = X_train[X_train.columns[feature]]

        X_test_K =  X_test[X_test.columns[feature]]

        RTactual.fit(X_train_K,y_train.values.ravel())

        Recall = recall_score(y_test,RTactual.predict(X_test_K))

        Precision = precision_score(y_test,RTactual.predict(X_test_K))

        recall.append(Recall)

        precision.append(Precision)

        print(f"{i} features give us a recall score of: {Recall} % and a precision score of {Precision} %")
plt.plot(np.arange(1,24),recall,label="Recall")

plt.plot(np.arange(1,24),precision,label="Precision")

plt.show()

from sklearn.feature_selection import f_classif
feature = f_classif(X_train,y_train)

P = feature[1]

X_train_ANova = X_train[X_train.columns[P<0.05]]

X_test_ANova = X_test[X_train.columns[P<0.05]]

RTactual.fit(X_train_ANova,y_train.values.ravel())
print(recall_score(y_test,RTactual.predict(X_test_ANova)))
fpr,tpr,thre = roc_curve(y_test,RTactual.predict_proba(X_test_ANova)[:,1])
plt.plot(fpr,tpr,marker="*")

plt.show()
print(roc_auc_score(y_test,RTactual.predict_proba(X_test_ANova)[:,1]))
#The same can be done by XGboost.Hope you all like...kindly let me know if u have any doubts