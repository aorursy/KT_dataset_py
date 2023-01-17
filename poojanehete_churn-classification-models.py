# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df= pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
df.head()
df.shape
pd.options.display.max_rows=None

pd.options.display.max_columns=None
import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats

df.isna().sum()
df.drop(['RowNumber', 'CustomerId','Surname'], axis=1, inplace=True )
df.describe()
numerical_features= df[['CreditScore','Age', 'Balance', 'EstimatedSalary']]

for i in numerical_features.columns:

    sns.distplot(df[i])

    plt.show()
sns.distplot(df['CreditScore'], hist=False)

plt.show()

scipy.stats.norm.interval(0.50, df['CreditScore'].mean(), df['CreditScore'].std() )
sns.countplot(df['Geography'])

plt.show()
df['HasCrCard'].value_counts().plot(kind='bar')

plt.title('Credit Card')

plt.show()
sns.countplot(df['NumOfProducts'])

plt.title('Number of products')

plt.show()
df['IsActiveMember'].value_counts().plot(kind='bar')

plt.title('Active member')

plt.show()
fig, axes= plt.subplots(1,2, figsize=(10,5))

sns.countplot(df['Gender'], ax=axes[0])

props = (df.groupby("Gender")['Exited'].value_counts(normalize=True).unstack())*100

props.plot(kind='bar', stacked='True', ax=axes[1])

plt.show()
sns.boxplot(x= df['Exited'], y=df['Age'])

plt.show()
fig, axes= plt.subplots(1,2, figsize=(15,6))

sns.countplot(df['Geography'], hue=df['Exited'], ax=axes[0])

props = (df.groupby("Geography")['Exited'].value_counts(normalize=True)*100).unstack()

props.plot(kind='bar', stacked='True', ax=axes[1])

plt.show()
props = (df.groupby("IsActiveMember")['Exited'].value_counts(normalize=True).unstack())*100

props.plot(kind='bar', stacked='True')

plt.show()
props = (df.groupby("Tenure")['Exited'].value_counts(normalize=True).unstack())*100

props.plot(kind='bar', stacked='True')

plt.show()
props = (df.groupby("HasCrCard")['Exited'].value_counts(normalize=True).unstack())*100

props.plot(kind='bar', stacked='True')

plt.show()
fig=plt.subplots(figsize=(15,15))

for i, j in enumerate(numerical_features):

    plt.subplot(8, 2, i+1)

    plt.subplots_adjust(hspace = 1.0)

    sns.boxplot(x=j,data = df)

    plt.xticks(rotation=90)

    #plt.title("Telecom")

    

plt.show()
# Number of outliers in each feature





outliers=[]

for i in numerical_features.columns:

    q1 = numerical_features[i].describe()['25%']

    q3 = numerical_features[i].describe()['75%']

    iqr = q3-q1

    data = numerical_features[(numerical_features[i] > (q1 - 1.5*iqr)) &

            (numerical_features[i] < (q3 + 1.5*iqr))]

    outliers.append(numerical_features.shape[0]-data.shape[0])

outlier= pd.DataFrame()

outlier['features']=numerical_features.columns

outlier['number']=outliers



outlier
#removing outliers:



def outliers(df,i):

    q1=df[i].quantile(0.25)

    q3=df[i].quantile(0.75)

    iqr=q3-q1

    ul=q3+(1.5*iqr)

    ll=q1-(1.5*iqr)

    clean_data= df.loc[(df[i]<ul) & (df[i]>ll)]

    return clean_data



clean_df=outliers(df, 'CreditScore')

clean_df=outliers(df, 'Age')



clean_df.shape
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
# Creating dummies for categorical features



categorical_features= clean_df.select_dtypes(include='O')



clean_df= pd.get_dummies(clean_df, prefix=categorical_features.columns , drop_first=True)
clean_df.head()
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))

scaler = sc.fit_transform(clean_df[numerical_features.columns])

df_scaled = pd.DataFrame(scaler, columns=numerical_features.columns)



clean_df.drop(['CreditScore', 'Age', 'Balance', 'EstimatedSalary'], axis=1, inplace=True)



clean_df= pd.concat([df_scaled.reset_index(drop=True), clean_df.reset_index(drop= True)], axis=1)
clean_df.head()
# Splitting the data in train and test sets

X = clean_df.drop('Exited', axis=1)

Y = clean_df['Exited']

x_train,x_test,y_train,y_test = train_test_split(X, Y ,test_size = 0.3,random_state = 25)
(df['Exited'].value_counts(normalize= True)*100).plot(kind='bar')

plt.show()
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, f1_score, precision_score,recall_score,confusion_matrix

from sklearn import metrics

from sklearn import model_selection

from imblearn.over_sampling import SMOTE

smt = SMOTE()

X_smo,Y_smo = smt.fit_sample(x_train ,y_train)



np.bincount(Y_smo)

#hyperparameter



# Random Forest

rfc= RandomForestClassifier(random_state=0)

hyper={'n_estimators':range(100,700,100), 'criterion':['gini','entropy']}

rfc_grid=GridSearchCV(estimator= rfc, param_grid=hyper, verbose=True)

rfc_grid.fit(X_smo, Y_smo)

rfc_grid.best_params_





#Decision Tree

dt= DecisionTreeClassifier(random_state=0)

dt_params= {'max_depth': np.arange(1,50), 'min_samples_leaf': np.arange(2,15)}

GS_dt= GridSearchCV(dt,dt_params, cv=5)



GS_dt.fit(X_smo, Y_smo)



GS_dt.best_params_



#gradient_boost

gb=GradientBoostingClassifier( random_state=0)

gb_params = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}



gsearch2 = GridSearchCV(gb,gb_params, scoring='roc_auc',iid=False, cv=5)



gsearch2.fit(X_smo, Y_smo)

gsearch2.best_params_



#knn

param_grid = { 'n_neighbors': np.arange(1,25), "metric" : [ "minkowski" , "manhattan" , "jaccard"] }

knn = KNeighborsClassifier(n_neighbors=7)

knn_grid = GridSearchCV ( knn , param_grid, cv = 5 , return_train_score = True )

knn_grid.fit(X_smo, Y_smo)

knn_grid.best_params_

# Models on smote data

seed= 0



RF= RandomForestClassifier(**rfc_grid.best_params_, random_state=seed)

dt= DecisionTreeClassifier(**GS_dt.best_params_, random_state=seed)

lr= LogisticRegression(max_iter=15000, random_state=seed)

bg= BaggingClassifier(random_state=seed)

adb= AdaBoostClassifier(random_state=seed)

gb= GradientBoostingClassifier(**gsearch2.best_params_, random_state=seed)

knn = KNeighborsClassifier(**knn_grid.best_params_)



models=[lr,RF, dt,knn, adb, bg, gb]



def score_ensemble_model(xtrain,ytrain,xtest,ytest):

    mod_columns=[]

    mod=pd.DataFrame(columns=mod_columns)

    i=0

    #read model one by one

    for model in models:

        model.fit(xtrain,ytrain)

        y_pred=model.predict(xtest)

        

        #compute metrics

        train_accuracy=model.score(xtrain,ytrain)

        test_accuracy=model.score(xtest,ytest)

        

        p_score=metrics.precision_score(ytest,y_pred)

        r_score=metrics.recall_score(ytest,y_pred)

        f1_score=metrics.f1_score(ytest,y_pred)

        # calculate the fpr and tpr for all thresholds of the classification

        probs = model.predict_proba(xtest)

        preds = probs[:,1]

        fp, tp, th = metrics.roc_curve(ytest, preds)

        

        #insert in dataframe

        mod.loc[i,"Model_Name"]=model.__class__.__name__

        mod.loc[i,"Precision"]=round(p_score,2)

        mod.loc[i,"Recall"]=round(r_score,2)

        mod.loc[i,"Train_Accuracy"]=round(train_accuracy,2)

        mod.loc[i,"Test_Accuracy"]=round(test_accuracy,2)

        mod.loc[i,"F1_Score"]=round(f1_score,2)

        mod.loc[i,'AUC'] = metrics.auc(fp, tp)

        

        i+=1

    

    #sort values by accuracy

    mod.sort_values(by=['AUC'],ascending=False,inplace=True)

    return(mod)



report=score_ensemble_model(X_smo, Y_smo, x_test, y_test)

report
# Models on origial data



seed= 0

lr= LogisticRegression(max_iter=15000, random_state=seed)

RF= RandomForestClassifier(random_state=seed)

dt= DecisionTreeClassifier(random_state=seed)

bg= BaggingClassifier(random_state=seed)

adb= AdaBoostClassifier(random_state=seed)

gb= GradientBoostingClassifier(random_state=seed)

knn = KNeighborsClassifier()



models=[lr,RF, dt,knn, adb, bg, gb]





def score_ensemble_model(xtrain,ytrain,xtest,ytest):

    mod_columns=[]

    mod=pd.DataFrame(columns=mod_columns)

    i=0

    #read model one by one

    for model in models:

        model.fit(xtrain,ytrain)

        y_pred=model.predict(xtest)

        

        #compute metrics

        train_accuracy=model.score(xtrain,ytrain)

        test_accuracy=model.score(xtest,ytest)

        

        p_score=metrics.precision_score(ytest,y_pred)

        r_score=metrics.recall_score(ytest,y_pred)

        f1_score=metrics.f1_score(ytest,y_pred)

        # calculate the fpr and tpr for all thresholds of the classification

        probs = model.predict_proba(xtest)

        preds = probs[:,1]

        fp, tp, th = metrics.roc_curve(ytest, preds)

        

        #insert in dataframe

        mod.loc[i,"Model_Name"]=model.__class__.__name__

        mod.loc[i,"Precision"]=round(p_score,2)

        mod.loc[i,"Recall"]=round(r_score,2)

        mod.loc[i,"Train_Accuracy"]=round(train_accuracy,2)

        mod.loc[i,"Test_Accuracy"]=round(test_accuracy,2)

        mod.loc[i,"F1_Score"]=round(f1_score,2)

        mod.loc[i,'AUC'] = metrics.auc(fp, tp)

        

        i+=1

    

    #sort values by accuracy

    mod.sort_values(by=['AUC'],ascending=False,inplace=True)

    return(mod)



report=score_ensemble_model(x_train, y_train, x_test, y_test)

report
# evaluate each model in turn

x= clean_df.drop('Exited', axis=1)

y= clean_df['Exited']

results = []

names = []



models = []

models.append(('MVLR', lr))

models.append(('decision tree', dt))

models.append(('RF', RF))

models.append(('Adaboost', adb))

models.append(('bagging', bg))

models.append(('gradient', gb))

models.append(('knn', knn))



print('name ',' bias ',' variance')

for name, model in models:

    kfold = model_selection.KFold(shuffle=True,n_splits=5,random_state=0)

    cv_results = model_selection.cross_val_score(model,x,y ,cv=kfold, scoring='roc_auc')

    results.append(cv_results)

    names.append(name)

    print("%s: %f (%f)" % (name, 1- np.mean(cv_results),np.var(cv_results,ddof=1)))



# bias calculation

bias= []

for i in list(results):

    bias.append(1- i)

    



# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(bias)

ax.set_xticklabels(names)

plt.show()
#Confusion matrix and classification report for Random Forest



RF= RandomForestClassifier(random_state=0)

RF.fit(x_train, y_train)



y_pred=  RF.predict(x_test)



print(confusion_matrix(y_test, y_pred))

      

print(classification_report(y_test, y_pred))
#Confusion matrix and classification report for Ada boost



adb= AdaBoostClassifier(random_state=0)

adb.fit(x_train, y_train)



y_pred=  adb.predict(x_test)



print(confusion_matrix(y_test, y_pred))

      

print(classification_report(y_test, y_pred))
#Confusion matrix and classification report for gradient boost



gb.fit(x_train, y_train)



y_pred=  gb.predict(x_test)



print(confusion_matrix(y_test, y_pred))

      

print(classification_report(y_test, y_pred))