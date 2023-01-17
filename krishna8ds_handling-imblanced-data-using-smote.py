# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing Libraries



# Importing Mathematical Library

import numpy as np

#Importing Data Manipulation Library

import pandas as pd



# Importing Visualization Libraries

import matplotlib.pyplot as plt # Basic Visualization Library

import seaborn as sns # Statistical Visualization Library



# Importing Machine Learning Libraries

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn import metrics



from sklearn.model_selection import RandomizedSearchCV



from prettytable import PrettyTable



from imblearn.over_sampling import SMOTE
data = pd.read_csv('../input/DATA.csv')
data.head(20)
# Understanding the Dataset

data.info()
# Understanding information of all the Columns in the Dataset

data.info(verbose=True)
def null_function():

    null_values = pd.DataFrame((data.isnull().sum()/len(data.index)*100),columns=['Percent_Null'])

    only_missing_variables = null_values[null_values['Percent_Null'] !=0 ]

    return pd.DataFrame(only_missing_variables.sort_values(by='Percent_Null', ascending=False))
pd.DataFrame(null_function())


# Finding the Total No. of Null Values in each Column:

pd.DataFrame(data.isnull().sum())[pd.DataFrame(data.isnull().sum())[0] != 0].sort_values(by=0, ascending=False).rename(columns={0:'Total No. of Null Values'})


# Displaying the Columns with null values more than 30%

null_values = pd.DataFrame((data.isnull().sum()/len(data.index)*100),columns=['Percent_Null'])

only_missing_variables = null_values[(null_values['Percent_Null'] !=0) & (null_values['Percent_Null'] >=30)]

pd.DataFrame(only_missing_variables.sort_values(by='Percent_Null', ascending=False))
# Dropping the Columns:

data.drop(['Medical_History_10','Medical_History_32','Medical_History_24','Medical_History_15','Family_Hist_5','Family_Hist_3','Family_Hist_2','Insurance_History_5','Family_Hist_4'],axis=1,inplace=True)
null_function()
np.dtype(data['Employment_Info_1'])
data['Employment_Info_1'].nunique()
data['Employment_Info_1'].value_counts().sort_values()


sns.boxplot(data['Employment_Info_1'])

# Note: distplot will not excute if there are nan in the data.
data['Employment_Info_1'].describe()


#Checking the median of 'Employment_Info_1':

data['Employment_Info_1'].median()


#Imputing missing values:

data['Employment_Info_1'] = data['Employment_Info_1'].fillna(data['Employment_Info_1'].median())


# Again Checking for Missing Values:

null_function()
np.dtype(data['Employment_Info_6'])
data['Employment_Info_6'].nunique()
sns.boxplot(data['Employment_Info_6'])


data['Employment_Info_6'].describe()
data['Employment_Info_6'] = data['Employment_Info_6'].fillna(data['Employment_Info_6'].median())
# Again Checking for Missing Values if any:

null_function()
np.dtype(data['Employment_Info_4'])
data['Employment_Info_4'].nunique()
sns.boxplot(data['Employment_Info_4'])
data['Employment_Info_4'].median()
data['Employment_Info_4'].value_counts()
sns.boxplot(data['Medical_History_1'])
data['Medical_History_1'].describe()
data['Medical_History_1'].nunique()
data['Medical_History_1'].value_counts()
data['Medical_History_1'].mode()
data['Medical_History_1'] = data['Medical_History_1'].fillna(data['Medical_History_1'].mode()[0])
#Checking for Null Values:

null_function()


data = data.drop(['Id'],axis=1)


data.head()
data['Product_Info_2_char'] = data.Product_Info_2.str[0]

data['Product_Info_2_num'] = data.Product_Info_2.str[1]
data['Product_Info_2_char'] = pd.factorize(data['Product_Info_2_char'])[0]

data['Product_Info_2_num'] = pd.factorize(data['Product_Info_2_num'])[0]


# Dropping 'Product_Info_2' Column

data = data.drop('Product_Info_2',axis=1)
data.head()
data.fillna(0,inplace=True)


#Checking the Shape of the Final Data:

data.shape
X = data.drop('Response',axis=1)

y = data.Response
# Splitting the data into Train and Test

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=2)






report= PrettyTable()

report.field_names=['Model name','Accuracy Score','Precision Score','Recall Score','F1_score','Cohen Kappa Score']





classifier=['LogisticRegression','KNN','DecisionTreeClassifier','RandomForestClassifier']

accuracy=[]

precision=[]

recall=[]

f1_score=[]

kappa=[]



for i in classifier:

    if i=='LogisticRegression':

        model1=LogisticRegression()

        model1.fit(X_train,y_train)

        log_pred=pd.DataFrame(model1.predict(X_test))

        #Evaluation metrics

        report.add_row([i,

                    metrics.accuracy_score(y_test,log_pred),

                    metrics.precision_score(y_test,log_pred,average='weighted'),

                    metrics.recall_score(y_test,log_pred,average='weighted'),

                    metrics.f1_score(y_test,log_pred,average='weighted'),

                    metrics.cohen_kappa_score(y_test,log_pred)])

        

    elif i=='KNN': 

        model2=KNeighborsClassifier()

        model2.fit(X_train,y_train)

        knn_pred=model2.predict(X_test)

        #Evaluation metrics

        report.add_row([i,

                    metrics.accuracy_score(y_test,knn_pred),

                    metrics.precision_score(y_test,knn_pred,average='weighted'),

                    metrics.recall_score(y_test,knn_pred,average='weighted'),

                    metrics.f1_score(y_test,knn_pred,average='weighted'),

                    metrics.cohen_kappa_score(y_test,log_pred)])

        

    elif i=='DecisionTreeClassifier':

        model3=DecisionTreeClassifier()

        model3.fit(X_train,y_train)

        dec_pred=model3.predict(X_test)

        #Evaluation metrics

        report.add_row([i,

                    metrics.accuracy_score(y_test,dec_pred),

                    metrics.precision_score(y_test,dec_pred,average='weighted'),

                    metrics.recall_score(y_test,dec_pred,average='weighted'),

                    metrics.f1_score(y_test,dec_pred,average='weighted'),

                    metrics.cohen_kappa_score(y_test,log_pred)])

        

    elif i=='RandomForestClassifier':

        model4=RandomForestClassifier()

        model4.fit(X_train,y_train)

        random_pred=model4.predict(X_test)

        #Evaluation metrics

        report.add_row([i,

                    metrics.accuracy_score(y_test,random_pred),

                    metrics.precision_score(y_test,random_pred,average='weighted'),

                    metrics.recall_score(y_test,random_pred,average='weighted'),

                    metrics.f1_score(y_test,random_pred,average='weighted'),

                    metrics.cohen_kappa_score(y_test,log_pred)])

print(report)

## Code to find the best hyper parameters for the KNN and DecisionTree and Random forest



from sklearn.model_selection import RandomizedSearchCV

## code to find the best model for the dataset





best_par= PrettyTable()

best_par.field_names=['Model name','Best Parameters','Best Score']





classifier=['KNN','DecisionTreeClassifier','RandomForestClassifier']





for i in classifier:

    if i=='KNN': 

        grid1={'n_neighbors': np.arange(1,50,2),'p': np.arange(1,50)}

        ran_search1=RandomizedSearchCV(model2,grid1,cv=5)

        ran_search1.fit(X_train,y_train)

        best_par.add_row([i,

                          ran_search1.best_params_,

                          ran_search1.best_score_])

        

    elif i=='DecisionTreeClassifier':

        grid2={'criterion':['gini','entropy'],'max_depth': np.arange(2,10),'max_leaf_nodes':np.arange(2,10),'min_samples_leaf':np.arange(2,10)}

        ran_search2=RandomizedSearchCV(model3,grid2,cv=5)

        ran_search2.fit(X_train,y_train)

        best_par.add_row([i,

                          ran_search2.best_params_,

                          ran_search2.best_score_])

        

    elif i=='RandomForestClassifier':

        grid3={'criterion':['gini','entropy'],'n_estimators':np.arange(1,100),'max_features':np.arange(1,10)}

        ran_search3=RandomizedSearchCV(model4,grid3,cv=3)

        ran_search3.fit(X_train,y_train)

        best_par.add_row([i,

                          ran_search3.best_params_,

                          ran_search3.best_score_])

        

print(best_par)
# Since Random Forest Classifier outperformed, let's use it with parameter obtained and test it with unseen test data.

model4=RandomForestClassifier(n_estimators=88,max_features=9,criterion='gini')

model4.fit(X_train,y_train)

random_pred=model4.predict(X_test)

    #Evaluation metrics

print("Accuracy Score:",metrics.accuracy_score(y_test,random_pred))

print("Precision Score:",metrics.precision_score(y_test,random_pred,average='weighted'))

print("Recall Score:",metrics.recall_score(y_test,random_pred,average='weighted'))

print("F1 Score:",metrics.f1_score(y_test,random_pred,average='weighted'))



print(metrics.confusion_matrix(y_test,random_pred))



print(metrics.classification_report(y_test,random_pred))


#Checking the distribution of Target Variable:

sns.countplot(data.Response)


# SMOTE Sampling Technique:

smote = SMOTE()

X_sm, y_sm = smote.fit_sample(X,y)


sns.countplot(y_sm)
# Splitting the Balanced Data for finally modelling:

X_sm_train, X_sm_test, y_sm_train, y_sm_test = train_test_split(X_sm,y_sm,test_size=0.3,random_state=2)






report= PrettyTable()

report.field_names=['Model name','Accuracy Score','Precision Score','Recall Score','F1_score','Cohen Kappa Score']





classifier=['LogisticRegression','KNN','DecisionTreeClassifier','RandomForestClassifier']

accuracy=[]

precision=[]

recall=[]

f1_score=[]

kappa=[]



for i in classifier:

    if i=='LogisticRegression':

        model1=LogisticRegression()

        model1.fit(X_sm_train,y_sm_train)

        log_pred=pd.DataFrame(model1.predict(X_sm_test))

        #Evaluation metrics

        report.add_row([i,

                    metrics.accuracy_score(y_sm_test,log_pred),

                    metrics.precision_score(y_sm_test,log_pred,average='weighted'),

                    metrics.recall_score(y_sm_test,log_pred,average='weighted'),

                    metrics.f1_score(y_sm_test,log_pred,average='weighted'),

                    metrics.cohen_kappa_score(y_sm_test,log_pred)])

        

    elif i=='KNN': 

        model2=KNeighborsClassifier()

        model2.fit(X_sm_train,y_sm_train)

        knn_pred=model2.predict(X_sm_test)

        #Evaluation metrics

        report.add_row([i,

                    metrics.accuracy_score(y_sm_test,knn_pred),

                    metrics.precision_score(y_sm_test,knn_pred,average='weighted'),

                    metrics.recall_score(y_sm_test,knn_pred,average='weighted'),

                    metrics.f1_score(y_sm_test,knn_pred,average='weighted'),

                    metrics.cohen_kappa_score(y_sm_test,log_pred)])

        

    elif i=='DecisionTreeClassifier':

        model3=DecisionTreeClassifier()

        model3.fit(X_sm_train,y_sm_train)

        dec_pred=model3.predict(X_sm_test)

        #Evaluation metrics

        report.add_row([i,

                    metrics.accuracy_score(y_sm_test,dec_pred),

                    metrics.precision_score(y_sm_test,dec_pred,average='weighted'),

                    metrics.recall_score(y_sm_test,dec_pred,average='weighted'),

                    metrics.f1_score(y_sm_test,dec_pred,average='weighted'),

                    metrics.cohen_kappa_score(y_sm_test,log_pred)])

        

    elif i=='RandomForestClassifier':

        model4=RandomForestClassifier()

        model4.fit(X_sm_train,y_sm_train)

        random_pred=model4.predict(X_sm_test)

        #Evaluation metrics

        report.add_row([i,

                    metrics.accuracy_score(y_sm_test,random_pred),

                    metrics.precision_score(y_sm_test,random_pred,average='weighted'),

                    metrics.recall_score(y_sm_test,random_pred,average='weighted'),

                    metrics.f1_score(y_sm_test,random_pred,average='weighted'),

                    metrics.cohen_kappa_score(y_sm_test,log_pred)])

print(report)




from sklearn.model_selection import RandomizedSearchCV

## code to find the best model for the dataset





best_par= PrettyTable()

best_par.field_names=['Model name','Best Parameters','Best Score']





classifier=['KNN','DecisionTreeClassifier','RandomForestClassifier']





for i in classifier:

    if i=='KNN': 

        grid1={'n_neighbors': np.arange(1,50,2),'p': np.arange(1,50)}

        ran_search1=RandomizedSearchCV(model2,grid1,cv=5)

        ran_search1.fit(X_sm_train,y_sm_train)

        best_par.add_row([i,

                          ran_search1.best_params_,

                          ran_search1.best_score_])

        

    elif i=='DecisionTreeClassifier':

        grid2={'criterion':['gini','entropy'],'max_depth': np.arange(2,10),'max_leaf_nodes':np.arange(2,10),'min_samples_leaf':np.arange(2,10)}

        ran_search2=RandomizedSearchCV(model3,grid2,cv=5)

        ran_search2.fit(X_sm_train,y_sm_train)

        best_par.add_row([i,

                          ran_search2.best_params_,

                          ran_search2.best_score_])

        

    elif i=='RandomForestClassifier':

        grid3={'criterion':['gini','entropy'],'n_estimators':np.arange(1,100),'max_features':np.arange(1,10)}

        ran_search3=RandomizedSearchCV(model4,grid3,cv=3)

        ran_search3.fit(X_sm_train,y_sm_train)

        best_par.add_row([i,

                          ran_search3.best_params_,

                          ran_search3.best_score_])

        

print(best_par)

# Since Random Forest Classifier outperformed, let's use it with parameter obtained and test it with unseen test data.

model4=RandomForestClassifier(n_estimators=96,max_features=4,criterion='gini')

model4.fit(X_sm_train,y_sm_train)

random_pred=model4.predict(X_sm_test)

    #Evaluation metrics

print("Accuracy Score:",metrics.accuracy_score(y_sm_test,random_pred))

print("Precision Score:",metrics.precision_score(y_sm_test,random_pred,average='weighted'))

print("Recall Score:",metrics.recall_score(y_sm_test,random_pred,average='weighted'))

print("F1 Score:",metrics.f1_score(y_sm_test,random_pred,average='weighted'),'\n')



print(metrics.confusion_matrix(y_sm_test,random_pred),'\n')



print(metrics.classification_report(y_sm_test,random_pred))