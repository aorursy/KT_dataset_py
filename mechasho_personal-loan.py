# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Important Library to handel the data



# For Mathematical caluclation and data handling

import numpy as np

import pandas as pd    



# For Data visulization

import seaborn as sns                      

import matplotlib.pyplot as plt

%matplotlib inline



# To understand better use of libraries I have imported library as per need further.
data = pd.read_csv('/kaggle/input/personal-loan-modeling/Bank_Personal_Loan_Modelling.csv')

data.head()
print('shape: (rows,columns)'+str(data.shape))
data.info()
# 5-point summary help to understand the center tendency value,spread and shape of data

data.describe()
#Dropping ID,Experience and ZIP code columns from the data.

data.drop('ID', axis=1, inplace=True)

data.drop('Experience', axis=1, inplace=True)

data.drop('ZIP Code', axis=1, inplace=True)

data.head()
#Replacing '0' value in mortgage column with its mean value.



data['Mortgage'] = data['Mortgage'].replace(0,data['Mortgage'].mean())

data.head()
#Visualization with histogram to know the frequency of each columns.

histogram = list(data)[0:]

data[histogram].hist(stacked=True, bins=10,figsize=(15,30),layout=(12,2));
#Personal Loan Distribution

PL_T = len(data.loc[data['Personal Loan']==True])       #PL_T = Personal Loan True value



PL_F = len(data.loc[data['Personal Loan']==False])      #PL_F = Personal Loan False value



print('Number of Personal Loan taken: '+ str(PL_T))



print('Percentage: '+ str(PL_T*100/(PL_F+PL_T))+'%')



print()



print('Number of Personal Loan not taken: '+ str(PL_F))



print('Percentage: '+ str(PL_F*100/(PL_F+PL_T))+'%')

#Checking correlation of personal column with each column.

corr = data.corr()                                        # function for correlation



corr.style.background_gradient(cmap='RdBu_r')            # We just style our correlation table
#Visulization of our 1st Hypothesis High salaries are less feasible to buy personal loans.



plt.figure(figsize=(10,8))



#Checking Hypothesis_1: High salaries are less feasible to buy personal loans

data.groupby('Personal Loan')['Income'].mean().plot(kind='bar',title='Income');
#Check for 2nd Hypothesis More the number of earning family members, less probability of buying personal loans.



H2 = pd.crosstab(data['Personal Loan'],data['Family'])

print(H2)



print()



for i in H2.iloc[:]:

    

    print('Chances of taking personal loan having family size as '+ str(i) +' '

          +str(H2.iloc[1,(i-1)]*100/(H2.iloc[0,(i-1)]+H2.iloc[1,(i-1)]))+' %')
#Check for 3rd Hypothesis People who are graduated or Advanced/Professionals are more to buy personal loans.



H3 = pd.crosstab(data['Personal Loan'],data['Education'])

print(H3)



print()



for i in H3.iloc[:]:

    print('Chances of taking personal loan as '+ str(i) +' '

          +str(H3.iloc[1,(i-1)]*100/(H3.iloc[0,(i-1)]+H3.iloc[1,(i-1)]))+' %')

#Since we fail to reject our Hypotesis,let's have a Visulization of hypothesis 3

H3.div(H3.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True);
# Before Checking for our 4th hypothesis we have to divide our age column into some range to have our hypothesis 4.

# here we are trying to create Age Group column. 

# As 22-30 a young age person,30-50 a mid age person and above 50 as Old age person.

# Note - Left bin is exclusive and right bin is inclusive.



data['Age_Group'] = pd.cut(x=data['Age'], bins=[22, 30, 50, 67], labels=['Young', 'Mid', 'Old'])



# For refernce of new function, here .cut() function is used to cut the columnn into a range of bin in category.

data.head()
# Check for 4th Hypothesis Customers with probably the age of 30–50 will buy personal loans.



H4 = pd.crosstab(data['Age_Group'],data['Personal Loan'])

print(H4)



print()



H4.div(H4.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True);
#dropping Age_Group column 



data.drop('Age_Group', axis=1, inplace=True)

data.head()
#Library for splitting of data

from sklearn.model_selection import train_test_split



#Libarary for confusion matrix

from sklearn import metrics



#Libarary for Roc curve score

from sklearn.metrics import roc_auc_score
#Spliting of data into train and test set



# 1) Split dependent and independent data

X = data.drop(['Personal Loan'], axis=1)

y = data['Personal Loan']



# 2) Spliting randomly in train and test set

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.30, random_state=1)
#Library for Logistic regression model

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
# 1) Fit model on the Train-set

LM_model = LogisticRegression(solver='liblinear')

LM_model.fit(X_train,y_train)



# 2) Predict on Test-set

ypredict_LM = LM_model.predict(X_test)



# 3) Coefficient and intercept of model

coef_LM = pd.DataFrame(LM_model.coef_)

coef_LM['intercept'] = LM_model.intercept_

print(coef_LM)

print()



# 4) Score of model

LM_model_score = LM_model.score(X_test,y_test)

print('Score of the model '+str(LM_model_score))
# 5) Confusion Matrix



LM_matrix = metrics.confusion_matrix(y_test, ypredict_LM)

print(LM_matrix)



print()



# For better understanding of confusion matrix we plot it on heatmap

LM_CM = pd.DataFrame(LM_matrix, index = [i for i in ['0','1']],

                    columns = [i for i in ['Predict 0', 'Predict 1']])



plt.figure(figsize=(7,5))



sns.heatmap(LM_CM,annot=True, fmt='g') 
# 6) CLassification Report

print(classification_report(y_test,ypredict_LM))
# 7) Area under curve

AUC_LM = round(roc_auc_score(y_test, ypredict_LM)*100)

print("AUC: ",  AUC_LM)
#Library for K-NN Model

from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore
# 1) Reading Personal Loan distribution by group

table=data.groupby(['Personal Loan']).count()

print(table)



# 2) Converting all attributes to z-score

XScaled = X.apply(zscore)
# 2) Spliting randomly in train and test set

X_train1,X_test1,y_train1,y_test1 = train_test_split(XScaled, y, test_size=0.30, random_state=1)
# 3) Creating Model on train set

Score = []

for k in range(1,10):

    KNN_model= KNeighborsClassifier(n_neighbors=k)

    KNN_model.fit(X_train1,y_train1)

    Score.append(KNN_model.score(X_test1,y_test1))

    

plt.plot(range(1,10),Score)



print(Score)
# 4) Confusion Matrix

ypredict_KNN = KNN_model.predict(X_test1)



KNN_matrix = metrics.confusion_matrix(y_test1, ypredict_KNN)

print(KNN_matrix)



KNN_CM = pd.DataFrame(KNN_matrix, index= [i for i in ['0','1']]

                     , columns= [i for i in ['Predict 0', 'Predict 1']])



plt.figure(figsize=(7,5))



sns.heatmap(KNN_CM, annot=True, fmt='g')
# 5) classification_report

print(classification_report(y_test1,ypredict_KNN))
# 6) Area under curve

AUC_KNN = round(roc_auc_score(y_test, ypredict_KNN)*100)



print("AUC: ",  AUC_KNN)
#Library for Naive Bayes

from sklearn.naive_bayes import GaussianNB
# 1) Creating model on train set

NB_model = GaussianNB()

NB_model.fit(X_train,y_train.ravel())



# 2) Predict on test set

ypredict_NB = NB_model.predict(X_test)

# 3) Confusion Matrix

NB_matrix = metrics.confusion_matrix(y_test,ypredict_NB)

print(NB_matrix)



NB_CM = pd.DataFrame(NB_matrix, index= [i for i in ['0','1']]

                     , columns= [i for i in ['Predict 0', 'Predict 1']])



plt.figure(figsize=(7,5))



sns.heatmap(NB_CM, annot=True, fmt='g')
# 5) CLassification Report

print(classification_report(y_test,ypredict_NB))
# 8) Area under curve

AUC_NB = round(roc_auc_score(y_test, ypredict_NB)*100)

print("AUC: ",  AUC_NB)
from sklearn.pipeline import Pipeline                               #For pipeline different models

from sklearn.preprocessing import StandardScaler                    #for Scaling the data

#Different types of classifier I will used for modeling and comparing

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
DT_model = Pipeline([('Scalar1', StandardScaler()),

           ('model_1',DecisionTreeClassifier())])        #Decision Tree Model



RF_model = Pipeline([('Scalar2', StandardScaler()),

           ('model_2', RandomForestClassifier())])        #Random Forest Model



Ada_model = Pipeline([('Scalar3', StandardScaler()),

                   ('model_3', AdaBoostClassifier())])     #ADa-Boosting model   



# Extra value like n_estimator and many more are taken as default you can change if needed
pipeline = [DT_model,RF_model, Ada_model]    # List of differnt model which are used for looping purpose



pipeline_dict = { 0: 'Decision Tree', 1: 'Random Forest', 2: 'Ada-Boosting'}
for test in pipeline:

    test.fit(X_train, y_train)

    

for i, model in enumerate(pipeline):

    print('{} Test Accuracy : {} '.format(pipeline_dict[i],model.score(X_test,y_test)))
# Decision Tree

ypredict_DT = DT_model.predict(X_test)

DT_matrix = metrics.confusion_matrix(y_test,ypredict_DT)

print('Decision Tree Confusion Matrix')

print(DT_matrix)



print()



# Random Forest 

ypredict_RF = RF_model.predict(X_test)

RF_matrix = metrics.confusion_matrix(y_test,ypredict_RF)

print('Random Forest Confusion Matrix')

print(RF_matrix)



print()



# Ada-Boosting

# Logistic Regression

ypredict_Ada = Ada_model.predict(X_test)

Ada_matrix = metrics.confusion_matrix(y_test,ypredict_Ada)

print('Ada Boosting Confusion Matrix')

print(Ada_matrix)
#for clear vision of confusion matrix

DT_CM = pd.DataFrame(DT_matrix, index=[i for i in [0,1]],

                    columns=[i for i in ["Predict 0", "Predict 1"]])

plt.figure(figsize=(8,5))

sns.heatmap(DT_CM, annot=True, fmt='g')
print(classification_report(y_test1,ypredict_DT))
AUC_DT = round(roc_auc_score(y_test1,ypredict_DT)*100)

print('AUC of Decision Tree model: ', AUC_DT)
#All the value are taken from aboved solved model for easy comparison among differnt types of model.

model_index = ['Logistic Model','KNN model', 'Naive Bayes', 'Decision Tree model']

column_lists = ['False Negative', 'True Postive','Recall', 'Accuracy (%)', 'AUC (%)']

models_info = {'False Negative' : [73,69,64,19],'True Postive' : [76,80,85,130],

               'Recall' : [0.51,0.54,0.57,0.87], 'Accuracy (%)' : [94,95,87,98],

               'AUC (%)' : [75,77,74,93]}

comparison_table = pd.DataFrame(models_info, model_index, column_lists)

comparison_table