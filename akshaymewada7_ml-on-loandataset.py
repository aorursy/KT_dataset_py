# Importing Neccesary Libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline 
#  Reading Data using pandas

Loan = pd.read_csv('../input/Loan payments data.csv') # path 



# printing head of Data

Loan.head()  
# Counting is there any null values

Loan.isnull().sum()
# Droping unwanted Data from Dataframe

Loan = Loan.drop(['paid_off_time','Loan_ID','effective_date','due_date'],axis = 1)
# printing Data Frame

Loan
# plotting loan_staus with respect to Age

plt.figure(figsize=(15,8))

sns.countplot(x='age',hue='loan_status',data=Loan)
# plotting loan_staus with respect to Education

plt.figure(figsize=(15,8))

sns.countplot(x='education',hue='loan_status',data=Loan)
# plotting loan_staus with respect to Gender

plt.figure(figsize=(15,8))

sns.countplot(x='Gender',hue='loan_status',data=Loan)
# plotting loan_staus with respect to Principal

plt.figure(figsize=(15,8))

sns.countplot(x='Principal',hue='loan_status',data=Loan)
# plotting loan_staus with respect to terms

plt.figure(figsize=(15,8))

sns.countplot(x='terms',hue='loan_status',data=Loan)
# Plotting heatmap of Data correlation

plt.figure(figsize=(15,8))

sns.heatmap(data=Loan.corr())
# Dividing age in three Diffrent Group.

def Age(age):

    if 18<=age<32:

        return 0

    elif 32<=age<46:

        return 1

    else:

        return 2

Loan['age'] = Loan['age'].apply(Age)
# Getting Unique values of : Principal , terms, education, Gender



Principle_list = list(Loan['Principal'].unique())

terms_list = list(Loan['terms'].unique())

education_list = list(Loan['education'].unique())

Gender_list = list(Loan['Gender'].unique())

loan_status_list = list(Loan['loan_status'].unique())

# printing list of unique values

print('Principal:',Principle_list)

print('terms:',terms_list)

print('education:',education_list)

print('Gender:',Gender_list)
# Creating Dictionary of List to append values in Data frame



Principle_dict = {v:k for k,v in enumerate(Principle_list)}

terms_dict = {v:k for k,v in enumerate(terms_list)}

education_dict = {v:k for k,v in enumerate(education_list)}

Gender_dict = {v:k for k,v in enumerate(Gender_list)}

loan_status_dict = {v:k for k,v in enumerate(loan_status_list)}
# Appending Values to key using lambda function



Loan['Principal'] = Loan['Principal'].apply(lambda x:Principle_dict[x])

Loan['terms'] = Loan['terms'].apply(lambda x:terms_dict[x])

Loan['education'] = Loan['education'].apply(lambda x:education_dict[x])

Loan['Gender'] = Loan['Gender'].apply(lambda x:Gender_dict[x])

Loan['past_due_days'] = Loan['past_due_days'].fillna(Loan['past_due_days'].median())

Loan['loan_status'] = Loan['loan_status'].apply(lambda x:loan_status_dict[x])
Loan
# Separating Train and Target Data.

Target = Loan['loan_status']

Train = Loan.drop(['loan_status','age'],axis=1)
#  Using train_test_split slitting data in training and testing Dataset

# train-test  ratio : 80% - 20%



from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test =  train_test_split(Train ,Target,test_size=0.2,random_state=120)
# importing the model for prediction



from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
# creating list of tuple wth model and its name  

models = []

models.append(('GNB',GaussianNB()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('DT',DecisionTreeClassifier()))

models.append(('RF',RandomForestClassifier()))

models.append(('LG',LogisticRegression()))
# imorting cross Validation for calcuting score

from sklearn.cross_validation import cross_val_score



acc = []   # list for collecting Accuracy of all model

names = []    # List of model name



for name, model in models:

    

    acc_of_model = cross_val_score(model, X_train, Y_train, cv=10, scoring='accuracy')

    

    # appending Accuray of different model to acc List

    acc.append(acc_of_model)

    

    # appending name of models

    names.append(name)

    

    # printing Output 

    Out = "%s: %f (%f)" % (name, acc_of_model.mean(), acc_of_model.std())

    print(Out)
# Compare Algorithms Accuracy with each other on same Dataset

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(acc)

ax.set_xticklabels(names)

plt.show()