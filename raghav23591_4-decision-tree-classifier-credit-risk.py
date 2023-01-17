import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math as m

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

%matplotlib inline
df=pd.read_csv("../input/train.csv")
df.head(5)
df.dtypes
df.isna().sum()
d=df.dropna()
print("before",df.shape[0])

print("after",d.shape[0])

df.shape[0]-d.shape[0]
d.columns
d.Dependents.unique()
sns.distplot(d.ApplicantIncome)
sns.distplot(d.CoapplicantIncome)
sns.distplot(d.LoanAmount)
sns.boxplot(y=d.LoanAmount)
d.LoanAmount.mean()
d.LoanAmount.describe()
d.Loan_Amount_Term.unique()
d.Loan_Amount_Term.value_counts()
d.Credit_History.unique()

#d['Credit_History'].unique()
d.Property_Area.unique()

#d['Property_Area'].unique()
d.Loan_Status.unique()
d.Loan_Status.value_counts()
# H0:Applicant income is very important. So that may impact the loan status

d.groupby('Loan_Status')['ApplicantIncome'].mean()
sns.scatterplot(d.ApplicantIncome,d.LoanAmount)
sns.scatterplot(d.ApplicantIncome,d.CoapplicantIncome)
# H0: Self Employed is related to Loan Status

d.groupby('Self_Employed')['Loan_Status'].value_counts()
d.groupby('Self_Employed')['Loan_Status'].count()
d.groupby('Self_Employed')['Loan_Status'].value_counts()/d.groupby('Self_Employed')['Loan_Status'].count()

# To know the probability, we can divide by counts()
#H0: Education and Loan Status are related to each other.. 

d.groupby('Education')['Loan_Status'].value_counts()
d.groupby('Education')['Loan_Status'].count()
d.groupby('Education')['Loan_Status'].value_counts()/d.groupby('Education')['Loan_Status'].count()
#H0: Education and Self Employed are related to each other

d.groupby('Education')['Self_Employed'].value_counts()/d.groupby('Education')['Loan_Status'].count()
#H0: Gender and Loan status are related to each other. 

d.groupby('Gender')['Loan_Status'].value_counts()/d.groupby('Gender')['Loan_Status'].count()
#H0: Married and Loan status are related to each other. 

d.groupby('Married')['Loan_Status'].value_counts()/d.groupby('Married')['Loan_Status'].count()

#H0: Dependants and Loan Status are related to each other. 

d.groupby('Dependents')['Loan_Status'].value_counts()/d.groupby('Dependents')['Loan_Status'].count()
# To find the mean of the Applicant Income who has Dependants



d.groupby('Dependents')['ApplicantIncome'].mean()
#H0: Property Area and Loan Status are related to each other. 

d.groupby('Property_Area')['Loan_Status'].value_counts()/d.groupby('Property_Area')['Loan_Status'].count()
d.columns
#Making the Data set ordinal:::



def datacleaning(x):

    x.Gender=x.Gender.map(lambda x:1 if x=='Male' else 0) #Assigning 1 if Gender is 'Male' and 0 if Gemder is 'Female'

    x.Married=x.Married.map(lambda x:1 if x=='Yes' else 0) #Assigning 1 if Married is 'Yes' and 0 if Married is 'No'

    x.Dependents=x.Dependents.map(lambda x:3 if x=='3+' else int(x)) #Assigning 3 if Dependents is '3+'' and same values if other than '3+'

    x.Education=x.Education.map(lambda x:1 if x=='Graduate' else 0) #Assigning 1 if Education is 'Graduate' and o if education is 'Not graduate'

    x.Self_Employed=x.Self_Employed.map(lambda x:1 if x=='Yes' else 0) #Assigning 1 if SelfEmployed is 'Yes' and 0 if selfEmployed is 'No'

    dummies=pd.get_dummies(x.Property_Area) #Get Dummies will create columns(=Unique values) and assign 1 and 0. 

    #x=x.join(dummies) #Joining dummies to the dataset

    x["TotalIncome"]=x.ApplicantIncome+x.CoapplicantIncome

    y=x.Loan_Status.map(lambda x:1 if x=='Y' else 0) #Assigning 1 if the Loan status is 'Y' and 0 if the Loan Status is 'N'

    x=x.drop(['Loan_ID','LoanAmount','Loan_Amount_Term','Property_Area','Loan_Status','ApplicantIncome','CoapplicantIncome'],axis=1) #Dropping the unwanted columns from the dataset

    return x,y

   
X,y=datacleaning(d.copy())
X.head(10)
#Testing Entropy

entropy = -0.55*m.log2(0.55)-0.45*m.log2(0.45)

print(entropy)
#In Other Way... 



def entropy(p,q):

    e=-p*m.log2(p)-q*m.log2(q)

    return e
y.value_counts()
y.value_counts()/len(y)
d.groupby('Gender')['Loan_Status'].value_counts()/d.groupby('Gender')['Loan_Status'].count()
entropy(0.627907,0.372093)
entropy(0.705584,0.294416)
d.index
df.index
test_index=df.index.difference(d.index)
test=df.loc[test_index]
test.isna().sum()
d.Credit_History.value_counts()
test.Gender=test.Gender.fillna("Male")

test.Married=test.Married.fillna("Yes")

test.Dependents=test.Dependents.fillna("0")

test.Self_Employed=test.Self_Employed.fillna("No")

test.LoanAmount=test.LoanAmount.fillna(d.LoanAmount.median())

test.Loan_Amount_Term=test.Loan_Amount_Term.fillna(d.Loan_Amount_Term.median())

test.Credit_History=test.Credit_History.fillna(1)
X_test,y_test=datacleaning(test)
dct=DecisionTreeClassifier(criterion="entropy")

dct.fit(X,y)
print("accuracy train",dct.score(X,y))

print("accuracy test",dct.score(X_test,y_test))
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(dct, out_file=dot_data,  feature_names =X.columns,class_names=["No","Yes"],

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
for height in range(1,6):

    for leaf in range(1,5):

        dc=DecisionTreeClassifier(criterion="entropy",max_depth=height,min_samples_leaf=leaf)

        dc.fit(X,y)

        print('depth',height,"leaf",leaf,"accuracy train",dc.score(X,y))

        print('depth',height,"leaf",leaf,"accuracy test",dc.score(X_test,y_test))
dc=DecisionTreeClassifier(criterion="entropy",max_depth=4,min_samples_leaf=3)

dc.fit(X,y)

dc.score(X_test,y_test)