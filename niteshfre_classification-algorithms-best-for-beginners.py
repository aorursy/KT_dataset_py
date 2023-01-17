import pandas as pd  #for loading the dataset as DataFrame

import numpy as np   #for handling multi-D arrays and mathematical computations

import seaborn as sb #highly interactive visualization of dataset

from matplotlib import pyplot as plt #visualization the data

from sklearn.model_selection import train_test_split  # split the data into trian and test sets

# different algorithms for comparisons

from sklearn.ensemble import RandomForestClassifier # also used for feature selection

from sklearn.ensemble import GradientBoostingClassifier #boosting algorithm

from sklearn.tree import DecisionTreeClassifier     

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC   # support vector classifier (SVC)
data = pd.read_csv('../input/loan-prediction.csv') 

#our dataset is about Loan Status of different applicants with several features
data.head() #overview of our dataset
data.isna().sum() #check how many number of Nan are present in each column
data.shape   #dimension of our dataset
data.fillna(method='bfill',inplace=True) # here we are use backward filling to remove our Nan from Dataset
#countplot of different gender on the basis of there loan status

sb.countplot(x='Gender',

             data=data,

             hue='Loan_Status',

             palette="GnBu_d") 
#here we are clearly obeserve the what is applicantIncome and whether he/she is self_employed or not,

#with there Loan Status

sb.catplot(x='Gender',

           y='ApplicantIncome',

           data=data,

           kind ='bar',

           hue='Loan_Status',

           col='Self_Employed')
#here we are clearly obeserve the what is co-applicantIncome and whether he/she is self_employed or not,

#with there Loan Status

sb.catplot(x='Gender',

           y='CoapplicantIncome',

           data=data,

           kind ='bar',

           hue='Loan_Status',

           col='Self_Employed')
data['ApplicantIncome'].plot(kind='hist',bins=50) #histogram of Applicant-Income

# we see that most of them are in between 0-10000
data['CoapplicantIncome'].plot(kind='hist',bins=50)

#histogram of coappicantIncome which almost similar to the ApplicantIncome's histogram
#it is more useful to use ApplicantIncome and CoapplicantIncome as one featue i.e, Total_Income

#for reducing the feature set

#So, I am creating new column Total_Income as the sum of ApplicantIncome and CopplicantIncome

data['Total_Income']=(data['ApplicantIncome']+data['CoapplicantIncome'])

data['Total_Income'].plot(kind='hist',bins=50) #histogram of Total_Income which is almost similar with above two

data.drop(columns=['ApplicantIncome','CoapplicantIncome'],inplace=True) 
sb.countplot(data.Dependents,data=data,hue='Loan_Status')

#count of different dependents with respect to there Loan_status
sb.countplot(data.Education,data=data,hue='Loan_Status',palette='Blues')

#count of graduated or non-graduated with respect to there Loan_status
sb.countplot(data.Married,data=data,hue='Loan_Status')

#count of Married or non-Married applicant with respect to there Loan_status
sb.barplot(x='Credit_History',y='Property_Area',data=data,hue='Loan_Status',orient='h')

# relation of credit history in different Property Area with respect to there Loan_Status
sb.barplot(x='Loan_Amount_Term',y='LoanAmount',data=data,hue='Loan_Status',palette='Blues')

#visualizing LoanAmount on the basis of LoanAmountTerm with respect to Loan_Status
#As above we observe that our there are so many columns with categorical values.

#which are useful feature for predicting our Loan Status at the end

#for the sake of simplicity I am coverting these categorical values in to numeric values.

x = pd.Categorical(data['Gender'])               # Male=1,Female=0

data['Gender']=x.codes



x = pd.Categorical(data['Married'])              # Yes=1,No=0

data['Married']=x.codes



x = pd.Categorical(data['Education'])            #Graduate=0,Non-graduated=1

data['Education']=x.codes



x = pd.Categorical(data['Self_Employed'])        #Yes=1,No=0

data['Self_Employed']=x.codes



x = pd.Categorical(data['Property_Area'])        # Rural=0,SemiUrban=1,Urban=2

data['Property_Area']=x.codes



x = pd.Categorical(data['Loan_Status'])          #Y=1,N=0

data['Loan_Status'] = x.codes



#in dependent column we clearly see that there is + sign for dependents more than 3

#which makes it column of object data type 

#So, I am going to remove this sign and convert it into numeric value

data['Dependents'] = data['Dependents'].str.replace('+','')     

data['Dependents'] = pd.to_numeric(data['Dependents'])
plt.figure(figsize=(10,7))

sb.heatmap(data.corr(),cmap='Greens',annot=True)

#Visualizing the correlation matrix using heatmap 
#We are going to predict the Loan_Status

Y=data['Loan_Status']

X=data.drop(columns=['Loan_Status','Loan_ID']) #X is all columns except Loan_Status and Loan_ID

# split the train and test dataset where test set is 30% of original dataset

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3) 
clf = RandomForestClassifier(n_estimators=400,max_depth=5) #defining RandomForest Classifier
clf = clf.fit(xtrain,ytrain)  #fitting our train dataset
clf.score(xtest,ytest)       #score on our test dataset
pd.Series(clf.feature_importances_,xtrain.columns).sort_values(ascending=False)

#feature importance in descending order

#So, I am using only top 4 features as my input
#Respliting the trianing and testing dataset

Y=data['Loan_Status']

X=data[['Credit_History','Total_Income','LoanAmount']] #X is top 3 feature having more feature importance values

# split the train and test dataset where test set is 30% of original dataset

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3) 
#Re-applying the RandomForest Classifiers

clf = RandomForestClassifier(n_estimators=400,max_depth=5) 

clf = clf.fit(xtrain,ytrain) 

clf.score(xtest,ytest)

#we can clearly observe that it increases the accuracy percentage
clf = LogisticRegression()  #defining Logistic Regression

clf = clf.fit(xtrain,ytrain) 

clf.score(xtest,ytest)
clf = SVC()  #defining Support Vector Classifier

clf = clf.fit(xtrain,ytrain) 

clf.score(xtest,ytest)
clf = KNeighborsClassifier(n_neighbors=3)  #defining K-nearest Neighbors(KNN) Classifier

clf = clf.fit(xtrain,ytrain) 

clf.score(xtest,ytest)
clf = DecisionTreeClassifier(max_depth=3)  #defining DecisionTree Classifier

clf = clf.fit(xtrain,ytrain) 

clf.score(xtest,ytest)
clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=2)  #defining Logistic Regression

clf = clf.fit(xtrain,ytrain) 

clf.score(xtest,ytest)