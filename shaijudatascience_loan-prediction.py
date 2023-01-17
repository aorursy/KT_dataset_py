# Check Data Directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np                       # For Linear Algebra

import pandas as pd                      # For data manipulation

import seaborn as sns                    # For data visualization

import matplotlib.pyplot as plt          # For data visualization      

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")        #To ignore any warnings
train = pd.read_csv("../input/loan-prediction-practice-av-competition/train_csv.csv") #Read train file

test  = pd.read_csv("../input/loan-prediction-practice-av-competition/test.csv.csv") # Read test file

submission=pd.read_csv("../input/loan-prediction-practice-av-competition/sample_submission.csv") #Read Submission file



#Lets Make Copy of file for not losing original data



train_df = train.copy()

test_df  = test.copy()
train_df.head()           #Display top five rows in the train_df dataset
test_df.head()         #Display top five rows in the test_df dataset.
train_df.columns # Display coloumn names in train_df.
test_df.columns # Display coloumn names in test_df.
print (train_df.shape) # Check the shape of train_df dataset

print (test_df.shape)  # Check the shape of test_df dataset
train_df.info()  # Print data type of each variable in train_df dataset
train_df.describe().T      #To Display the Basic Description of train_df.
train_df.describe(include=['object'])  # To see the statistics of non-numerical features
test_df.describe().T        #To Display the Basic Description of test_df.
train_df.nunique()  #To dislay the uniqe categories in train_df
test_df.nunique() #To dislay the uniqe categories in test_df
train_df.isnull().sum()
# Fill missing values in the categorical variable using mode function.



train_df['Gender'].fillna(train_df['Gender'].mode()[0], inplace=True) 



train_df['Married'].fillna(train_df['Married'].mode()[0], inplace=True) 



train_df['Dependents'].fillna(train_df['Dependents'].mode()[0], inplace=True) 



train_df['Self_Employed'].fillna(train_df['Self_Employed'].mode()[0], inplace=True) 



train_df['Credit_History'].fillna(train_df['Credit_History'].mode()[0], inplace=True)

train_df['Loan_Amount_Term'].value_counts()
# Use mode function to fill the missing value in Loan_Amount_Term.



train_df['Loan_Amount_Term'].fillna(train_df['Loan_Amount_Term'].mode()[0], inplace=True)
# Print mean and median value of LoanAmount.



print(train_df['LoanAmount'].mean())

print(train_df['LoanAmount'].median())
# The median value is smaller as compared to mean value in LoanAmount variable.



train_df['LoanAmount'].fillna(train_df['LoanAmount'].median(), inplace=True)
train_df.isnull().sum()
test_df.isnull().sum()
# Fill missing values in the test_df dataset.



test_df['Gender'].fillna(test_df['Gender'].mode()[0], inplace=True) 



test_df['Dependents'].fillna(test_df['Dependents'].mode()[0], inplace=True) 



test_df['Self_Employed'].fillna(test_df['Self_Employed'].mode()[0], inplace=True) 



test_df['Credit_History'].fillna(test_df['Credit_History'].mode()[0], inplace=True)



test_df['Loan_Amount_Term'].fillna(test_df['Loan_Amount_Term'].mode()[0], inplace=True)



test_df['LoanAmount'].fillna(test_df['LoanAmount'].median(), inplace=True)
test_df.isnull().sum()
# Visualize Target Variable ['Loan_Status']



train_df['Loan_Status'].value_counts()
train_df['Loan_Status'].value_counts(normalize=True).plot.bar()
# Visualize Independent Variable (Categorical)



plt.figure(figsize = (20,10))



plt.subplot(221)

train_df['Gender'].value_counts(normalize=True).plot.bar(figsize= (20,10), title = 'Gender')



plt.subplot(222)

train_df['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 



plt.subplot(223)

train_df['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')



plt.subplot(224)

train_df['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')



plt.show()
# Visualize Independent Variable (Ordinal)



plt.figure(figsize = (20,10))



plt.subplot(221)

train_df['Dependents'].value_counts(normalize=True).plot.bar(figsize= (20,10), title = 'No of Dependents')



plt.subplot(222)

train_df['Education'].value_counts(normalize=True).plot.bar(figsize= (20,10), title = 'Type of Education')



plt.subplot(223)

train_df['Property_Area'].value_counts(normalize=True).plot.bar(figsize= (20,10), title = 'Property_Area')



plt.show()
# Visualization of ApplicantIncome Variable (Numerical)



plt.figure(figsize = (20,10))



plt.subplot(121) 

sns.distplot(train_df['ApplicantIncome'],color="m", ) 



plt.subplot(122) 

train_df['ApplicantIncome'].plot.box(figsize=(16,5)) 



plt.show()
print('Mean value of ApplicantIncome is  :',train_df['ApplicantIncome'].mean())

print('Median value of ApplicantIncome is  :',train_df['ApplicantIncome'].median())
# Visualization of CoapplicantIncome Variable (Numerical)



plt.figure(figsize = (20,10))



plt.subplot(121) 

sns.distplot(train_df['CoapplicantIncome'],color="r", ) 



plt.subplot(122) 

train_df['CoapplicantIncome'].plot.box(figsize=(16,5)) 



plt.show()
# Visualization of LoanAmount Variable (Numerical)



plt.figure(figsize = (20,10))



plt.subplot(121) 

sns.distplot(train_df['LoanAmount'],color="b", ) 



plt.subplot(122) 

train_df['LoanAmount'].plot.box(figsize=(16,5)) 



plt.show()
train_df['LoanAmount_log'] = np.log(train_df['LoanAmount']) 



train_df['LoanAmount_log'].hist(bins=20) 



test_df['LoanAmount_log'] = np.log(test_df['LoanAmount']) 



test_df['LoanAmount_log'].hist(bins=20) 
train_df['Loan_Amount_Term'].value_counts()
# Visualization of Gender Variable vs Loan_Status



Gender=pd.crosstab(train_df['Gender'],train_df['Loan_Status']) 



Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,5))



plt.show()

# Visualization of Married Variable vs Loan_Status



Married=pd.crosstab(train_df['Married'],train_df['Loan_Status']) 



Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,4)) 



plt.show()

# Visualization of Dependents Variable vs Loan_Status



Dependents=pd.crosstab(train_df['Dependents'],train_df['Loan_Status']) 



Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(6,4))



plt.show()
# Visualization of Self_Employed Variable vs Loan_Status



Self_Employed=pd.crosstab(train_df['Self_Employed'],train_df['Loan_Status']) 



Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



plt.show()
# Visualization of Education Variable vs Loan_Status



Education =pd.crosstab(train_df['Education'],train_df['Loan_Status']) 



Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,4))



plt.show()
# Visualization of Credit_History vs Loan_Status



Credit_History=pd.crosstab(train_df['Credit_History'],train_df['Loan_Status'])



Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,4)) 



plt.show()
# Visualization of Property_Area vs Loan_Status



Property_Area=pd.crosstab(train_df['Property_Area'],train_df['Loan_Status'])



Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 



plt.show()
# Visualization of ApplicantIncome vs Loan_Status



bins=[0,2500,4000,6000,81000] 



group=['Low','Average','High', 'Very high'] 



train_df['Income_bin']=pd.cut(train_df['ApplicantIncome'],bins,labels=group)



Income_bin=pd.crosstab(train_df['Income_bin'],train_df['Loan_Status']) 



Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)



plt.xlabel('ApplicantIncome') 



P = plt.ylabel('Percentage')
# Visualization of CoapplicantIncome vs Loan_Status



bins=[0,1000,3000,42000] 



group=['Low','Average','High'] 



train_df['Coapplicant_Income_bin']=pd.cut(train_df['CoapplicantIncome'],bins,labels=group)



Coapplicant_Income_bin=pd.crosstab(train_df['Coapplicant_Income_bin'],train_df['Loan_Status']) 



Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 



plt.xlabel('CoapplicantIncome') 



P = plt.ylabel('Percentage')
# Visualization of Total_Income vs Loan_Status



train_df['Total_Income']=train_df['ApplicantIncome']+train_df['CoapplicantIncome']



bins=[0,2500,4000,6000,81000] 



group=['Low','Average','High', 'Very high'] 



train_df['Total_Income_bin']=pd.cut(train_df['Total_Income'],bins,labels=group)



Total_Income_bin=pd.crosstab(train_df['Total_Income_bin'],train_df['Loan_Status']) 



Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 



plt.xlabel('Total_Income') 



P = plt.ylabel('Percentage')
# Visualization of LoanAmount vs Loan_Status



bins=[0,100,200,700] 



group=['Low','Average','High'] 



train_df['LoanAmount_bin']=pd.cut(train_df['LoanAmount'],bins,labels=group)



LoanAmount_bin=pd.crosstab(train_df['LoanAmount_bin'],train_df['Loan_Status']) 



LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 



plt.xlabel('LoanAmount') 



P = plt.ylabel('Percentage')
#Lets drop the variables (Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income') in the train_df



train_df=train_df.drop([ 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)



train_df['Dependents'].replace('3+', 3,inplace=True)  #Replace Dependents (3+ as 3)

 

train_df['Loan_Status'].replace('N', 0,inplace=True)  #Replace Loan Status (N as 0)



train_df['Loan_Status'].replace('Y', 1,inplace=True)  ##Replace Loan Status (Y as 1)



train_df.head()
# Lets visualize the correlation of variables using heatmap



matrix = train_df.corr()



ax = plt.subplots(figsize=(9, 6)) 



sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
train_df.head()
train_lgr=train_df.drop(['Loan_ID','Income_bin'],axis=1) 



X = train_lgr.drop('Loan_Status',1)

y = train_lgr.Loan_Status



X = pd.get_dummies(X)

X.head()
X.head()
print(X.shape)

print(y.shape)
# Use  train_test_split function from sklearn to divide our train dataset.



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X,y, test_size =0.2,random_state=1)
print(x_train.shape)

print(y_train.shape)



print(x_test.shape)

print(y_test.shape)
#Import LogisticRegression and accuracy_score from sklearn and fit the logistic regression model.



from sklearn.linear_model import LogisticRegression 



from sklearn.metrics import accuracy_score



model_log = LogisticRegression(random_state=1)



model_log.fit(x_train, y_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_lgr = model_log.predict(x_test)



acc_log = accuracy_score(y_test,pred_lgr)*100 



acc_log
#Predict the test file using Log Regression Model



test_lr=test_df.drop('Loan_ID',axis=1)



test_lgr=pd.get_dummies(test_lr)



test_lgr.shape



pred_test_lr = model_log.predict(test_lgr)

#Final Submission



submission['Loan_Status'] = pred_test_lr 



submission['Loan_ID']     = test_df['Loan_ID']



submission['Loan_Status'].replace(0, 'N',inplace=True)



submission['Loan_Status'].replace(1, 'Y',inplace=True)



pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')



submission.head()
from sklearn.svm import SVC



model_svc=  SVC(gamma='auto')



model_svc.fit(x_train, y_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_svc = model_svc.predict(x_test)



acc_svc = accuracy_score(y_test,pred_svc)*100



acc_svc
from sklearn.neighbors import KNeighborsClassifier



model_knn= KNeighborsClassifier(n_neighbors = 3)



model_knn.fit(x_train, y_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_knn = model_knn.predict(x_test)



acc_knn = accuracy_score(y_test,pred_knn)*100



acc_knn
from sklearn.ensemble import RandomForestClassifier



model_rfc= RandomForestClassifier(n_estimators=100,random_state = 1)



model_rfc.fit(x_train, y_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_rfc = model_rfc.predict(x_test)



acc_rfc = accuracy_score(y_test,pred_rfc)*100



acc_rfc
from sklearn.naive_bayes import GaussianNB



model_gnb= GaussianNB()



model_gnb.fit(x_train, y_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_gnb = model_gnb.predict(x_test)



acc_gnb = accuracy_score(y_test,pred_gnb)*100



acc_gnb
from sklearn.linear_model import Perceptron



model_ptn= Perceptron()



model_ptn.fit(x_train, y_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_ptn = model_ptn.predict(x_test)



acc_ptn = accuracy_score(y_test,pred_ptn)*100



acc_ptn
from sklearn.tree import DecisionTreeClassifier



model_dtc= DecisionTreeClassifier(random_state=1)



model_dtc.fit(x_train, y_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_dtc = model_dtc.predict(x_test)



acc_dtc = accuracy_score(y_test,pred_dtc)*100



acc_dtc
import lightgbm as lgb



model_lgb=lgb.LGBMClassifier()



model_lgb.fit(x_train, y_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_lgb = model_lgb.predict(x_test)



acc_lgb = accuracy_score(y_test,pred_lgb)*100



acc_lgb
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron',  

              'Decision Tree','LGBMClassifier'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_rfc, acc_gnb, acc_ptn, 

             acc_dtc,acc_lgb]})

models.sort_values(by='Score', ascending=False)
#Adding total income by combining applicant's income and coapplicant's income



train_df['Total_Income']=train_df['ApplicantIncome']+train_df['CoapplicantIncome'] 



test_df['Total_Income']=test_df['ApplicantIncome']+test_df['CoapplicantIncome']
# Drop  Loan_Id,Income_bin and LoanAmount_log variables in train_fe 



train_fe = train_df.drop(['Loan_ID','Income_bin','LoanAmount_log'],axis=1)



# Drop  Loan_Id and LoanAmount_log variables in test_fe 



test_fe  = test_df.drop(['Loan_ID','LoanAmount_log'],axis=1)
train_fe.head()
#Convert Dependent Column into numeric feature



train_fe = train_fe.replace({'Dependents': r'3+'}, {'Dependents': 3}, regex=True)



test_fe = test_fe.replace({'Dependents': r'3+'}, {'Dependents': 3}, regex=True)
# process column, apply LabelEncoder to categorical features



from sklearn.preprocessing import LabelEncoder



lbl = LabelEncoder()



lbl.fit(list(train_fe["Dependents"].values))



train_fe["Dependents"] = lbl.transform(list(train_fe["Dependents"].values))



lbl.fit(list(test_fe["Dependents"].values))



test_fe["Dependents"] = lbl.transform(list(test_fe["Dependents"].values))



# shape 



print('Shape all_data: {}'.format(train_fe.shape))



print('Shape all_data: {}'.format(test_fe.shape))
train_fe = pd.get_dummies(train_fe)



test_fe = pd.get_dummies(test_fe)



train_fe.head()
test_fe.head()
train_fe.shape,test_fe.shape
from sklearn.feature_selection import SelectKBest,f_classif



X_fe  = train_fe.drop(['Loan_Status'],axis=1)



y_fe  = train_fe.Loan_Status



selector = SelectKBest(f_classif, k=6)



X_new = selector.fit_transform(X_fe, y_fe)



print(X_new)



X_new.shape

# Get back the features we've kept, zero out all other features



selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=X_fe.index, 

                                 columns=X_fe.columns)

selected_features.head()
# Dropped columns have values of all 0s, so var is 0, drop them



selected_columns = selected_features.columns[selected_features.var() != 0]



# Get the dataset with the selected features.



X_fe[selected_columns].head()
from sklearn.model_selection import train_test_split



xf_train, xf_test, yf_train, yf_test = train_test_split(X_fe[selected_columns],y_fe, test_size =0.2,random_state=1)



xf_train.shape,xf_test.shape,yf_train.shape,yf_test.shape
#Import LogisticRegression and accuracy_score from sklearn and fit the logistic regression model.



from sklearn.linear_model import LogisticRegression 



from sklearn.metrics import accuracy_score



model_log1 = LogisticRegression(random_state=1)



model_log1.fit(xf_train, yf_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



yf_pred = model_log1.predict(xf_test)



acc_log1 = accuracy_score(yf_test,yf_pred )*100 



acc_log1
#Support Vector Classifier



from sklearn.svm import SVC



model_svc1=  SVC(gamma='auto')



model_svc1.fit(xf_train, yf_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_svc1 = model_svc1.predict(xf_test)



acc_svc1 = accuracy_score(yf_test,pred_svc1)*100



acc_svc1
#KNN



from sklearn.neighbors import KNeighborsClassifier



model_knn1= KNeighborsClassifier(n_neighbors = 3)



model_knn1.fit(xf_train, yf_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_knn1 = model_knn1.predict(xf_test)



acc_knn1 = accuracy_score(yf_test,pred_knn1)*100



acc_knn1
#Random Forest Classifier



from sklearn.ensemble import RandomForestClassifier



model_rfc1= RandomForestClassifier(n_estimators=100,random_state = 1)



model_rfc1.fit(xf_train, yf_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_rfc1 = model_rfc1.predict(xf_test)



acc_rfc1 = accuracy_score(yf_test,pred_rfc1)*100



acc_rfc1
#Gaussian NB



from sklearn.naive_bayes import GaussianNB



model_gnb1= GaussianNB()



model_gnb1.fit(xf_train, yf_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_gnb1 = model_gnb1.predict(xf_test)



acc_gnb1 = accuracy_score(yf_test,pred_gnb1)*100



acc_gnb1
#Perceptron



from sklearn.linear_model import Perceptron



model_ptn1= Perceptron()



model_ptn1.fit(xf_train, yf_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_ptn1 = model_ptn1.predict(xf_test)



acc_ptn1 = accuracy_score(yf_test,pred_ptn1)*100



acc_ptn1
#Decision Tree Classifier



from sklearn.tree import DecisionTreeClassifier



model_dtc1= DecisionTreeClassifier(random_state=1)



model_dtc1.fit(xf_train, yf_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_dtc1 = model_dtc1.predict(xf_test)



acc_dtc1 = accuracy_score(yf_test,pred_dtc1)*100



acc_dtc1
#Lightgbm Classifier



import lightgbm as lgb



model_lgb1=lgb.LGBMClassifier()



model_lgb1.fit(xf_train, yf_train)



# Use Prdict method to predict the loan status and calculate accuracy score in the validation set.



pred_lgb1 = model_lgb1.predict(xf_test)



acc_lgb1 = accuracy_score(yf_test,pred_lgb1)*100



acc_lgb1
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron',  

              'Decision Tree','LGBMClassifier'],

    'Score': [acc_svc1, acc_knn1, acc_log1, 

              acc_rfc1, acc_gnb1, acc_ptn1, 

             acc_dtc1,acc_lgb1]})

models.sort_values(by='Score', ascending=False)
test_ptn = test_fe[selected_columns] 



test_ptn.head()
#Final Submission



pred_ptn = model_ptn1.predict(test_ptn)



submission['Loan_Status'] = pred_ptn



submission['Loan_ID']     = test_df['Loan_ID']



submission['Loan_Status'].replace(0, 'N',inplace=True)



submission['Loan_Status'].replace(1, 'Y',inplace=True)



pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('perceptron.csv')



submission.head()
#Cross Validation using Logistic Regression



from sklearn.model_selection import StratifiedKFold 

from sklearn.model_selection import cross_val_score





skfold = StratifiedKFold (n_splits=10,shuffle=True, random_state= 1)



Log_skf = LogisticRegression()



Log_skf1 = cross_val_score(Log_skf, X_fe[selected_columns], y_fe, cv=skfold)



print(Log_skf1)



acc_log2 = Log_skf1.mean()*100.0



acc_log2 
#Cross Validation using SVC



skfold = StratifiedKFold (n_splits=5,shuffle=True, random_state= 1)



svc_sk = SVC(gamma='auto')



svc_sk1 = cross_val_score(svc_sk, X_fe[selected_columns], y_fe, cv=skfold)



print(svc_sk1)



acc_svc2 = svc_sk1.mean()*100.0



acc_svc2 
#Cross Validation using KNN



skfold = StratifiedKFold (n_splits=10,shuffle=True, random_state= 1)



knn_sk = KNeighborsClassifier(n_neighbors = 3)



knn_sk1 = cross_val_score(knn_sk, X_fe[selected_columns], y_fe, cv=skfold)



print(knn_sk1)



acc_knn2 = knn_sk1.mean()*100.0



acc_knn2

#Cross Validation using RandomForest Classifier



skfold = StratifiedKFold (n_splits=10,shuffle=False, random_state= None)



rfc_sk = RandomForestClassifier(n_estimators=100,random_state = 1)



rfc_sk1 = cross_val_score(rfc_sk, X_fe[selected_columns], y_fe, cv=skfold)



print(rfc_sk1)



acc_rfc2 = rfc_sk1.mean()*100.0



acc_rfc2



#Cross Validation using GaussianNB



skfold = StratifiedKFold (n_splits=5,shuffle=False, random_state= None)



gnb_sk = GaussianNB()



gnb_sk1 = cross_val_score(gnb_sk, X_fe[selected_columns], y_fe, cv=skfold)



print(gnb_sk1)



acc_gnb2 = gnb_sk1.mean()*100.0



acc_gnb2

#Cross Validation using Perceptron



skfold = StratifiedKFold (n_splits=5,shuffle=True, random_state= None)



ptn_sk = Perceptron()



ptn_sk1 = cross_val_score(ptn_sk, X_fe[selected_columns], y_fe, cv=skfold)



print(ptn_sk1)



acc_ptn2 = ptn_sk1.mean()*100.0



acc_ptn2

#Cross Validation using DecisionTree Classifier



skfold = StratifiedKFold (n_splits=5,shuffle=True, random_state= None)



dt_sk = DecisionTreeClassifier(random_state=1)



dt_sk1 = cross_val_score(dt_sk, X_fe[selected_columns], y_fe, cv=skfold)



print(dt_sk1)



acc_dt2 = dt_sk1.mean()*100.0



acc_dt2
#Cross Validation using LGBMClassifier



skfold = StratifiedKFold (n_splits=10,shuffle=True, random_state= None)



lgb_sk = lgb.LGBMClassifier()



lgb_sk1 = cross_val_score(lgb_sk, X_fe[selected_columns], y_fe, cv=skfold)



print(lgb_sk1)



acc_lgb2 = lgb_sk1.mean()*100.0



acc_lgb2
#Model Validation



models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron',  

              'Decision Tree','LGBMClassifier'],

    'Mean Accuracy': [acc_svc2, acc_knn2, acc_log2, 

              acc_rfc2, acc_gnb2, acc_ptn2, 

             acc_dt2,acc_lgb2]})

models.sort_values(by='Mean Accuracy', ascending=False)