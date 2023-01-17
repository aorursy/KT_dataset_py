import pandas as pd

df=pd.read_csv("E:\\data\\train.csv")

df.head()
df.describe()
X=df.drop(["Loan_ID","Loan_Status"],axis=1)

X.shape
y=df["Loan_Status"]

print(y)
X.isnull().sum()
X["Gender"].value_counts()
X["Gender"].fillna("Male",inplace=True)
X.isnull().sum()
X["Married"].value_counts()
X.isnull().sum()
X["Married"].fillna("Yes",inplace=True)
X["Married"].value_counts()
X["Dependents"].value_counts()
X["Dependents"].fillna("0",inplace=True)
X.isnull().sum()
X["Self_Employed"].value_counts()
X["Self_Employed"].fillna("No",inplace=True)
X.isnull().sum()
X["LoanAmount"].value_counts()
X["LoanAmount"].fillna(X["LoanAmount"].mean(),inplace=True)
X["LoanAmount"].value_counts()
X.isnull().sum()
X["Loan_Amount_Term"].value_counts()
X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].mean(),inplace=True)



X['Credit_History'].fillna(X['Credit_History'].mean(),inplace=True)



X.isnull().sum()
X=pd.get_dummies(X)

X.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=30)
X_test.shape
X_train.shape
y_train.shape
y_test.shape
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr.fit(X_train,y_train)
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier()
dtf.fit(X_train,y_train)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_predict=lr.predict(X_test)

y_predict1=svc.predict(X_test)

y_predict2=dtf.predict(X_test)

y_predict3=nb.predict(X_test)

y_predict4=knn.predict(X_test)

df1=pd.DataFrame({'Actual':y_test,'Predicted_LR':y_predict,'Predicted_svc':y_predict1,'Predicted_dtr':y_predict2,'Predicted_nb':y_predict3,'Predicted_knn':y_predict4 })

df1.to_csv("Day16_Output.csv")
print(lr.score(X_test,y_test))

print(svc.score(X_test,y_test))

print(dtf.score(X_test,y_test))

print(nb.score(X_test,y_test))

print(knn.score(X_test,y_test))
gender=input("What is your gender:")

married=input("Married:")

dependents=input("dependents value:")

Education=input("enter your education")

SelfEmployed=input("Self Employed:")

Applicantincome=int(input("enter applicant income"))

coapplicantincome=int(input("enter co applicant income:"))

loanamount=int(input("enter loan amount:"))

loanamountterm=int(input("enter loan amount term:"))

credithistory=int(input("enter credit history:"))

propertyarea=input("enter property area:")

data = [[gender,married,dependents,Education,SelfEmployed,Applicantincome,coapplicantincome,loanamount,loanamountterm,credithistory,propertyarea]]

newdf = pd.DataFrame(data, columns = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area'])

newdf.head()
newdf=pd.get_dummies(newdf)

newdf.head()
newdf.head()
X_train.columns
missing_col=set(X_train.columns)-set(newdf.columns)

print(missing_col)
for c in missing_col:

    newdf[c]=0
newdf=newdf[X_train.columns]
yp=nb.predict(newdf)

print(yp)
if (yp[0]=='Y'):

    print("Your Loan is approved, Please contact at HDFC Bank Any Branch for further processing")

else:

    print("Sorry ! Your Loan is not approved")