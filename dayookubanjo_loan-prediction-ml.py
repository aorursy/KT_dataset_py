#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline
#import data
data = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
#preview top five rows in the dataset
data.head()
#check the number of row and columns in the imported data
data.shape   #614 rows and 13 columns
#check the description of the data such as mean,count, min, max for each numeric cell
data.describe()
data["Credit_History"].median() #median value of 1
#check if Credit_History is in % or 0 and 1 values
data.groupby(["Credit_History"])["Loan_ID"].count() 
#check the number of null records per column
data.isnull().sum()

#remove the Loan_ID column as it's not needed as an independent variable for the model. 
#it's just an ID to uniquely identify each row and does not describe the data in anyway.
data = data.drop(columns = ("Loan_ID") )
data.head()
#Drop null values in all the categorical columns
data = data.dropna(subset = ["Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area"])
data.isnull().sum()
#Fill null values in columns with continous data with the mean of each column
data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].mean())
data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mean())
data.isnull().sum()
#Loan Status by Marital Status
sb.countplot(x=data["Married"], hue = data["Loan_Status"], data = data)
#LoanAmount distribution
data["LoanAmount"].plot.hist(data["LoanAmount"])
fig = data.groupby("Dependents")["Loan_Status"].count().plot.bar(color = "red")
fig.set_ylabel('Count')
fig.set_title("Count of loans requested by No of Dependents")



#Check for correlation between all numeric columns
plt.title('Correlation Matrix')
sb.heatmap(data.corr(),annot=True)
#this scatter plot also shows the correlation above.
data.plot.scatter("ApplicantIncome", "LoanAmount", color = "blue")
data.groupby(["Gender"])["Loan_Status"].count() #Female and Male
data.groupby(["Married"])["Loan_Status"].count() #Yes and No
data.groupby(["Dependents"])["Loan_Status"].count() #0, 1, 2, 3+ and Male
data.groupby(["Education"])["Loan_Status"].count() #Graduate and Non Graduate
data.groupby(["Self_Employed"])["Loan_Status"].count() #Yes or No
data.groupby(["Loan_Status"])["Loan_Status"].count() #Y or N
data.groupby(["Property_Area"])["Loan_Status"].count() #Rural, Semiurban or Urban
data["Gender"] = data["Gender"].replace(["Female","Male"], [0, 1])
data["Married"] = data["Married"].replace(["No","Yes"], [0, 1])
data["Dependents"] = data["Dependents"].replace(["0","1","2","3+"], [0, 1,2,3])
data["Education"] = data["Education"].replace(["Not Graduate","Graduate"], [0, 1])
data["Self_Employed"] = data["Self_Employed"].replace(["No","Yes"], [0, 1])
data["Loan_Status"] = data["Loan_Status"].replace(["N","Y"], [0, 1])
data["Property_Area"] = data["Property_Area"].replace(["Rural","Semiurban", "Urban"], [0, 1, 2])
data.head()
#check data types of all the columns to make sure they are all numeric (float or integer)
data.dtypes
fig, ax = plt.subplots(figsize=(10,5))  
sb.heatmap(data.corr(),annot = True, ax=ax)


#Import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

#Define x and y (independent variable(s) and dependent variable respectively)
y = pd.DataFrame(data.iloc[:,11:])
x = pd.DataFrame(data.iloc[:,0:11])

#Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state = 1)

#Initialize and fit Logistic Regression Model
classifier = LogisticRegression(max_iter = 10000)
classifier.fit(x_train,y_train.values.ravel())
#Pass the test data (x_test) to the model to predict y
y_pred = classifier.predict(x_test)
y_pred
#score your model
classifier.score(x_train,y_train)
confusion_matrix(y_test,y_pred)
#check the share of total predictions that were accurate (possitive and negative)
accuracy_score(y_test,y_pred)
#check the share of total positive prediction that were accurate
precision_score(y_test,y_pred)
#checks the share of all the positives in the test data that the model was able to accurately predict
recall_score(y_test,y_pred)
#weighted average of the precision score and recall score
f1_score(y_test,y_pred)
#Initialize and fit Decision Tree Model
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train,y_train)
#Pass the test data (x_test) to the model to predict y
dt_y_pred = dt_classifier.predict(x_test)
dt_y_pred
#score your model
dt_classifier.score(x_train,y_train)
confusion_matrix(y_test,dt_y_pred)
#check the share of total predictions that were accurate (both positive and negative)
accuracy_score(y_test,dt_y_pred)
#check the share of total positive prediction that were accurate
precision_score(y_test,dt_y_pred)
#checks the share of all the positives in the test data that the model was able to accurately predict
recall_score(y_test,dt_y_pred)
#weighted average of the precision score and recall score
f1_score(y_test,dt_y_pred)
#Initialize and fit Random Forest Model
rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train,y_train.values.ravel())
#Pass the test data (x_test) to the model to predict y
rf_y_pred = rf_classifier.predict(x_test)
rf_y_pred
#score your model
rf_classifier.score(x_train,y_train)
confusion_matrix(y_test,rf_y_pred)
#check the share of total predictions that were accurate (both positive and negative)
accuracy_score(y_test,rf_y_pred)
#check the share of total positive prediction that were accurate
precision_score(y_test,rf_y_pred)
#checks the share of all the positives in the test data that the model was able to accurately predict
recall_score(y_test,rf_y_pred)
#weighted average of the precision score and recall score
f1_score(y_test,rf_y_pred)