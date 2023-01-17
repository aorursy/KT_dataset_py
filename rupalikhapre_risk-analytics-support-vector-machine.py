# Useful packages to load 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#These API customise display feature or behavioural feature
pd.set_option('display.max_columns',None)
pd.set_option('display.max_row',None)
# By running the below command it will read the file
train_data = pd.read_csv('../input/risk_analytics_train.csv')
# It is used to display atleast 5 rows of the dataset
train_data.head()
# The below ommmand will display the row and columns of the dataset
train_data.shape
# Check the null values in the dataset
print(train_data.isnull().sum())
# Create list of the column names
colname1 = ["Gender","Married","Dependents","Self_Employed","Loan_Amount_Term"]
colname1
# Below command is used replace the null value of the categorical variable column by mode
for x in colname1[:]:
     train_data[x].fillna(train_data[x].mode()[0],inplace = True)
# Again check for the null value in the categorial variable column
print(train_data.isnull().sum())

# Imputing the mean value to the numerical variable column using the mean and check the null value in the nuerical variable column
train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(),inplace = True)
print(train_data.isnull().sum())

# Impute zero in the Credit History column for the missing value
train_data["Credit_History"].fillna(value = 0,inplace = True)
print(train_data.isnull().sum())

# It will display the datatypes of all the column of the dataset
train_data.dtypes

# Transforming the categorical data to numerical data
from sklearn import preprocessing
colname1 = ["Gender","Married","Education","Self_Employed","Property_Area","Loan_Status"]

le = {}

for x in colname1:
    le[x]= preprocessing.LabelEncoder()
    
for x in colname1:
    train_data[x] = le[x].fit_transform(train_data[x])
# Again check the datatype of the all the column of the dataset
train_data.dtypes
# By running the below command it will read the test file
test_data = pd.read_csv('../input/risk_analytics_test.csv')
# It is used to display atleast 5 rows of the test dataset
test_data.head()
# It dispaly the number of column and rows of the test dataset
test_data.shape

# Check the null value in the test dataset
print(test_data.isnull().sum())
# Create the list of the categorial column
colname2 = ["Gender","Dependents","Self_Employed","Loan_Amount_Term"]
colname2
#Replace the null value of the categorical column by the mode value
for x in colname2[:]:
     test_data[x].fillna(test_data[x].mode()[0],inplace = True)
# Again check the null value
print(test_data.isnull().sum())
#imputing numerial missing data with mean value
test_data["LoanAmount"].fillna(test_data["LoanAmount"].mean(),inplace = True)
print(test_data.isnull().sum())
#impuatating values for the credir history column differently
test_data["Credit_History"].fillna(value = 0,inplace = True)
print(test_data.isnull().sum())
#Tranforming cartegorial data to numerical 
from sklearn import preprocessing
colname2 = ["Gender","Married","Education","Self_Employed","Property_Area"]

le = {}

for x in colname2:
    le[x]= preprocessing.LabelEncoder()
    
for x in colname2:
    test_data[x] = le[x].fit_transform(test_data[x])
# Separate the X_train and Y_train of the train dataset
X_train=train_data.values[:,1:-1]
Y_train=train_data.values[:,-1]
# Convert Y_train to int
Y_train=Y_train.astype(int)
#Convert the test dataset in the X_test
X_test = test_data.values[:,1:]
# Scale the dataset (for the normal distribution)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)


# Apply the SVC for the classification and predict the Y_test 
from sklearn.linear_model import LogisticRegression
svc_model= LogisticRegression()
svc_model.fit(X_train, Y_train)
Y_pred = svc_model.predict(X_test)
print(list(Y_pred))
# Convert the above predicted value in the list 
Y_pred_col = list(Y_pred)
# Again the read the test dataset and concate the Y predicted value
test_data = pd.read_csv('../input/risk_analytics_test.csv')
test_data["Y_Prediction"] = Y_pred_col
test_data.head()
# Cross validation to find the accuracy for the data 
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
#performing kfold_cross_validation
from sklearn import cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
