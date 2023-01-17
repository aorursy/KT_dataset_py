import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





sns.set(style = 'darkgrid')
trainfile = "../input/train-file.csv"

testfile = "../input/test-file.csv"



df_train = pd.read_csv(trainfile)

df_test = pd.read_csv(testfile)



# df_train.shape

# print("--------------------------------------" )

df_train.head(5)
df_train.columns
print("The training dataset has: " + str(df_train.shape[0]) + " rows " + str(df_train.shape[1]) + " columns") 
# check missing values for both datasets



print("The training dataset has null value: ")

print(df_train.isnull().sum())

print("---------------------------")

print("The testing dataset has null value: ")

print(df_test.isnull().sum())
# check gender count 

sns.countplot(x = 'Gender', hue = 'Married', data = df_train)
sns.countplot(x = 'Gender', hue = 'Education', data = df_train)
sns.catplot(x = 'Gender', y = 'ApplicantIncome', hue = 'Dependents', data = df_train, kind = 'bar')
sns.relplot(x="ApplicantIncome", y="CoapplicantIncome", col="Gender", data=df_train)
grid = sns.FacetGrid(df_train, col = 'Education', row = "Self_Employed", size = 2.8, aspect = 1.5)

grid.map(sns.barplot, "Gender", "ApplicantIncome", alpha = .5, ci = None)

grid.add_legend()
sns.countplot(x = 'Credit_History', hue = "Gender", data = df_train)
df_train['Dependents'].unique()
df_train.describe()
df_train = df_train.drop('Loan_ID', axis = 1)

df_test = df_test.drop('Loan_ID', axis = 1)



print("\nThe training dataset has column: ", df_train.shape[1])

print("\nThe testing dataset has column: ", df_test.shape[1])
combine = [df_train, df_test]
for dataset in combine:

    dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].dropna().mode().values[0])

    dataset["Dependents"] = dataset['Dependents'].fillna(dataset['Dependents'].dropna().mode().values[0])

    dataset["Self_Employed"] = dataset['Self_Employed'].fillna(dataset['Self_Employed'].dropna().mode().values[0])

    dataset["LoanAmount"] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].dropna().mean())

    dataset["Loan_Amount_Term"] = dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].dropna().mean())

    dataset["Credit_History"] = dataset['Credit_History'].fillna(dataset['Credit_History'].dropna().mode().values[0])



print("The Training data ")

print(df_train.isnull().sum())

print("---------------------------")

print(df_test.isnull().sum())
df_train['Married'] = df_train['Married'].fillna(df_train['Married'].dropna().mode().values[0])
# check missing value for the training dataset 

df_train.isnull().sum()
df_train.head(3)

df_train['Property_Area'].unique()
df_train.dtypes
sex = {"Male": 1, "Female": 0}

Married_code = {"Yes": 1, "No": 0}

Edu_code = {"Graduate": 1, "Not Graduate": 0}

SelfEmployed_code = {"Yes": 1, "No": 0}

Prop_code = {"Urban": 1, "Semiurban": 2, "Rural": 3}

Loanstatus = {"Y": 1, "N": 0}



for dataset in combine:

    dataset["Gender"] = dataset["Gender"].map(sex)

    dataset["Married"] = dataset["Married"].map(Married_code)

    dataset["Education"] = dataset["Education"].map(Edu_code)

    dataset["Self_Employed"] = dataset["Self_Employed"].map(SelfEmployed_code)

    dataset["Property_Area"] = dataset["Property_Area"].map(Prop_code)

    



df_train['Loan_Status'] = df_train['Loan_Status'].map(Loanstatus)

df_train.head()

    
for dataset in combine:

    dataset["Credit_History"] = dataset["Credit_History"].astype(int)

    

# print(df_train.dtypes)

# print("-------------------------")

# print(df_test.dtypes)
for dataset in combine:

    dataset['Dependents'] = dataset['Dependents'].replace("3+", 3)



print(df_train['Dependents'].unique())
# seperate variables and target

X = df_train.drop("Loan_Status", axis = 1)

y = df_train['Loan_Status']
# load machine learning package 

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import accuracy_score

LG_score = cross_val_score(LogisticRegression(), X, y)

DT_score = cross_val_score(DecisionTreeClassifier(), X, y)



LG_mean = LG_score.mean()

DT_mean = DT_score.mean()



print("\nLG model score mean: ", LG_mean)

print("DT model score mean: ", DT_mean)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
lg_model = LogisticRegression()

lg_model.fit(X_train, y_train)

lg_predict = lg_model.predict(X_test)



lg_score = accuracy_score(y_test, lg_predict)



print("The Accuracy Score for LG model: ", lg_score)
DT_model = DecisionTreeClassifier()

DT_model.fit(X_train, y_train)

DT_model_predict = DT_model.predict(X_test)



DT_model_score = accuracy_score(y_test, DT_model_predict)

print("The Accuracy Score for LG model: ", DT_model_score)
