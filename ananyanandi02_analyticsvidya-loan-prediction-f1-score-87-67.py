import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
train = pd.read_csv('../input/analytics-vidhya-loan-prediction/train.csv')
train.head(5)
test = pd.read_csv('../input/analytics-vidhya-loan-prediction/test.csv')
test.head(5)
print(train.shape)
print(test.shape)
train.info()
train.isnull().sum()
test.isnull().sum()
test['Gender'].value_counts()
train['LoanAmount'].value_counts()
test['Dependents'].value_counts()
test['Credit_History'].value_counts()
train['Self_Employed'].value_counts()
train['Loan_Amount_Term'].value_counts()
train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0]) #imputing missing values for gender
test['Gender'] = test['Gender'].fillna(test['Gender'].mode()[0])
train['Married'] = train['Married'].fillna(train['Married'].mode()[0]) #imputing missing values for married
test['Married'] = test['Married'].fillna(test['Married'].mode()[0]) 
train['Dependents'] = train['Dependents'].fillna(train['Dependents'].mode()[0]) #imputing missing values for Dependents
test['Dependents'] = test['Dependents'].fillna(test['Dependents'].mode()[0]) 
train['Self_Employed'].fillna('No',inplace=True)
test['Self_Employed'].fillna('No',inplace=True)
train['Credit_History'] = train['Credit_History'].fillna(train['Credit_History'].mode()[0])
test['Credit_History'] = test['Credit_History'].fillna(test['Credit_History'].mode()[0])
train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].median())
test['LoanAmount'] = test['LoanAmount'].fillna(test['LoanAmount'].median())
train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].median())
test['Loan_Amount_Term'] = test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].median())
train.isnull().sum()
test.isnull().sum()
#Credit History 
train["Credit_History"]=train["Credit_History"].astype("object")
test["Credit_History"]=test["Credit_History"].astype("object")
train['Loan_Amount_Term']=train['Loan_Amount_Term'].astype(int)
df_chi=train.copy()
df_chi.head()
#Assigning levels to the categories
lis = []
for i in range(0, df_chi.shape[1]):
    if(df_chi.iloc[:,i].dtypes == 'object'):
        df_chi.iloc[:,i] = pd.Categorical(df_chi.iloc[:,i])
        df_chi.iloc[:,i] = df_chi.iloc[:,i].cat.codes 
        lis.append(df_chi.columns[i])
cat_var=["Gender","Married","Dependents","Education","Self_Employed","Credit_History","Loan_Status"] 
catdf=df_chi[cat_var]
catdf.info()
from sklearn.feature_selection import chi2
n= 7
for i in range(0,6):
    X=catdf.iloc[:,i+1:n]
    y=catdf.iloc[:,i]
    chi_scores = chi2(X,y)
    p_values = pd.Series(chi_scores[1],index = X.columns)
    print("for",i)
    print(p_values)
    for j in range (0, len(p_values)):
        if (p_values[j]<0.05):
            print(p_values[j])
#Loan ID is dropped because it is not required
train=train.drop(["Gender"],axis=1)
test=test.drop(["Gender"],axis=1)
#gender is correlated with married and dependent
train=train.drop(["Loan_ID"],axis=1)
test=test.drop(["Loan_ID"],axis=1)
train=train.drop(["Dependents"],axis=1)
test=test.drop(["Dependents"],axis=1)
train.info()
df_final= pd.get_dummies(train[["Married","Education","Self_Employed","Credit_History","Property_Area"]], drop_first=True, dtype=bool)
df_final.info()
train=train.drop(["Married","Education","Self_Employed","Credit_History","Property_Area"],axis=1)
mergedDf = train.merge(df_final, left_index=True, right_index=True)
#Standardizing the numerical variables
mergedDf["ApplicantIncome"]= (mergedDf["ApplicantIncome"] - mergedDf["ApplicantIncome"].mean())/mergedDf["ApplicantIncome"].std()
mergedDf["CoapplicantIncome"]= (mergedDf["CoapplicantIncome"] - mergedDf["CoapplicantIncome"].mean())/mergedDf["CoapplicantIncome"].std()
mergedDf["LoanAmount"]= (mergedDf["LoanAmount"] - mergedDf["LoanAmount"].mean())/mergedDf["LoanAmount"].std()
mergedDf["Loan_Amount_Term"]= (mergedDf["Loan_Amount_Term"] - mergedDf["Loan_Amount_Term"].mean())/mergedDf["Loan_Amount_Term"].std()

lis = []
for i in range(0, mergedDf.shape[1]):
    if(mergedDf.iloc[:,i].dtypes == 'object'):
        mergedDf.iloc[:,i] = pd.Categorical(mergedDf.iloc[:,i])
        mergedDf.iloc[:,i] = mergedDf.iloc[:,i].cat.codes 
        lis.append(mergedDf.columns[i])
mergedDf.tail(5)
mergedDf.info()
X=mergedDf.drop(["Loan_Status"],axis=1)
Y=mergedDf["Loan_Status"]
Y=Y.astype(int)
x=np.array(X)
y=np.array(Y)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#Naive Bayes
parameters = {'priors':[[0.01, 0.99],[0.1, 0.9], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7],[0.35, 0.65], [0.4, 0.6],[0.45,0.55],[0.5,0.5],[0.55,0.45],[0.6,0.4]]}
nb = GridSearchCV(GaussianNB(), parameters, scoring = 'f1', n_jobs=-1)
nb.fit(x, y)
scores = cross_val_score(nb, x, y, cv=5,scoring = 'f1')
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
from sklearn import tree
from sklearn.pipeline import Pipeline
decisiontree = tree.DecisionTreeClassifier()
pipe = Pipeline(steps=[('decisiontree', decisiontree)])
criterion = ['gini', 'entropy']
max_depth = list(range(1,20))
parameters = dict(decisiontree__criterion=criterion,decisiontree__max_depth=max_depth)
dt = GridSearchCV(pipe, parameters,cv=5,scoring="f1", n_jobs=-1)
dt.fit(x,y)
scores = cross_val_score(dt, x, y,scoring="f1", cv=5)
print("DT Cross validation f1 score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
number_of_neighbors = range(1,20)
params = {'n_neighbors':number_of_neighbors}
knn = KNeighborsClassifier()
knnmodel = GridSearchCV(knn, params, cv=5,scoring="f1", n_jobs=-1)
knnmodel.fit(x,y)
scores = cross_val_score(knnmodel, x, y,scoring="f1", cv=5)
print("KNN Cross validation f1 score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
df_cat= pd.get_dummies(test[["Married","Education","Self_Employed","Credit_History","Property_Area"]], drop_first=True, dtype=bool)
df_cat.head()
test=test.drop(["Married","Education","Self_Employed","Credit_History","Property_Area"],axis=1)
finaldDf = test.merge(df_cat, left_index=True, right_index=True)
finaldDf.info()
finaldDf['Loan_Amount_Term']=finaldDf['Loan_Amount_Term'].astype(int)
finaldDf["ApplicantIncome"]= (finaldDf["ApplicantIncome"] - finaldDf["ApplicantIncome"].mean())/finaldDf["ApplicantIncome"].std()
finaldDf["CoapplicantIncome"]= (finaldDf["CoapplicantIncome"] - finaldDf["CoapplicantIncome"].mean())/finaldDf["CoapplicantIncome"].std()
finaldDf["LoanAmount"]= (finaldDf["LoanAmount"] - finaldDf["LoanAmount"].mean())/finaldDf["LoanAmount"].std()
finaldDf["Loan_Amount_Term"]= (finaldDf["Loan_Amount_Term"] - finaldDf["Loan_Amount_Term"].mean())/finaldDf["Loan_Amount_Term"].std()
finaldDf.head()
x=np.array(finaldDf)
#Prediction for ouput variable
ypred=dt.predict(x)
test=pd.read_csv("../input/analytics-vidhya-loan-prediction/test.csv")
test["Loan_Status"]=ypred
dict = {1: 'Y', 0: 'N'} 
test['Loan_Status']= test['Loan_Status'].map(dict) 
test.columns
test=test.drop(['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area'],axis=1)
test.head()
test.to_csv("submission.csv")

