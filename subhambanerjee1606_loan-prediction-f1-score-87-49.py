import pandas as pd
import numpy as np
from sklearn import preprocessing 
from fancyimpute import KNN   
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.pipeline import Pipeline
train=pd.read_csv("../input/analytics-vidhya-loan-prediction/train.csv")
test=pd.read_csv("../input/analytics-vidhya-loan-prediction/test.csv")
train.head(2)
#Loan ID is dropped because it is not required
train=train.drop(["Loan_ID"],axis=1)
test=test.drop(["Loan_ID"],axis=1)
print(train.isnull().sum())
train.info()
#Credit History and Loan Amount Term are categorical variables
train["Credit_History"]=train["Credit_History"].astype("object")
#KNN imputation
#Assigning levels to the categories
lis = []
for i in range(0, train.shape[1]):
    if(train.iloc[:,i].dtypes == 'object'):
        train.iloc[:,i] = pd.Categorical(train.iloc[:,i])
        train.iloc[:,i] = train.iloc[:,i].cat.codes 
        train.iloc[:,i] = train.iloc[:,i].astype('object')
        lis.append(train.columns[i])
#replace -1 with NA to impute
for i in range(0, train.shape[1]):
    train.iloc[:,i] = train.iloc[:,i].replace(-1, np.nan) 
#Apply KNN imputation algorithm
train = pd.DataFrame(KNN(k = 3).fit_transform(train), columns = train.columns)
#Convert into proper datatypes
for i in lis:
    train.loc[:,i] = train.loc[:,i].round()
    train.loc[:,i] = train.loc[:,i].astype('object')
train.head()
#Checking correlation between continuous variable
numvar=["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
df_corr = train.loc[:,numvar]
sns.heatmap(df_corr.corr(), annot = True, fmt='.2g',cmap= 'coolwarm')
#Standardizing the numerical variables
train["ApplicantIncome"]= (train["ApplicantIncome"] - train["ApplicantIncome"].mean())/train["ApplicantIncome"].std()
train["CoapplicantIncome"]= (train["CoapplicantIncome"] - train["CoapplicantIncome"].mean())/train["CoapplicantIncome"].std()
train["LoanAmount"]= (train["LoanAmount"] - train["LoanAmount"].mean())/train["LoanAmount"].std()
train["Loan_Amount_Term"]= (train["Loan_Amount_Term"] - train["Loan_Amount_Term"].mean())/train["Loan_Amount_Term"].std()
X=train.drop(["Loan_Status"],axis=1)
Y=train["Loan_Status"]
Y=Y.astype(int)
x=np.array(X)
y=np.array(Y)
decisiontree = tree.DecisionTreeClassifier()
pipe = Pipeline(steps=[('decisiontree', decisiontree)])
criterion = ['gini', 'entropy']
max_depth = list(range(1,20))
parameters = dict(decisiontree__criterion=criterion,decisiontree__max_depth=max_depth)
dt = GridSearchCV(pipe, parameters,cv=5,scoring="f1", n_jobs=-1)
dt.fit(x,y)
scores = cross_val_score(dt, x, y,scoring="f1", cv=5)
print("DT Cross validation f1 score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
#Naive Bayes
parameters = {'priors':[[0.01, 0.99],[0.1, 0.9], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7],[0.35, 0.65], [0.4, 0.6],[0.45,0.55],[0.5,0.5],[0.55,0.45],[0.6,0.4]]}
nb = GridSearchCV(GaussianNB(), parameters, scoring = 'f1', n_jobs=-1)
nb.fit(x, y)
scores = cross_val_score(nb, x, y,scoring = 'f1', cv=5)
print("NB Cross validation F1 score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
number_of_neighbors = range(1,20)
params = {'n_neighbors':number_of_neighbors}
knn = KNeighborsClassifier()
knnmodel = GridSearchCV(knn, params, cv=5,scoring="f1", n_jobs=-1)
knnmodel.fit(x,y)
scores = cross_val_score(knnmodel, x, y,scoring="f1", cv=5)
print("KNN Cross validation f1 score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
test["Credit_History"]=test["Credit_History"].astype("object")
#KNN imputation
#Assigning levels to the categories
lis = []
for i in range(0, test.shape[1]):
    if(test.iloc[:,i].dtypes == 'object'):
        test.iloc[:,i] = pd.Categorical(test.iloc[:,i])
        test.iloc[:,i] = test.iloc[:,i].cat.codes 
        test.iloc[:,i] = test.iloc[:,i].astype('object')
        lis.append(test.columns[i])
#replace -1 with NA to impute
for i in range(0, test.shape[1]):
    test.iloc[:,i] = test.iloc[:,i].replace(-1, np.nan) 
#Apply KNN imputation algorithm
test = pd.DataFrame(KNN(k = 3).fit_transform(test), columns = test.columns)
#Convert into proper datatypes
for i in lis:
    test.loc[:,i] = test.loc[:,i].round()
    test.loc[:,i] = test.loc[:,i].astype('object')
test["ApplicantIncome"]= (test["ApplicantIncome"] - test["ApplicantIncome"].mean())/test["ApplicantIncome"].std()
test["CoapplicantIncome"]= (test["CoapplicantIncome"] - test["CoapplicantIncome"].mean())/test["CoapplicantIncome"].std()
test["LoanAmount"]= (test["LoanAmount"] - test["LoanAmount"].mean())/test["LoanAmount"].std()
test["Loan_Amount_Term"]= (test["Loan_Amount_Term"] - test["Loan_Amount_Term"].mean())/test["Loan_Amount_Term"].std()
x=np.array(test)
#Prediction
ypred=dt.predict(x)
test=pd.read_csv("../input/analytics-vidhya-loan-prediction/test.csv")
test["Loan_Status"]=ypred
dict = {0 : 'N', 1: 'Y'} 
test['Loan_Status']= test['Loan_Status'].map(dict) 
test.columns
test=test.drop(['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area'],axis=1)
test.head()
test.to_csv("submissionfinal.csv")
