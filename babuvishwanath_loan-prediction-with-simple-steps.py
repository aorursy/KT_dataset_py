import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, fbeta_score, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#NAIVE_BAYES MODEL
from sklearn.naive_bayes import GaussianNB

#SVC 
from sklearn.svm import SVC

#XGBOOST
from xgboost import XGBClassifier
import pandas as pd

from sklearn.metrics import classification_report
df1 = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
df1.info()
df1
df2 = pd.read_csv("../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv")
df2.info()
df2
df = df1.append(df2)
df.info()
df
df.dtypes
df["Credit_History"].value_counts()
df["Loan_Status"].value_counts()
df.columns = map(str.lower, df.columns)
df["credit_history"]=df["credit_history"].replace(np.nan,1.0)
df["credit_history"]= df["credit_history"].astype(str)
df.isna().sum()
#MODE IMPUTATION FOR LOAN_STATUS 
df["loan_status"] = df["loan_status"].replace(np.nan,"Y")
df["loan_status"]= df["loan_status"].replace("Y",1).astype(str)
df["loan_status"]= df["loan_status"].replace("N",0).astype(str)
#REMOVOING THE LOAN_ID AS IT IS NOT AN IMPORTANT VARIABLE
df.drop("loan_id",axis=1,inplace=True)
df.duplicated().any()
df.shape
df.drop_duplicates(keep=False, inplace=True)
df.duplicated().any()
df.shape
#user-defined function for knowing the number of cat and num varriables in a data-set
def cat_num(df):
    total = 0
    cat = 0
    num = 0
    for col in df.columns.values:
        if df[col].dtype == "object":
            cat = cat+1
        else:
            num=num+1
    print("numerical:",num)
    print("categorical:",cat)
cat_num(df) 

#TOTALLY 4 NUMERICAL AND 8 CATEGORICAL VARIABLES ARE THERE 
#lets split the data-frame into Numerical and categorical variable
category = [col for col in df.columns.values if df[col].dtype == 'object']

# CATEGORICAL
data_cat = df[category]

#Numerical variable
data_num = df.drop(category,axis =1)
data_num.isna().sum()
#LOAN_AMOUNT
amt = data_num["loanamount"]
amtfil = amt.fillna(amt.median())
data_num["loanamount"] = amtfil

#LOAN_AMOUNT_TERM
amt1 = data_num["loan_amount_term"]
amt1fil = amt1.fillna(amt1.median())
data_num["loan_amount_term"] = amt1fil
data_num.isna().sum()
data_cat.info()
data_cat.isna().sum()
import warnings
warnings.filterwarnings("ignore")
data_cat["gender"].value_counts()
data_cat["gender"]= data_cat["gender"].replace(np.nan,"Male")

data_cat["self_employed"]=data_cat["self_employed"].replace(np.nan,"No")

data_cat["dependents"]=data_cat["dependents"].replace(np.nan,0)

data_cat["married"]= data_cat["married"].replace(np.nan,"Yes")
data_cat.isna().sum()
#OUTLIERS CHECK FOR Numerical data
for i in data_num.columns:
    print(i)
    sns.set(style="whitegrid")
    sns.boxplot(data_num[i])
    plt.show()
    
#Categorical data count plot percentage
for i in data_cat.columns:
    print(i)
    total = float(len(data_cat))
    plt.figure(figsize=(8,10))
    sns.set(style="whitegrid")
    ax = sns.countplot(data_cat[i])
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center") 
    plt.show()
df["gender"].value_counts()
# let's look at the target percentage

plt.figure(figsize=(8,6))
sns.countplot(df['loan_status']);

print('The percentage of Y class : %.2f' % (df['loan_status'].value_counts()[1] / len(df)))
print('The percentage of N class : %.2f' % (df['loan_status'].value_counts()[0] / len(df)))

# We can consider it as imbalanced data, but for now i will not
#Credit_History

grid = sns.FacetGrid(df,col='loan_status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'credit_history');

# we didn't give a loan for most people who got Credit History = 0
# but we did give a loan for most of people who got Credit History = 1
# so we can say if you got Credit History = 1 , you will have better chance to get a loan

# important feature
# Gender

grid = sns.FacetGrid(df,col='loan_status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'gender');

# most males got loan and most females got one too so (No pattern)

# i think it's not so important feature...
# Married
plt.figure(figsize=(15,5))
sns.countplot(x='married', hue='loan_status', data=df);

# most people who get married did get a loan
# if you'r married then you have better chance to get a loan :)

# Dependents

plt.figure(figsize=(15,5))
sns.countplot(x='dependents', hue='loan_status', data=df);

# first if Dependents = 0 , we got higher chance to get a loan ((very hight chance))

# Education

grid = sns.FacetGrid(df,col='loan_status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'education');

# If you are graduated or not, you will get almost the same chance to get a loan (No pattern)
# Here you can see that most people did graduated, and most of them got a loan
# on the other hand, most of people who did't graduate also got a loan, but with less percentage from people who graduated

# Self_Employed

grid = sns.FacetGrid(df,col='loan_status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'self_employed')
df.groupby('loan_status').median() # median because Not affected with outliers

# we can see that when we got low median in CoapplicantInocme we got Loan_Status = N
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
import warnings
warnings.filterwarnings("ignore")

data_cat["loan_status"]=data_cat["loan_status"].replace("1","Y")
data_cat["loan_status"]=data_cat["loan_status"].replace("0","N")
data_cat
# transform the target column
import warnings
warnings.filterwarnings("ignore")
target_values = {'Y': 1 , 'N' : 0}

target = data_cat['loan_status']
data_cat.drop('loan_status', axis=1, inplace=True)

target = target.map(target_values)
#Label Encoding
from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()
import warnings
warnings.filterwarnings("ignore")
data_cat["gender"]=le.fit_transform(data_cat["gender"])

data_cat["married"]=le.fit_transform(data_cat["married"])

data_cat["education"]=le.fit_transform(data_cat["education"])

data_cat["self_employed"]=le.fit_transform(data_cat["self_employed"])

data_cat["property_area"]=le.fit_transform(data_cat["property_area"])
data_cat.drop("dependents",axis=1,inplace=True)
data_num.skew()
for col in data_num.columns:
    data_num[col] = (data_num[col]-data_num[col].min())/(data_num[col].max() - data_num[col].min())
    
data_num.head()
data= pd.concat([data_num, data_cat],axis=1)
data
X = data#independent variable

y = target #dependant variable
#train and test data split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)

print(X_train.shape, X_test.shape)
#NOTE:
#.values will store the values in the form of array
#if you not give x will store the values in series
#Random-Forest Model
model = RandomForestClassifier(n_estimators = 100, random_state = 42).fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))
#StratifiedShuffleSplit to split the data 

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train, test in sss.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    
print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)

#FOR-NOW JUST TAKING THE THREE VARIABLES
models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)
}
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score

def loss(y_true, y_pred, retu=False):
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    if retu:
        return pre, rec, f1, loss, acc
    else:
        print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' % (pre, rec, f1, loss, acc))
# train_eval_train

def train_eval_train(models, X, y):
    for name, model in models.items():
        print(name,':')
        model.fit(X, y)
        loss(y, model.predict(X))
        print('-'*30)
        
train_eval_train(models, X_train, y_train)

# we can see that best model is LogisticRegression at least for now, SVC is just memorizing the data so it is overfitting .

#Stratified K-fold corss validation
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

def train_eval_cross(models, X, y, folds):
    
    X = pd.DataFrame(X) 
    y = pd.DataFrame(y)
    idx = [' pre', ' rec', ' f1', ' loss', ' acc']
    for name, model in models.items():
        ls = []
        print(name,':')

        for train, test in folds.split(X, y):
            model.fit(X.iloc[train], y.iloc[train]) 
            y_pred = model.predict(X.iloc[test]) 
            ls.append(loss(y.iloc[test], y_pred, retu=True))
        print(pd.DataFrame(np.array(ls).mean(axis=0), index=idx)[0])  
        print('-'*30)
        
train_eval_cross(models, X_train, y_train, skf)
