# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("../input/hackerearth/Train.csv")
train.head(10)
test=pd.read_csv("../input/hackerearth/Test.csv")
test.head(10)
train.columns
train.count()
train.describe()
train.info()
train.head(5)
import matplotlib.pyplot as plt



def bar_plot(variable):

    

    """

        input: variable ex:"Sex"

        output: bar plot & value count

    """

    #get future

    var= train[variable]

    #count number of categorical variable(value/sample)

    varValue=var.value_counts()

    

    #visualize

    

    plt.figure(figsize=(10,4))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    

    print("{}: \n {}".format(variable,varValue))

    
category1= ["Gender","Age","Education_Level","Relationship_Status","Hometown","Unit","Decision_skill_possess","Time_of_service"]



for c in category1:

    bar_plot(c)
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train[variable],bins=50)

    

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar =["Age"]



for n in numericVar:

    plot_hist(n)

 
print(train['Age'])
train.head(5)
train['Age'].unique()
train['Relationship_Status'].unique()
train['Unit'].unique()
train['Decision_skill_possess'].unique()
train['Compensation_and_Benefits'].unique()
train.head(2)
train[["Hometown","Attrition_rate"]].groupby(["Hometown"], as_index=False).mean().sort_values(by="Attrition_rate",ascending=False)
train[["Time_since_promotion","Attrition_rate"]].groupby(["Time_since_promotion"], as_index=False).mean().sort_values(by="Attrition_rate",ascending=False)
from collections import Counter 

def detect_outliers(df,features):

    outlier_indices=[]

    

    for c in features:

        # 1st quartile

        Q1=np.percentile(df[c],25)

        

        # 3rd quartile

        Q3=np.percentile(df[c],75)

        

        # IQR

        IQR= Q3-Q1

        

        # Outlier Step

        outlier_step= IQR * 1.5

        

        # Detect outlier and their indeces 

        outlier_list_col = df[(df[c]< Q1 - outlier_step)|( df[c] > Q3 + outlier_step)].index

        

        # Store indices 

        outlier_indices.extend(outlier_list_col)

    

    outliers_indices = Counter(outlier_indices)

    

    multiple_outliers = list(i for i , v in outliers_indices.items() if v>2 )

    

    return multiple_outliers



train.loc[detect_outliers(train,["Age","VAR1","VAR2","VAR3","Pay_Scale"])]

train.columns
train.columns[train.isnull().any()]
train.isnull().sum()
train.boxplot(column="VAR1",by="Gender")

plt.show()



train.boxplot(column="VAR2",by="Gender")

plt.show()



train.boxplot(column="VAR3",by="Gender")

plt.show()



train.boxplot(column="VAR4",by="Gender")

plt.show()



train.boxplot(column="VAR5",by="Gender")

plt.show()



train.boxplot(column="VAR6",by="Gender")

plt.show()



train.boxplot(column="VAR7",by="Gender")

plt.show()



train.boxplot(column="Attrition_rate",by="Gender")

plt.show()

import seaborn as sns
train[train["Age"].isnull()]
train[train["Age"].isnull()]
train["Age"]=train["Age"].fillna(train['Age'].median())



train['Age'].unique()
train[train["Age"].isnull()]
train[train["Time_of_service"].isnull()]


train["Time_of_service"]=train["Time_of_service"].fillna(train['Time_of_service'].median())
train.isnull().sum()
train[train["Work_Life_balance"].isnull()]


train["Work_Life_balance"]=train["Work_Life_balance"].fillna(train['Work_Life_balance'].median())
train.isnull().sum()
train["VAR2"]=train["VAR2"].fillna(train['VAR2'].median())

train["VAR4"]=train["VAR4"].fillna(train['VAR4'].median())

train["Pay_Scale"]=train["Pay_Scale"].fillna(train['Pay_Scale'].median())
train.isnull().sum()
train["Pay_Scale"]=train["Pay_Scale"].fillna(train["Pay_Scale"].median())
train.head()
train["Gender"]=[1 if i =="M" else 0 for i in train["Gender"]]
train['Relationship_Status'].unique()
train["Relationship_Status"]=[1 if i =="Married" else 0 for i in train["Relationship_Status"]]
train['Hometown'].unique()
from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column 'species'. 

train['Hometown']= label_encoder.fit_transform(train['Hometown']) 

  

train['Hometown'].unique() 
train.head(2)
train['Unit']= label_encoder.fit_transform(train['Unit']) 

train['Decision_skill_possess']= label_encoder.fit_transform(train['Decision_skill_possess']) 

train['Compensation_and_Benefits']= label_encoder.fit_transform(train['Compensation_and_Benefits']) 

train.head(5)
train.info()
train.head(5)
id=[]

for i in list(train.Employee_ID):

    if not i.isdigit():

        id.append(i.replace("EID_","").strip().split(" ")[0])

train["Employee_ID"]=id
train.head(5)
train.info()


train["Employee_ID"] = train["Employee_ID"].astype(str).astype(int)

print(train.dtypes)
train.head(5)
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_len=len(train)
test=train[:train_len]

test.drop(labels=["Attrition_rate"],axis=1,inplace=True)

test.head(10)
train_df=train[:train_len]

X_train=train_df.drop(labels=["Attrition_rate"],axis=1)

y_train=train_df["Attrition_rate"]

X_train,X_test ,y_train,y_test=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))



print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train, y_train)

'''LinearRegression()'''

from sklearn.metrics import mean_squared_error

def rmse(y_test,y_pred):

      return np.sqrt(mean_squared_error(y_test,y_pred))



y_pred = lr.predict(X_test)

print("Linear Regressor score on testing set: ", rmse(y_test, y_pred))



Score = 100 * max(0 , 1 - rmse(y_test, y_pred))

print('Accuracy',Score)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



model = XGBRegressor(n_estimators=1000,learning_rate=0.05,n_jobs=-1)

model.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error

def rmse(y_test,y_pred):

      return np.sqrt(mean_squared_error(y_test,y_pred))



y_pred = model.predict(X_test)

print("XGBoost score on testing set: ", rmse(y_test, y_pred))
Score = 100 * max(0 , 1 - rmse(y_test, y_pred))

print('Accuracy',Score)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

  

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)





from sklearn.linear_model import Lasso

best_alpha = 0.0099



regr = Lasso(alpha=best_alpha, max_iter=50000)

regr.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

def rmse(y_test,y_pred):

      return np.sqrt(mean_squared_error(y_test,y_pred))

y_pred = regr.predict(X_test)

print("Lasso score on testing set: ", rmse(y_test, y_pred))
Score = 100 * max(0 , 1 - rmse(y_test, y_pred))

print('Accuracy',Score)