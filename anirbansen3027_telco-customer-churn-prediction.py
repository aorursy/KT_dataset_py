# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
# Importing the dataset

dataset = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#Data Cleaning

dataset.replace(' ', np.nan, inplace=True)

dataset.isna().sum() #11 values.Hence, replacing them with 0
nanrows=dataset[dataset.isna().any(axis=1)] 

nanrows.head()
#We can see that all have tenure 0 and total_charges missing.

#We can replace that with 0 as they never paid any chargegender

dataset['TotalCharges'] = dataset['TotalCharges'].replace(np.nan, 0).astype('float64')

dataset.isnull().sum()
#Summary Statistics

dataset.dtypes
list(set(dataset.SeniorCitizen)) # Senior Citizen is binary. So converting it
dataset['SeniorCitizen'] = dataset['SeniorCitizen'].astype('bool')
#Univariate analysis

#Categorical Variables - 1

a1 = plt.subplot2grid((3,3),(0,0))

a1 = sns.countplot(y='gender',data=dataset)



a1 = plt.subplot2grid((3,3),(0,1))

a1 = sns.countplot(y='SeniorCitizen',data=dataset)



a1 = plt.subplot2grid((3,3),(0,2))

a1 = sns.countplot(y='Partner',data=dataset)



a1 = plt.subplot2grid((3,3),(1,0))

a1 = sns.countplot(y='Dependents',data=dataset)



a1 = plt.subplot2grid((3,3),(1,1))

a1 = sns.countplot(y='PhoneService',data=dataset)



a1 = plt.subplot2grid((3,3),(1,2))

a1 = sns.countplot(y='MultipleLines',data=dataset)



a1 = plt.subplot2grid((3,3),(2,0))

a1 = sns.countplot(y='InternetService',data=dataset)



a1 = plt.subplot2grid((3,3),(2,1))

a1 = sns.countplot(y='OnlineSecurity',data=dataset)



a1 = plt.subplot2grid((3,3),(2,2))

a1 = sns.countplot(y='OnlineBackup',data=dataset)



plt.tight_layout()

plt.show()
b1 = plt.subplot2grid((3,3),(0,0))

b1 = sns.countplot(y='DeviceProtection',data=dataset)



b1 = plt.subplot2grid((3,3),(0,1))

b1 = sns.countplot(y='TechSupport',data=dataset)



b1 = plt.subplot2grid((3,3),(0,2))

b1 = sns.countplot(y='StreamingTV',data=dataset)



b1 = plt.subplot2grid((3,3),(1,0))

b1 = sns.countplot(y='StreamingMovies',data=dataset)



b1 = plt.subplot2grid((3,3),(1,1))

b1 = sns.countplot(y='Contract',data=dataset)



b1 = plt.subplot2grid((3,3),(1,2))

b1 = sns.countplot(y='PaperlessBilling',data=dataset)



b1 = plt.subplot2grid((3,3),(2,0))

b1 = sns.countplot(y='PaymentMethod',data=dataset)



b1 = plt.subplot2grid((3,3),(2,1))

b1 = sns.countplot(y='Churn',data=dataset)



plt.tight_layout()

plt.show()

c1 = plt.subplot2grid((3,1),(0,0))

c1 = sns.distplot(dataset['tenure'])



c1 = plt.subplot2grid((3,1),(1,0))

c1 = sns.distplot(dataset['MonthlyCharges'])



c1 = plt.subplot2grid((3,1),(2,0))

c1 = sns.distplot(dataset['TotalCharges'],bins=10)



plt.tight_layout()

plt.show()
#Bivariate

#Categorical Variables - 1

a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.gender=='Male'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Male")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.gender=='Female'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Female")

plt.tight_layout()

plt.show()

#Gender is not a good factor 
a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.SeniorCitizen==True].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Senior Citizen")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.SeniorCitizen==False].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Not Senior Citizen")

plt.tight_layout()

plt.show()

#The churn rate is quite high in Senior Citizen
a1 = plt.subplot2grid((3,3),(0,2))

a1 = sns.countplot(y='Partner',hue="Churn",data=dataset)

a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.Partner=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have Partners")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.Partner=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have Partners")

plt.tight_layout()

plt.show()

#CHurn rate is higher for people who dont have partners

a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.Dependents=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have Dependents")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.Dependents=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have Dependents")

plt.tight_layout()

plt.show()



#Churn rate is higher for people who dont have dependets
a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.PhoneService=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have PhoneService")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.PhoneService=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have PhoneService")

plt.tight_layout()

plt.show()

#There is not much difference but people who have phone service churn more
a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.MultipleLines=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have MultipleLines")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.MultipleLines=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have MultipleLines")

plt.tight_layout()

plt.show()

#People who have multiple line churn more

a1 = plt.subplot2grid((1,3),(0,0))

a1= dataset[dataset.InternetService=='Fiber optic'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Fiber Optic")

a1 = plt.subplot2grid((1,3),(0,1))

a1= dataset[dataset.InternetService=='DSL'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="DSL")

a1 = plt.subplot2grid((1,3),(0,2))

a1= dataset[dataset.InternetService=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have InternetService")

plt.tight_layout()

plt.show()

#Customers with Fiber Optic Cable churn the most (42% churn)
a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.OnlineSecurity=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have OnlineSecurity")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.OnlineSecurity=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have OnlineSecurity")

plt.tight_layout()

plt.show()

#Customers who dont have Online Security churn more (42%churn)
a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.OnlineBackup=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have OnlineBackup")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.OnlineBackup=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have OnlineBackup")

plt.tight_layout()

plt.show()

#Customers who dont have Online Backup churn more (40%churn)
a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.DeviceProtection=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have DeviceProtection")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.DeviceProtection=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have DeviceProtection")

plt.tight_layout()

plt.show()

#Customers who dont have DeviceProtection churn more (40%churn)
a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.TechSupport=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have TechSupport")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.TechSupport=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have TechSupport")

plt.tight_layout()

plt.show()

#Customers who dont have Tech Support churn more (42%churn)
a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.StreamingTV=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have StreamingTV")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.StreamingTV=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have StreamingTV")

plt.tight_layout()

plt.show()

#Dont have much differnce. Although people who dont have a streaming tv churn more(33%)
a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.StreamingMovies=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have StreamingMovies")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.StreamingMovies=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont have StreamingMovies")

plt.tight_layout()

plt.show()

#Dont have much differnce. Although people who dont have a streaming movies churn more(34%)
a1 = plt.subplot2grid((1,3),(0,0))

a1= dataset[dataset.Contract=='Month-to-month'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Month-to-month Contract")

a1 = plt.subplot2grid((1,3),(0,1))

a1= dataset[dataset.Contract=='One year'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="One year Contract")

a1 = plt.subplot2grid((1,3),(0,2))

a1= dataset[dataset.Contract=='Two year'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Two year Contract")

plt.tight_layout()

plt.show()

#Customers with Month on month contract churn more(43%)
a1 = plt.subplot2grid((1,2),(0,0))

a1= dataset[dataset.PaperlessBilling=='Yes'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Have PaperlessBilling")

a1 = plt.subplot2grid((1,2),(0,1))

a1= dataset[dataset.PaperlessBilling=='No'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Dont Have PaperlessBilling")

plt.tight_layout()

plt.show()

#Customers who have paperless billing churn more(34%)
a1 = plt.subplot2grid((2,2),(0,0))

a1= dataset[dataset.PaymentMethod=='Electronic check'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Electronic check")

a1 = plt.subplot2grid((2,2),(0,1))

a1= dataset[dataset.PaymentMethod=='Mailed check'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Mailed check")

a1 = plt.subplot2grid((2,2),(1,0))

a1= dataset[dataset.PaymentMethod=='Bank transfer (automatic)'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Bank transfer (automatic)")

a1 = plt.subplot2grid((2,2),(1,1))

a1= dataset[dataset.PaymentMethod=='Credit card (automatic)'].Churn.value_counts().plot(kind="pie",autopct ='%1.1f%%',title="Credit card (automatic)")



plt.tight_layout()

plt.show()

# 45% of the customers who pay through Elecctronic check churn
#Continous 

def kdeplot(feature):

    plt.figure(figsize=(9, 4))

    plt.title("KDE for {}".format(feature))

    ax0 = sns.kdeplot(dataset[dataset['Churn'] == 'No'][feature].dropna(), color= 'navy', label= 'Churn: No')

    ax1 = sns.kdeplot(dataset[dataset['Churn'] == 'Yes'][feature].dropna(), color= 'orange', label= 'Churn: Yes')
kdeplot('tenure')

#People with a small tenure churn out more
kdeplot('MonthlyCharges')

#People with a higher monthly charge churn out mor
kdeplot('TotalCharges')

#people with a lesser total charge churn out more 
#Plotting correlation

dataset.corr().style.background_gradient(cmap='coolwarm')
#Data Preprocessing:

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



#customer id col

Id_col     = ['customerID']

#Target columns

target_col = ["Churn"]

#categorical columns

cat_cols   = dataset.nunique()[dataset.nunique() < 6].keys().tolist()

cat_cols   = [x for x in cat_cols if x not in target_col]

#numerical columns

num_cols   = [x for x in dataset.columns if x not in cat_cols + target_col + Id_col]

#Binary columns with 2 values

bin_cols   = dataset.nunique()[dataset.nunique() == 2].keys().tolist()

#Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]



#Label encoding Binary columns

le = LabelEncoder()

for i in bin_cols :

    dataset[i] = le.fit_transform(dataset[i])

    

#Duplicating columns for multi value columns

dataset = pd.get_dummies(data = dataset,columns = multi_cols )



#Scaling Numerical columns

# Initialise the Scaler 

scaler = StandardScaler() 

scaled = pd.DataFrame(scaler.fit_transform(dataset[num_cols]))



#dropping original values merging scaled values for numerical columns

dataset = dataset.drop(columns = Id_col,axis = 1)

dataset = dataset.drop(columns = num_cols,axis = 1)

dataset = dataset.merge(scaled,left_index=True,right_index=True,how = "left")
#Modelling 

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

train,test = train_test_split(dataset,test_size = .20 ,random_state = 0)

    

##seperating dependent and independent variables

cols    = [i for i in dataset.columns if i not in Id_col + target_col]

train_X = train[cols]

train_Y = train[target_col]

test_X  = test[cols]

test_Y  = test[target_col]



def churn_prediction(classifier,train_X,train_Y,test_X,test_Y):

#Logistic Regression

    # Train the classifier

    classifier.fit(train_X, train_Y)

    # Apply The Full Featured Classifier To The Test Data

    y_pred = classifier.predict(test_X)

    # View The Accuracy Of Our Full Feature (4 Features) Model

    return accuracy_score(test_Y, y_pred),f1_score(test_Y,y_pred) 
# Create a random forest classifier

clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

accuracy,f1Score = churn_prediction(clf,train_X,train_Y,test_X,test_Y)

accuracy,f1Score #77.4 #51.2

#Logistic Regression

clf = LogisticRegression()

accuracy,f1Score = churn_prediction(clf,train_X,train_Y,test_X,test_Y)

accuracy,f1Score #79.7 #57.5
#KnearestNeighbours

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

           metric_params=None, n_jobs=1, n_neighbors=5, p=2,

           weights='uniform')

accuracy,f1Score = churn_prediction(knn,train_X,train_Y,test_X,test_Y)

accuracy ,f1Score #75.3, #52.0

#LightGBM

lgbm_c = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

                        learning_rate=0.5, max_depth=7, min_child_samples=20,

                        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,

                        n_jobs=-1, num_leaves=500, objective='binary', random_state=None,

                        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,

                        subsample_for_bin=200000, subsample_freq=0)



accuracy,f1Score = churn_prediction(lgbm_c,train_X,train_Y,test_X,test_Y)

accuracy,f1Score #75 #50