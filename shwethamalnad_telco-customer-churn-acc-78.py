import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

import glob

import seaborn as sns 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df.shape
df = df.drop("customerID",axis=1)
# Replacing the  ' ' in Totalcharges to NA values and then converting into float  

df['TotalCharges']=df['TotalCharges'].replace(' ',np.nan)

df["TotalCharges"] = df["TotalCharges"].astype(float)

# Changing the senior citizen value to YES or NO 

df['SeniorCitizen']=df['SeniorCitizen'].replace(1,"Yes")

df['SeniorCitizen']=df['SeniorCitizen'].replace(0,"No")





#defining the continuoes variable and Categoical variable 

cat_var = []

data=df.mean()

data.index

cont_var =df[data.index]

cont_var

for i in df : 

    if i in data:

        print(i)

    else: 

        cat_var.append(i)

print(cat_var)
i="No internet service"

j=i.split()

j[0]
#Changing the Vlaue No internet service and No phone service into NO

cat_yes_no =["MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]

for value in cat_yes_no:

    

    for i in df[value]:

        

        split_val = i.split()

        

        if split_val[0] =="No":

            df[value]=df[value].replace(i,"No")

            

        

df[cat_yes_no].nunique()

       



  
""" iterate through all the columns of a dataframe and modify the data type

    to reduce memory usage.        

"""

start_mem = df.memory_usage().sum() / 1024**2

print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



for col in df.columns:

    col_type = df[col].dtype



    if col_type != object:

        c_min = df[col].min()

        c_max = df[col].max()

        if str(col_type)[:3] == 'int':

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                df[col] = df[col].astype(np.int8)

            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                df[col] = df[col].astype(np.int16)

            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                df[col] = df[col].astype(np.int32)

            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                df[col] = df[col].astype(np.int64)  

        else:

            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                df[col] = df[col].astype(np.float16)

            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float32)

            else:

                df[col] = df[col].astype(np.float64)

    else:

        df[col] = df[col].astype('category')



end_mem = df.memory_usage().sum() / 1024**2

print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))   
df.isna().sum()


def null_values(base_dataset):

    print(base_dataset.isna().sum())

    ## null value percentage     

    null_value_table=(base_dataset.isna().sum()/base_dataset.shape[0])*100

    ## null value percentage beyond threshold drop , else treat the columns 

    

    retained_columns=null_value_table[null_value_table<30].index

    # if any variable as null value greater than input(like 30% of the data) value than those variable are consider as drop

    drop_columns=null_value_table[null_value_table>30].index

    base_dataset.drop(drop_columns,axis=1,inplace=True)

    len(base_dataset.isna().sum().index)

    cont=base_dataset.describe().columns

    cat=[i for i in base_dataset.columns if i not in base_dataset.describe().columns]

    for i in cat:

        base_dataset[i].fillna(base_dataset[i].value_counts().index[0],inplace=True)

    for i in cont:

        base_dataset[i].fillna(base_dataset[i].median(),inplace=True)

    print(base_dataset.isna().sum())

    return base_dataset,cat,cont

null_values(df)
df.dtypes
sns.boxplot(df["MonthlyCharges"])
sns.boxplot(df["TotalCharges"])
sns.boxplot(df["tenure"])
for i in cat_var : 

    print((df[i].value_counts()/df.shape[0])*100)
for j in cat_var:

    sns.countplot(df[j],hue=df["Churn"])

    plt.xticks(rotation=45)

    plt.show()

   
sns.pairplot(df,hue="Churn")
sns.catplot(x="Churn",y="tenure",data=df,jitter=False)
for i in cat_var:

    sns.boxplot(x=df[i],y=df["TotalCharges"],hue=df["Churn"])

    plt.show()

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

df[cat_var].nunique()

bin_cols   = df.nunique()[df.nunique() == 2].keys().tolist()

multi_col=[]

for i in cat_var:

    if  i not in bin_cols:

        multi_col.append(i)

multi_col

le=LabelEncoder()

for i in bin_cols:

    df[i]=le.fit_transform(df[i])

df.head()

df = pd.get_dummies(data=df,columns=multi_col)





Y =pd.DataFrame(df["Churn"])

X = df.drop("Churn",axis=1)





from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
X_train.shape,X_test.shape,Y_test.shape,Y_train.shape
from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

from sklearn.tree import DecisionTreeClassifier



model=[RandomForestClassifier,BaggingClassifier,DecisionTreeClassifier]

for i in model:

    classifier = i()

    classifier.fit(X_train,Y_train)

    y_pred= classifier.predict(X_test)

    cm=confusion_matrix(Y_test,y_pred)

    acc_score = accuracy_score(Y_test,y_pred)

    print(cm,acc_score)



            
