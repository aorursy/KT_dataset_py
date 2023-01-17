# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

data=pd.read_csv("../input/application_train.csv")

data.head()

d=data.iloc[:1000,:]
d.shape
#preprocessing

#Drop the columns based on the given threshold value 

def drop_Null_threshold(data):

    colname=data.columns

    coltype=data.dtypes

    total=data.isna().sum()

    percent=(total/data.shape[0])*100

    dict={'ColumnName':colname,

             'ColumnsType':coltype,

             'Null_percent':percent}

    nulldf=pd.DataFrame(dict)

    #print(nulldf)

    threshold=float(input("Please enter threshold value: "))

    for i in data:

        if nulldf.loc[i,'Null_percent']>threshold:

            data.drop(i,axis=1,inplace=True)

    return data

d=drop_Null_threshold(d)

def f5_variabletype(data):

    colname=data.columns

    coltype=data.dtypes

    variabletype=[]

    for i in data:

        if (data[i].nunique()>8) and (data[i].dtype=='int64' or data[i].dtype=='float64'):

            variabletype.append('Continuous')

        #elif (data[i].nunique()<=8):

         #   variabletype.append('Class')

        else:

            variabletype.append('Class')

    variabletype

    dict={'ColumnName':colname,

         'Column_dtype':coltype,

          'Variable_Type':variabletype}

    return pd.DataFrame(dict)

#if there are null value, then replace continuous with MEDIAN and categorical with MODE

def f6_NullValueTreatment(data):

    typedf=f5_variabletype(data)

    for i in data:

        if typedf.loc[i,'Variable_Type'] is'Continuous':

            data[i].fillna(value=data[i].median(),inplace=True)

        elif typedf.loc[i,'Variable_Type'] is'Class':

            data[i].fillna(value=data[i].mode().loc[0],inplace=True)

    return data

d=f6_NullValueTreatment(d)
y=d["AMT_CREDIT"]

d.drop("AMT_CREDIT", axis=1, inplace=True)
# Outliers Code



import numpy as np

import statistics as sts

def outlier_detect(df):

    for i in df.columns:

        x=np.array(df[i])

        p=[]

        Q1 = df[i].quantile(0.25)

        Q3 = df[i].quantile(0.75)

        IQR = Q3 - Q1

        LTV= Q1 - 1.5 * IQR

        UTV= Q3 + 1.5 * IQR

        for j in x:

            if j < LTV or j>UTV:

                p.append(sts.median(x))

            else:

                p.append(j)

        df[i]=p

    return df

d=outlier_detect(d[d.describe().columns])


x=d
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 100) 
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x_train, y_train)
y_pred = reg.predict(x_test)
plott=pd.DataFrame()

plott["actual"]=y_test.values

plott["predicted"]=y_pred

plott.plot(figsize=(50,10))
from sklearn.ensemble import RandomForestRegressor

lm=RandomForestRegressor()

lm.fit(x_train,y_train)

lm.predict(x_test)

pred_values_1=lm.predict(x_test)
plott["random forest"]=pred_values_1

plott.plot(figsize=(20,5))