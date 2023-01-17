# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd





data_predict=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

print('Data Show Info\n')

data_predict.info()





data_predict=data_predict.rename(columns={'Serial No.':'SerialNo','GRE Score':'GREScore','TOEFL Score':'TOEFLScore','LOR ':'LOR','University Rating':'UniversityRating','Chance of Admit ':'ChanceOfAdmit'})







data_predict=data_predict.drop(['SerialNo'],axis=1) #1 column 0 row



import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

#Correlation Matrix



#create correlation matrix without class

Tf_corr=data_predict.iloc[:,0:8]



#Correlation Matrix 



corrmat=Tf_corr.corr()



#Features :columns and rows

sns.heatmap(corrmat, 

 xticklabels=corrmat.columns,

 yticklabels=corrmat.columns)



def find_maxCorrelations(dataset,threshold):

    col_corr=set()

    corr_matrix=dataset.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if(abs(corr_matrix.iloc[i,j])>threshold):

                colname=corr_matrix.columns[i]

                col_corr.add(colname)

    return col_corr





corr_features=find_maxCorrelations(corrmat,0.8)



print(corrmat.iloc[:,7:8])
def normalize(df):

    result = df.copy()

    for feature_name in df.columns:

        max_value = df[feature_name].max()

        min_value = df[feature_name].min()

        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result





normalized_data=normalize(data_predict)
x=normalized_data.iloc[:,0:7]



y=normalized_data.iloc[:,7:8]



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)



from sklearn import datasets, linear_model

from sklearn.linear_model import LinearRegression





model = LinearRegression()

model.fit(x_train,y_train)





y_pred = model.predict(x_test)

y_pred=pd.DataFrame(y_pred)



score=model.score(x_test,y_test)

print("Accuray: "+str(score))

#Basic Feature Selecion



corr_features=find_maxCorrelations(normalized_data,0.7)

print(corr_features)



#dataset_with_correlated_features 

new_Dataset=normalized_data[corr_features]

y=new_Dataset.iloc[:,-1]



x=new_Dataset.drop(['ChanceOfAdmit'],axis=1)



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)



from sklearn import datasets, linear_model

from sklearn.linear_model import LinearRegression





model = LinearRegression()

model.fit(x_train,y_train)





y_pred = model.predict(x_test)

y_pred=pd.DataFrame(y_pred)



score=model.score(x_test,y_test)

print("Accuray: "+str(score))


