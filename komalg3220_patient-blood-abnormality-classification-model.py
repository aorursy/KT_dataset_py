

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn import linear_model

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing  # to normalisation

from sklearn.model_selection import train_test_split as dsplit







print(os.listdir("../input"))



df = pd.read_csv('../input/Training Data - Classification of Patients with Abnormal Blood Pressure (N2000)_27-Jul-2016.csv')



df.drop(['Patient_Number',],axis=1,inplace=True) #to drop column



df.head(6)

#null value

df.isnull().any()
#to check how manu null values are there



df.isnull().sum()
#To check unique values 

df['Pregnancy'].value_counts()
df[df['Sex'] == 1]['Pregnancy'].count()



for i in df[df['Sex'] == 1]['Pregnancy'].unique():

    print (i, list(df[df['Sex'] == 1]['Pregnancy']).count(i))

    

    

'''def FillNAPregnancy(row):

    if row == 0:

        return 0

print('Columns Consisting of nan')



data['Pregnancy']=data['Sex'].apply(FillNAPregnancy)



data = data.drop(['Pregnancy'], axis=1)

data['Genetic_Pedigree_Coefficient']=data['Genetic_Pedigree_Coefficient'].fillna(data['Genetic_Pedigree_Coefficient'].mean())

data['alcohol_consumption_per_day']=data['alcohol_consumption_per_day'].fillna(data['alcohol_consumption_per_day'].mean())

print(data[data.columns[data.isna().any()].tolist()].isnull().sum())'''    

    
df.drop(['Pregnancy',],axis=1,inplace=True) #to drop column
#To check unique values 

df['Genetic_Pedigree_Coefficient'].value_counts()
df["Genetic_Pedigree_Coefficient"].fillna(value=df["Genetic_Pedigree_Coefficient"].mean(),inplace=True)
df['alcohol_consumption_per_day'].value_counts()
df["alcohol_consumption_per_day"].fillna(value=df["alcohol_consumption_per_day"].mean(),inplace=True)
df.isnull().sum()
#To check the data type of coloumn



df.dtypes
#To check wheater column is contnious or categorical



for column in df.columns:

    print(column,len(df[column].unique()))
def plotBarChart(df,col,label):

    g = sns.FacetGrid(df, col=col)

    g.map(plt.hist, label, bins=10)



for val in ['Genetic_Pedigree_Coefficient','Level_of_Hemoglobin','Chronic_kidney_disease','Adrenal_and_thyroid_disorders','Age','BMI','Physical_activity']:

    plotBarChart(df,'Blood_Pressure_Abnormality',val)   


#Scatter plot for continous Value

for column in ['Physical_activity','salt_content_in_the_diet','alcohol_consumption_per_day','Level_of_Hemoglobin']:



    x=df[column]    

    y=df['Blood_Pressure_Abnormality']

    plt.scatter(x, y)

    plt.xlabel('x')

    plt.ylabel('y')

    plt.show()
##creating new features out of existing

##converting gpc to distant occurrence & immediate occurence

print(df['Genetic_Pedigree_Coefficient'].mean())

#as mean is 0.49 we will consider values above it to be immediate and below it to be distant

def distantoccurence(row):

    if row<0.5:

        return 1

    else:

        return 0 

def immediateoccurance(row):

    if row>=0.5:

        return 1

    else:

        return 0   

df['distantoccurence']=df['Genetic_Pedigree_Coefficient'].apply(distantoccurence)

df['immediateoccurance']=df['Genetic_Pedigree_Coefficient'].apply(immediateoccurance)



def hemoglobin(row):

    if row['Sex']==1:

        if row['Level_of_Hemoglobin']<12.0:

            return 'lowHg'

        elif (row['Level_of_Hemoglobin']>=12.0 and row['Level_of_Hemoglobin']<=15.0):

            return 'normalHg'

        elif (row['Level_of_Hemoglobin']>15.0):

            return 'highHg'

    elif row['Sex']==0:

        if row['Level_of_Hemoglobin']<14.0:

            return 'lowHg'

        elif (row['Level_of_Hemoglobin']>=14.0 and row['Level_of_Hemoglobin']<=17.0):

            return 'normalHg'

        elif (row['Level_of_Hemoglobin']>17.0):

            return 'highHg'

for i, row in df.iterrows():

    val=hemoglobin(row)

    df.at[i,'hemoglobinBin'] = val

dummy=pd.get_dummies(df['hemoglobinBin'])

df = pd.concat([df, dummy], axis=1, sort=False)



def BmiToBinaryData(row):

    if row>30:

        return 1

    else:

        return 0

df['obesity']=df['BMI'].apply(BmiToBinaryData)

print(df.dtypes)
def plotBarChart(df,col,label):

    g = sns.FacetGrid(df, col=col)

    g.map(plt.hist, label, bins=10)



for val in ['Genetic_Pedigree_Coefficient','Level_of_Hemoglobin','Chronic_kidney_disease','Adrenal_and_thyroid_disorders','Age','BMI','Physical_activity','highHg','lowHg','normalHg','obesity']:

    plotBarChart(df,'Blood_Pressure_Abnormality',val)   
# To check Correlation

#df.corr() #df.corr(method='spearman')

df.corr().abs().unstack().sort_values()['Blood_Pressure_Abnormality']
#To define x and y

#x = df.loc[:, df.columns != 'Blood_Pressure_Abnormality'] #to  select multiple column except one data point may be that we want to predict

#y=df['Blood_Pressure_Abnormality'].values #.values = to get the numpy array and dataset dont return index value and column with selected column



x = df[['Genetic_Pedigree_Coefficient','Level_of_Hemoglobin','Chronic_kidney_disease','Adrenal_and_thyroid_disorders','Age','BMI','Physical_activity']]

y = df["Blood_Pressure_Abnormality"]



#convert categorical values (either text or integer) 

#df = pd.get_dummies(df, columns=['type'])

#x=pd.get_dummies(x,columns=[''])

#print(x.columns)



#To Normalise the equation

#x=preprocessing.normalize(x)

print(x.head())

print(y)









x_train, x_test, y_train, y_test = dsplit(x, y, random_state = 1)

reg = LogisticRegression()

reg.fit(x_train, y_train)

predicted = reg.predict(x_test)

from sklearn.metrics import accuracy_score

#r_square = metrics.score(y_test, predicted)

accuracy_score(y_test, predicted)





'''#train and test dataset creation

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

logistic = linear_model.LogisticRegression()

logistic.fit(x_train,y_train)

predicted_Values = regression.predict(x_test)

print(predicted_Values)

print(y_test)'''