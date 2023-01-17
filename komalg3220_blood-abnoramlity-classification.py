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

import warnings



warnings.filterwarnings("ignore")



print(os.listdir("../input"))



data = pd.read_csv('../input/Training Data - Classification of Patients with Abnormal Blood Pressure (N2000)_27-Jul-2016.csv')





print('DataType in Dataset')

print(data.dtypes)

print('Number of Columns containing Null Value')

print(data.isnull().any().sum(), ' / ', len(data.columns))

print('Number of rows containing null in either column')

print(data.isnull().any(axis=1).sum(), ' / ', len(data))

print('Checking colinearity with Blood Pressure Abnormality')

print(data.corr().abs().unstack().sort_values()['Blood_Pressure_Abnormality'])



data.head(6)
from sklearn.feature_selection import RFE, f_regression

from sklearn.linear_model import (LinearRegression, Ridge, Lasso)

from sklearn import tree, linear_model

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc

from matplotlib.legend_handler import HandlerLine2D





def ModelSelection(test_data,features,label):

    MLA = [

    ensemble.RandomForestClassifier(),

           

    linear_model.LogisticRegressionCV(),

    linear_model.SGDClassifier(),

                    

    tree.DecisionTreeClassifier(),

                

    ]

    

    MLA_columns = ['MLA Name', 'MLA Parameters','MLA Score']

    MLA_compare = pd.DataFrame(columns = MLA_columns)

    x_train,x_test,y_train,y_test = train_test_split (data[features],data[label],test_size=0.2,random_state=0)

    row_index = 0

    MLA_predict = data[label]

    for alg in MLA:



        MLA_name = alg.__class__.__name__

        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        alg.fit(x_train, y_train)

        MLA_predict[MLA_name] = alg.predict(x_test)

        MLA_compare.loc[row_index, 'MLA Score']=alg.score(x_test,y_test)

        row_index+=1



    

    MLA_compare.sort_values(by = ['MLA Score'], ascending = False, inplace = True)

    return MLA_compare,x_train,x_test,y_train,y_test
print(data[data.columns[data.isna().any()].tolist()].isnull().sum())



def FillNAPregnancy(row):

    if row == 0:

        return 0

print('Columns Consisting of nan')



data['Pregnancy']=data['Sex'].apply(FillNAPregnancy)



data = data.drop(['Pregnancy'], axis=1)

data['Genetic_Pedigree_Coefficient']=data['Genetic_Pedigree_Coefficient'].fillna(data['Genetic_Pedigree_Coefficient'].mean())

data['alcohol_consumption_per_day']=data['alcohol_consumption_per_day'].fillna(data['alcohol_consumption_per_day'].mean())

print(data[data.columns[data.isna().any()].tolist()].isnull().sum())
##creating new features out of existing

##converting gpc to distant occurrence & immediate occurence

print(data['Genetic_Pedigree_Coefficient'].mean())

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

data['distantoccurence']=data['Genetic_Pedigree_Coefficient'].apply(distantoccurence)

data['immediateoccurance']=data['Genetic_Pedigree_Coefficient'].apply(immediateoccurance)



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

for i, row in data.iterrows():

    val=hemoglobin(row)

    data.at[i,'hemoglobinBin'] = val

dummy=pd.get_dummies(data['hemoglobinBin'])

data = pd.concat([data, dummy], axis=1, sort=False)



def BmiToBinaryData(row):

    if row>30:

        return 1

    else:

        return 0

data['obesity']=data['BMI'].apply(BmiToBinaryData)

print(data.dtypes)
print(data.corr().abs().unstack().sort_values()['Blood_Pressure_Abnormality'])
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

#tips = sns.load_dataset("tips")

ax = sns.scatterplot(x="Genetic_Pedigree_Coefficient", y="Level_of_Hemoglobin",hue="Blood_Pressure_Abnormality", data=data)
def plotBarChart(data,col,label):

    g = sns.FacetGrid(data, col=col)

    g.map(plt.hist, label, bins=10)



for val in ['Genetic_Pedigree_Coefficient','Level_of_Hemoglobin','Chronic_kidney_disease','Adrenal_and_thyroid_disorders','Age','BMI','Physical_activity','highHg','lowHg','normalHg','obesity']:

    plotBarChart(data,'Blood_Pressure_Abnormality',val)   
features=['Genetic_Pedigree_Coefficient','Level_of_Hemoglobin','Chronic_kidney_disease','Adrenal_and_thyroid_disorders','highHg','lowHg','normalHg']

MLA_compare,x_train,x_test,y_train,y_test=ModelSelection(data,features,'Blood_Pressure_Abnormality')

print(MLA_compare[['MLA Name','MLA Score']].head())