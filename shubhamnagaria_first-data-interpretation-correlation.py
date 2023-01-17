# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

from math import log10

sns.set_style('whitegrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# get data csv files as a DataFrame

database_df = pd.read_csv("../input/database.csv")

# database_df.columns.values to visualize the name of every columns
# Preview of data :



# Remove Unknown data and some categories 



database_df = database_df.replace('Unknown', np.nan)

database_df = database_df.dropna()





data = (database_df.drop(['Record ID', 'Agency Code','Crime Solved', 'Agency Name', 'Agency Type',

       'Record Source'],axis=1))



ID = database_df['Record ID'].values





data.head()
# Definition of a small function to construct a dictionnary

def list_par(x,l):

    if x not in l :

        l[x]=len(l)

    return l
# Definition of all dictionnaries to allow an acp on data



dic_city = {}

for city in data['City']:

    (list_par(city,dic_city))



dic_state = {}

for state in data['State']:

    (list_par(state,dic_state))

    

dic_crime_type = {}

for crime_type in data['Crime Type']:

    (list_par(crime_type,dic_crime_type))

    

dic_victim_race = {}

for victim_race in data['Victim Race']:

    (list_par(victim_race,dic_victim_race))

    

dic_victim_ethnicity = {}

for victim_ethnicity in data['Victim Ethnicity']:

    (list_par(victim_ethnicity,dic_victim_ethnicity))



    

dic_perpetrator_sex = {}

for perpetrator_sex in data['Perpetrator Sex']:

    (list_par(perpetrator_sex,dic_perpetrator_sex))

    

dic_perpetrator_ethnicity = {}

for perpetrator_ethnicity in data['Perpetrator Ethnicity']:

    (list_par(perpetrator_ethnicity,dic_perpetrator_ethnicity))

    

dic_relationship = {}

for relationship in data['Relationship']:

    (list_par(relationship,dic_relationship))



dic_weapon = {}

for weapon in data['Weapon']:

    (list_par(weapon,dic_weapon))



dic_month = {}

for month in data['Month']:

    (list_par(month,dic_month))

    

dic_sexe = {}

for sexe in data['Victim Sex']:

    (list_par(sexe,dic_sexe))

       

dic_perpetrator_race = {}

for perpetrator_race in data['Perpetrator Race']:

    (list_par(perpetrator_race,dic_perpetrator_race))





data['Crime Type']=data['Crime Type'].map(dic_crime_type)

data['Victim Race']=data['Victim Race'].map(dic_victim_race)

data['Victim Ethnicity']=data['Victim Ethnicity'].map(dic_victim_ethnicity)

data['Perpetrator Race']=data['Perpetrator Race'].map(dic_perpetrator_race)

data['Perpetrator Sex']=data['Perpetrator Sex'].map(dic_perpetrator_sex)

data['Perpetrator Ethnicity']=data['Perpetrator Ethnicity'].map(dic_perpetrator_ethnicity)

data['Relationship']=data['Relationship'].map(dic_relationship)

data['Weapon']=data['Weapon'].map(dic_weapon)

data['Month']=data['Month'].map(dic_month)

data['Victim Sex']=data['Victim Sex'].map(dic_sexe)

data['City']=data['City'].map(dic_city)

data['State']=data['State'].map(dic_state)
# New dataframes 

data.head()
# Correlation between all datas 

data.corr()
# Visualization of the correlations, code by nirajverma

correlation = data.corr()

plt.figure(figsize=(15,15))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')
# Remove incident data to only keep crime data



data_crime = data [ data["Victim Count"] >= 1 ]



# Remove high correlation data 

data_crime = (data_crime.drop(['Incident','State','City','Month'],axis=1))
# Correlation between all datas 

data_crime.corr()
# Visualization of the correlations, code by nirajverma

correlation = data_crime.corr()

plt.figure(figsize=(15,15))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')
# Number of incident with at least one victim

len(data_crime)
# Remove crime data to only keep incident data



data_incident = data [ data["Victim Count"] == 0 ]

data_incident = (data_incident.drop(['Victim Count'],axis=1))
# Correlation between all datas 

data_incident.corr()
# Visualization of the correlations, code by nirajverma

correlation = data_incident.corr()

plt.figure(figsize=(15,15))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')
# Number of incident without victim

len(data_incident)
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()



#data_crime = (data_crime.drop(['Year'],axis=1))

data_crime_std = ss.fit_transform(data_crime)



from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=6)

Y_sklearn = sklearn_pca.fit_transform(data_crime_std)



print(sklearn_pca.explained_variance_ratio_) 
# Display PCA

from matplotlib import pyplot

import pylab

from mpl_toolkits.mplot3d import Axes3D



fig = pylab.figure()

ax = Axes3D(fig)

ax.scatter(Y_sklearn[:,0], Y_sklearn[:,1], Y_sklearn[:,2])

plt.show()

            
plt.figure(figsize=(5,5))

plt.scatter(Y_sklearn[:,0], Y_sklearn[:,1],s=10)

plt.show()
plt.figure(figsize=(5,5))

plt.scatter(Y_sklearn[:,0], Y_sklearn[:,2],s=10)

plt.show()
plt.figure(figsize=(5,5))

plt.scatter(Y_sklearn[:,2], Y_sklearn[:,0],s=10)

plt.show()
plt.figure(figsize=(5,5))

plt.scatter(Y_sklearn[:,2], Y_sklearn[:,1],s=10)

plt.show()
ID1 = database_df[database_df["Victim Count"] >= 1 ]



ID_crime = ID1['Record ID']



s1 = pd.Series(ID_crime, name='ID Crime')



data_crime = pd.concat([data_crime, s1], axis=1)



data_crime.head()
a = Y_sklearn[:,0]

#s2 = pd.Series(a, name='k0')

s2 = DataFrame(index = (ID_crime-1), data =a)



data_crime1 = pd.concat([data_crime, s2], axis=1)



data_crime1 = data_crime1[data_crime1[0] >= 5 ]



data_crime1
s3 = DataFrame(index = (ID_crime-1), data =Y_sklearn[:,2])



data_crime2 = pd.concat([data_crime, s3], axis=1)



data_crime2 = data_crime2[data_crime2[0] >= 8 ]



data_crime2.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



data_X = (data.drop(['Perpetrator Sex','Perpetrator Age','Perpetrator Race','Perpetrator Ethnicity','Relationship','Perpetrator Count'],axis=1))



Perpetrator_Sex_y = data['Perpetrator Sex'].values

Perpetrator_Age_y = data['Perpetrator Age'].values

Perpetrator_Race_y = data['Perpetrator Race'].values

Perpetrator_Ethnicity_y = data['Perpetrator Ethnicity'].values

Relationship_y = data['Relationship'].values

Perpetrator_Count_y = data['Perpetrator Count'].values



misslabel = []

misslabel1 = []

misslabel2 = []

misslabel3 = []

misslabel4 = []

misslabel5 = []



for i in range(10):

    np.random.seed(i)

    indices = np.random.permutation(155851)

    indices_train = indices[:-25000]

    indices_test = indices[-25000:]



    data_X_train = data_X.iloc[indices_train]

    Perpetrator_Race_y_train = Perpetrator_Race_y[indices_train]

    Perpetrator_Sex_y_train = Perpetrator_Sex_y[indices_train]

    Perpetrator_Age_y_train = Perpetrator_Age_y[indices_train]

    Perpetrator_Ethnicity_y_train = Perpetrator_Ethnicity_y[indices_train]

    Relationship_y_train = Relationship_y[indices_train]

    Perpetrator_Count_y_train = Perpetrator_Count_y[indices_train]

    

    



    data_X_test  = data_X.iloc[indices_test]

    Perpetrator_Race_y_test  = Perpetrator_Race_y[indices_test]

    Perpetrator_Sex_y_test = Perpetrator_Sex_y[indices_test]

    Perpetrator_Age_y_test = Perpetrator_Age_y[indices_test]

    Perpetrator_Ethnicity_y_test = Perpetrator_Ethnicity_y[indices_test]

    Relationship_y_test = Relationship_y[indices_test]

    Perpetrator_Count_y_test = Perpetrator_Count_y[indices_test]







    

    gnb_race = GaussianNB()

    y_pred_race = gnb_race.fit(data_X_train, Perpetrator_Race_y_train).predict(data_X_test)

    misslabel.append((Perpetrator_Race_y_test != y_pred_race).sum()/25000)

    

    gnb_sex = GaussianNB()

    y_pred_sex = gnb_sex.fit(data_X_train, Perpetrator_Sex_y_train).predict(data_X_test)



    misslabel1.append((Perpetrator_Sex_y_test != y_pred_sex).sum()/25000)

    

    #gnb_age = GaussianNB()

    #y_pred_age = gnb_age.fit(data_X_train, Perpetrator_Age_y_train).predict(data_X_test)



    #misslabel2.append((Perpetrator_Age_y_test != y_pred_age).sum()/25000)

    

    gnb_ethni = GaussianNB()

    y_pred_ethni = gnb_ethni.fit(data_X_train, Perpetrator_Ethnicity_y_train).predict(data_X_test)



    misslabel3.append((Perpetrator_Ethnicity_y_test != y_pred_ethni).sum()/25000)

     

    gnb_rela = GaussianNB()

    y_pred_rela = gnb_rela.fit(data_X_train, Relationship_y_train).predict(data_X_test)



    misslabel4.append((Relationship_y_test != y_pred_rela).sum()/25000)

    

    gnb_count = GaussianNB()

    y_pred_count = gnb_count.fit(data_X_train, Perpetrator_Count_y_train).predict(data_X_test)



    misslabel5.append((Perpetrator_Count_y_test != y_pred_count).sum()/25000)

    





    knn_rela = KNeighborsClassifier()

    knn_rela.fit(data_X_train, Relationship_y_train)

    

    misslabel4.append((Relationship_y_test != knn_rela.predict(data_X_test)).sum()/25000)

    

    

print("Misslabel Perpetrator Race : ",misslabel)

print("Misslabel Perpetrator Sex : ",misslabel1)

print("Misslabel Perpetrator Ethnicity : ",misslabel3)

print("Misslabel Perpetrator Count : ",misslabel5)

print("Misslabel Perpetrator Age : ",misslabel2)

print("Misslabel Perpetrator Relationship : ",misslabel4)
