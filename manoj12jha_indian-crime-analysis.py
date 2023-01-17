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





import random as rnd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier        

        



# Any results you write to the current directory are saved as output.
rape_data = pd.read_csv('../input/crime-in-india/20_Victims_of_rape.csv')

state_geo = pd.read_csv('../input/indian-state-details/latitude.csv')

#rape_data.rename(columns={'Area_Name': 'State'}, inplace=True)

rape_data.head()
state_geo.head()
#df2['new'] = df2.first_id.isin(df1.id).astype(np.int8)

#rape_data['latitude'] = rape_data.Area_Name.isin(state_geo.State).astype(np.float64)
rape_data.info()
#rape_data.State.unique()
'''def set_value(row_number, assigned_value): 

    return assigned_value[row_number]'''



latitude_mapping = {"Andaman & Nicobar Islands": 11.7400867, "Andhra Pradesh":15.91, "Arunachal Pradesh":28.21, "Assam":26.20, "Bihar":25.09

                , "Chandigarh":30.7333, "Chhattisgarh":21.27, "Dadra & Nagar Haveli":20.1809, "Daman & Diu":20.4283, "Delhi":28.7041, "Goa":15.29, "Gujarat":22.25

                , "Haryana":29.05, "Himachal Pradesh":31.10, "Jammu & Kashmir":33.77, "Jharkhand":23.61, "Karnataka":15.31, "Kerala":10.85,

                "Lakshadweep":10,"Madhya Pradesh":22.97,"Maharashtra":19.75,"Manipur":24.66,"Meghalaya":25.46,"Mizoram":23.16,"Nagaland":26.15,"Odisha":20.95,"Puducherry":11.9416,

                "Punjab":31.14,"Rajasthan":27.02,"Sikkim":27.53,"Tamil Nadu":11.12,"Tripura":23.94,"Uttar Pradesh":26.84,"Uttarakhand":30.06,"West Bengal":22.98}



#event_dictionary ={'Music' : 1500, 'Poetry' : 800, 'Comedy' : 1200} 

    

# Add a new column named 'Price' 

#df['Price'] = df['Event'].apply(set_value, args =(event_dictionary, )) 

#rape_data['Latitude'] = rape_data['Area_Name'].apply(set_value, args=(latitude_mapping))





rape_data['Latitude'] = rape_data['Area_Name'].map(latitude_mapping)

  



#print(rape_data) 



'''for dataset in rape_data.iteritems():

    #print(dataset)

    dataset['Latitude'] = dataset['Latitude'].map(latitude_mapping)

    #dataset[1] = dataset[1].map(latitude_mapping)

    #print(dataset)

    dataset['Latitude'] = dataset['Latitude'].fillna(0)'''

    

rape_data.head()

#rape_data.drop(columns=['latitude'])
rape_data.isnull().sum()
longitude_mapping = {"Andaman & Nicobar Islands": 92.6586, "Andhra Pradesh":79.74, "Arunachal Pradesh":94.72, "Assam":92.93, "Bihar":85.31

                , "Chandigarh":76.7794, "Chhattisgarh":81.86, "Dadra & Nagar Haveli":73.0169, "Daman & Diu":72.8397, "Delhi":77.1025, "Goa":74.12, "Gujarat":71.19

                , "Haryana":76.08, "Himachal Pradesh":77.17, "Jammu & Kashmir":76.57, "Jharkhand":85.27, "Karnataka":75.71, "Kerala":76.27,

                "Lakshadweep":72.1833,"Madhya Pradesh":78.65,"Maharashtra":75.71,"Manipur":93.90,"Meghalaya":91.36,"Mizoram":92.93,"Nagaland":94.56,"Odisha":85.09,"Puducherry":79.8083,

                "Punjab":75.34,"Rajasthan":74.21,"Sikkim":88.51,"Tamil Nadu":78.65,"Tripura":91.98,"Uttar Pradesh":80.94,"Uttarakhand":79.01,"West Bengal":87.85}



rape_data['Longitude'] = rape_data['Area_Name'].map(longitude_mapping)

rape_data.head()
sns.set(rc={'figure.figsize':(10,10)})

sns.scatterplot(x='Longitude', y='Latitude', hue='Rape_Cases_Reported',s=1500, data=rape_data)
sns.pairplot(rape_data, vars=["Rape_Cases_Reported", "Victims_Above_50_Yrs","Victims_Between_10-14_Yrs", "Victims_Between_14-18_Yrs","Victims_Between_18-30_Yrs","Victims_Between_30-50_Yrs"])
sns.barplot(x="Rape_Cases_Reported", y="Area_Name", data=rape_data)

sns.factorplot(x="Year", y="Rape_Cases_Reported", data=rape_data, kind='bar')
dw_crime_against_women = pd.read_csv('../input/crime-in-india/crime/42_District_wise_crimes_committed_against_women_2013.csv')

dw_crime_against_women
dw_crime_against_women.DISTRICT.unique()
#dw_crime_against_women.to_csv(r'../File_Name.csv')

dw_crime_against_women.to_csv('mycsvfile.csv',index=False)