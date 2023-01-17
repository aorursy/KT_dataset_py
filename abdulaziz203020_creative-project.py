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
#Importing_libraries



import numpy as np #Linear Algebra

import pandas as pd #Data processing

from sklearn.preprocessing import LabelEncoder #Data Preprocessing

from sklearn.preprocessing import StandardScaler #Data Preprocessing
#Data Overview



telcom = pd.read_csv('C:/Users/User/Desktop/ML-P/Data.csv')

#first few rows

telcom.head()



#Data_Overview

print ("Rows     : " ,telcom.shape[0])

print ("Columns  : " ,telcom.shape[1])

print ("\nFeatures : \n" ,telcom.columns.tolist())

print ("\nMissing values :  ", telcom.isnull().sum().values.sum())

print ("\nUnique values :  \n",telcom.nunique())
#telcom.info()
#Data Manipulation





#Replacing spaces with null values in total charges column

telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ",np.nan)



#telcom.info()

#print(telcom['TotalCharges'].notnull())



#Dropping null values from total charges column which contain .15% missing data 

telcom = telcom[telcom["TotalCharges"].notnull()]

telcom = telcom.reset_index()[telcom.columns]   

   

#convert string to float type

telcom["TotalCharges"] = telcom["TotalCharges"].astype(dtype=np.float64)



#replace 0 and 1 values to Yes and No

telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1:"Yes",0:"No"}) 

    



#replace 'No internet service' to No for the following columns

replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

                'TechSupport','StreamingTV', 'StreamingMovies']

for i in replace_cols : 

    telcom[i]  = telcom[i].replace({'No internet service' : 'No'}) 

    

#replace 'No phone service' to No for the MultipleLines column

#replace_cols = [ 'MultipleLines']

#for i in replace_cols : 

    #telcom[i]  = telcom[i].replace({'No phone service' : 'No'})     



#Tenure to categorical column

def tenure_lab(telcom) :

    

    if telcom["tenure"] <= 12 :

        return "Tenure_0-12"

    elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24 ):

        return "Tenure_12-24"

    elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48) :

        return "Tenure_24-48"

    elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60) :

        return "Tenure_48-60"

    elif telcom["tenure"] > 60 :

        return "Tenure_gt_60"

telcom["tenure_group"] = telcom.apply(lambda telcom:tenure_lab(telcom),axis = 1)

#telcom.info()
####Data Exploring and Visualisation####
#Data Preprocessing





#Separating churn and non churn customers

#Separating catagorical and numerical columns

#Separating Binary columns with 2 values and more than 3 values

churn     = telcom[telcom["Churn"] == "Yes"]

not_churn = telcom[telcom["Churn"] == "No"]

#customer id col

Id_col     = ['customerID']

#Target columns

target_col = ["Churn"]

#categorical columns

cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()

cat_cols   = [x for x in cat_cols if x not in target_col]

#numerical columns

num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]

#Binary columns with 2 values

bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()

#Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]

#Label encoding Binary columns

le = LabelEncoder()

for i in bin_cols :

    telcom[i] = le.fit_transform(telcom[i])

    

#Duplicating columns for multi value columns

telcom = pd.get_dummies(data = telcom,columns = multi_cols )

#Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(telcom[num_cols])

scaled = pd.DataFrame(scaled,columns=num_cols)

#dropping original values merging scaled values for numerical columns

df_telcom_og = telcom.copy()

telcom = telcom.drop(columns = num_cols,axis = 1)

telcom = telcom.merge(scaled,left_index=True,right_index=True,how = "left")

#print (scaled)