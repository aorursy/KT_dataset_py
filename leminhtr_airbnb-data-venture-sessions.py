# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Session 2 : Data Cleansing



import os #appel system



import numpy as np

import pandas as pd
df_train=pd.read_csv("../input/train_users_2.csv") # training data

df_train.sample(n=5)
df_test=pd.read_csv("../input/test_users.csv") # données de test

df_test.sample(n=5)
# Concaténation du training et test data

df_all=pd.concat((df_train,df_test), axis=0, ignore_index=True)

df_all.head(n=5)

#/!\ : valeur nulle pour date_first_booking dans test data car pas encore de réservation (c'est ce qu'on essaye de prédire)
# Remove date_first_booking column

df_all.drop('date_first_booking',axis=1,inplace=True)
df_all.sample(n=5)
# Format datetime of date_account_created

df_all['date_account_created']=pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')
df_all.head(n=5)
# Format datetime of date_account_created

df_all['timestamp_first_active']=pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')
# Format datetime of timestamp_first_active

df_all.head(n=5)
def remove_age_outliers(x, min_value=15, max_value=90):

    if np.logical_or(x<=min_value, x>=max_value):

        return np.nan

    else:

        return x
# Sort age column

df_all['age']=df_all['age'].apply(lambda x: remove_age_outliers(x) if(not np.isnan(x)) else x) #apply : from 1st row to last

#si la valeur x n'est pas nan (<=> int) then apply(remove_age_outliers(x))

#else return x.
df_all['age'].head(50)

df_all['age'].fillna(-1, inplace=True)
df_all['age'].head(50)
# Convert age from float to age

df_all.age=df_all.age.astype(int)

df_all.sample(5)
#Function to count number of NaN values in df column

def check_NaN_Values_in_df(df):

    for col in df:

        nan_count=df[col].isnull().sum()

        

        if nan_count !=0:

            print(col + " -> " + str(nan_count) + " NaN Values")

        
check_NaN_Values_in_df(df_all)
df_all['first_affiliate_tracked'].fillna(-1, inplace=True)
df_all.sample(10)

df_all.drop('timestamp_first_active',axis=1, inplace=True)

df_all.sample(n=5)
# Drop all row where date_account_created<=2013/02/01



df_all=df_all[df_all['date_account_created']>'2013-02-01']

df_all.sample(5)
#Makedir output

if not os.path.exists("output"):

    os.makedirs("output")



#Export to CSV

df_all.to_csv("output/cleaned.csv", sep =',', index=False)