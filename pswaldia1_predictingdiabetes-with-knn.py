import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings    #warnings to ignore any kind of warnings that we may recieve.

warnings.filterwarnings('ignore')
def display_all(df):

    '''

    input: dataframe

    description: it takes a dataframe and allows use to show a mentioned no. of rows and columns in the screen

    '''

    with pd.option_context("display.max_rows",10,"display.max_columns",9):  #you might want to change these numbers.

        display(df)
df=pd.read_csv('../input/diabetes.csv')

df.shape
display_all(df)
def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        

        return mis_val_table_ren_columns
missing_values_table(df)
features_with_missing_values=['BMI','SkinThickness','BloodPressure','Insulin','Glucose']

for i in features_with_missing_values:

    df[i]=df[i].replace(0,np.median(df[i].values))
target=df['Outcome'].values

df.drop(['Outcome'],inplace=True,axis=1)
#from sklearn importing standard scalar that will convert the provided dataframe into standardised one.

from sklearn.preprocessing import StandardScaler                                              

sta=StandardScaler()

input=sta.fit_transform(df)    #will give numpy array as output
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(input,target,test_size=0.1,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)