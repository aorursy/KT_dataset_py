import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd
#Read Dataset



telcom=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(telcom.info())

print ("Rows     : " ,telcom.shape[0])

print ("Columns  : " ,telcom.shape[1])

print ("\nFeatures : \n" ,telcom.columns.tolist())

print ("\nMissing values :  ", telcom.isnull().sum().values.sum())

print ("\nUnique values :  \n",telcom.nunique())

#telcom['TotalCharges'].mean() gives error due to string value ("") 

#replacing "" to NA value

telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ",np.nan)



#Removing NA

print(telcom['TotalCharges'].isnull().sum())

telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

telcom=telcom[telcom["TotalCharges"].notnull()]

telcom = telcom.reset_index()[telcom.columns]



# Replacing 'No internet service'to 'No' for many columns

replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

                'TechSupport','StreamingTV', 'StreamingMovies']

for i in replace_cols:

    telcom[i]=telcom[i].replace('No internet service','No')



#converting into categorical variable

telcom["SeniorCitizen"]=telcom["SeniorCitizen"].replace({1:"Yes",0:"No"})





#converting tenure into tenure group

def tenure_lab(telcom):

    if telcom["tenure"]<=12:

        return "Tenure_0-12"

    elif (telcom["tenure"]>12) & (telcom["tenure"]<=24):

        return "Tenure_12-24"

    elif (telcom["tenure"]>24) & (telcom["tenure"]<=48):

        return "Tenure_24-48"

    elif (telcom["tenure"]>48) & (telcom["tenure"]<=60):

        return "Tenure_48-60"

    elif (telcom["tenure"]>60):

        return "Tenure_gt-60"

telcom["tenure_group"]=telcom.apply(lambda telcom:tenure_lab(telcom),axis=1)





#Separating churn and not churn customer

churn=telcom[telcom["Churn"]=="Yes"]

not_churn=telcom[telcom["Churn"]=="No"]



#Separating Numerical and categorical columns

Id_col=['customerID']

target_col=['Churn']

cat_cols=telcom.nunique()[telcom.nunique()<6].keys().tolist()

cat_cols=[x for x in cat_cols if x not in target_col ]

num_cols=[x for x in telcom.columns if x not in cat_cols+Id_col+target_col]





lab=telcom["Churn"].value_counts().keys().tolist()

print(lab)

val = telcom["Churn"].value_counts().values.tolist()

print(val)

    
#Exploratory Data Analysis

import matplotlib.pyplot as plt

%matplotlib inline

lab=telcom["Churn"].value_counts().keys().tolist()

val=telcom["Churn"].value_counts().values.tolist()

plt.pie(val,labels=lab,shadow=True,radius=1,autopct='%1.1f%%')

plt.show()

# As per pie chart data seems imbalanced









#Univariate anlysis for categorical data

print(cat_cols)

def pie_plot(column):

    val=telcom[column].value_counts().values.tolist()

    lab=telcom[column].value_counts().keys().tolist()

    plt.pie(val,labels=lab,radius=1,autopct='%1.1f%%')

    plt.show()

for i in cat_cols:

 pie_plot(i)

    

    
#Univariate Analysis for Quantative variable

def histogram(column):

    plt.hist(telcom[column],rwidth=0.5)

    plt.suptitle(column+" attrition distribution", fontsize=10)

    plt.show()

for i in num_cols :

    histogram(i)

 

#Bivariate Analysis

import matplotlib.gridspec as gridspec

def plot_pie(column):

    lab1=churn[column].value_counts().keys().tolist()

    val1 = churn[column].value_counts().values.tolist()

    lab2=not_churn[column].value_counts().keys().tolist()

    val2 = not_churn[column].value_counts().values.tolist()

    the_grid = gridspec.GridSpec(2, 2)

    plt.subplot(the_grid[0, 0], aspect=1, title='churn')

    plt.axis("equal")

    plt.pie(val1,labels=lab1, shadow=True,radius=1,autopct='%1.1f%%')

    plt.subplot(the_grid[0, 1], aspect=1, title='not churn')

    plt.pie(val2,labels=lab2, shadow=True,radius=1,autopct='%1.1f%%')

    plt.suptitle(column+" attrition distribution", fontsize=10)

    plt.show()

for i in cat_cols :

    plot_pie(i)
def histogram(column):

    plt.xlabel(column)

    plt.hist([churn[column],not_churn[column]],rwidth=0.95, color=['green','orange'],label=['churn','not churn'])

    plt.legend()

    plt.suptitle(column+" attrition distribution", fontsize=10)

    plt.show()

for i in num_cols :

    histogram(i)
 #customer churn by tenure group

plt.xlabel("tenure_group")

plt.hist([churn["tenure_group"],not_churn["tenure_group"]],rwidth=0.95, color=['green','orange'],label=['churn','not churn'])

plt.legend()

plt.suptitle(" attrition distribution", fontsize=10)

plt.show()
#Data Preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



Id_col     = ['customerID']

target_col = ["Churn"]

cat_cols =telcom.nunique()[telcom.nunique() < 6].keys().tolist()

cat_cols=[x for x in cat_cols if x not in target_col]

num_cols=[x for x in telcom.columns if x not in target_col+cat_cols+Id_col]

bin_cols=telcom.nunique()[telcom.nunique()==2].keys().tolist()

multi_cols = [i for i in cat_cols if i not in bin_cols]

le=LabelEncoder()

for i in bin_cols:

    telcom[i]=le.fit_transform(telcom[i])

telcom = pd.get_dummies(data = telcom,columns = multi_cols )



#Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(telcom[num_cols])

scaled = pd.DataFrame(scaled,columns=num_cols)

print(scaled)



df_telcom_og = telcom.copy()

telcom = telcom.drop(columns = num_cols,axis = 1)

telcom = telcom.merge(scaled,left_index=True,right_index=True,how = "left")
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