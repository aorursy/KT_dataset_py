# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt
#Importing the data set

data =pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
data.head()
data.tail()
data.info()
data.isnull().sum()
data.dtypes
data.describe()
#x is our features data frame

x = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction',

         'Age']]

y = data['Outcome']
print(y.value_counts())

sns.countplot(y)

plt.xticks(range(len(data['Outcome'].unique())))

plt.xlabel('Outcome')

plt.ylabel('Count')

plt.show()
x.describe()
#We will first normalize our x data frame so that our all features can come on same scale

data_std = (x - x.mean()) / x.std()

data_std.describe()
data_part = pd.concat([y,data_std],axis=1)

data_part.head()
data_part = pd.melt(data_part,id_vars = 'Outcome',

                   var_name = 'features',

                   value_name = 'value')

data_part.head()
plt.figure(figsize=(18,10))

ax = sns.violinplot(x='features',y='value',hue = 'Outcome',data = data_part,split=True,inner = 'quart')

plt.xticks(rotation = 45)

plt.show(ax)
plt.figure(figsize=(18,10))

ax = sns.boxplot(x='features',y='value',hue = 'Outcome',data = data_part)

plt.xticks(rotation = 45)

plt.show(ax)
#Now we will analyze x

sns.heatmap(x.corr(),annot =True)

plt.figure(figsize=(100,100))

plt.show()
#Lets visualize age variable

sns.countplot(data['Pregnancies'])

plt.show()
preg = pd.concat([y,data_std['Pregnancies']],axis=1)

preg = pd.melt(preg,id_vars='Outcome',var_name='Pregnancies',value_name='values')

preg.head()
plt.figure(figsize=(10,10))

sns.swarmplot(x='Pregnancies',y='values',hue='Outcome',data=preg)

plt.show()
# Lets visualize the relation between Pregnancy and other variables

fig, axarr = plt.subplots(3, 2, figsize=(12, 12))

plt.xticks(np.arange(max(data['Pregnancies'])+1))

sns.lineplot(x='Pregnancies',y='Glucose',data = x,ax=axarr[0][0]).set(title="Pregnancies And Glucose")

sns.lineplot(x='Pregnancies',y='BloodPressure',data = x,ax=axarr[0][1]).set(title="Pregnancies And Blood Pressure")

sns.lineplot(x='Pregnancies',y='SkinThickness',data = x,ax=axarr[1][0]).set(title="Pregnancies And Skin Thickness")

sns.lineplot(x='Pregnancies',y='Insulin',data = x,ax=axarr[1][1]).set(title="Pregnancies And Insuline")

sns.lineplot(x='Pregnancies',y='BMI',data = x,ax=axarr[2][0]).set(title="Pregnancies And BMI")

sns.lineplot(x='Pregnancies',y='DiabetesPedigreeFunction',data = x,ax=axarr[2][1]).set(title="Pregnancies And Diabetes Pedigree Function")

plt.subplots_adjust(hspace=.9)

plt.show()

sns.lineplot(x='Pregnancies',y='Age',data = x).set(title="Pregnancies And Age")

plt.show()
#let see how many distinct values we have in glucose an visualize it

x['Glucose'].unique()
plt.figure(figsize=(20, 25))

sns.countplot(data['Glucose'])

plt.show()
Glucose = pd.concat([y,data_std['Glucose']],axis=1)

Glucose= pd.melt(Glucose,id_vars='Outcome',var_name='Glucose',value_name='values')

Glucose.head()
plt.figure(figsize=(10,10))

sns.swarmplot(x='Glucose',y='values',hue='Outcome',data = Glucose)

plt.show()
# Lets visualize the relation between Glucose and other variables

fig, axarr = plt.subplots(3, 2, figsize=(12, 12))

sns.scatterplot(x='Glucose',y='BloodPressure',data = x,ax=axarr[0][0]).set(title="Glucose And Blood Pressure")

sns.scatterplot(x='Glucose',y='SkinThickness',data = x,ax=axarr[0][1]).set(title="Glucose And Skin Thickness")

sns.scatterplot(x='Glucose',y='Insulin',data = x,ax=axarr[1][0]).set(title="Glucose And Insuline")

sns.scatterplot(x='Glucose',y='BMI',data = x,ax=axarr[1][1]).set(title="Glucose And BMI")

sns.scatterplot(x='Glucose',y='DiabetesPedigreeFunction',data = x,ax=axarr[2][0]).set(title="Glucose And Diabetes Pedigree Function")

sns.scatterplot(x='Glucose',y='Age',data = x,ax=axarr[2][1]).set(title="Glucose And Age")

plt.subplots_adjust(hspace=.9)

plt.show()
x['BloodPressure'].unique()
plt.figure(figsize=(20,10))

sns.countplot(x['BloodPressure'])

plt.show()
Bp = pd.concat([y,data_std['BloodPressure']],axis=1)

Bp= pd.melt(Bp,id_vars='Outcome',var_name='Bp',value_name='values')

Bp.head()
plt.figure(figsize=(10,10))

sns.swarmplot(x='Bp',y='values',hue='Outcome',data=Bp)

plt.show()
# Lets visualize the relation between Blood Pressure and other variables

fig, axarr = plt.subplots(2, 2, figsize=(12, 12))

sns.scatterplot(x='BloodPressure',y='SkinThickness',data = x,ax=axarr[0][0]).set(title="Blood Pressure And Skin Thickness")

sns.scatterplot(x='BloodPressure',y='Insulin',data = x,ax=axarr[0][1]).set(title="Blood Pressure And Insuline")

sns.scatterplot(x='BloodPressure',y='BMI',data = x,ax=axarr[1][0]).set(title="Blood Pressure And BMI")

sns.scatterplot(x='BloodPressure',y='DiabetesPedigreeFunction',data = x,ax=axarr[1][1]).set(title="Blood Pressure And Diabetes Pedigree Function")

plt.subplots_adjust(hspace=.9)

plt.show()

sns.scatterplot(x='BloodPressure',y='Age',data = x).set(title="Blood Pressure And Age")

plt.show()
#Lets visualize the count plot of skin Thickness

plt.figure(figsize=(10,10))

sns.countplot(x['SkinThickness'])

plt.show()
St = pd.concat([y,data_std['SkinThickness']],axis = 1)

St =pd.melt(St,id_vars = 'Outcome',var_name ='SkinThickness',value_name='values')

St.head()
plt.figure(figsize=(10,10))

sns.swarmplot(x='SkinThickness',y='values',hue='Outcome',data = St)

plt.show()
# Lets visualize the relation between Skin Thickness and other variables

fig, axarr = plt.subplots(2, 2, figsize=(12, 12))

sns.scatterplot(x='SkinThickness',y='Insulin',data = x,ax=axarr[0][0]).set(title="Skin Thickness And Insulin")

sns.scatterplot(x='SkinThickness',y='BMI',data = x,ax=axarr[0][1]).set(title="Skin Thickness And BMI")

sns.scatterplot(x='SkinThickness',y='DiabetesPedigreeFunction',data = x,ax=axarr[1][0]).set(title="Skin Thickness And Diabetes Pedigree Function")

sns.lineplot(x='SkinThickness',y='Age',data = x,ax=axarr[1][1]).set(title="Skin Thickness And Age")

plt.subplots_adjust(hspace=.9)

plt.show()
x['Insulin'].sort_values().value_counts()
plt.figure(figsize=(10,10))

sns.countplot(x['Insulin'].sort_values())

plt.show()
insulin = pd.concat([y,data_std['Insulin']],axis = 1)

insulin = pd.melt(insulin,id_vars='Outcome',var_name='insulin',value_name='value')
plt.figure(figsize=(10,10))

sns.swarmplot(x='insulin',y='value',hue='Outcome',data = insulin)

plt.show()
# Lets visualize the relation between Insulin and other variables

sns.scatterplot(x='Insulin',y='BMI',data = x).set(title="Insulin And BMI")

plt.show()

sns.scatterplot(x='Insulin',y='DiabetesPedigreeFunction',data=x).set(title="Insulin And Diabetes Pedigree Function")

plt.show()

sns.lineplot(x='Insulin',y='Age',data = x).set(title="Insulin And Age")

plt.show()
x['BMI'].sort_values().value_counts()
plt.figure(figsize=(30,10))

sns.countplot(x['BMI'])

plt.show()
insulin = pd.concat([y,data_std['BMI']],axis = 1)

insulin = pd.melt(insulin,id_vars='Outcome',var_name='BMI',value_name='value')
plt.figure(figsize=(10,10))

sns.swarmplot(x='BMI',y='value',hue='Outcome',data = insulin)

plt.show()
# Lets visualize the relation between Insulin and other variables

sns.scatterplot(x='BMI',y='DiabetesPedigreeFunction',data=x).set(title="Insulin And Diabetes Pedigree Function")

plt.show()

sns.lineplot(x='BMI',y='Age',data = x).set(title="Insulin And Age")

plt.show()
x['DiabetesPedigreeFunction'].sort_values().value_counts()
plt.figure(figsize=(30,10))

sns.countplot(x['DiabetesPedigreeFunction'])

plt.show()
DPF = pd.concat([y,data_std['DiabetesPedigreeFunction']],axis = 1)

DPF = pd.melt(DPF,id_vars='Outcome',var_name='DPF',value_name='value')
plt.figure(figsize=(10,10))

sns.swarmplot(x='DPF',y='value',hue='Outcome',data = DPF)

plt.show()
# Lets visualize the relation between Insulin and other variables

sns.scatterplot(x='DiabetesPedigreeFunction',y='Age',data=x).set(title="Diabetes Pedigree Function and Age")

plt.show()
x['Age'].value_counts()
plt.figure(figsize=(20,10))

sns.countplot(x['Age'])

plt.show()
# **6. Realtion of BMI variable with other independent variable**
Age = pd.concat([y,data_std['Age']],axis = 1)

Age = pd.melt(Age,id_vars='Outcome',var_name='Age',value_name='value')
Age.head()
plt.figure(figsize=(10,10))

sns.swarmplot(x='Age',y='value',hue='Outcome',data = Age)

plt.show()