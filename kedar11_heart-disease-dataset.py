# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import numpy as np

from sklearn.linear_model import LogisticRegression

from collections import Counter

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/heart.csv')
#age distibution

df.age.unique()

sb.distplot(df.age)

plt.title('Age Distribution')
# Male Female percentage

sb.countplot(df.sex)

print('Total People[0,1]: ',len(df.sex))

total_male=[i for i in  df.sex if i=='Male']

print('Total Male[1]: ',len(total_male))

total_female=len(df.sex)-len(total_male)

print('Total Female[0]: ',total_female)

df.sex.replace(to_replace=[0,1],value=['Female','Male'],inplace=True)



male_percentage=(len(total_male)*100)/len(df.sex)

female_percentage=100-male_percentage

print('Percentage of Male: '+str(male_percentage)+' %')

print('Percentage of Female: '+str(female_percentage)+' %')
df.cp.replace(to_replace=[0,1,2,3],value=['general','sevior','moderate','critical'],inplace=True)

df.cp.unique()

general_cp=[i for i in  df.cp if i=='general']

sevior_cp=[i for i in  df.cp if i=='sevior']

moderate_cp=[i for i in  df.cp if i=='moderate']

critical_cp=[i for i in  df.cp if i=='critical']

print('People with Level-1 Pain :',len(general_cp))

print('People with Level-2 Pain :',len(moderate_cp))

print('People with Level-3 Pain :',len(sevior_cp))

print('People with Level-4 Pain :',len(critical_cp))



general_cpp=(len(general_cp)*100)/len(df.cp)

sevior_cpp=(len(sevior_cp)*100)/len(df.cp)

moderate_cpp=(len(moderate_cp)*100)/len(df.cp)

critical_cpp=(len(critical_cp)*100)/len(df.cp)

print('\n')

print('Percentage with Level-1 Pain :',str(general_cpp)+' %')

print('Percentage with Level-2 Pain :',str(sevior_cpp)+' %')

print('Percentage with Level-3 Pain :',str(moderate_cpp)+' %')

print('Percentage with Level-4 Pain :',str(critical_cpp)+' %')



labels=['Level-1:general','Level-2:sevior','Level-3:moderate','Level-4:critical']

values=[general_cpp,sevior_cpp,moderate_cpp,critical_cpp]

plt.pie(values,labels=labels,wedgeprops={'linewidth':3},autopct='%1.1f%%')

rcParams['figure.figsize']=(5,4)

plt.title('Pie-Chart for pain type')

plt.show()
# age group of suffering from critical chest pain who need urgent treatment

age_critical_cp=[]

index_critical_cp=[]  # index no. of people with critical chest pain

for i in range(len(df)-1):

    if df.cp[i]=='critical':

             index_critical_cp.append(i)

             age_critical_cp.append(df.age[i])

sb.countplot(age_critical_cp)

plt.xlabel('Age')

plt.title('age group having critical chest pain')

print('list of age group with critical chest pain :',list(Counter(age_critical_cp)))

sb.regplot(x=df.trestbps,y=df.chol)

plt.title('Blood Pressure vs Cholestrol')
# identity of people who have critical chest pain and have very high BP and Cholestrol

index_highbp_chol=[]

index_highbp_cp_c=set

for i in range(len(df)-1):

    if df.trestbps[i]>140 and df.chol[i]>280:

        index_highbp_chol.append(i)

print('People who need immediate treatment : ',

      index_highbp_cp_c.intersection(set(index_highbp_chol),set(index_critical_cp)))

sb.countplot(df.fbs )

plt.title('fasting blood sugar') # 1:(>120 mg) and 0:(<120 mg)

male_with_fbs=[]

female_with_fbs=[]

for i in range(len(df)-1):

    if df.fbs[i]==1 and df.sex[i]=='Male':

        male_with_fbs.append(i)

    elif df.fbs[i]==1 and df.sex[i]=='Female':

        female_with_fbs.append(i)

print('Total patients with positive sugar: ',len([i for i in df.fbs if i==1]))

print('No. of Males with positive sugar: ',len(male_with_fbs))

print('No. of Females with positive sugar: ',len(female_with_fbs))
#Relation between thalach and exang

df.exang.replace(to_replace=[0,1],value=['No','Yes'],inplace=True)

positive_exang=[i for i in df.exang if i=='Yes']

index_with_high_thalach_exang=[]

for i in range(len(df)-1):

    if df.thalach[i]>df.thalach.median()+20 and df.exang[i]=='Yes':

        index_with_high_thalach_exang.append(i) # index(ID) of people with high thalach and have exang(sevior pain)

print('People with this index have have symptoms of heart-rate problem :',list(index_with_high_thalach_exang))

sb.countplot(df.age[index_with_high_thalach_exang])
#Logistic Regression between trestbps and fbs

logreg=LogisticRegression()

X=df.trestbps.values.reshape(-1,1)

y=df.fbs

logreg.fit(X,y)

print('score of Regression : '+str(logreg.score(X,y)*100)+' %')

sb.regplot(X,y,data=df,logistic=True)
#People with sevior heart problem and having heart defect too

df.thal.replace(to_replace=[1,2,3],value=['Normal','Fixed Defect','reversable defect'],inplace=True)

defect=[]

for i in range(len(df)-1):

    if df.thal[i]=='Fixed Defect':

        defect.append(i)

critical_patient=set

critical_patient.intersection(set(defect),set(index_with_high_thalach_exang))

print('Patient who need urgent treatment: ',critical_patient.intersection(set(defect),set(index_with_high_thalach_exang)))
sb.set()

sb.pairplot(df[['age','sex','trestbps','chol','thalach','thal','target']])