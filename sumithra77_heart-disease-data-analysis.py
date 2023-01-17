import numpy as np

from scipy.stats import chisquare

from scipy.stats import chi2

import pandas as pd

import scipy.stats as stats





# visualization

import seaborn as sns

from pandas.plotting import scatter_matrix

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



# And some function we will need

import statsmodels.api as sm

from statsmodels.discrete.discrete_model import Logit

from scipy.special import logit



# machine learning

from sklearn.model_selection import train_test_split

from sklearn import datasets, linear_model

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from pylab import rcParams



import itertools

import warnings

warnings.filterwarnings("ignore")

import io
Heart_df= pd.read_csv('../input/heart.csv')
Heart_df.head(10)
Heart_rows = Heart_df.shape[0]

Heart_columns = Heart_df.shape[1]



print('Number of rows in Heart Disease data set: ',Heart_rows)

print('Number of columns in Heart Disease data set: ',Heart_columns)
Heart_df.columns
Heart_df.describe()
Heart_df.info()
Heart_df.isnull().sum()
#**thal:** 2 = normal; 1 = fixed defect; 3 = reversable defect; 0 = missing values

# Finding the records with missing values

Heart_df[Heart_df.thal == 0].index
Heart_df.loc[[48,281], :]
Heart_df.loc[(Heart_df.age>=50)&(Heart_df.cp==2)&(Heart_df.sex==0)&(Heart_df.ca==0)&(Heart_df.target==1)&(Heart_df.restecg==0)

            &(Heart_df.slope==2)]
Heart_df.loc[(Heart_df.age>=50)&(Heart_df.cp==0)&(Heart_df.sex==1)&(Heart_df.ca==0)&(Heart_df.target==0)&(Heart_df.restecg==1)

            &(Heart_df.slope==1)]
Heart_df.loc[48, 'thal']=int(2)

Heart_df.loc[281, 'thal']=int(2)
Heart_df.hist(figsize=(16,15))
HaveDisease = len(Heart_df[Heart_df.target == 0])

NoDisease = len(Heart_df[Heart_df.target == 1])

print("Number of people having heart disease : ",HaveDisease)

print("Number of people having no heart disease : ",NoDisease)

print("Percentage of patients who don't have heart disease: {:.2f}%".format((NoDisease / (len(Heart_df.target))*100)))

print("Percentage of patients who have heart disease: {:.2f}%".format((HaveDisease / (len(Heart_df.target))*100)))
fig,ax=plt.subplots(figsize=(10,4))

plt.subplot(1, 2, 1)

g1=sns.countplot(x="target", data=Heart_df)

g1.set_ylabel('Frequency')

g1.set_xlabel('Target')



plt.subplot(1, 2, 2)

g2=sns.countplot(x='sex',hue='target',data=Heart_df)

g2.set_ylabel('Frequency')

g2.set_xlabel('sex')
# Target Verses continuous variables



fig,ax=plt.subplots(figsize=(32,8))

plt.subplot(1, 4, 1)

cho_bins = [100,150,200,250,300,350,400,450]

Heart_df['bin_chol']=pd.cut(Heart_df['chol'], bins=cho_bins)

g1=sns.countplot(x='bin_chol',data=Heart_df,hue='target',palette='plasma',linewidth=3)

g1.set_title("Cholestoral vs Heart Disease")



plt.subplot(1, 4, 2)

thal_bins = [60,80,100,120,140,160,180,200,220]

Heart_df['bin_thalch']=pd.cut(Heart_df['thalach'], bins=thal_bins)

g2=sns.countplot(x='bin_thalch',data=Heart_df,hue='target',palette='plasma',linewidth=3)

g2.set_title("Thalach vs Heart Disease")



plt.subplot(1, 4, 3)

trestbps_bins = [60,80,100,120,140,160,180,200,220]

Heart_df['bin_trestbps']=pd.cut(Heart_df['trestbps'], bins=trestbps_bins)

g3=sns.countplot(x='bin_trestbps',data=Heart_df,hue='target',palette='plasma',linewidth=3)

g3.set_title("Trestbps vs Heart Disease")



plt.subplot(1, 4, 4)

oldpeak_bins= [0,1,2,3,4,5,6]

Heart_df['bin_oldpeak']=pd.cut(Heart_df['oldpeak'], bins=oldpeak_bins)

g4=sns.countplot(x='bin_oldpeak',data=Heart_df,hue='target',palette='plasma',linewidth=3)

g4.set_title("oldpeak vs Heart Disease")

plt.show()

#Target verses categorical variables

# thal, slope , restecg

fig,ax=plt.subplots(figsize=(30,8))





plt.subplot(141)

x1=sns.countplot(x='thal',data=Heart_df,hue='target',palette='YlOrRd',linewidth=3)

x1.set_title('Thal Vs Target')



plt.subplot(142)

x2=sns.countplot(x='slope',data=Heart_df,hue='target',palette='YlOrRd',linewidth=3)

x2.set_title('slope Vs Target')



plt.subplot(143)

x3=sns.countplot(x='restecg',data=Heart_df,hue='target',palette='YlOrRd',linewidth=3)

x3.set_title('Restecg Vs Target')



plt.subplot(144)

x4=sns.countplot(x='ca',data=Heart_df,hue='target',palette='YlOrRd',linewidth=3)

x4.set_title('Ca Vs Target')

plt.show()





plt.figure(figsize=(8,6))

labels = 'No','Yes'

sizes = [len(Heart_df[Heart_df['exang'] == 0]),len(Heart_df[Heart_df['exang'] == 1])]

colors = ['skyblue', 'yellowgreen']

explode = (0.1, 0)  # explode 1st slice



plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', 

        shadow=True, startangle=90)

plt.axis('equal')

plt.title('Exang')

plt.show()
pd.crosstab([Heart_df.sex], [Heart_df.target])

ct_sex = pd.crosstab([Heart_df.sex], [Heart_df.target], normalize='index')

ct_sex

pd.crosstab([Heart_df.fbs], [Heart_df.target])

ct_fbs = pd.crosstab([Heart_df.fbs], [Heart_df.target], normalize='index')

ct_fbs

pd.crosstab([Heart_df.restecg], [Heart_df.target])

ct_ecg = pd.crosstab([Heart_df.restecg], [Heart_df.target], normalize='index')

ct_ecg

pd.crosstab([Heart_df.exang], [Heart_df.target])

ct_exang = pd.crosstab([Heart_df.exang], [Heart_df.target], normalize='index')

ct_exang

pd.crosstab([Heart_df.slope], [Heart_df.target])

ct_slope = pd.crosstab([Heart_df.slope], [Heart_df.target], normalize='index')

ct_slope

pd.crosstab([Heart_df.ca], [Heart_df.target])

ct_ca = pd.crosstab([Heart_df.ca], [Heart_df.target], normalize='index')

ct_ca

pd.crosstab([Heart_df.thal], [Heart_df.target])

ct_thal = pd.crosstab([Heart_df.thal], [Heart_df.target], normalize='index')

ct_thal

pd.crosstab([Heart_df.cp], [Heart_df.target])

ct_cp = pd.crosstab([Heart_df.cp], [ Heart_df.target], normalize='index')

ct_cp
Heart_df.hist("age", bins=75)
plt.figure(figsize=(15,6))

sns.countplot(x='age',data = Heart_df, hue = 'target',palette='GnBu')

plt.show()
plt.subplot(1,3,1)

Heart_df.age.hist(figsize=(12,4))

plt.title('Age for all')

plt.subplot(1,3,2)

Heart_df[Heart_df.target == 0].age.hist()

plt.title('Age for Target = 0')

plt.subplot(1,3,3)

Heart_df[Heart_df.target == 1].age.hist()

plt.title('Age for Target = 1')
Heart_df['AgeRange']=0



youngAge=Heart_df[(Heart_df.age>=29)&(Heart_df.age<=40)].index

middleAge=Heart_df[(Heart_df.age>40)&(Heart_df.age<=55)].index

elderlyAge=Heart_df[(Heart_df.age>55)].index





#Assigning values to different agerange based on the agegroups

Heart_df.loc[youngAge, 'AgeRange'] = 0

Heart_df.loc[middleAge, 'AgeRange'] = 1

Heart_df.loc[elderlyAge, 'AgeRange'] = 2

#Heart_df['AgeRange'] = Heart_df['AgeRange'].astype(int)



# Target based on the agegroup



#Youngeage

youngeage_0=len(Heart_df[(Heart_df.target==0)&(Heart_df.AgeRange==0)])

youngeage_1=len(Heart_df[(Heart_df.target==1)&(Heart_df.AgeRange==0)])

print(" There are {} patients with heart disease compared to {} patients without it among {} youngsters "

      .format(youngeage_0,youngeage_1,len(youngAge))) 



middleage_0=len(Heart_df[(Heart_df.target==0)&(Heart_df.AgeRange==1)])

middleage_1=len(Heart_df[(Heart_df.target==1)&(Heart_df.AgeRange==1)])

print(" There are {} patients with heart disease compared to {} patients without it among {} middle aged "

      .format(middleage_0,middleage_1,len(middleAge))) 



elderlyage_0=len(Heart_df[(Heart_df.target==0)&(Heart_df.AgeRange==2)])

elderlyage_1=len(Heart_df[(Heart_df.target==1)&(Heart_df.AgeRange==2)])

print(" There are {} patients with heart disease compared to {} patients without it among {} eldely aged "

      .format(elderlyage_0,elderlyage_1,len(elderlyAge))) 





# Plotting the above data

pd.crosstab(Heart_df.AgeRange,Heart_df.target).plot(kind="bar",figsize=(8,4))

plt.title('Heart Disease Frequency for AgeGroup')

plt.xlabel('Agegroups (0-young ages 1-middle ages 2-elderly ages)')

plt.xticks(rotation = 0)

plt.ylabel('Frequency')

plt.show()

sns.swarmplot(x="AgeRange", y="age",hue='sex',

              palette=["r", "c", "y"], data=Heart_df)

from matplotlib.gridspec import GridSpec



plt.figure(1, figsize=(14,8))

the_grid = GridSpec(3, 3)

colors = ['blue','green']

explode = [0,0]



plt.subplot(the_grid[0, 0], aspect=1, title='Youngsters')

plt.pie([youngeage_0,youngeage_1], explode=explode, labels=['Disease ','No Disease'],

        colors=colors, autopct='%1.1f%%')

plt.title(' Youngsters ',color = 'blue',fontsize = 12)



plt.subplot(the_grid[0, 1], aspect=1, title='Middleagers')

plt.pie([middleage_0,middleage_1], explode=explode, labels=['Disease ','No Disease'], 

        colors=colors, autopct='%1.1f%%')

plt.title('Middleagers ',color = 'blue',fontsize = 12)



plt.subplot(the_grid[0, 2], aspect=1, title='Middleagers')

plt.pie([elderlyage_0,elderlyage_1], explode=explode, labels=[' Disease ','No Disease'], 

        colors=colors, autopct='%1.1f%%')

plt.title(' Elderlyagers ',color = 'blue',fontsize = 12)



plt.suptitle('Percentage of Heart Disease found in different age groups', fontsize=16)

plt.show()



male_count=len(Heart_df[Heart_df['sex']==1])

female_count=len(Heart_df[Heart_df['sex']==0])



fig, (ax1,ax2) = plt.subplots(1,2,figsize = (12,5),constrained_layout=True)

plt.subplots_adjust(wspace = 0.5)



ax1.bar(Heart_df.sex.unique(),Heart_df.sex.value_counts(),width = 0.8)

ax1.set_xticks(Heart_df.sex.unique())

ax1.set_xticklabels(('Male','Female'))



ax2.pie((male_count,female_count), labels = ('Male','Female'), autopct='%1.1f%%', shadow=True, startangle=90, explode=[0,0.3])

plt.show()



print("There are {} female and {} male patients out of {} patients in total".

      format(female_count,male_count,len(Heart_df.sex)))



print("The percentage of female and male patients are {:.2f} and {:.2f} ".

      format((female_count / (len(Heart_df.sex))*100),(male_count / (len(Heart_df.sex))*100)))



plt.subplot(2,3,1)

Heart_df[Heart_df.sex == 1].age.hist(figsize=(12,10))

plt.title('Age for males')

plt.subplot(2,3,2)

Heart_df[(Heart_df.target == 0) & (Heart_df.sex == 1)].age.hist()

plt.title('Age for Target = 0 males')

plt.subplot(2,3,3)

Heart_df[(Heart_df.target == 1) & (Heart_df.sex == 1)].age.hist()

plt.title('Age for Target = 1 males')



plt.subplot(2,3,4)

Heart_df[Heart_df.sex == 0].age.hist()

plt.title('Age for females')

plt.subplot(2,3,5)

Heart_df[(Heart_df.target == 0) & (Heart_df.sex == 0)].age.hist()

plt.title('Age for Target = 0 females')

plt.subplot(2,3,6)

Heart_df[(Heart_df.target == 1) & (Heart_df.sex == 0)].age.hist()

plt.title('Age for Target = 1 females')
data= Heart_df

agerange='AgeRange'

female_young=len(data[(data.AgeRange==0)&(data.sex==0)])

male_young =len(data[(data.AgeRange==0)&(data.sex==1)])



female_middle=len(data[(data.AgeRange==1)&(data.sex==0)])

male_middle =len(data[(data.AgeRange==1)&(data.sex==1)])



female_elderly=len(data[(data.AgeRange==2)&(data.sex==0)])

male_elderly =len(data[(data.AgeRange==2)&(data.sex==1)])



female_young_0=len(data[(data.target==0)&(data.AgeRange==0)&(data.sex==0)])

female_young_1=len(data[(data.target==1)&(data.AgeRange==0)&(data.sex==0)])

male_young_0=len(data[(data.target==0)&(data.AgeRange==0)&(data.sex==1)])

male_young_1=len(data[(data.target==1)&(data.AgeRange==0)&(data.sex==1)])



female_middle_0=len(data[(data.target==0)&(data.AgeRange==1)&(data.sex==0)])

female_middle_1=len(data[(data.target==1)&(data.AgeRange==1)&(data.sex==0)])

male_middle_0=len(data[(data.target==0)&(data.AgeRange==1)&(data.sex==1)])

male_middle_1=len(data[(data.target==1)&(data.AgeRange==1)&(data.sex==1)])



female_elderly_0=len(data[(data.target==0)&(data.AgeRange==2)&(data.sex==0)])

female_elderly_1=len(data[(data.target==1)&(data.AgeRange==2)&(data.sex==0)])

male_elderly_0=len(data[(data.target==0)&(data.AgeRange==2)&(data.sex==1)])

male_elderly_1=len(data[(data.target==1)&(data.AgeRange==2)&(data.sex==1)])





print("** The tabular data of findings of disease among the 303 Patients**")

from tabletext import to_text



results =[["Agegroup", "Female","Male","Female-0","Female-1","Male-0","Male-1"],

["Youngage", female_young, male_young,female_young_0,female_young_1,

 male_young_0, male_young_1],

["Middleage", female_middle, male_middle, female_middle_0, 

          female_middle_1, male_middle_0, male_middle_1],

["Elderlyage", female_elderly, male_elderly, female_elderly_0, 

          female_elderly_1, female_elderly_0, male_elderly_1]]

print (to_text(results))
plt.subplot(2,3,1)

Heart_df[Heart_df.sex == 1].chol.hist(figsize=(12,10))

plt.title('chol male')

plt.subplot(2,3,2)

Heart_df[(Heart_df.target == 0) & (Heart_df.sex == 1)].chol.hist()

plt.title('chol Target = 0 for males')

plt.subplot(2,3,3)

Heart_df[(Heart_df.target == 1) & (Heart_df.sex == 1)].chol.hist()

plt.title('chol Target = 1 for males')



plt.subplot(2,3,4)

Heart_df[Heart_df.sex == 0].chol.hist()

plt.title('chol female')

plt.subplot(2,3,5)

Heart_df[(Heart_df.target == 0) & (Heart_df.sex == 0)].chol.hist()

plt.title('chol Target = 0 for females')

plt.subplot(2,3,6)

Heart_df[(Heart_df.target == 1) & (Heart_df.sex == 0)].chol.hist()

plt.title('chol Target = 1 for females')
plt.subplot(2,3,1)

Heart_df[Heart_df.sex == 1].thalach.hist(figsize=(12,10))

plt.title('max heart rate male')

plt.subplot(2,3,2)

Heart_df[(Heart_df.target == 0) & (Heart_df.sex == 1)].thalach.hist()

plt.title('max heart-rate Target = 0 for males')

plt.subplot(2,3,3)

Heart_df[(Heart_df.target == 1) & (Heart_df.sex == 1)].thalach.hist()

plt.title('max heart-rate Target = 1 for males')



plt.subplot(2,3,4)

Heart_df[Heart_df.sex == 0].thalach.hist()

plt.title('max heart rate female')

plt.subplot(2,3,5)

Heart_df[(Heart_df.target == 0) & (Heart_df.sex == 0)].thalach.hist()

plt.title('max heart-rate Target = 0 for females')

plt.subplot(2,3,6)

Heart_df[(Heart_df.target == 1) & (Heart_df.sex == 0)].thalach.hist()

plt.title('max heart-rate Target = 1 for females')
plt.subplot(2,3,1)

Heart_df[Heart_df.sex == 1].oldpeak.hist(figsize=(12,10))

plt.title('oldpeak male')

plt.subplot(2,3,2)

Heart_df[(Heart_df.target == 0) & (Heart_df.sex == 1)].oldpeak.hist()

plt.title('oldpeak Target = 0 for males')

plt.subplot(2,3,3)

Heart_df[(Heart_df.target == 1) & (Heart_df.sex == 1)].oldpeak.hist()

plt.title('oldpeak Target = 1 for males')



plt.subplot(2,3,4)

Heart_df[Heart_df.sex == 0].oldpeak.hist()

plt.title('oldpeak female')

plt.subplot(2,3,5)

Heart_df[(Heart_df.target == 0) & (Heart_df.sex == 0)].oldpeak.hist()

plt.title('oldpeak Target = 0 for females')

plt.subplot(2,3,6)

Heart_df[(Heart_df.target == 1) & (Heart_df.sex == 0)].oldpeak.hist()

plt.title('oldpeak Target = 1 for females')
plt.figure(figsize=(8,6))





labels = 'Chest Pain Type:0','Chest Pain Type:1','Chest Pain Type:2','Chest Pain Type:3'

sizes = [len(Heart_df[Heart_df['cp'] == 0]),len(Heart_df[Heart_df['cp'] == 1]),

         len(Heart_df[Heart_df['cp'] == 2]),

         len(Heart_df[Heart_df['cp'] == 3])]

colors = ['skyblue', 'yellowgreen','orange','gold']

explode = (0, 0,0,0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=180)

 

plt.axis('equal')

plt.show()
pd.crosstab([Heart_df.cp], [Heart_df.target])

ct_cp = pd.crosstab([Heart_df.cp], [ Heart_df.target], normalize='index')

ct_cp
pd.crosstab(Heart_df.cp,data.target).plot(kind="bar",figsize=(15,6),color=['brown','pink' ])

plt.title('Heart Disease Frequency According To chest pain type')

plt.xlabel('chest pain type - (0-asymptomatic 1-atypical angina 2-non-anginal pain 3-typical angina pain)')

plt.xticks(rotation = 0)

plt.legend(["Disease", " No Disease"])

plt.ylabel('Frequency')

plt.show()
g = sns.FacetGrid(Heart_df, col="cp",  row="sex")

g = g.map(plt.hist, "target",  color="r")
g = sns.FacetGrid(Heart_df, col='cp')

g.map(plt.hist, 'chol', bins=10)



g = sns.FacetGrid(Heart_df, col='cp')

g.map(plt.hist, 'trestbps', bins=10)



g = sns.FacetGrid(Heart_df, col='cp')

g.map(plt.hist, 'restecg', bins=10)



g = sns.FacetGrid(Heart_df, col='cp')

g.map(plt.hist, 'exang', bins=10)



g = sns.FacetGrid(Heart_df, col='cp')

g.map(plt.hist, 'ca', bins=10)
sns.countplot(x='fbs',data=Heart_df,hue='target',palette='YlOrRd',linewidth=3)

plt.title('Fbs Vs Target')
sns.barplot(x = 'fbs', y = 'target', hue = 'sex', data=Heart_df)

ax.set_title('Fbs Type vs Target based on gender ')
plt.figure(1, figsize=(10,8))

the_grid = GridSpec(4, 4)



labels = 'fbs < 120 mg/dl','fbs > 120 mg/dl'

sizes = [len(Heart_df[Heart_df['fbs'] == 0]),len(Heart_df[Heart_df['cp'] == 0])]

colors = ['skyblue', 'yellowgreen','orange','gold']

explode = (0.1, 0.1)  # explode 1st slice



plt.subplot(the_grid[0, 0], aspect=1, title='cp-0')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', 

        shadow=True, startangle=180)

plt.axis('equal')

plt.title(' cp-0 ',color = 'blue',fontsize = 12)



sizes = [len(Heart_df[Heart_df['fbs'] == 0]),len(Heart_df[Heart_df['cp'] == 1])]

plt.subplot(the_grid[0, 1], aspect=1, title='cp-1')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', 

        shadow=True, startangle=180)

plt.axis('equal')

plt.title(' cp-1' ,color = 'blue',fontsize = 12)



sizes = [len(Heart_df[Heart_df['fbs'] == 0]),len(Heart_df[Heart_df['cp'] == 2])]

plt.subplot(the_grid[0, 2], aspect=1, title='cp-2')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', 

        shadow=True, startangle=180)

plt.axis('equal')

plt.title(' cp-2' ,color = 'blue',fontsize = 12)

          

sizes = [len(Heart_df[Heart_df['fbs'] == 0]),len(Heart_df[Heart_df['cp'] == 3])]

plt.subplot(the_grid[0, 3], aspect=1, title='cp-3')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', 

        shadow=True, startangle=180)

plt.axis('equal')

plt.title(' cp-3'  ,color = 'blue' ,fontsize = 12)



plt.show()

pd.crosstab([ Heart_df[Heart_df.sex==0].fbs], [ Heart_df[Heart_df.sex==0].target], normalize='index')

pd.crosstab([ Heart_df[Heart_df.sex==1].fbs], [ Heart_df[Heart_df.sex==1].target], normalize='index')
sns.barplot(x=Heart_df.thalach.value_counts()[:20].index,y=Heart_df.thalach.value_counts()[:20].values)

plt.xlabel('Thalach')

plt.ylabel('Count')

plt.title('Thalach Counts')

plt.xticks(rotation=45)

plt.show()
age_unique=sorted(Heart_df.age.unique())

age_thalach_values=Heart_df.groupby('age')['thalach'].count().values

mean_thalach=[]

for i,age in enumerate(age_unique):

    mean_thalach.append(sum(Heart_df[Heart_df['age']==age].thalach)/age_thalach_values[i])
plt.figure(figsize=(10,5))

sns.pointplot(x=age_unique,y=mean_thalach,color='red',alpha=0.8)

plt.xlabel('Age',fontsize = 15,color='blue')

plt.xticks(rotation=45)

plt.ylabel('Thalach',fontsize = 15,color='blue')

plt.title('Age vs Thalach',fontsize = 15,color='blue')

plt.grid()

plt.show()
sns.barplot(x = 'thal', y = 'target', hue = 'sex', data=Heart_df)

ax.set_title('Thal vs Target based on gender ')
Heart_df.thal.value_counts()
g = sns.FacetGrid(Heart_df, col='thal')

g.map(plt.hist, 'target', bins=10)

g = sns.FacetGrid(Heart_df, col='thal')

g.map(plt.hist, 'sex', bins=10)

g = sns.FacetGrid(Heart_df, col='thal')

g.map(plt.hist, 'cp', bins=10)

g = sns.FacetGrid(Heart_df, col='thal')

g.map(plt.hist, 'chol', bins=10)

g = sns.FacetGrid(Heart_df, col='thal')

g.map(plt.hist, 'thalach', bins=10)
#Target 1

a=len(Heart_df[(Heart_df['target']==1)&(Heart_df['thal']==0)])

b=len(Heart_df[(Heart_df['target']==1)&(Heart_df['thal']==1)])

c=len(Heart_df[(Heart_df['target']==1)&(Heart_df['thal']==2)])

d=len(Heart_df[(Heart_df['target']==1)&(Heart_df['thal']==3)])

print('Target 1 Thal 0: ',a)

print('Target 1 Thal 1: ',b)

print('Target 1 Thal 2: ',c)

print('Target 1 Thal 3: ',d)



#so,Apparently, there is a rate at Thal 2.Now, draw graph

print('*'*50)

#Target 0

e=len(Heart_df[(Heart_df['target']==0)&(Heart_df['thal']==0)])

f=len(Heart_df[(Heart_df['target']==0)&(Heart_df['thal']==1)])

g=len(Heart_df[(Heart_df['target']==0)&(Heart_df['thal']==2)])

h=len(Heart_df[(Heart_df['target']==0)&(Heart_df['thal']==3)])

print('Target 0 Thal 0: ',e)

print('Target 0 Thal 1: ',f)

print('Target 0 Thal 2: ',g)

print('Target 0 Thal 3: ',h)
f,ax=plt.subplots(figsize=(7,7))

sns.barplot(y=['T 1&0 Th 0','T 1&0 Th 1','T 1&0 Th 2','Ta 1&0 Th 3'],x=[1,6,131,28],color='green',alpha=0.5,label='Target 1 Thal State')

sns.barplot(y=['T 1&0 Th 0','T 1&0 Th 1','T 1&0 Th 2','Ta 1&0 Th 3'],x=[1,12,37,89],color='red',alpha=0.7,label='Target 0 Thal State')

ax.legend(loc='lower right',frameon=True)

ax.set(xlabel='Target State and Thal Counter',ylabel='Target State and Thal State',title='Target VS Thal')

plt.xticks(rotation=90)

plt.show()
#fig,ax=plt.plot(figsize=(30,8))

#***Categorical Variables: *** Sex, cp, restecg, fbs, slope, exang, thal,  ca<br>

#***Continuos Variables: ***age, trestbps, chol, thalch,oldpeak<br>

#***Predictor Variable: *** target

sns.boxplot(x='cp',y='age',data=Heart_df,hue='target',palette='hot',linewidth=3)

plt.title("Figure 1")

plt.show()



sns.boxplot(x='cp',y='chol',data=Heart_df,hue='target',palette='hot',linewidth=3)

plt.title("Figure 2")

plt.show()



sns.boxplot(x='slope',y='age',data=Heart_df,hue='target',palette='hot',linewidth=3)

plt.title("Figure 3")

plt.show()



sns.boxplot(x='slope',y='ca',data=Heart_df,hue='target',palette='hot',linewidth=3)

plt.title("Figure 4")

plt.show()



sns.boxplot(x='slope',y='oldpeak',data=Heart_df,hue='target',palette='hot',linewidth=3)

plt.title("Figure 5")

plt.show()
Heart_df= Heart_df.drop(['bin_chol','bin_thalch','bin_trestbps','bin_oldpeak','AgeRange'],axis=1)

Heart_df.columns
#  Lets look at the correlation matrix and plot it using Pandas Style and Matplotlib

Heart_df.corr().round(decimals =2).style.background_gradient(cmap = 'Oranges')
#  Correlation FEMALE - filter dataframe for male/female

dataFemale = Heart_df[(Heart_df['sex'] == 0)]                       # female

dataFemaleCorr = dataFemale.drop(["sex"], axis=1).corr()    # female corr

plt.figure(figsize=(10,10))

plt.title('correlation Heart Disease - FEMALE', fontsize=14)

sns.heatmap(dataFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')





#  Correlation MALE - filter dataframe for male/female

dataMale   = Heart_df[(Heart_df['sex'] == 1)]                       # male

dataMaleCorr = dataMale.drop(["sex"], axis=1).corr()        # male corr

plt.figure(figsize=(10,10))

plt.title('correlation Heart Disease - MALE', fontsize=14)

sns.heatmap(dataMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')
# Corelation with target



x = Heart_df.corr()

pd.DataFrame(x['target']).sort_values(by='target',ascending = False).style.background_gradient(cmap = 'Greens')

print('correlation Heart Disease - MALE')

dataMaleCorr['target'].sort_values(ascending=False)

print('correlation Heart Disease -FEMALE')

dataFemaleCorr['target'].sort_values(ascending=False)
sns.pairplot(Heart_df,hue="target")

plt.show()
# Set X as feature data and y as target data 



X = Heart_df.drop(['target'],axis =1)

y = Heart_df.target

from sklearn import feature_selection

chi2, pval = feature_selection.chi2(X,y)

print(chi2)
Heart_df.columns
dep = pd.DataFrame(chi2)

dep.columns = ['Dependency']

dep.index = X.columns

dep.sort_values('Dependency', ascending = False).style.background_gradient(cmap = 'terrain')
#Creating dummy variables for the categorical variables



dummy_restecg = pd.get_dummies(Heart_df['restecg'], prefix='restecg')

Heart_df = Heart_df.join(dummy_restecg)

dummy_cp = pd.get_dummies(Heart_df['cp'], prefix='cp')

Heart_df = Heart_df.join(dummy_cp)

dummy_slope = pd.get_dummies(Heart_df['slope'], prefix='slope')

Heart_df = Heart_df.join(dummy_slope)

dummy_thal = pd.get_dummies(Heart_df['thal'], prefix='thal')

Heart_df = Heart_df.join(dummy_thal)

Heart_df.head()
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

Heart_df.iloc[:, [0,3,4,7,9]] = sc_X.fit_transform(Heart_df.iloc[:, [0,3,4,7,9]])

Heart_df.head()
Heart_df.columns
#Using stats model for Logistic Regression for all the variables

# Avoid the dummy variable trap by excluding the 0th categorical variable

predictors = [ 'age', 'sex', 'cp_1', 'cp_2', 'cp_3', 'trestbps', 'chol', 'fbs', 'restecg_1', 'restecg_2',

                  'thalach', 'exang', 'oldpeak', 'slope_1', 'slope_2', 'ca', 'thal_1', 'thal_2', 'thal_3']

# Fit the model

m = Logit(Heart_df['target'], Heart_df[predictors])

m = m.fit()
m.summary2()
conf_mat = m.pred_table()



tn, fp, fn, tp = conf_mat.flatten()

print ('True Positive :', tp)

print ('False Negative:', fn)

print ('False Positive:', fp)

print ('True Negative :', tn)



acc = (tp + tn) / np.sum(conf_mat)

print ("Accuracy of the model is: %1.2f" % acc)



prc = tp / (tp + fp)

rec = tp / (tp + fn)

print ("Model's precision is %1.2f and it's recall is %1.2f" % (prc, rec))



mcc = (tp * tn - fp * fn)/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))

print ("Mathews correlation coefficient between the estimates and the true response is: %1.2f" % mcc)
#Adding the probability for each record from the model

Heart_df.loc[:, 'p'] = m.predict(Heart_df[predictors])
##Plotting the model probability output

Heart_df = Heart_df.sort_values(by = ['p'], axis = 0)

plt.plot(logit(Heart_df.p), Heart_df.p, '--')

plt.plot(logit(Heart_df.p), Heart_df.target, '+')

plt.xlabel('logit(p)')

plt.ylabel('p')

plt.show()
# Set discrimination thresholds

ths = np.arange(0., 1.0, 0.025)

#th = 0

# Containers

sensitivity = []

specificity = []

accuracy = []

matthews = []



old_settings = np.seterr(all='ignore')  #seterr to known value

np.seterr(divide='ignore')

# Main loop

for th in ths:

    # Generate estimates

    conf_mat = m.pred_table(threshold=th)

    # Extract TN, FP, ...

    tn, fp, fn, tp = conf_mat.flatten()

    

    # Calculate sensitivity and specificity

    sens = (1. * tp) / (tp + fn)

    spec = (1. * tn) / (tn + fp)    

    

    # Calculate ACC and MCC

    acc = (tp + tn) / np.sum(conf_mat)

    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))    



    # Add to containers

    sensitivity.append(sens)

    specificity.append(spec)

    accuracy.append(acc)

    if np.isnan(mcc) == True:

        mcc = 0

    matthews.append(mcc)
# Bind all the numbers together

roc = pd.DataFrame({'discret_thr' : ths, 

                    'sensitivity' : sensitivity, 

                    'specificity' : specificity,

                    '_specificity' : [1 - x for x in specificity],

                    'accuracy' : accuracy, 

                    'matthews' : matthews})



# Sort by 1 - specificity so we can plot it easily

roc = roc.sort_values(by = "_specificity")

roc.head()


plt.plot(roc._specificity, roc.sensitivity, label = 'ROC')

plt.plot(np.arange(0., 1., 0.01), 

         np.arange(0., 1., 0.01), 

         '--')



plt.legend(loc = 4)
auc = np.trapz(y = roc.sensitivity, x = roc._specificity)

print ("Area under ROC curve = %1.2f" % auc)
predictors_mod = [ 'sex', 'cp', 'trestbps', 'restecg',

                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']



# Fit the model

m_mod = Logit(Heart_df['target'], Heart_df[predictors_mod])

m_mod = m_mod.fit()
m_mod.summary2()
conf_mat_mod = m_mod.pred_table()

tn, fp, fn, tp = conf_mat_mod.flatten()



print ('True Positive :', tp)

print ('False Negative:', fn)

print ('False Positive:', fp)

print ('True Negative :', tn)



acc = (tp + tn) / np.sum(conf_mat_mod)

print ("Accuracy of the model is: %1.2f" % acc)



prc = tp / (tp + fp)

rec = tp / (tp + fn)

print ("Model's precision is %1.2f and it's recall is %1.2f" % (prc, rec))
Heart_df.columns
#Logistic Regression using sklearn

#Splitting data into train and test dataset(70-30) 

X1 = Heart_df.iloc[:, [0,1,3,4,5,7,8,9,11,15,16,18,19,20,22,23,25,26]].values

Y1 = Heart_df.iloc[:, 13].values

X1_Train, X1_Test, Y1_Train, Y1_Test = train_test_split(X1, Y1, test_size = 0.3, random_state = 101)
# Fitting the Logistic Regression into the Training set

#from sklearn.linear_model import LogisticRegression

classifier1 = LogisticRegression(random_state = 0)

classifier1.fit(X1_Train, Y1_Train)

print('Intercept: ',classifier1.intercept_)

print('Coefficients: ', classifier1.coef_)
print("Logistic Train accuracy: ",classifier1.score(X1_Train,Y1_Train))

print("Logistic Test accuracy: ", classifier1.score(X1_Test,Y1_Test))
Y1_Pred = classifier1.predict(X1_Test)

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(Y1_Test, Y1_Pred)

cm1

conf_matrix=pd.DataFrame(data=cm1,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


from sklearn.metrics import classification_report

target_names = ['Heart Disease', 'No Heart Disease']

print(classification_report(Y1_Test, Y1_Pred, target_names=target_names))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(Y1_Test, classifier1.predict(X1_Test))

fpr, tpr, thresholds = roc_curve(Y1_Test, classifier1.predict_proba(X1_Test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
#sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])

from sklearn import metrics

print("Area under Curve = ", metrics.roc_auc_score(Y1_Test, Y1_Pred))
#Naive Bayes Modelling using scales variables only, no dummy variables

#Splitting data into train and test dataset(70-30) 

X = Heart_df.iloc[:, 0:13].values

Y = Heart_df.iloc[:, 13].values

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.3, random_state = 101)
clf = GaussianNB()

# Test options and evaluation metric

scoring = 'accuracy'



#Fitting the training set for the gaussian classifier

clf.fit(X_Train, Y_Train) 



#calling the cross validation function

cv_results = model_selection.cross_val_score(clf, X_Train, Y_Train, scoring=scoring)



#Model Performance

#displaying the mean and standard deviation of the prediction

msg = "%s: %f (%f)" % ('NB model accuracy', cv_results.mean(), cv_results.std())

print(msg)



#Predicting for the Test(Validation) Set

pred_clf = clf.predict(X_Test)



print("NB test Accuracy :", clf.score(X_Test, Y_Test))
cm = metrics.confusion_matrix(Y_Test, pred_clf)

conf_matrix = pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
target_names = ['Heart Disease', 'No Heart Disease']

print(classification_report(Y_Test, pred_clf, target_names=target_names))
roc_auc = roc_auc_score(Y_Test, clf.predict(X_Test))

fpr2, tpr2, thresholds2 = roc_curve(Y_Test, clf.predict_proba(X_Test)[:,1])

plt.figure()

plt.plot(fpr2, tpr2, label='Naive Bayes ROC (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
#sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])

metrics.roc_auc_score(Y_Test, pred_clf)