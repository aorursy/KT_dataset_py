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
%precision %.2f
import plotly.graph_objs as go
df=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.sample(4)
df.info()
df.describe(include='all').fillna(' ')
print('Attrition count')

print(df.Attrition.value_counts())

print('Attrition in percentage')

print(df.Attrition.value_counts()*100/len(df))
import plotly.plotly as py

import cufflinks as cf

cf.set_config_file(offline=True, world_readable=True, theme='ggplot')

df.Attrition.value_counts().iplot(kind='bar',title='Bar graph of attrition count in both category')
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

style.use('ggplot')

import warnings

warnings.filterwarnings('ignore')
plt.figure(figsize=(20,10))

plt.subplot(221)

sns.distplot(df.Age[df.Gender=='Male'],label='Male avg age=36.65',norm_hist=True,color='c',hist=False)

plt.ylabel('Density')

plt.axvline(df.Age[df.Gender=='Male'].mean(),linestyle='dashed',color='c',linewidth=1)

sns.distplot(df.Age[df.Gender=='Female'],label='Female avg age=37.32',norm_hist=True,color='k',hist=False)

plt.axvline(df.Age[df.Gender=='Female'].mean(),linestyle='dashed',color='k',linewidth=2)

plt.subplot(222)

sns.distplot(df.Age,norm_hist=True,label='Age',hist=False,color='b')

plt.axvline(df.Age.mean())
plt.figure(figsize=(20,10))

plt.subplot(221)

sns.distplot(df.Age[df.Attrition=='Yes'],label='attrition',norm_hist=True,color='c')

#plt.ylabel('Density')

plt.axvline(df.Age[df.Attrition=='Yes'].mean(),linestyle='dashed',color='c',linewidth=1)

plt.subplot(222)

sns.distplot(df.Age[df.Attrition=='No'],label='Age dist of non attrition',norm_hist=True,color='k')

plt.axvline(df.Age[df.Gender=='Female'].mean(),linestyle='dashed',color='k',linewidth=2)

plt.subplot(223)

sns.distplot(df.Age,norm_hist=True,label='Age',hist=False,color='b')

plt.axvline(df.Age.mean())
print('Employee dept wise')

print(df.Department.value_counts())

print('....Attrition departmet wise....')

print(df.Department[df.Attrition=='Yes'].value_counts())

sns.countplot(df.Department[df.Attrition=='Yes'])
print('Attriion rate for the R & D depat =',100*133/961)

print('Attriion rate for the Sales depat =',100*92/446)

print('Attriion rate for the HR depat =',100*12/63)
#Lets check thes summary of daily rate and monthly rate in attrition categaory

print('--------Daily and Monthly rate of the Attrition category--------')

print(df[df.Attrition=='Yes'].describe()[['DailyRate','MonthlyRate']])

print('--------Daily and Monthly rate of the Non-Attrition category--------')

print(df[df.Attrition=='No'].describe()[['DailyRate','MonthlyRate']])

#Lets check the distribution

plt.figure(figsize=(20,10))

plt.subplot(121)

sns.distplot(df.DailyRate[df.Attrition=='Yes'],label='Attrition',norm_hist=True,color='c',hist=False)

sns.distplot(df.DailyRate[df.Attrition=='No'],label='No Attrition',norm_hist=True,color='b',hist=False)

plt.title('Distribution of Daily rate')

plt.subplot(122)

sns.distplot(df.MonthlyRate[df.Attrition=='Yes'],label='Attrition',norm_hist=True,color='c',hist=False)

sns.distplot(df.MonthlyRate[df.Attrition=='No'],label='No Attrition',norm_hist=True,color='b',hist=False)

plt.title('Distribution of monthly rate')
#Now try to analyze the business travel impact on the attrition rate

sns.countplot(x='BusinessTravel',hue='Attrition',data=df)
print('Percentage of attriton in Travel_Rarely',

      100*len(df[(df.BusinessTravel=='Travel_Rarely') & (df.Attrition=='Yes')])

      /len(df[df.BusinessTravel=='Travel_Rarely']))



print('Percentage of attriton in Travel_Frequently',

      100*len(df[(df.BusinessTravel=='Travel_Frequently') & (df.Attrition=='Yes')])

      /len(df[df.BusinessTravel=='Travel_Frequently']))



print('Percentage of attriton in Non_Travel',

      100*len(df[(df.BusinessTravel=='Non-Travel') & (df.Attrition=='Yes')])

      /len(df[df.BusinessTravel=='Non-Travel']))
#Now plot the distribtion of distance from home



print('--------Sumary of Distance from home in the Attrition category--------')

print(df[df.Attrition=='Yes'].describe()[['DistanceFromHome']])

print('--------Daily and Monthly rate of the Non-Attrition category--------')

print(df[df.Attrition=='No'].describe()[['DistanceFromHome']])

#Lets check the distribution

plt.figure(figsize=(7,7))

sns.distplot(df.DistanceFromHome[df.Attrition=='Yes'],label='Attrition',norm_hist=True,color='c',hist=False)

sns.distplot(df.DistanceFromHome[df.Attrition=='No'],label='No Attrition',norm_hist=True,color='b',hist=False)

dict(zip([1,2,3,4,5],['Below_College','College','Bachelor','Master','Doctor']))
def education(value):

    edu=dict(zip([1,2,3,4,5],['Below_College','College','Bachelor','Master','Doctor']))

    if value in edu:

        #print(value,edu[value])

        return edu[value]
#list(map(lambda x:education(x),list(df.Education)))

#df.Education.apply(lambda x:education(x))

df['Education_labels']=list(map(lambda x:education(x),list(df.Education)))
pd.concat([df.Education.apply(lambda x:education(x)),df.EducationField,df.Attrition],axis=1,ignore_index=True).head(2)
#Education 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'

#Let uss find out the percentage of attrition in each education



for i in set(list(df.Education_labels)):

    print('Percentage Attrition in {0} degree = {1} %'.format(i,100*len(df[(df.Education_labels==i) & (df.Attrition=='Yes')])/

                                                           len(df[(df.Education_labels==i)])))



#See the distribution of attrition across the education field and education_labels(Master,bachelors

#degree etc)

g=sns.catplot(x='EducationField',hue='Attrition',col='Education_labels',data=df,

            kind='count',height=4,aspect=0.9,col_wrap=3,sharex =True)

g.set_xticklabels(rotation=45)
for i in set(list(df.EducationField)):

    print('Percentage Attrition in {0} field = {1} %'.format(i,100*len(df[(df.EducationField==i) & (df.Attrition=='Yes')])/

                                                           len(df[(df.EducationField==i)])))

def EnvironmentSatisfaction(value):

    env=dict(zip([1,2,3,4],['Low','Medium','High','Very_High']))

    if value in env:

        return env[value]
df['EnvSatisfaction_labels']=list(map(lambda x:EnvironmentSatisfaction(x),list(df.EnvironmentSatisfaction)))
for i in set(list(df.EnvSatisfaction_labels)):

    print('Percentage Attrition in {0} env satisfaction = {1} %'.format(i,100*len(df[(df.EnvSatisfaction_labels==i) & (df.Attrition=='Yes')])/

                                                           len(df[(df.EnvSatisfaction_labels==i)])))

plt.figure(figsize=(10,5))

plt.subplot(121)

sns.countplot(x='EnvSatisfaction_labels', hue='Attrition',data=df)

#plt.subplot(122)

sns.catplot(x='EnvSatisfaction_labels', hue='Gender',data=df,col='Attrition',kind='count')
plt.figure(figsize=(10,5))

sns.distplot(df.HourlyRate[df.Attrition=='Yes'],label='Attrion',color='c',norm_hist=True,hist=False)

sns.distplot(df.HourlyRate[df.Attrition=='No'],label='No Attrition',color='black',hist=False)
#Job role wise 



jr_A=dict(df.JobRole[df.Attrition=='Yes'].value_counts())

plt.figure(figsize=(20,10))

plt.subplot(121)

plt.pie(jr_A.values(),labels=jr_A.keys(),autopct='%1.1f')

plt.title('Job role under attrition')



plt.subplot(122)

ms=dict(df.MaritalStatus[df.Attrition=='Yes'].value_counts())

plt.pie(ms.values(),labels=ms.keys(),autopct='%1.1f')

plt.title('MaritalStatus under attrition')

print('Average monthly income of the people who left the company in dollar= ',df.MonthlyIncome[df.Attrition=='Yes'].mean())

print('Median income of the people who left the company in dollar= ',df.MonthlyIncome[df.Attrition=='Yes'].median())

print('Average monthly income of the people who dont left the company in dollar= ',df.MonthlyIncome[df.Attrition=='No'].mean())

print('Median income of the people who don''t left the company in dollar= ',df.MonthlyIncome[df.Attrition=='No'].median())

print('Average Income of the people who left the company is {0} percentage below then the who dont left'.

      format(100*(df.MonthlyIncome[df.Attrition=='No'].mean()-df.MonthlyIncome[df.Attrition=='Yes'].mean())/df.MonthlyIncome[df.Attrition=='Yes'].mean()))

#g=sns.FacetGrid(col='Attrition',data=df,height=6,aspect=0.9)

#g.map(plt.hist,'MonthlyIncome')

plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(df.NumCompaniesWorked)

plt.title('Employee count with respect to no. of companies worked')



plt.subplot(132)

sns.countplot(df.NumCompaniesWorked[df.Attrition=='Yes'])

plt.title('Attrition')



plt.subplot(133)

sns.countplot(df.NumCompaniesWorked[df.Attrition=='No'])

plt.title('Non-Attrition')
plt.figure(figsize=(13,5))

plt.subplot(131)

sns.countplot(df.OverTime)

plt.title('Overall')

plt.subplot(132)

sns.countplot(df.OverTime[df.Attrition=='Yes'])

plt.title('Attrition')

plt.subplot(133)

sns.countplot(df.OverTime[df.Attrition=='No'])

plt.title('Non-Attriton')


plt.figure(figsize=(20,5))

plt.subplot(121)

sns.violinplot(x='PercentSalaryHike',y='Attrition',data=df,hue='Gender',scale='count',inner='quartile',split=True)

plt.subplot(122)

sns.violinplot(x='PercentSalaryHike',y='Attrition',data=df,hue='PerformanceRating',inner='quartile',split='True')
#g=sns.FacetGrid(col='Department',data=df,height=6,aspect=0.9)

#g.map(plt.hist,'TotalWorkingYears',normed=1)

plt.figure(figsize=(15,7))

sns.violinplot(x='TotalWorkingYears',y='Attrition',hue='Department',data=df,inner='quartile',palette='Set2')
#t=dict(df.groupby(['Attrition','TrainingTimesLastYear'])['TrainingTimesLastYear'].count())

#t.key(('No',0))

t_a=dict(df.TrainingTimesLastYear[df.Attrition=='Yes'].value_counts())

t_a.keys()
plt.figure(figsize=(15,7))

plt.subplot(121)

t_a=dict(df.TrainingTimesLastYear[df.Attrition=='Yes'].value_counts())

plt.pie(t_a.values(),labels=t_a.keys(),autopct='%1.1f%%',colors = ['silver', 'yellowgreen', 'lightcoral', 'lightskyblue'])

plt.title('Attrition')

plt.xlabel('TrainingTimesLastYear')

plt.subplot(122)

t_a=dict(df.TrainingTimesLastYear[df.Attrition=='No'].value_counts())

#sns.countplot(df.TrainingTimesLastYear[df.Attrition=='No'],normed=1)

plt.pie(t_a.values(),labels=t_a.keys(),autopct='%1.1f%%')

plt.title('No Attrition')

plt.xlabel('TrainingTimesLastYear')
dict_WorkLifeBalance={ 1:'Bad', 2:'Good', 3:'Better' ,4:'Best'}

dict1=dict(df.WorkLifeBalance[df.Attrition=='Yes'].value_counts().sort_index())

dict3=dict((dict_WorkLifeBalance[key],val*100/len(df.Attrition[df.Attrition=='Yes'])) for key,val in dict1.items())

plt.figure(figsize=(15,4))

plt.subplot(121)

sns.barplot(x=list(dict3.keys()),y=list(dict3.values()))

plt.xlabel('WorkLifeBalance')

plt.ylabel('% of people who attrited')

plt.subplot(122)

dict1=dict(df.WorkLifeBalance[df.Attrition=='No'].value_counts().sort_index())

dict3=dict((dict_WorkLifeBalance[key],val*100/len(df.Attrition[df.Attrition=='No'])) for key,val in dict1.items())

sns.barplot(x=list(dict3.keys()),y=list(dict3.values()))

plt.xlabel('WorkLifeBalance')

plt.ylabel('% of people who do not attrited')
#Drop the addition variables created in exploratory analysis

df.drop(['Education_labels','EnvSatisfaction_labels'],axis=1,inplace=True)

#Lets Drop the not useful predictors

df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1,inplace=True)

print('no of unique values in dept  ',np.unique(df.Department))

print('no of unique values in EducationField  ',np.unique(df.EducationField))

print('no of unique values in Gender  ',np.unique(df.Gender))

print('no of unique values in JobRole  ',np.unique(df.JobRole))

print('no of unique values in MaritalStatus  ',np.unique(df.MaritalStatus))

print('no of unique values in OverTime  ',np.unique(df.OverTime))

#Lets use the one hot encoding to transform the categorical fatures

from sklearn.preprocessing import OneHotEncoder

oneh_enc=OneHotEncoder()

oneh_features=oneh_enc.fit_transform(df[['BusinessTravel','EducationField','Gender','JobRole','MaritalStatus','OverTime']])
oneh_features=pd.DataFrame(oneh_features.toarray(),columns=oneh_enc.get_feature_names())

oneh_features.sample(3)
oneh_features.drop(['x5_No','x2_Female','x5_No'],axis=1,inplace=True)
l_ind=[]

#Lets perform chi square test of independence

from sklearn.feature_selection import chi2

chisq,pval=chi2(oneh_features,df[['Attrition']])

for i in pval:

    if i<0.05:

        l_ind.append('Y')

    else:

        l_ind.append('N')

data={'chisq':chisq,'pval':pval,'ind':l_ind}

data=pd.DataFrame(data=data,index=[i for i in list(oneh_enc.get_feature_names()) if i not in ['x5_No','x2_Female','x5_No']])
data.sort_values(by=['chisq','ind'],ascending=False)
#Create a df having only significance levels

oneh_features=oneh_features[list(data[data.ind=='Y'].index)]
#Observed frequency table

Observed=pd.crosstab(df.Attrition,df.BusinessTravel,margins=True)

Observed.index=['No', 'Yes', 'row_total']

Observed.columns=['Non-Travel', 'Travel_Frequently', 'Travel_Rarely', 'col_total']

Observed
#Calculate the expected frquency table

Expected=pd.DataFrame(np.outer(Observed.iloc[0:2:,3:4],Observed.iloc[2:3:,0:3])/1470)

Expected.index=['No', 'Yes']

Expected.columns=['Non-Travel', 'Travel_Frequently', 'Travel_Rarely']
Expected
#pd.options.display.float_format = '{:,.2f}'.format



from scipy.stats import chi2_contingency

Observed=pd.crosstab(df.Attrition,df.BusinessTravel)

chi2_contingency(observed=Observed)# Displays chi2, p, dof, ex
#create an empty dataframe to hold the chi2 and p value of the categorical predictors

#As you can see, Gender is independent from the attrition hece we will drop it now

l_chisq=[]

l_pval=[]

independent_status=[]

for col in ['BusinessTravel','EducationField','Gender','JobRole','MaritalStatus','OverTime']:

    

    Observed=pd.crosstab(df.Attrition,df[col])

    chi2, p, dof, ex=chi2_contingency(observed=Observed)

    chi2=round(chi2,4)

    p=round(p,3)

    l_chisq.append(chi2)

    l_pval.append(p)

    if p<0.05:

        independent_status.append('Y')

    else:

        independent_status.append('N')

chisq_dict={'chisq':l_chisq,'pval':l_pval,'indicator':independent_status}

pd.DataFrame(data=chisq_dict,index=['BusinessTravel','EducationField','Gender','JobRole','MaritalStatus','OverTime'])
#Now lets use another feature selection methods for categorical data

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold

#Lets use variance threshold method



from sklearn.feature_selection import VarianceThreshold

selector=VarianceThreshold(threshold=0.2)

selector.fit_transform(oneh_features)

selector.get_support(indices=True)
#Using mutual information to select the categorical features

from sklearn.feature_selection import mutual_info_classif

mi=mutual_info_classif(oneh_features,df[['Attrition']])

data={'Features':list(oneh_features.columns),'MI val':list(mi)}

print('Chi squae columns',oneh_features.columns)

pd.DataFrame(data=data).sort_values(by='MI val',ascending=False)

#['NumCompaniesWorked','StockOptionLevel',

#                             'TrainingTimesLastYear','TotalWorkingYears','YearsAtCompany',

#                             'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']

sns.distplot(df.Age)
#from statsmodels.graphics.gofplots import qqplot

#qqplot(df.Age)

#df['TotalWorkingYears'].quantile([0, .25, .5, .75, 1.])
#Let us use the Anderson Darlington test

#H0: the sample has a Gaussian distribution.

#H1: the sample does not have a Gaussian distribution.

#Tests whether a data sample has a Gaussian distribution.

#Assumptions

#Observations in each sample are independent and identically distributed (iid).



from scipy.stats import anderson



for col in ['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','PercentSalaryHike']:

    result=anderson(df[col])

    print('Statistics for the {0} variable'.format(col),result)

from scipy.stats import shapiro

for col in ['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','PercentSalaryHike']:

    result=shapiro(df[col])

    print('Statistics for the {0} variable statistics= {1} and P-val= {2}'.format(col,result[0],result[1]))
from scipy.stats import kstest

np.random.seed(987654321)

for col in ['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','PercentSalaryHike']:

    result=kstest(np.array(df[col]),cdf='norm')

    print('Statistics for the {0} variable statistics= {1} and P-val= {2}'.format(col,result[0],result[1]))
from scipy.stats import mannwhitneyu

#Under the null hypothesis H0, the distributions of both populations are equal.[3]

#The alternative hypothesis H1 is that the distributions are not equal.

#mannwhitneyu(np.array(pd.get_dummies(df[['Attrition']],drop_first=True)).ravel(),np.array(df[['Age']]).ravel())

for col in ['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','PercentSalaryHike']:

    

    stat,p=mannwhitneyu(df[col][df.Attrition=='Yes'],df[col][df.Attrition=='No'],alternative='less')

    print('----------------------------{0}----------------------------'.format(col))

    print('Statistics=%.3f,p=%.4f'%(stat,p))

    alpha=0.05

    if p>alpha:

        print('Same dist of feature {0} on both the categories, fail to reject null hypothesis'.format(col))

    else:

        print('Accept the alternate hypothesis i.e. values in attrition population of {0} is less than non attrition population of {1}'.format(col,col))
from scipy.stats import kruskal

for col in ['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','PercentSalaryHike']:

    

    stat,p=kruskal(df[col][df.Attrition=='Yes'],df[col][df.Attrition=='No'],alternative='less')

    print('----------------------------{0}----------------------------'.format(col))

    print('Statistics=%.3f,p=%.4f'%(stat,p))

    alpha=0.05

    if p>alpha:

        print('Same dist of feature {0} on both the categories, fail to reject null hypothesis'.format(col))

    else:

        print('Accept the alternate hypothesis i.e. values in attrition population of {0} is differ than non attrition population of {1}'.format(col,col))