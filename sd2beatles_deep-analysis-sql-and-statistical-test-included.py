import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from  scipy import stats



data=pd.read_csv(r"../input/facebook-data/pseudo_facebook.csv")

data.head()
#to check if there are any missing values

data.isnull().any() 
data=data.dropna()

data.isnull().any()
categorical=[col for col in data.columns if data[col].dtype=='object']

for i,j in enumerate(categorical):

    print("{0}: {1} has the unique elements:{2}".format(i+1,j,data[j].unique()))
# Summanry of each filed in the data 

data.describe()
labels=['13-19','20-34','35-49','50-65','over 65']

data['age_group']=pd.cut(data.age,bins=[12,19,34,49,65,data.age.max()],labels=labels,include_lowest=True)

data.head()

gender_no=data.groupby("gender")["tenure"].count()

fig,ax=plt.subplots(figsize=(13,7))

gender_no.plot.pie(ax=ax,autopct='%0.2f%%') #autopct cacluate the percentage value automatically

sns.set()

fig,ax=plt.subplots(figsize=(10,8))

test=data.groupby("age_group")['tenure'].count()

test=test/sum(test)*100

color=["royalblue","cornflowerblue","steelblue","lightblue","teal"]

ax.bar(test.index,test.values,color=color,alpha=0.5)

for i in ax.patches:

    ax.text(i.get_x()+0.25,i.get_height()+0.2,str(round(i.get_height(),2))+'%')

ax.set_xlabel("Age")

ax.set_ylabel("percentage")

ax.set_title("Age Distribution of Facebook Users",fontsize=15)
fig,ax=plt.subplots(figsize=(13,7))

color=['r','b']

test=data.pivot_table('tenure',index='age_group',columns='gender',aggfunc='count')

#conversion into percenatage



for col in test.columns:

    test[col]=test[col]/sum(test[col])*100

test.plot(kind='bar',color=color,ax=ax,alpha=0.7)

ax.set_xticklabels(test.index,rotation=360)

ax.set_xlabel("Age group",fontsize=14)

ax.set_ylabel("Percentage",fontsize=14)

ax.set_title('Age and Gender Distribution of Facebook Users',fontsize=14)
#conversion of tenure based on yearly basis

data["tenure_yearly"]=data["tenure"].apply(lambda x:x/365)
labels=['0-1 year','1-2 years','2-3 years','3-4 years','4-5 years','5-6 years','over 6 years']

#Note that the maximum tenure_yearly is 8.6.Thefore we set the upper limit to 9.

data['tenure_group']=pd.cut(data["tenure_yearly"],bins=[0,1,2,3,4,5,6,9],labels=labels,include_lowest=True)

test=data.groupby("tenure_group")["tenure_yearly"].count()

test=test/test.sum()*100



a=pd.Series([''])

b=pd.Series(np.unique(test.index))

c=pd.Series([''])

labels=pd.concat([a,b,c])

fig,axes=plt.subplots(1,2,figsize=(17,13))

test.plot(ax=axes[0],marker='.')

axes[0].set_xticks([-0.5,0,1,2,3,4,5,6,7])

axes[0].set_xticklabels(labels)

axes[0].set_title("Bar Chart:Rentation Rate across 7 intervals",fontsize=15)





#pie_charts

explode=[0.1,0.0,0.0,0.0,0.0,0.4,0.8]

test.plot.pie(ax=axes[1],autopct='%.2f%%',explode=explode)

axes[1].set_ylabel("")

axes[1].set_title("Pie Chart: Retation Rate",fontsize=15)
test=pd.read_csv(r"../input/sql-data/consistency.csv")

test=test.set_index('index_name')

days=np.arange(30,750,30) 

index=[str(i)+' days repeated' for i in days]

test=test.reindex(index)#rearrange the index from 30 days to 720 days

print(test)
# draw a horizontal bar graph

fig,ax=plt.subplots(figsize=(14,10))

test.plot(kind="barh",ax=ax)

ax.set_xlabel("Percentage(%)",fontsize=15)

ax.set_ylabel("Repeated Days",fontsize=15)

male=data.loc[data.gender=='male','tenure']

female=data.loc[data.gender=='female','tenure']

test=[male,female]



fig, ax = plt.subplots(figsize=(11,8))

labels=['female','male']

colors=['pink','lightblue']



bplot=ax.boxplot(test,labels=labels,patch_artist=True)

for patch, color in zip(bplot['boxes'], colors):

    patch.set_facecolor(color)



length=len(data.loc[data.gender=='female','gender'])

male=np.array(data.loc[data.gender=='male','tenure'].iloc[:length])#matching the length of index 

female=np.array(data.loc[data.gender=='female','tenure'])

diff=male-female



import plotly.figure_factory as ff

from scipy import stats





def result_matrix(x):

    result=[['Test_type','Test_static','Significance Level','p-value','Comment']]

    level=0.05

    

    alpha=0.05

    for i in range(2):

        if i==0:

            test='shapiro_test'

            test_static,p_value=stats.shapiro(x)

        else:

            test='Kolmogrove-sminorve test'

            test_static,p_value=stats.kstest(x,'norm')

        alpha=0.05

        if p_value>0.05:

            comment='Meeting normality condition'

        else:

            comment='Not meeting normality condition'

        result.append([test,test_static,alpha,p_value,comment])

    return result





table=ff.create_table(result_matrix(diff))

table.layout.height=200

table.layout.margin.update({'t':30,'b':50})

table.layout.update({'title':'Test For Normality Assumption'})

table.show()

sns.set()

stats.probplot(diff,dist='norm',plot=plt)

plt.title('Normal Q-Q plot')

plt.xlabel("Sample Quantile")

plt.show()

#kurtosis, skewness



def result_matrix(x):

    result=[['Test type','Statics','Significance level','P vlaue','Comment']]

    alpha=0.05

    for i in range(2):

        if i==0:

            test='skewness'

            statics,pvalue=stats.skewtest(x)

        else:

            test='kurtosis'

            statics,pvalue=stats.kurtosistest(x)

        if pvalue>alpha:

            comment='Fail to reject null hypoethesis'

        else:

            comment='Reject null hypothesis'

        result.append([test,statics,alpha,pvalue,comment])

    return result

table=ff.create_table(result_matrix(diff))

table.layout.height=200

table.layout.margin.update({'t':30,'b':50})

table.layout.update({'title':'Test For Kurtosis and Skewness'})

table.show()





#drawing the distribution

fig,ax=plt.subplots(1,2,figsize=(15,7))

sns.distplot(male,ax=ax[0])

ax[0].set_title('Tenure Distribution of Male',fontsize=14)

sns.distplot(female,ax=ax[1],color='coral')

ax[1].set_title('Tenure Distribution of Female',fontsize=14)



result=[['Test','Statics','Significance','P value','Comment']]

statics,pvalue=stats.wilcoxon(male,female)

if pvalue>0.05:

    comment="Fail to reject Null"

else:

    comment='Reject Null'

result.append(['Wilcoxon Signed-Rank test',statics,0.05,pvalue,comment])

table=ff.create_table(result)

table.layout.height=200

table.layout.margin.update({'t':50,'b':30})

table.layout.update({'title':'Wilcoxon Signed-Test'})

table.show()