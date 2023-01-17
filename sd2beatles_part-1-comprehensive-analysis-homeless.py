import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import plotly.figure_factory as ff

import copy

import category_encoders as ce

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import seaborn as sns

from scipy.stats import chi2_contingency

from scipy.stats import chi2

data=pd.read_csv('../input/homeless-dataset/homeless_prep.csv')
data.head()

data=data.drop('CLIENT_KEY',axis=1)

columns={'CLIENT_KEY':'client_key','AGE':'age','GENDER':'gender','VETERAN':'veteran','INCOME':'income','NIGHTS':'nights'}

data=data.rename(columns=columns)
def statistical_measure(data):

    test=[['Feature','Type','NA availability','Missing_Ratio','Skewness','Kurtosis']]

    for i in data.columns:

        column_name=i

        if data[i].dtype=='O':

            column_type='Object'

        else:

            column_type=data[i].dtype

        if data[i].isnull().sum()==0:

            missing_data='None'

            missing_ratio=0

        else:

            missing_data='Yes'

            missing_ratio=round(data[i].isnull().sum()/data[i].shape[0],2)

        if data[i].dtype=='O':

            skew='Not Applicable'

            kurtosis='Not Applicable'

        else:

            skew=round(data[i].skew(),2)

            kurtosis=round(data[i].kurt(),2)

        test.append([column_name,column_type,missing_data,missing_ratio,skew,kurtosis])

    return test

        
ff.create_table(statistical_measure(data))
data.describe()

features=['substanceabuse','completed','probation','required']

for i in features:

    data[i]=data[i].map(lambda x: 'No'if x==0 else 'Yes')
content=[['Features','Elements']]

#store the categorical data and its unique memebers

for i,j in zip(data.columns,data.dtypes):

    if j=='object':

        content.append([i,list(data.loc[:,i].unique())])  

#present the result in the tablet format

ff.create_table(content).show()
#Converting Categorical data into integer

ml_data=data.copy()

for i,j in enumerate(ml_data.dtypes):

    if j=='object' or 'category':

        ce_ord=ce.OrdinalEncoder()

        ml_data.iloc[:,i]=ce_ord.fit_transform(ml_data.iloc[:,i])

sns.set()

fig,ax=plt.subplots(figsize=(14,5))

sns.scatterplot(ml_data.income,ml_data.nights)
ml_data=ml_data[(ml_data.income<2000)&(ml_data.nights<300)]

ml_data.shape[0]

print('Number of Observations before filtering:{0}'.format(data.shape[0]))

print('Number of Observations After filtering: {0}'.format(ml_data.shape[0]))
income_veteran=ml_data.income*ml_data.veteran

sns.set()

fig,ax=plt.subplots(figsize=(13,5))

sns.scatterplot(income_veteran,ml_data.nights,ax=ax,color='r')
ml_data=ml_data[income_veteran<2000]

print('Number of Observations before filtering:{0}'.format(data.shape[0]))

print('Number of Observations After filtering: {0}'.format(ml_data.shape[0]))
ml_data.isnull().any()
#Selecting only 'applicable' rows of original data

data=data.iloc[ml_data.index,:]

print('Number of Observation after filtering out ouliers: {0} rows'.format(data.shape[0]))
sns.set()

pie=data.groupby('gender')['age'].count()

fig,axes=plt.subplots(1,2,figsize=(12,5))

def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:.1f}%\n({:d} people)".format(pct, absolute)

axes[0].pie(pie, autopct=lambda x: func(x, pie),textprops=dict(color="w"),colors=['R','B'],explode=[0.1,0])

axes[0].set_title('Pie Chart For Homeless By Gender',fontsize=15)

axes[1].bar(x=pie.index,height=pie.values,color='coral',alpha=0.7)

axes[1].set_title('Bar Chart For Homeless By Gender',fontsize=15)
data1=data.pivot_table('required',index='veteran',columns='gender',aggfunc='count',margins=True)

data2=data.pivot_table('required',index=['veteran','probation'],columns='gender',aggfunc='count',fill_value=0)

display(data1)

display(data2)
sns.set_style('white')

labels=[['No Supervision','Supervision'],['Never Joining Armed Fprces','Having Served Armed Forces']]

labels=list(pd.MultiIndex.from_product(labels))

fig,ax=plt.subplots(figsize=(15,7))

data2.plot(kind='bar',color=['R','B','Green'],ax=ax,alpha=0.8)

ax.set_xticks([-0.2,0.6,1.6,2.6])

ax.set_xticklabels(labels=labels,rotation=36,fontsize=14)

ax.set_xlabel('Assistance , Veteran',fontsize=15)

ax.set_ylabel('Number of Peple',fontsize=15)

ax.set_title('Numer of Homeless According to Veteran and Supervision over an offender Required',fontsize=20)
print('Table 1 :Living Status of Homeless By Gender')

display(data.pivot_table('nights',index='gender',columns='completed',aggfunc='count',fill_value=0,margins=True))



print('Table 2: Average of Nights Acccording to Ceasing to Stay at Shelters and Gender ')

display(round(data.pivot_table('nights',index='gender',columns='completed',aggfunc='mean',fill_value=0),2))



fig,ax=plt.subplots(figsize=(10,4))

sns.boxplot(x=data.completed,y=data.nights,hue=data.gender,palette='YlOrBr')

ax.set_title('Boxtplot for Table 2',fontsize=16)

ax.set_xlabel('Complted',fontsize=14)

ax.set_ylabel('Number of Nights',fontsize=14)
data.pivot_table('nights',index='substanceabuse',columns='probation',aggfunc='count',margins=True)
result=data.pivot_table('nights',index='substanceabuse',columns='probation',aggfunc='count')



# interpret test-statistic

def chi_square(data,prob):

    table=[['Test_Type','Critical Value','Alpha','p_value','Result']]

    stats,p,dof,expected=chi2_contingency(data)

    critical=chi2.ppf(prob,dof)

    alpha = 1.0 - prob

    if p <= alpha:

        result='Dependent (reject H0)'

    else:

        result='Independent (fail to reject H0)'

    outcome=['Chi_Square',critical,round(alpha,2),round(p,2),result]

    table.append(outcome)

    return table



ff.create_table(chi_square(result,0.95))
data.age.describe()
labels=['20-29','30-39','40-49','50-59','over 60']

data['age_group']=pd.cut(data.age,bins=[19,29,39,49,59,data.age.max()],labels=labels)
data.pivot_table('nights',index=['probation','completed'],columns='age_group',aggfunc='count',fill_value=0)