import numpy as np 

# large, multi-dimensional arrays and matrices, 

# along with a large collection of high-level mathematical functions to operate on these arrays.

import pandas as pd

# data structures and operations for manipulating numerical tables and time series

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

from matplotlib.ticker import PercentFormatter

# plotting

import plotly.express as px

# graph

import plotly.graph_objects as go

# graph

import seaborn as sns

# t-test

from scipy import stats

# regression

from sklearn import datasets, linear_model

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from statsmodels.formula.api import ols

# Word Cloud

from wordcloud import WordCloud
data=pd.read_csv('../input/data-engineer-jobs/DataEngineer.csv')
data.head(2)
data.describe(include='all')
# Check for missing values

def missing_values_table(df):

    # number of missing values

    mis_val = df.isnull().sum()

    # % of missing values

    mis_val_percent = 100 * mis_val / len(df)

    # make table # axis '0' concat along index, '1' column

    mis_val_table = pd.concat([mis_val,mis_val_percent],axis=1) 

    # rename columns

    mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0:'Missing Values',1:'% of Total Values'})

    # sort by column

    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1]!=0].sort_values(

        '% of Total Values',ascending=False).round(1) #Review

    print("Your selected datset has "+str(df.shape[1])+" columns and "+str(len(df))+" observations.\n"

         "There are "+str(mis_val_table_ren_columns.shape[0])+" columns that have missing values.")

    # return the dataframe with missing info

    return mis_val_table_ren_columns



missing_values_table(data)
data['Easy Apply'].value_counts()
data['Competitors'].value_counts()
# Replace -1 or -1.0 or '-1' to NaN

data=data.replace(-1,np.nan)

data=data.replace(-1.0,np.nan)

data=data.replace('-1',np.nan)
missing_values_table(data)
#Remove '\n' from Company Name. 

data['Company Name'],_=data['Company Name'].str.split('\n', 1).str

# 1st column after split, 2nd column after split (delete when '_')

# string.split(separator, maxsplit) maxsplit default -1, which means all occurrances
# Split salary into two columns min salary and max salary.

data['Salary Estimate'],_=data['Salary Estimate'].str.split('(', 1).str
# Split salary into two columns min salary and max salary.

data['Min_Salary'],data['Max_Salary']=data['Salary Estimate'].str.split('-').str

data['Min_Salary']=data['Min_Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype('int')

data['Max_Salary']=data['Max_Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype('int')

# lstrip is for removing leading characters

# rstrip is for removing rear characters
#Drop the original Salary Estimate column

data.drop(['Salary Estimate'],axis=1,inplace=True)
# To estimate the salary with regression and other analysis, better come up with one number: Est_Salary = (Min_Salary+Max_Salary)/2

data['Est_Salary']=(data['Min_Salary']+data['Max_Salary'])/2
# Create a variable for how many years a firm has been founded

data['Years_Founded'] = 2020 - data['Founded']
# A final look at the data before analysis

data.head(2)
plt.figure(figsize=(13,5))

sns.set() #style==background

sns.distplot(data['Min_Salary'], color="b")

sns.distplot(data['Max_Salary'], color="r")



plt.xlabel("Salary ($'000)")

plt.legend({'Min_Salary':data['Min_Salary'],'Max_Salary':data['Max_Salary']})

plt.title("Distribution of Min & Max Salary",fontsize=19)

plt.xlim(0,210)

plt.xticks(np.arange(0, 210, step=10))

plt.tight_layout()

plt.show()
min_max_view = data.sort_values(['Min_Salary','Max_Salary'],ascending=True).reset_index(drop=True).reset_index()
f, (ax_box, ax_line) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios": (0.05,1)},figsize=(13,5))

mean=min_max_view['Est_Salary'].mean()

median=min_max_view['Est_Salary'].median()



bpv = sns.boxplot(y='Est_Salary',data=min_max_view, ax=ax_box).set(ylabel="Est. Salary ($'000)")

ax_box.axhline(mean, color='k', linestyle='--')

ax_box.axhline(median, color='y', linestyle='-')



lp1 = sns.lineplot(x='index',y='Min_Salary',data=min_max_view, color='b')

lp2 = sns.lineplot(x='index',y='Max_Salary',ax=ax_line,data=min_max_view, color='r')

ax_line.axhline(mean, color='k', linestyle='--')

ax_line.axhline(median, color='y', linestyle='-')



plt.legend({'Min_Salary':data['Min_Salary'],'Max_Salary':data['Max_Salary'],'Mean':mean,'Median':median})

plt.title("Salary Estimates of Each Engineer",fontsize=19)

plt.xlabel("Observations")

plt.tight_layout()

plt.show()
sns.set(style='white')



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},figsize=(13,5))

mean=data['Est_Salary'].mean()

median=data['Est_Salary'].median()



bph = sns.boxplot(data['Est_Salary'], ax=ax_box).set(xlabel="")

ax_box.axvline(mean, color='k', linestyle='--')

ax_box.axvline(median, color='y', linestyle='-')



dp = sns.distplot(data['Est_Salary'],ax=ax_hist, color="g").set(xlabel="Est. Salary ($'000)")

ax_hist.axvline(mean, color='k', linestyle='--')

ax_hist.axvline(median, color='y', linestyle='-')



plt.legend({'Mean':mean,'Median':median})

plt.xlim(0,210)

plt.xticks(np.arange(0,210,step=10))

plt.tight_layout() #Adjust the padding between and around subplots

plt.show()
sns.set(style='white')



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},figsize=(13,5))

mean=data['Years_Founded'].mean()

median=data['Years_Founded'].median()



bph = sns.boxplot(data['Years_Founded'], ax=ax_box).set(xlabel="")

ax_box.axvline(mean, color='k', linestyle='--')

ax_box.axvline(median, color='y', linestyle='-')



dp = sns.distplot(data['Years_Founded'],ax=ax_hist, color="g").set(xlabel="Years_Founded")

ax_hist.axvline(mean, color='k', linestyle='--')

ax_hist.axvline(median, color='y', linestyle='-')



plt.legend({'Mean':mean,'Median':median})

plt.xlim(0,240)

plt.xticks(np.arange(0,240,step=10))

plt.tight_layout() #Adjust the padding between and around subplots

plt.show()
sns.set(style='white')



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},figsize=(13,5))

mean=data['Rating'].mean()

median=data['Rating'].median()



bph = sns.boxplot(data['Rating'], ax=ax_box).set(xlabel="")

ax_box.axvline(mean, color='k', linestyle='--')

ax_box.axvline(median, color='y', linestyle='-')



dp = sns.distplot(data['Rating'],ax=ax_hist, color="g").set(xlabel="Ratings")

ax_hist.axvline(mean, color='k', linestyle='--')

ax_hist.axvline(median, color='y', linestyle='-')



plt.legend({'Mean':mean,'Median':median})

plt.xlim(0,6)

plt.xticks(np.arange(0,6,step=1))

plt.tight_layout() #Adjust the padding between and around subplots

plt.show()
# First I count the positions opened by the companies.

df_by_firm=data.groupby('Company Name')['Job Title'].count().reset_index().sort_values(

    'Job Title',ascending=False).head(20).rename(columns={'Job Title':'Hires'})

# When we reset the index, the old index is added as a column, and a new sequential index is used
# Merge with original data to get salary estimates.

Sal_by_firm = df_by_firm.merge(data,on='Company Name',how='left')
sns.set(style="white")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Hires',y='Company Name',data=Sal_by_firm,ax=ax_bar, palette='Set2').set(ylabel="")

sns.pointplot(x='Est_Salary',y='Company Name',data=Sal_by_firm, join=False,ax=ax_point).set(

    ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
df_by_city=data.groupby('Location')['Job Title'].count().reset_index().sort_values(

    'Job Title',ascending=False).head(20).rename(columns={'Job Title':'Hires'})

Sal_by_city = df_by_city.merge(data,on='Location',how='left')
sns.set(style="white")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Hires',y='Location',data=Sal_by_city,ax=ax_bar, palette='Set2').set(ylabel="")

sns.pointplot(x='Est_Salary',y='Location',data=Sal_by_city, join=False,ax=ax_point).set(

    ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
data['City'],data['State'] = data['Location'].str.split(', ',1).str
data['State']=data['State'].replace('Arapahoe, CO','CO')
stateCount = data.groupby('State')[['Job Title']].count().reset_index().rename(columns={'Job Title':'Hires'}).sort_values(

    'Hires', ascending=False).reset_index(drop=True)

stateCount = stateCount.merge(data, on='State',how='left')
sns.set(style="whitegrid")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Hires',y='State',data=stateCount,ax=ax_bar)

sns.pointplot(x='Est_Salary',y='State',data=stateCount, join=False,ax=ax_point).set(ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
data['HQCity'],data['HQState'] = data['Headquarters'].str.split(', ',1).str
data['HQState']=data['HQState'].replace('NY (US), NY','NY')
HQCount = data.groupby('HQState')[['Job Title']].count().reset_index().rename(columns={'Job Title':'Hires'}).sort_values(

    'Hires', ascending=False).head(20).reset_index(drop=True)

HQCount = HQCount.merge(data, on='HQState',how='left')
sns.set(style="whitegrid")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Hires',y='HQState',data=HQCount,ax=ax_bar)

sns.pointplot(x='Est_Salary',y='HQState',data=HQCount, join=False,ax=ax_point).set(ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
RevCount = data.groupby('Revenue')[['Job Title']].count().reset_index().rename(columns={'Job Title':'Hires'}).sort_values(

    'Hires', ascending=False).reset_index(drop=True)
#Make the Revenue column clean

RevCount["Revenue_USD"]=['Unknown','10+ billion','100-500 million','50-100 million','2-5 billion','10-25 million','25-50 million','1-5 million','5-10 billion','<1 million','1-2 billion','0.5-1 billion','5-10 million']

#Merge the new Revenue back to data

RevCount2 = RevCount[['Revenue','Revenue_USD']]

RevCount = RevCount.merge(data, on='Revenue',how='left')
data=data.merge(RevCount2,on='Revenue',how='left')
sns.set(style="whitegrid")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Hires',y='Revenue_USD',data=RevCount,ax=ax_bar)

sns.pointplot(x='Est_Salary',y='Revenue_USD',data=RevCount, join=False,ax=ax_point).set(ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
SizeCount = data.groupby('Size')[['Job Title']].count().reset_index().rename(columns={'Job Title':'Hires'}).sort_values(

    'Hires', ascending=False).reset_index(drop=True)

SizeCount = SizeCount.merge(data, on='Size',how='left')
sns.set(style="whitegrid")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Hires',y='Size',data=SizeCount,ax=ax_bar)

sns.pointplot(x='Est_Salary',y='Size',data=SizeCount, join=False,ax=ax_point).set(ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
SecCount = data.groupby('Sector')[['Job Title']].count().reset_index().rename(columns={'Job Title':'Hires'}).sort_values(

    'Hires', ascending=False).reset_index(drop=True)

SecCount = SecCount.merge(data, on='Sector',how='left')

SecCount = SecCount[SecCount['Hires']>29]
sns.set(style="whitegrid")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Hires',y='Sector',data=SecCount,ax=ax_bar)

sns.pointplot(x='Est_Salary',y='Sector',data=SecCount, join=False,ax=ax_point).set(ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
OwnCount = data.groupby('Type of ownership')[['Job Title']].count().reset_index().rename(columns={'Job Title':'Hires'}).sort_values(

    'Hires', ascending=False).reset_index(drop=True)

OwnCount = OwnCount.merge(data, on='Type of ownership',how='left')
sns.set(style="whitegrid")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Hires',y='Type of ownership',data=OwnCount,ax=ax_bar)

sns.pointplot(x='Est_Salary',y='Type of ownership',data=OwnCount, join=False,ax=ax_point).set(ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
# create a new dataset from original data

text_Analysis = data[['Job Title','Job Description','Est_Salary','Max_Salary','Min_Salary','City','State','Easy Apply','Revenue_USD','Rating','Size','Industry','Sector','Type of ownership','Years_Founded','Company Name','HQState']]

# remove special characters and unify some word use

text_Analysis['Job_title_2']= text_Analysis['Job Title'].str.upper().replace('[^A-Za-z0-9]+', ' ',regex=True)

text_Analysis['Job_title_2']= text_Analysis['Job_title_2'].str.upper().replace(

    ['Â','AND ','WITH ','SYSTEMS','OPERATIONS','ANALYTICS','SERVICES','ENGINEERS','NETWORKS','GAMES','MUSICS','INSIGHTS','SOLUTIONS','JR ','MARKETS','STANDARDS','FINANCE','PRODUCTS','DEVELOPERS','SR '],

    ['','','','SYSTEM','OPERATION','ANALYTIC','SERVICE','ENGINEER','NETWORK','GAME','MUSIC','INSIGHT','SOLUTION','JUNIOR ','MARKET','STANDARD','FINANCIAL','PRODUCT','DEVELOPER','SENIOR '],regex=True)
# unify some word use

text_Analysis['Job_title_2']= text_Analysis['Job_title_2'].str.upper().replace(

    ['BUSINESS INTELLIGENCE','INFORMATION TECHNOLOGY','QUALITY ASSURANCE','USER EXPERIENCE','USER INTERFACE','DATA WAREHOUSE','DATA ANALYST','DATA BASE','DATA QUALITY','DATA GOVERNANCE','BUSINESS ANALYST','DATA MANAGEMENT','REPORTING ANALYST','BUSINESS DATA','SYSTEM ANALYST','DATA REPORTING','QUALITY ANALYST','DATA ENGINEER','BIG DATA','SOFTWARE ENGINEER','MACHINE LEARNING','FULL STACK','DATA SCIENTIST','DATA SCIENCE','DATA CENTER','ENTRY LEVEL','NEURAL NETWORK','SYSTEM ENGINEER'],

    ['BI','IT','QA','UX','UI','DATA_WAREHOUSE','DATA_ANALYST','DATABASE','DATA_QUALITY','DATA_GOVERNANCE','BUSINESS_ANALYST','DATA_MANAGEMENT','REPORTING_ANALYST','BUSINESS_DATA','SYSTEM_ANALYST','DATA_REPORTING','QUALITY_ANALYST','DATA_ENGINEER','BIG_DATA','SOFTWARE_ENGINEER','MACHINE_LEARNING','FULL_STACK','DATA_SCIENTIST','DATA_SCIENCE','DATA_CENTER','ENTRY_LEVEL','NEURAL_NETWORK','SYSTEM_ENGINEER'],regex=True)
# unify some word use

text_Analysis['Job_title_2']= text_Analysis['Job_title_2'].str.upper().replace(

    ['DATA_ENGINEER JUNIOR','DATA_ENGINEER SENIOR','DATA  REPORTING_ANALYST'],

    ['JUNIOR DATA_ENGINEER','SENIOR DATA_ENGINEER','DATA_REPORTING_ANALYST'],regex=True)
jobCount=text_Analysis.groupby('Job_title_2')[['Job Title']].count().reset_index().rename(

    columns={'Job Title':'Count'}).sort_values('Count',ascending=False)

jobSalary = text_Analysis.groupby('Job_title_2')[['Max_Salary','Est_Salary','Min_Salary']].mean().sort_values(

    ['Max_Salary','Est_Salary','Min_Salary'],ascending=False)

jobSalary['Spread']=jobSalary['Max_Salary']-jobSalary['Est_Salary']

jobSalary=jobSalary.merge(jobCount,on='Job_title_2',how='left').sort_values('Count',ascending=False).head(20)
f, axs = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios":(1,0.5)},figsize=(13,8))



ax = axs[0]

ax.errorbar(x='Job_title_2',y='Est_Salary',data=jobSalary,yerr=jobSalary['Spread'],fmt='o')

ax.set_ylabel('Est. Salary ($\'000)')



ax = axs[1]

sns.barplot(x=jobSalary['Job_title_2'],y=jobSalary['Count']).set(xlabel="")



plt.xticks(rotation=65,horizontalalignment='right')

plt.tight_layout()
# get top keywords

s = text_Analysis['Job_title_2'].str.split(expand=True).stack().value_counts().reset_index().rename(

    columns={'index':'KW',0:'Count'})

S = s[s['Count']>29]

S
# write get_keyword method

def get_keyword(x):

   x_ = x.split(" ")

   keywords = []

   try:

      for word in x_:

         if word in np.asarray(S['KW']):

            keywords.append(word)

   except:

      return -1



   return keywords
# get keywords from each row

text_Analysis['KW'] = text_Analysis['Job_title_2'].apply(lambda x: get_keyword(x))
# create dummy columns by keywords

kwdummy = pd.get_dummies(text_Analysis['KW'].apply(pd.Series).stack()).sum(level=0).replace(2,1)

text_Analysis = text_Analysis.merge(kwdummy,left_index=True,right_index=True).replace(np.nan,0)
# run t-test for top keywords to see their correlation with salaries

text_columns = list(text_Analysis.columns)

ttests=[]

for word in text_columns:

    if word in set(S['KW']):

        ttest = stats.ttest_ind(text_Analysis[text_Analysis[word]==1]['Est_Salary'],

                                     text_Analysis[text_Analysis[word]==0]['Est_Salary'])

        ttests.append([word,ttest])

        

ttests = pd.DataFrame(ttests,columns=['KW','R'])

ttests['R']=ttests['R'].astype(str).replace(['Ttest_indResult\(statistic=','pvalue=','\)'],['','',''],regex=True)

ttests['Statistic'],ttests['P-value']=ttests['R'].str.split(', ',1).str

ttests=ttests.drop(['R'],axis=1).sort_values('P-value',ascending=True)

ttests
# Selecting keywords with p-value <0.1 into multiple regression model.

ttest_pass = list(ttests[ttests['P-value'].astype(float)<0.1]['KW'])

print(*ttest_pass,sep=' + ')
TitleBar=ttests[ttests['P-value'].astype(float)<0.05]

TitleBar['Statistic']=(TitleBar['Statistic'].astype(float)/101)*100

TitleBar=TitleBar.sort_values('Statistic',ascending=False).replace(

    'ENGINEER','*OTHER*_ENGINEER').replace('_',' ',regex=True)

TitleBar['KW']='"' + TitleBar['KW'] + '"'
fig = plt.figure(figsize=(13, 5))

sns.barplot(x='KW',y='Statistic',data=TitleBar).set(xlabel="",ylabel="Salary Performance (%) \n Against Average")



plt.xticks(rotation=45,horizontalalignment='right')
# run regression

# Remove variables with p-value >0.05 one by one until all <0.05

titleMod_final = ols("Est_Salary ~ SOFTWARE_ENGINEER + NETWORK + SECURITY + SYSTEM_ENGINEER + SYSTEM + SOFTWARE + MACHINE_LEARNING + ENGINEER + DATA_ENGINEER",

               data=text_Analysis).fit()

print(titleMod_final.summary())
# Plot with scatterplots

fig = plt.figure(figsize=(13, 13))

fig = sm.graphics.plot_partregress_grid(titleMod_final,fig=fig)

fig.tight_layout(pad=1.0)

# Sorry somebody tell me how to remove that "Partial Regression Plot"
text_Analysis['Job_Desc2'] = text_Analysis['Job Description'].replace('[^A-Za-z0-9]+', ' ',regex=True)
text_Analysis['Job_Desc2'] = text_Analysis['Job_Desc2'].str.upper().replace(

    ['COMPUTER SCIENCE','ENGINEERING DEGREE',' MS ','BUSINESS ANALYTICS','SCRUM MASTER','MACHINE LEARNING',' ML ','POWER BI','ARTIFICIAL INTELLIGENCE',' AI ','ALGORITHMS','DEEP LEARNING','NEURAL NETWORK','NATURAL LANGUAGE PROCESSING','DECISION TREE','CLUSTERING','PL SQL'],

    ['COMPUTER_SCIENCE','ENGINEERING_DEGREE',' MASTER ','BUSINESS_ANALYTICS','SCRUM_MASTER','MACHINE_LEARNING',' MACHINE_LEARNING ','POWER_BI','ARTIFICIAL_INTELLIGENCE',' ARTIFICIAL_INTELLIGENCE ','ALGORITHM','DEEP_LEARNING','NEURAL_NETWORK','NATURAL_LANGUAGE_PROCESSING','DECISION_TREE','CLUSTER','PLSQL'],regex=True)
# Create a list of big data buzzwords to see if those words in JD would influence the salary

buzzwords = ['COMPUTER_SCIENCE','MASTER','MBA','SQL','PYTHON','R','PHD','BUSINESS_ANALYTICS','SAS','PMP','SCRUM_MASTER','STATISTICS','MATHEMATICS','MACHINE_LEARNING','ARTIFICIAL_INTELLIGENCE','ECONOMICS','TABEAU','AWS','AZURE','POWER_BI','ALGORITHM','DEEP_LEARNING','NEURAL_NETWORK','NATURAL_LANGUAGE_PROCESSING','DECISION_TREE','REGRESSION','CLUSTER','ORACLE','EXCEL','TENSORFLOW','HADOOP','SPARK','NOSQL','SAP','ETL','API','PLSQL','MONGODB','POSTGRESQL','ELASTICSEARCH','REDIS','MYSQL','FIREBASE','SQLITE','CASSANDRA','DYNAMODB','OLTP','OLAP','DEVOPS','PLATFORM','NETWORK','APACHE','SECURITY']
# Count the JD keywords.

S2 = text_Analysis['Job_Desc2'].str.split(expand=True).stack().value_counts().reset_index().rename(

    columns={'index':'KW',0:'Count'})

S2 = S2[S2['KW'].isin(buzzwords)].reset_index(drop=True)

# .sort_values('Count',ascending=False)

S2_TOP = S2[S2['Count']>29]

S2_TOP_JD = S2_TOP

S2_TOP_JD['KW'] = S2_TOP_JD['KW'] +'_JD'

S2_TOP_JD
wordCloud = WordCloud(width=450,height= 300).generate(' '.join(S2['KW']))

plt.figure(figsize=(19,9))

plt.axis('off')

plt.title("Keywords in Data Engineer Job Descriptions",fontsize=20)

plt.imshow(wordCloud)

plt.show()
# write get_keyword method

def get_keyword(x):

   x_ = x.split(" ")

   keywords = []

   try:

      for word in x_:

         if word + '_JD' in np.asarray(S2_TOP_JD['KW']):

            keywords.append(word + '_JD')

   except:

      return -1



   return keywords
# get keywords from each row

text_Analysis['JDKW'] = text_Analysis['Job_Desc2'].apply(lambda x: get_keyword(x))
# create dummy columns by keywords

kwdummy = pd.get_dummies(text_Analysis['JDKW'].apply(pd.Series).stack()).sum(level=0)

# Since a JD sometimes repeat a keyword, the value may >1

# But what we want to know is whether the appearance of the keyword impact the salary, not frequency

# So values >1 have to be replaced by 1, but there must be a better way than coding like this ↓

kwdummy = kwdummy.replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,35,39],

                         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
# merge back the dummy columns to the main dataset

text_Analysis = text_Analysis.merge(kwdummy,left_index=True,right_index=True,how='left').replace(np.nan,0)
# let's see if number of buzzwords contained or how wordy the JD is would have impact.

text_Analysis['JDKWlen']=text_Analysis['JDKW'].str.len()

text_Analysis['JDlen']=text_Analysis['Job Description'].str.len()
# run t-test for top keywords to see their correlation with salaries

text_columns = list(text_Analysis.columns)

ttests_JD=[]

for word in text_columns:

    if word in set(S2_TOP_JD['KW']):

        ttest2 = stats.ttest_ind(text_Analysis[text_Analysis[word]>0]['Est_Salary'],

                                 text_Analysis[text_Analysis[word]==0]['Est_Salary'])

        ttests_JD.append([word,ttest2])



ttests_JD = pd.DataFrame(ttests_JD,columns=['KW','R'])

ttests_JD['R']=ttests_JD['R'].astype(str).replace(['Ttest_indResult\(statistic=','pvalue=','\)'],['','',''],regex=True)

ttests_JD['Statistic'],ttests_JD['P-value']=ttests_JD['R'].str.split(', ',1).str

ttests_JD=ttests_JD.drop(['R'],axis=1).sort_values('P-value',ascending=True)

ttests_JD
JDBar=ttests_JD[(ttests_JD['P-value'].astype(float)<0.05)&(ttests_JD['Statistic'].astype(float)>0)]

JDBar['Statistic']=(JDBar['Statistic'].astype(float)/101)*100

JDBar=JDBar.sort_values('Statistic',ascending=False).replace('_JD','',regex=True).replace('_',' ',regex=True)

JDBar['KW']='"' + JDBar['KW'] + '"'
fig = plt.figure(figsize=(13, 5))

sns.barplot(x='KW',y='Statistic',data=JDBar).set(xlabel="",ylabel="Salary Markup %")



plt.xticks(rotation=45,horizontalalignment='right')
#Selecting keywords with p-value <0.1 into multiple regression model.

ttest_JD_pass1 = list(ttests_JD[ttests_JD['P-value'].astype(float)<0.05]['KW'])

print(*ttest_JD_pass1,sep=' + ')
#Run regression and remove variables with p-value >0.05 one by one until all <0.05

JDMod = ols("Est_Salary ~ CLUSTER_JD + COMPUTER_SCIENCE_JD + ALGORITHM_JD + PLATFORM_JD + PYTHON_JD + CASSANDRA_JD",

               data=text_Analysis).fit()

print(JDMod.summary())
fig = plt.figure(figsize=(13, 13))

fig = sm.graphics.plot_partregress_grid(JDMod,fig=fig)

fig.tight_layout(pad=1.0)
# create dummy columns by State

kwdummy = pd.get_dummies(text_Analysis['State'].apply(pd.Series).stack()).sum(level=0)

text_Analysis = text_Analysis.merge(kwdummy,left_index=True,right_index=True,how='left').replace(np.nan,0)
S3 = text_Analysis['State'].value_counts().reset_index().rename(

    columns={'index':'State','State':'Count'})

S3_Top = S3[S3['Count']>29]

S3_Top
#run t-test for top states hiring engineers to see their correlation with salaries

text_columns = list(text_Analysis.columns)

ttests_state=[]

for word in text_columns:

    if word in set(S3_Top['State']):

        ttest3 = stats.ttest_ind(text_Analysis[text_Analysis[word]>0]['Est_Salary'],

                                 text_Analysis[text_Analysis[word]==0]['Est_Salary'])

        ttests_state.append([word,ttest3])



ttests_state = pd.DataFrame(ttests_state,columns=['State','R'])

ttests_state['R']=ttests_state['R'].astype(str).replace(['Ttest_indResult\(statistic=','pvalue=','\)'],['','',''],regex=True)

ttests_state['Statistic'],ttests_state['P-value']=ttests_state['R'].str.split(', ',1).str

ttests_state=ttests_state.drop(['R'],axis=1).sort_values('P-value',ascending=True)

ttests_state
#Selecting states with p-value <0.1 into multiple regression model.

ttest_state_pass = list(ttests_state[ttests_state['P-value'].astype(float)<0.1]['State'])

print(*ttest_state_pass,sep=' + ')
StateMod = ols("Est_Salary ~ FL + CA + TX",

               data=text_Analysis).fit()

print(StateMod.summary())
fig = plt.figure(figsize=(13, 13))

fig = sm.graphics.plot_partregress_grid(StateMod,fig=fig)

fig.tight_layout(pad=1.0)
text_Analysis['City']=text_Analysis['City'].str.replace(' ','_',regex=True)
S35 = text_Analysis['City'].value_counts().reset_index().rename(

    columns={'index':'City','City':'Count'})

S35_Top = S35[S35['Count']>29]
# create dummy columns by City

kwdummy = pd.get_dummies(text_Analysis[text_Analysis['City'].isin(np.asarray(S35_Top['City']))]['City'].apply(pd.Series).stack()).sum(level=0)

text_Analysis = text_Analysis.merge(kwdummy,left_index=True,right_index=True,how='left').replace(np.nan,0)
#run t-test for top cities hring data engineers to see their correlation with salaries

text_columns = list(text_Analysis.columns)

ttests_city=[]

for word in text_columns:

    if word in set(S35_Top['City']):

        ttest35 = stats.ttest_ind(text_Analysis[text_Analysis[word]>0]['Est_Salary'],

                                 text_Analysis[text_Analysis[word]==0]['Est_Salary'])

        ttests_city.append([word,ttest35])



ttests_city = pd.DataFrame(ttests_city,columns=['City','R'])

ttests_city['R']=ttests_city['R'].astype(str).replace(['Ttest_indResult\(statistic=','pvalue=','\)'],['','',''],regex=True)

ttests_city['Statistic'],ttests_city['P-value']=ttests_city['R'].str.split(', ',1).str

ttests_city=ttests_city.drop(['R'],axis=1).sort_values('P-value',ascending=True)

ttests_city
#Selecting cities with p-value <0.1 into multiple regression model.

ttest_city_pass = list(ttests_city[ttests_city['P-value'].astype(float)<0.1]['City'])

print(*ttest_city_pass,sep=' + ')
CityMod = ols("Est_Salary ~ Irving + Jacksonville + Houston + Fort_Worth + San_Jose + San_Diego + Los_Angeles + San_Antonio + Sunnyvale",

               data=text_Analysis).fit()

print(CityMod.summary())
S31 = text_Analysis['HQState'].value_counts().reset_index().rename(

    columns={'index':'HQState','HQState':'Count'}).replace(0,'Unknown_State')

S31_Top = S31[S31['Count']>29]

S31_Top['HQState_HQ'] = [s + '_HQ' for s in S31_Top['HQState']]
# create dummy columns by HQ State

kwdummy = pd.get_dummies(S31_Top['HQState_HQ'].apply(pd.Series).stack()).sum(level=0)

S31_Top2 = S31_Top.merge(kwdummy,left_index=True,right_index=True,how='left').drop(['Count'],axis=1)

text_Analysis = text_Analysis.merge(S31_Top2,on='HQState',how='left').replace(np.nan,0)
text_columns = list(text_Analysis.columns)

ttests_HQstate=[]

for word in text_columns:

    if word in set(S31_Top['HQState_HQ']):

        ttest31 = stats.ttest_ind(text_Analysis[text_Analysis[word]>0]['Est_Salary'],

                                 text_Analysis[text_Analysis[word]==0]['Est_Salary'])

        ttests_HQstate.append([word,ttest31])



ttests_HQstate = pd.DataFrame(ttests_HQstate,columns=['HQState_HQ','R'])

ttests_HQstate['R']=ttests_HQstate['R'].astype(str).replace(['Ttest_indResult\(statistic=','pvalue=','\)'],['','',''],regex=True)

ttests_HQstate['Statistic'],ttests_HQstate['P-value']=ttests_HQstate['R'].str.split(', ',1).str

ttests_HQstate=ttests_HQstate.drop(['R'],axis=1).sort_values('P-value',ascending=True)

ttests_HQstate
ttest_HQstate_pass = list(ttests_HQstate[ttests_HQstate['P-value'].astype(float)<0.1]['HQState_HQ'])

print(*ttest_HQstate_pass,sep=' + ')
HQStateMod = ols("Est_Salary ~ FL_HQ + TX_HQ + CA_HQ",

               data=text_Analysis).fit()

print(HQStateMod.summary())
#Remove special characters.

text_Analysis['Revenue_USD'] = text_Analysis['Revenue_USD'].replace('[^A-Za-z0-9]+', '_',regex=True).replace(['_1_million','Unknown','5_10_billion'],['Small_Business','RevUnknown','Large_Corp'])

text_Analysis['Size'] = text_Analysis['Size'].replace('[^A-Za-z0-9]+', '_',regex=True).replace(['51_to_200_employees','10000_employees'],['SMB','Giant']).replace('Unknown','SizeUnknown')

text_Analysis['Sector'] = text_Analysis['Sector'].replace('[^A-Za-z0-9]+', '_',regex=True).replace('Unknown','SectorUnknown').replace(['Government','Unknown'],['GovSec','SectorUnknown'])

text_Analysis['Industry'] = text_Analysis['Industry'].replace('[^A-Za-z0-9]+', '_',regex=True).replace('Unknown','IndUnknown')

text_Analysis['Type of ownership'] = text_Analysis['Type of ownership'].replace('[^A-Za-z0-9]+', '_',regex=True).replace('Unknown','OwnUnknown')
#Rename column name for running regression later.

text_Analysis = text_Analysis.rename(columns={"Easy Apply":"Easy_Apply"})
# create dummy columns by Revenue

kwdummy = pd.get_dummies(text_Analysis['Revenue_USD'].apply(pd.Series).stack()).sum(level=0)

text_Analysis = text_Analysis.merge(kwdummy,left_index=True,right_index=True,how='left').replace(np.nan,0)
S4 = text_Analysis['Revenue_USD'].value_counts().reset_index().rename(

    columns={'index':'Revenue_USD','Revenue_USD':'Count'})

S4_Top = S4[S4['Count']>29]

S4_Top
#run t-test to see the salary differences by companies' revenue.

text_columns = list(text_Analysis.columns)

ttests_rev=[]

for word in text_columns:

    if word in set(S4_Top['Revenue_USD']):

        ttest4 = stats.ttest_ind(text_Analysis[text_Analysis[word]>0]['Est_Salary'],

                                 text_Analysis[text_Analysis[word]==0]['Est_Salary'])

        ttests_rev.append([word,ttest4])



ttests_rev = pd.DataFrame(ttests_rev,columns=['Revenue_USD','R'])

ttests_rev['R']=ttests_rev['R'].astype(str).replace(['Ttest_indResult\(statistic=','pvalue=','\)'],['','',''],regex=True)

ttests_rev['Statistic'],ttests_rev['P-value']=ttests_rev['R'].str.split(', ',1).str

ttests_rev=ttests_rev.drop(['R'],axis=1).sort_values('P-value',ascending=True)

ttests_rev
#Selecting revenues with p-value <0.1 into multiple regression model.

ttest_rev_pass = list(ttests_rev[ttests_rev['P-value'].astype(float)<0.1]['Revenue_USD'])

print(*ttest_rev_pass,sep=' + ')
kwdummy = pd.get_dummies(text_Analysis['Size'].apply(pd.Series).stack()).sum(level=0)

text_Analysis = text_Analysis.merge(kwdummy,left_index=True,right_index=True,how='left').replace(np.nan,0)
S5 = text_Analysis['Size'].value_counts().reset_index().rename(

    columns={'index':'Size','Size':'Count'})

S5_Top = S5[S5['Count']>29]

S5_Top
text_columns = list(text_Analysis.columns)

ttests_size=[]

for word in text_columns:

    if word in set(S5_Top['Size']):

        ttest5 = stats.ttest_ind(text_Analysis[text_Analysis[word]>0]['Est_Salary'],

                                 text_Analysis[text_Analysis[word]==0]['Est_Salary'])

        ttests_size.append([word,ttest5])



ttests_size = pd.DataFrame(ttests_size,columns=['Size','R'])

ttests_size['R']=ttests_size['R'].astype(str).replace(['Ttest_indResult\(statistic=','pvalue=','\)'],['','',''],regex=True)

ttests_size['Statistic'],ttests_size['P-value']=ttests_size['R'].str.split(', ',1).str

ttests_size=ttests_size.drop(['R'],axis=1).sort_values('P-value',ascending=True)

ttests_size
ttest_size_pass = list(ttests_size[ttests_size['P-value'].astype(float)<0.1]['Size'])

print(*ttest_size_pass,sep=' + ')
kwdummy = pd.get_dummies(text_Analysis['Sector'].apply(pd.Series).stack()).sum(level=0)

text_Analysis = text_Analysis.merge(kwdummy,left_index=True,right_index=True,how='left').replace(np.nan,0)
S6 = text_Analysis['Sector'].value_counts().reset_index().rename(

    columns={'index':'Sector','Sector':'Count'})

S6_Top = S6[S6['Count']>29]

S6_Top
text_columns = list(text_Analysis.columns)

ttests_sec=[]

for word in text_columns:

    if word in set(S6_Top['Sector']):

        ttest6 = stats.ttest_ind(text_Analysis[text_Analysis[word]>0]['Est_Salary'],

                                 text_Analysis[text_Analysis[word]==0]['Est_Salary'])

        ttests_sec.append([word,ttest6])



ttests_sec = pd.DataFrame(ttests_sec,columns=['Sector','R'])

ttests_sec['R']=ttests_sec['R'].astype(str).replace(['Ttest_indResult\(statistic=','pvalue=','\)'],['','',''],regex=True)

ttests_sec['Statistic'],ttests_sec['P-value']=ttests_sec['R'].str.split(', ',1).str

ttests_sec=ttests_sec.drop(['R'],axis=1).sort_values('P-value',ascending=True)

ttests_sec
ttest_sec_pass = list(ttests_sec[ttests_sec['P-value'].astype(float)<0.1]['Sector'])

print(*ttest_sec_pass,sep=' + ')
kwdummy = pd.get_dummies(text_Analysis['Type of ownership'].apply(pd.Series).stack()).sum(level=0)

text_Analysis = text_Analysis.merge(kwdummy,left_index=True,right_index=True,how='left').replace(np.nan,0)
S8 = text_Analysis['Type of ownership'].value_counts().reset_index().rename(

    columns={'index':'Type_of_ownership','Type of ownership':'Count'})

S8_Top = S8[S8['Count']>29]

S8_Top
text_columns = list(text_Analysis.columns)

ttests_own=[]

for word in text_columns:

    if word in set(S8_Top['Type_of_ownership']):

        ttest8 = stats.ttest_ind(text_Analysis[text_Analysis[word]>0]['Est_Salary'],

                                 text_Analysis[text_Analysis[word]==0]['Est_Salary'])

        ttests_own.append([word,ttest8])



ttests_own = pd.DataFrame(ttests_own,columns=['Type_of_ownership','R'])

ttests_own['R']=ttests_own['R'].astype(str).replace(['Ttest_indResult\(statistic=','pvalue=','\)'],['','',''],regex=True)

ttests_own['Statistic'],ttests_own['P-value']=ttests_own['R'].str.split(', ',1).str

ttests_own=ttests_own.drop(['R'],axis=1).sort_values('P-value',ascending=True)

ttests_own
ttest_own_pass = list(ttests_own[ttests_own['P-value'].astype(float)<0.1]['Type_of_ownership'])

print(*ttest_own_pass,sep=' + ')
ModC = ols("Est_Salary ~ FL + CA + TX + CASSANDRA_JD + ENGINEER + Irving + Houston + Fort_Worth + San_Diego + San_Antonio",

               data=text_Analysis).fit()

# Rating, Years_Founded, Easy_Apply, PHD/Master, Sector, Size, Type_of_ownership not significant

print(ModC.summary())
# Trying different interaction terms.

text_Analysis['CA_CA_HQ']=text_Analysis['CA']*text_Analysis['CA_HQ']

text_Analysis['PYTHON_CASSANDRA']=text_Analysis['PYTHON_JD']*text_Analysis['CASSANDRA_JD']

text_Analysis['CA_PYTHON']=text_Analysis['CA']*text_Analysis['PYTHON_JD']

text_Analysis['CA_CASSANDRA']=text_Analysis['CA']*text_Analysis['CASSANDRA_JD']

text_Analysis['ENGINEER_FL']=text_Analysis['ENGINEER']*text_Analysis['FL']

text_Analysis['ENGINEER_CA']=text_Analysis['ENGINEER']*text_Analysis['CA']

text_Analysis['ENGINEER_TX']=text_Analysis['ENGINEER']*text_Analysis['TX']

text_Analysis['ENGINEER_CA_HQ']=text_Analysis['ENGINEER']*text_Analysis['CA_HQ']

text_Analysis['ENGINEER_PYTHON']=text_Analysis['ENGINEER']*text_Analysis['PYTHON_JD']

text_Analysis['ENGINEER_CASSANDRA']=text_Analysis['ENGINEER']*text_Analysis['CASSANDRA_JD']

text_Analysis['ENGINEER_Irving']=text_Analysis['ENGINEER']*text_Analysis['Irving']

text_Analysis['ENGINEER_Houston']=text_Analysis['ENGINEER']*text_Analysis['Houston']

text_Analysis['ENGINEER_Fort_Worth']=text_Analysis['ENGINEER']*text_Analysis['Fort_Worth']

text_Analysis['ENGINEER_San_Antonio']=text_Analysis['ENGINEER']*text_Analysis['San_Antonio']
# Final model considering interaction terms.

ModC = ols("Est_Salary ~ FL + CA + TX + CASSANDRA_JD + ENGINEER + Irving + Houston + Fort_Worth + San_Diego + San_Antonio",

               data=text_Analysis).fit()

# Rating, Years_Founded, Easy_Apply, PHD, Sector, Size, Type_of_ownership not significant

print(ModC.summary())
fig = plt.figure(figsize=(13, 26))

fig = sm.graphics.plot_partregress_grid(ModC,fig=fig)

fig.tight_layout(pad=1.0)
# create a separate dataset for CA

data_CA = data[data['State']=='CA']
pd.set_option('display.max_columns', None)

data_CA.describe(include='all')
sns.set(style='white')



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},figsize=(13,5))

mean=data['Est_Salary'].mean()

median=data['Est_Salary'].median()



bph = sns.boxplot(data['Est_Salary'], ax=ax_box).set(xlabel="")

ax_box.axvline(mean, color='k', linestyle='--')

ax_box.axvline(median, color='y', linestyle='-')



dp1 = sns.distplot(data_CA['Est_Salary'],ax=ax_hist, color="r").set(xlabel="Est. Salary ($'000)")

dp2 = sns.distplot(data['Est_Salary'],ax=ax_hist, color="g").set(xlabel="Est. Salary ($'000)")

ax_hist.axvline(mean, color='k', linestyle='--')

ax_hist.axvline(median, color='y', linestyle='-')



plt.legend({'Mean (All)':mean,'Median (All)':median,'California':data_CA['Est_Salary'],'All':data['Est_Salary']})

plt.xlim(0,210)

plt.xticks(np.arange(0,210,step=10))

plt.tight_layout() #Adjust the padding between and around subplots

plt.show()
# Create a table for heatmap of number of companies with different sizes and revenues

Firm_Size = data.pivot_table(columns="Size",index="Revenue_USD",values="Company Name",aggfunc=pd.Series.nunique).reset_index()

Firm_Size = Firm_Size[['Revenue_USD','1 to 50 employees','51 to 200 employees','201 to 500 employees','501 to 1000 employees','1001 to 5000 employees','5001 to 10000 employees','10000+ employees']]

Firm_Size = Firm_Size.reindex([11,2,9,4,7,10,5,0,1,6,8,3,12])

Firm_Size = Firm_Size.set_index('Revenue_USD').replace(np.nan,0)



# Create a table for heatmap of number of companies with different sizes and revenues in CA

Firm_Size_CA = data_CA.pivot_table(columns="Size",index="Revenue_USD",values="Company Name",aggfunc=pd.Series.nunique).reset_index()

Firm_Size_CA = Firm_Size_CA[['Revenue_USD','1 to 50 employees','51 to 200 employees','201 to 500 employees','501 to 1000 employees','1001 to 5000 employees','5001 to 10000 employees','10000+ employees']]

Firm_Size_CA = Firm_Size_CA.reindex([11,2,9,4,7,10,5,0,1,6,8,3,12])

Firm_Size_CA = Firm_Size_CA.set_index('Revenue_USD').replace(np.nan,0)



# Create table for heatmap of salaries by companies with different sizes and revenues

Firm_Size_Sal = data.pivot_table(columns="Size",index="Revenue_USD",values="Est_Salary",aggfunc=np.mean).reset_index()

Firm_Size_Sal = Firm_Size_Sal[['Revenue_USD','1 to 50 employees','51 to 200 employees','201 to 500 employees','501 to 1000 employees','1001 to 5000 employees','5001 to 10000 employees','10000+ employees']]

Firm_Size_Sal = Firm_Size_Sal.reindex([11,2,9,4,7,10,5,0,1,6,8,3,12])

Firm_Size_Sal = Firm_Size_Sal.set_index('Revenue_USD').replace(np.nan,0)



# Create table for heatmap of salaries by companies with different sizes and revenues in CA

Firm_Size_CA_Sal = data_CA.pivot_table(columns="Size",index="Revenue_USD",values="Est_Salary",aggfunc=np.mean).reset_index()

Firm_Size_CA_Sal = Firm_Size_CA_Sal[['Revenue_USD','1 to 50 employees','51 to 200 employees','201 to 500 employees','501 to 1000 employees','1001 to 5000 employees','5001 to 10000 employees','10000+ employees']]

Firm_Size_CA_Sal = Firm_Size_CA_Sal.reindex([11,2,9,4,7,10,5,0,1,6,8,3,12])

Firm_Size_CA_Sal = Firm_Size_CA_Sal.set_index('Revenue_USD').replace(np.nan,0)
f, axs = plt.subplots(nrows=2,ncols=2, sharey=True,sharex=True, figsize=(13,9))



fs = sns.heatmap(Firm_Size,annot=True,fmt='.0f',annot_kws={"size": 12},cmap="YlGnBu", ax=axs[0,0]).set(title="Number of Companies in the US",xlabel="")

fsc = sns.heatmap(Firm_Size_CA,annot=True,fmt='.0f',annot_kws={"size": 12},cmap="YlGnBu", ax=axs[0,1]).set(title="Number of Companies in CA",xlabel="",ylabel="")

fss = sns.heatmap(Firm_Size_Sal,annot=True,fmt='.0f',annot_kws={"size": 12},cmap="Oranges",ax=axs[1,0]).set(title="Avg. Salaries in the US")

fscs = sns.heatmap(Firm_Size_CA_Sal,annot=True,fmt='.0f',annot_kws={"size": 12},cmap="Oranges",ax=axs[1,1]).set(title="Avg. Salaries in CA",ylabel="")



plt.setp([a.get_xticklabels() for a in axs[1,:]],rotation=45,ha='right')

plt.tight_layout()

plt.show()
ca_sal_by_firm = data_CA.groupby('Company Name')[['Est_Salary']].mean().reset_index()
SmallHighPay = data_CA[((data_CA['Revenue_USD']=='5-10 million')|(data_CA['Revenue_USD']=='10-25 million')|(data_CA['Revenue_USD']=='25-50 million'))&(

    data_CA['Size']=='51 to 200 employees')]['Company Name'].value_counts().reset_index().rename(

    columns={'index':'Company Name','Company Name':'Hires'})
SmallHighPay = SmallHighPay.merge(ca_sal_by_firm, on='Company Name',how='left')

SmallHighPay = SmallHighPay.merge(data_CA[['Company Name','Rating','Headquarters','Type of ownership','Industry','Sector','Years_Founded','Competitors']], on='Company Name',how='left')

SmallHighPay = SmallHighPay.drop_duplicates().reset_index(drop=True)

SmallHighPay
SmallHighPay.describe(include='all')
MLHighPay = data_CA[((data_CA['Revenue_USD']=='50-100 million')|(

    data_CA['Revenue_USD']=='100-500 million')|(

    data_CA['Revenue_USD']=='0.5-1 billion'))&(

    data_CA['Size']=='1001 to 5000 employees')]['Company Name'].value_counts().reset_index().rename(

    columns={'index':'Company Name','Company Name':'Hires'})
MLHighPay = MLHighPay.merge(ca_sal_by_firm, on='Company Name',how='left')

MLHighPay = MLHighPay.merge(data_CA[['Company Name','Rating','Headquarters','Type of ownership','Industry','Sector','Years_Founded','Competitors']], on='Company Name',how='left')

MLHighPay = MLHighPay.drop_duplicates().reset_index(drop=True)

MLHighPay
MLHighPay.describe(include='all')
RevCountCA = data_CA.groupby('Revenue_USD')[['Job Title']].count().reset_index().rename(columns={'Job Title':'Count'}).sort_values(

    'Count', ascending=False).reset_index(drop=True)

RevCountCA = RevCountCA.merge(data_CA, on='Revenue_USD',how='left')
sns.set(style="whitegrid")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Count',y='Revenue_USD',data=RevCountCA,ax=ax_bar)

sns.pointplot(x='Est_Salary',y='Revenue_USD',data=RevCountCA, join=False,ax=ax_point).set(ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
SizeCountCA = data_CA.groupby('Size')[['Job Title']].count().reset_index().rename(columns={'Job Title':'Count'}).sort_values(

    'Count', ascending=False).reset_index(drop=True)

SizeCountCA = SizeCountCA.merge(data_CA, on='Size',how='left')
sns.set(style="whitegrid")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Count',y='Size',data=SizeCountCA,ax=ax_bar)

sns.pointplot(x='Est_Salary',y='Size',data=SizeCountCA, join=False,ax=ax_point).set(ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
SecCountCA = data_CA.groupby('Sector')[['Job Title']].count().reset_index().rename(columns={'Job Title':'Count'}).sort_values(

    'Count', ascending=False).head(12).reset_index(drop=True)

SecCountCA = SecCountCA.merge(data_CA, on='Sector',how='left')
sns.set(style="whitegrid")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Count',y='Sector',data=SecCountCA,ax=ax_bar)

sns.pointplot(x='Est_Salary',y='Sector', join=False,data=SecCountCA,ax=ax_point).set(ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()
OwnCountCA = data_CA.groupby('Type of ownership')[['Job Title']].count().reset_index().rename(columns={'Job Title':'Count'}).sort_values(

    'Count', ascending=False).reset_index(drop=True)

OwnCountCA = OwnCountCA.merge(data_CA, on='Type of ownership',how='left')
sns.set(style="whitegrid")

f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))

sns.barplot(x='Count',y='Type of ownership',data=OwnCountCA,ax=ax_bar)

sns.pointplot(x='Est_Salary',y='Type of ownership',data=OwnCountCA, join=False,ax=ax_point).set(ylabel="",xlabel="Salary ($'000)")



plt.tight_layout()