# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')

df.head()

df=df.drop('ID',axis=1)

df.describe()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,12))

ax1=fig.add_subplot(521)

ax2=fig.add_subplot(522)

ax3=fig.add_subplot(523)

ax4=fig.add_subplot(524)

ax5=fig.add_subplot(525)

df.Age.hist(bins=50,ax=ax1)

df.Experience.hist(bins=50,ax=ax2)

df.Income.hist(bins=50,ax=ax3)

df.CCAvg.hist(bins=50,ax=ax4)

df.Mortgage.hist(bins=20,ax=ax5)
df[df['Experience']<0].shape

# experience cannot be negative, make it to zero
df1=df[df['Experience']>=0]

m=df1['Experience'].median()

for i in range (0,len(df.Experience)):

    if df['Experience'][i]<0:

        df['Experience'][i]=m

    if df['ZIP Code'][i]<10000:

        df.drop(df.index[i],inplace=True)

df['Experience'].describe()
df.info()
df.isnull().sum()

# no missing values
df.apply(lambda x: len(x.unique()))
import seaborn as sns

sns.countplot(x='Personal Loan',hue='Family',data=df)

table=pd.crosstab(df['Family'],df['Personal Loan'])

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Family vs Purchase')

plt.xlabel('Family')

plt.ylabel('Proportion of Loans')

# family size of more than 3 are more likely to get the loan
sns.countplot(x='Personal Loan',hue='Education',data=df)

table=pd.crosstab(df['Education'],df['Personal Loan'])

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Education vs Purchase')

plt.xlabel('Education')

plt.ylabel('Proportion of Loans')

# undergraduate has very less prob of taking the loan
sns.countplot(x='Personal Loan',hue='Securities Account',data=df)

table=pd.crosstab(df['Securities Account'],df['Personal Loan'])

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Securities Account vs Purchase')

plt.xlabel('Securities Account')

plt.ylabel('Proportion of Loans')

# having securities account wont prefer loan
sns.countplot(x='Personal Loan',hue='CD Account',data=df)

table=pd.crosstab(df['CD Account'],df['Personal Loan'])

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of CD Account vs Purchase')

plt.xlabel('CD Account')

plt.ylabel('Proportion of Loans')

# customers having COD account have high prob of taking loan
sns.distplot( df[df['Personal Loan'] == 0]['CCAvg'], color = 'r')

sns.distplot( df[df['Personal Loan'] == 1]['CCAvg'], color = 'g')

# higher CCAvg spending customer is more likely to get the personal loan
sns.distplot( df[df['Personal Loan'] == 0]['Age'], color = 'r')

sns.distplot( df[df['Personal Loan'] == 1]['Age'], color = 'g')

# no variation
sns.distplot( df[df['Personal Loan'] == 0]['Experience'], color = 'r')

sns.distplot( df[df['Personal Loan'] == 1]['Experience'], color = 'g')

# no variation
sns.distplot( df[df['Personal Loan'] == 0]['Income'], color = 'r')

sns.distplot( df[df['Personal Loan'] == 1]['Income'], color = 'g')

# Income with more than 50 is more likely to get the personal loan
sns.distplot( df[df['Personal Loan'] == 0]['Mortgage'], color = 'r')

sns.distplot( df[df['Personal Loan'] == 1]['Mortgage'], color = 'g')

# higher mortgage is more likely to get the loan
sns.countplot(x='Personal Loan',hue='Online',data=df)

table=pd.crosstab(df.Online,df['Personal Loan'])

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Education vs Purchase')

plt.xlabel('Online')

plt.ylabel('Proportion of Loans')
sns.countplot(x='Personal Loan',hue='CreditCard',data=df)

table=pd.crosstab(df.CreditCard,df['Personal Loan'])

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of CreditCard vs Purchase')

plt.xlabel('CreditCard')

plt.ylabel('Proportion of Loans')
sns.set(style='ticks',color_codes=True)

sns.catplot(x='Personal Loan',y='Income',data=df, height=7, aspect=2)

# Income below 50 are not getting a loan
sns.set(style='ticks',color_codes=True)

sns.catplot(x='Personal Loan',y='Mortgage',data=df, height=7, aspect=2)
sns.set(style='ticks',color_codes=True)

sns.catplot(x='Personal Loan',y='Age',data=df, height=7, aspect=2)
sns.set(style='ticks',color_codes=True)

sns.catplot(x='Personal Loan',y='Experience',data=df, height=7, aspect=2)
sns.set(style='ticks',color_codes=True)

sns.catplot(x='Personal Loan',y='CCAvg',data=df, height=7, aspect=2)
sns.set(style='ticks',color_codes=True)

sns.catplot(x='Personal Loan',y='ZIP Code',data=df, height=7, aspect=2)

# there should not be any dependency of ZIP code
#sns.pairplot(df.iloc[:,:])
sns.boxplot(x='Family',y='Income',hue='Personal Loan',data=df)

# Family with income less than 100k are less likely to take loan
sns.boxplot(x='Education',y='Mortgage',hue='Personal Loan',data=df)

# Customers having high mortgages need personal loan
sns.boxplot(x='Education',y='CCAvg',hue='Personal Loan',data=df)

# Customers having high CCAvg need personal loan
sns.set(style='ticks',color_codes=True)

sns.catplot(x='CD Account',y='Income',data=df, height=7, aspect=2)
sns.set(style='ticks',color_codes=True)

sns.catplot(x='Education',y='Income',data=df, height=7, aspect=2)
sns.set(style='ticks',color_codes=True)

sns.catplot(x='Family',y='Income',data=df, height=7, aspect=2)
corr = df.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)

# Experience will be removed

# CCAvg may or may not be removed 
df=df.drop('Experience', axis=1)
import pandas.core.algorithms as algos

from pandas import Series

import scipy.stats.stats as stats

import re

import traceback

import string



max_bin = 20

force_bin = 3



# define a binning function

def mono_bin(Y, X, n = max_bin):

    

    df1 = pd.DataFrame({"X": X, "Y": Y})

    justmiss = df1[['X','Y']][df1.X.isnull()]

    notmiss = df1[['X','Y']][df1.X.notnull()]

    r = 0

    while np.abs(r) < 1:

        try:

            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})

            d2 = d1.groupby('Bucket', as_index=True)

            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)

            n = n - 1 

        except Exception as e:

            n = n - 1



    if len(d2) == 1:

        n = force_bin         

        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))

        if len(np.unique(bins)) == 2:

            bins = np.insert(bins, 0, 1)

            bins[1] = bins[1]-(bins[1]/2)

        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 

        d2 = d1.groupby('Bucket', as_index=True)

    

    d3 = pd.DataFrame({},index=[])

    d3["MIN_VALUE"] = d2.min().X

    d3["MAX_VALUE"] = d2.max().X

    d3["COUNT"] = d2.count().Y

    d3["EVENT"] = d2.sum().Y

    d3["NONEVENT"] = d2.count().Y - d2.sum().Y

    d3=d3.reset_index(drop=True)

    

    if len(justmiss.index) > 0:

        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])

        d4["MAX_VALUE"] = np.nan

        d4["COUNT"] = justmiss.count().Y

        d4["EVENT"] = justmiss.sum().Y

        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y

        d3 = d3.append(d4,ignore_index=True)

    

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT

    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT

    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT

    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT

    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["VAR_NAME"] = "VAR"

    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       

    d3 = d3.replace([np.inf, -np.inf], 0)

    d3.IV = d3.IV.sum()

    

    return(d3)



def char_bin(Y, X):

        

    df1 = pd.DataFrame({"X": X, "Y": Y})

    justmiss = df1[['X','Y']][df1.X.isnull()]

    notmiss = df1[['X','Y']][df1.X.notnull()]    

    df2 = notmiss.groupby('X',as_index=True)

    

    d3 = pd.DataFrame({},index=[])

    d3["COUNT"] = df2.count().Y

    d3["MIN_VALUE"] = df2.sum().Y.index

    d3["MAX_VALUE"] = d3["MIN_VALUE"]

    d3["EVENT"] = df2.sum().Y

    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    

    if len(justmiss.index) > 0:

        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])

        d4["MAX_VALUE"] = np.nan

        d4["COUNT"] = justmiss.count().Y

        d4["EVENT"] = justmiss.sum().Y

        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y

        d3 = d3.append(d4,ignore_index=True)

    

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT

    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT

    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT

    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT

    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["VAR_NAME"] = "VAR"

    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      

    d3 = d3.replace([np.inf, -np.inf], 0)

    d3.IV = d3.IV.sum()

    d3 = d3.reset_index(drop=True)

    

    return(d3)



def data_vars(df1, target):

    

    stack = traceback.extract_stack()

    filename, lineno, function_name, code = stack[-2]

    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]

    final = (re.findall(r"[\w']+", vars_name))[-1]

    

    x = df1.dtypes.index

    count = -1

    

    for i in x:

        if i.upper() not in (final.upper()):

            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:

                conv = mono_bin(target, df1[i])

                conv["VAR_NAME"] = i

                count = count + 1

            else:

                conv = char_bin(target, df1[i])

                conv["VAR_NAME"] = i            

                count = count + 1

                

            if count == 0:

                iv_df = conv

            else:

                iv_df = iv_df.append(conv,ignore_index=True)

    

    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})

    iv = iv.reset_index()

    return(iv_df,iv)
final_iv, IV = data_vars(df,df['Personal Loan'])
IV.sort_values('IV')
df['California']=(df['ZIP Code']<96200).astype(int)



df['undergraduate']=(df['Education']==1).astype(int)

df['graduate']=(df['Education']==2).astype(int)



df['family_1']=(df['Family']==1).astype(int)

df['family_2']=(df['Family']==2).astype(int)

df['family_3']=(df['Family']==3).astype(int)





df=df.drop('ZIP Code',axis=1)

df=df.drop('Education',axis=1)

df=df.drop('Family',axis=1)
df['Age_0_25']=(df['Age']<=25).astype(int)

df['Age_25_30']=(df['Age']>25).astype(int) & (df['Age']<=30).astype(int)

df['Age_30_35']=(df['Age']>30).astype(int) & (df['Age']<=35).astype(int)

df['Age_35_40']=(df['Age']>35).astype(int) & (df['Age']<=40).astype(int)

df['Age_40_45']=(df['Age']>40).astype(int) & (df['Age']<=45).astype(int)

df['Age_45_50']=(df['Age']>45).astype(int) & (df['Age']<=50).astype(int)

df['Age_50_55']=(df['Age']>50).astype(int) & (df['Age']<=55).astype(int)

df['Age_55_60']=(df['Age']>55).astype(int) & (df['Age']<=60).astype(int)

df['Age_60_65']=(df['Age']>60).astype(int) & (df['Age']<=65).astype(int)



df['CC_0_1']=(df['CCAvg']<=1).astype(int)

df['CC_1_2']=(df['CCAvg']>1).astype(int) & (df['CCAvg']<=2).astype(int)

df['CC_2_3']=(df['CCAvg']>2).astype(int) & (df['CCAvg']<=3).astype(int)

df['CC_3_4']=(df['CCAvg']>3).astype(int) & (df['CCAvg']<=4).astype(int)

df['CC_4_5']=(df['CCAvg']>4).astype(int) & (df['CCAvg']<=5).astype(int)

df['CC_5_6']=(df['CCAvg']>5).astype(int) & (df['CCAvg']<=6).astype(int)

df['CC_6_7']=(df['CCAvg']>6).astype(int) & (df['CCAvg']<=7).astype(int)

df['CC_7_8']=(df['CCAvg']>7).astype(int) & (df['CCAvg']<=8).astype(int)

df['CC_8_9']=(df['CCAvg']>8).astype(int) & (df['CCAvg']<=9).astype(int)



df['Income_0_20']=(df['Income']<=20).astype(int)

df['Income_20_40']=(df['Income']>20).astype(int) & (df['Income']<=40).astype(int)

df['Income_40_60']=(df['Income']>40).astype(int) & (df['Income']<=60).astype(int)

df['Income_60_80']=(df['Income']>60).astype(int) & (df['Income']<=80).astype(int)

df['Income_80_100']=(df['Income']>80).astype(int) & (df['Income']<=100).astype(int)

df['Income_100_120']=(df['Income']>100).astype(int) & (df['Income']<=120).astype(int)

df['Income_120_140']=(df['Income']>120).astype(int) & (df['Income']<=140).astype(int)

df['Income_140_160']=(df['Income']>140).astype(int) & (df['Income']<=160).astype(int)

df['Income_160_180']=(df['Income']>160).astype(int) & (df['Income']<=180).astype(int)

df['Income_180_200']=(df['Income']>180).astype(int) & (df['Income']<=200).astype(int)



df['Mortgage_0_75']=(df['Mortgage']==0).astype(int)

df['Mortgage_75_125']=(df['Mortgage']>=75).astype(int) & (df['Mortgage']<125).astype(int)

df['Mortgage_125_175']=(df['Mortgage']>=125).astype(int) & (df['Mortgage']<175).astype(int)

df['Mortgage_175_225']=(df['Mortgage']>=175).astype(int) & (df['Mortgage']<225).astype(int)

df['Mortgage_225_275']=(df['Mortgage']>=225).astype(int) & (df['Mortgage']<275).astype(int)

df['Mortgage_275_325']=(df['Mortgage']>=275).astype(int) & (df['Mortgage']<325).astype(int)

df['Mortgage_325_400']=(df['Mortgage']>=325).astype(int) & (df['Mortgage']<400).astype(int)

df['Mortgage_400_500']=(df['Mortgage']>=400).astype(int) & (df['Mortgage']<500).astype(int)



df[['Age_sq','Income_sq','CCAvg_sq','Mortgage_sq']]=df[['Age','Income','CCAvg','Mortgage']].apply(lambda x: np.square(x))

df[['Age_sqrt','Income_sqrt','CCAvg_sqrt','Mortgage_sqrt']]=df[['Age','Income','CCAvg','Mortgage']].apply(lambda x: np.sqrt(x))

df[['Age_ln','Income_ln','CCAvg_ln','Mortgage_ln']]=df[['Age','Income','CCAvg','Mortgage']].apply(lambda x: np.log(x))

#df[['Age_Inv','Income_Inv','CCAvg_Inv']]=df[['Age','Income','CCAvg']].apply(lambda x: np.reciprocal(x))





#df=df.drop('Age',axis=1)

#df=df.drop('CCAvg',axis=1)

#df=df.drop('Income',axis=1)

#df=df.drop('Mortgage',axis=1)
df.loc[df['Mortgage_ln']<0,['Mortgage_ln']]= 0
#sns.set(style='ticks',color_codes=True)

#sns.catplot(x='Personal Loan',y='Mortgage',data=df, height=7, aspect=2)

correl = df.corr().abs()



# Select upper triangle of correlation matrix

upper = correl.where(np.triu(np.ones(correl.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.75

to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]

to_drop

correl.to_csv('file.csv',index=False)

correl.style.background_gradient(cmap='coolwarm').set_precision(2)

#df.drop(df[to_drop], axis=1)
# can remove age and its transformations

# can remove mortgage and its transformations except mortgage_sq

# can remove income transformations

# can remove all CCavg variables

df.columns

df_new=df.drop(['Age','CCAvg','Income','Mortgage','Age_sq', 'Income_sq', 'CCAvg_sq',

                'Age_sqrt','Income_sqrt', 'CCAvg_sqrt', 'Mortgage_sqrt', 'Age_ln','Income_ln',

                'CCAvg_ln', 'Mortgage_ln','Age_0_25', 'Age_25_30', 'Age_30_35','Age_35_40', 

                'Age_40_45', 'Age_45_50', 'Age_50_55', 'Age_55_60','Age_60_65','California',

                'Mortgage_sq'],axis=1)



# no dependency on online, creditcard, securities account

# Customers having high CCAvg need personal loan

# Family with income less than 100k are less likely to take loan

# higher mortgage is more likely to get the loan

# Income with more than 50 is more likely to get the personal loan

# customers having COD account have high prob of taking loan

# undergraduate has very less prob of taking the loan

# family size of more than 3 are more likely to get the loan

# 'CC_0_1', 'CC_1_2', 'CC_2_3', 'CC_3_4', 'CC_4_5', 'CC_5_6','CC_6_7', 'CC_7_8', 'CC_8_9',

# 'Mortgage_0_75', 'Mortgage_75_125', 'Mortgage_125_175','Mortgage_175_225','Mortgage_225_275', 'Mortgage_275_325', 'Mortgage_325_400', 'Mortgage_400_500',
df_new.columns
#sns.set(style='ticks',color_codes=True)

#sns.catplot(x='Personal Loan',y='Mortgage',data=df, height=7, aspect=2)

correl1 = df_new.corr().abs()



# Select upper triangle of correlation matrix

upper = correl1.where(np.triu(np.ones(correl1.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.75

to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]

to_drop

correl1.to_csv('file.csv',index=False)

correl1.style.background_gradient(cmap='coolwarm').set_precision(2)

#df.drop(df[to_drop], axis=1)
from scipy.stats import zscore

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

#cols = list(df_new.columns)

#a, b = cols.index('Personal Loan'), cols.index('Income')

#cols[b], cols[a] = cols[a], cols[b]

#df_new1=df_new[cols]

X=df_new.iloc[:,1:]

y=df_new.iloc[:,0]

#X_array=X[:].values.astype(float)

#min_max_scaler = MinMaxScaler()

#scaled_array = min_max_scaler.fit_transform(X_array)

#scaled_df=pd.DataFrame(scaled_array)

#X=scaled_df.iloc[:,:]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
import statsmodels.api as sm

logit_model=sm.Logit(endog=y_train,exog=X_train)

result=logit_model.fit()

print(result.summary())
def elimination(x,sl,y):

    numvars=len(x.columns)

    for i in range(0,numvars):

        lr=sm.Logit(y,x.values).fit()

        maxvar=max(lr.pvalues)

        if maxvar>sl:

            for j in range(0,numvars-i):

                if(lr.pvalues[j]==maxvar):

                    del x[x.columns[j]]

    lr.summary()

    return x



sl = 0.05

x_model=elimination(X_train,sl,y_train)

lr=sm.Logit(endog=y_train,exog=x_model).fit()

print(lr.summary())



vif=pd.DataFrame()

vif['VIF Factor']=[variance_inflation_factor(x_model.values,i) for i in range(x_model.shape[1])]

vif['features']=x_model.columns

vif
del x_model['CC_0_1']

vif=pd.DataFrame()

vif['VIF Factor']=[variance_inflation_factor(x_model.values,i) for i in range(x_model.shape[1])]

vif['features']=x_model.columns

vif
del x_model['Online']

vif=pd.DataFrame()

vif['VIF Factor']=[variance_inflation_factor(x_model.values,i) for i in range(x_model.shape[1])]

vif['features']=x_model.columns

vif
del x_model['CreditCard']

vif=pd.DataFrame()

vif['VIF Factor']=[variance_inflation_factor(x_model.values,i) for i in range(x_model.shape[1])]

vif['features']=x_model.columns

vif
del x_model['CC_8_9']

vif=pd.DataFrame()

vif['VIF Factor']=[variance_inflation_factor(x_model.values,i) for i in range(x_model.shape[1])]

vif['features']=x_model.columns

vif
del x_model['CC_1_2']

del x_model['CC_2_3']

del x_model['CC_7_8']

vif=pd.DataFrame()

vif['VIF Factor']=[variance_inflation_factor(x_model.values,i) for i in range(x_model.shape[1])]

vif['features']=x_model.columns

vif
del x_model['CC_6_7']

del x_model['Income_80_100']

vif=pd.DataFrame()

vif['VIF Factor']=[variance_inflation_factor(x_model.values,i) for i in range(x_model.shape[1])]

vif['features']=x_model.columns

vif
mylist=list(x_model.columns)

X_test=X_test.loc[:, X_test.columns.str.contains('|'.join(mylist))]
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

model1 = LogisticRegression()

model2= DecisionTreeClassifier(random_state = 0)

model3= RandomForestClassifier(n_estimators=10,random_state = 0)

model4= GaussianNB()

model1.fit(x_model, y_train)

model2.fit(x_model, y_train)

model3.fit(x_model, y_train)

model4.fit(x_model, y_train)



y_pred1 = model1.predict(X_test)

y_pred2 = model2.predict(X_test)

y_pred3 = model3.predict(X_test)

y_pred3_t = model3.predict(x_model)

y_pred4 = model4.predict(X_test)
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, y_pred1)

plt.show()

skplt.metrics.plot_confusion_matrix(y_test, y_pred2)

plt.show()

skplt.metrics.plot_confusion_matrix(y_test, y_pred3)

plt.show()

skplt.metrics.plot_confusion_matrix(y_train, y_pred3_t)

plt.show()
y_probas1 = model1.predict_proba(X_test)

skplt.metrics.plot_cumulative_gain(y_test, y_probas1)

plt.show()



y_probas2 = model2.predict_proba(X_test)

skplt.metrics.plot_cumulative_gain(y_test, y_probas2)

plt.show()



y_probas3 = model3.predict_proba(X_test)

skplt.metrics.plot_cumulative_gain(y_test, y_probas3)

plt.show()



y_probas3_t = model3.predict_proba(x_model)

skplt.metrics.plot_cumulative_gain(y_train, y_probas3_t)

plt.show()
skplt.metrics.plot_precision_recall(y_test, y_probas1)

plt.show()



skplt.metrics.plot_precision_recall(y_test, y_probas2)

plt.show()



skplt.metrics.plot_precision_recall(y_test, y_probas3)

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred3))
Importance = pd.DataFrame({'Importance':model3.feature_importances_*100}, index=x_model.columns)

Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kfold = model_selection.KFold(n_splits=10, random_state=12345)

cv_results1 = model_selection.cross_val_score(model1, X, y, cv=kfold, scoring='accuracy')

cv_results2 = model_selection.cross_val_score(model2, X, y, cv=kfold, scoring='accuracy')

cv_results3 = model_selection.cross_val_score(model3, X, y, cv=kfold, scoring='accuracy')

print(cv_results1.mean())

print(cv_results2.mean())

print(cv_results3.mean())