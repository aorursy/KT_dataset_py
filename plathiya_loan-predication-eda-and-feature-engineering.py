'''

(Features)

Loan_ID	            - Unique Loan ID                                String

Gender              - Gender of person                              Categorical(nominal): Male/ Female

Married	            - Applicant married                             Categorical(nominal): Y/N

Dependents          - Number of people dependent on that person     Categorical(nominal): 0/1/2/3+

Education           - Applicant Education                           Categorical(nominal): Graduate/ Not Graduate

Self_Employed	    - Self employed                                 Categorical(nominal): Y/N

ApplicantIncome	    - Applicant income                              Numerical           : in $'s

CoapplicantIncome   - Coapplicant income                            Numerical           : in $'s

LoanAmount	        - Loan amount                                   Numerical           : in thousands of $'s

Loan_Amount_Term	- Term of loan                                  Numerical           : in number of months

Credit_History	    - Credit history meets guidelines               Categorical         : 1/0   

Property_Area	    - Borrower's property at stake location         Categorical         : Urban/ Semi Urban/ Rural



(Target)

Loan_Status         - Loan approved (Y/N)





Notes: 

1. Credit history => record of a borrower's responsible repayment of debts

2. A co-applicant refers to a person who applies along with the borrower for a loan. 

   This is done so that the income of the co-applicant can be used to supplement the borrower's income and increase his/her eligibility

3. Having dependents means you have higher commitments, which in turn lower your disposable income.

'''
# Standard imports

import numpy as np

import pandas as pd

from glob import glob

pd.set_option('display.max_columns',500)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Custom tools

from plotting_helper import *
data_train, data_test = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv', encoding='UTF-8'), pd.read_csv('../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv', encoding='UTF-8')

data_train['which_data'] = 'data_train'

data_test['which_data'] = 'data_test'
# Combine all data to fill/analyse/transform/etc. according to whole dataset(train/test)

data_all = pd.concat([data_train,data_test], axis=0)    # test target col will contain all nan's

data_train.shape, data_test.shape, data_all.shape
data_all.head()
data_all.info()
# Fill missing values with -1 for now so that errors are avoided while casting dtype

data_all.fillna(value=-1, inplace=True)
# Check Max values to alter columns' dtypes accordingly

data_all.max()[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]
# Reduce unnecessary memory usage

data_all['ApplicantIncome'] = data_all['ApplicantIncome'].astype('int32')

data_all['CoapplicantIncome'] = data_all['CoapplicantIncome'].astype('float32')

data_all[['LoanAmount','Loan_Amount_Term']] = data_all[['LoanAmount','Loan_Amount_Term']].astype('float16')

data_all['Credit_History'] = data_all['Credit_History'].astype('int8')

data_all[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']].dtypes
data_all.replace(-1,np.nan, inplace=True)
# Drop un-necessary cols

data_all.drop(labels=['Loan_ID'], inplace=True, axis=1)
# IS there any duplicate row? remove it.

data_train.duplicated().any()
# It should have been cleaned and numeric

data_all['Dependents'].unique()
dependent_dict = {'0':0,'1':1,'2':2,'3+':3}

data_all['Dependents'] = data_all['Dependents'].map(dependent_dict)
contineous_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',]

categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History'\

                    , 'Property_Area', 'Loan_Status']
# Overall Categorical data distribution

plt.figure(figsize=(20,8))

for index, col in enumerate(categorical_features, start=1):

    plt.subplot(2,4,index)

    plt.title(col)

    plt.pie(data_all[col].value_counts().values,autopct='%1.0f%%', labels=data_all[col].value_counts().index)

plt.tight_layout()
# Overall Contineous Data Distribution(ideally, should be normal/gaussian distributed)

plt.figure(figsize=(20,5))

plt.tight_layout()

for index, col in enumerate(contineous_features, start=1):

    plt.subplot(1,4,index)

    plt.title(col)

    plt.xlabel('Range')

    plt.ylabel('Values')

    sns.distplot(data_all[col], kde_kws={'bw':0.1})
# Analysing validity of default Mathemetical Outlier Removal techniques

# IQR Ranges

Q1 = data_all[contineous_features].quantile(0.25)

Q3 = data_all[contineous_features].quantile(0.75)

IQR = Q3-Q1



lower_range = Q1 - 1.5*IQR

upper_range = Q3 + 1.5*IQR

iqr_range = pd.DataFrame(pd.concat([lower_range,upper_range], axis=1))

iqr_range.columns = ['IQR Lower','IQR Upper']



# Z-score range

means = data_all[contineous_features].mean()

stds = data_all[contineous_features].std()



lower_zscore = means - 3*stds

upper_zscore = means + 3*stds

z_range = pd.DataFrame(pd.concat([lower_zscore,upper_zscore], axis=1))

z_range.columns =['Z Lower','Z Upper']



# Both Into DataFrame

pd.concat([iqr_range,z_range], axis=1)
# Looking for patterns between any two features

sns.pairplot(data_all)
# Looking for individual features' relation with Loan Status

plt.figure(figsize=(20,10))

for index,col in enumerate(categorical_features, start=1):

    plt.subplot(2,4,index)

    plot_frequency_sns(data=data_all, feature_name=col, hue='Loan_Status', annotate=True, annotate_distance=5, annotate_rotation='horizontal', palette='Blues')

    plt.legend(['Loan Dis-approved','Loan Approved']);

plt.tight_layout()
# color palettes: BuGn_r,Blues,GnBu_d

for index,col in enumerate(categorical_features, start=1):

    data = pd.crosstab(data_all[col],data_all['Loan_Status'])

    data.div(data.sum(axis=1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
# Looking for a pattern combining two two features with Loan Status

plt.figure(figsize=(20,10))

index=1

for cat_col,cat_hue in zip(['Married','Married','Married','Gender','Self_Employed','Self_Employed','Dependents','Property_Area']\

                           ,['Education','Self_Employed','Gender','Dependents','Education',None,'Married','Education']):

    plt.subplot(2,4,index)

    plot_frequency_sns(data=data_all, feature_name=cat_col, hue=cat_hue, annotate=True, annotate_distance=5, annotate_rotation='horizontal', palette='Blues')

    plt.legend(data_all[cat_hue].unique()) if cat_hue != None else _

    index+=1

plt.tight_layout()
print(f'Married-Graduate: {275/72, 485/146}')

print(f'Self_Employed-Graduate: {626/181, 94/25}')

print(f'Dependents-Married: {(((124+146+79)/3)/((124+146+79+36+14+12)/6))*100:.2f} %')

print(273/69, 216/74, 274/75)

626/181, 94/25
sns.scatterplot(data_all['Self_Employed'], data_all['LoanAmount'])

plt.axhline(y=300, c='green', alpha=0.3, linestyle='--');
data_all[  (data_all['Dependents']==0) & (data_all['Married']=='No') & (data_all['Education']=='Not Graduate')]['Credit_History'].value_counts(normalize=True)*100
data_all[  (data_all['Dependents']==0) & (data_all['Married']=='No') & (data_all['Education']=='Graduate')]['Credit_History'].value_counts(normalize=True)
fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(20,5))



# Direct

data_all.groupby('Dependents')['Property_Area'].value_counts().plot(kind='bar', ax=axs[0]);



# More visual

dic = {}

for x in data_all.groupby('Property_Area')['Dependents']:

    dic[x[0]] = x[1].value_counts()

df = pd.DataFrame(dic)



df.plot(kind='bar', ax=axs[1])

plt.title('Number of dependents from different areas')

plt.xlabel('Count')

plt.ylabel('No. of people');
d = pd.crosstab(data_all['Credit_History'],data_all['Dependents']).T

dd = d.div(d.sum(axis=1), axis=0)*100

dd
plt.figure(figsize=(20,4))

plt.subplot(1,2,1)

sns.regplot(x='ApplicantIncome', y='LoanAmount', data=data_all);

plt.axvline(x=22_000, c='green', alpha=0.3, linestyle='--');



plt.subplot(1,2,2)

plt.plot(data_all['LoanAmount'], marker="*", linestyle='')

plt.plot(data_all['ApplicantIncome'], marker=".", linestyle='');
plt.figure(figsize=(20,6))

plt.plot(data_all['LoanAmount'][50:100].apply(lambda x: x/data_all['LoanAmount'][50:100].max()), marker="*", linestyle='-', label='LoanAmount')

plt.plot(data_all['ApplicantIncome'][50:100].apply(lambda x: x/data_all['ApplicantIncome'][50:100].max()), marker=".", linestyle='--', label='ApplicantIncome')

plt.xlabel('Data Points')

plt.ylabel('Amount (normalised)')

plt.title('LoanAmount vs ApplicantIncome')

plt.legend();
plt.figure(figsize=(20,8))

plt.title("Distribution of income according to applicant's property area ")

plt.axhline(y=22000, c='orange', alpha=0.7, linestyle='--')

sns.scatterplot(x=data_all['Property_Area'], y=data_all['ApplicantIncome']);
plt.figure(figsize=(20,4))

plt.subplot(1,2,1)



sns.scatterplot(x=data_all['Gender'], y=data_all['ApplicantIncome'])

max_female_income = data_all[ data_all['Gender']=='Female' ]['ApplicantIncome'].max()

plt.annotate(s=max_female_income, xy=(1,-1), xytext=(0.9,max_female_income+3000))

plt.axhline(y=max_female_income, c='orange', alpha=0.7, linestyle='--');



plt.subplot(1,2,2)

sns.scatterplot(x=data_all['Gender'], y=data_all['CoapplicantIncome'])

max_female_co_income = data_all[ data_all['Gender']=='Female' ]['CoapplicantIncome'][284]

plt.annotate(s=max_female_co_income, xy=(1,-1), xytext=(0.9,max_female_co_income+3000))

plt.axhline(y=max_female_co_income, c='orange', alpha=0.7, linestyle='--');
plt.figure(figsize=(20,4))

plt.subplot(1,2,1)

sns.scatterplot(x='ApplicantIncome', y='Dependents', data=data_all, hue='Credit_History');

plt.axvline(x=13_000, linestyle='--', c='g', alpha=0.3)

plt.axhline(y=2, linestyle='--', c='g', alpha=0.3)

plt.subplot(1,2,2)

plot_frequency_sns(data=data_all, feature_name="Dependents", hue="Credit_History", annotate=True, annotate_distance=3);
423/78, 125/20, 125/25, 63/20
data_all.groupby('Education')['ApplicantIncome'].max(), data_all.groupby('Self_Employed')['ApplicantIncome'].max()
plt.figure(figsize=(20,4))

plt.subplot(1,2,1)

sns.scatterplot(data_all['Education'], data_all['ApplicantIncome'])

plt.axhline(18165, linestyle="--", alpha=0.3)



plt.subplot(1,2,2)

sns.scatterplot(data_all['Self_Employed'], data_all['ApplicantIncome'])

plt.axhline(39147, linestyle="--", alpha=0.3);
data_all.boxplot(column='ApplicantIncome', by = 'Education');

plt.ylabel('Applicant Income');

plt.title("");
data_all.head(2)
# Turn Applicant Income into Categorical Feature

bins=[0,2500,4000,6000,81000]

group=['Low','Average','High', 'Very high']

data_all['Income_bin'] = pd.cut(data_all['ApplicantIncome'],bins,labels=group)
# Check it's correlation with Loan Status Percentage wise

data = pd.crosstab(data_all['Income_bin'], data_all['Loan_Status'])

data.div(data.sum(axis=1), axis=0)*100
data.div(data.sum(axis=1), axis=0).plot(kind='bar', stacked=True);

plt.ylabel('Percentage');
# Let's apply the same concept on Co-applicant Income and form categories

bins=[0,1000,3000,42000] 

group=['Low','Average','High'] 

data_all['Coapplicant_Income_bin']=pd.cut(data_all['CoapplicantIncome'],bins,labels=group)



Coapplicant_Income_bin=pd.crosstab(data_all['Coapplicant_Income_bin'],data_all['Loan_Status']) 

Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

plt.xlabel('CoapplicantIncome') 

plt.ylabel('Percentage');
# Engineer Total Income feature and again make it categorical

data_all['Total_Income']=data_all['ApplicantIncome']+data_all['CoapplicantIncome']



bins=[0,2500,4000,6000,81000]

group=['Low','Average','High', 'Very high'] 

data_all['Total_Income_bin']=pd.cut(data_all['Total_Income'],bins,labels=group)



Total_Income_bin=pd.crosstab(data_all['Total_Income_bin'],data_all['Loan_Status'])

Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Total_Income') 

plt.ylabel('Percentage');
# Let's feature engineer another column(Note: Loan amount is in thousands)

interest_rate=0.08

data_all['EMI'] = data_all.apply(lambda x: ((x['LoanAmount']*1000)/x['Loan_Amount_Term']) ,axis=1)

data_all['Residual_monthly_income'] = (data_all['Total_Income']/12)-(data_all['EMI'])
# See their distributions

plt.figure(figsize=(20,4))

for i,col in enumerate(['Total_Income','EMI','Residual_monthly_income'],start=1):

    plt.subplot(1,3,i)

    sns.distplot(data_all[col], rug=True);
# Allocate negative status to people with -ve residual income and analyse with Credit History feature

data_all['Redisual_Status'] = data_all['Residual_monthly_income'].apply(lambda x: 0 if x<0 else 1)

d = pd.crosstab(data_all['Redisual_Status'], data_all['Credit_History'])

d.div(d.sum(axis=1), axis=0)*100
plot_frequency_sns(data=data_all, feature_name='Redisual_Status', annotate=True, annotate_distance=0, hue='Loan_Status')
data_all[ (data_all['Redisual_Status']==0) & (data_all['Loan_Status']=='Y')]['Credit_History'].value_counts()
data_all[ (data_all['Redisual_Status']==0) & (data_all['Loan_Status']=='Y') & (data_all['Credit_History']==0)]
# Check for a pattern of newely generated features with the Loan Status

plt.figure(figsize=(20,4))

for i,col in enumerate(['Total_Income','EMI','Residual_monthly_income'],start=1):

    plt.subplot(1,3,i)

    sns.scatterplot(x=data_all['Loan_Status'], y=data_all[col])

    if i==3:

        plt.axhline(0, linestyle="--", alpha=0.3, c='g');
data_all[ data_all['EMI']==data_all['EMI'].max() ]
data_all['Loan_Amount_Term'].value_counts()
sns.distplot(data_all['LoanAmount']);
# Add another feature

data_all['Remaining_family_income'] = data_all.apply(lambda x: x['Residual_monthly_income']/x['Dependents'] if x['Dependents']!=0 else x['Residual_monthly_income'],axis=1)

sns.scatterplot(x=data_all['Loan_Status'], y=data_all['Remaining_family_income'])

plt.axhline(0, linestyle="--", alpha=0.3, c='g');
# data_all['Loan_Status'] = data_all['Loan_Status'].map({'Y':1,'N':0})
sns.heatmap(data_all[['Dependents','Residual_monthly_income','Remaining_family_income','Credit_History','Loan_Status']].corr(), annot=True, cmap='YlGnBu');
data_all['Safe_Applicant'] = data_all.apply(lambda x: 1 if (x['Education']=='Graduate' and x['Dependents']>0 and x['Married']=='Yes' and x['Self_Employed']=='No' and (x['Income_bin']=='High' or x['Income_bin']=='Very high')) else 0, axis=1)

plot_frequency_sns(data=data_all, feature_name='Safe_Applicant', annotate=True,annotate_distance=-4, hue='Loan_Status')



d = pd.crosstab(data_all['Safe_Applicant'], data_all['Loan_Status'])

dd = d.div(d.sum(axis=1), axis=0)*100

dd.plot.bar(stacked=True);