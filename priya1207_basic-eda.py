# import dependencies

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
# read data

df = pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")



# view head of data

df.head()



#rename column for convenience

df.rename(columns ={"default.payment.next.month":"default payment" },inplace=True)
# Shape of the data

print(df.shape)

# features included

df.columns
df.info()
# Cardinality

df.nunique()



# Due to the cardinality and features description we can conclude that 

# 'SEX','EDUCATION', 'MARRIAGE' can be considered as categorical features
# Check for missing values



sns.heatmap(df.isna())
# Categorical features

df[['SEX','EDUCATION','MARRIAGE']].describe()
# Numeric feature 'PAY'

df[['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].describe()
# Numeric feature 'BILL_AMT'

df[['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].describe()
# Numeric feature 'PAY_AMT'

df[['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].describe()
# the data seems to be imbalanced

sns.countplot(df['default payment'])
# General visualization



features_cat  = ['SEX','EDUCATION','MARRIAGE']



fig, axes = plt.subplots(ncols=2,nrows=3,figsize=(15,9))



for i in range(0,len(features_cat)):

    sns.countplot(x = df[features_cat[i]], hue = df['default payment'],ax = axes[i][0] )

    sns.countplot(df[features_cat[i]],ax = axes[i][1])



# Ratio of defaulters in each class of categorical features



fig, axes = plt.subplots(ncols=1,nrows=3,figsize=(15,10))



features_cat  = ['SEX','EDUCATION','MARRIAGE']



for i in range(0,len(features_cat)):

    feature = features_cat[i]

    classes = df[feature].unique().tolist()

    ratio = []

    for c in classes:

        total = df[ df[feature] == c ].shape[0]

        defaulters = df[ (df[feature] == c) & (df['default payment']==1)].shape[0]

        r = defaulters/total

        ratio.append(r)

    sns.barplot(x = classes,y=ratio,ax = axes[i])

    axes[i].set_title(feature)

    
# for PAY

fig,axes = plt.subplots(ncols=2,nrows=3,figsize=(15,10))

sns.countplot(x = 'PAY_0',hue = 'default payment',data=df,ax = axes[0][0])

sns.countplot(x = 'PAY_2',hue = 'default payment',data=df,ax = axes[0][1])

sns.countplot(x = 'PAY_3',hue = 'default payment',data=df,ax = axes[1][0])

sns.countplot(x = 'PAY_4',hue = 'default payment',data=df,ax = axes[1][1])

sns.countplot(x = 'PAY_5',hue = 'default payment',data=df,ax = axes[2][0])

sns.countplot(x = 'PAY_6',hue = 'default payment',data=df,ax = axes[2][1])
def plot_numeric_feature(feature,feature_list,df):

    default = df[ df['default payment']==1 ]

    non_default = df[ df['default payment']==0 ]

    

    if (len(feature_list)==1):

        fig, axes = plt.subplots(ncols=1,nrows=1,figsize=(9,6))

    else:

        fig, axes = plt.subplots(ncols=2,nrows=3,figsize=(15,9))

        

    fig.suptitle("Distributions for " + feature)



    for i in range (0,len(feature_list)):

        f = feature_list[i]

        if (i%2==0):

            k = 0

            j = int(i/2)

        else:

            k = 1

            j = int(i/2)

        if (len(feature_list)!=1):

            sns.distplot(default[f],label="default",color='red',ax = axes[j][k])

            sns.distplot(non_default[f],label="non-default",color='green',ax=axes[j][k])

        else:

            sns.distplot(default[f],label="default",color='red')

            sns.distplot(non_default[f],label="non-default",color='green')

    fig.legend()

    

            

            

    
# 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'

# 'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'

# 'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'



feature_pay = ['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

feature_bill_amt = ['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

feature_pay_amt = ['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']



plot_numeric_feature("PAY",feature_pay,df)



plot_numeric_feature("BILL AMT",feature_bill_amt,df)



plot_numeric_feature("PAY AMT",feature_pay_amt,df)



# The distributions for all numeric features appear similar for defaulters and non-defaulters



bal = ['LIMIT_BAL']

plot_numeric_feature("LIMIT_BAL",bal,df)



# The distribution for defaulters and non-defaulters is similar, no starking differences observed.

# The max balance limit for defaulers and non-defaulters is same.



# The height difference indicates that people with limit balance less than 100000 tend to default more.



# Which implies that given that a person has limit balance value <= 100000  , it is more likely that he/she

# will default.
df_defaulter_education = df[ (df['EDUCATION']==2) | (df['EDUCATION']==3) ]

bal = ['LIMIT_BAL']

plot_numeric_feature("LIMIT_BAL for education category 2 or 3",bal,df_defaulter_education)
values = df['LIMIT_BAL'].value_counts().index

freq = df['LIMIT_BAL'].value_counts().values

defaults = [   df[ (df['LIMIT_BAL'] == bal) & (df['default payment']==1) ].shape[0] for bal in values    ]



df_limit_bal = pd.DataFrame(list(zip(values, freq,defaults)), columns =['values', 'freq','defaults'])  



sns.set_color_codes("pastel")

fig = plt.subplots(figsize=(15,20))



ax = sns.barplot(x="freq", y="values", data=df_limit_bal,label="total",orient='h',color="b")



sns.barplot(x="defaults", y="values", data=df_limit_bal,label="defaults-involved",orient='h', color="r")



ax.set(ylabel="balance values",xlabel="Frequency count")

plt.legend()

df_limit_bal['default_ratio'] = df_limit_bal.apply(lambda x: x['defaults']/x['freq'],axis=1 )



sns.set_color_codes("pastel")

fig = plt.subplots(figsize=(15,20))



sns.barplot(x = "default_ratio",y = "values",data=df_limit_bal,orient='h')

# default ratio is max for bal = 327680 followed by bal = 740000, but the sample size is very small hence we cant

# cant conclude



df_limit_bal.sort_values(by = "default_ratio",ascending=False).tail(25)
# PAY

# 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'

data_pay = df[['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

sns.heatmap(data_pay.corr())



# Correlation decreases with previous previous months. 

# There is least correlation between PAY_0(first value) and PAY_6(latest value)
# BILL AMT

# 'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'

data_bill_amt = df[['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]

sns.heatmap(data_bill_amt.corr())



# BILL AMT also shows similar trends like PAY
# PAY AMT

# 'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'



data_pay_amt = df[['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

sns.heatmap(data_pay_amt.corr())



# No correlation between pay amounts at all.
# Scale the data

scaler = StandardScaler()

scaler.fit(df.drop('default payment',axis=1))

scaled_features = scaler.transform(df.drop('default payment',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_feat['default payment'] = df['default payment']
df_feat.head()