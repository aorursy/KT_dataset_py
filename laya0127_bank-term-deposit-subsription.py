import seaborn as sns

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt
df =pd.read_csv("/kaggle/input/portuguesebankidirectmarkcampaignsphonecalls/bank-full2.csv")
df.head()
df.info()
df.describe()
df['y'].value_counts()
df['y'].value_counts(normalize=True)
df2=df.copy()
df['y']=df['y'].map({'no':0,'yes':1})
df.head()
df.isnull().sum()
cat_col = df.select_dtypes('object').columns
cat_col
for i in cat_col:

    

    print("Feature : ",i)

    print(df[i].value_counts(normalize=True))

    print("\n")
df['poutcome'].value_counts(normalize=True)
df[df['poutcome']=='unknown'][['pdays','poutcome']]
df[(df['poutcome']=='unknown') & (df['pdays']!=-1)][['pdays','poutcome']]
df[df['pdays']==-1].shape
df[(df['poutcome']=='unknown') & (df['pdays']==-1)].shape
df.loc[(df['poutcome']=='unknown') & (df['pdays']==-1),'poutcome'] = 'not_contacted_prev'
df[(df['poutcome']=='not_contacted_prev') & (df['pdays']==-1)].shape
df[df['poutcome']=='other']
df[(df['poutcome']=='other') & (df['pdays']==-1)]
print("Analysis of feature : 'Poutcome'")

fig, ax1 = plt.subplots(1,2,figsize=(15,5))

sns.countplot(y='poutcome',order=df['poutcome'].value_counts().index,data=df,ax=ax1[0])

prop_df = (df['y'].groupby(df['poutcome']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())

sns.barplot(y='poutcome',x='Percentage',hue='y',data=prop_df,order=df['poutcome'].value_counts().index,ax=ax1[1])

ax1[1].set(xticks = np.array(range(0,100,5)))

plt.show()
prop_df
fig, ax1 = plt.subplots(1,2,figsize=(15,7))

sns.countplot(x='poutcome',data=df,hue='y',ax=ax1[0])

prop_df = (df['poutcome'].groupby(df['y']).value_counts().rename('Values').reset_index())

prop_df = prop_df[prop_df['y']==1]

plt.pie(prop_df['Values'],labels=prop_df['poutcome'],autopct='%1.1f%%')

ax1[1].set_title("Percentage of each category out of Subcribed people")

plt.show()
df['job'].value_counts(normalize=True)
df[df['job']=='unknown'].shape
df[(df['job']=='unknown') & (df['pdays']==-1)].shape
print("Analysis of feature : 'Job'")

fig, ax1 = plt.subplots(1,2,figsize=(15,5))

sns.countplot(y='job',order=df['job'].value_counts().index,data=df,ax=ax1[0])

prop_df = (df['y'].groupby(df['job']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())

sns.barplot(y='job',x='Percentage',hue='y',data=prop_df,order=df['job'].value_counts().index,ax=ax1[1])

ax1[1].set(xticks = np.array(range(0,100,5)))

plt.show()
prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)
fig, ax1 = plt.subplots(1,2,figsize=(15,7))

sns.countplot(y='job',data=df,hue='y',order=df['job'].value_counts().index,ax=ax1[0])

prop_df = (df['job'].groupby(df['y']).value_counts().rename('Values').reset_index())

prop_df = prop_df[prop_df['y']==1]

plt.pie(prop_df['Values'],labels=prop_df['job'],autopct='%1.1f%%')

ax1[1].set_title("Percentage of each category out of Subcribed people")

plt.show()
df['education'].value_counts(normalize=True)
df[df['education']=='unknown'].shape
df[(df['education']=='unknown') & (df['pdays']==-1)].shape
print("Analysis of feature : 'Education'")

fig, ax1 = plt.subplots(1,2,figsize=(15,5))

sns.countplot(y='education',order=df['education'].value_counts().index,data=df,ax=ax1[0])

prop_df = (df['y'].groupby(df['education']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())

sns.barplot(y='education',x='Percentage',hue='y',data=prop_df,order=df['education'].value_counts().index,ax=ax1[1])

ax1[1].set(xticks = np.array(range(0,100,5)))

plt.show()
prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)
fig, ax1 = plt.subplots(1,2,figsize=(15,7))

sns.countplot(y='education',data=df,hue='y',order=df['education'].value_counts().index,ax=ax1[0])

prop_df = (df['education'].groupby(df['y']).value_counts().rename('Values').reset_index())

prop_df = prop_df[prop_df['y']==1]

plt.pie(prop_df['Values'],labels=prop_df['education'],autopct='%1.1f%%')

ax1[1].set_title("Percentage of each category out of Subcribed people")

plt.show()
df['marital'].value_counts(normalize=True)
print("Analysis of feature : 'Marital'")

fig, ax1 = plt.subplots(1,2,figsize=(15,5))

sns.countplot(y='marital',order=df['marital'].value_counts().index,data=df,ax=ax1[0])

prop_df = (df['y'].groupby(df['marital']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())

sns.barplot(y='marital',x='Percentage',hue='y',data=prop_df,order=df['marital'].value_counts().index,ax=ax1[1])

ax1[1].set(xticks = np.array(range(0,100,5)))

plt.show()
prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)
fig, ax1 = plt.subplots(1,2,figsize=(15,7))

sns.countplot(y='marital',data=df,hue='y',order=df['marital'].value_counts().index,ax=ax1[0])

prop_df = (df['marital'].groupby(df['y']).value_counts().rename('Values').reset_index())

prop_df = prop_df[prop_df['y']==1]

plt.pie(prop_df['Values'],labels=prop_df['marital'],autopct='%1.1f%%')

ax1[1].set_title("Percentage of each category out of Subcribed people")

plt.show()
df['default'].value_counts(normalize=True)
print("Analysis of feature : 'Default'")

fig, ax1 = plt.subplots(1,2,figsize=(15,5))

sns.countplot(y='default',order=df['default'].value_counts().index,data=df,ax=ax1[0])

prop_df = (df['y'].groupby(df['default']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())

sns.barplot(y='default',x='Percentage',hue='y',data=prop_df,order=df['default'].value_counts().index,ax=ax1[1])

ax1[1].set(xticks = np.array(range(0,100,5)))

plt.show()
prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)
fig, ax1 = plt.subplots(1,2,figsize=(15,7))

sns.countplot(y='default',data=df,hue='y',order=df['default'].value_counts().index,ax=ax1[0])

prop_df = (df['default'].groupby(df['y']).value_counts().rename('Values').reset_index())

prop_df = prop_df[prop_df['y']==1]

plt.pie(prop_df['Values'],labels=prop_df['default'],autopct='%1.1f%%')

ax1[1].set_title("Percentage of each category out of Subcribed people")

plt.show()
df['housing'].value_counts(normalize=True)
print("Analysis of feature : 'Housing'")

fig, ax1 = plt.subplots(1,2,figsize=(15,5))

sns.countplot(y='housing',order=df['housing'].value_counts().index,data=df,ax=ax1[0])

prop_df = (df['y'].groupby(df['housing']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())

sns.barplot(y='housing',x='Percentage',hue='y',data=prop_df,order=df['housing'].value_counts().index,ax=ax1[1])

ax1[1].set(xticks = np.array(range(0,100,5)))

plt.show()
prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)
fig, ax1 = plt.subplots(1,2,figsize=(15,7))

sns.countplot(y='housing',data=df,hue='y',order=df['housing'].value_counts().index,ax=ax1[0])

prop_df = (df['housing'].groupby(df['y']).value_counts().rename('Values').reset_index())

prop_df = prop_df[prop_df['y']==1]

plt.pie(prop_df['Values'],labels=prop_df['housing'],autopct='%1.1f%%')

ax1[1].set_title("Percentage of each category out of Subcribed people")

plt.show()
df['housing'].value_counts(normalize=True)
print("Analysis of feature : 'Housing'")

fig, ax1 = plt.subplots(1,2,figsize=(15,5))

sns.countplot(y='housing',order=df['housing'].value_counts().index,data=df,ax=ax1[0])

prop_df = (df['y'].groupby(df['housing']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())

sns.barplot(y='housing',x='Percentage',hue='y',data=prop_df,order=df['housing'].value_counts().index,ax=ax1[1])

ax1[1].set(xticks = np.array(range(0,100,5)))

plt.show()
prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)
fig, ax1 = plt.subplots(1,2,figsize=(15,7))

sns.countplot(y='housing',data=df,hue='y',order=df['housing'].value_counts().index,ax=ax1[0])

prop_df = (df['housing'].groupby(df['y']).value_counts().rename('Values').reset_index())

prop_df = prop_df[prop_df['y']==1]

plt.pie(prop_df['Values'],labels=prop_df['housing'],autopct='%1.1f%%')

ax1[1].set_title("Percentage of each category out of Subcribed people")

plt.show()
cat_col
df[df['education']=='unknown'].shape
df[(df['education']=='unknown') & (df['pdays']==-1)].shape
df[df['contact']=='unknown'].shape
df[(df['contact']=='unknown') & (df['pdays']==-1)].shape
num_cols = df.select_dtypes('int64').columns
for i in num_cols:

    sns.boxplot(y=i,data = df)

    plt.show()
for i in num_cols:

    sns.boxplot(y='age',x='y',data = df)

    plt.show()
#df_log = df[num_cols]

#df_log.apply(np.log)
sns.pairplot(df2)
sns.pairplot(df2,hue='y')
df2.corr()
for i in num_cols:

    sns.barplot(x='y',y=i,data = df)

    plt.show()
df0=df[df['y']=='no']

df1=df[df['y']=='yes']
for i in num_cols:

    sns.distplot(df0[i])

    sns.distplot(df1[i])

    plt.show()
cat_col1 = df2.select_dtypes('object').columns

cat_col1
df2['prev_contacted'] = list(map(lambda x : 'Yes' if x != -1 else 'No',df2['pdays']))
df2[df2['prev_contacted']=='Yes']
#fig, ax1 = plt.subplots(figsize=(15,5))

#plt.figure(figsize=(15,5))



for col in cat_col1:

    print("Analysis of feature : ",col)

    fig, ax1 = plt.subplots(1,2,figsize=(15,5))

    #sns.countplot(y=col,order=df[col].value_counts().index,hue='y',data=df,ax=ax1[0])

    sns.countplot(y=col,order=df2[col].value_counts().index,data=df2,ax=ax1[0])

    prop_df = (df2['y'].groupby(df2[col]).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())

    sns.barplot(y=col,x='Percentage',hue='y',data=prop_df,order=df2[col].value_counts().index,ax=ax1[1])

    ax1[1].set(xticks = np.array(range(0,100,5)))

    plt.show()
for col in cat_col1:

    print("Analysis of feature : ",col)

    print(df2.groupby(col)['y'].value_counts(normalize=True))
df[df['campaign']==0]