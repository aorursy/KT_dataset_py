import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline
df1=pd.read_csv('../input/presidents_one.csv', low_memory=False)

df2=pd.read_csv('../input/presidents_two.csv', low_memory=False)
print(list(df1.columns))

print(list(df2.columns))
df2['Petitions Pending'] = df2['Petitions Pending (Commutations)'] + df2['Petitions Pending (Pardons)']

df2.drop(['Petitions Pending (Commutations)', 'Petitions Pending (Pardons)'], axis=1, inplace=True)



df2['Petitions Received'] = df2['Petitions Received (Pardons)'] + df2['Petitions Received (Commutations)']

df2.drop(['Petitions Received (Pardons)', 'Petitions Received (Commutations)'], axis=1, inplace=True)



df2['Petitions Denied'] = df2['Petitions Denied (Pardons)'] + df2['Petitions Denied (Commutations)']

df2.drop(['Petitions Denied (Commutations)', 'Petitions Denied (Pardons)'], axis=1, inplace=True)



df2['Petitions Closed Without Presidential Action'] = df2['Petitions Closed Without Presidential Action (Pardons)'] + df2['Petitions Closed Without Presidential Action (Commutations)']

df2.drop(['Petitions Closed Without Presidential Action (Pardons)', 'Petitions Closed Without Presidential Action (Commutations)'], axis=1, inplace=True)



df2['Petitions Denied or Closed Without Presidential Action'] = df2['Petitions Denied or Closed Without Presidential Action (Pardons)'] + df2['Petitions Denied or Closed Without Presidential Action (Commutations)']

df2.drop(['Petitions Denied or Closed Without Presidential Action (Pardons)', 'Petitions Denied or Closed Without Presidential Action (Commutations)'], axis=1, inplace=True)



df1.drop(['Respites'], axis=1, inplace=True)
#Check that all the columns match

list(map(lambda x,y: x==y,sorted(list(df1.columns)),sorted(list(df2.columns))))
#Check that all the columns are in the same order

list(map(lambda x,y: x==y,list(df1.columns),list(df2.columns)))
#Re-order df2 columns

df2=df2.loc[:,list(df1.columns)]
#Check again that all the columns are in the same order

list(map(lambda x,y: x==y,list(df1.columns),list(df2.columns)))
#Now that the columns match, we can combime the dataframes. But first, we make sure their order matches

df=pd.concat([df1,df2], axis=0, ignore_index=True)

df.head()
#A brief summary of the dataframe

df.describe()
#to do this, we use the xor operator: ^

all(pd.isnull(df['Petitions Denied or Closed Without Presidential Action']) ^ pd.isnull(df['Petitions Closed Without Presidential Action']))
df['Petitions Closed Without Presidential Action']=df['Petitions Denied or Closed Without Presidential Action'].fillna(0)+df['Petitions Closed Without Presidential Action'].fillna(0)

df.drop('Petitions Denied or Closed Without Presidential Action',axis=1,inplace=True)
df['Percent Granted']=df['Petitions Granted']/df['Petitions Received']

df['Percent Denied']=df['Petitions Denied']/df['Petitions Received']
# Visualize number of pardons each year



colors = sns.color_palette("Set1", len(df['President'].unique()))

im = sns.lmplot(data=df, x='Fiscal Year', y="Pardons", hue='President', fit_reg=False, palette=colors)

im.set(xlim=(1890,2020), title="Pardons Issued by Year")



#The first thing I notice is that for most presidents, the number of pardons increases during their tenure.  

#As expected, the number of pardons tends to be highest in their last year.
# Visualize number of pardons each year

colors = sns.color_palette("Set1", len(df['President'].unique()))

im = sns.lmplot(data=df, x='Fiscal Year', y="Petitions Granted", hue='President', fit_reg=False, palette=colors)

im.set(xlim=(1890,2020), title="Petitions Granted by Year")

#The first thing I notice is that for most presidents, the number of petitions granted increases during their tenure.  

#As expected, the number of pardons tends to be highest in their last year.
# Number of petitions

colors = sns.color_palette("Set1", len(df['President'].unique()))

g = sns.lmplot(data=df, x='Fiscal Year', y="Petitions Received", hue='President', fit_reg=False, palette=colors)

g.set(xlim=(1900,2020), title="Petitions Received by Year")
df['Year in Office']=pd.Series([1]*len(df))

df.loc[0,'Year in Office']=3 #data starts at McKinley's third year in office

for i in range(1,len(df)):

    if df.loc[i-1, 'President'] == df.loc[i,'President']:

        df.loc[i,'Year in Office'] = df.loc[i-1,'Year in Office']+1



# NOTE THAT THIS CAN BE VECTORIZED using the pandas .shift(1) function, 

# but since the dataset is so small, I chose readability over the unnoticeable speedup.
df['Year in Office'].value_counts()
#combine two rows if the second is the 20-day period at the end of a president's outgoing term

to_drop=[]

for i in range(0,len(df)-1):

    if (df.loc[i, 'Year in Office'] in [5,9]) and (df.loc[i+1,'Year in Office'] == 1):

        df.iloc[i,3:] = df.iloc[i,3:]+df.iloc[i+1,3:]

        to_drop.append(i+1)



df.drop(df.index[to_drop], axis=0, inplace=True)



# We will still have some duplicate years, but only when presidents died in office. 

# In those cases, they served for a reasonable amount of time during that year, so we include it in our analysis
# Redo 'Year in Office' column

df.reset_index(inplace=True, drop=True)

df['Year in Office']=pd.Series([1]*len(df))

df.loc[0,'Year in Office']=3 #data starts at McKinley's third year in office

for i in range(1,len(df)):

    if df.loc[i-1, 'President'] == df.loc[i,'President']:

        df.loc[i,'Year in Office'] = df.loc[i-1,'Year in Office']+1
#Add a column for information on what party each president belonged to

# Since the dataset is small, I took 30 seconds to enter this information by hand.

d={1:'R',0:'D'}

party=list(map(lambda x: d[x],[1,1,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,0,1,0]))

partymap=dict(zip(df.President.unique(),party))

df['Party']=df['President'].map(partymap)
# Visualize number of pardons each year by party

colors = sns.color_palette("Set1", len(df['President'].unique()))

im = sns.lmplot(data=df, x='Fiscal Year', y="Petitions Granted", hue='Party', fit_reg=False, palette=colors)

im.set(xlim=(1890,2020), title="Petitions Granted Issued by Year")



#At a glance, it seems democrats grant more petitions, but if you exclude the obvious outliers they are roughly equivalent
df['Knew Outgoing']=pd.Series([1]*len(df))

for pres in ['William McKinley', 'Warren Harding', 'Franklin D. Roosevelt', 'John F. Kennedy']:

    df.loc[df['President']==pres,'Knew Outgoing']=0

#These four presidents died in office.  Nixon also left office early, but he knew he was about to do so.


presidents=list(df['President'].unique())

terms=list(map(lambda pres: int(np.ceil((df['President']==pres).sum()/4)), presidents))

termdict=dict(zip(presidents, terms))

termdict['William McKinley']=1 #The abouve counting method would miscount McKinley as not having served

df['Terms']=df['President'].map(termdict)

df.head()
OneTermdf=df.loc[(df['Terms']==1) & (df['Knew Outgoing']==1)]

TwoTermdf=df.loc[(df['Terms']==2) & (df['Knew Outgoing']==1)]
# Visualize number of pardons each year for presidents serving two full terms

colors = sns.color_palette("Set1", len(TwoTermdf['President'].unique()))

im = sns.lmplot(data=TwoTermdf, x='Fiscal Year', y="Petitions Granted", hue='President', fit_reg=False, palette=colors)

im.set(xlim=(1890,2020), title="Petitions Granted by Year")
colors = sns.color_palette("Set1", len(TwoTermdf['President'].unique()))

im = sns.lmplot(data=TwoTermdf, x='Fiscal Year', y="Petitions Granted", hue='Party', fit_reg=False, palette=colors)

im.set(xlim=(1890,2020), title="Petitions Granted by Year")
colors = sns.color_palette("Set1", len(TwoTermdf['President'].unique()))

im = sns.lmplot(data=TwoTermdf, x='Fiscal Year', y="Percent Granted", hue='President', fit_reg=False, palette=colors)

im.set(xlim=(1890,2020), title="Percent of Petitions Granted by Year")
colors = sns.color_palette("Set1", len(TwoTermdf['President'].unique()))

im = sns.lmplot(data=TwoTermdf, x='Fiscal Year', y="Percent Denied", hue='President', fit_reg=False, palette=colors)

im.set(xlim=(1890,2020), title="Percent of Petitions Denied by Year")
# Visualize number of pardons each year for presidents serving one full terms

colors = sns.color_palette("Set1", len(OneTermdf['President'].unique()))

im = sns.lmplot(data=OneTermdf, x='Fiscal Year', y="Petitions Granted", hue='President', fit_reg=False, palette=colors)

im.set(xlim=(1890,2020), title="Petitions Granted by Year")
colors = sns.color_palette("Set1", len(OneTermdf['President'].unique()))

im = sns.lmplot(data=OneTermdf, x='Fiscal Year', y="Petitions Granted", hue='Party', fit_reg=False, palette=colors)

im.set(xlim=(1890,2020), title="Petitions Granted by Year")
byyear=TwoTermdf.groupby('Year in Office').mean()

byyear
byyear.plot(y=['Petitions Received','Petitions Granted','Petitions Denied','Petitions Closed Without Presidential Action'])
byyear.plot(y=['Percent Granted','Percent Denied'])
im = sns.lmplot(data=TwoTermdf, x='Year in Office', y="Petitions Granted", hue='Party', fit_reg=True, palette=colors)

im.set(xlim=(0,10), title="Petitions Granted by Year in office")
TwoTermDemdf=TwoTermdf.loc[TwoTermdf['Party']=='D']
dembyyear=TwoTermdf.groupby('Year in Office').mean()

dembyyear
byyear.plot(y=['Petitions Received','Petitions Granted','Petitions Denied','Petitions Closed Without Presidential Action'])
byyear.plot(y=['Percent Granted','Percent Denied'])
temp=TwoTermdf[['Year in Office','Party','Percent Granted']].dropna()

temp['Democrat']=temp['Party'].map({'R':0,'D':1})

temp.drop('Party',axis=1,inplace=True)

X=temp[['Year in Office','Democrat']]

X=sm.add_constant(X)

Y=temp['Percent Granted']
model=sm.OLS(Y,X)

result=model.fit()

result.summary()
#Choose a regressor and construct the set of explanatory variables

regressor='Percent Granted'



regressiondf=df[['Fiscal Year','Petitions Pending','Petitions Received','Year in Office', 'Party','Terms', regressor]].copy()

regressiondf['Democrat']=regressiondf['Party'].map({'R':0,'D':1})

regressiondf.drop('Party',axis=1,inplace=True)

regressiondf.dropna(inplace=True)



Y=regressiondf[regressor]

regressiondf.drop(regressor,axis=1,inplace=True)
#Normalize explanatory variables

X=(regressiondf-regressiondf.mean())/regressiondf.std()
model=sm.OLS(Y,sm.add_constant(X), missing='drop')

results=model.fit()

results.summary()
regressiondf.corr()
sns.heatmap(regressiondf.corr())
#Choose a regressor and construct the set of explanatory variables

regressor='Percent Granted'



regressiondf=df[['Fiscal Year','Petitions Pending','Year in Office', 'Party','Terms', regressor]].copy()

regressiondf['Democrat']=regressiondf['Party'].map({'R':0,'D':1})

regressiondf.drop('Party',axis=1,inplace=True)

regressiondf.dropna(inplace=True)



Y=regressiondf[regressor]

regressiondf.drop(regressor,axis=1,inplace=True)
#Normalize explanatory variables

X=(regressiondf-regressiondf.mean())/regressiondf.std()
model=sm.OLS(Y,sm.add_constant(X), missing='drop')

results=model.fit()

results.summary()
#Choose a regressor and construct the set of explanatory variables

regressor='Percent Granted'



regressiondf=df[['Fiscal Year','Petitions Received','Year in Office', 'Party', regressor]].copy()

regressiondf['Democrat']=regressiondf['Party'].map({'R':0,'D':1})

regressiondf.drop('Party',axis=1,inplace=True)

regressiondf.dropna(inplace=True)



Y=regressiondf[regressor]

regressiondf.drop(regressor,axis=1,inplace=True)
#Normalize explanatory variables

X=(regressiondf-regressiondf.mean())/regressiondf.std()
model=sm.OLS(Y,sm.add_constant(X), missing='drop')

results=model.fit()

results.summary()
number=len(regressiondf.columns)

fig, axarray = plt.subplots(number, figsize=(6,40),sharey=True)

for i in range(number):

    axarray[i].scatter(x=regressiondf[regressiondf.columns[i]], y=results.resid)

    axarray[i].set_title('Residual vs '+ regressiondf.columns[i])
plt.scatter(y=results.resid,x=regressiondf['Fiscal Year'])
#Choose a regressor and construct the set of explanatory variables

regressor='Percent Granted'



regressiondf=df[['Fiscal Year','Petitions Received','Year in Office', 'Party', regressor]].copy()

regressiondf['Democrat']=regressiondf['Party'].map({'R':0,'D':1})

regressiondf.drop('Party',axis=1,inplace=True)

regressiondf.dropna(inplace=True)



Y=regressiondf[regressor]

regressiondf.drop(regressor,axis=1,inplace=True)
#add new feature

regressiondf['Year Squared']=(regressiondf['Fiscal Year']-regressiondf['Fiscal Year'].mean())**2

#Normalize explanatory variables

X=(regressiondf-regressiondf.mean())/regressiondf.std()
model=sm.OLS(Y,sm.add_constant(X), missing='drop')

results=model.fit()

results.summary()
sns.heatmap(regressiondf.corr())
#Choose a regressor and construct the set of explanatory variables

regressor='Percent Granted'



regressiondf=df[['Fiscal Year','Year in Office', regressor]].copy()

regressiondf.dropna(inplace=True)



Y=regressiondf[regressor]

regressiondf.drop(regressor,axis=1,inplace=True)
#add new feature

regressiondf['Year Squared']=(regressiondf['Fiscal Year']-regressiondf['Fiscal Year'].mean())**2

#Normalize explanatory variables

X=(regressiondf-regressiondf.mean())/regressiondf.std()
model=sm.OLS(Y,sm.add_constant(X), missing='drop')

results=model.fit()

results.summary()
number=len(regressiondf.columns)

fig, axarray = plt.subplots(number, figsize=(6,20),sharey=True)

for i in range(number):

    axarray[i].scatter(x=regressiondf[regressiondf.columns[i]], y=results.resid)

    axarray[i].set_title('Residual vs '+ regressiondf.columns[i])
#Choose a regressor and construct the set of explanatory variables

regressor='Petitions Granted'



regressiondf=df[['Fiscal Year','Petitions Pending','Year in Office', 'Party','Terms', regressor]].copy()

regressiondf['Democrat']=regressiondf['Party'].map({'R':0,'D':1})

regressiondf.drop('Party',axis=1,inplace=True)

regressiondf.dropna(inplace=True)



Y=regressiondf[regressor]

regressiondf.drop(regressor,axis=1,inplace=True)



#add new feature

regressiondf['Year Squared']=(regressiondf['Fiscal Year']-regressiondf['Fiscal Year'].mean())**2

#Normalize explanatory variables

X=(regressiondf-regressiondf.mean())/regressiondf.std()



model=sm.OLS(Y,sm.add_constant(X), missing='drop')

results=model.fit()

results.summary()
number=len(regressiondf.columns)

fig, axarray = plt.subplots(number, figsize=(6,40),sharey=True)

for i in range(number):

    axarray[i].scatter(x=regressiondf[regressiondf.columns[i]], y=results.resid)

    axarray[i].set_title('Residual vs '+ regressiondf.columns[i])
#Choose a regressor and construct the set of explanatory variables

regressor='Petitions Denied'



regressiondf=df[['Fiscal Year','Petitions Pending','Year in Office', 'Party', 'Terms', regressor]].copy()

regressiondf['Democrat']=regressiondf['Party'].map({'R':0,'D':1})

regressiondf.drop('Party',axis=1,inplace=True)

regressiondf.dropna(inplace=True)



Y=regressiondf[regressor]

regressiondf.drop(regressor,axis=1,inplace=True)



#add new feature

regressiondf['Year Squared']=(regressiondf['Fiscal Year']-regressiondf['Fiscal Year'].mean())**2

#Normalize explanatory variables

X=(regressiondf-regressiondf.mean())/regressiondf.std()



model=sm.OLS(Y,sm.add_constant(X), missing='drop')

results=model.fit()

results.summary()
number=len(regressiondf.columns)

fig, axarray = plt.subplots(number, figsize=(6,40),sharey=True)

for i in range(number):

    axarray[i].scatter(x=regressiondf[regressiondf.columns[i]], y=results.resid)

    axarray[i].set_title('Residual vs '+ regressiondf.columns[i])