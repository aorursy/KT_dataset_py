import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/FEHSurvey8Stars.csv')

df.drop('Timestamp', axis=1, inplace=True)

df.columns
df.columns = ['begin', 'spend', 'available', 'merged', 'inherited', 'home', 'catalog', 'barracks']

df.head()
df.shape
df.isnull().sum()
def show_count_freq(df, column):

    counts = pd.DataFrame(df[column].value_counts())

    freqs = df[column].value_counts(normalize=True)

    count_freq = pd.concat([counts, freqs.rename('proportion')], axis=1)

    count_freq.columns = ['Total', 'Proportion']

    return count_freq

    
show_count_freq(df, 'begin')
df['begin'].value_counts(normalize=True).plot(kind='bar', title='Month in which Respondents Started Playing FEH (2017)')

plt.ylabel('Proportion of Respondents')
show_count_freq(df, 'spend')
spend = df['spend'][df['spend'] != 'I don\'t know / don\'t remember']

order = ['None (F2P)', '$1 - $50', '$51 - $100', '$101 - $200', '$201 - $400', '$401 - $600', '$601 - $800', '$801 - $1000', '$1001 - $2000', '$2001 - $3000', '$3001 - $4000', '$4001 - $5000', '$5000+']

spend.value_counts(normalize=True).loc[order].plot(kind='bar')

plt.title('Amount of Money spent in FEH')

plt.ylabel('Proportion of Respondents')

plt.xlabel('Amount Spent (USD $)')





spend_dict = {

    'None (F2P)' : 'None (F2P)',

    '$1 - $50': '1-200',

    '$51 - $100': '1-200', 

    '$101 - $200': '1-200',

    '$201 - $400': '201-800', 

    '$401 - $600': '201-800', 

    '$601 - $800': '201-800',

    '$801 - $1000': '801+', 

    '$1001 - $2000': '801+', 

    '$2001 - $3000': '801+', 

    '$3001 - $4000': '801+', 

    '$4001 - $5000': '801+', 

    '$5000+': '801+',

    "I don't know / don't remember" : np.NaN,

    np.NaN : np.NaN

}



   

df['spendgroup'] = df['spend'].map(lambda x: spend_dict[x])



show_count_freq(df, 'spendgroup')

df['spendgroup'].value_counts(normalize=True).plot(kind='bar')

plt.title('Amount of Money spent in FEH')

plt.ylabel('Proportion of Respondents')

plt.xlabel('Amount Spent (USD $)')
df['available'].describe()
def ecdf(df, column):

    x = np.sort(df[column])

    y = np.arange(1, len(x)+1)/len(x)

    return x, y



pd.DataFrame(df['available'].describe()).T
df['available'].hist(bins = 30)

plt.title('Number of Available 5 Star Heroes')

plt.ylabel('Number of Respondents')





plt.figure()

availx, availy =  ecdf(df, 'available')

plt.plot (availx, availy, marker= '.', linestyle='none')

plt.xlabel('Available 5 Star Heroes')

plt.ylabel('Cummulative Proportion')

plt.title('Empirical Distribution Function of Available 5 Star Heroes')
df['available'].groupby(df['spendgroup']).describe().sort_values(by='mean')
sns.boxplot(x='spendgroup', y='available', order = ['None (F2P)', '1-200', '201-800','801+'], data=df)

plt.title('Available 5 Stars by Amount Spent')

plt.xlabel('Amount Spent (USD $)')

plt.ylabel('Available 5 stars')
df['available'].groupby(df['begin']).describe().loc[['February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November']]
sns.boxplot(x='begin', y='available', order=['February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November'], data=df)

plt.title('Available 5 Stars by Month Respondent Started FEH')

plt.xlabel('Amount Spent (USD $)')

plt.xticks(rotation=90)

plt.ylabel('Month Started FEH')
df[(df['begin']=='November') & (df['available']>100)]
pd.DataFrame(df['merged'].describe()).T
df[df['merged']>=200]
mergedx, mergedy =  ecdf(df, 'merged')

plt.plot (mergedx, mergedy, marker= '.', linestyle='none')

plt.xlabel('Merged 5 Star Heroes')

plt.ylabel('Cummulative Proportion')

plt.title('Empirical Distribution Function of Merged 5 Star Heroes')



plt.figure()

plt.plot (mergedx, mergedy, marker= '.', linestyle='none')

plt.xlabel('Merged 5 Star Heroes')

plt.ylabel('Cummulative Proportion')

plt.title('Empirical Distribution Function of Merged 5 Star Heroes (Zoomed)')

plt.axis([0,40, 0, 1.0])
df['merged'].groupby(df['spendgroup']).describe()
sns.boxplot(x='spendgroup', y='merged', order = ['None (F2P)', '1-200', '201-800','801+'], data=df, showfliers=False)

plt.title('Merged 5 Stars by Amount Spent')

plt.xlabel('Amount Spent (USD $)')

plt.ylabel('Merged 5 stars')
pd.DataFrame(df['inherited'].describe()).T
df[df['inherited']>=100]
df.loc[1929, 'inherited']  = np.NaN
pd.DataFrame(df['inherited'].describe()).T
df['inherited'].hist(range=(0,50), bins=20)
inheritedx, inheritedy =  ecdf(df, 'inherited')

plt.plot (inheritedx, inheritedy, marker= '.', linestyle='none')

plt.xlabel('Inherited 5 Star Heroes')

plt.ylabel('Cummulative Proportion of Respondents')

plt.title('Empirical Distribution Function of Inherited 5 Star Heroes')



plt.figure()

plt.plot (inheritedx, inheritedy, marker= '.', linestyle='none')

plt.xlabel('Inherited 5 Star Heroes')

plt.ylabel('Cummulative Proportion of Respondents')

plt.title('Empirical Distribution Function of Inherited 5 Star Heroes (Zoomed)')

plt.axis([0,30,0,1.0])
df['inherited'].groupby(df['spendgroup']).describe()
sns.boxplot(x='spendgroup', y='inherited', order = ['None (F2P)', '1-200', '201-800','801+'], data=df, showfliers=False)

plt.title('Inherited 5 Stars by Amount Spent')

plt.xlabel('Amount Spent (USD $)')

plt.ylabel('Inherited 5 stars')
df['home'].unique()
salvage_data = {

    '~6' : 6,

    '1 (Lachesis)': 1,

    'o': 0,

    'O': 0,

    '0, I am not crazy': 0,

    'Zero': 0,

    '1 it was Lucius lol sorry buddy': 1,

    '1 (Fcking Marth he sucks so badly)': 1,

    '0 why the fuck would you do this' : 0,

    '2(peri)': 2,

    'zero': 0,

    'Clive': 1,

    'None': 0

}



df['home'] = df['home'].replace(salvage_data)
df['homenum'] = pd.to_numeric(df['home'], errors='coerce')

pd.DataFrame(df['homenum'].describe()).T
(df[df['homenum']>=15]).sort_values(by='homenum')
mask = df.homenum >= 100

column_name = 'homenum'

df.loc[mask, 'homenum'] = np.NaN
pd.DataFrame(df['homenum'].describe()).T
homex, homey =  ecdf(df, 'homenum')

plt.plot (homex, homey, marker= '.', linestyle='none')

plt.xlabel('5 Star Heroes Sent Home')

plt.ylabel('Cummulative Proportion of Respondents')

plt.title('ECDF of No. of 5 Star Heroes Sent Home')



plt.figure()

df['homenum'].hist(normed=True, range=(0,10), bins=10)
df['homenum'].groupby(df['spendgroup']).describe()
df['barracks'] = df['barracks'].apply(pd.to_numeric, errors='coerce')

pd.DataFrame(df['barracks'].describe()).T
df[df['barracks'] == 1000]
print ((df['barracks'] == 200).sum(), ((df['barracks'] == 200).sum())/(df['barracks'].notnull().sum()))

# Number of respondents with barrack size 200 + proportion
df['barracks'].hist(bins=20)

plt.title('Barrack Size')

plt.ylabel('Number of Respondents')

plt.xlabel('Size')
barracksx, barracksy =  ecdf(df, 'barracks')

plt.plot (barracksx, barracksy, marker= '.', linestyle='none')

plt.xlabel('Barrack Size')

plt.ylabel('Cummulative Proportion of Respondents')

plt.title('Empirical Distribution Function of Barrack Size')
df.groupby('spendgroup')['barracks'].describe()
sns.boxplot(x='spendgroup', y='barracks', order = ['None (F2P)', '1-200', '201-800','801+'], data=df)

plt.title('Barrack Size by Amount Spent')

plt.xlabel('Amount Spent (USD $)')

plt.ylabel('Barrack Size')
pd.DataFrame(df['catalog'].describe()).T
len(df[df['catalog']>190])
df['catalog'].hist(bins=15, range=(0,190), normed=1)

plt.title('Units Unlocked in Catalog')

plt.ylabel('Number of Respondents')

plt.xlabel('Number of Units Unlocked')
catx, caty =  ecdf(df, 'catalog')

plt.plot (catx, caty , marker= '.', linestyle='none')

plt.xlabel('Units Unlocked in Catalog')

plt.ylabel('Cummulative Proportion of Respondents')

plt.title('Empirical Distribution Function of Heroes Unlcoked in Catalog')

plt.axis([0,190,0,1.0])
df.groupby('spendgroup')['catalog'].describe()
sns.boxplot(x='spendgroup', y='catalog', order = ['None (F2P)', '1-200', '201-800','801+'], data=df, showfliers=False)

plt.title('Catalog Unlocked by Amount Spent')

plt.xlabel('Amount Spent (USD $)')

plt.ylabel('Catalog Unlocked')