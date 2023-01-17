import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)



%matplotlib inline
df = pd.read_csv("../input/cs-training.csv")
df.head(10)
df.info()
df.rename(columns = {df.columns[0]:'ID'}, inplace = True) 



df.describe()
df.drop(df.columns[[4, 8, 10]], axis=1, inplace=True)

df.head()
P = df.groupby('SeriousDlqin2yrs')['ID'].count().reset_index()



P['Percentage'] = 100 * P['ID']  / P['ID'].sum()



print(P)
df['SeriousDlqin2yrs'].value_counts(normalize=True).plot(kind='barh')
df['RevolvingUtilizationOfUnsecuredLines'].describe()
df3=df.loc[df['RevolvingUtilizationOfUnsecuredLines'] <=1]

sns.distplot(df3['RevolvingUtilizationOfUnsecuredLines'])
len(df[(df['RevolvingUtilizationOfUnsecuredLines']>1)])
df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].map(lambda x: np.NaN if x >1 else x)
df['RevolvingUtilizationOfUnsecuredLines'].describe()
df['RevolvingUtilizationOfUnsecuredLines'].fillna(method='ffill', inplace=True)
df['RevolvingUtilizationOfUnsecuredLines'].describe()
df['age'].describe()
sns.distplot(df['age'])
df.loc[df['age']>80, 'age']=80

df.loc[df['age']<18, 'age']=18
sns.distplot(df['age'])
df['age'].describe()
df['DebtRatio'].describe()
df2=df[df['DebtRatio']<=1]

sns.distplot(df2['DebtRatio'])
df2=df[df['DebtRatio']>1]

df2['DebtRatio'].describe()
df.loc[df['DebtRatio']>1, 'DebtRatio']=np.NaN
df['DebtRatio'].describe()
df['DebtRatio'].fillna(method='ffill', inplace=True)
df['DebtRatio'].describe()
sns.distplot(df['DebtRatio'])
df['NumberOfOpenCreditLinesAndLoans'].describe()
sns.distplot(df['NumberOfOpenCreditLinesAndLoans'])
df.loc[df['NumberOfOpenCreditLinesAndLoans']>30, 'NumberOfOpenCreditLinesAndLoans']=30
df['NumberOfOpenCreditLinesAndLoans'].describe()
sns.distplot(df['NumberOfOpenCreditLinesAndLoans'])
df['MonthlyIncome'].describe()
df['MonthlyIncome'].isnull().sum()
len(df[df['MonthlyIncome']<1000])
sns.distplot(df['MonthlyIncome'].dropna())
df2=df[df['MonthlyIncome']<50000]

sns.distplot(df2['MonthlyIncome'].dropna())
df.loc[df['MonthlyIncome']>25000, 'MonthlyIncome']=25000

df['MonthlyIncome'].describe()
df.loc[df['MonthlyIncome']<1000, 'MonthlyIncome']=np.NaN

df['MonthlyIncome'].describe()
df['MonthlyIncome'].fillna(method='ffill', inplace=True)

df['MonthlyIncome'].describe()
sns.distplot(df['MonthlyIncome'])
df['NumberRealEstateLoansOrLines'].describe()
sns.distplot(df['NumberRealEstateLoansOrLines'])
df2=df[df['NumberRealEstateLoansOrLines']<6]

sns.distplot(df2['NumberRealEstateLoansOrLines'].dropna())
df.loc[df['NumberRealEstateLoansOrLines']>5, 'NumberRealEstateLoansOrLines']=5

df['NumberRealEstateLoansOrLines'].describe()
sns.distplot(df['NumberRealEstateLoansOrLines'])
df['NumberOfDependents'].describe()
sns.distplot(df['NumberOfDependents'].dropna())
df.loc[df['NumberOfDependents']>5, 'NumberOfDependents']=5

df['NumberOfDependents'].describe()
#df['NumberOfDependents'].fillna(df['NumberOfDependents'].mean(), inplace=True)

df['NumberOfDependents'].fillna(method='ffill', inplace=True)

df['NumberOfDependents'].describe()
df.describe()
df.to_pickle("gmsc_clean.pkl")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from scipy.stats import chisquare

from scipy.stats import chi2_contingency

sns.set(color_codes=True)



%matplotlib inline
df = pd.read_pickle('gmsc_clean.pkl')

df.head()
df.describe()
df.info()
sns.countplot(x='SeriousDlqin2yrs', data=df)
f,ax=plt.subplots(1,2,figsize=(14,6))

df['SeriousDlqin2yrs'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=False)

ax[0].set_title('SeriousDlqin2yrs')

ax[0].set_ylabel('')

sns.countplot('SeriousDlqin2yrs',data=df,ax=ax[1])

ax[1].set_title('SeriousDlqin2yrs')

plt.show()
for column in df.columns[2:]:

    print(column)

    #s=df['column']

    s=df[column]

    mu, sigma =norm.fit(s)

    count, bins, ignored = plt.hist(s, 30, normed=True, color='g')

    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=1, color='r')



    title = "Plot used: mu = %.2f,  std = %.2f" % (mu, sigma)

    plt.title(title, loc='right')



    plt.show()
df.groupby('SeriousDlqin2yrs')['RevolvingUtilizationOfUnsecuredLines'].agg(['count','mean'])
df['RevolvingUtilizationOfUnsecuredLines'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 
df['RevolvingUtilizationOfUnsecuredLines'].describe()
def cat_ruul(ruul):

    if ruul <0.03:

        return 1

    elif 0.03<= ruul <0.14:

        return 2

    elif 0.14<= ruul <0.52:

        return 3

    else:

        return 4
df['ruul_cat'] = df['RevolvingUtilizationOfUnsecuredLines'].apply(cat_ruul)

df.head(3)
# lets check if the categorization was done correctly

df.groupby('ruul_cat')['RevolvingUtilizationOfUnsecuredLines'].agg(['min','max'])
pd.crosstab(df.SeriousDlqin2yrs, df.ruul_cat, normalize='columns')
sb=pd.crosstab(df.ruul_cat, df.SeriousDlqin2yrs, normalize=0)

sb.plot.bar(stacked=True)
df2=pd.crosstab(df.SeriousDlqin2yrs, df.ruul_cat)

chi2_contingency(df2)
df.groupby('SeriousDlqin2yrs')['age'].agg(['count','mean'])
df['age'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 
df['age'].describe()
def cat_ruul(ruul):

    if ruul <41:

        return 1

    elif 41<= ruul <52:

        return 2

    elif 52<= ruul <63:

        return 3

    else:

        return 4
df['age_cat'] = df['age'].apply(cat_ruul)

df.head(3)
# lets check if the categorization was done correctly

df.groupby('age_cat')['age'].agg(['min','max'])
pd.crosstab(df.SeriousDlqin2yrs, df.age_cat, normalize='columns')
sb=pd.crosstab(df.age_cat, df.SeriousDlqin2yrs, normalize=0)

sb.plot.bar(stacked=True)
df2=pd.crosstab(df.SeriousDlqin2yrs, df.age_cat)

chi2_contingency(df2)
df.groupby('SeriousDlqin2yrs')['DebtRatio'].agg(['count','mean'])
df['DebtRatio'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 
df['DebtRatio'].describe()
def cat_ruul(ruul):

    if ruul <0.13:

        return 1

    elif 0.13<= ruul <0.27:

        return 2

    elif 0.27<= ruul <0.43:

        return 3

    else:

        return 4
df['DebtRatio_cat'] = df['DebtRatio'].apply(cat_ruul)

df.head(3)
# lets check if the categorization was done correctly

df.groupby('DebtRatio_cat')['DebtRatio'].agg(['min','max'])
pd.crosstab(df.SeriousDlqin2yrs, df.DebtRatio_cat, normalize='columns')
sb=pd.crosstab(df.DebtRatio_cat, df.SeriousDlqin2yrs, normalize=0)

sb.plot.bar(stacked=True)
df2=pd.crosstab(df.SeriousDlqin2yrs, df.DebtRatio_cat)

chi2_contingency(df2)
df.groupby('SeriousDlqin2yrs')['MonthlyIncome'].agg(['count','mean'])
df['MonthlyIncome'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 
df['MonthlyIncome'].describe()
def cat_ruul(ruul):

    if ruul <3600:

        return 1

    elif 3600<= ruul <5500:

        return 2

    elif 5500<= ruul <8333:

        return 3

    else:

        return 4
df['MonthlyIncome_cat'] = df['MonthlyIncome'].apply(cat_ruul)

df.head(3)
# lets check if the categorization was done correctly

df.groupby('MonthlyIncome_cat')['MonthlyIncome'].agg(['min','max'])
pd.crosstab(df.SeriousDlqin2yrs, df.MonthlyIncome_cat, normalize='columns')
sb=pd.crosstab(df.MonthlyIncome_cat, df.SeriousDlqin2yrs, normalize=0)

sb.plot.bar(stacked=True)
df2=pd.crosstab(df.SeriousDlqin2yrs, df.MonthlyIncome_cat)

chi2_contingency(df2)
df.groupby('SeriousDlqin2yrs')['NumberOfOpenCreditLinesAndLoans'].agg(['count','mean'])
df['NumberOfOpenCreditLinesAndLoans'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 
df['NumberOfOpenCreditLinesAndLoans'].describe()
def cat_ruul(ruul):

    if ruul <5:

        return 1

    elif 5<= ruul <8:

        return 2

    elif 8<= ruul <11:

        return 3

    else:

        return 4
df['NOCLL_Cat'] = df['NumberOfOpenCreditLinesAndLoans'].apply(cat_ruul)

df.head(3)
# lets check if the categorization was done correctly

df.groupby('NOCLL_Cat')['NumberOfOpenCreditLinesAndLoans'].agg(['min','max'])
pd.crosstab(df.SeriousDlqin2yrs, df.NOCLL_Cat, normalize='columns')
sb=pd.crosstab(df.NOCLL_Cat, df.SeriousDlqin2yrs, normalize=0)

sb.plot.bar(stacked=True)
df2=pd.crosstab(df.SeriousDlqin2yrs, df.NOCLL_Cat)

chi2_contingency(df2)
df.groupby('SeriousDlqin2yrs')['NumberRealEstateLoansOrLines'].agg(['count','mean'])
df['NumberRealEstateLoansOrLines'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 
df['NumberRealEstateLoansOrLines'].describe()
def cat_ruul(ruul):

    if ruul <=0:

        return 1

    elif 0< ruul <=1:

        return 2

    elif 1< ruul <=2:

        return 3

    else:

        return 4
df['NRELL_Cat'] = df['NumberRealEstateLoansOrLines'].apply(cat_ruul)

df.head(3)
# lets check if the categorization was done correctly

df.groupby('NRELL_Cat')['NumberRealEstateLoansOrLines'].agg(['min','max'])
pd.crosstab(df.SeriousDlqin2yrs, df.NRELL_Cat, normalize='columns')
sb=pd.crosstab(df.NRELL_Cat, df.SeriousDlqin2yrs, normalize=0)

sb.plot.bar(stacked=True)
df2=pd.crosstab(df.SeriousDlqin2yrs, df.NRELL_Cat)

chi2_contingency(df2)
df.groupby('SeriousDlqin2yrs')['NumberOfDependents'].agg(['count','mean'])
df['NumberOfDependents'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 
df['NumberOfDependents'].describe()
def cat_ruul(ruul):

    if ruul <=0:

        return 1

    elif 0< ruul <=1:

        return 2

    elif 1< ruul <=2:

        return 3

    else:

        return 4
df['NOD_Cat'] = df['NumberOfDependents'].apply(cat_ruul)

df.head(3)
# lets check if the categorization was done correctly

df.groupby('NOD_Cat')['NumberOfDependents'].agg(['min','max'])
pd.crosstab(df.SeriousDlqin2yrs, df.NOD_Cat, normalize='columns')
sb=pd.crosstab(df.NOD_Cat, df.SeriousDlqin2yrs, normalize=0)

sb.plot.bar(stacked=True)
df2=pd.crosstab(df.SeriousDlqin2yrs, df.NOD_Cat)

chi2_contingency(df2)