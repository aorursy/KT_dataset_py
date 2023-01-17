import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
df.loc[:, 'age'] = df['age'].str.replace(' years','')

df=df[df['year']<2016]

df.rename(columns={'gdp_per_capita ($)':'gdp'}, inplace=True)

df.rename(columns={"suicides/100k pop": "Suicides_percapita"})

df.isna().sum()
df.describe()
df["generation"].value_counts().sort_values(ascending=False).plot(kind='bar')

plt.xticks(rotation=30)
plt.figure(figsize=(30,10))

sns.relplot(x="year", y="suicides_no",col='generation', data=df,palette="tab10",linewidth=2,kind='line')
sns.relplot(x="year", y="suicides_no", hue="generation",

            col="age", height=3,

            kind="line", data=df);
plt.figure(figsize=(20,7))

sns.lineplot(x="year", y="suicides_no",hue='sex', data=df)

plt.title('Suicides by Sex over time')
from numpy import median

sns.catplot(x="sex", y="suicides_no",col='age', data=df, estimator=median,palette="tab10",height=4, aspect=.7,kind='bar')
# suicides per capita

df.groupby(by=['country'])['suicides/100k pop'].mean().reset_index().sort_values(['suicides/100k pop'],ascending=True).tail(20).plot(x='country',y='suicides/100k pop',kind='barh')
se_X=df["gdp"]

se_Y=df["suicides/100k pop"]

se_n=pd.concat([se_X,se_Y],axis=1)

se_nx=se_X.values.reshape(-1,1)

se_ny=se_Y.values.reshape(-1,1)



from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(se_nx,se_ny)

print("Y intercept:", lin_reg.intercept_)

print("Slope:", lin_reg.coef_)

plt.scatter(se_nx, se_ny, color='black', marker='o', alpha=.5)

sns.regplot(x='gdp', y='suicides/100k pop', data=se_n, scatter=None, color="r")

plt.xlabel('GDP per capita ($)')

plt.ylabel('Suicides / 100K')

plt.legend(loc="best")

plt.show()