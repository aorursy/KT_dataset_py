import numpy as np

import pandas as pd 

import os

import seaborn as sns

import matplotlib.pyplot as plt
df= pd.read_csv('../input/beauty.csv', delimiter=',')
df.head()
#checking for na

df.info()
df.describe()
plt.figure(figsize=(10,6))

sns.heatmap(df.corr(), annot=True,cmap="YlGnBu")
sns.pairplot(df[['wage', 'female', 'educ',

       'exper']]);
plt.figure(1 , figsize = (15 , 6))

for gender in [1 , 0]:

    plt.scatter(x = 'wage' , y = 'exper' , data = df[df['female'] == gender] ,

                s = 200 , alpha = 0.5 , label = gender)

plt.xlabel('Wage in thousands/month'), 

plt.ylabel('Years of Expertise') 

plt.title('Level of Expertise compared to Wage with the distinction gender')

plt.legend(['Female','Male'])

plt.show()
w_dev_wage = df[df['female']==1]['wage'].groupby(df['exper']).mean()
m_dev_wage = df[df['female']==0]['wage'].groupby(df['exper']).mean()
plt.figure(1 , figsize = (15 , 6))

sns.lineplot(data= w_dev_wage)

sns.lineplot(data= m_dev_wage)

plt.xlabel('Years of Expertise'), 

plt.ylabel('Mean Wage in thousand/Month') 

plt.title('Years of Expertise compared to Wage with the distinction gender')

plt.legend(['Female','Male'])

plt.show()
g = sns.PairGrid(df[['female', 'wage', 'educ',

       'exper']])

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=8);
data = df[["wage","educ"]].groupby(["educ"],as_index=False).mean().sort_values(by="educ", ascending=False)
sns.barplot(x="educ",y="wage",data=data)
df.married.value_counts()/df.married.size
married_ppl = df[df.married ==1]

married_ppl.female.value_counts()/ married_ppl.female.size
df[["female","service"]].groupby("female").mean()
sns.boxplot(x="female",y="wage",data=df)
sns.boxplot(x="female",y="wage",data=df, showfliers=False)
sns.regplot(x="educ", y="exper", data=df)
df["looks"].value_counts().sort_values(ascending=False)
df["looks"].value_counts().sort_values(ascending=False).plot.bar()
sns.jointplot(x="looks",y="wage", data = df, kind="hex")
sns.regplot(x="looks",y="wage", data = df)