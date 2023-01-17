import pandas as pd

df = pd.read_csv('../input/HR_comma_sep.csv')

df.info() 
df.head()
sales = df.groupby(by='sales').count()

sales
df.rename(columns={'sales': 'department'}, inplace=True)

df.head()
def salary(row):

    if row['salary'] == 'high':

        return 3

    elif row['salary'] == 'medium':

        return 2

    else:

        return 1

    

df['Salary2'] = df.apply(salary, axis=1)



df.describe()
import seaborn as sns

df_grouped = df.groupby(by=['department'],as_index=False).count()

ax = sns.barplot(x="department", y="left", data=df_grouped)
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style="white")



corr = df.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



vf, ax = plt.subplots(figsize=(11, 9))



sns.heatmap(corr, mask=mask, vmax=.3, center=0,annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax = sns.distplot(df['satisfaction_level'])
def sat(row):

    if .3 <= row['satisfaction_level'] < .5:

        return 'Medium'

    elif row['satisfaction_level'] >= .5:

        return 'High'

    else:

        return 'Low'



df['Sat'] = df.apply(sat, axis=1)



df_gr = df.groupby(by='Sat',as_index=False).mean()

df_gr.head()
df_grouped = df.groupby(by=['Sat'])

left_rate = df_grouped['left'].sum() / df_grouped['left'].count()

ax = left_rate.plot(kind='barh')
df_grouped = df.groupby(by=['salary'],as_index=False).count()

df_grouped
ax = sns.barplot(x="salary", y="left", data=df_grouped)
df_grouped = df.groupby(by=['salary'])

left_rate = df_grouped['left'].sum() / df_grouped['left'].count()

ax = left_rate.plot(kind='barh')
# Set a default value

df_grouped = df.groupby(by=['salary','Sat'])

left_rate = df_grouped['left'].sum() / df_grouped['left'].count()

ax = left_rate.plot(kind='barh')