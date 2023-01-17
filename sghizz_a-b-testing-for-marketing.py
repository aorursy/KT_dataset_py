# This Python 3 environment comes with many helpful analytics libraries installed

import os

print(os.listdir("../input"))
%matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv("../input/WA_Fn-UseC_-Marketing-Campaign-Eff-UseC_-FastF.csv")
df.shape
df.head(15)
ax = df.groupby(

   'Promotion'

).sum()[

    'SalesInThousands'

].plot.pie(

    figsize=(7, 7),

    autopct='%1.0f%%'

)



ax.set_ylabel('')

ax.set_title('Sales distributions across different promotions')



plt.show()
ax = df.groupby([

    'Promotion', 'MarketSize'

]).count()[

    'MarketID'

].unstack(

    'MarketSize'

).plot(

    kind='bar',

    figsize=(12, 10),

    grid=True

)



ax.set_ylabel('count')

ax.set_title('Breakdowns of market sizes across different promotions')



plt.show()
# or you can visualize the plot above with a stacked bar chart



ax = df.groupby([

    'Promotion', 'MarketSize'

]).sum()[

    'SalesInThousands'

].unstack(

    'MarketSize'

).plot(

    kind='bar',

    figsize=(12, 10),

    grid=True,

    stacked=True

)



ax.set_ylabel('Sales (in thousands)')

ax.set_title('Breakdowns of market sizes across different promotions')



plt.show()
ax = df.groupby(

    'AgeOfStore'

).count()[

    'MarketID'

].plot(

    kind='bar',

    color='magenta',

    figsize=(10, 7),

    grid=True

)



ax.set_xlabel('age')

ax.set_ylabel('count')

ax.set_title('Overall distributions of age of store')



plt.show()
ax = df.groupby(

    ['AgeOfStore', 'Promotion']

).count()[

    'MarketID'

].unstack(

    'Promotion'

).iloc[::-1].plot(

    kind='barh', 

    figsize=(12,15),

    grid=True

)



ax.set_ylabel('age')

ax.set_xlabel('count')

ax.set_title('Overall distributions of age of store')



plt.show()
# Look at the summary statistics of store ages across the three promotion groups



df.groupby('Promotion').describe()['AgeOfStore']
df.groupby('Week').count()['MarketID']
df.groupby(['Promotion', 'Week']).count()['MarketID']
df.groupby(['Promotion']).count()['MarketID']
ax1, ax2, ax3 = df.groupby(

    ['Week', 'Promotion']

).count()[

    'MarketID'

].unstack('Promotion').plot.pie(

    subplots=True,

    figsize=(24, 8),

    autopct='%1.0f%%'

)



ax1.set_ylabel('Promotion #1')

ax2.set_ylabel('Promotion #2')

ax3.set_ylabel('Promotion #3')



ax1.set_xlabel('distribution across different weeks')

ax2.set_xlabel('distribution across different weeks')

ax3.set_xlabel('distribution across different weeks')



plt.show()
import numpy as np

from scipy import stats
means = df.groupby('Promotion').mean()['SalesInThousands']

means
stds = df.groupby('Promotion').std()['SalesInThousands']

stds
varis = df.groupby('Promotion').std()['SalesInThousands']**2

varis
ns = df.groupby('Promotion').count()['SalesInThousands']

ns
t_1_vs_2 = (

    means.iloc[0] - means.iloc[1]

)/ np.sqrt(

    (stds.iloc[0]**2/ns.iloc[0]) + (stds.iloc[1]**2/ns.iloc[1])

)
numerator_df_1_vs_2 = (varis.iloc[0]/ns.iloc[0] + varis.iloc[1]/ns.iloc[1])**2



denominator_df_1_vs_2 = (varis.iloc[0]**2/(ns.iloc[0]**2*(ns.iloc[0]-1)))+(varis.iloc[1]**2/(ns.iloc[1]**2*(ns.iloc[1]-1)))



df_1_vs_2 = numerator_df_1_vs_2 / denominator_df_1_vs_2
p_1_vs_2 = (1 - stats.t.cdf(t_1_vs_2, df=df_1_vs_2))*2
print("The t-value is %0.10f and the p-value is %0.10f." % (t_1_vs_2, p_1_vs_2))
t, p = stats.ttest_ind(

    df.loc[df['Promotion'] == 1, 'SalesInThousands'].values,

    df.loc[df['Promotion'] == 2, 'SalesInThousands'].values,

    equal_var=False

)
print("The t-value is %0.10f and the p-value is %0.10f." % (t, p))
t_1_vs_3 = (

    means.iloc[0] - means.iloc[2]

)/ np.sqrt(

    (stds.iloc[0]**2/ns.iloc[0]) + (stds.iloc[2]**2/ns.iloc[2])

)
numerator_df_1_vs_3 = (varis.iloc[0]/ns.iloc[0] + varis.iloc[2]/ns.iloc[2])**2



denominator_df_1_vs_3 = (varis.iloc[0]**2/(ns.iloc[0]**2*(ns.iloc[0]-1)))+(varis.iloc[2]**2/(ns.iloc[2]**2*(ns.iloc[2]-1)))



df_1_vs_3 = numerator_df_1_vs_3 / denominator_df_1_vs_3
p_1_vs_3 = (1 - stats.t.cdf(t_1_vs_3, df=df_1_vs_3))*2
print("The t-value is %0.10f and the p-value is %0.10f." % (t_1_vs_3, p_1_vs_3))
t, p = stats.ttest_ind(

    df.loc[df['Promotion'] == 1, 'SalesInThousands'].values,

    df.loc[df['Promotion'] == 3, 'SalesInThousands'].values,

    equal_var=False

)
print("The t-value is %0.10f and the p-value is %0.10f." % (t, p))