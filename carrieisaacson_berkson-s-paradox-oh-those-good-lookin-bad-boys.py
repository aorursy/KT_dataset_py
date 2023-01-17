import numpy as np

import pandas as pd



from numpy.random import multivariate_normal, seed



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



seed(42)

df = pd.DataFrame( np.round(multivariate_normal([5,5],[[6,0],[0,6]],100),1, ),

                 columns = ['attractiveness', 'personality'])

df = df.applymap(lambda x: 0 if x < 1 else x)

df = df.applymap(lambda x: 10 if x > 10 else x)



fig, axarr = plt.subplots(1, 2, figsize=(16, 4))

plt.subplot(1,2,1)

_ = sns.regplot(x = "attractiveness", y = "personality", data=df)

plt.title('All Potential Dates')

plt.xlim([-0.5,10.5])

plt.ylim([-0.5,10.5])

plt.subplot(1,2,2)

sns.regplot(x = "attractiveness",

                y = "personality", 

                data=df[np.logical_or(df.attractiveness > 6, df.personality > 6)])

plt.title('At least a 6')

plt.xlim([-0.5,10.5])

plt.ylim([-0.5,10.5])

_ = plt.fill_between(x=[-0.5,6], y1=[6,6], y2=[-0.5,-0.5], hatch='/', edgecolor="r", facecolor="none")
seed(101)

df = pd.DataFrame( np.round(multivariate_normal([5,5],[[6,0],[0,6]],100),1, ),

                 columns = ['book', 'movie'])

df = df.applymap(lambda x: 0 if x < 1 else x)

df = df.applymap(lambda x: 10 if x > 10 else x)



fig, axarr = plt.subplots(1, 2, figsize=(16, 4))

plt.subplot(1,2,1)

_ = sns.regplot(x = "book", y = "movie", data=df, color='indigo')

plt.title('All Movies')

plt.xlim([-0.5,10.5])

plt.ylim([-0.5,10.5])

plt.subplot(1,2,2)

sns.regplot(x = "book",

            y = "movie", 

            data=df[df.book + df.movie > 8],

            color='indigo')

plt.title('Constant Minimum Quality')

plt.xlim([-0.5,10.5])

plt.ylim([-0.5,10.5])

_ = plt.fill_between(x=[-0.5,8], y1=[8,0], y2=[-0.5,-0.5], hatch='/', edgecolor="r", facecolor="none")
df = pd.DataFrame({'cholecystitis': [28, 68],

                    'not cholecystitis': [548, 2606]},

                   index=['diabetes', 'needs glasses'])

df['prevalence %'] = round(100*df['cholecystitis'] / (df['cholecystitis'] + df['not cholecystitis']),1)

df
df = pd.DataFrame({'cholecystitis': [3000 ,  29700],

                    'not cholecystitis': [97000 , 960300 ]},

                   index=['diabetes', 'needs glasses'])

df['prevalence %'] = round(100*df['cholecystitis'] / (df['cholecystitis'] + df['not cholecystitis']),1)

df