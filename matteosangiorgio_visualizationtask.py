import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import os

from dateutil.relativedelta import relativedelta

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('../input/train_electricity.csv')

test_df = pd.read_csv('../input/test_electricity.csv')
print("Dataset has", len(df), "entries.")



print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")

for col_name in df.columns:

    col = df[col_name]

    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")
def add_datetime_features(df):

    features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",

                "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",

                "Hour", "Minute", "Quarter"]

    one_hot_features = ["Month", "Dayofweek", "Quarter"]



    datetime = pd.to_datetime(df.Date * (10 ** 9))



    df['Datetime'] = datetime  # <-- We won't use this for training, but we'll remove it later



    for feature in features:

        new_column = getattr(datetime.dt, feature.lower())

        if feature in one_hot_features:

            df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)

        else:

            df[feature] = new_column

            

    return df



df = add_datetime_features(df)



print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")

for col_name in df.columns:

    col = df[col_name]

    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")
test_df = add_datetime_features(test_df)

print(df.columns)

print(test_df.columns)
#first run the notebook without running this cell. Observe the outliers than come back to this cell and re-run the code



def remove_outliers(df):

    df = df.loc[np.invert(df['Coal_MW']>4500) | np.invert(df['Coal_MW']<300) , :]

    df = df.loc[np.invert(df['Gas_MW']<30), :]

    df = df.loc[np.invert(df['Nuclear_MW']<400), :]

    df = df.loc[np.invert(df['Biomass_MW']>80), :]

    df = df.loc[np.invert(df['Production_MW']<3000), :]

    df = df.loc[np.invert(df['Consumption_MW']>11000) | np.invert(df['Consumption_MW']<2000) , :]

    return df



df = remove_outliers(df)
# # plot consumption (no test set consumption available)

fig, axs = plt.subplots(1, 1, figsize=(15, 5))

axs.plot(df['Datetime'], df['Consumption_MW'], color='royalblue')

axs.set_xlabel('Date')

axs.set_ylabel('Consumption_MW')

plt.show()
# plot consumption (no test set consumption available) - 2017 only 

fig, axs = plt.subplots(1, 1, figsize=(15, 5))

axs.plot(df.loc[df['Year'] == 2017, 'Datetime'],

         df.loc[df['Year'] == 2017, 'Consumption_MW'], color='royalblue')

axs.set_xlabel('Date')

axs.set_ylabel('Consumption_MW')

plt.show()
# plot consumption (no test set consumption available) - 2017 only 

fig, axs = plt.subplots(1, 1, figsize=(15, 5))

axs.plot(df.loc[(df['Year'] == 2017).values & (df['Month_1'] == 1).values, 'Datetime'],

         df.loc[(df['Year'] == 2017).values & (df['Month_1'] == 1).values, 'Consumption_MW'], color='royalblue')

axs.set_xlabel('Date')

axs.set_ylabel('Consumption_MW')

plt.show()
# plot productions

production_list = ['Coal_MW', 'Gas_MW', 'Hidroelectric_MW', 'Nuclear_MW',

                   'Wind_MW', 'Solar_MW', 'Biomass_MW', 'Production_MW']



i = 0

fig, axs = plt.subplots(len(production_list), 1, figsize=(15, 5*len(production_list)))

for production in production_list:

    axs[i].plot(df['Datetime'], df[production], color='royalblue')

    axs[i].set_xlabel('Date')

    axs[i].set_ylabel(production)

    axs[i].plot(test_df['Datetime'], test_df[production], color='orangered')

    i = i+1

    

plt.show()
# plot sum of all the productions - Production_MW

fig, axs = plt.subplots(1, 1, figsize=(15, 5))

axs.plot(df['Datetime'], df['Coal_MW']+df['Gas_MW']+df['Hidroelectric_MW']+df['Nuclear_MW']+df['Wind_MW']+

         df['Solar_MW']+df['Biomass_MW']-df['Production_MW'], color='royalblue')

axs.set_xlabel('Date')

axs.set_ylabel('sum productions - Production_MW')

axs.plot(test_df['Datetime'], test_df['Coal_MW']+test_df['Gas_MW']+test_df['Hidroelectric_MW']+test_df['Nuclear_MW']+

         test_df['Wind_MW']+test_df['Solar_MW']+test_df['Biomass_MW']-test_df['Production_MW'], color='orangered')



plt.show()
# plot Production_MW vs Consumption_MW (no test set consumption available)

fig, axs = plt.subplots(1, 1, figsize=(10, 10))

axs.plot(df['Production_MW'], df['Consumption_MW'], '.', color='royalblue')

axs.set_xlabel('Production_MW')

axs.set_ylabel('Consumption_MW')

axs.set_xlim(3000, 12000)

axs.set_ylim(3000, 12000)

plt.show()
# plot Production_MW vs Consumption_MW (no test set consumption available)

fig, axs = plt.subplots(1, 1, figsize=(10, 10))

axs = sns.kdeplot(df['Production_MW'], df['Consumption_MW'], cmap="Blues", shade=True, cbar=True)

axs.set_xlabel('Production_MW')

axs.set_ylabel('Consumption_MW')

axs.set_xlim(3000, 12000)

axs.set_ylim(3000, 12000)

plt.show()
# check static distribution of each variable (principal diagonal)

# and eventual correlations between each couple of variables (out of the principal)



sns.pairplot(df[['Consumption_MW', 'Coal_MW', 'Gas_MW', 'Hidroelectric_MW',

                 'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW', 'Production_MW']])
fig, axs = plt.subplots(1, 1, figsize=(15, 5))



axs = sns.violinplot(x=(df[['Month_1']].values[:,0]*1+

                        df[['Month_2']].values[:,0]*2+

                        df[['Month_3']].values[:,0]*3+

                        df[['Month_4']].values[:,0]*4+

                        df[['Month_5']].values[:,0]*5+

                        df[['Month_6']].values[:,0]*6+

                        df[['Month_7']].values[:,0]*7+

                        df[['Month_8']].values[:,0]*8+

                        df[['Month_9']].values[:,0]*9+

                        df[['Month_10']].values[:,0]*10+

                        df[['Month_11']].values[:,0]*11+

                        df[['Month_12']].values[:,0]*12),

                     y=df[['Consumption_MW']].values[:,0], color='royalblue')

axs.set_xlabel('Month')

axs.set_ylabel('Consumption_MW')

axs.set_xticks(range(0,12))

axs.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])



plt.show()
fig, axs = plt.subplots(1, 1, figsize=(15, 5))



axs = sns.violinplot(x=(df[['Dayofweek_0']].values[:,0]*1+

                        df[['Dayofweek_1']].values[:,0]*2+

                        df[['Dayofweek_2']].values[:,0]*3+

                        df[['Dayofweek_3']].values[:,0]*4+

                        df[['Dayofweek_4']].values[:,0]*5+

                        df[['Dayofweek_5']].values[:,0]*6+

                        df[['Dayofweek_6']].values[:,0]*7),

                     y=df[['Consumption_MW']].values[:,0], color='royalblue')

axs.set_xlabel('Day of the week')

axs.set_ylabel('Consumption_MW')

axs.set_xticks(range(0,7))

axs.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])



plt.show()
pca = PCA(n_components=10)

principalComponents = pca.fit_transform(df.drop(columns=['Date', 'Consumption_MW', 'Datetime']).values)



fig, axs = plt.subplots(1, 1, figsize=(8, 4))

axs.bar(np.arange(pca.explained_variance_ratio_.shape[0])+1, np.cumsum(pca.explained_variance_ratio_))

axs.set_xlabel('principal components')

axs.set_ylabel('cumulated explained variance ratio')

axs.set_xticks(list(range(1, pca.explained_variance_ratio_.shape[0]+1)))

plt.show()
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(df.drop(columns=['Date', 'Consumption_MW', 'Datetime']).values)



fig, axs = plt.subplots(1, 1, figsize=(9, 7))

cb = axs.scatter(principalComponents[:, 0], principalComponents[:, 1], c=df[['Consumption_MW']].values[:,0], cmap='jet')

axs.set_xlabel('pc 1')

axs.set_ylabel('pc 2')

plt.colorbar(cb)

plt.show()
pca = PCA(n_components=3)

principalComponents = pca.fit_transform(df.drop(columns=['Date', 'Consumption_MW', 'Datetime']).values)



fig = plt.figure(figsize=(13, 10))

ax = fig.add_subplot(111, projection='3d')

cb = ax.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2],

                c=df[['Consumption_MW']].values[:,0], cmap='jet', alpha=0.2)

ax.set_xlabel('pc 1')

ax.set_ylabel('pc 2')

ax.set_zlabel('pc 3')

plt.colorbar(cb)

plt.show()