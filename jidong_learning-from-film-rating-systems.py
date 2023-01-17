# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input/data"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import numpy as np

import pandas as pd

import zipfile

from subprocess import check_output

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression





import matplotlib



# Pyplot configuration

matplotlib.style.use('fivethirtyeight')
with zipfile.ZipFile('../input/data/fandango.zip','r') as z: z.extractall('.')

    

print(check_output(["ls", "fandango"]).decode("utf8"))
fandango = pd.read_csv('fandango/fandango_score_comparison.csv')

fandango.head()
# %%% List of films alphabetically sorted %%%



films_sorted = sorted(fandango['FILM'])



print(films_sorted)
# Display list of keys (column names)



fandango.keys()
# WATCH OUT for the following typo: 'Metacritic_user_nom'



# Rename key



fandango.rename(columns={'Metacritic_user_nom':'Metacritic_user_norm'}, inplace=True)



fandango.keys()
# Set index

fandango.set_index('FILM')



# Sort by index

fandango.sort_values(by='FILM', ascending=True, inplace=True)



# Reset numerical index

fandango.reset_index(drop=True, inplace=True)



fandango.head()
fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(8.,6.))



fandango['Fandango_Stars'].plot.hist(alpha=0.5, bins=5, label='Fandango_Stars', ax=axes[0])

fandango['IMDB_norm'].plot.hist(alpha=0.5, bins=10, label='IMDB_norm', ax=axes[0])

axes[0].legend(loc='upper left')

axes[0].set_xlabel('Stars')

axes[0].set_xlim([0.,5.])

axes[0].set_ylim([0.,60.])



fandango['RT_user_norm'].plot.hist(alpha=0.5, bins=10, label='RT_user_norm', ax=axes[1])

fandango['RT_norm'].plot.hist(alpha=0.5, bins=10, label='RT_norm', ax=axes[1])

axes[1].legend(loc='upper left')

axes[1].set_title(' ')



fandango['Metacritic_user_norm'].plot.hist(alpha=0.5, bins=10, label='Metacritic_user_norm', ax=axes[2])

fandango['Metacritic_norm'].plot.hist(alpha=0.5, bins=10, label='Metacritic_norm', ax=axes[2])

axes[2].legend(loc='upper left')

axes[2].set_title(' ')



plt.subplots_adjust(hspace=0.2)



plt.show()

plt.close()
fig, axes = plt.subplots()



rankings_lst = ['Fandango_Stars', 'RT_user_norm', 'RT_norm', 'IMDB_norm', 'Metacritic_user_norm', 'Metacritic_norm']



fandango[rankings_lst].boxplot(vert=False)



axes.set_xlabel('Stars')



plt.show()

plt.close()
fig, axes = plt.subplots()



fandango['Fandango_Stars'].plot.hist(alpha=0.5, bins=5, label='Fandango_Stars', ax=axes)

fandango['Fandango_Ratingvalue'].plot.hist(alpha=0.5, bins=10, label='Fandango_Ratingvalue', ax=axes)

axes.legend(loc='upper left')

axes.set_xlabel('Stars')

axes.set_xlim([0.,5.])

axes.set_ylim([0.,60.])



plt.show()

plt.close()
fig, axes = plt.subplots()



fandango[['Fandango_Stars', 'Fandango_Ratingvalue']].boxplot(vert=False)



axes.set_xlabel('Stars')



plt.show()

plt.close()
fig, axes = plt.subplots()



only_rt_80 = fandango['RT_norm'] >= 4.

rankings_lst = ['Fandango_Stars', 'RT_user_norm', 'IMDB_norm', 'Metacritic_user_norm', 'Metacritic_norm']



with matplotlib.style.context('fivethirtyeight'):

    fandango[rankings_lst].boxplot(vert=False)



with matplotlib.style.context('ggplot'):

    fandango[only_rt_80][rankings_lst].boxplot(vert=False)



axes.set_xlabel('Stars')



plt.title('Red boxes: RT best movies only', fontsize=14)



plt.show()

plt.close()
fig, axes = plt.subplots()



rankings_lst = ['Fandango_Stars', 'RT_user_norm', 'RT_norm', 'IMDB_norm', 'Metacritic_user_norm', 'Metacritic_norm']



cax = axes.matshow(fandango[rankings_lst].corr())



axes.set_yticklabels(['']+rankings_lst)

axes.set_xticklabels(['']+rankings_lst, rotation=90)



fig.colorbar(cax)



plt.show()

plt.close()
fig, axes = plt.subplots()



rankings_lst = ['Fandango_Stars', 'RT_user_norm', 'RT_norm', 'IMDB_norm', 'Metacritic_user_norm', 'Metacritic_norm']



cax = axes.matshow(fandango[only_rt_80][rankings_lst].corr())



axes.set_yticklabels(['']+rankings_lst)

axes.set_xticklabels(['']+rankings_lst, rotation=90)



fig.colorbar(cax)



plt.show()

plt.close()
fig, axes = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(8.,6.))



axes[0].scatter(fandango['RT_norm'], fandango['RT_user_norm'], color='black', alpha=0.5)

axes[0].scatter(fandango[only_rt_80]['RT_norm'], fandango[only_rt_80]['RT_user_norm'], color='red', alpha=0.5)

axes[0].set_ylabel('Stars')

axes[0].set_xlim([0.,5.])

axes[0].set_ylim([0.,5.5])

axes[0].set_title('RT versus RT users', fontsize=14)



axes[1].scatter(fandango['RT_norm'], fandango['Metacritic_norm'], color='black', alpha=0.5)

axes[1].scatter(fandango[only_rt_80]['RT_norm'], fandango[only_rt_80]['Metacritic_norm'], color='red', alpha=0.5)

axes[1].set_ylabel('Stars')

axes[1].set_title('RT versus Metacritic', fontsize=14)



axes[2].scatter(fandango['RT_norm'], fandango['IMDB_norm'], color='black', alpha=0.5)

axes[2].scatter(fandango[only_rt_80]['RT_norm'], fandango[only_rt_80]['IMDB_norm'], color='red', alpha=0.5)

axes[2].set_ylabel('Stars')

axes[2].set_title('RT versus IMDB', fontsize=14)



axes[3].scatter(fandango['Metacritic_norm'], fandango['Fandango_Stars'], color='black', alpha=0.5)

axes[3].scatter(fandango[only_rt_80]['Metacritic_norm'], fandango[only_rt_80]['Fandango_Stars'], color='red', alpha=0.5)

axes[3].set_ylabel('Stars')

axes[3].set_xlabel('Stars')

axes[3].set_title('Metacritic versus Fandango', fontsize=14)



plt.subplots_adjust(hspace=0.3)



plt.show()

plt.close()
# create a feature matrix 'X' by selecting two DataFrame columns

feature_cols = ['RT_user_norm', 'RT_norm', 'Metacritic_user_norm', 'Metacritic_norm']

X = fandango.loc[:, feature_cols]

X.shape



# create a response vector 'y' by selecting a Series

y = fandango['IMDB_norm']

y.shape



# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43)

# Change 'random_state' value to obtain different final results
# Train model

linreg = LinearRegression()

linreg.fit(X_train, y_train)
# use the fitted model to make predictions for the testing set observations

pred = linreg.predict(X_test)
learnt_df = X_test



learnt_df.insert(loc=0, column='IMDB_norm_predicted', value=pd.Series(data=pred, index=learnt_df.index))

learnt_df.insert(loc=0, column='IMDB_norm_actual', value=y_test)



learnt_df[['IMDB_norm_actual', 'IMDB_norm_predicted']].head()
fig, axes = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(8.,6.))



dot1 = axes[0].scatter(fandango['Metacritic_norm'], fandango['IMDB_norm'], color='blue', alpha=0.5)

dot2 = axes[0].scatter(learnt_df['Metacritic_norm'], learnt_df['IMDB_norm_predicted'], color='red', alpha=0.5)

axes[0].set_ylabel('Stars')

axes[0].set_xlim([0.,5.])

axes[0].set_ylim([0.,5.5])

axes[0].set_title('Metacritic versus IMDB', fontsize=14)

axes[0].legend((dot1, dot2),

           ('full dataset', 'predited'),

           scatterpoints=1,

           loc='upper left',

           ncol=3,

           fontsize=8)



axes[1].scatter(fandango['Metacritic_user_norm'], fandango['IMDB_norm'], color='blue', alpha=0.5)

axes[1].scatter(learnt_df['Metacritic_user_norm'], learnt_df['IMDB_norm_predicted'], color='red', alpha=0.5)

axes[1].set_ylabel('Stars')

axes[1].set_title('Metacritic users versus IMDB', fontsize=14)



axes[2].scatter(fandango['RT_norm'], fandango['IMDB_norm'], color='blue', alpha=0.5)

axes[2].scatter(learnt_df['RT_norm'], learnt_df['IMDB_norm_predicted'], color='red', alpha=0.5)

axes[2].set_ylabel('Stars')

axes[2].set_title('RT versus IMDB', fontsize=14)



axes[3].scatter(fandango['RT_user_norm'], fandango['IMDB_norm'], color='blue', alpha=0.5)

axes[3].scatter(learnt_df['RT_user_norm'], learnt_df['IMDB_norm_predicted'], color='red', alpha=0.5)

axes[3].set_ylabel('Stars')

axes[3].set_title('RT users versus IMDB', fontsize=14)

axes[3].set_xlabel('Stars')





plt.subplots_adjust(hspace=0.3)



plt.show()

plt.close()