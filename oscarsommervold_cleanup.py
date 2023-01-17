import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
bruker_json = pd.read_json('../input/raw-data/bruker.json', orient = 'index').transpose()

columns = bruker_json['columns'][:5].values

data = bruker_json['data']
bruker = pd.DataFrame(columns = columns)

for user in range(len(data)):

    bruker.loc[len(bruker)] = data[user]

bruker
film = pd.read_excel('../input/raw-data/film.xlsx', sheet_name = 'film', index_col = 0)

film
rangering = pd.read_csv('../input/raw-data/rangering.dat', sep = '::', engine = 'python')

rangering.columns =['BrukerID','FilmID','Rangering','Tidstempel']

rangering
# For user-profiles we have a bunch of 'none' to fill out

# For 'Kjønn' i would like to keep the values binary.

print(bruker['Kjonn'].unique())

bruker['Kjonn'] = bruker['Kjonn'].replace([None,'nan'],np.nan)

#Find the probability distribution among genders

gender_prob_dist = bruker.Kjonn.value_counts(normalize = True)

#Create array with Male and Female values based on prob dist

fill = np.random.choice(['M','F'],

                     p=[gender_prob_dist[0], gender_prob_dist[1]], 

                     size = len(bruker[bruker.Kjonn.isna()]))

#Fill all nan values with prob dist array

bruker.loc[bruker.Kjonn.isna(),'Kjonn'] = fill
# For the alder-column we fill all non-valid values with the last observed value, with a limit of 1

print(bruker['Alder'].unique())

bruker['Alder'] = bruker['Alder'].replace([None,'nan'],np.nan)

bruker.Alder.fillna(method = 'ffill', limit = 1, inplace = True)

# The remaining non-valid values are filled with the median value

bruker.Alder.fillna(bruker.Alder.median(), inplace = True)
# For the jobb-column we fill all non-valid values with 0, which is pre-defined as 'not specified' in the readme

print(bruker['Jobb'].unique())

bruker['Jobb'] = bruker['Jobb'].replace([None,'nan'],0)



# For the postcode-column we fill with 0

bruker['Postkode'] = bruker['Postkode'].replace([None,'nan'],0)

bruker.set_index('BrukerID',inplace = True)
bruker.index = bruker.index.map(int)

bruker
# Find all ratings with 10-star scale. That is, find all ratings prior to 01.08.2000



#First we look for missing data

pd.isna(rangering['Tidstempel']).value_counts()

#fill with 0

#Create deep copy to analyse ratings with missing timestamps later on

rangering_copy = rangering.copy()

rangering['Tidstempel'].fillna(0, inplace = True)

# Adding new columns Date and Hour translating the Timestamp values.

date = []

hour = []

for tstemp in rangering.Tidstempel.values:

    t = pd.to_datetime(tstemp,unit = 's')

    date.append(t.date())

    hour.append(t.hour)

rangering['Date'] = date

rangering['Hour'] = hour
# Adding a new feature 'DayOrNight' where 1 is day and 0 is night

# I suspect those who watch movies during nighttime have different rating-habits and movie preferences than those who watch movies during daytime

rangering['DayOrNight'] = rangering.Hour.map(lambda x: 1 if (x>6 & x<23) else 0)

rangering.DayOrNight.value_counts()
#Now we can get all ratings prior to 01.08.2000 with valid timestamps, and change the rating to a 5-star system

rangering.loc[(rangering['Date'] >= datetime.date(1970,1,1)) & (rangering['Date'] < datetime.date(2000,8,1)),'Rangering'] = np.ceil(rangering['Rangering']/2).astype(int)
#now all entries are in the 5-star range

print(rangering['Rangering'].value_counts())

#But what about 10-star scale ratings with invalid timestamps with ratings less than 5?

#We need to find out if they exist
inv_timestamp = rangering_copy[pd.isna(rangering_copy['Tidstempel'])]

inv_timestamp['Rangering'].value_counts()

# For values 1-5 the distribution look very similar to the cleaned table.

# I want to assume that there are no 10-star scale ratings with invalid timestamps with ratings less than 5

# I normalize the distribution of the 1-5 star ratings with invalid timestamps, and compare with the normalized distribution of the cleaned table

# to see if the distribution is similiar

inv_5star = inv_timestamp[inv_timestamp['Rangering'] < 6]

print('Distribution of ratings in the 1-5 range with invalid timestamps \n',inv_5star['Rangering'].value_counts(normalize = True))

print('\n Distribution of ratings from cleaned table \n' ,rangering['Rangering'].value_counts(normalize = True))

# the normalized distributions confirms my assumption, so I assume that there are no 10-star scale ratings with invalid timestamps with ratings less than 5

# Therefore I only convert the ratings with invalid timestamps and a rating greater than 5

#Removing the 'Date' column so that the structure match that of sample_data for further analysis

#I havent decided if i want to further explore the Date column, so I store a deep copy just in case.

rangering_modified = rangering.copy().drop(columns = ['Date','Hour'])

rangering = rangering.drop(columns = ['Date','Hour','DayOrNight'])

rangering_modified
#Moving on to the movie dataset, here I want to create a column for each genre, with binary values.

#I also need to check the data in MovieID's to ensure there are no problems.

genre_dummies = pd.get_dummies(film.Sjanger.str.split('|',expand = True).stack(dropna = False)).sum(level = 0)

genre_dummies

film = pd.concat([film.drop(columns = 'Sjanger'), genre_dummies], axis = 1)

film
#I notice that a column named 'Ukjennt' has appeared

#Apparently there are some movies with 'unknown' genres

ukjent = film['Ukjennt'].value_counts()

ukjent_id = film[film['Ukjennt'] == 1]['FilmID']

#Checking if the movies with unknown genres has been rated by users

ukjent_rangering = rangering[rangering.FilmID.isin(ukjent_id)]

print(ukjent_id.values)

print(ukjent_rangering.FilmID.value_counts())

#The three movies iwth unknown genres has been rated. I'm thinking of manually adding their genres in order to remove the 'Ukjennt' column

film[film['Ukjennt'] == 1]['Tittel']

#I search for the movies using IMDB and find their respective genres.
#Setting FilmID as index to match structure of sample_data

film.set_index('FilmID',inplace = True)
print(film[film.index == 1881].to_string())

#Adding genres for Railroaded! (1947)

film.loc[1881,'Crime'] = 1

film.loc[1881,'Drama'] = 1

film.loc[1881,'Film-Noir'] = 1



print(film[film.index == 2554].to_string())

#Adding genres for Name of the Rose, The (1986)

film.loc[2554,'Crime'] = 1

film.loc[2554,'Drama'] = 1

film.loc[2554,'Mystery'] = 1



print(film[film.index == 3090].to_string())

#Adding genres for Love Jones (1997)

film.loc[3090,'Drama'] = 1

film.loc[3090,'Romance'] = 1



film[film['Ukjennt'] == 1]
film.drop(columns = 'Ukjennt', inplace = True)

film
print(pd.isna(film['Tittel']).value_counts())

# There are no missing values in the Tittel column

# Checking for duplicate values

tittel_unq = film['Tittel'].value_counts()

filmid_unq = film.index.value_counts()

print((filmid_unq > 1).where(lambda x : x == True).dropna())

print((tittel_unq > 1).where(lambda x : x == True).dropna())

# There are no duplicate values, the movie dataset has the same structure as in sample_data and is ready for further analysis
#Fixing the Children genre. At the moment we have both Children and Children's

filmcop = film.copy()

filmcop['Chill'] = film.apply(lambda row: 1 if (row.Children == 1 or row["Children's"] == 1) else 0, axis = 1)

film.drop(columns = ['Children', "Children's"], inplace = True)

film["Children's"] = filmcop.Chill
#we know from earlier analysis that the genres Children's and Animation are highly correlated. Adding this as a new feature

film['Animation | Children'] = film.apply(lambda row: 1 if (row.Animation == 1 and row["Children's"] == 1) else 0, axis = 1)
userrank_df = rangering_modified.merge(bruker,left_on = 'BrukerID', right_on = 'BrukerID')

final_df = userrank_df.merge(film,left_on = 'FilmID', right_on = 'FilmID')
models_val_rmse = []

models_gen_rmse = []

for (i,j) in enumerate(bruker.index):

    X = final_df.loc[final_df['BrukerID'] == j].drop(columns = ['BrukerID','Tittel','Tidstempel','Postkode','Rangering','Postkode','FilmID','Kjonn','Jobb','Alder']).values

    y = final_df.loc[final_df['BrukerID'] == j].Rangering.values

    

    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, train_size = 0.8, test_size=0.2, random_state=42)

    

    lr = LinearRegression()

    lr.fit(X_train,y_train)

    models_val_rmse.append(np.sqrt(mean_squared_error(y_val, np.clip(lr.predict(X_val).round(),1,5))))

    models_gen_rmse.append(np.sqrt(mean_squared_error(y_test, np.clip(lr.predict(X_test).round(),1,5))))
print('The mean RMSE for validation data with new features is:' ,pd.DataFrame(models_val_rmse).mean()[0])

print('\nThe estimated generalization error of the model with new features is: ',pd.DataFrame(models_gen_rmse).mean()[0])
smodels_val_rmse = []

smodels_gen_rmse = []

for (i,j) in enumerate(bruker.index):

    X = final_df.loc[final_df['BrukerID'] == j].drop(columns = ['BrukerID','Tittel','Tidstempel','Postkode','Rangering','Postkode','FilmID','DayOrNight','Animation | Children','Kjonn','Jobb','Alder']).values

    y = final_df.loc[final_df['BrukerID'] == j].Rangering.values

    

    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, train_size = 0.8, test_size=0.2, random_state=42)

    

    lr = LinearRegression()

    lr.fit(X_train,y_train)

    smodels_val_rmse.append(np.sqrt(mean_squared_error(y_val, np.clip(lr.predict(X_val).round(),1,5))))

    smodels_gen_rmse.append(np.sqrt(mean_squared_error(y_test, np.clip(lr.predict(X_test).round(),1,5))))
print('The mean RMSE for validation data is:' ,pd.DataFrame(smodels_val_rmse).mean()[0])

print('\nThe estimated generalization error of the model is: ',pd.DataFrame(smodels_gen_rmse).mean()[0])
#export cleaned dataframes to csv

bruker.to_csv('bruker_cleaned.csv',index = True)

rangering_modified.to_csv('rangering_modified_cleaned.csv', index = False)

rangering.to_csv('rangering_cleaned.csv',index = False)

film.to_csv('film_cleaned.csv', index = True)