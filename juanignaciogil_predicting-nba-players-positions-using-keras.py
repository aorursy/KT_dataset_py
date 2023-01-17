%load_ext autoreload

%autoreload 2
import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer, StandardScaler



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout
stats = pd.read_csv(r'../input/Seasons_Stats.csv', index_col=0)
stats_clean = stats.drop(['blanl', 'blank2', 'Tm'], axis=1)
stats_clean.head()
players = pd.read_csv(r'../input/Players.csv', index_col=0)

players.head(10)
data = pd.merge(stats_clean, players[['Player', 'height', 'weight']], left_on='Player', right_on='Player', right_index=False,

      how='left', sort=False).fillna(value=0)

data = data[~(data['Pos']==0) & (data['MP'] > 200)]

data.reset_index(inplace=True, drop=True)

data['Player'] = data['Player'].str.replace('*','')



totals = ['PER', 'OWS', 'DWS', 'WS', 'OBPM', 'DBPM', 'BPM', 'VORP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA',

         'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']



for col in totals:

    data[col] = 36 * data[col] / data['MP']
data.tail()
X = data.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()

y = data['Pos'].as_matrix()



encoder = LabelBinarizer()

y_cat = encoder.fit_transform(y)

nlabels = len(encoder.classes_)



scaler =StandardScaler()

Xnorm = scaler.fit_transform(X)



stats2017 = (data['Year'] == 2017)

X_train = Xnorm[~stats2017]

y_train = y_cat[~stats2017]

X_test = Xnorm[stats2017]

y_test = y_cat[stats2017]
model = Sequential()

model.add(Dense(40, activation='relu', input_dim=46))

model.add(Dropout(0.5))

model.add(Dense(30, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(nlabels, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.2, verbose=1)
model.test_on_batch(X_test, y_test, sample_weight=None)
# Production model, using all data

model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0, verbose=1)
first_team_members = ['Russell Westbrook', 'James Harden', 'Anthony Davis', 'LeBron James', 'Kawhi Leonard']

first_team_stats = data[[((x[1]['Player'] in first_team_members) & (x[1]['Year']==2017)) for x in data.iterrows()]]

first_team_stats
pd.DataFrame(index=first_team_stats.loc[:, 'Player'].values, data={'Real': first_team_stats.loc[:, 'Pos'].values,

    'Predicted':encoder.inverse_transform(model.predict(Xnorm[first_team_stats.index, :]))})
mvp = [(1956, 'Bob Pettit'), (1957, 'Bob Cousy'), (1958, 'Bill Russell'), (1959, 'Bob Pettit'), 

(1960, 'Wilt Chamberlain'), (1961, 'Bill Russell'), (1962, 'Bill Russell'), (1963, 'Bill Russell'),

(1964, 'Oscar Robertson'), (1965, 'Bill Russell'), (1966, 'Wilt Chamberlain'), (1967, 'Wilt Chamberlain'),

(1968, 'Wilt Chamberlain'), (1969, 'Wes Unseld'), (1970, 'Willis Reed'), (1971, 'Lew Alcindor'), 

(1972, 'Kareem Abdul-Jabbar'), (1973, 'Dave Cowens'), (19704, 'Kareem Abdul-Jabbar'), (1975, 'Bob McAdoo'),

(1976, 'Kareem Abdul-Jabbar'), (1977, 'Kareem Abdul-Jabbar'), (1978, 'Bill Walton'), (1979, 'Moses Malone'), 

(1980, 'Kareem Abdul-Jabbar'), (1981, 'Julius Erving'), (1982, 'Moses Malone'), (1983, 'Moses Malone'), 

(1984, 'Larry Bird'), (1985, 'Larry Bird'), (1986, 'Larry Bird'), (1987, 'Magic Johnson'), 

(1988, 'Michael Jordan'), (1989, 'Magic Johnson'), (1990, 'Magic Johnson'), (1991, 'Michael Jordan'),

(1992, 'Michael Jordan'), (1993, 'Charles Barkley'), (1994, 'Hakeem Olajuwon'), (1995, 'David Robinson'),  

(1996, 'Michael Jordan'), (1997, 'Karl Malone'), (1998, 'Michael Jordan'), (1999, 'Karl Malone'), 

(2000, 'Shaquille O\'Neal'), (2001, 'Allen Iverson'), (2002, 'Tim Duncan'), (2003, 'Tim Duncan'), 

(2004, 'Kevin Garnett'), (2005, 'Steve Nash'), (2006, 'Steve Nash'), (2007, 'Dirk Nowitzki'), 

(2008, 'Kobe Bryant'), (2009, 'LeBron James'), (2010, 'LeBron James'), (2011, 'Derrick Rose'), 

(2012, 'LeBron James'), (2013, 'LeBron James'), (2014, 'Kevin Durant'), (2015, 'Stephen Curry'),

(2016, 'Stephen Curry')]
mvp_stats = pd.concat([data[(data['Player'] == x[1]) & (data['Year']==x[0])] for x in mvp], axis=0)
mvp_stats
mvp_pred = pd.DataFrame(index=mvp_stats.loc[:, 'Player'].values, data={'Real': mvp_stats.loc[:, 'Pos'].values,

    'Predicted':encoder.inverse_transform(model.predict(Xnorm[mvp_stats.index, :]))})
mvp_pred
curry2017 = data[(data['Player'] == 'Stephen Curry') & (data['Year']==2017)] 

pettit1956 = data[(data['Player'] == 'Bob Pettit') & (data['Year']==1956)]
time_travel_curry = pd.concat([curry2017 for year in range(1956, 2018)], axis=0)

time_travel_curry['Year'] = range(1956, 2018)



X = time_travel_curry.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()

y = time_travel_curry['Pos'].as_matrix()



y_cat = encoder.transform(y)

Xnorm = scaler.transform(X)



time_travel_curry_pred = pd.DataFrame(index=time_travel_curry.loc[:, 'Year'].values, 

                                data={'Real': time_travel_curry.loc[:, 'Pos'].values,

    'Predicted':encoder.inverse_transform(model.predict(Xnorm))})





time_travel_pettit = pd.concat([pettit1956 for year in range(1956, 2018)], axis=0)

time_travel_pettit['Year'] = range(1956, 2018)



X = time_travel_pettit.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()

y = time_travel_pettit['Pos'].as_matrix()



y_cat = encoder.transform(y)

Xnorm = scaler.transform(X)



time_travel_pettit_pred = pd.DataFrame(index=time_travel_pettit.loc[:, 'Year'].values, 

                                data={'Real': time_travel_pettit.loc[:, 'Pos'].values,

    'Predicted':encoder.inverse_transform(model.predict(Xnorm))})
pd.concat([time_travel_curry_pred,time_travel_pettit_pred],axis=1,keys=['Stephen Curry','Bob Pettit'])
magic = data[(data['Player'] == 'Magic Johnson')] 

jordan = data[(data['Player'] == 'Michael Jordan')]
# Magic

X = magic.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()

y = magic['Pos'].as_matrix()



y_cat = encoder.transform(y)

Xnorm = scaler.transform(X)



magic_pred = pd.DataFrame(index=magic.loc[:, 'Age'].values, 

                                data={'Real': magic.loc[:, 'Pos'].values,

    'Predicted':encoder.inverse_transform(model.predict(Xnorm))})



# Jordan

X = jordan.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()

y = jordan['Pos'].as_matrix()



y_cat = encoder.transform(y)

Xnorm = scaler.transform(X)



jordan_pred = pd.DataFrame(index=jordan.loc[:, 'Age'].values, 

                                data={'Real': jordan.loc[:, 'Pos'].values,

    'Predicted':encoder.inverse_transform(model.predict(Xnorm))})
pd.concat([magic_pred,jordan_pred],axis=1,keys=['Magic Johnson','Michael Jordan'])
first_team_stats
multiplier = np.arange(0.8,1.2,0.02)

growing_predicted = []



for p in first_team_stats.iterrows():

    growing = pd.concat([p[1].to_frame().T for x in multiplier], axis=0)

    growing['height'] = growing['height'] * multiplier

    growing['weight'] = growing['weight'] * (multiplier ** 3)



    X = growing.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()

    y = growing['Pos'].as_matrix()



    y_cat = encoder.transform(y)

    Xnorm = scaler.transform(X)



    growing_predicted.append(pd.DataFrame(index=multiplier, data={'height': growing.loc[:, 'height'].values,

            'Real': growing.loc[:, 'Pos'].values, 'Predicted':encoder.inverse_transform(model.predict(Xnorm))}))
pd.concat(growing_predicted,axis=1,keys=first_team_stats['Player'])