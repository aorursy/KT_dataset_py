import pandas as pd

import matplotlib.pyplot as pl

from keras.models import Sequential

from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
videogame_db = pd.read_csv('../input/videogame-db/video_games.csv')

videogame_db.head()
columns_keep = ['Features.Handheld?', 'Features.Multiplatform?', 'Features.Online?', 

                'Metadata.Genres', 'Metadata.Licensed?', 'Metadata.Publishers', 

                'Metadata.Sequel?', 'Metrics.Review Score', 'Metrics.Sales', 

                'Release.Console', 'Length.All', 'PlayStyles.Average', 'Length.Completionists.Average', 

                'Length.Main + Extras.Average', 'Length.Main Story.Average']



videogame_db = videogame_db.drop(videogame_db.columns.difference(columns_keep), 'columns')



videogame_db[pd.isnull(videogame_db).any(axis=1)]
videogame_db['Metadata.Publishers'] = videogame_db['Metadata.Publishers'].fillna('Other')
videogame_db[pd.isnull(videogame_db).any(axis=1)]
X = videogame_db.loc[:, videogame_db.columns != 'Metrics.Review Score'].values

y = videogame_db.loc[:, ['Metrics.Review Score']]

X[0]

label_encoder = LabelEncoder()

X[:, 3] = label_encoder.fit_transform(X[:, 3])

X[:, 5] = label_encoder.fit_transform(X[:, 5])

X[:, 8] = label_encoder.fit_transform(X[:, 8])


oneHotEncoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [3, 5, 8])], 

                                  remainder='passthrough')

X = oneHotEncoder.fit_transform(X).toarray()
X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.25)
regressor = Sequential()

regressor.add(Dense(units=48, activation='relu', input_dim=94))

regressor.add(Dense(units=48, activation='relu'))

regressor.add(Dense(units=1, activation='linear'))

regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
history = regressor.fit(X_training, y_training, batch_size=20, epochs=100, validation_data=(X_test, y_test))
pl.title('Learning Curves')

pl.xlabel('Epochs')

pl.ylabel('Cross Entropy')

pl.plot(history.history['loss'], label='train')

pl.plot(history.history['val_loss'], label='val')

pl.legend()

pl.show()