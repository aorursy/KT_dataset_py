import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



custom_style = {'axes.edgecolor': 'white',

 'axes.facecolor': '#EAEAF2',

 'axes.labelcolor': '.15',

 'grid.color': 'white',

 'text.color': '.15',

 'xtick.color': '.15',

 'ytick.color': '.15'}

sns.set_style("darkgrid", rc=custom_style) 

# This dict is to make the seaborn visuals readable in the Jupyter Notebook Dark Theme

# credits to mwaskom and Kyle Kelley on Stack Overflow



import matplotlib.pyplot as plt



import warnings

warnings.simplefilter("ignore")

warnings.filterwarnings("ignore")

tracks = pd.read_csv('../input/ultimate-spotify-tracks-db/SpotifyFeatures.csv',index_col=0)

tracks.head()
tracks.isnull().sum()
genres = tracks.loc[~tracks.index.duplicated()]

list(genres.index)
data = tracks.drop(labels = ['Movie',

 'R&B',

 'A Capella',

 'Alternative',

 'Country',

 'Dance',

 'Electronic',

 'Anime',

 'Folk',

 'Blues',

 'Opera',

 "Children's Music",

 'Children’s Music',

 'Indie',

 'Classical',

 'Pop',

 'Reggae',

 'Reggaeton',

 'Jazz',

 'Rock',

 'Ska',

 'Comedy',

 'Soul',

 'Soundtrack',

 'World'])

data
data.loc[data['track_name'] == 'MIDDLE CHILD']
joint = data.loc[data.duplicated(subset='track_name', keep=False)==True]

joint
joint.describe()
sns.distplot(joint['popularity']).set_title('Popularity Distribution of Hip-Hop/Rap Songs')
popular = joint.loc[joint.popularity >= 65]

popular
popular.corr()
sns.pairplot(popular, y_vars=['popularity'], x_vars=['acousticness','danceability','duration_ms',

                                                       'energy','instrumentalness','liveness','loudness',

                                                       'speechiness','tempo','valence'])
sns.scatterplot(data=popular,x='danceability',y='popularity',hue='mode').set_title('Popularity vs. Danceability of Popular Hip-Hop/Rap Songs')
sns.scatterplot(data=popular,y='popularity',x='loudness',hue='mode').set_title('Popularity vs. Loudness of Popular Hip-Hop/Rap Songs')
sns.scatterplot(data=popular,y='popularity',x='tempo',hue='mode').set_title('Popularity vs. Tempo of Popular Hip-Hop/Rap Songs')
sns.barplot(data=popular,x='popularity',y='key',

           order=['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']).set_title('Popularity For The Different Keys')
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,ExtraTreesRegressor,ExtraTreesClassifier

from xgboost import XGBRegressor, XGBClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder
features = ['acousticness','danceability','duration_ms','energy','instrumentalness','key',

           'liveness','loudness','mode','speechiness','tempo','time_signature','valence']
X = popular[features]

y = popular['popularity']
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state = 1)
columns = X.dtypes == 'object'

categorical = columns[columns].index

label_train_X = train_X.copy()

label_valid_X = valid_X.copy()



label_encoder = LabelEncoder()

for col in categorical:

    label_train_X[col] = label_encoder.fit_transform(train_X[col])

    label_valid_X[col] = label_encoder.transform(valid_X[col])
def model_accuracy(model, X_t=label_train_X, X_v=label_valid_X, y_t=train_y, y_v=valid_y):

    model.fit(X_t,y_t)

    predictions = model.predict(X_v)

    return accuracy_score(y_v, predictions)
def model_score(model, X_t=label_train_X, X_v=label_valid_X, y_t=train_y, y_v=valid_y):

    model.fit(X_t,y_t)

    predictions = model.predict(X_v)

    return mean_absolute_error(y_v, predictions)
RFR1 = RandomForestRegressor(n_estimators = 100, random_state = 1)

RFR2 = RandomForestRegressor(n_estimators = 250, random_state = 1)

RFR3 = RandomForestRegressor(n_estimators = 500, random_state = 1)



RFR_Model_1 = RFR1.fit(label_train_X,train_y)

RFR_Model_2 = RFR2.fit(label_train_X,train_y)

RFR_Model_3 = RFR3.fit(label_train_X,train_y)



models = [RFR_Model_1, RFR_Model_2, RFR_Model_3]



for i in range(0,len(models)):

    RFRscore = model_score(models[i])

    print("Random Forest Regression %d Mean Absolute Error: %f" % (i+1,RFRscore))
ETR1 = ExtraTreesRegressor(n_estimators = 100, random_state = 1)

ETR2 = ExtraTreesRegressor(n_estimators = 250, random_state = 1)

ETR3 = ExtraTreesRegressor(n_estimators = 500, random_state = 1)



ETR_Model_1 = ETR1.fit(label_train_X,train_y)

ETR_Model_2 = ETR2.fit(label_train_X,train_y)

ETR_Model_3 = ETR3.fit(label_train_X,train_y)



models = [ETR_Model_1, ETR_Model_2, ETR_Model_3]



for i in range(0,len(models)):

    ETRscore = model_score(models[i])

    print("Extra Trees Regression %d Mean Absolute Error: %f" % (i+1,ETRscore))
XGBR1 = XGBRegressor(n_estimators = 100, random_state = 1,verbosity=0)

XGBR2 = XGBRegressor(n_estimators = 250, random_state = 1,verbosity=0)

XGBR3 = XGBRegressor(n_estimators = 500, random_state = 1,verbosity=0)



XGBR_Model_1 = XGBR1.fit(label_train_X,train_y)

XGBR_Model_2 = XGBR2.fit(label_train_X,train_y)

XGBR_Model_3 = XGBR3.fit(label_train_X,train_y)



models = [XGBR_Model_1, XGBR_Model_2, XGBR_Model_3]



for i in range(0,len(models)):

    XGBRscore = model_score(models[i])

    print("Extreme Gradient Boost Regression %d Mean Absolute Error: %f" % (i+1,XGBRscore))
RFC1 = RandomForestClassifier(n_estimators = 100, random_state = 1)

RFC2 = RandomForestClassifier(n_estimators = 250, random_state = 1)

RFC3 = RandomForestClassifier(n_estimators = 500, random_state = 1)



RFC_Model_1 = RFC1.fit(label_train_X,train_y)

RFC_Model_2 = RFC2.fit(label_train_X,train_y)

RFC_Model_3 = RFC3.fit(label_train_X,train_y)



models = [RFC_Model_1, RFC_Model_2, RFC_Model_3]



for i in range(0,len(models)):

    RFCaccuracy = model_accuracy(models[i])

    print("Random Forest Classifier %d Accuracy: %f" % (i+1,RFCaccuracy))
ETC1 = ExtraTreesClassifier(n_estimators = 100, random_state = 1)

ETC2 = ExtraTreesClassifier(n_estimators = 250, random_state = 1)

ETC3 = ExtraTreesClassifier(n_estimators = 500, random_state = 1)



ETC_Model_1 = ETC1.fit(label_train_X,train_y)

ETC_Model_2 = ETC2.fit(label_train_X,train_y)

ETC_Model_3 = ETC3.fit(label_train_X,train_y)



models = [ETC_Model_1, ETC_Model_2, ETC_Model_3]



for i in range(0,len(models)):

    ETCaccuracy = model_accuracy(models[i])

    print("Extra Trees Classifier %d Accuracy: %f" % (i+1,ETCaccuracy))
XGBC1 = XGBClassifier(n_estimators = 100, random_state = 1)

XGBC2 = XGBClassifier(n_estimators = 250, random_state = 1)

XGBC3 = XGBClassifier(n_estimators = 500, random_state = 1)



XGBC_Model_1 = XGBC1.fit(label_train_X,train_y)

XGBC_Model_2 = XGBC2.fit(label_train_X,train_y)

XGBC_Model_3 = XGBC3.fit(label_train_X,train_y)



models = [XGBC_Model_1, XGBC_Model_2, XGBC_Model_3]



for i in range(0,len(models)):

    XGBCaccuracy = model_accuracy(models[i])

    print("Extreme Gradient Boost Classifier %d Accuracy: %f" % (i+1,XGBCaccuracy))
sns.distplot(joint['popularity']).set_title('Popularity Distribution of Hip-Hop/Rap Songs')
from scipy.stats import beta 

sns.distplot(joint['popularity'], fit=beta)