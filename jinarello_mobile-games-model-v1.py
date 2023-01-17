#Going to try and predict average rating from other data.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



game_data = pd.read_csv("../input/17k-apple-app-store-strategy-games/appstore_games.csv", index_col="ID", parse_dates=['Original Release Date', 'Current Version Release Date'])

game_data.head()
from sklearn import preprocessing

from sklearn.impute import SimpleImputer



#clean and fill the data with reasonable 0 or EN for languages. Maybe can put in if to change average user rating to None if rating count=0?

game_data["In-app Purchases"] = game_data["In-app Purchases"].fillna(0, axis=0)

game_data["Average User Rating"] = game_data["Average User Rating"].fillna(0, axis=0)

game_data["User Rating Count"] = game_data["User Rating Count"].fillna(0, axis=0)

game_data["Price"] = game_data["Price"].fillna(0, axis=0)

game_data["Languages"] = game_data["Languages"].fillna("EN", axis=0)

game_data['Size'].fillna(game_data['Size'].mean(), inplace=True)



game_data['Size'] = round(game_data['Size']/10**6, 2)



import datetime as dt

game_data['Original Release Year'] = game_data['Original Release Date'].dt.year

game_data['Current Version Release Year'] = game_data['Current Version Release Date'].dt.year



#change to integer example yyyymmdd to plot in seaborn cause its dumd, otherwise the dtype is fine to use in matplot

#game_data['Original Release Date'] = game_data['Original Release Date'].astype(str).replace({'-':''}, regex=True).astype(int)

#game_data['Current Version Release Date'] = game_data['Current Version Release Date'].astype(str).replace({'-':''}, regex=True).astype(int)



def to_binary(x):

    if x == 0:

        return 0

    else:

        return 1

        

game_data['In-app Purchases'] = game_data['In-app Purchases'].apply(to_binary)





#only needs run once, unless done on copy of data and work with that from then on. Dropped data is unneccesary here anyway tho.

def count_lang(y):

    return len(y.split(','))



game_data['Languages'] = game_data['Languages'].apply(count_lang)

game_data = game_data.drop(['Subtitle', 'Icon URL'], axis=1)

game_data.drop_duplicates(inplace=True)



game_data.head()

#visualization kernel if want it

import seaborn as sb

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline



#size histogram, drop NaN or won't work

#plot=sb.distplot(game_data["Size"], color="r")



#plot=sb.distplot(game_data["Price"], color="r", hist=False)

#plt.xlim(0,1000)



#plot=sb.kdeplot(game_data["Average User Rating"], color="b",)

#plt.figure(figsize=(10, 10))

#sb.regplot(data=game_data, x="Original Release Date", y="Average User Rating")



sb.regplot(x=game_data['Size'], y=game_data['Original Release Year'])

sb.regplot(data=game_data, x="Original Release Year", y="Average User Rating")

#game_data.Developer.value_counts()



game_data['In-app Purchases']=round(game_data['In-app Purchases'].astype(np.float64), 1)

game_data['Languages']=round(game_data['Languages'].astype(np.float64),1)

game_data['Original Release Year']=round(game_data['Original Release Year'].astype(np.float64), 1)

game_data['Current Version Release Year']=round(game_data['Current Version Release Year'].astype(np.float64),1)

game_data['User Rating Count']=round(game_data['User Rating Count'].astype(np.float64),1)

game_data.dtypes





#game_data.loc[game_data['Developer']=='Giancarlo Cavalcante']

#deleted about 100-150 duplicates (how'd those get in there?), compared .count() to nunique() before and after.



#inf = lambda df: df[df.isinf().any(axis=1)]

#inf(game_data)

#which is NaN? Used to see if NaN for user ratings is NaN because no ratings been given, and other NaNs to get rid of

#nans = lambda df: df[df.isnull().any(axis=1)]

#nans(game_data)
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

import category_encoders as ce



features = ['User Rating Count', 'Price', 'In-app Purchases', 'Languages', 'Size', 'Original Release Year', 'Current Version Release Year']



label_cols = ['Age Rating']

le = preprocessing.LabelEncoder()

# le.fit(X_train['Age Rating'])

# X_train['Age Rating'] = le.transform(X_train['Age Rating'])

# X_val['Age Rating'] = le.transform(X_val['Age Rating'])

encoded = game_data[label_cols].apply(le.fit_transform)

data = game_data[features].join(encoded)



#might move to after traintestsplit to try and avoid data leakage, but I dont think given what the data represents that its a big deal.

count_cols = ['Developer'] 

count_enc = ce.CountEncoder(cols=count_cols)

# for col in game_data[count_cols]:

#     count_enc.fit(X_train[col])

#     X_train[col] = count_enc.transform(X_train[col])

#     X_val[col] = count_enc.transform(X_val[col])

finished_data = data.join(count_enc.fit_transform(game_data[count_cols]).add_suffix("_count"))

#finished_data = data.join(count_enc.transform(game_data[count_cols]).add_suffix("_count"))



y = game_data["Average User Rating"]         

X = finished_data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)



game_model = RandomForestRegressor(n_estimators=150, max_leaf_nodes=100, random_state=1)

game_model.fit(X_train, y_train)



game_model_val_prediction = game_model.predict(X_val)

game_model_val_mae = mean_absolute_error(game_model_val_prediction, y_val)

game_model_R2 = r2_score(game_model_val_prediction, y_val)



print("Validation MAE for Random Forest Model: {}".format(game_model_val_mae))

print("R^2 score for RFM: {}".format(game_model_R2))

#average user rating is from 0 to 5 range, so an MAE of 0.223547.. is 4.67% error. Seems suspiciously low... maybe...

finished_data.describe()



#XGBOOOOOOOST

XG_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state=1)

XG_model.fit(X_train, y_train,early_stopping_rounds=5, eval_set=[(X_val, y_val)], verbose=False)



XG_model_val_prediction = XG_model.predict(X_val)

XG_model_val_mae = mean_absolute_error(XG_model_val_prediction, y_val)

XG_model_R2 = r2_score(XG_model_val_prediction, y_val)



print("Validation MAE for XGBoost: {}".format(XG_model_val_mae))

print("R^2 score for XG: {}".format(game_model_R2))
