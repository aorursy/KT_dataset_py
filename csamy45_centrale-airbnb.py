import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
listings = pd.read_csv("../input/listings.csv/listings.csv")

calendar = pd.read_csv("../input/calendar.csv/calendar.csv")
listings.head(3)
print("Il y a %d annonces concernant %d logements parisiens." %(len(calendar), len(listings)))

print("Les annonces sont réparties du " + calendar["date"].min() + " au " + calendar["date"].max() + " (date au format aaaa/mm/jj)")

print("Chaque logement possède %d caractéristiques." %listings.columns.size)
listings.price = listings.price.str.replace("[$, ]","").astype("float32")
listings.groupby("neighbourhood_cleansed").price.mean().sort_values(ascending = False).plot.bar()
listings_null = listings.isnull().mean()*100

listings_null[listings_null > 0].sort_values(ascending = False).plot.bar(figsize=(20,5),fontsize=15)
features_to_keep = ['description','id', "price", 'host_listings_count',

       'neighbourhood_cleansed', 'property_type', 'room_type', 'accommodates',

       'bathrooms', 'bedrooms', 'beds', 'bed_type',

       'number_of_reviews', 'amenities']



df = listings[features_to_keep]
df['metres'] = df['description'].str.extract('(\d{2,3}\s?[mM])')

df['metres'] = df['metres'].str.replace("\D", "")

df['metres'] = df['metres'].astype(float)

df['sq_ft'] = df['description'].str.extract('(\d{2,3}\s?(sq|Sq|SQ))')[0]

df['sq_ft'] = df['sq_ft'].str.replace("\D", "")

df['sq_ft'] = df['sq_ft'].astype(float)

# on convertit les square feet en m2 :

df['sq_ft'] = df['sq_ft']/10.764

# on remplace les valeurs manquantes de sq_ft par la valeur dans la colonne metres

df['metres'] = df['sq_ft'].fillna(df["metres"])

# on enlève les valeurs abérrantes

df["metres"] = df["metres"].apply(lambda row: np.NaN if (row < 10 or row > 200) else row)

df = df.drop(columns = ['sq_ft','description'])
missing = df.columns[df.isna().sum() > 0]

for col in missing:

    df[col] = df[col].fillna(df[col].median())
df.head()
categorical = ["neighbourhood_cleansed","property_type","room_type","bed_type"]



for col in categorical:

    df = pd.concat([df, pd.get_dummies(df[col])], axis = 1)



df = df.drop(columns=categorical)
df.head()
df["amenities"] = df['amenities'].map(

    lambda amns: "|".join([amn.replace("}", "").replace("{", "").replace('"', "") for amn in amns.split(",")])

)
amenities_list = np.unique(np.concatenate(df.amenities.map(lambda amenity : amenity.split("|"))))[1:]

amenity_arr = np.array([df['amenities'].map(lambda amns: amn in amns) for amn in amenities_list])

df = pd.concat([df, pd.DataFrame(data=amenity_arr.T, columns=amenities_list)], axis=1)

df = df.drop(columns = ["amenities","id"])
df = df.drop(columns = ["translation missing: en.hosting_amenity_49","translation missing: en.hosting_amenity_50","Other"])
df.head(3)
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split #split

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error #metrics
TEST_SIZE = 0.3

RAND_STATE = 42



X = df.drop(columns = ['price'])[(df.price < 600) & (df.price >= 5)]

y = df[(df.price < 600) & (df.price >= 5)].price



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state=RAND_STATE)
moyenne_train = np.zeros(y_train.shape[0]) + y_train.mean()

moyenne_test = np.zeros(y_test.shape[0]) + y_train.mean()

print('Moyenne MAE train: %.3f, test: %.3f' % (

        mean_absolute_error(y_train, moyenne_train),

        mean_absolute_error(y_test, moyenne_test)))

print('Moyenne MAPE train: %.3f, test: %.3f' % (

        np.sum(np.abs((y_train - moyenne_train) / y_train)),

        np.sum(np.abs((y_test - moyenne_test) / y_test))))
lineaire = LinearRegression()

lineaire.fit(X_train, y_train)

y_train_preds = lineaire.predict(X_train)

y_test_preds = lineaire.predict(X_test)



print('Linear Regression MAE train: %.3f, test: %.3f' % (

        mean_absolute_error(y_train, y_train_preds),

        mean_absolute_error(y_test, y_test_preds)))

print('Linear Regression MAPE train: %.3f, test: %.3f' % (

        np.sum(np.abs((y_train - y_train_preds) / y_train)),

        np.sum(np.abs((y_test - y_test_preds) / y_test))))
lasso = Lasso(alpha=0.01)

lasso.fit(X_train, y_train)

y_train_preds = lasso.predict(X_train)

y_test_preds = lasso.predict(X_test)



print('Lasso MAE train: %.3f, test: %.3f' % (

        mean_absolute_error(y_train, y_train_preds),

        mean_absolute_error(y_test, y_test_preds)))

print('Lasso MAPE train: %.3f, test: %.3f' % (

        np.sum(np.abs((y_train - y_train_preds) / y_train)),

        np.sum(np.abs((y_test - y_test_preds) / y_test))))
from joblib import dump

dump(lasso, 'modelLasso.joblib') 
forest = RandomForestRegressor(n_estimators=40, 

                               criterion='mse', 

                               random_state=RAND_STATE, 

                               n_jobs=-1)

forest.fit(X_train, y_train)

y_train_preds = forest.predict(X_train)

y_test_preds = forest.predict(X_test)



print('Random Forest MAE train: %.3f, test: %.3f' % (

        mean_absolute_error(y_train, y_train_preds),

        mean_absolute_error(y_test, y_test_preds)))

print('Random Forest MAPE train: %.3f, test: %.3f' % (

        np.sum(np.abs((y_train - y_train_preds) / y_train)),

        np.sum(np.abs((y_test - y_test_preds) / y_test))))