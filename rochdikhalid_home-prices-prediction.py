import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

import pickle
delhi_data = pd.read_csv('../input/delhi-house-price-prediction/MagicBricks.csv')

delhi_data.head()
delhi_data.shape
delhi_data.columns
delhi_data.describe()
delhi_data.dtypes
col_with_missing_val = delhi_data.isnull().any()

col_with_missing_val
dataframe = delhi_data.drop(['Parking', 'Status', 'Type'], axis = 1)

dataframe.head()
count_missing_vals_by_col = dataframe.isnull().sum()

print(count_missing_vals_by_col[count_missing_vals_by_col > 0])
dataframe['Price'] = dataframe['Price'] / 100

dataframe.head()
dataframe['Per_Sqft'] = dataframe['Price'] / dataframe['Area']

dataframe['Per_Sqft']
dataframe.head()
locality_stats = dataframe['Locality'].value_counts(ascending = False)

print(len(locality_stats))

print(len(locality_stats[locality_stats > 10]))

print(len(locality_stats[locality_stats <= 10])) # Listed as "other" locality
locality_less_than_10 = locality_stats[locality_stats <= 10]

locality_less_than_10                                   
len(dataframe.Locality.unique())
dataframe['Locality'] = dataframe.Locality.apply(lambda x: 'other' if x in locality_less_than_10 else x)

len(dataframe.Locality.unique())
dataframe[dataframe['Locality'] == 'other']
dataframe.Furnishing = dataframe['Furnishing'].fillna(method = 'bfill')

dataframe['Furnishing'].isnull().any()
dataframe.Bathroom = dataframe['Bathroom'].fillna(method = 'bfill')

dataframe['Bathroom'].isnull().any()
dataframe.isnull().any()
furnishing = LabelEncoder()

transaction = LabelEncoder()

locality = LabelEncoder()



dataframe['Furnishing'] = furnishing.fit_transform(dataframe['Furnishing'])

dataframe['Transaction'] = transaction.fit_transform(dataframe['Transaction'])

dataframe['Locality'] = locality.fit_transform(dataframe['Locality'])



dataframe.head()



# Furnishing categories: Furnished: 0 | Semi-Furnished: 1 | Unfurnished: 2

# Transaction categories: New property: 0 | Resale: 1

# Locality categories: "other" category: 27 
dataframe['Locality'].value_counts(ascending = False)
df = dataframe.rename(columns = {'Area': 'Area (sf)', 'BHK': 'Bedroom', 'Price': 'Price ($)', 'Per_Sqft': 'Per Sf ($)'})

df.head()
min_thresold, max_thresold = df['Price ($)'].quantile([0.001, 0.999])

min_thresold, max_thresold
df_outliers = df[(df['Price ($)'] < min_thresold) | (df['Price ($)'] > max_thresold)]

df_outliers
df = df[(df['Price ($)'] > min_thresold) & (df['Price ($)'] < max_thresold)]

df
X = df.drop(['Price ($)', 'Per Sf ($)'], axis = 1)

y = df['Price ($)']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



print(len(X_train))

print(len(X_test))
X
y
def train_model(trained_model):

    

    model = trained_model

    model.fit(X_train, y_train)

    

    y_predicted = model.predict(X_test)  

    

    print(mean_absolute_error(y_test, y_predicted))

    

print('MAE using Random Forest Regressor ($): ')

train_model(RandomForestRegressor(random_state = 42, n_estimators = 300))



print('MAE using Linear Regression ($): ')

train_model(LinearRegression())



print('MAE using Decision Tree Regressor ($): ')

train_model(DecisionTreeRegressor())
def get_kfold_cross(model, data, target):

    

    folds = StratifiedKFold(n_splits = 5)

    

    scores = cross_val_score(model, X, y)

    

    return scores.mean()



print(get_kfold_cross(RandomForestRegressor(n_estimators = 300, random_state = 42), X, y))

print(get_kfold_cross(LinearRegression(), X, y))

print(get_kfold_cross(DecisionTreeRegressor(), X, y))  
model = RandomForestRegressor(n_estimators = 300, random_state = 42)

model.fit(X_train, y_train)
model.predict([[375, 3, 1, 2, 7, 0]])
model.predict([[800, 3, 2, 1, 27, 0]])
model.predict([[300, 3, 1, 2, 9, 0]])
with open('price_prediction_model_pickle', 'wb') as file:

    pickle.dump(model, file)
with open('price_prediction_model_pickle', 'rb') as file:

    trained_model = pickle.load(file)
trained_model.predict([[800, 3, 2, 1, 27, 0]])
trained_model.predict([[300, 3, 1, 2, 9, 0]])