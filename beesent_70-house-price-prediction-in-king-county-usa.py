import pandas as pd



housing_data = pd.read_csv("../input/kc_house_data.csv")

features = [u'bedrooms', u'bathrooms', u'sqft_living',

       u'floors', u'condition', u'grade', u'sqft_lot15',

            u'sqft_lot',

       u'sqft_above',  u'sqft_living15', u'sqft_basement']



price = housing_data['price']



housing_data = pd.DataFrame(housing_data, columns=features)

housing_data.head()
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(housing_data, price, random_state=0)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
from pandas import read_csv, DataFrame

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor



models = [LinearRegression(),

	          RandomForestRegressor(n_estimators=100, max_features ='sqrt'), 

	          KNeighborsRegressor(n_neighbors=6), 

	          SVR(kernel='linear'), 

	          ]



train_results = []



for model in models:

    model.fit(X_train_std, y_train)

    y_train_pred = model.predict(X_train_std)

    y_test_pred = model.predict(X_test_std)

    train_results.append([model, y_train_pred, y_test_pred])

    accuracy = model.score(X_test_std, y_test)

    print("Accuracy: {}%".format(int(round(accuracy * 100))))