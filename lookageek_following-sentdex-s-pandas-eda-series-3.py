import pandas as pd



df = pd.read_csv('../input/diamonds.csv', index_col=0)

df.head()
df['cut'].astype("category").cat.codes[:200]
cut_class_dict = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}

color_dict = {"J": 1,"I": 2,"H": 3,"G": 4,"F": 5,"E": 6,"D": 7}

clarity_dict = {"I3": 1, "I2": 2, "I1": 3, "SI2": 4, "SI1": 5, "VS2": 6, "VS1": 7, "VVS2": 8, "VVS1": 9, "IF": 10, "FL": 11}
# Mapping using these dictionaries in the dataframe

df['cut'] = df['cut'].map(cut_class_dict)

df['color'] = df['color'].map(color_dict)

df['clarity'] = df['clarity'].map(clarity_dict)



df.head()
import sklearn

from sklearn import svm



# Shuffle the dataframe using sk learn, you can use pandas reindex method with np.random too

df = sklearn.utils.shuffle(df)
# X is the feature set - a list of list of features

X = df.drop('price', axis=1).values

# y is the target - a list of prices

y = df['price'].values
X = sklearn.preprocessing.scale(X)
# Video uses a manual method, but sklearn gives a nice method to do this declaratively

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)
%%time

# Support Vector Regression with Linear Kernel

model = svm.SVR(kernel='linear')

model.fit(X_train, y_train)
model.score(X_test, y_test)
for X, y in list(zip(X_test, y_test))[:50]:

    print(f"Predicted: {model.predict([X])[0]}, Actual: {y}")
%%time

# Support Vector Regression with RBF Kernel

model = svm.SVR(kernel='rbf')

model.fit(X_train, y_train)
print(f"--- Score: {model.score(X_test, y_test)} ---")



for X, y in list(zip(X_test, y_test))[:50]:

    print(f"Predicted: {model.predict([X])[0]}, Actual: {y}")
%%time

# Using SGD Regressor

model = sklearn.linear_model.SGDRegressor(max_iter=10_000)

model.fit(X_train, y_train)
print(f"--- Score: {model.score(X_test, y_test)} ---")



for X, y in list(zip(X_test, y_test))[:50]:

    print(f"Predicted: {model.predict([X])[0]}, Actual: {y}")
%%time

# Using Linear Regression

model = sklearn.linear_model.LinearRegression()

model.fit(X_train, y_train)
print(f"--- Score: {model.score(X_test, y_test)} ---")



for X, y in list(zip(X_test, y_test))[:50]:

    print(f"Predicted: {model.predict([X])[0]}, Actual: {y}")
df = pd.read_csv('../input/diamonds.csv', index_col=0)

dummies_df = pd.get_dummies(df)

dummies_df.head()
# X is the feature set - a list of list of features

X = dummies_df.drop('price', axis=1)

# y is the target - a list of prices

y = dummies_df['price'].values



# Scale our features

X[['depth', 'carat', 'table', 'x', 'y', 'z']] = sklearn.preprocessing.scale(X[['depth', 'carat', 'table', 'x', 'y', 'z']])

X = X.values



X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)
%%time

# Support Vector Regression with Linear Kernel

model = svm.SVR(kernel='linear')

model.fit(X_train, y_train)
print(f"--- Score: {model.score(X_test, y_test)} ---")



for X, y in list(zip(X_test, y_test))[:50]:

    print(f"Predicted: {model.predict([X])[0]}, Actual: {y}")
%%time

# Support Vector Regression with RBF Kernel

model = svm.SVR(kernel='rbf')

model.fit(X_train, y_train)
print(f"--- Score: {model.score(X_test, y_test)} ---")



for X, y in list(zip(X_test, y_test))[:50]:

    print(f"Predicted: {model.predict([X])[0]}, Actual: {y}")
%%time

# Using SGD Regressor

model = sklearn.linear_model.SGDRegressor(max_iter=10_000)

model.fit(X_train, y_train)
print(f"--- Score: {model.score(X_test, y_test)} ---")



for X, y in list(zip(X_test, y_test))[:50]:

    print(f"Predicted: {model.predict([X])[0]}, Actual: {y}")
%%time

# Using Linear Regression

model = sklearn.linear_model.LinearRegression()

model.fit(X_train, y_train)
print(f"--- Score: {model.score(X_test, y_test)} ---")



for X, y in list(zip(X_test, y_test))[:50]:

    print(f"Predicted: {model.predict([X])[0]}, Actual: {y}")