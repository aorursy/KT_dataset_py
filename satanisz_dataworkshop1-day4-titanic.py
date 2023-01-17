import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
df = pd.read_csv("../input/train.csv")

df.head()
# Prepare Data

df['Sex_cat'] = df['Sex'].factorize()[0]  # Faktoryzacja

median = df['Age'].median()     # Zamiana brakujacych danych

df['Age'].fillna(median, inplace=True) # przez mediane



feats = ['Pclass', 'Fare', 'Sex_cat', 'Age']



X = df[ feats ].values

y = df['Survived'].values





# Trenuj

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train.shape, X_test.shape
model = DecisionTreeClassifier(max_depth=15)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



accuracy_score(y_test, y_pred)



model = RandomForestClassifier(max_depth=15, n_estimators=50)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



accuracy_score(y_test, y_pred)
df['Age_norm']= df['Age']/df['Age'].max()

df['Fare_norm']= df['Fare']/df['Fare'].max()

feats = ['Pclass', 'Fare', 'Sex_cat', 'Age_norm']



X = df[ feats ].values

y = df['Survived'].values





# Trenuj

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train.shape, X_test.shape



model = RandomForestClassifier(max_depth=25)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



accuracy_score(y_test, y_pred)