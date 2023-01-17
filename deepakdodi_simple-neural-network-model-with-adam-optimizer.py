import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential

from keras.layers import Dense
train = pd.read_csv("../input/train.csv")

train.head()
train.isnull().sum().sum()
#There are no null values in the data provided
test = pd.read_csv("../input/test.csv")

test.head()
X_train = train.drop(['ID_code', 'target'], axis = 1)

X_train.head()
y_train = train['target']

y_train.head()
X_test = test.drop(['ID_code'], axis = 1)

X_test.head()
test.isnull().sum().sum()
X_train.var()
# As the variance is high, we need to scale out the features
ss = StandardScaler()

X_train_Scaled = ss.fit_transform(X_train)
X_test_Scaled = ss.transform(X_test)
y_train.value_counts(dropna=False)
encoder = LabelEncoder()

encoder.fit(y_train)

encoded_Y = encoder.transform(y_train)
model = Sequential()

model.add(Dense(200, activation='relu'))

model.add(Dense(25, activation='relu'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train_Scaled, encoded_Y, epochs=200, batch_size = 32, shuffle=True, verbose=1)
pred = model.predict(X_test_Scaled)
result = pd.DataFrame({"ID_code": test['ID_code'], "target": pred[:,0]})

result.head()
result.to_csv("submission1.csv", index=False)