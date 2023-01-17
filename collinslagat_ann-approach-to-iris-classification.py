import pandas as pd

dataset = pd.read_csv('../input/Iris.csv')
dataset.head()
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
y = to_categorical(y)
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(32, input_dim = 4, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()
model.fit(X_train, y_train, batch_size = 64, epochs = 50, verbose = 0)
model.evaluate(X_test, y_test)
model.fit(X_train, y_train, batch_size = 64, epochs = 100, verbose = 0)
model.evaluate(X_test, y_test)