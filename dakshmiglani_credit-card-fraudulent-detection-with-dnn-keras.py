import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import keras
df = pd.read_csv('../input/creditcard.csv')

df.head(1)
df['Class'].unique() # 0 = no fraud, 1 = fraudulent
X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
clf = Sequential([

    Dense(units=16, kernel_initializer='uniform', input_dim=30, activation='relu'),

    Dense(units=18, kernel_initializer='uniform', activation='relu'),

    Dropout(0.25),

    Dense(20, kernel_initializer='uniform', activation='relu'),

    Dense(24, kernel_initializer='uniform', activation='relu'),

    Dense(1, kernel_initializer='uniform', activation='sigmoid')

])
clf.summary()
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
clf.fit(X_train, Y_train, batch_size=15, epochs=2)
score = clf.evaluate(X_test, Y_test, batch_size=128)

print('\nAnd the Score is ', score[1] * 100, '%')