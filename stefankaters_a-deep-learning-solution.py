import numpy as np



import pandas as pd

pd.set_option('display.max_columns', None)



import seaborn as sns



import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split



from keras.utils import to_categorical

from keras.models import Sequential 

from keras.layers import Dense, Activation
df = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
df.describe()
df.head()
sns.countplot(data=df, x='blueWins')
df.shape
df.drop(columns='gameId', inplace=True)
df.isnull().sum()
df.drop(columns=df.iloc[:,20:].columns, inplace=True)

df.drop(columns='blueDeaths', inplace=True)
plt.figure(figsize=(20,20))

corr = df.corr()

sns.heatmap(corr,annot=True)
corr['blueWins'].sort_values(ascending=False)
X = df[corr.columns[1:]]
X.head()
scaler = StandardScaler()

scaler.fit(X)

X = pd.DataFrame(scaler.transform(X), columns=X.columns)
X.head()
y = df['blueWins']

y = to_categorical(y, 2)

y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = Sequential()

model.add(Dense(units=2, activation='softmax', input_dim=len(X.columns)))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.evaluate(X_test, y_test)