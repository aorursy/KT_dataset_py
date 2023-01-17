import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from numpy.random import seed
import tensorflow as tf
dftrain = pd.read_csv('../input/instagram-fake-spammer-genuine-accounts/train.csv')
dftest = pd.read_csv('../input/instagram-fake-spammer-genuine-accounts/test.csv')

df = pd.concat([dftrain, dftest], axis=0, sort=True)
df.head()
sns.countplot(x='profile pic', data=df, palette='hls', hue='fake')
plt.xticks(rotation=45)
plt.show()
sns.countplot(x='private', data=df, palette='hls', hue='fake')
plt.xticks(rotation=45)
plt.show()
# Scale Continuous Features
continuous_features = ['nums/length username', 'description length', '#posts', '#followers', '#follows']

scaler = StandardScaler()
for feature in continuous_features:
    df[feature] = df[feature].astype('float64')
    df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))

dftrain.head()
# Let's create our train test split
X_train = df[pd.notnull(df['fake'])].drop(['fake'], axis=1)
y_train = df[pd.notnull(df['fake'])]['fake']
X_test = df[pd.isnull(df['fake'])].drop(['fake'], axis=1)
model = Sequential()
model.add(Dense(11, input_dim=X_train.shape[1], activation='linear', name='input_layer'))
model.add(Dense(22, activation='linear', name='hidden_layer'))
model.add(Dropout(0.0))
model.add(Dense(1, activation='sigmoid', name='output_layer'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
training = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
val_acc = np.mean(training.history['accuracy'])
print("\n%s: %.2f%%" % ('accuracy', val_acc*100))
# summarize history for accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
