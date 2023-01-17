import warnings

warnings.filterwarnings("ignore")



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



import keras

from keras.models import Sequential

from keras.layers import Dense
df = pd.read_csv('Churn_Modelling.csv')
df.head(3)
def create_enc(df, columns):

    '''

    The following will create encoded columns based on a list of columns as an argument. The original column

    will stay intact.  Ex: create_enc(df, ['sport', 'nationality'])

    '''

    for col in columns:

        df[col+'_enc'] = df[col]

        encoder = LabelEncoder()

        encoder.fit(df[col])

        df[col+'_enc'] = encoder.transform(df[col])

    return df

df = create_enc(df, ['Geography', 'Gender'])

df = df[['CreditScore', 'Geography_enc',

       'Gender_enc', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',

       'IsActiveMember', 'EstimatedSalary', 'Exited']]
df.head(3)
X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values
X.shape, y.shape
X[:5]
y[:5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
cf = Sequential()

cf.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=10))

cf.add(Dense(output_dim=6, init='uniform', activation='relu'))

cf.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
cf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cf.fit(X_train, y_train, nb_epoch=50, batch_size=20)
scores = cf.evaluate(X, y)

print('%s: %.2f%%' % (cf.metrics_names[1], scores[1]*100))
y_pred = cf.predict(X_test)

y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)

cm