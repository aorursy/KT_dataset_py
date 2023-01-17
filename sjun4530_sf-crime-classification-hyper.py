import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')



from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(list(train["Address"]) + list(test["Address"]))



haha = tokenizer.texts_to_sequences(train["Address"])

haha2 = tokenizer.texts_to_sequences(test["Address"])



from keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(haha, maxlen=7)

padded2 = pad_sequences(haha2, maxlen=7)



from keras import Sequential

from keras.layers import Flatten, Embedding, Dense 

model = Sequential()

model.add(Embedding(2201, 1, input_length=7))

model.add(Flatten())

model.add(Dense(39, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])



from sklearn.preprocessing import LabelEncoder

le2 = LabelEncoder()

y= le2.fit_transform(train['Category'])

model.fit(padded, y, epochs = 10 , batch_size = 2048)



preds = model.predict(padded)

preds2 = model.predict(padded2)



from sklearn.decomposition import PCA



pca1 = PCA(n_components=2)

X_low = pca1.fit_transform(preds)

X_low2 = pca1.transform(preds2)



train['Date'] = train['Dates'].dt.date

train['n_days'] = (train['Date'] - train['Date'].min()).apply(lambda x: x.days)

train['Day'] = train['Dates'].dt.day

train['DayOfWeek'] = train['Dates'].dt.weekday

train['Month'] = train['Dates'].dt.month

train['Year'] = train['Dates'].dt.year

train['Hour'] = train['Dates'].dt.hour

train['Minute'] = train['Dates'].dt.minute

train['Block'] = train['Address'].str.contains('block', case=False)

train['ST'] = train['Address'].str.contains('ST', case=False)

train["X_Y"] = train["X"] - train["Y"]

train["XY"] = train["X"] + train["Y"]



test['Date'] = test['Dates'].dt.date

test['n_days'] = (test['Date'] - test['Date'].min()).apply(lambda x: x.days)

test['Day'] = test['Dates'].dt.day

test['DayOfWeek'] = test['Dates'].dt.weekday

test['Month'] = test['Dates'].dt.month

test['Year'] = test['Dates'].dt.year

test['Hour'] = test['Dates'].dt.hour

test['Minute'] = test['Dates'].dt.minute

test['Block'] = test['Address'].str.contains('block', case=False)

test['ST'] = test['Address'].str.contains('ST', case=False)

test["X_Y"] = test["X"] - test["Y"]

test["XY"] = test["X"] + test["Y"]



from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()

train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])

test['PdDistrict'] = le1.transform(test['PdDistrict'])



le2 = LabelEncoder()

y= le2.fit_transform(train['Category'])



train = pd.concat([train, pd.DataFrame(X_low)], 1)

test = pd.concat([test, pd.DataFrame(X_low2)], 1)



le3 = LabelEncoder()

le3.fit(list(train['Address']) + list(test['Address']))

train['Address'] = le3.transform(train['Address'])

test['Address'] = le3.transform(test['Address'])



train.drop(['Dates','Date','Descript','Resolution', 'Category'], 1, inplace=True)

test.drop(['Dates','Date',], 1, inplace=True)



train.to_csv("train.csv", index=False)

test.to_csv("test.csv", index=False)



from lightgbm import LGBMClassifier

hyper = {'colsample_bytree': 0.625,

 'is_unbalance': False,

 'learning_rate': 0.025,

 'min_child_samples': 105,

 'num_class': 39,

 'num_leaves': 233,

 'objective': 'multiclass',

 'reg_alpha': 0.4000134592012641,

 'reg_lambda': 0.5082596745249518,

 'subsample': 0.9338693244190213,

 'subsample_for_bin': 140000,

 'n_estimators': 600}

model = LGBMClassifier(**hyper)

model.fit(train, y, categorical_feature=["PdDistrict", "DayOfWeek"])

preds = model.predict_proba(test)

submission = pd.DataFrame(preds, columns=le2.inverse_transform(np.linspace(0, 38, 39, dtype='int16')), index=test.index)

submission.to_csv('LGBM_final.csv', index_label='Id')