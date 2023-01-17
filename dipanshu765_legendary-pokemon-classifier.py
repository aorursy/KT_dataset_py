import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import explained_variance_score

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout

from tensorflow.keras.callbacks import EarlyStopping
df = pd.read_csv("/kaggle/input/pokemon/pokemon.csv")
df.head(5)
df.drop(['abilities'], axis=1, inplace=True)
df.info()
print("*"*15, "Unique Values", "*"*15,"\n\n\n")

for column in df.columns:

    print(f"{column[0].upper()}{column[1:]}: {len(df[column].unique())}")
df.drop(['name','japanese_name','pokedex_number', 'classfication'], axis=1, inplace=True)
df.head(5)
plt.figure(figsize=(16,6))

sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
df.drop(['type2'], axis=1, inplace=True)
df.describe()
df
df.dropna(subset=['height_m'], inplace=True)
df.isnull().sum()
plt.figure(figsize=(30,30))

sns.heatmap(df.corr(), annot=True)
df = df.drop(['percentage_male'], axis=1)
columns_to_drop = []

for column in df.corr().columns:

    temp = float(df.corr()['is_legendary'][column])

    if temp>-0.1 and temp<0.1:

        columns_to_drop.append(column)
df = df.drop(columns_to_drop, axis=1)
df.info()
plt.figure(figsize=(10, 6))

sns.set_style('darkgrid')

sns.countplot(x='is_legendary', data=df, palette='coolwarm', saturation=1)
plt.figure(figsize=(12, 6))

sns.scatterplot(x='attack', y='defense', hue='is_legendary', data=df)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

sns.countplot(x='against_dark', hue='is_legendary', data=df, palette='OrRd_r', ax=axes[0], alpha=0.8)

sns.countplot(x='against_ghost', hue='is_legendary', data=df, palette='OrRd_r', ax=axes[1], alpha=0.8)

sns.countplot(x='against_psychic', hue='is_legendary', data=df, palette='OrRd_r', ax=axes[2], alpha=0.8)
fig, axes2 = plt.subplots(2, 2, figsize=(16, 8))

sns.distplot(df['base_total'], ax=axes2[0][0])

sns.distplot(df['sp_defense'], ax=axes2[0][1])

sns.distplot(df['sp_attack'], ax=axes2[1][0])

sns.distplot(df['speed'], ax=axes2[1][1])
plt.figure(figsize=(16, 9))

sns.countplot(y='type1', hue='is_legendary', data=df, palette='GnBu_d', alpha=0.8)
plt.figure(figsize=(16, 9))

plt.xlabel("Index")

sns.scatterplot(y='hp', x=df.index, data=df, hue='is_legendary')
df = pd.get_dummies(df, columns=['capture_rate', 'type1'], drop_first=True)
smote = SMOTE(sampling_strategy=1)
X1 = df.drop(['is_legendary'], axis=1)

y1 = df['is_legendary']
X1 = pd.DataFrame(StandardScaler().fit_transform(X1), columns=X1.columns)
X1_Train, X1_CV, y1_train, y1_cv = train_test_split(X1, y1, test_size=0.4)

X1_CV, X1_Test, y1_cv, y1_test = train_test_split(X1_CV, y1_cv, test_size=0.5)
X1_Train, y1_train = smote.fit_sample(X1_Train, y1_train)
gs_lr = GridSearchCV(

            estimator=LogisticRegression(max_iter=150),

            param_grid={'C': (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0)}

        )
gs_lr.fit(X1_Train, y1_train)
lr = gs_lr.best_estimator_
print(classification_report(y_true=y1_train, y_pred=lr.predict(X1_Train)))
print(classification_report(y_true=y1_test, y_pred=lr.predict(X1_Test)))
svc = GridSearchCV(SVC(),

                  param_grid={

                      'C': [0.1, 1, 10, 30, 100, 300, 1000],

                      'gamma':[1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]

                  }, 

                  verbose=3).fit(X1_Train, y1_train).best_estimator_
print(classification_report(y_true=y1_train, y_pred=svc.predict(X1_Train)))
print(classification_report(y_true=y1_test, y_pred=svc.predict(X1_Test)))
model = Sequential()

model.add(Dense(66, activation='relu'))

model.add(Dense(45, activation='relu'))

model.add(Dense(30, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(5, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
epochs=500
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=25)
model.fit(x=X1_Train, y=y1_train, epochs=epochs, callbacks=[early_stop], validation_data=(X1_CV, y1_cv))
losses = pd.DataFrame(model.history.history)

losses[['loss', 'val_loss']].plot()
print("                         Performance on Training Set\n\n\n")

print(classification_report(y_true=y1_train, y_pred=model.predict_classes(X1_Train)))

print("\n\n\n")

print("                         Performance on CV Set\n\n\n")

print(classification_report(y_true=y1_cv, y_pred=model.predict_classes(X1_CV)))
print(classification_report(y_true=y1_test, y_pred=model.predict_classes(X1_Test)))
rfc = GridSearchCV(RandomForestClassifier(),

                  param_grid={

                      'n_estimators': np.linspace(50, 350, 10).astype(int),

                      'min_samples_leaf': [2, 4, 6, 8, 10],

                      'max_depth': [int(x) for x in np.linspace(10, 150, 8)],

                      'max_features': ['auto', 'sqrt']

                  }

                  ).fit(X1_Train, y1_train).best_estimator_
print(classification_report(y_true=y1_train, y_pred=rfc.predict(X1_Train)))
print(classification_report(y_true=y1_test, y_pred=rfc.predict(X1_Test)))