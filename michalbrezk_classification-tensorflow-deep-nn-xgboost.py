import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.utils import resample



%matplotlib inline

pd.options.display.max_columns = 500



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.info()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['TotalCharges'] = df['TotalCharges'].fillna(value=0)



df['tenure'] = df['tenure'].astype('float64')
df.drop('customerID', axis=1, inplace=True)
col_cat = df.select_dtypes(include='object').drop('Churn', axis=1).columns.tolist()

col_num = df.select_dtypes(exclude='object').columns.tolist()
for c in col_cat:

    print('Column {} unique values: {}'.format(c, len(df[c].unique())))
plt.figure(figsize=(20,20))

for i,c in enumerate(col_cat):

    plt.subplot(5,4,i+1)

    sns.countplot(df[c], hue=df['Churn'])

    plt.title(c)

    plt.xlabel('')
plt.figure(figsize=(20,5))

for i,c in enumerate(['tenure', 'MonthlyCharges', 'TotalCharges']):

    plt.subplot(1,3,i+1)

    sns.distplot(df[df['Churn'] == 'No'][c], kde=True, color='blue', hist=False, kde_kws=dict(linewidth=2), label='No')

    sns.distplot(df[df['Churn'] == 'Yes'][c], kde=True, color='Orange', hist=False, kde_kws=dict(linewidth=2), label='Yes')

    plt.title(c)
plt.figure(figsize=(20,5))

for i,c in enumerate(col_num):

    plt.subplot(1,4,i+1)

    sns.violinplot(x=df['Churn'], y=df[c])

    plt.title(c)
df.head()
dfT = pd.get_dummies(df, columns=col_cat)

dfT.head()
dfT['Churn'] = dfT['Churn'].map(lambda x: 1 if x == 'Yes' else 0)
plt.figure(figsize=(5, 5))

sns.countplot(dfT['Churn'])

plt.title('Imbalanced dataset, it seems ratio is 2:5 for Yes:No')

plt.show()
minority = dfT[dfT.Churn==1]

majority = dfT[dfT.Churn==0]



minority_upsample = resample(minority, replace=True, n_samples=majority.shape[0])

dfT = pd.concat([minority_upsample, majority], axis=0)

dfT = dfT.sample(frac=1).reset_index(drop=True)
plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)

sns.countplot(df['Churn'])

plt.title('Imbalanced dataset')



plt.subplot(1,2,2)

sns.countplot(dfT['Churn'])

plt.title('Balanced dataset')

plt.show()
rs = RobustScaler()

dfT['tenure'] = rs.fit_transform(dfT['tenure'].values.reshape(-1,1))

dfT['MonthlyCharges'] = rs.fit_transform(dfT['MonthlyCharges'].values.reshape(-1,1))

dfT['TotalCharges'] = rs.fit_transform(dfT['TotalCharges'].values.reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(dfT.drop('Churn', axis=1).values, dfT['Churn'].values, test_size=0.2)
xg = XGBClassifier()

xg.fit(X_train, y_train)

y_test_hat_xg = xg.predict(X_test)
print(classification_report(y_test, y_test_hat_xg))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
model = Sequential()



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.1))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.1))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.1))



model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.45))



model.add(Dense(1, activation='sigmoid'))



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1,patience=10, min_lr=0.0000000001)

early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)



model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping_cb, reduce_lr])

y_test_hat_tf = model.predict(X_test)
y_test_hat_tf2 = [1 if x > 0.5 else 0 for x in y_test_hat_tf ]
print(classification_report(y_test, y_test_hat_tf2))