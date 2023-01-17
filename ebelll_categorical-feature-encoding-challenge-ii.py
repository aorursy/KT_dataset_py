!pip install decision-tree-id3
!pip install scikit-learn==0.20.3
import numpy as np
import pandas as pd
import os
import string
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import xgboost as xgb
import lightgbm as lgb
from id3 import Id3Estimator
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
df = train.copy()
train.shape, test.shape
train.head()
train.info()
msno.matrix(train)
plt.show()
fig, ax = plt.subplots(1,1,figsize=(8,5))
sns.countplot(train.target, ax=ax)
fig, axes = plt.subplots(1, 5, figsize=(30,8))
for i in range(5):
    sns.countplot(f'bin_{i}', data=train, ax=axes[i])
    axes[i].set_ylim([0, 600000])
    axes[i].set_title(f'bin_{i}', fontsize=15)
fig.suptitle("Binary Feature", fontsize=20)
plt.show()
fig, axes = plt.subplots(2, 3, figsize=(30,16))
for i in range(5):
    sns.countplot(f'nom_{i}', data=train, ax=axes[i//3][i%3],
                 order=train[f'nom_{i}'].value_counts().index)
    axes[i//3][i%3].set_ylim([0, 350000])
    axes[i//3][i%3].set_title(f'nom_{i}', fontsize=15)
fig.suptitle("Nominal Feature 1 to 5", fontsize=20)
plt.show()
train[[f'nom_{i}' for i in range(5, 10)]].describe(include='O')
fig, ax = plt.subplots(2,1, figsize=(30, 10))
for i in range(7,9): 
    sns.countplot(f'nom_{i}', data= train, ax=ax[i-7],
                  order = train[f'nom_{i}'].dropna().value_counts().index)
    ax[i-7].set_ylim([0, 5500])
    ax[i-7].set_title(f'nom_{i}', fontsize=15)
    ax[i-7].set_xticks([])
fig.suptitle("Nominal Feature 7&8", fontsize=20)
plt.show()
train[[f'ord_{i}' for i in range(6)]].describe(include='all')
fig, ax = plt.subplots(1,3, figsize=(30, 8))

ord_order = [
    [1.0, 2.0, 3.0],
    ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],
    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']
]

for i in range(3): 
    sns.countplot(f'ord_{i}', hue='target', data= train, ax=ax[i],
                  order = ord_order[i]
                 )
    ax[i].set_ylim([0, 200000])
    ax[i].set_title(f'ord_{i}', fontsize=15)
fig.suptitle("Ordinal Features", fontsize=20)
plt.show()
fig, ax = plt.subplots(1,2, figsize=(24, 8))

for i in range(3, 5): 
    sns.countplot(f'ord_{i}', hue='target', data= train, ax=ax[i-3],
                  order = sorted(train[f'ord_{i}'].dropna().unique())
                 )
    ax[i-3].set_ylim([0, 75000])
    ax[i-3].set_title(f'ord_{i}', fontsize=15)
fig.suptitle("Ordinal Feature 3&4", fontsize=20)
plt.show()
def bin_encoding(df):
    bin_encoder = {'F': 0, 'T': 1, 'Y': 1, 'N': 0}
    df['bin_3'] = df['bin_3'].map(bin_encoder)
    df['bin_4'] = df['bin_4'].map(bin_encoder)
    df = df.fillna(value={'bin_0': df.bin_0.mode()[0], 'bin_1':df.bin_1.mode()[0], 'bin_2':df.bin_2.mode()[0], 'bin_3':df.bin_3.mode()[0], 
                        'bin_4':df.bin_4.mode()[0]})
    return df
ex = bin_encoding(df)
ex[['bin_0','bin_1', 'bin_2', 'bin_3', 'bin_4']].head()
def ord_encoding(df):
    map_ord1 = {'Novice':1, 
                'Contributor':2, 
                'Expert':4, 
                'Master':5, 
                'Grandmaster':6}
    
    map_ord2 = {'Freezing':1, 
                'Cold':2, 
                'Warm':3, 
                'Hot':4, 
                'Boiling Hot':5, 
                'Lava Hot':6}
    df['ord_1'] = df['ord_1'].map(map_ord1)
    df['ord_2'] = df['ord_2'].map(map_ord2)

    ord3_by_ord = df['ord_3'].map(ord, na_action='ignore')
    map_ord3 = {key:value for value,key in enumerate(sorted(df['ord_3'].dropna().unique()))}
    df['ord_3'] = df['ord_3'].map(map_ord3)
    
    ord4_by_ord = df['ord_4'].map(ord, na_action='ignore')
    map_ord4 = {key:value for value,key in enumerate(sorted(df['ord_4'].dropna().unique()))}
    df['ord_4'] = df['ord_4'].map(map_ord4)
    
    ord_5 = list(df['ord_5'].apply(lambda x: str(x)).values)
    ord_5_encoded = []
    alphabet = list(string.ascii_lowercase + string.ascii_uppercase)
    
    for val in ord_5:
        if val != 'nan':
            ord_5_encoded.append(np.log(alphabet.index(val[0]) * 10 + alphabet.index(val[1])))
        else:
            ord_5_encoded.append(np.nan)
    df['ord_5'] = ord_5_encoded
    df = df.fillna(value={'ord_0':df['ord_0'].median(), 'ord_1':df['ord_1'].median(), 
                          'ord_2':df['ord_2'].median(), 'ord_3':df['ord_3'].median(), 
                          'ord_4':df['ord_4'].median(), 'ord_5':df['ord_5'].median()})
    return df
df = ord_encoding(df)
df[[x for x in df.columns if 'ord_' in x]].head()
def nom_encoding(df):
    oh_encoder = OneHotEncoder(handle_unknown='ignore')
    
    ## fill missing values
    nom_cols = ["nom_"+str(i) for i in range(0, 10)]
    df[nom_cols] = df[nom_cols].fillna('nan')
    
    oh_enc_cols = ['nom_0', 'nom_1','nom_2','nom_3','nom_4']
    target_enc_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
    
    # One-Hot Encoding
    nom_oh = pd.get_dummies(df[oh_enc_cols])
    
    y = df['target']
    X = df.copy()
    # Target Encoding
    for index_fit, index_transform in StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(X, y):
        target_enc = TargetEncoder(cols = target_enc_cols, smoothing=0.5)
        target_enc.fit(X.iloc[index_fit,:], y.iloc[index_fit])
        df.loc[index_transform, :] = target_enc.transform(X.iloc[index_transform, :])
    
    df = pd.concat([df.drop(oh_enc_cols, axis=1), nom_oh], axis=1)
    return df
df = train.copy()
ex = nom_encoding(df)
nom_cols = ["nom_"+str(i) for i in range(5, 10)]
ex[nom_cols].head()
import datetime

def date_encoding(df):
    df = df.fillna(value={'day':8, 'month':13})
    df['sin_day'] = np.sin(2*np.pi*df['day']/8)
    df['cos_day'] = np.cos(2*np.pi*df['day']/8)
    df['sin_month'] = np.sin(2*np.pi*df['month']/13)
    df['cos_month'] = np.cos(2*np.pi*df['month']/13)
    df.drop(columns = ['day', 'month'], inplace=True)
    return df
def data_encoding(df):
    df_cpy = df.copy()
    df_cpy = bin_encoding(df_cpy)
    df_cpy = ord_encoding(df_cpy)
    df_cpy = nom_encoding(df_cpy)
    df_cpy = date_encoding(df_cpy)
    
    return df_cpy
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
train_encoded = data_encoding(train)
train_encoded.head().T
#nom_5 ~ nom_9 is object type, we need to convert to float type
train_encoded[[f'nom_{i}' for i in range(5,10)]] = train_encoded[[f'nom_{i}' for i in range(5,10)]].astype(float)
y = train_encoded['target']
X = train_encoded.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify = y, random_state=77)
id3 = Id3Estimator(max_depth=5)
id3.fit(X_train, y_train)
pred = id3.predict(X_test)
print('Test data prediction accuracy (ID3): ',accuracy_score(y_test, pred))
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred = dt.predict(X_test)
print('Test data prediction accuracy (CART): ',accuracy_score(y_test, pred))
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
print('Test data prediction accuracy (Random Forest): ',accuracy_score(y_test, pred))
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)
pred = xgb_clf.predict(X_test)
print('Test data prediction accuracy (XGBoost): ',accuracy_score(y_test, pred))
lgb_clf = lgb.LGBMClassifier()
lgb_clf.fit(X_train, y_train)
pred = lgb_clf.predict(X_test)
print('Test data prediction accuracy (LightGBM): ',accuracy_score(y_test, pred))
gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)
print('Test data prediction accuracy (NaiveBayes): ',accuracy_score(y_test, pred))
vt_clf = VotingClassifier(estimators=[
    ('id3', id3), ('dt', dt), ('rf', rf_clf), ('xgb', xgb_clf), ('lgb', lgb_clf), ('nb', gnb)])
vt_clf.fit(X_train, y_train)
pred = vt_clf.predict(X_test)
print('Test data prediction accuracy (Ensemble): ',accuracy_score(y_test, pred))