import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer 

from sklearn.impute import KNNImputer                                     # imputation of missing data

from imblearn.over_sampling import SMOTE                                  # upsampling

from catboost import CatBoostClassifier

from sklearn.metrics import f1_score
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/pet-adoption/train.csv')

test = pd.read_csv('/kaggle/input/pet-adoption/test.csv')

print(train.shape,test.shape)
train.head()
test.head()
train.dtypes
test.dtypes
train['issue_date'] = train['issue_date'].astype('datetime64')

train['listing_date'] = train['listing_date'].astype('datetime64')

test['issue_date'] = test['issue_date'].astype('datetime64')

test['listing_date'] = test['listing_date'].astype('datetime64')
train.describe(include='all')
test.describe(include='all')
train['length(m)']*=100

test['length(m)']*=100

train.rename(columns={'length(m)':'length(cm)'}, inplace=True)

test.rename(columns={'length(m)':'length(cm)'}, inplace=True)
(train['length(cm)']==0).sum()
(test['length(cm)']==0).sum()
train.loc[train['length(cm)']==0,'length(cm)'] = np.nan

test.loc[test['length(cm)']==0,'length(cm)'] = np.nan
train['ratio l/h'] = train['length(cm)']/train['height(cm)']

test['ratio l/h'] = test['length(cm)']/test['height(cm)']
train['difference'] = (train['listing_date'] - train['issue_date']).dt.days

test['difference'] = (test['listing_date'] - test['issue_date']).dt.days
train['issue_month'] = train['issue_date'].dt.month

test['issue_month'] = test['issue_date'].dt.month
train['listing_month'] = train['listing_date'].dt.month

test['listing_month'] = test['listing_date'].dt.month
train.drop(columns=['listing_date', 'issue_date'], inplace=True)

test.drop(columns=['listing_date', 'issue_date'], inplace=True)
sns.set(style="white")

corr = train.drop(columns=['pet_id']).corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr, mask=mask, cmap='seismic_r', vmax=.5, center=0, annot=True,

            square=True, linewidths=.9, cbar_kws={"shrink": .5}, fmt='.2f')

plt.show()
sns.set(style="white")

corr = test.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr, mask=mask, cmap='Spectral', vmax=.3, center=0, annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')

plt.show()
plt.figure(figsize=(10,2))

sns.boxplot(train['difference'], palette='winter')

plt.title('Train Difference')

plt.xlabel("Days")

plt.show()
plt.figure(figsize=(10,2))

sns.boxplot(test['difference'], palette='afmhot')

plt.title('Train Difference')

plt.xlabel("Days")

plt.show()
sns.countplot(x=train['condition'], hue=train['breed_category'], palette='Set1', saturation=0.89)

plt.legend(title='Breed Category', bbox_to_anchor=(1.35, 0.5), loc='right', ncol=1)

plt.xlabel("Condition")

plt.ylabel("Count")

plt.title("Value counts for Condition and Breed Category", fontsize=15, pad=20)

plt.show()
sns.countplot(x=train['condition'], hue=train['pet_category'], log=True, palette='magma', saturation=1)

plt.legend(title='Pet Category', bbox_to_anchor=(1.3, 0.5), loc='right', ncol=1)

plt.xlabel("Condition")

plt.ylabel("Count")

plt.title("Value counts for Condition and Pet Category", fontsize=15, pad=20)

plt.show()
nan_cols_train = train.isna().sum()

nan_cols_test = test.isna().sum()



print(nan_cols_train[nan_cols_train>0],'\n')

print(nan_cols_test[nan_cols_test>0])
plt.figure(figsize=(10,5))

sns.countplot(x=train['condition'].isna(), hue=train['X1'], log=True, palette='Set1', saturation=1)

plt.xlabel("Is NaN ?")

plt.ylabel("Count")

plt.title("Train Condition (Is NaN?) and X1", fontsize=15, pad=20)

plt.legend(title='X1', bbox_to_anchor=(1.12, 0.5), loc='right', ncol=1, title_fontsize=14)

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(x=test['condition'].isna(), hue=test['X1'], log=True, palette='Set1', saturation=1)

plt.xlabel("Is NaN ?")

plt.ylabel("Count")

plt.title("Test Set: Condition (Is NaN?) and X1", fontsize=15, pad=20)

plt.legend(title='X1', bbox_to_anchor=(1.12, 0.5), loc='right', ncol=1, title_fontsize=14)

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(11,5))

sns.countplot(x=train['condition'].isna(), hue=train['X2'], log=True, palette='gist_ncar', saturation=0.9, ax=ax[0])

ax[0].set_xlabel("Is NaN ?")

ax[0].set_ylabel("Count")

ax[0].set_title("Train Set: Condition (Is NaN?) and X2", fontsize=15, pad=20)

ax[0].get_legend().remove()



sns.countplot(x=test['condition'].isna(), hue=test['X2'], log=True, palette='gist_ncar', saturation=0.9, ax=ax[1])

ax[1].set_xlabel("Is NaN ?")

ax[1].set_ylabel("Count")

ax[1].set_title("Test Set: Condition (Is NaN?) and X2", fontsize=15, pad=20)

ax[1].get_legend().remove()



lines, labels = fig.axes[-1].get_legend_handles_labels()

fig.legend(lines, labels, loc = 'right', title='X2', title_fontsize=14)

fig.show()
plt.figure(figsize=(4,4))

sns.countplot(x=train['condition'].isna(), hue=train['breed_category'], palette='CMRmap')

plt.xlabel("Is NaN ?")

plt.ylabel("Count")

plt.title("Condition (Is NaN?) and Breed Category", fontsize=15, pad=20)

plt.legend(title='Breed Category', bbox_to_anchor=(1.55, 0.5), loc='right', ncol=1, title_fontsize=13)

plt.show()
plt.figure(figsize=(4,4))

sns.countplot(x=train['condition'].isna(), hue=train['pet_category'], palette="plasma", saturation=1)

plt.xlabel("Is NaN ?")

plt.ylabel("Count")

plt.title("Condition (Is NaN?) and Breed Category", fontsize=15, pad=20)

plt.legend(title='Breed Category', bbox_to_anchor=(1.55, 0.5), loc='right', ncol=1, title_fontsize=13)

plt.show()
clr_tr = set(train['color_type'].values)

clr_te = set(test['color_type'].values)

clr_tr-clr_te
train.drop(train[(train['color_type'] == 'Black Tiger') | (train['color_type'] == 'Brown Tiger')].index , inplace=True, axis=0)

train.reset_index(drop=True, inplace=True)
cols = ['color_type','X1', 'X2', 'issue_month', 'listing_month']



ohe = OneHotEncoder(sparse=False, dtype='int8')

tr = pd.DataFrame(ohe.fit_transform(train[cols]), columns=ohe.get_feature_names(cols))

te = pd.DataFrame(ohe.transform(test[cols]), columns=ohe.get_feature_names(cols))
cols.extend(['pet_id'])

X_test = pd.concat([test.drop(columns=cols), te], axis=1)



cols.extend(['breed_category', 'pet_category'])

X = pd.concat([train.drop(columns=cols), tr], axis=1)
y1 = train['breed_category'].astype('int')

y2 = train['pet_category']
X['imputed_condition'] = train['condition'].isna()

X_test['imputed_condition'] = test['condition'].isna()
X['imputed_length'] = train['length(cm)'].isna()

X_test['imputed_length'] = test['length(cm)'].isna()
knni = KNNImputer(n_neighbors=4, weights='distance')

X = pd.DataFrame(knni.fit_transform(X), columns=X.columns)

X_test = pd.DataFrame(knni.transform(X_test), columns=X.columns)
X['id_1'] = train['pet_id'].apply(lambda x: int(x[5]))

X_test['id_1'] = test['pet_id'].apply(lambda x: int(x[5]))
X['id_2'] = train['pet_id'].apply(lambda x: int(x[6]))

X_test['id_2'] = test['pet_id'].apply(lambda x: int(x[6]))
X['id_3'] = train['pet_id'].apply(lambda x: int(x[7]))

X_test['id_3'] = test['pet_id'].apply(lambda x: int(x[7]))
X['id_4'] = train['pet_id'].apply(lambda x: int(x[8]))

X_test['id_4'] = test['pet_id'].apply(lambda x: int(x[8]))
X['id_12'] = train['pet_id'].apply(lambda x: int(x[5:7]))

X_test['id_12'] = test['pet_id'].apply(lambda x: int(x[5:7]))
X['id_23'] = train['pet_id'].apply(lambda x: int(x[6:8]))

X_test['id_23'] = test['pet_id'].apply(lambda x: int(x[6:8]))
X.head()
X_test.head()
y1.value_counts()
y2.value_counts()
over_y1 = SMOTE(sampling_strategy='minority', random_state=42118231, k_neighbors=4)

X1_over, y1_over = over_y1.fit_sample(X, y1)
cb1 = CatBoostClassifier(boosting_type='Ordered', random_state=42)

cb1.fit(X1_over, y1_over, verbose=100)
{x:y for x,y in zip(cb1.feature_names_,cb1.feature_importances_)}
pred_y1 = cb1.predict(X_test)
y1_f1 = f1_score(y1, cb1.predict(X), average="weighted")

print("F1 - y1:", y1_f1)
lbbc = LabelBinarizer()

X = pd.concat([X, pd.DataFrame(lbbc.fit_transform(y1)).add_prefix('breed_')], axis=1)

X_test = pd.concat([X_test, pd.DataFrame(lbbc.transform(pred_y1)).add_prefix('breed_')], axis=1)
over_y2 = SMOTE(sampling_strategy='not majority', random_state=42118231, k_neighbors=4)

X2_over, y2_over = over_y2.fit_sample(X, y2)
cb2 = CatBoostClassifier(boosting_type='Ordered', random_state=42)

cb2.fit(X2_over, y2_over, verbose=100)
{x:y for x,y in zip(cb2.feature_names_,cb2.feature_importances_)}
pred_y2 = cb2.predict(X_test)
y2_f1 = f1_score(y2,cb2.predict(X), average="weighted")

print("F1 - y2:", y2_f1)
print("Score: ", (y1_f1+y2_f1)*50)
predicted = pd.DataFrame(data={'pet_id' : test['pet_id'].values,

                               'breed_category' : pred_y1.reshape(-1),

                               'pet_category' : pred_y2.reshape(-1)})

predicted.shape
predicted.head()
predicted.to_csv('submit.csv', index=False)