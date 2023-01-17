import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, train_test_split



from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/allstate-claims-severity/train.csv')

test = pd.read_csv('../input/allstate-claims-severity/test.csv')

sample = pd.read_csv('../input/allstate-claims-severity/sample_submission.csv')
print(train.shape, test.shape, sample.shape)
train.head()
fig, ax = plt.subplots(1,2,figsize=(20,8))

sns.distplot(train['loss'],kde=False, ax=ax[0])

sns.distplot(train['loss'],hist=False, ax=ax[1])
train.dtypes.value_counts()
train.isna().any().sum()
train.describe()
train.corr()
plt.figure(figsize=(14,10))

sns.heatmap(train.corr(), annot=True)
train_correlations = train.drop(["loss"], axis=1).corr()

train_correlations = train_correlations.values.flatten()

train_correlations = train_correlations[train_correlations != 1]



test_correlations = test.corr()

test_correlations = test_correlations.values.flatten()

test_correlations = test_correlations[test_correlations != 1]



plt.figure(figsize=(20,5))

sns.distplot(train_correlations, color="Red", label="train")

sns.distplot(test_correlations, color="Green", label="test")

plt.xlabel("Correlation values found in train (except 1)")

plt.ylabel("Density")

plt.title("Are there correlations between features?"); 

plt.legend();
sns.pairplot(train.sample(frac=0.1), vars=['cont1', 'cont2', 'cont3', 'cont4', 'cont5','cont6', 'cont7'])
sns.pairplot(train.sample(frac=0.1), x_vars=['cont1', 'cont2', 'cont3', 'cont4', 'cont5','cont6', 'cont7'], y_vars=['cont8', 'cont9', 'cont10', 'cont11', 'cont12',

       'cont13', 'cont14'])
sns.pairplot(train.sample(frac=0.1), vars=['cont8', 'cont9', 'cont10', 'cont11', 'cont12',

       'cont13', 'cont14'])
sns.pairplot(train.sample(frac=0.1), x_vars=['cont8', 'cont9', 'cont10', 'cont11', 'cont12',

       'cont13', 'cont14'], y_vars=['cont1', 'cont2', 'cont3', 'cont4', 'cont5','cont6', 'cont7'])
train = train.drop(['cont1', 'cont11', 'cont10'], axis=1)

test = test.drop(['cont1', 'cont11', 'cont10'], axis=1)
fig,axes = plt.subplots(39,3,figsize=(20,180))

ax = axes.flatten()



for i in range(116):

    sns.countplot(train[f'cat{i+1}'], ax=ax[i])
cat_cols = train.select_dtypes(include='object').columns



le = LabelEncoder()



for i in cat_cols:

    test_unique = test[i].unique()

    train_unique = train[i].unique()

    labels = list(set(test_unique) | set(train_unique))

    

    le.fit(labels)

    train[i] = le.transform(train[i])

    test[i] = le.transform(test[i])
X = train.drop(['loss'], axis=1)

y = np.log(train['loss']+1)
X_train,X_val,y_train, y_val = train_test_split(X,y,test_size=0.1)



model = LGBMRegressor(n_estimators=300, learning_rate=0.1, random_state=123)

model.fit(X_train,y_train)

preds = model.predict(X_val)



print(mean_absolute_error(preds,y_val))
feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X.columns)), columns=['Value','Feature'])



fig,ax = plt.subplots(1,1,figsize=(20,30))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False), ax=ax)

plt.title('LightGBM Features')
test_predictions = model.predict(test)

sample['loss'] = np.expm1(test_predictions)

sample.to_csv('submission.csv', index=False)
plt.figure(figsize=(8,6))

sns.distplot(sample['loss'])