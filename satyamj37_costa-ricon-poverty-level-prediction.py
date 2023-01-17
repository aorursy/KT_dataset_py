import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from imblearn.over_sampling import RandomOverSampler

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

%matplotlib inline

pd.pandas.set_option("display.max_columns", None)

# pd.pandas.set_option("display.max_rows", None)
train_data = pd.read_csv('../input/income-qualification/train.csv')
train_data.head()
train_data.info()
# checking for the columns that has the null values in them

column_nan = [feature for feature in train_data.columns if train_data[feature].isnull().any() == True]

print(column_nan)
# checking for numerical features

numerical_features = [feature for feature in train_data.columns if train_data[feature].dtype != 'object']

train_data[numerical_features].head()
# checking for categorical features

train_data.select_dtypes('object').head()
print(train_data.groupby('Target')['Target'].count())

sns.countplot(train_data.Target)
# since the data seems to be baised so we will first we will do over sampling of the data so to get the balanced data

# minority specifies the sampler to resmaple those data which are less in count and 

# to reach them at the same count with the other majority class



oversampler = RandomOverSampler() #sampling_strategy='minority')
X_over, y_over = oversampler.fit_resample(train_data.drop('Target', axis=1), train_data['Target'])

X_over.shape, y_over.shape
dataset_over = X_over.merge(y_over, left_index=True, right_index=True)
train_data.shape, dataset_over.shape
# Let's again check the biasnes of data

sns.countplot(dataset_over.Target)
same_poverty = 0

no_family_head = 0



for idhogar in dataset_over['idhogar'].unique():

    if len(dataset_over[dataset_over['idhogar'] == idhogar]['Target'].unique()) == 1:

        same_poverty += 1

    if (dataset_over[dataset_over['idhogar'] == idhogar]['parentesco1'] == 0).all():

        no_family_head += 1
print('Family with the same poverty level:', same_poverty)

print('Family with the diff poverty level:', len(dataset_over['idhogar'].unique()) - same_poverty)

print('House without a Family head:', no_family_head)
dataset_over[column_nan].isnull().sum() / dataset_over.shape[0]
zero_var_col = [feature for feature in dataset_over.columns if len(dataset_over[feature].unique()) == 1]

print('columns with zero variance', zero_var_col)
# removing columns with zero variance

dataset_over.drop(['elimbasu5'], axis=1, inplace=True)
# We will be performing Feature Engineering on the copy of the original data

train_df = dataset_over.copy()
# Since dependency column decimal values so that we will convert into integer value using ceil value. 

# But before that we will have to convert the yes and no values to integer value

# print(train_df['dependency'].unique())

train_df.loc[train_df['dependency'] == 'no', 'dependency'] = 0

# print(train_df['dependency'].unique())

train_df.loc[train_df['dependency'] == 'yes', 'dependency'] = train_df[train_df['dependency'] != 'yes']['dependency'].astype('float').mean()

# print(train_df['dependency'].unique())

train_df['dependency'] = train_df['dependency'].astype('float').apply(np.ceil)

# print(train_df['dependency'].unique())

train_df['dependency'].plot.box()

plt.show()
# Since we have some outliers in the dependency column so we will treat them by updating those rows from mean(including non-zero values only) which have outliers

# train_df = train_df[train_df['dependency'] < 4]

train_df.loc[train_df['dependency']>3, 'dependency'] = np.ceil(train_df[train_df['dependency']>0]['dependency'].mean())

train_df['dependency'].plot.box()
train_df['v2a1'].fillna(0, inplace=True)

train_df[train_df['v2a1'] > 0]['v2a1'].plot.box()

plt.show()

train_df[(train_df['v2a1'] < 330000) & (train_df['v2a1'] > 0)]['v2a1'].plot.box()

plt.show()
train_df.loc[train_df['v2a1'] > 350000, 'v2a1'] = np.round(train_df[train_df['v2a1'] > 0]['v2a1'].mean())

train_df[train_df['v2a1'] > 0]['v2a1'].plot.box()
# Removing uneccessary features from the dataset



feature_remove = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin',

       'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']

train_df.drop(feature_remove, axis=1, inplace=True)
train_df.head()
for idhogar in train_df['idhogar'].unique():

    train_df.loc[train_df['idhogar'] == idhogar, 'Target'] = stats.mode(train_df[train_df['idhogar'] == idhogar]['Target']).mode[0]
column_null = [feature for feature in train_df.columns if train_df[feature].isnull().any() == True]

print(column_null)
train_df.drop(['v18q1', 'rez_esc'], axis=1, inplace=True)
train_df.head()
train_df['meaneduc'].fillna(0, inplace=True)

train_df.drop(['Id', 'idhogar'], axis=1, inplace=True)
train_df.select_dtypes('object').head()
print(train_df['edjefe'].unique())

train_df.loc[train_df['edjefe'] == 'no', 'edjefe'] = 0

train_df.loc[train_df['edjefe'] == 'yes', 'edjefe'] = train_df[train_df['edjefe'] != 'yes']['edjefe'].astype('float').mean()

print(train_df['edjefe'].unique())

train_df['edjefe'] = train_df['edjefe'].astype('float').apply(np.ceil)

sns.boxplot(train_df['edjefe'])

plt.show()

train_df.loc[train_df['edjefe'] > 15, 'edjefe'] = np.ceil(train_df[train_df['edjefe']>0]['edjefe'].mean())

sns.boxplot(train_df['edjefe'])

plt.show()
print(train_df['edjefa'].unique())

train_df.loc[train_df['edjefa'] == 'no', 'edjefa'] = 0

train_df.loc[train_df['edjefa'] == 'yes', 'edjefa'] = train_df[train_df['edjefa'] != 'yes']['edjefa'].astype('float').mean()

print(train_df['edjefa'].unique())

train_df['edjefa'] = train_df['edjefa'].astype('float').apply(np.ceil)

print(train_df['edjefa'].unique())

sns.boxplot(train_df['edjefa'])

plt.show()

train_df.loc[train_df['edjefa'] > 15, 'edjefa'] = np.ceil(train_df[train_df['edjefa']>0]['edjefa'].mean())

sns.boxplot(train_df['edjefa'])

plt.show()
sns.boxplot(train_df['meaneduc'])

plt.show()

train_df.loc[train_df['meaneduc'] > 17, 'meaneduc'] = np.ceil(train_df[train_df['meaneduc']>0]['meaneduc'].mean())

sns.boxplot(train_df['meaneduc'])

plt.show()
pca = PCA()
pca.fit(train_df)

pca.explained_variance_ratio_
X = train_df.drop(['Target'], axis=1)

y = train_df['Target'].values

pca = PCA(n_components=10)

X_pca = pca.fit_transform(X)
X_pca[:1]
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
rfc_model = RandomForestClassifier(n_estimators = 100)
rfc_model.fit(X_train, y_train)
y_pred = rfc_model.predict(X_test)
print(accuracy_score(y_test, y_pred))

print(precision_score(y_test, y_pred, average='weighted'))

print(recall_score(y_test, y_pred, average='weighted'))

print(f1_score(y_test, y_pred, average='weighted'))
from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(rfc_model, X, y, cv=10)

print(cv_score)

print('Accuracy after cross validation: ', cv_score.mean())