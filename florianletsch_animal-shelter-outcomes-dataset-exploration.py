# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# !gunzip /kaggle/input/shelter-animal-outcomes/train.csv.gz -O /kaggle
PATH_TRAIN = '/kaggle/input/shelter-animal-outcomes/train.csv.gz'
PATH_TEST = '/kaggle/input/shelter-animal-outcomes/test.csv.gz'
PATH_SAMPLE = '/kaggle/input/shelter-animal-outcomes/sample_submission.csv.gz'
animals = pd.read_csv(PATH_TRAIN, compression='gzip')
animals
animals.AnimalType.value_counts().plot(kind='bar')
animals.OutcomeType.value_counts().plot(kind='bar')
min_count = 1000
breeds = animals.Breed.fillna('Unknown')
breeds_filtered = breeds.value_counts() > min_count
breeds_filtered.values
animals[animals.AnimalType=='Dog'].Breed.fillna('Unknown').value_counts().head(30).plot(kind='bar')
animals[animals.AnimalType=='Cat'].Breed.fillna('Unknown').value_counts().head(30).plot(kind='bar')
# We consider a breed 'pure' if the breed name does not contain "Mix"
animals['PureBred'] = ~animals.Breed.str.contains('Mix', regex=False)
animals.PureBred.value_counts()
animals[animals.AnimalType == 'Cat'].PureBred.value_counts().plot(kind='bar')
animals[animals.AnimalType == 'Dog'].PureBred.value_counts().plot(kind='bar')
from datetime import timedelta
max_int = 20
time_map = {f'{n} {word}': n*delta
            for n in range(max_int+1)
            for word, delta in [
                ('day', timedelta(days=1)),
                ('days', timedelta(days=1)),
                ('week', timedelta(weeks=1)),
                ('weeks', timedelta(weeks=1)),
                ('month', timedelta(days=30)), # close enough
                ('months', timedelta(days=30)), # close enough
                ('year', timedelta(days=365)), # close enough
                ('years', timedelta(days=365)) # close enough
            ]}
time_map[np.nan] = None
time_map
assert all(age in time_map for age in animals.AgeuponOutcome.unique())
animals.AgeuponOutcome.unique()

animals['AgeInDays'] = animals.AgeuponOutcome.map(time_map).map(lambda age: age.days)
animals.AgeInDays.hist()
animals[['OutcomeType', 'AgeInDays']].groupby('OutcomeType').agg(['mean', 'median'])
animals[(animals.AnimalType=='Dog') & (animals.OutcomeType == 'Died')].AgeInDays.hist()
animals[animals.OutcomeType == 'Euthanasia'].AgeInDays.hist()
animals[animals.OutcomeType == 'Euthanasia'].OutcomeSubtype.value_counts().plot(kind='bar')
animals[animals.OutcomeType == 'Return_to_owner'].AgeInDays.hist()
animals[animals.OutcomeType == 'Transfer'].AgeInDays.hist()
animals[animals.OutcomeType == 'Transfer'].OutcomeSubtype.value_counts().plot(kind='bar')
# TODO: is 'Transfer' a bad outcome?
animals['HasBadOutcome'] = animals.OutcomeType.isin(['Euthanasia', 'Died'])
animals['HasBadOutcome'].value_counts()
animals.head()
animals.Color.unique()
animals[animals.AnimalType=='Cat'].Color.fillna('Unknown').value_counts().head(40).plot(kind='bar')
animals[animals.AnimalType=='Dog'].Color.fillna('Unknown').value_counts().head(40).plot(kind='bar')

def append_column_of_frequent_values(df, base_col_name, new_col_name, min_count, value_other):
    values = df[base_col_name]
    counts = pd.value_counts(values)
    mask_frequent = values.isin(counts[counts > min_count].index)

    # Add default 'other value' to all entries first
    df[new_col_name] = value_other
    df[new_col_name][mask_frequent] = df[mask_frequent][base_col_name]
    



# counts = pd.value_counts(values)
# mask_frequent = animals.Color.isin(counts[counts > 100].index)

# # Add column for most frequent colors
# animals['ColorCleaned'] = 'Other Color'
# animals['ColorCleaned'][mask_frequent] = animals[mask_frequent]['Color']

append_column_of_frequent_values(
    df=animals,
    base_col_name='Color',
    new_col_name='ColorCleaned',
    min_count=80,
    value_other='Other Color'
)

animals['ColorCleaned'].unique()
animals.ColorCleaned.fillna('Unknown').value_counts().head(50).plot(kind='bar')
animals['IsTabby'] = animals.Color.str.contains('Tabby')
animals['IsPoint'] = animals.Color.str.contains('Point')
animals['IsTortie'] = animals.Color.str.contains('Tortie')
animals.head()
animals.columns.values.tolist()
# multi-class categorical (n_classes > 2)
categorical_vars = [
    'AnimalType',
    'SexuponOutcome',
    'ColorCleaned'
]

# numerical vars, including binary categorical - which we consider numerical
numerical_vars = [
    'AgeInDays',
    'PureBred',
    'IsTabby',
    'IsPoint',
    'IsTortie'
]


data = pd.DataFrame()

for var in categorical_vars:
    # drop_first ? For interpretability, I prefer to keep all
    categorical_cols = pd.get_dummies(animals[var])
    data = data.join(categorical_cols, how='outer')
    
# append numerical columns
data = pd.concat(
    [data, animals[numerical_vars]],
    axis='columns'
)
    
data
for col in data.columns:
    if data[col].isnull().values.any():
        print(f'{col} has NaN value')
data['AgeInDays'] = (
    data['AgeInDays']
    .fillna(data['AgeInDays'].median())
)
index = animals['OutcomeType'] != 'Transfer'

X = data[index]
y = animals[index]['HasBadOutcome']
y.value_counts()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
# Normalize data by removing mean and scaling to unit variance
normalizer = StandardScaler()
X_norm = normalizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.20, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

value_counts = y_test.value_counts()
guessing_baseline = value_counts[False] / (value_counts[False] + value_counts[True])
print(f'Always guessing most frequent class (False) yields accuracy: {guessing_baseline}')
print(f'Model accuracy: {metrics.accuracy_score(y_test, y_pred)}')
confmat = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confmat, annot=True)
print(metrics.classification_report(y_test, y_pred))


features = zip(X.columns.values.tolist(), model.coef_[0])
features = sorted(features, key=lambda ft: abs(ft[1]), reverse=True)

pd.DataFrame({
    'feature': [f[0] for f in features],
    'weight': [f[1] for f in features]
}).plot.bar(figsize=(15,6), x='feature', y='weight')
