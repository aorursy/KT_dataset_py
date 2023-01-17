import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train['set'], test['set'] = 'train', 'test'
combined = pd.concat([train, test])
combined.isnull().sum()
# combined = combined.drop(['Cabin', 'Embarked'], axis=1)
pclass = combined.loc[combined.Fare.isnull(), 'Pclass'].values[0]
median_fare = combined.loc[combined.Pclass== pclass, 'Fare'].median()
combined.loc[combined.Fare.isnull(), 'Fare'] = median_fare
# Select everything before the . as title
combined['Title'] = combined['Name'].str.extract('([A-Za-z]+)\.', expand=True)
combined['Title'].unique()
title_reduction = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 
                   'Master': 'Master', 'Don': 'Mr', 'Rev': 'Rev',
                   'Dr': 'Dr', 'Mme': 'Miss', 'Ms': 'Miss',
                   'Major': 'Mr', 'Lady': 'Mrs', 'Sir': 'Mr',
                   'Mlle': 'Miss', 'Col': 'Mr', 'Capt': 'Mr',
                   'Countess': 'Mrs','Jonkheer': 'Mr',
                   'Dona': 'Mrs'}
combined['Title'] = combined['Title'].map(title_reduction)
combined['Title'].unique()
for title, age in combined.groupby('Title')['Age'].median().iteritems():
    print(title, age)
    combined.loc[(combined['Title']==title) & (combined['Age'].isnull()), 'Age'] = age
combined.isnull().sum()
def other_family_members_survived(dataset, label='family_survival'):
    """
    Check if other family members survived
      -> 0 other did not survive
      -> 1 at least one other family member survived
      -> 0.5 unknown if other members survived or person was alone
    
    Parameters
    ----------
    dataset : DataFrame
      The sub-dataframe containing the family
    """
    ds = dataset.copy()
    if len(dataset) == 1:
        ds[label] = 0.5
        return ds
    result = []
    for ix, row in dataset.iterrows():
        survived_fraction = dataset.drop(ix)['Survived'].mean()
        if np.isnan(survived_fraction):
            result.append(0.5)
        elif survived_fraction == 0:
            result.append(0)
        else:
            result.append(1)
    ds[label] = result
    return ds
combined['surname'] = combined['Name'].apply(lambda x: x.split(",")[0])
combined = combined.groupby(['surname', 'Fare']).apply(other_family_members_survived).reset_index(drop=True)
combined = combined.groupby(['Ticket']).apply(lambda x: other_family_members_survived(x, label='family_survival_ticket')).reset_index(drop=True)
combined.loc[combined['family_survival'] == 0.5, 'family_survival'] = combined.loc[combined['family_survival'] == 0.5, 'family_survival_ticket']
combined['family_size'] = combined['Parch'] + combined['SibSp']
combined['Sex'] = LabelEncoder().fit_transform(combined['Sex'])
combined.loc[:, 'Age'] = pd.qcut(combined['Age'], 4, labels=False)
combined.loc[:, 'Fare'] = pd.qcut(combined['Fare'], 5, labels=False)
selected = ['Pclass', 'Sex', 'Age', 'Fare', 'family_size', 'family_survival']
scaler  = StandardScaler()
scaler.fit(combined[selected])
combined[selected] = scaler.transform(combined[selected])
combined.to_parquet('titanic_family_survivabillity.parquet', index=False)
train = combined.loc[combined['set'] == 'train'].drop('set', axis=1).reset_index(drop=True)
test = combined.loc[combined['set'] == 'test'].drop(['set', 'Survived'], axis=1).reset_index(drop=True)
def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2)**2))

# test function
vec1 = np.array([3, 0])
vec2 = np.array([0, 4])

# this is the 3:4:5 triangle and therefore, it should return 5 (Long live Pythagoras)
euclidean_distance(vec1, vec2)
# A first implementation
def get_nearest_neighbor(vector, dataset, number_of_neighbors=1, ignore_cols=['Survived']):
    distances = []
    for ix, row in dataset.loc[:, ~dataset.columns.isin(ignore_cols)].iterrows():
        distance = euclidean_distance(row, vector)
        distances.append((distance, ix))
    indices = [x[1] for x in sorted(distances, key=lambda x: x[0])]
    neighbors = dataset.loc[indices[:number_of_neighbors]]
    return neighbors

# Another implementation using Pandas
def get_nearest_neighbor(vector, dataset, number_of_vectors=1, ignore_cols=['Survived'], not_count_duplicates=False):
    ds = dataset.copy()
    ds['distance'] = ds.loc[:, ~ds.columns.isin(ignore_cols)].apply(
        lambda x: euclidean_distance(x, vector), axis=1)
    if not_count_duplicates:
        distances = sorted(ds.distance.unique())[:number_of_vectors]
        return ds.loc[ds.distance <= max(distances)].drop('distance', axis=1)
    return ds.sort_values('distance', ascending=True).head(number_of_vectors).drop('distance', axis=1)
        
# test function
dataset = pd.DataFrame([
    {'a': 1, 'b': 1, 'Survived': 1},
    {'a': 2, 'b': 2, 'Survived': 1},
    {'a': 3, 'b': 3, 'Survived': 0},
    {'a': 4, 'b': 4, 'Survived': 0},
    {'a': 5, 'b': 5, 'Survived': 0},
])
vector = pd.Series({'a': 2.5, 'b': 2.5})

# should be (2,2) and (3,3) (if keeping track of duplicates)
get_nearest_neighbor(vector, dataset)
def predict(vector, dataset, number_of_neighbors=1, y='Survived'):
    neighbors = get_nearest_neighbor(vector, dataset, number_of_neighbors)
    return round(neighbors[y].mean())

# test function
print(predict(vector, dataset))
print(predict(pd.Series({'a': 4.5, 'b': 4.5}), dataset))
def predict_dataset(dataset, number_of_neighbors=1):
    ds = dataset.copy()
    def predict_row(vector, dataset):
        subset = dataset.loc[~(dataset.index==vector.name)]
        if vector.name % 100 == 0:
            print(vector.name)
        return int(predict(vector, subset, number_of_neighbors))

    ds['predicted'] = ds.loc[:, ds.columns.isin(selected)].apply(
        lambda x: predict_row(x, ds), axis=1)
    
    return ds

ds = predict_dataset(train, number_of_neighbors=10)

print('Accuracy:', sum(ds['Survived'] == ds['predicted']) / len(ds))

def predict_testset(test_dataset, train_dataset, number_of_neighbors=1):
    ds = test_dataset.copy()
    select = selected + ['Survived']
    
    def predict_row(vector, dataset):
        if vector.name % 100 == 0:
            print(vector.name)
        return int(predict(vector, dataset[select], number_of_neighbors))

    ds['Survived'] = ds.loc[:, ds.columns.isin(selected)].apply(
        lambda x: predict_row(x, train_dataset), axis=1)
    
    return ds
final_test = predict_testset(test, train, number_of_neighbors=10)
result = final_test[['PassengerId', 'Survived']].copy()
result
result.to_csv('results.csv', index=False)
