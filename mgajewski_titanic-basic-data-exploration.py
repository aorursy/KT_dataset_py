import pandas as pd

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

print('train', train_df.shape)
print('test', test_df.shape)
print(train_df.shape[0] + test_df.shape[0])
train_df.info()
train_df.head()
print(train_df.groupby(['Pclass'])['Survived'].value_counts())
print(train_df.groupby(['Pclass', 'Sex'])['Survived'].value_counts(normalize='True'))
describe_fields = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
print('\nTrain males:')
print(train_df[train_df.Sex == 'male'][describe_fields].describe())

print('\nTest males:')
print(test_df[test_df.Sex == 'male'][describe_fields].describe())

print('\nTrain females:')
print(train_df[train_df.Sex == 'female'][describe_fields].describe())

print('\nTest females:')
print(test_df[test_df.Sex == 'female'][describe_fields].describe())