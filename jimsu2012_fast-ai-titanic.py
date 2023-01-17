from fastai.tabular import *
path = Path('data')
path.mkdir(parents=True, exist_ok=True)
! cp '../input/train.csv' 'data/train.csv'
! cp '../input/test.csv' 'data/test.csv'
path.ls()
df = pd.read_csv(path/'train.csv')
print(df.info())
print(df.describe())
df.head()
df['Pclass'].hist()
df['Sex'].value_counts().plot(kind='bar')
df['Age'].hist()
plt.scatter(df['Age'], df['Survived'])
print(df['Age'].idxmax())



df = df.drop(df['Age'].idxmax())



df.reset_index(drop=True, inplace=True)
plt.scatter(df['Age'], df['Survived'])
plt.scatter(df['Fare'], df['Survived'])
df = df[df['Fare'] < 400]

df.reset_index(drop=True, inplace=True)
len(df)
plt.scatter(df['Fare'], df['Survived'])
df['Embarked'].value_counts().plot(kind='bar')
test_df = pd.read_csv(path/'test.csv')
print(test_df.info())
print(test_df.describe())
test_df.head()
test_df['Pclass'].hist()
test_df['Sex'].value_counts().plot(kind='bar')
test_df['Age'].hist()
test_df['Embarked'].value_counts().plot(kind='bar')
all_df = pd.concat([df, test_df], sort=False)

fare_med = all_df['Fare'].median()

print(fare_med)



for dfa in [df, test_df]:

    for i, cabin in enumerate(dfa['Cabin']):

        if pd.isna(cabin):

            dfa.loc[i, 'Cabin'] = 0

        else:

            dfa.loc[i, 'Cabin'] = 1



for i, fare in enumerate(test_df['Fare']):

    if pd.isna(fare):

        test_df.loc[i, 'Fare'] = fare_med



df['Title'] = df['Name'].str.split(',').str[1].str.split(' ').str[1]

test_df['Title'] = test_df['Name'].str.split(',').str[1].str.split(' ').str[1]



print(df.info())
len(df)
df.head()
df['Title'].value_counts().plot(kind='bar')
dep_var = 'Survived'

cat_names = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'Cabin']

cont_names = ['Age', 'Fare']

procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(test_df, path=path, cat_names=cat_names, cont_names=cont_names)
"""data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                    .split_by_idx(list(range(len(df)-120, len(df)-50)))

                    .label_from_df(cols=dep_var)

                    .add_test(test)

                    .databunch())"""

data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                    .split_none()

                    .label_from_df(cols=dep_var)

                    .add_test(test)

                    .databunch())
data.show_batch(rows=10)
learner = tabular_learner(data, layers=[200, 100], ps=[0.01, 0.1], emb_drop=0.2, metrics=accuracy)
learner.lr_find()
learner.recorder.plot()
learner.fit(12, slice(2e-2))
predictions, *_ = learner.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)
sub_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': labels})

sub_df.to_csv('submission.csv', index=False)