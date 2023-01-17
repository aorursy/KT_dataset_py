from fastai.tabular import *
path = Path('../input/titanic')

w_path = Path('/kaggle/working')

path.ls()
train = pd.read_csv(path/'train.csv')

kaggle_test_df = pd.read_csv(path/'test.csv')

len(train), len(kaggle_test_df)

train.info()
train.describe()
train.head()
plt.hist(train.Pclass)

train.Age.hist()
plt.scatter(train['Age'], train['Survived'])
train['Age'].idxmax()
train = train.drop(train['Age'].idxmax())

train.reset_index(drop=True, inplace=True)
plt.scatter(train['Age'], train['Survived'])
train = train[train['Fare'] < 400]

train.reset_index(drop=True, inplace=True)
plt.scatter(train.Fare, train.Survived)
train.Fare.describe()
df_all = pd.concat([train, kaggle_test_df], sort=False)

len(df_all)
fare_med = df_all['Fare'].median()



for i, fare in enumerate(kaggle_test_df['Fare']):

  if pd.isna(fare):

    kaggle_test_df.loc[i, 'Fare'] = fare_med

age_med = df_all['Age'].median()



for dfa in [train, kaggle_test_df]:

  for i, age in enumerate(dfa['Age']):

    if pd.isna(age):

      dfa.loc[i, 'Age'] = age_med
kaggle_test_df.isna().sum(),train.isna().sum()
for dfa in [train, kaggle_test_df]:

  for i, cabin in enumerate(dfa['Cabin']):

    if pd.isna(cabin):

      dfa.loc[i, 'Cabin'] = 0

    else:

      dfa.loc[i, 'Cabin'] = 1

  
train['Title'] =  train['Name'].str.split(', ').str[1].str.split('.').str[0]

kaggle_test_df['Title'] =  kaggle_test_df['Name'].str.split(', ').str[1].str.split('.').str[0]
train.columns
def make_fares(df):

    fare_intervals = pd.qcut(df['Fare'], 4, labels =['fare_0-8', 'fare_8-14', 'fare_14-31', 'fare_31-263'])

    fares = pd.get_dummies(fare_intervals, dtype=np.int64)

    df.drop(['Fare'], inplace=True, axis=1)

    return pd.concat([df, fares], axis=1)
train = make_fares(train)

kaggle_test_df = make_fares(kaggle_test_df)
def make_rich_fem(df):

    df['rich_fem'] = 0

    index = (df['Sex'] == 'female') & ((df['fare_31-263'] == 1) | df['Pclass'] == 1)

    df.loc[index, 'rich_fem'] = 1

    return df
train = make_rich_fem(train)

kaggle_test_df = make_rich_fem(kaggle_test_df)
plt.matshow(train.corr())

plt.show()
corr = train.corr()

corr.style.background_gradient(cmap='coolwarm')
train.head()
kaggle_test_df.head()
train.drop(['Name', 'Ticket'], inplace=True, axis=1)

kaggle_test_df.drop(['Name', 'Ticket'], inplace=True, axis=1)
train.head()

train.info()
dep_var = 'Survived'

cat_names = ['Sex', 'Pclass', 'Cabin', 'SibSp', 'Parch', 'Embarked', 'Title', 'fare_0-8', 'fare_8-14', 'fare_14-31','fare_31-263', 'rich_fem']

cont_names = ['Age']

procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(train.iloc[600:800].copy(), path=w_path, cat_names=cat_names, cont_names=cont_names)
test = TabularList.from_df(kaggle_test_df, path=w_path, cat_names=cat_names, cont_names=cont_names)
data = (TabularList.from_df(train, path=w_path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           #.split_by_idx(list(range(600,800)))

                           .split_none()

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch())
data.show_batch(rows=10)
learn.destroy()
learn = tabular_learner(data, layers=[200,100], ps=[0.01, 0.1], emb_drop=0.2, metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit(2, 1e-2)
learn.recorder.plot_losses()
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)





sub_df = pd.DataFrame({'PassengerId': kaggle_test_df['PassengerId'], 'Survived': labels})

sub_df.to_csv(w_path/'submission_6_rfem.csv', index=False)