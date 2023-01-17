%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from fastai.tabular import *
train = pd.read_csv('../input/titanic/train.csv')

train.head()
train.shape
train.isnull().sum()
test = pd.read_csv('../input/titanic/test.csv')

test.isnull().sum()
test.shape
train['Embarked'].value_counts() 
train[train['Embarked'].isnull()]
train['Sex'].loc[train['Embarked'] == 'S'].value_counts()
train['Sex'].loc[train['Embarked'] == 'C'].value_counts()
train.loc[train['Cabin'] == 'B28']
train.loc[(train['Embarked'] == 'S') & (train['Survived'] == 1) & (train['Sex'] == 'female')]
train.loc[(train['Embarked'] == 'C') & (train['Survived'] == 1) & (train['Sex'] == 'female')]
print(str(140*100 / 203) + ' chances of a female from S embarkment to survive.')

print(str(64*100 / 73) + ' chances of a female from C embarkment to survive.')
# Filling with S since it's largest

train["Embarked"] = train["Embarked"].fillna("S")



test['Fare'].fillna(test['Fare'].median(), inplace = True)



## Assigning all the null values as "N"

train['Cabin'].fillna("NA", inplace=True)

test['Cabin'].fillna("NA", inplace=True)
print(train.isnull().sum(), test.isnull().sum())
train["Title"] = pd.Series([i.split(",")[1].split(".")[0].strip() for i in train["Name"]])

train["Title"].head()
test["Title"] = pd.Series([i.split(",")[1].split(".")[0].strip() for i in test["Name"]])

test["Title"].head()
grouped = train.groupby(['Sex','Pclass', 'Title'])  
grouped.head()
grouped['Age'].median()
# apply the grouped median value on the Age NaN

train['Age'] = grouped['Age'].apply(lambda x: x.fillna(x.median()))
# Same on test

test_grouped = test.groupby(['Sex','Pclass', 'Title'])  

test_grouped['Age'].median()

test['Age'] = grouped['Age'].apply(lambda x: x.fillna(x.median()))
print(train.isnull().sum(), test.isnull().sum())
dep_var = 'Survived'

cat_names  = ['Title', 'Sex', 'Ticket', 'Cabin', 'Embarked']

cont_names  = [ 'Age', 'SibSp', 'Parch', 'Fare']

procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)



data = (TabularList.from_df(train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)

                        .split_by_idx(list(range(0,200)))

                        .label_from_df(cols = dep_var)

                        .add_test(test, label=0)

                        .databunch())
data.show_batch(rows=10)
np.random.seed(40)
learn = tabular_learner(data, layers=[180, 120], metrics=accuracy, emb_drop=0.1)
learn.lr_find()
learn.recorder.plot()
learn.fit(5,slice(1e-01))
learn.recorder.plot_losses()
test_temp = pd.read_csv('../input/titanic/test.csv')
# Predict our target value

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)



# create submission file to submit in Kaggle competition

submission = pd.DataFrame({'PassengerId': test_temp['PassengerId'] , 'Survived': labels})

submission.to_csv('submission.csv', index=False)

submission.head()
submission.shape