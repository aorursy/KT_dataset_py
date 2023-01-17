from fastai import *

from fastai.tabular import *
train_df = pd.read_csv('../input/train.csv')

train_df.info()
test_df = pd.read_csv('../input/test.csv')

test_df.info()
for df in [train_df, test_df]:

    df['Title'] = df['Name'].str.split(',').str[1].str.split(' ').str[1]

    df['Deck'] = df['Cabin'].str[0]



# find mean age for each Title across train and test data sets

all_df = pd.concat([train_df, test_df], sort=False)

mean_age_by_title = all_df.groupby('Title').mean()['Age']

mean_fare_by_deck = all_df.groupby('Deck').mean()['Fare']

mean_fare = all_df.mean()['Fare']

# update missing ages

for df in [train_df, test_df]:

    for title, age in mean_age_by_title.iteritems():

        df.loc[df['Age'].isnull() & (df['Title'] == title), 'Age'] = age

        

    for deck, fare in mean_fare_by_deck.iteritems():

        df.loc[df['Fare'].isnull() & (df['Deck'] == deck), 'Fare'] = fare

        

    df.loc[df['Fare'].isnull() & (df['Deck'].isnull()), 'Fare'] = mean_fare        
dependency_var = 'Survived'

cat_names = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'SibSp', 'Parch']

cont_names = ['Age', 'Fare']

procs = [FillMissing, Categorify, Normalize]



test = TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names, procs=procs)

data = (TabularList.from_df(train_df, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_idx(list(range(0,200)))

                           .label_from_df(cols=dependency_var)

                           .add_test(test, label=0)

                           .databunch())
data.show_batch(rows=10)
np.random.seed(99)

learn = tabular_learner(data, layers=[400,200], metrics=accuracy)

learn.fit(5)
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)
sub_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': labels})

sub_df.to_csv('submission.csv', index=False)
sub_df.tail()