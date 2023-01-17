#!pip install fastai --upgrade


from fastai import *

from fastai.tabular import *

import pandas as pd
df_test = pd.read_csv('../input/test.csv')

df_train = pd.read_csv('../input/train.csv')



def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set on axis 0

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def divide_df(all_data):

    # Returns divided dfs of training and test set

    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)



df_all = concat_df(df_train, df_test)



df_train.name = 'Training Set'

df_test.name = 'Test Set'

df_all.name = 'All Set' 



dfs = [df_train, df_test]



#print(dfs[:5])

print(df_test[:5])
print('Training examples = {}'.format(df_train.shape[0]))

print('Test examples = {}'.format(df_test.shape[0]))



print('\nTraining columns:\n',df_train.columns)

print('\nTesting colums:\n',df_test.columns)
#Display missing values



def display_missing(df):    

    for col in df.columns.tolist():          

        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))

    print('\n')

    

for df in dfs:

    print('{}'.format(df.name))

    display_missing(df)



#Extract title from name

#Extract deck from the first letter in cabin number

#Fill missing values in age 



for df in [df_train, df_test]:

    df['Title'] = df['Name'].str.split(',').str[1].str.split(' ').str[1]

    df['Deck'] = df['Cabin'].str[0]



# find mean age for each Title across train and test data sets

all_df = pd.concat([df_train, df_test], sort=False)

mean_age_by_title = all_df.groupby('Title').mean()['Age']

# update missing ages

for df in [df_train,df_test]:

    for title, age in mean_age_by_title.iteritems():

        df.loc[df['Age'].isnull() & (df['Title'] == title), 'Age'] = age
print(mean_age_by_title)
#Two missing values for embarked in training set

df_all[df_all['Embarked'].isnull()]
#https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html

#Mrs Stone embarked from Southamptonn with her maid

df_all['Embarked'] = df_all['Embarked'].fillna('S')
#missing value for Fare

df_all[df_all['Fare'].isnull()]
#Median fare for passenger in class 3, Parch 0 and 0 Siblings&Spouse

med_fare = df_all.groupby(['Pclass','Parch','SibSp']).Fare.median()[3][0][0]

df_test.Fare.fillna(med_fare,inplace=True)
dep_var = 'Survived'

cat_names = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']

cont_names = ['Age', 'Fare', 'SibSp', 'Parch']

procs = [FillMissing, Categorify, Normalize]



test = TabularList.from_df(df_test, cat_names=cat_names, cont_names=cont_names, procs=procs)

data = (TabularList.from_df(df_train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_idx(list(range(0,200)))

                           #.split_by_idx(valid_idx=range(200,400))

                           .label_from_df(cols=dep_var)

                           .add_test(test, label=0)

                           .databunch())
np.random.seed(101)
learn = tabular_learner(data, layers=[600,200], metrics=accuracy, emb_drop=0.1)

learn.fit_one_cycle(20)
#learn.lr_find()
#learn.recorder.plot()
#learn.fit(15, 1e-02)
# get predictions

preds, targets = learn.get_preds()



predictions = np.argmax(preds, axis = 1)

pd.crosstab(predictions, targets)
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)
sub_df = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': labels})

sub_df.to_csv('submission.csv', index=False)
sub_df.tail()