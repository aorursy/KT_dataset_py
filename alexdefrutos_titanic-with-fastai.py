from fastai.tabular import *

import seaborn as sns

path=Path('/kaggle/input/')
train_data_original = pd.read_csv(path/'train.csv', low_memory=False, 

                                 )#parse_dates=["date_account_created"])
test_data_original = pd.read_csv(path/'test.csv', low_memory=False, 

                                )#parse_dates=["date_account_created"])



#test_data_original.drop('PassengerId',axis=1,inplace=True);
train_data_original.head()



test_data_original.head()
train_data_original.isnull().sum().sort_index()/len(train_data_original)
test_data_original.isnull().sum().sort_index()/len(test_data_original)
filler=test_data_original.Fare.median()
test_data_original.Fare=test_data_original.Fare.fillna(filler)
for df in [train_data_original, test_data_original]:

    df['Title'] = df['Name'].str.split(',').str[1].str.split(' ').str[1]

    df['Deck'] = df['Cabin'].str[0]



# find mean age for each Title across train and test data sets

all_df = pd.concat([train_data_original, test_data_original], sort=False)

mean_age_by_title = all_df.groupby('Title').mean()['Age']

# update missing ages



for df in [train_data_original, test_data_original]:

    for title, age in mean_age_by_title.iteritems():

        df.loc[df['Age'].isnull() & (df['Title'] == title), 'Age'] = age
dep_var = 'Survived'

cat_names = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']

cont_names = ['Age', 'Fare', 'SibSp', 'Parch']

procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(test_data_original, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(train_data_original, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           #.split_by_idx(list(range(1200,1460)))

                           .split_by_idx(list(range(0,200)))         

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch())





data.show_batch(rows=10)
learn = tabular_learner(data, layers=[600,200], metrics=accuracy, emb_drop=0.1)
#learn.lr_find()

#learn.recorder.plot()
learn.fit_one_cycle(10, 1e-3)
learn.save('mini_train')

preds,__=learn.get_preds(ds_type=DatasetType.Test)
preds = np.argmax(preds, 1)
test_data_original = pd.read_csv(path/'test.csv', low_memory=False, 

                                )#parse_dates=["date_account_created"])
submission_pd = pd.DataFrame({'PassengerId':test_data_original['PassengerId'],'Survived':preds})
submission_pd.head()
submission_pd.to_csv('submission_DL_10082019',index=False)