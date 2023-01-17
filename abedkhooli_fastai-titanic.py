import os
print(os.listdir("../input"))
!pip3 install git+https://github.com/fastai/fastai.git
!pip3 install git+https://github.com/pytorch/pytorch
from fastai import *
from fastai.tabular import * 

import fastai; 
fastai.show_install(1)
from sklearn.metrics import accuracy_score 
from sklearn.utils import shuffle
np.random.seed(114)
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.sample(3)
def class_age(age):
    if age <= 15: return 'child'  # was 10
    #elif age <= 21: return 'adult'
    elif age <= 60: return 'young'
    else: return 'old'
    
# can also code ticket
def class_fare(fare):
    if fare <= 15.0: return 'third'
    elif fare <= 100.0: return 'second'
    else: return 'first'
# dealing with nulls 
# 1 Fare in test_df (all_df[all_df['Fare'].isnull()]), use media = 14.55 from all_df.describe()
mask = test_df['Fare'].isnull()
test_df.loc[mask, 'Fare'] = 14.55
# 2 embarked in train_df (all_df[all_df['Embarked'].isnull()]) survived is not null, so train 
mask2 = train_df['Embarked'].isnull()
train_df.loc[mask2, 'Embarked'] = 'S'
# 263 ages, use median per sex: all_df.groupby('Sex').median()['Age'] => F 27, M: 28 
mask3 = (train_df['Age'].isnull()) & (train_df['Sex'] == 'female')
train_df.loc[mask3, 'Age'] = 27.0
mask4 = (train_df['Age'].isnull()) & (train_df['Sex'] == 'male')
train_df.loc[mask4, 'Age'] = 28.0
mask5 = (test_df['Age'].isnull()) & (test_df['Sex'] == 'female')
test_df.loc[mask5, 'Age'] = 27.0
mask6 = (test_df['Age'].isnull()) & (test_df['Sex'] == 'male')
test_df.loc[mask6, 'Age'] = 28.0
for df in [train_df, test_df]:
    df['AgeGroup'] = df['Age'].apply(lambda x: class_age(x))
    df['FareGroup'] = df['Fare'].apply(lambda x: class_fare(x))
    df['Alone'] = df['SibSp'] + df['Parch'] == 0
    df['SibCh'] = df['SibSp'] * df['Parch'] > 0
    df['Relatives'] = df['SibSp'] + df['Parch']
    
train_df.corr()
train_df = shuffle(train_df)
dep_var = 'Survived'
cat_names = ['Pclass', 'Sex', 'Alone', 'SibCh', 'Embarked'] # 'AgeGroup', 'FareGroup',
cont_names = ['Age', 'Fare', 'Relatives' ] # 'Parch', 'SibSp'
procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(train_df, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
                           #.split_by_idx(list(range(len(train_df)-225,len(train_df))))
                            .split_by_idx(valid_idx=range(len(train_df)-175,len(train_df)))
                           .label_from_df(cols=dep_var)
                           .add_test(test, label=0)
                           .databunch())
data.show_batch(3)
emb_szs={'Pclass':6,  'Alone': 4, 'Sex': 4, 'Embarked':6, 'SibCh':4}
learn = tabular_learner(data, layers=[60,40], emb_szs= emb_szs,  metrics=accuracy) 
learn.lr_find()
learn.recorder.plot()
lr = 5e-2
learn.fit_one_cycle(4, lr)
learn.validate()
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
res_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': labels})
res_df.to_csv('titanic-83.csv', index=False)
