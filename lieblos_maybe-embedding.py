import pandas as pd
import numpy as np
import re
import sklearn

# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
PassengerId = test['PassengerId']
full_data = [train, test]

# basis for feature engineering borrow from
# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Feature engineering steps taken from Sina
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# Create a New feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)
gp = train[['Sex','Age','FamilySize','Title']].groupby(by=['Sex','Age','FamilySize'])[['Title']].count().reset_index().rename(index=str, columns={'Title': 'sex_age_familysize_title'})
train = train.merge(gp, on=['Sex','Age','FamilySize'], how='left')
gp = train[['Fare','Pclass','IsAlone','Title']].groupby(by=['Fare','Pclass','IsAlone'])[['Title']].count().reset_index().rename(index=str, columns={'Title': 'fare_pclass_isalone_title'})
train = train.merge(gp, on=['Fare','Pclass','IsAlone'], how='left')
gp = train[['Sex','Pclass','FamilySize','Title']].groupby(by=['Sex','Pclass','FamilySize'])[['Title']].count().reset_index().rename(index=str, columns={'Title': 'sex_pclass_familysize_title'})
train = train.merge(gp, on=['Sex','Pclass','FamilySize'], how='left')
gp = train[['Parch','Embarked','Title']].groupby(by=['Parch','Embarked'])[['Title']].count().reset_index().rename(index=str, columns={'Title': 'parch_embarked_title'})
train = train.merge(gp, on=['Parch','Embarked'], how='left')
gp = train[['Has_Cabin','IsAlone', 'Sex', 'Age']].groupby(by=['Has_Cabin', 'Sex', 'IsAlone'])[['Age']].count().reset_index().rename(index=str, columns={'Age': 'hascabin_isalone_sex_age'})
train = train.merge(gp, on=['Has_Cabin','IsAlone', 'Sex'], how='left')
gp = train[['Age','IsAlone','Fare','Sex']].groupby(by=['Age','IsAlone','Fare'])[['Sex']].count().reset_index().rename(index=str, columns={'Sex': 'age_familysize_fare_sex_title'})
train = train.merge(gp, on=['Age', 'IsAlone','Fare'], how='left')
gp = train[['Sex','Age']].groupby(by=['Sex'])[['Age']].count().reset_index().rename(index=str, columns={'Age': 'sex_age'})
train = train.merge(gp, on=['Sex'], how='left')
gp = train[['Pclass','Age']].groupby(by=['Pclass'])[['Age']].count().reset_index().rename(index=str, columns={'Age': 'pclass_age'})
train = train.merge(gp, on=['Pclass'], how='left')

gp = test[['Sex','Age','FamilySize','Title']].groupby(by=['Sex','Age','FamilySize'])[['Title']].count().reset_index().rename(index=str, columns={'Title': 'sex_age_familysize_title'})
test = test.merge(gp, on=['Sex','Age','FamilySize'], how='left')
gp = test[['Fare','Pclass','IsAlone','Title']].groupby(by=['Fare','Pclass','IsAlone'])[['Title']].count().reset_index().rename(index=str, columns={'Title': 'fare_pclass_isalone_title'})
test = test.merge(gp, on=['Fare','Pclass','IsAlone'], how='left')
gp = test[['Sex','Pclass','FamilySize','Title']].groupby(by=['Sex','Pclass','FamilySize'])[['Title']].count().reset_index().rename(index=str, columns={'Title': 'sex_pclass_familysize_title'})
test = test.merge(gp, on=['Sex','Pclass','FamilySize'], how='left')
gp = test[['Parch','Embarked','Title']].groupby(by=['Parch','Embarked'])[['Title']].count().reset_index().rename(index=str, columns={'Title': 'parch_embarked_title'})
test = test.merge(gp, on=['Parch','Embarked'], how='left')
gp = test[['Has_Cabin','IsAlone', 'Sex', 'Age']].groupby(by=['Has_Cabin', 'Sex', 'IsAlone'])[['Age']].count().reset_index().rename(index=str, columns={'Age': 'hascabin_isalone_sex_age'})
test = test.merge(gp, on=['Has_Cabin','IsAlone', 'Sex'], how='left')
gp = test[['Age','IsAlone','Fare','Sex']].groupby(by=['Age','IsAlone','Fare'])[['Sex']].count().reset_index().rename(index=str, columns={'Sex': 'age_familysize_fare_sex_title'})
test = test.merge(gp, on=['Age', 'IsAlone','Fare'], how='left')
gp = test[['Sex','Age']].groupby(by=['Sex'])[['Age']].count().reset_index().rename(index=str, columns={'Age': 'sex_age'})
test = test.merge(gp, on=['Sex'], how='left')
gp = test[['Pclass','Age']].groupby(by=['Pclass'])[['Age']].count().reset_index().rename(index=str, columns={'Age': 'pclass_age'})
test = test.merge(gp, on=['Pclass'], how='left')
y_train = train['Survived'].values
train.drop(['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'Has_Cabin', 'FamilySize', 'IsAlone', 'Title', 'Survived'],1,inplace=True)
train.head()
test.drop(['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'Has_Cabin', 'FamilySize', 'IsAlone', 'Title'],1,inplace=True)
test.head()
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

max_sex_age_familysize_title = np.max([train['sex_age_familysize_title'].max(), test['sex_age_familysize_title'].max()])+1
max_fare_pclass_isalone_title = np.max([train['fare_pclass_isalone_title'].max(), test['fare_pclass_isalone_title'].max()])+1
max_sex_pclass_familysize_title = np.max([train['sex_pclass_familysize_title'].max(), test['sex_pclass_familysize_title'].max()])+1
max_parch_embarked_title = np.max([train['parch_embarked_title'].max(), test['parch_embarked_title'].max()])+1
max_hascabin_isalone_sex_age = np.max([train['hascabin_isalone_sex_age'].max(), test['hascabin_isalone_sex_age'].max()])+1
max_age_familysize_fare_sex_title = np.max([train['age_familysize_fare_sex_title'].max(), test['age_familysize_fare_sex_title'].max()])+1
max_sex_age = np.max([train['sex_age'].max(), test['sex_age'].max()])+1
max_pclass_age = np.max([train['pclass_age'].max(), test['pclass_age'].max()])+1
def get_keras_data(dataset):
    X = {
        'sex_age_familysize_title': np.array(dataset.sex_age_familysize_title),
        'fare_pclass_isalone_title': np.array(dataset.fare_pclass_isalone_title),
        'sex_pclass_familysize_title': np.array(dataset.sex_pclass_familysize_title),
        'parch_embarked_title': np.array(dataset.parch_embarked_title),
        'hascabin_isalone_sex_age': np.array(dataset.hascabin_isalone_sex_age),
        'age_familysize_fare_sex_title': np.array(dataset.age_familysize_fare_sex_title),
        'sex_age': np.array(dataset.sex_age),
        'pclass_age': np.array(dataset.pclass_age)
    }
    return X
train_df = get_keras_data(train)
test_df = get_keras_data(test)
emb_n = 50
dense_n = 1000
in_sex_age_familysize_title = Input(shape=[1], name = 'sex_age_familysize_title')
emb_sex_age_familysize_title = Embedding(max_sex_age_familysize_title, emb_n)(in_sex_age_familysize_title)
in_fare_pclass_isalone_title = Input(shape=[1], name = 'fare_pclass_isalone_title')
emb_fare_pclass_isalone_title = Embedding(max_fare_pclass_isalone_title, emb_n)(in_fare_pclass_isalone_title)
in_sex_pclass_familysize_title = Input(shape=[1], name = 'sex_pclass_familysize_title')
emb_sex_pclass_familysize_title = Embedding(max_sex_pclass_familysize_title, emb_n)(in_sex_pclass_familysize_title)
in_parch_embarked_title = Input(shape=[1], name = 'parch_embarked_title')
emb_parch_embarked_title = Embedding(max_parch_embarked_title, emb_n)(in_parch_embarked_title)
in_hascabin_isalone_sex_age = Input(shape=[1], name = 'hascabin_isalone_sex_age')
emb_hascabin_isalone_sex_age = Embedding(max_hascabin_isalone_sex_age, emb_n)(in_hascabin_isalone_sex_age) 
in_age_familysize_fare_sex_title = Input(shape=[1], name = 'age_familysize_fare_sex_title')
emb_age_familysize_fare_sex_title = Embedding(max_age_familysize_fare_sex_title, emb_n)(in_age_familysize_fare_sex_title) 
in_pclass_age = Input(shape=[1], name = 'pclass_age')
emb_pclass_age = Embedding(max_pclass_age, emb_n)(in_pclass_age) 
in_sex_age = Input(shape=[1], name = 'sex_age')
emb_sex_age = Embedding(max_sex_age, emb_n)(in_sex_age)
fe = concatenate([(emb_sex_age_familysize_title), (emb_fare_pclass_isalone_title), (emb_sex_pclass_familysize_title),
                  (emb_parch_embarked_title), (emb_hascabin_isalone_sex_age), (emb_age_familysize_fare_sex_title),
                  (emb_sex_age), (emb_pclass_age)])
s_dout = SpatialDropout1D(0.2)(fe)
x = Flatten()(s_dout)
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[in_sex_age_familysize_title,
                      in_fare_pclass_isalone_title,
                      in_sex_pclass_familysize_title,
                      in_parch_embarked_title,
                      in_age_familysize_fare_sex_title,
                      in_hascabin_isalone_sex_age,
                      in_pclass_age,
                      in_sex_age], outputs=outp)
batch_size = 150
epochs = 10
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(train_df) / batch_size) * epochs
lr_init, lr_fin = 0.001, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=0.001, decay=lr_decay)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])

model.summary()
model.fit(train_df, y_train, batch_size=batch_size, epochs=10, shuffle=True)
model.save_weights('dl_support.h5')
predictions = model.predict(test_df, batch_size=batch_size)
sub = pd.DataFrame()
sub['PassengerId'] = PassengerId.astype('int')
sub['Survived'] = model.predict(test_df, batch_size=batch_size)
sub['Survived'] = sub['Survived'].apply(round).apply(int)
sub.to_csv('sub.csv',index=False)