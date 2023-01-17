# by Grossmend, 2018
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import time



from xgboost import XGBClassifier

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score
# load data



# train data

train_data = pd.read_csv('/kaggle/input/train.csv')



# test data

test_data = pd.read_csv('/kaggle/input/test.csv')
# check duplecated field "PassengerId"

print(any(train_data['PassengerId'].duplicated()))

print(any(test_data['PassengerId'].duplicated()))
# concat train and test data in one DataFrame

all_data = pd.concat([train_data.set_index('PassengerId'), test_data.set_index('PassengerId')], keys=['train', 'test'], axis=0, sort=False)



# show first 10 row data

all_data.head(10)
# description of columns:



# Survived - Survival (0 = No; 1 = Yes)

# Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)

# Name - Name

# Sex - Sex

# Age - Age

# SibSp - Number of Siblings/Spouses Aboard

# Parch - Number of Parents/Children Aboard

# Ticket - Ticket Number

# Fare - Passenger Fare ()

# Cabin - Cabin (Number)

# Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# show info

all_data.info()

print(pd.__version__)
# count empty values

print('Empty values:')

all_data.isnull().sum()
all_data['Counter'] = 1
# select 'Title' from field 'Name'



def title_parser(name):

    

    if not isinstance(name, str):

        return name

        

    if len(name.split()) == 1:

        return name

    

    try:

        parser_name = name.split(',')[1].split('.')[0].strip()

    except Exception as e:

        parser_name = 'error_parse'

    

    if parser_name == 'Mlle' or parser_name == 'Miss':

        parser_name = 'Miss'

    elif parser_name == 'Mme' or parser_name == 'Lady' or parser_name == 'Ms' or parser_name == 'Mrs':

        parser_name = 'Mrs'

    elif parser_name == 'Master':

        parser_name = 'Master'

    elif parser_name == 'Mr':

        parser_name = 'Mr'

    elif parser_name == 'error_parse':

        parser_name = 'error_parse'

    else:

        parser_name = 'Other'

        

    return parser_name





# processing field 'Name'

all_data['Name'] = all_data['Name'].apply(title_parser)



# unique count field 'Name' after processing

all_data['Name'].value_counts()
# normalize between 0 and 1 field 'Name'



def name_to_number(name):

    

    if not isinstance(name, str):

        return name

    

    if name == 'Mr':

        number_name = 0

    elif name == 'Miss':

        number_name = 1

    elif name == 'Mrs':

        number_name = 2

    elif name == 'Master':

        number_name = 3

    elif name == 'Other':

        number_name = 4

    else:

        number_name = -1

        

    return float(number_name)



# convert field 'Name' string to number

all_data['Name'] = all_data['Name'].apply(name_to_number)



# # alternative methods

# all_data['Name'] = all_data['Name'].replace(['Mr', 'Miss', 'Mrs', 'Master', 'Other'], [0, 1, 2, 3, 4])

# all_data['Name'] = all_data['Name'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Other': 4})



# normalize between 0 and 1 field 'Name'

scaler = MinMaxScaler()

all_data['Name'] = scaler.fit_transform(all_data[['Name']])

all_data['Name'].value_counts()
# look at survival by sex

print(all_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
# look at count survival by 'sex' and 'class'



all_data.groupby(["Pclass", "Sex"])["Survived"].value_counts()
# normalize between 0 and 1 field 'Sex'



def sex_to_number(name):

    

    if not isinstance(name, str):

        return name

    

    if name == 'male':

        number_sex = 0

    elif name == 'female':

        number_sex = 1

    else:

        number_sex = -1

        

    return float(number_sex)



# convert field 'Name' string to number

all_data['Sex'] = all_data['Sex'].apply(sex_to_number)



# normalize between 0 and 1 field 'Name'

scaler = MinMaxScaler()

all_data['Sex'] = scaler.fit_transform(all_data[['Sex']])

all_data['Sex'].value_counts()
print(np.dtype(all_data['SibSp']))

print(np.dtype(all_data['Parch']))



# create new field 'Family' from sum 'SibSp' + 'Parch'

all_data['Family'] = all_data['SibSp'].astype(int) + all_data['Parch'] + 1



# convert column to float

all_data['Family'] = all_data["Family"]



# view how influence 'Family' survival

df_family = all_data[['Family', 'Survived', 'Counter']].copy()

df_family.groupby('Family', as_index=False).agg({'Survived': 'mean', 'Counter': 'count'}).rename(columns={'Counter': 'Count'})
def family_agr(family_count):

    

    """ function group family count """

    

    if not isinstance(family_count, int):

        return family_count

    

    if family_count == 1:

        family_group = 1

    elif (family_count == 2) or (family_count == 3):

        family_group = 2

    elif family_count == 4:

        family_group = 3

    elif (family_count == 5) or (family_count == 6) or (family_count == 7):

        family_group = 4

    elif (family_count == 8) or (family_count == 11):

        family_group = 5

    else:

        family_group = 6

    

    return float(family_group)

    

    

# convert field 'Name' string to number

all_data['Family'] = all_data['Family'].apply(family_agr)

all_data.groupby('Family', as_index=False).agg({'Survived': 'mean', 'Counter': 'count'}).rename(columns={'Counter': 'Count'})
# normalize field 'Family'

scaler = MinMaxScaler()

all_data['Family'] = scaler.fit_transform(all_data[['Family']])

all_data['Family'].value_counts()
# count empty value field "Fare"

print('Count empty "Fare":', all_data['Fare'].isnull().sum())
# fill empty values mean group from filed "Pclass"

all_data['Fare'] = all_data['Fare'].fillna(all_data.groupby('Pclass')['Fare'].transform('mean'))

print('Count empty "Fare":', all_data['Fare'].isnull().sum())
# field "Fare" contains zero values

print('Zero counts in field "Fare"', all_data[all_data['Fare'] == 0].shape[0])



# fill zero values mean group from fields "Pclass" and "Sex" (slow method)

all_data['Fare'] = all_data['Fare'].replace(0, all_data.groupby('Pclass')['Fare'].transform('mean'))



print('Count zero "Fare":', all_data['Fare'][all_data['Fare']==0].count())
# view "Fare" values

plt.plot(all_data['Fare'].sort_values().reset_index(drop=True));

plt.title('"Fare" field sort values')
# convert fielf "Fare" to categorical

all_data['Fare'] = pd.cut(all_data['Fare'], bins=10, labels=False).astype('float')
# normalize field 'Fare'

scaler = MinMaxScaler()

all_data['Fare'] = scaler.fit_transform(all_data[['Fare']])

all_data['Fare'].value_counts()
print('Count empty "Age" field:', all_data['Age'].isnull().sum())

print('Percentage empty "Age" field:', round(all_data['Age'].isnull().sum() / all_data.shape[0] * 100, 2), '%')
# plot distribution "Age" by "Class"



all_data['Age'][all_data['Pclass'] == 1].plot(kind='kde');

all_data['Age'][all_data['Pclass'] == 2].plot(kind='kde');

all_data['Age'][all_data['Pclass'] == 3].plot(kind='kde');

plt.title("Distribution 'Age' by 'Class'");

plt.legend(('1st class', '2nd class','3rd class'),loc='best');
# fill empty values "Age" by "Name" (processing previos)

all_data['Age'] = all_data['Age'].fillna(all_data.groupby('Name')['Age'].transform('mean'))
# convert fielf "Fare" to categorical

all_data['Age'] = pd.cut(all_data['Age'], bins=10, labels=False).astype('float')
# normalize field 'Fare'

scaler = MinMaxScaler()

all_data['Age'] = scaler.fit_transform(all_data[['Age']])

all_data['Age'].value_counts()
# all_data['Counter'] = 1

# all_data[['Pclass', 'Fare']].groupby('Pclass', as_index=False).mean()

# all_data[['Pclass', 'Survived', 'Sex', 'Fare', 'Counter']].groupby(['Pclass', 'Survived', 'Sex'], as_index=False).agg({'Fare': np.mean, 'Counter': np.sum})



# # plt.plot(all_data['Fare'].sort_values().reset_index()['Fare'])
# fill empty values

all_data['Cabin'].fillna('Z',inplace=True)
# select 1st element string "Cabin"

if not np.issubdtype(all_data['Cabin'].dtype, np.number):

    all_data['Cabin'] = all_data['Cabin'].map(lambda x : x[0])
# look "Survived" and "Counter" by "Cabin"

cabin = all_data.groupby(['Cabin'])['Survived', 'Counter'].agg({'Survived': np.mean, 'Counter': np.sum}).sort_values(by=['Counter'], ascending=[0]).reset_index()

cabin
# string to numeric field "Cabin"

if not np.issubdtype(all_data['Cabin'].dtype, np.number):

    all_data['Cabin'] = all_data['Cabin'].map(dict(zip(cabin['Cabin'].values, cabin.index.values))).astype(float)
# normalize field 'Cabin'

scaler = MinMaxScaler()

all_data['Cabin'] = scaler.fit_transform(all_data[['Cabin']])

all_data['Cabin'].value_counts()
# add field "is alone"

all_data['isAlone'] = 0

all_data.loc[all_data['Family'] == 1, 'isAlone'] = 1
all_data.head(10)
# count_empty values

all_data['Embarked'].value_counts()
# replace empty values

all_data['Embarked'].fillna('N', inplace=True)
embarked = all_data.groupby(['Embarked'])['Survived', 'Counter'].agg({'Survived': np.mean, 'Counter': np.sum}).sort_values(by=['Counter'], ascending=[0]).reset_index()

embarked
# string to numeric field "Embarked"

if not np.issubdtype(all_data['Embarked'].dtype, np.number):

    all_data['Embarked'] = all_data['Embarked'].map(dict(zip(embarked['Embarked'].values, embarked.index.values))).astype(float)
# normalize field 'Cabin'

scaler = MinMaxScaler()

all_data['Embarked'] = scaler.fit_transform(all_data[['Embarked']])

all_data['Embarked'].value_counts()
# normalize field 'Pclass'

scaler = MinMaxScaler()

all_data['Pclass'] = scaler.fit_transform(all_data[['Pclass']].astype(float))

all_data['Pclass'].value_counts()
all_data['Ticket'] = all_data['Ticket'].apply(lambda x: len(x) if isinstance(x, str) else x)

all_data['Ticket'] = pd.cut(all_data['Fare'], bins=10, labels=False).astype('float')



scaler = MinMaxScaler()

all_data['Ticket'] = scaler.fit_transform(all_data[['Ticket']].astype(float))

all_data['Ticket'].value_counts()



all_data.groupby(['Ticket'])['Survived', 'Counter'].agg({'Survived': 'mean', 'Counter': 'sum'})
# let's look at the processing result "all_data"

all_data.head(10)
# very nice. Select fields for ML

data_for_ml = all_data[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Fare', 'Cabin', 'Embarked', 'Family', 'isAlone', 'Ticket']]

data_for_ml.head(10)
# split dataset train data and test data for ML

X_model = data_for_ml.loc['train'].drop('Survived', axis=1).select_dtypes(include=[np.number])

y_model = data_for_ml.loc['train']['Survived']



Y_finish = data_for_ml.loc['test'].drop('Survived', axis=1).select_dtypes(include=[np.number])



print('size train data:', X_model.shape)

print('size train labels:', y_model.shape)

print('size finish test data:', Y_finish.shape)
# cross-validation K blocks



from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score



model = MLPClassifier(solver='adam',

                      random_state=0,

                      max_iter=1000,

                      batch_size=10,

                      learning_rate_init=0.001,

                      alpha=0.01,

                      hidden_layer_sizes=[10],

                      activation='relu')



scores = cross_val_score(model, X_model.values, y_model.values, cv=4)

plt.plot(scores)

print('mean scores:', np.mean(scores))
# save CSV result



model.fit(X_model, y_model)

out = model.predict(Y_finish).T.astype('int')



df_out = pd.DataFrame(data=out, index=Y_finish.index).reset_index()

df_out.columns = ['PassengerId', 'Survived']

df_out.to_csv('submission_nn.csv', index=False, sep=',')
# XGBClassifier model

xgb = XGBClassifier(n_estimators=100, max_depth=3)
# cross k validation

scores = cross_val_score(xgb, X_model.values, y_model.values, cv=3)

plt.plot(scores)

print(np.round(scores, 3))
# save CSV result



xgb.fit(X_model, y_model)

out = xgb.predict(Y_finish).T.astype('int')



df_out = pd.DataFrame(data=out, index=Y_finish.index).reset_index()

df_out.columns = ['PassengerId', 'Survived']

df_out.to_csv('submission_xgb.csv', index=False, sep=',')
from keras import models

from keras import layers

from keras import regularizers

from keras import optimizers



def build_model(insh=10):

    model = models.Sequential()

    model.add(layers.Dense(8, activation='relu', input_shape=(insh,)))

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(4, activation='tanh'))

    model.add(layers.Dense(1, activation='sigmoid'))

    opt = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model
# split data train and test

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2)
# fit NN model and show accurancy

input_shape = X_train.shape[1]

DL_model = None

DL_model = build_model(insh=input_shape)



# get initial weights model

initial_weights = DL_model.get_weights()



history = DL_model.fit(X_train.values,

                       y_train.values,

                       epochs=300,

                       verbose=0,

                       batch_size=64,

                       validation_data=(X_test.values, y_test.values))
# check scores model

DL_model.evaluate(X_test.values, y_test.values)[1]
# plotting training



n = 0



# Plot training & validation accuracy values

plt.subplots(figsize=(17,4))

plt.plot(history.history['acc'][n:])

plt.plot(history.history['val_acc'][n:])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.subplots(figsize=(17,4))

plt.plot(history.history['loss'][n:])

plt.plot(history.history['val_loss'][n:])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# kross k validation



kfold = StratifiedKFold(n_splits=3)



scores = []



nn_model = None

nn_model = build_model(X_model.shape[1])



for i in range(3):

    shf = shuffle(X_model, y_model)

    for train, test in kfold.split(shf[0].reset_index(drop=True), shf[1].reset_index(drop=True)):

        nn_model.set_weights(initial_weights)

        nn_model.fit(X_model.iloc[train].values,

                     y_model.iloc[train].values,

                     epochs=300,

                     verbose=0,

                     batch_size=64)

        acc = nn_model.evaluate(X_model.iloc[test].values, y_model.iloc[test].values, verbose=0)[1]

        scores.append(acc)

        print('accuracy step ' + str(len(scores)) + ': ', acc)

print('mean:', np.mean(scores))
# save CSV result



# fit NN model and show accurancy

input_shape = X_model.shape[1]

DL_model = None

DL_model = build_model(insh=input_shape)



DL_model.set_weights(initial_weights)



DL_model.fit(X_model.values,

             y_model.values,

             epochs=300,

             verbose=0,

             batch_size=64)



# predict finish labels

out = DL_model.predict_classes(Y_finish.values)



print(out.mean())



df_out = pd.DataFrame(data=out, index=Y_finish.index).reset_index()

df_out.columns = ['PassengerId', 'Survived']

df_out.to_csv('submission_nn_keras.csv', index=False, sep=',')