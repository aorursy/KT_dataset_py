import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None) # Setting pandas to display a N number of columns
pd.set_option('display.max_rows', None) # Setting pandas to display a N number rows
pd.set_option('display.width', 1000) # Setting pandas dataframe display width to N

import pandas_profiling 
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
dataset = [train, test]

# Let's get a summary of our datasets

print('Entries in training set: ', len(train), '\nEntries in testing set: ',len(test))

for df in dataset:
    print(df.isna().sum())

# A combination of training and test dataset would be helpful in data analysis

train_test_comb = pd.concat([train, test], axis=0)
# Visualizing the 'Age' distribution among the passengers
sns.distplot(train_test_comb['Age'])

# Replace 'Age' column missing values with mean
for df in dataset:
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    
# Drop the 'Cabin' column
for df in dataset:
    df.drop(['Cabin'], axis=1, inplace=True)  
    
    
# Replace 'Embarked' column missing values with 'S'
for df in dataset:
    df['Embarked'].fillna('S', inplace=True)
# Create new 'Familysize' and 'Title' column

for df in dataset:
    df['Familysize'] = df['SibSp']+df['Parch']
    df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    #print(list(df['Title'].unique()))

    # Replace the titles that has less than 20 ocurrences with 'Misc'
    title_names = (df['Title'].value_counts()> 10) #this will create a true false series with title name as index
    
    df['Title'] = df['Title'].apply(lambda x: x if title_names.loc[x] == True else 'Misc')

    
# Let's do this again as we have new features
train_test_comb = pd.concat([train, test], axis=0)

print(train_test_comb['Title'].value_counts())
# Some barplots for the discrete features sex, embarked, passengerclass and title, with respect to survival

fig, ax= plt.subplots(1,2, figsize=(16,6))
sns.barplot(x = 'Sex', y = 'Survived', ax = ax[0], data=train)
sns.barplot(x = 'Embarked', y = 'Survived', ax = ax[1], data=train)

fig, ax= plt.subplots(1,2, figsize=(16,6))
sns.barplot(x = 'Pclass', y = 'Survived', ax = ax[0], data=train)
sns.barplot(x = 'Title', y = 'Survived', ax = ax[1], data=train)
# Boxplots for continuos features like age, fare and familysize

fig, ax= plt.subplots(1,3, figsize=(25,10))
sns.boxplot(x='Fare', orient='v', meanline = False, ax= ax[0], data=train)
sns.boxplot(x='Age', orient='v', meanline = False, ax= ax[1], data=train)
sns.boxplot(x='Familysize', orient='v', meanline = False, ax= ax[2], data=train)
# Histogram of age, fare and familysize by Survival

fig, ax= plt.subplots(1, 3, figsize=(25, 8))
sns.distplot( train[train['Survived']==1]['Age'] , kde=False, label='Survived', color='g', ax=ax[0])
sns.distplot( train[train['Survived']==0]['Age'] , kde=False, label='Didn\'t survive', color='r', ax=ax[0])

sns.distplot(train[train['Survived']==1]['Fare'], kde=False, label='Survived', color = 'g', ax=ax[1])
sns.distplot(train[train['Survived']==0]['Fare'], kde=False, label='Didn\'t survive', color = 'r', ax=ax[1])

sns.distplot(train[train['Survived']==1]['Familysize'], kde=False, label='Survived', color = 'g', ax=ax[2])
sns.distplot(train[train['Survived']==0]['Familysize'], kde=False, label='Didn\'t survive', color = 'r', ax=ax[2])

plt.legend(prop={'size': 18})
fig, ax = plt.subplots(1, 2,figsize=(16,6))

sns.pointplot(x="Embarked", y="Survived", hue="Sex", data=train, linestyles=["-", "--"], ax = ax[0])
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train, linestyles=["-", "--"], ax = ax[1])
for df in dataset:
    df['Fare_cat'] = pd.qcut(df['Fare'], q=4, labels=(1,2,3,4))
    df['Age_cat'] = pd.qcut(df['Age'], q=4, labels=(1,2,3,4))
    
    # lambda function to change the values of 'Familysize'
    df['Familysize'] = df['Familysize'].apply(lambda x: 'Alone' if x==0 else('Small' if x>0 and x<5 else('Medium' if x>=5 and x<7 else 'Large')))
fig, ax = plt.subplots(1, 3,figsize=(18,6) )   

sns.pointplot(x = 'Fare_cat', y = 'Survived',  data=train, ax = ax[0])
sns.pointplot(x = 'Age_cat', y = 'Survived',  data=train, ax = ax[1])
sns.pointplot(x = 'Familysize', y = 'Survived', data=train, ax = ax[2])
# Convert categorical dtypes into numerical dtypes
for df in dataset:
    # Convert category dtypes to integers
    df['Age_cat'] = df['Age_cat'].astype(np.int32)
    df['Fare_cat'] = df['Fare_cat'].astype(np.int32)
    
    # Replace string values with integer values
    df.Title.replace({'Mr':1, 'Mrs':2, 'Miss':3, 'Master':4, 'Misc':5}, inplace=True)
    df.Sex.replace({'female':0, 'male': 1}, inplace=True)
    df.Embarked.replace({'S':1, 'C':2, 'Q':3}, inplace=True)
# One-hot encoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

features = ['Age_cat', 'Fare_cat', 'Pclass', 'Sex', 'Embarked', 'Title', 'Familysize']
encoded_fearures = []

for df in dataset:
  for feature in features:
    encoded = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
    n = df[feature].nunique()
    cols = [f'{feature}_{n}' for n in range(1, n + 1)]
    encoded_df = pd.DataFrame(encoded, columns=cols)
    encoded_df.index = df.index
    encoded_fearures.append(encoded_df)

train_one = pd.concat([train, *encoded_fearures[:7]], axis=1)
test_one = pd.concat([test, *encoded_fearures[7:]], axis=1)

dataset = [train_one, test_one]
for df in dataset:
    print(df.columns)
for df in dataset:
  df.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 
           'Ticket', 'Fare', 'Embarked', 'Familysize', 'Title', 'Fare_cat', 'Age_cat' ], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

features = [x for x in train_one.columns if x!='Survived']

x = train_one[features].to_numpy()
y = train_one['Survived'].to_numpy()

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = int(0.95*len(train_one)), shuffle=False ,random_state=1400)

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

clf = RandomForestClassifier(criterion='gini', 
                        n_estimators=300,
                        max_depth=4,
                        min_samples_split=4,
                        min_samples_leaf=7,
                        max_features='auto',
                        oob_score=True,
                        random_state=1400,
                        n_jobs=-1)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_val)

cm = confusion_matrix(y_val, y_pred)
print(cm)
print(classification_report(y_val, y_pred))
test_data = test_one[features].to_numpy()

prediction_clf = clf.predict(test_data)
print(len(prediction_clf))

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': prediction_clf})
output.to_csv('/kaggle/working/my_submission.csv', index=False)
# Using Tensorflow Neural Network
import tensorflow as tf
import tensorflow.keras as keras

seed = 1400

tf.random.set_seed(seed)
my_init = keras.initializers.glorot_uniform(seed=seed)

model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(x_train.shape[1],)))
model.add(keras.layers.Dense(360, activation='selu', kernel_initializer=my_init))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(360, activation='selu', kernel_initializer=my_init))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(360, activation='selu', kernel_initializer=my_init))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss = keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, mode='max', restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=3, mode='max', min_lr=0)

model.fit(x_train, y_train, epochs = 50, batch_size = 2, callbacks=[reduce_lr, early_stopping], verbose = 1)

val_loss, val_acc = model.evaluate(x_val, y_val, verbose=1)
print('\nValidation accuracy:', val_acc)
target_col =[]

test_data = test_one[features].to_numpy()
prediction_nn = model.predict(test_data)

for i in prediction_nn:
  target_col.append(int(round(i[0])))

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': target_col})
output.to_csv('my_submission.csv', index=False)