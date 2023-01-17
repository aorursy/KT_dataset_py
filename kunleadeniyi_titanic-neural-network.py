# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import scale

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import tensorflow as tf

sns.set()        

# Any results you write to the current directory are saved as output.
raw_train_data = pd.read_csv('../input/titanic/train.csv')

raw_test_data = pd.read_csv('../input/titanic/test.csv')
test_df = raw_test_data.copy()

train_df = raw_train_data.copy()

dfs = [train_df, test_df]
for dataset in dfs:

    print(dataset.info())

    print('\n')
# getting the number of empty cells for each colums

for dataset in dfs:

    print(dataset.isna().sum())

    print('\n')
# Dealing with missing values

for df in dfs:

    df['Age'].fillna(df['Age'].median(), inplace=True)

    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    df['Embarked'].fillna('S', inplace=True) # For some reason df['Embarked'].mode() is not working
for df in dfs:

    print(df.info())

    print('\n')
# Making new insights/columns

for df in dfs:

    df['Family_size'] = df['SibSp'] + df['Parch'] + 1

    df['Is_alone'] = 0

    df['Title'] = df['Name'].map(lambda x : x.split(' ')[1])

for df in dfs:

    df['Is_alone'] = df['Family_size'].map(lambda x: 1 if x == 1 else 0)

    df['binned_age']= pd.qcut(df['Age'], q=4) #, [0,10,20,30,40,50,60,70,80,90])
train_df['binned_age'].value_counts()
train_df
sns.pairplot(train_df)



plt.show()
plt.figure(figsize=(16,17))



corrMatrix = train_df.corr()

sns.heatmap(corrMatrix, annot=True)



plt.show()
train_df.columns
to_drop = ['Age', 'PassengerId', 'Name', 'Ticket', 'Cabin']
train_df.dtypes
for df in dfs:

    sex_encoder = LabelEncoder()

    df['sex_code'] = sex_encoder.fit_transform(df['Sex'])

    

    em_encoder = LabelEncoder()

    df['embarked_code'] = em_encoder.fit_transform(df['Embarked'])



    age_encoder = LabelEncoder()

    df['binned_age_code'] = age_encoder.fit_transform(df['binned_age'])



    title_encoder = LabelEncoder()

    df['title_code'] = title_encoder.fit_transform(df['Title'])

    
for df in dfs:

    fare = np.array(df['Fare']).reshape(-1,1)

    df['scaled_fare'] = scale(fare)
train_df.columns.tolist()
final_train_df = train_df[[

 'Survived',

'PassengerId',

 'Pclass',

 'SibSp',

 'Parch',

 'scaled_fare',

 'Family_size',

 'Is_alone',

 'sex_code',

 'embarked_code',

 'binned_age_code',

 'title_code']]



final_test_df = test_df[[

'PassengerId',

 'Pclass',

 'SibSp',

 'Parch',

 'scaled_fare',

 'Family_size',

 'Is_alone',

 'sex_code',

 'embarked_code',

 'binned_age_code',

 'title_code']]
final_train_df
# Setting features and targets

features = final_train_df.drop(labels=['Survived','PassengerId'], axis=1)

target = final_train_df['Survived']
# Converting Dataframes to nd arrays for the neural network

features = features.to_numpy()

target = target.to_numpy()



# convert test data into nd array

final_test_df = final_test_df.to_numpy()
# Split into training and validation

x_train, x_test, y_train, y_test= train_test_split(features, target, random_state=42, test_size=0.1)
input_size = 10

output_size = 2

hidden_layer_size = 50



max_epochs=100

batch_size=100



model = tf.keras.Sequential([

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(output_size, activation='softmax')

])



early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)



model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(x_train, y_train,

          verbose=2,

          epochs=max_epochs,

          callbacks=[early_stopping],

          batch_size=batch_size,

          shuffle=True,

          validation_split=0.1)
loss, acc = model.evaluate(x_test, y_test, verbose=2)

print("Loss: ", loss)

print("Accuracy: ", acc)
pred=model.predict(final_test_df[:,1:])

pred=np.around(pred)
predictions=pd.DataFrame()

predictions['PassengerId'] = final_test_df[:,0]

predictions['Survived']= pred[:,1]
predictions.to_csv('Submission.csv')