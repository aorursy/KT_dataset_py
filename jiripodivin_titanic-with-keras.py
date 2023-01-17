!pip install -q -U keras-tuner
!pip install -q -U pandas
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Concatenate, Flatten, Dropout, BatchNormalization
from tensorflow.keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import kerastuner as kt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.columns
test_df.columns
train_df = train_df.sample(train_df.shape[0])
train_df.head()
train_df[['Sex', 'PassengerId']].groupby('Sex').count()
train_df[['Pclass', 'PassengerId']].groupby('Pclass').count()
train_df[['PassengerId', 'SibSp']].groupby('SibSp').count()
train_df[['PassengerId', 'Parch']].groupby('Parch').count()
train_df['Cabin'].unique()
grouped = train_df[['PassengerId', 'Cabin']].groupby('Cabin').count()
grouped[['PassengerId']].sort_values('PassengerId')

cabins = train_df
cabins['HadCabin'] = cabins['Cabin'].isna()

cabins[['PassengerId', 'HadCabin']].groupby('HadCabin').count()
train_df[['Fare', 'Age']].median()
train_df[['Fare', 'Age']].std()
def plot_cont(variable):
    fig, ax = plt.subplots(1, 1) 
    
    fig.suptitle(variable)
    
    ax.hist(train_df[variable], bins=20)
    ax.set_xlabel(variable)
    ax.set_ylabel('Passengers')
    
plot_cont('Fare')
plot_cont('Age')

train_df[['Pclass', 'Sex', 'PassengerId']].groupby(['Pclass', 'Sex']).count()
train_df[['Pclass', 'Parch', 'PassengerId']].groupby(['Pclass', 'Parch']).count()
train_df[['SibSp', 'Pclass', 'PassengerId']].groupby(['Pclass', 'SibSp']).count()
train_df[['SibSp', 'Pclass']].groupby('Pclass').mean()
train_df[['Parch', 'Pclass']].groupby('Pclass').mean()
train_df[['Age', 'Sex']].groupby('Sex').median()
train_df[['Age', 'Sex', 'Pclass']].groupby(['Sex', 'Pclass']).median()
train_df[['Fare', 'Sex', 'Pclass']].groupby(['Sex', 'Pclass']).median()
train_df[['Fare', 'Sex', 'Pclass']].groupby(['Sex', 'Pclass']).median()
def plot_categorical(category):
    unique_vals = train_df[category].unique()
    fig, ax = plt.subplots()

    width = 0.5
    offset = 0
    survival = train_df[['Survived', category]].groupby([category], as_index=False).mean()
    
    for unique_val in unique_vals:
        offset += width*2
        ax.bar(offset, 
                survival.loc[survival[category]==unique_val]['Survived'],
                label=str(unique_val))
        
    ax.set_ylabel('Survival')
    ax.set_xticks([width*2*i for i in range(1,len(unique_vals)+1)])
    ax.set_xticklabels(unique_vals)
    ax.set_xlabel(category)
    plt.show()
    
categorical_vars = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']

for variable in categorical_vars:
    plot_categorical(variable)

train_df[['Parch', 'Pclass', 'SibSp', 'Survived']].groupby(['Pclass', 'SibSp', 'Parch']).mean()
train_df[['Sex', 'Parch', 'Pclass', 'Survived']].groupby(['Pclass','Parch','Sex']).mean()
train_df[['Sex', 'SibSp', 'Pclass', 'Survived']].groupby(['Pclass','SibSp','Sex']).mean()
train_df[['Age', 'Sex', 'Pclass', 'Survived']].groupby(['Sex', 'Pclass', 'Survived']).median()
train_df[['Fare', 'Sex', 'Pclass', 'Survived']].groupby(['Sex', 'Pclass', 'Survived']).median()
def plot_cont_by_survival(variable):
    fig, ax = plt.subplots(1, 2) 
    
    fig.suptitle(variable)
    
    ax[0].hist(train_df.loc[train_df['Survived']==1, variable], bins=20)
    ax[1].hist(train_df.loc[train_df['Survived']==0, variable], bins=20)
    ax[0].set_xlabel('Embarked')
    ax[1].set_xlabel('Survivors')
    ax[0].set_ylabel('Passengers')
    
plot_cont_by_survival('Fare')
plot_cont_by_survival('Age')

def title_extraction(name):
    #Name column values take form of FirstName, Title. SecondName
    title = name.split(',')[1].split('.')[0].lstrip()
    return title
train_df['Title'] = train_df['Name'].apply(title_extraction)
train_df['NameL'] = train_df['Name'].apply(lambda x: len(x))

test_df['Title'] = test_df['Name'].apply(title_extraction)
test_df['NameL'] = test_df['Name'].apply(lambda x: len(x))
train_df['HadCabin'] = train_df['Cabin'].isna().astype(float)
train_df['Cabin'] = train_df['Cabin'].fillna('X')
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: x[0])

test_df['HadCabin'] = test_df['Cabin'].isna().astype(float)
test_df['Cabin'] = test_df['Cabin'].fillna('X')
test_df['Cabin'] = test_df['Cabin'].apply(lambda x: x[0])
categorical_vars = ['Pclass', 'Embarked', 'SibSp', 'Parch', 'Title', 'Cabin']

train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x=='male' else 0)

for variable in categorical_vars:
    train_df = pd.concat([train_df, pd.get_dummies(train_df[variable], dummy_na=True, prefix=variable)], axis=1)


test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x=='male' else 0)

for variable in categorical_vars:
    test_df = pd.concat([test_df, pd.get_dummies(test_df[variable], dummy_na=True, prefix=variable)], axis=1)
train_df['NanAge'] = train_df['Age'].isna().astype(float)
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

test_df['NanAge'] = test_df['Age'].isna().astype(float)
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
labels = train_df['Survived'].values



train_df = train_df.reindex(columns=train_df.columns.join(test_df.columns, 'outer').array, fill_value=0)
test_df = test_df.reindex(columns=test_df.columns.join(train_df.columns, 'outer').array, fill_value=0)

train_df = train_df.drop(columns=['Name', 'Embarked', 'PassengerId', 'Survived', 'Cabin', 'Ticket']+categorical_vars)

print(train_df.columns.difference(test_df.columns))
print(test_df.columns.difference(train_df.columns))
train_df.head()
test_df.head()
for column in train_df.columns:
    if train_df[column].min() != train_df[column].max():
        train_df[column] = (train_df[column]-train_df[column].min())/(train_df[column].max()-train_df[column].min())
        
for column in train_df.columns:
    if test_df[column].min() != test_df[column].max():
        test_df[column] = (test_df[column]-test_df[column].min())/(test_df[column].max()-test_df[column].min())       

train_df.head()
test_df.head()
validation_split = 7/8
batch_size = 32

training_data = train_df.values[:int(len(train_df.values)*validation_split)]
validation_data = train_df.values[int(len(train_df.values)*validation_split):]
train_labels = labels[:int(train_df.values.shape[0]*validation_split)]
training_data = tf.data.Dataset.from_tensor_slices((training_data, train_labels/1.0)).shuffle(training_data.shape[0]).repeat().batch(batch_size)

validation_labels = labels[int(train_df.values.shape[0]*validation_split):]
validation_data = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels/1.0)).shuffle(validation_data.shape[0]).repeat().batch(batch_size)
models_performance = []
def plot_metrics(records, metric_keys, loss=True):
    fig, ax = plt.subplots()
    fig.set_size_inches((20, 8))

    for name, record in records:
        n_metric_keys = [key for key in record.keys() if any([m_key in str(key) for m_key in metric_keys])]

        for metric in n_metric_keys:
            ax.plot([i for i in range(len(record[metric]))], 
                    record[metric],
                    label=name + '-' + metric, 
                    ls='-',
                   marker='x',
                   linewidth=1)

    ax.set_ylabel('Validation - '+' / '.join(metric_keys))
    ax.set_xlabel('Epoch')
    ax.legend()
    plt.show()
    
    if loss:
        fig, ax = plt.subplots()
        fig.set_size_inches((20, 8))

        for name, record in records:    
            ax.plot([i for i in range(len(record['val_loss']))], 
                    record['val_loss'],
                    label=name, 
                    ls='-',
                   marker='o',
                   linewidth=1)

        ax.set_ylabel('Validation Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        plt.show()
def simple_model(hyper_parameters):
    
    def optimizers(optimizer, learning_rate):
        return {
            'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
            'adadelta': tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
            'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate)
        }[optimizer]
    
    model_input = Input(shape=(train_df.values.shape[1],))
    model_hidden = model_input
    
    depth = hyper_parameters.Int(
        'network_depth',
        1,
        10,
        default=1)
    
    dropout = hyper_parameters.Float(
        'dropout',
        0.0,
        0.3,
        sampling='linear')
    
    layer_size = hyper_parameters.Int(
        'layer_size',
        32,
        256,
        step=32)
    
    residual_connection = hyper_parameters.Boolean(
        'residual_connection',
        default=False)
    
    optimizer = hyper_parameters.Choice(
        'optimizer',
        ['adam', 'adadelta', 'rmsprop', 'sgd'])
    
    learning_rate = hyper_parameters.Float(
        'lr',
        1e-7,
        1e-1, 
        sampling='log')
    
    optimizer = optimizers(optimizer, learning_rate)
    
    if residual_connection:
        line = [model_hidden]
        for i in range(depth):
            if dropout > 0.0:
                model_hidden = Dropout(dropout)(model_hidden)
            if len(line) > 1:
                #layer block    
                model_hidden = Dense(
                    layer_size, 
                    activation='relu')(concat_layer)
            else:
                #layer block    
                model_hidden = Dense(
                    layer_size, 
                    activation='relu')(model_hidden)

            line = line+[model_hidden]
            concat_layer = Concatenate()(line)
            
        model_hidden = concat_layer
    
    else:
        for i in range(depth):     

            if dropout > 0.0:
                model_hidden = Dropout(dropout)(model_hidden)
            model_hidden = Dense(
                layer_size, 
                activation='relu')(model_hidden)        

    model_output = Dense(1,
                         activation='sigmoid')(model_hidden)

    model = Model(model_input,
                  model_output,
                  name='Simple'+str(depth)+'_'+str(layer_size))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizer, 
                  metrics=[tf.keras.metrics.Accuracy(),
                           tf.keras.metrics.Precision(), 
                           tf.keras.metrics.Recall()])
    
    return model
tuner = kt.BayesianOptimization(                    
                    simple_model,
                    seed=1,
                    objective='val_accuracy',
                    max_trials=70,
                    overwrite=True)

tuner.search(
    training_data,
    validation_data=validation_data,
    epochs=50,
    steps_per_epoch=30,
    validation_steps=10)

for index, best_hyperparameters in enumerate(tuner.get_best_hyperparameters(5)):
    print("Model_", index)
    for param_key in ['network_depth', 'dropout', 'layer_size', 'residual_connection', 'lr', 'optimizer']:
        print(param_key, ": ", best_hyperparameters.get(param_key))
    print("\n")
    
metrics = []
models = []

for index, best_hyperparameters in enumerate(tuner.get_best_hyperparameters(5)):
    model = simple_model(best_hyperparameters)
    metrics.append(("Model_"+str(index), model.fit(
        training_data,
        validation_data=validation_data,
        epochs=50,
        steps_per_epoch=30,
        validation_steps=30,
        verbose=0).history))
    models.append(("Model_"+str(index), model)) 
plot_metrics(metrics, ['val_accuracy'])
training_data = train_df.values
train_labels = labels
metrics = []
best_models = []

training_data = tf.data.Dataset.from_tensor_slices(
                        (training_data, train_labels/1.0)
                    ).shuffle(training_data.shape[0]).repeat().batch(batch_size)

for index in [2, 3, 4]:
    test_model = simple_model(tuner.get_best_hyperparameters(5)[index])
    best_models.append(test_model)
    metrics.append(('final_model_'+str(index),
                    test_model.fit(
                               training_data,
                               epochs=50,                        
                               steps_per_epoch=30,
                               verbose=0).history))

metric_keys = ['accuracy']

plot_metrics(
    metrics,
    metric_keys,
    loss = False)

for index, model in enumerate(best_models):
    test_df['Survived'] = np.nan_to_num(
                            np.round(
                                test_model.predict(
                                    test_df[train_df.columns.array].values)
                            )).astype(int)

    test_df[['PassengerId', 'Survived']].to_csv(
        'output' + str(index) + '.csv',
        index=False)