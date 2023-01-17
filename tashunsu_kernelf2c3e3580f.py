# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
# feature used in tensorflow

feature = [

    'Pclass',

    'Sex',

    'Age',

    'FamilySize',

    'Fare',

    # 'Cabin'

]



# expect result

train_output = train_data.loc[:, 'Survived'].to_numpy().astype(np.float32)
# clean data

def tf_input_array(data):

    data = data.copy()

    

    # FamilySize

    data['FamilySize'] = data['SibSp'] + data['Parch']

    

    # Unknown Age

    data['Age'].fillna(data['Age'].median(), inplace=True)

    

    # Sex to number

    s = data['Sex']

    s.replace('male', 0, inplace=True)

    s.replace('female', 1, inplace=True)

    

    ret = data.loc[:, feature].to_numpy()

    

    ret = ret.astype(np.float32)

    return ret
train_input = tf_input_array(train_data)

train_input
model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(len(feature),)),

    tf.keras.layers.Dense(4 * len(feature), activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(4 * len(feature), activation='relu'),

    tf.keras.layers.Dense(len(feature)),

    tf.keras.layers.Dense(2)

])
# loss function

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# testing 

predictions = model(train_input[:1]).numpy()

print('output', predictions)

predictions = tf.nn.softmax(predictions).numpy()

print('predictions', predictions)

print('loss', loss_fn(train_output[:1], predictions).numpy())
# compile modle

model.compile(optimizer='adam',

              loss=loss_fn,

              metrics=['accuracy'])



model.summary()
# check points setup

checkpoint_path = '.tf_checkpoints/analysis-{epoch:04d}.ckpt'



check_point_cb = tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_path,

    save_weights_only=True,

    verbose=0

)



# Save the weights using the `checkpoint_path` format

model.save_weights(checkpoint_path.format(epoch=0))



# training model

total_epochs = 10

progress_cb = tf.keras.callbacks.LambdaCallback(

    on_epoch_end=lambda _epoch, _log: print('epoch', _epoch, '/', total_epochs, '   ', end='\r')

)



model.fit(train_input, train_output, 

          epochs=total_epochs,

          callbacks=[check_point_cb, progress_cb],

          verbose=0

         )

    
# or load previous

model.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path)))



# then check performance

train_loss, train_acc = model.evaluate(train_input, train_output, verbose=2)

print(f'loss : {train_loss}, accuracy : {train_acc*100}%')
# to Probability[Died, Survived]

probability_model = tf.keras.Sequential([

    model,

    tf.keras.layers.Softmax()

])
def to_result_table(data, result):

    if not isinstance(result, np.ndarray):

        result = result.numpy()

    ret = pd.DataFrame({

        'PassengerId': data['PassengerId'],

        'Survived': np.zeros_like(data['PassengerId'])

    })

    

    ret.loc[result[:, 0] < result[:, 1], 'Survived'] = 1

    

    return ret



def performance(data, predict):

    ret = pd.merge(data, predict, on='PassengerId', suffixes=('', '_p'))



    total = len(ret)



    if total == 0:

        return 0.0



    match = np.count_nonzero(ret['Survived'] == ret['Survived_p'])



    return 100 * match / total

#

print('pref(train)',

      performance(train_data,

                  to_result_table(train_data, 

                                  probability_model(train_input))))
# predict test data set

test_input = tf_input_array(test_data)

test_result = probability_model(test_input)

result_table = to_result_table(test_data, test_result)

result_table.to_csv('/kaggle/working/submission.csv', index=False)

result_table