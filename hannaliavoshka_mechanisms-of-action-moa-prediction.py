# Get the list of all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# DATA LOADING

# 'sig_id' is a complex unique ID for each example, we will use simple numerical indexing instead of it



import pandas as pd



train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train_features.drop(['sig_id'], axis='columns', inplace=True)



train_target_full = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_target = train_target_full.copy()

train_target.drop(['sig_id'], axis='columns', inplace=True)



test_features_full = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

test_features = test_features_full.copy()

test_features.drop(['sig_id'], axis='columns', inplace=True)

# CATEGORICAL DATA PROCESSING

# There are 3 categorical features, we need to encoder them to numerical form



from sklearn.preprocessing import LabelEncoder



label_train_features = train_features.copy()

label_test_features = test_features.copy()



object_cols = ['cp_type', 'cp_time', 'cp_dose']

label_encoder = LabelEncoder()

for col in object_cols:

    label_train_features[col] = label_encoder.fit_transform(train_features[col])

    label_test_features[col] = label_encoder.transform(test_features[col])



# Check the encoded features

label_test_features.loc[0:5,['cp_type', 'cp_time', 'cp_dose']]
# DEEP LEARNING

# The number of layers and the number of neurons per layer were chosen using keras-tuner:

# https://www.tensorflow.org/tutorials/keras/keras_tuner



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Dropout



def get_model(n_inputs, n_outputs):

    model = Sequential()

    model.add(Dropout(0.2, input_shape=(n_inputs,)))

    model.add(Dense(306, input_dim=n_inputs, kernel_initializer='he_uniform', activation='sigmoid'))

    model.add(Dropout(0.5)) 

    model.add(Dense(256, activation='sigmoid'))

    model.add(Dropout(0.5))

    model.add(Dense(856, activation='sigmoid'))

    model.add(Dropout(0.5))

    model.add(Dense(n_outputs, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics = ['accuracy'])

    return model
# TRAIN THE MODEL AND GET PREDICTIONS



n_inputs, n_outputs = label_train_features.shape[1], train_target.shape[1]

model = get_model(n_inputs, n_outputs)



model.fit(label_train_features, train_target, verbose=1, epochs=100, validation_split=0.2, batch_size=32)



predictions = model.predict(label_test_features)
# SAVE PREDICTIONS FOR SUBMISSION



data = pd.DataFrame.from_records(predictions)

data.insert(0,'sig_id',test_features_full['sig_id'])

data.columns = train_target_full.columns

data.to_csv('submission.csv', index=False)