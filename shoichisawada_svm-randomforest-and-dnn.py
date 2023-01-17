# Read data

import pandas as pd

import pandas_profiling as pdp

train = pd.read_csv("../input/prudential-life-insurance-assessment/train.csv")

test = pd.read_csv("../input/prudential-life-insurance-assessment/test.csv")

print("train:", train.shape)

print("test:", test.shape)



# Check data

print(train.columns.to_numpy())

#pdp.ProfileReport(train)
# Split x and y

x = train.iloc[:,1:-1]

y = train['Response']

print(y.value_counts())
# One-hot encode categorical data 

x = pd.get_dummies(x)

x.shape
# Normalize

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

x
# Complete missing values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

x
# Preprocess test data

z = test.iloc[:,1:]

z = pd.get_dummies(z)

z = pd.DataFrame(scaler.transform(z), columns=z.columns)

z = pd.DataFrame(imputer.transform(z), columns=z.columns)
# Split train data and test data

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=0)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
# SVM (grid search)

from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit



param = {'C': [5, 10, 20], 'dual': [False], 'penalty': ['l1', 'l2']}

gscv = GridSearchCV(LinearSVC(), param, cv=4, verbose=2)

gscv.fit(x, y)



result = pd.DataFrame.from_dict(gscv.cv_results_)

result
# SVM (train)

from sklearn.svm import LinearSVC

svm_clf = LinearSVC(C=10, penalty='l1', dual=False)

svm_clf.fit(x, y)
import pickle



# Save model

filename = 'svm_clf.bin'

pickle.dump(svm_clf, open(filename, 'wb'))



# Load model

#model = pickle.load(open(filename, 'rb'))
# Random forest (grid search)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit



param = {

    "n_estimators":[50,75,100],

    "criterion":["gini","entropy"],

    "max_depth":[15,20,25],

    "random_state":[0],

}

gscv = GridSearchCV(RandomForestClassifier(), param, cv=4, verbose=2)

gscv.fit(x, y)



result = pd.DataFrame.from_dict(gscv.cv_results_)

result
# Random forest (train)

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=25)

rf_clf.fit(x, y)
import pickle



# Save model

filename = 'rf_clf.bin'

pickle.dump(rf_clf, open(filename, 'wb'))



# Load model

#model = pickle.load(open(filename, 'rb'))
# Inference

result = rf_clf.predict(z)

submission = pd.DataFrame({'Id': test['Id'].astype('int').values, 'Response': result})

submission.to_csv('submission_rf.csv', index=False)

submission
# DNN

from tensorflow import keras

def get_model():

    model = keras.Sequential([

        keras.layers.Flatten(input_shape=[x.shape[-1]]),

        keras.layers.Dense(512, activation='relu'),

        keras.layers.Dense(256, activation='relu'),

        keras.layers.Dense(128, activation='relu'),

        keras.layers.Dense(64, activation='relu'),

        keras.layers.Dense(32, activation='relu'),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(9, activation='softmax')

    ])

    model.compile(optimizer='adam', 

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

    return model
# DNN (fit and validation)

import tensorflow as tf



batch_size = 512

train_ds = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values)).shuffle(len(x_train)).batch(batch_size)

val_ds = tf.data.Dataset.from_tensor_slices((x_val.values, y_val.values)).batch(batch_size)



model = get_model()

fit = model.fit(train_ds, validation_data=val_ds, epochs=20)
# DNN (accuracy by epoch)

import matplotlib.pyplot as plt



plt.plot(fit.history['accuracy'])

plt.plot(fit.history['val_accuracy'])

plt.show()
batch_size = 512

train_ds = tf.data.Dataset.from_tensor_slices((x.values, y.values)).shuffle(len(x_train)).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((z.values)).batch(batch_size)



# Train

model = get_model()

model.fit(train_ds, epochs=5)



# Inference

result = model.predict(test_ds)

result = [pd.np.argmax(res) for res in result]

submission = pd.DataFrame({'Id': test['Id'].astype('int').values, 'Response': result})

submission.to_csv('submission_dnn.csv', index=False)

submission