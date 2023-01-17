from pylab import rcParams

rcParams['figure.figsize'] = 30, 5
import numpy   as np

import pandas  as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.utils import np_utils

from sklearn.metrics import confusion_matrix,classification_report
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

df.head()
columns = ['age', 'anaemia', 'diabetes',

       'ejection_fraction', 'high_blood_pressure',

       'serum_creatinine', 'serum_sodium', 'sex', 'smoking',

       'DEATH_EVENT']
def plot_data(name):

    result = df[[name]].value_counts()

    result.plot(kind="bar")
for item in columns:

    plot_data(item)

    plt.legend()

    plt.figure()
rel_with_target = df.corr()[['DEATH_EVENT']].sort_values(['DEATH_EVENT'],ascending=True)

rel_with_target.plot(kind="bar")
rel_with_target.plot()
rel_with_target
sns.heatmap(df.corr())
feature_matrix = df.drop("DEATH_EVENT",axis=1)

target         = df[['DEATH_EVENT']]
new_target = np_utils.to_categorical(target)
normalized_fm=(feature_matrix-feature_matrix.min())/(feature_matrix.max()-feature_matrix.min())
X_train,X_test,y_train,y_test = train_test_split(normalized_fm,new_target,test_size=.2)
n_col = X_train.shape[1]
def build_model(n_col):

    model = Sequential()

    model.add(Dense(32, input_dim=n_col, activation='relu'))

    model.add(Dropout(.1))

    model.add(Dense(32, input_dim=n_col, activation='relu'))

    model.add(Dropout(.1))

    model.add(Dense(64, input_dim=n_col, activation='relu'))

    model.add(Dropout(.1))

    model.add(Dense(64, input_dim=n_col, activation='relu'))

    model.add(Dropout(.1))

    model.add(Dense(64, input_dim=n_col, activation='relu'))

    model.add(Dropout(.1))

    model.add(Dense(64, input_dim=n_col, activation='relu'))

    model.add(Dropout(.1))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    # compile the keras model

    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

    return model
model= build_model(n_col)
model.summary()
history = model.fit(X_train,y_train,epochs=100,validation_data=(X_test, y_test))
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
predicted_value = []

for item in model.predict(X_test):

    predicted_value.append(np.argmax(item))

print(predicted_value)    

    

    
actual_result = []

for item in y_test:

    actual_result.append(np.argmax(item))
pd.DataFrame(predicted_value).value_counts().plot(kind="bar")
pd.DataFrame(actual_result).value_counts().plot(kind="bar")
loss,acc = model.evaluate(X_test,y_test)
print("LOSS OF THE MODEL     : {}".format(loss))

print("ACCURACY OF THE MODEL : {}".format(acc))
report = classification_report(y_pred=predicted_value,y_true=actual_result)
print(report)
cm = confusion_matrix(predicted_value,actual_result)
sns.heatmap(cm,annot=True)