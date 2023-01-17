import csv

import datetime

import numpy as np

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential

from keras.utils import np_utils

from keras import applications

from keras import regularizers

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from keras.layers.advanced_activations import PReLU
def get_data(fn):

  data = []

  with open(fn) as f:

    reader = csv.DictReader(f)

    data = [row for row in reader]

  return data



def get_fields(data, fields):

  extracted = []

  for row in data:

    extract = []

    for field, f in sorted(fields.items()):

      info = f(row[field])

      if type(info) == list:

        extract.extend(info)

      else:

        extract.append(info)

    extracted.append(np.array(extract, dtype=np.float32))

  return extracted



def preprocess_data(X, scaler=None):

  if not scaler:

    scaler = StandardScaler()

    scaler.fit(X)

  X = scaler.transform(X)

  return X





def dateConvertion(x):

  date, time = x.split(' ')

  year, month, day = map(int, date.split('-'))

  hour, minute, second = time.split(':')

  return [day, month, year, hour, minute, datetime.datetime(year, month, day).isocalendar()[1], getHourPart(int(hour))]



def getHourPart(hour):

    if(hour >= 2 and hour < 8): return 1;

    if(hour >= 8 and hour < 12): return 2;

    if(hour >= 12 and hour < 14): return 3;

    if(hour >= 14 and hour < 18): return 4;

    if(hour >= 18 and hour < 22): return 5;

    if(hour < 2 or hour >= 22): return 6;



days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']



data_fields = {

    'X': lambda x: float(x),

    'Y': lambda x: float(x),

    'Dates' : lambda x : dateConvertion(x),

    'DayOfWeek' : lambda x : [days.index(x), 1 if days.index(x) > 4 else 0],

    'Address': lambda x: [1 if ('/' in x.lower() and 'of' not in x.lower()) else 0],

    'PdDistrict': lambda x: districts.index(x),

}



print('Loading training data...')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import zipfile

z1 = zipfile.ZipFile('../input/sf-crime/train.csv.zip')

z2 = zipfile.ZipFile('../input/sf-crime/test.csv.zip')

z1.extractall()

z2.extractall()

raw_train = get_data('../working/train.csv')    
districts = np.unique([row['PdDistrict'] for row in raw_train]).tolist()

labels = np.unique([row['Category'] for row in raw_train]).tolist()

label_fields = {'Category': lambda x: labels.index(x.replace(',', ''))}

print('days')

print(days)

print('districts')

print(districts)

print('labels')

print(labels)
print('Creating training data...')

X = np.array(get_fields(raw_train, data_fields), dtype=np.float32)

print('Creating training labels...')

y = np.array(get_fields(raw_train, label_fields))

del raw_train
X = preprocess_data(X)

Y = np_utils.to_categorical(y)



input_dim = X.shape[1]

output_dim = len(labels)



def build_model(input_dim, output_dim, hn=64, dp=0.5, layers=1):

    model = Sequential()

    model.add(BatchNormalization())

    model.add(Dense(output_dim, input_shape=(input_dim,)))

    model.add(PReLU())

    

    model.add(Dense(100, init='uniform'))

    model.add(Activation('tanh'))

    model.add(Dropout(0.5))

    

    model.add(Dense(hn, input_shape=(hn,), init='glorot_uniform',  activity_regularizer=regularizers.l1(0.01)))



    model.add(Dense(output_dim, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    return model



EPOCHS = 40

BATCHES = 128

HN = 256

LAYERS = 0

DROPOUT = 0.01

ITERATIONS = 5



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42,stratify=Y)

model = build_model(input_dim, output_dim, HN, DROPOUT, LAYERS)

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCHES, validation_data=(X_test, y_test), verbose=2)

prediction = model.predict(X_test, verbose=2)

predictedVals = []

for row in prediction:

    predictedVals.append(np.argmax(row))

validVals = []

for row in y_test:

    validVals.append(np.argmax(row))
print("F1 score (micro): ", f1_score(predictedVals, validVals, average='micro'))

print("F1 score (macro): ", f1_score(predictedVals, validVals, average='macro'))

print("F1 score (weighted): ", f1_score(predictedVals, validVals, average='weighted'))

c = confusion_matrix(predictedVals, validVals)

reverse_c = list(zip(*np.array(c)))

for i in range(len(c[1])):

    print(labels[i])

    fn = sum(c[i]) - c[i][i]

    fp = sum(reverse_c[i]) - c[i][i]

    print("Правильных результатов: " + str(c[i][i]))

    print("Ошибки первого рода: "+ str(fn))

    print("Ошибки второго рода: " + str(fp))
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

plt.hist(predictedVals, alpha=.5, color='blue')

plt.hist(validVals, alpha=.5, color='red')

plt.legend('PA')

#plt.xlabel('1 - Predicted, 2 - Actual')

plt.show()