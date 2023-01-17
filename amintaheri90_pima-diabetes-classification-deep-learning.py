# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf 

import matplotlib.pyplot as plt

import sklearn as sk
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import seaborn as sns

import warnings

warnings.simplefilter('ignore')

sns.set(rc={'figure.figsize' : (10, 5)})

sns.set_style("darkgrid", {'axes.grid' : True})
diabetes = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

dia = diabetes.copy()
diabetes.describe().T
# diabetes.Pregnancies = diabetes.Pregnancies.replace(0,333)

# diabetes.Outcome = diabetes.Outcome.replace(0,333)

# diabetes



# diabetes = diabetes.replace(0, np.nan)

# diabetes = diabetes.dropna()

# # df = df.replace(np.nan, 0.0)



# diabetes.Pregnancies = diabetes.Pregnancies.replace(333,0)

# diabetes.Outcome = diabetes.Outcome.replace(333,0)

# diabetes
corrMatrix = diabetes.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
from sklearn.dummy import DummyClassifier

x = diabetes.drop(columns = 'Outcome')

y = diabetes['Outcome']
x.head()
y.head()
dummy = DummyClassifier('most_frequent') #returining most frequent class in this case 1/

results = dummy.fit(x,y)

results.score(x,y)
len( # Number of recrds that have at least one zero in it

    x[(x.Glucose == 0) |

    (x.BloodPressure ==0) |

    (x.SkinThickness==0) |

    (x.Insulin==0) |

    (x.BMI==0) |

    (x.DiabetesPedigreeFunction==0) |

    (x.Age==0)]

)
from tensorflow import keras

import tensorflow.keras.backend as K



def get_f1(y_true, y_pred): #taken from old keras source code

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val
def max_metric (history):

    max_acc = max(history.history['accuracy'])

    max_f1 = max(history.history['get_f1'])

    min_loss = min(history.history['loss'])

    max_val_acc = max(history.history['val_accuracy'])

    max_val_f1 = max(history.history['val_get_f1'])

    min_val_loss = min(history.history['val_loss'])

    print(f"Maximum Accuracy: {max_acc} \nMaximum F1 Score: {max_f1} \nMinimum Binary CrossEntropy Loss: {min_loss} \nMaximum Validation Accuracy: {max_val_acc} \nMaximum Validation F1 Score: {max_val_f1} \nMaximum Validation Binary CrossEntropy Loss: {min_val_loss} \n")

def plot_this(history):

    # summarize history for accuracy

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

    

    # summarize history for f1

    plt.plot(history.history['get_f1'])

    plt.plot(history.history['val_get_f1'])

    plt.title('model f1')

    plt.ylabel('f1')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
diabetes.columns
# normalize the data

# we do not want to modify our label column Exited

cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',

       'DiabetesPedigreeFunction', 'Age']



# copy churn dataframe to churn_norm to do not affect the original data

dia_norm = diabetes.copy()



# normalize churn_norm dataframe 

dia_norm[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/ (x.max() - x.min()) )



x = dia_norm.drop(columns = 'Outcome')

y = dia_norm['Outcome']
dia_norm
dia_norm.describe().T
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

import warnings
min_max_scaler = preprocessing.MinMaxScaler()

X_train_minmax = min_max_scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X_train_minmax, y, test_size=0.33, random_state=42)
parameters = {'C': np.linspace(0.0001, 100, 40)}

grid_search = GridSearchCV(LogisticRegression(max_iter=3000, class_weight={0:0.35, 1:0.65}), parameters, n_jobs=-1)

grid_search.fit(X_train, y_train)



print('best parameters: ', grid_search.best_params_)

print('best scrores: ', grid_search.best_score_)
lr_clf = LogisticRegression(C=2.5642, max_iter=3000, class_weight={0:0.35, 1:0.65})

lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)
from sklearn.metrics import f1_score

y_hat = lr_clf.predict(X_test)

f1 = f1_score(y_test, y_hat)

print (f"f1 socre is: {f1} ")
#lr_clf.predict_proba(x.iloc[[700]]) #FOR PREDICtion
parameters = {'C': np.linspace(0.0001, 100, 40)}

grid_search = GridSearchCV(svm.SVC(probability=True, max_iter=300, class_weight={1: 0.65, 0:0.35}), parameters, n_jobs=-1)

grid_search.fit(X_train, y_train)



print('best parameters: ', grid_search.best_params_)

print('best scrores: ', grid_search.best_score_)
from sklearn import svm



clf = svm.SVC(C=28.205199999999998, gamma='auto', probability=True, verbose=True, max_iter=3000, class_weight={1: 0.65, 0:0.35})

clf.fit(X_train, y_train)
clf.score(X_test, y_test)
from sklearn.metrics import f1_score

y_hat = clf.predict(X_test)

f1 = f1_score(y_test, y_hat)

print (f"f1 socre is: {f1} ")
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)

print (f"f1 socre is: {f1} ")
model2 = tf.keras.Sequential()

model2.add(tf.keras.layers.Dense(16, input_dim=x.shape[1], kernel_regularizer=tf.keras.regularizers.l2(0.001)))#activation = 'relu' ))

model2.add(tf.keras.layers.ELU(alpha=1))

model2.add(tf.keras.layers.Dropout(0.2))

model2.add(tf.keras.layers.Dense(16,kernel_regularizer=tf.keras.regularizers.l2(0.001)))# activation='relu'))

model2.add(tf.keras.layers.ELU(alpha=1))

model2.add(tf.keras.layers.Dropout(0.2))

model2.add(tf.keras.layers.Dense(16,kernel_regularizer=tf.keras.regularizers.l2(0.001)))# activation='relu'))

model2.add(tf.keras.layers.ELU(alpha=1))

model2.add(tf.keras.layers.Dropout(0.2))

model2.add(tf.keras.layers.Dense(16,kernel_regularizer=tf.keras.regularizers.l2(0.001)))# activation='relu'))

model2.add(tf.keras.layers.ELU(alpha=1))

model2.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model2.compile(optimizer='rmsprop', loss='MSE', metrics=['accuracy', get_f1])
history2 = model2.fit(X_train, y_train, validation_split=0.20, batch_size=64, workers=-1, epochs=100, verbose=2, class_weight={0:0.35, 1:0.65})
max_metric(history2)

plot_this(history2)
# model2.save("model2.h5")
# from  tensorflow.keras.utils import plot_model

# plot_model(model2, to_file='model.png', show_shapes=True, rankdir="LR", expand_nested=False ,dpi=200)
from sklearn.metrics import f1_score



y_pred = model2.predict(X_test, verbose=1)

y_pred = y_pred>0.5

f1 = f1_score(y_test, y_pred)

print (f"f1 socre is: {f1} ")
# apply(lambda x: (x - x.min())/ (x.max() - x.min()) 

# normalize the data

# we do not want to modify our label column Exited

# Pregnancies                  0.000

# Glucose                      0.000

# BloodPressure                0.000

# SkinThickness                0.000

# Insulin                      0.000

# BMI                          0.000

# DiabetesPedigreeFunction     0.078

# Age                         21.000





# Pregnancies                  17.00

# Glucose                     199.00

# BloodPressure               122.00

# SkinThickness                99.00

# Insulin                     846.00

# BMI                          67.10

# DiabetesPedigreeFunction      2.42

# Age                          81.00





## Sorry for hardcoding, i wll fix it ASAP!



def norm_a_data(data):

    data[0] = (data[0] - 0)    / (17 - 0)

    data[1] = (data[1] - 0)    / (199 - 0)

    data[2] = (data[2] - 0)    / (122 - 0)

    data[3] = (data[3] - 0)    / (99 - 0)

    data[4] = (data[4] - 0)    / (846 - 0)

    data[5] = (data[5] - 0)    / (67 - 0)

    data[6] = (data[6] - 0.078) / (2 - 0.078)

    data[7] = (data[7] - 21)    / (81 - 21)

    return data[:]
my_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]

my_data = diabetes.iloc[245].values[:8]

my_data = norm_a_data(my_data)

my_data=np.array(my_data)

my_data = my_data.reshape(8,1)

# pd.DataFrame(my_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',

#        'DiabetesPedigreeFunction', 'Age'])

model2.predict(my_data.transpose())
diabetes.iloc[245].values[-1]