#Quick load dataset and check
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
TRAIN_SET_PATH = "/content/drive/My Drive/colab/train_set.csv"
TEST_SET_PATH = "/content/drive/My Drive/colab/test_set.csv"
data_train = pd.read_csv(TRAIN_SET_PATH)
data_test = pd.read_csv(TEST_SET_PATH)
n = 20

#Helper Functions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer


def getTrainingValues(properties=[], impute=True):
  df = pd.read_csv(TRAIN_SET_PATH)
  
  if(impute):
    simple_imp = SimpleImputer(missing_values=-1, strategy='mean')
    new_df = pd.DataFrame(simple_imp.fit_transform(df))
    new_df.columns = df.columns
    new_df.index = df.index
    df = new_df

  if len(properties) == 0:
    # separate target & id from values 
    properties = list(df.columns.values)
    properties.remove('target')
    properties.remove('id')

  X = df[properties]
  y = df['target'].astype(int)
  return X, y

def getTestValues(properties=[]):
  df = pd.read_csv(TEST_SET_PATH)
  orig = df

  # do we need to impute here??
  simple_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
  new_df = pd.DataFrame(simple_imp.fit_transform(df))
  new_df.columns = df.columns
  new_df.index = df.index
  df = new_df

  if len(properties) == 0:
    properties = list(df.columns.values)

  X = df[properties]
  return orig, X

def getNBestFeatures(n):
  X, y = getTrainingValues(impute = True)
  bestfeatures = SelectKBest(score_func=f_classif, k=n)
  fit = bestfeatures.fit(X,y)
  dfscores = pd.DataFrame(fit.scores_)
  dfcolumns = pd.DataFrame(X.columns)
  featureScores = pd.concat([dfcolumns,dfscores],axis=1)
  featureScores.columns = ['Specs','Score']
  # print(featureScores.nlargest(n,'Score'))
  return featureScores.nlargest(n,'Score')['Specs']

def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

def calc_class_weights(target):
  neg, pos = np.bincount(target)
  total = neg + pos
  print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
  weight_for_0 = (1 / neg)*(total)/2.0 
  weight_for_1 = (1 / pos)*(total)/2.0
  class_weight = {0: weight_for_0, 1: weight_for_1}
  print('Weight for class 0: {:.2f}'.format(weight_for_0))
  print('Weight for class 1: {:.2f}'.format(weight_for_1))
  return class_weight

getNBestFeatures(n)
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import imblearn
from imblearn.over_sampling import SMOTE

X, y = getTrainingValues(getNBestFeatures(n), impute=False)

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(n,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#variate between epochs and batch size
model.fit(X_train, y_train, epochs=10, batch_size=64)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
from sklearn import metrics

def get_score(y_test,y_pred):
  score = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', zero_division='warn')
  print("binary f1 score is: ",score)
  score = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', zero_division='warn')
  print("weighted f1 score is: ",score)
  score = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro', zero_division='warn')
  print("macro f1 score is: ",score)
  score = metrics.accuracy_score(y_test,y_pred)
  print("total acc is: ",score)
  return score
# test our results on ones and zeros 

# treshold for the probability to predict 0/1
TRESH = 0.5

y_pred = model.predict(X_test)

y_pred[y_pred < TRESH]  = 0
y_pred[y_pred >= TRESH] = 1
get_score(y_test, y_pred)


X_pos, y_pos = extrac_one_label(X_test, y_test, 1)
X_neg, y_neg = extrac_one_label(X_test, y_test, 0)

y_negpred = model.predict(X_neg)
y_negpred[y_negpred < TRESH]  = 0
y_negpred[y_negpred >= TRESH] = 1
print("Accuracy of predicting 0:", sum(y_negpred==0)/len(y_negpred))
print("sum 0:", sum(y_negpred==0))


y_pospred = model.predict(X_pos)
y_pospred[y_pospred < TRESH]  = 0
y_pospred[y_pospred >= TRESH] = 1
print("Accuracy of predicting 1:", sum(y_pospred==1)/len(y_pospred))
print("sum 1:", sum(y_pospred==1))

data_real, data_test = getTestValues(getNBestFeatures(n))
y_target = model.predict(data_test)

print((y_target))


y_target[y_target < TRESH]  = 0
y_target[y_target >= TRESH] = 1


print(sum(y_target))

data_out = pd.DataFrame(data_real['id'].copy())
data_out.insert(1, "target", y_target.astype(int), True)
data_out.to_csv('submission.csv',index=False)
data_out