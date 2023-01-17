# these requirements are needed for some parts of our code
!pip install imblearn
%pip install tensorflow
# Quick load dataset and check
import pandas as pd
import os
running_local = True if os.getenv('JUPYTERHUB_USER') is None else False
os.listdir('/data')
if ~running_local:
    path = "/data/final-project-dataset/"
else:
    path = "./"
filename = path + "train_set.csv"
data_train = pd.read_csv(filename)
filename = path + "test_set.csv"
data_test = pd.read_csv(filename)

data_train.describe()
data_test.describe()

from sklearn.tree import DecisionTreeClassifier
## Select target and features
fea_col = data_train.columns[2:]

data_Y = data_train['target']
data_X = data_train[fea_col]


clf = DecisionTreeClassifier()
clf = clf.fit(data_X,data_Y)
y_pred = clf.predict(data_X)
sum(y_pred==data_Y)/len(data_Y)

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)
clf = DecisionTreeClassifier(min_impurity_decrease = 0.001)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)

sum(y_pred==y_val)/len(y_val)
def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)


sum(y_pospred==y_pos)/len(y_pos)
X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
sum(y_negpred==y_neg)/len(y_neg)
print(sum(data_Y==0)/len(data_Y), sum(data_Y==1))
import numpy as np
zeros = np.sum(data_Y==0)
ones = np.sum(data_Y==1)
print("amounf of 0: ", zeros)
print("amounf of 1: ", ones)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
labels = ['0', '1']
values = [zeros, ones]
ax.bar(labels, values)

for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(data_Y)), 
                (p.get_x() + 0.3, p.get_height() + 10000))

plt.show()
vars_with_missing = []
for f in data_X.columns:
    missings = data_X[data_X[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/data_X.shape[0]
        
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
# Dropping columns with too many -1 values
data_X = data_X.drop(vars_to_drop, axis=1)
from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(missing_values=-1, strategy='median')

# replace missing values with median
data_X['ps_ind_02_cat'] = imp_median.fit_transform(data_X[['ps_ind_02_cat']])
data_X['ps_ind_04_cat'] = imp_median.fit_transform(data_X[['ps_ind_04_cat']])
data_X['ps_ind_05_cat'] = imp_median.fit_transform(data_X[['ps_ind_05_cat']])
data_X['ps_reg_03'] = imp_median.fit_transform(data_X[['ps_reg_03']])
data_X['ps_car_01_cat'] = imp_median.fit_transform(data_X[['ps_car_01_cat']])
data_X['ps_car_02_cat'] = imp_median.fit_transform(data_X[['ps_car_02_cat']])
data_X['ps_car_07_cat'] = imp_median.fit_transform(data_X[['ps_car_07_cat']])
data_X['ps_car_09_cat'] = imp_median.fit_transform(data_X[['ps_car_09_cat']])
data_X['ps_car_11'] = imp_median.fit_transform(data_X[['ps_car_11']])
data_X['ps_car_12'] = imp_median.fit_transform(data_X[['ps_car_12']])
data_X['ps_car_14'] = imp_median.fit_transform(data_X[['ps_car_14']])
vars_with_missing = []
for f in data_X.columns:
    missings = data_X[data_X[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/data_X.shape[0]
        
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

x_train_s = x_train[0:10000]
y_train_s = y_train[0:10000]

x_val_s = x_val[0:10000]
y_val_s = y_val[0:10000]
print(x_train.shape)
print(x_val.shape)
import imblearn

# import the SMOTETomek
from imblearn.over_sampling import SMOTE

# create the  object with the desired sampling strategy.
smote = SMOTE(sampling_strategy='minority')

# fit the object to our training data
x_train_smtenn, y_train_smtenn = smote.fit_sample(x_train, y_train)
print(sum(y_train_smtenn == 1))
print(sum(y_train_smtenn == 0))
print(sum(y_train ==1))
# import the ADASYN object.
from imblearn.over_sampling import ADASYN

# create the object to resample the majority class.
adasyn = ADASYN(sampling_strategy='minority')

# fit the object to the training data.
x_train_adasyn, y_train_adasyn = adasyn.fit_sample(x_train, y_train)
# import the Random Over Sampler object.
from imblearn.over_sampling import RandomOverSampler

# create the object.
over_sampler = RandomOverSampler(random_state=42)

# fit the object to the training data.
x_train_over, y_train_over = over_sampler.fit_sample(x_train, y_train)
# import the resample object.
from sklearn.utils import resample

train = pd.concat([x_train, y_train], 1)
c0 = train[train['target'] == 0]
c1 = train[train['target'] == 1]
c1_modified = resample(c1, replace=True, n_samples=int(len(c0)/1.05))
resampled = pd.concat([c0, c1_modified])
resampled.sort_index(inplace=True)
x_train_re = resampled.drop('target', 1)
y_train_re = resampled['target']
import tensorflow as tf
from numpy import loadtxt
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential

X = x_train_re
y = y_train_re

# define the keras model
model = Sequential()
model.add(Dense(100, input_dim=55, activation='relu'))
model.add(Dense(50, input_dim=55, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=200, batch_size=100, validation_data=(x_val, y_val))

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
rf_pred = model.predict_classes(x_val)
from sklearn.metrics import f1_score
f1_score(y_val, rf_pred, average='macro')
from sklearn.neural_network import MLPClassifier
X = x_train
y = y_train
clf = MLPClassifier(solver='adam', activation='relu', learning_rate='adaptive', verbose=True, max_iter=40, batch_size=100)

clf.fit(X, y)
Y_target = clf.predict(x_val)
print(np.sum(y_val == 1))
print(np.sum(Y_target == 1))
from sklearn.metrics import accuracy_score
score = accuracy_score(y_val,Y_target)

print(score)
from sklearn.metrics import f1_score

f1_score(y_val, Y_target, average='macro')
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=200, n_jobs=-1, min_samples_leaf=55, min_samples_split=150)

# Fit on training data
model.fit(x_train_re, y_train_re)
rf_pred = model.predict(x_val)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_val,rf_pred))
print(accuracy_score(y_val,rf_pred))
print("-------------------------------------------")
print("real ", np.sum(y_val==1))
print("predicted ", np.sum(rf_pred==1))
X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = model.predict(X_pos)

sum(y_pospred==y_pos)/len(y_pos)
from sklearn.metrics import f1_score
f1_score(y_val, rf_pred, average='macro')
import pickle

# Save to file in the current working directory
pkl_filename = "tree.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
data_test = data_test.drop(['ps_car_03_cat', 'ps_car_05_cat'], axis = 1)
fea_col = data_test.columns[1:]

data_test_X = data_test[fea_col]

data_test_X = data_test.drop(columns=['id'])
rf_predictions = model.predict(data_test_X)
print(sum(rf_predictions==1))

data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", rf_predictions, True) 
data_out.to_csv('submission.csv',index=False)
data_out

print(data_out.shape)
