# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
####################### Train data #############################################
train['HF1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
train['HF2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HR1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['HR2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['FR1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['FR2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
train['ele_vert'] = train.Elevation-train.Vertical_Distance_To_Hydrology

train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2 

####################### Test data #############################################
test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])
test['ele_vert'] = test.Elevation-test.Vertical_Distance_To_Hydrology

test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2
train.head()
y = train.Cover_Type
X = train.drop(["Id","Cover_Type"],axis =1)
X_test = test.iloc[:,1:]
important = list(X.iloc[:,:10].columns) + list(X.iloc[:,-10:].columns)
X_imp = X[important]
X_nimp = X.drop(important,axis = 1)
test_imp = X_test[important]
test_nimp = X_test.drop(important,axis = 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_imp)
X_imp = sc.transform(X_imp)
test_imp = sc.transform(test_imp)
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# y_train = le.fit_transform(y)
import os
import sys
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model,Sequential
from sklearn.metrics import roc_auc_score
from keras.optimizers import Adam, RMSprop, Adagrad,Adadelta,SGD
from keras.layers import Flatten
from keras.layers.merge import concatenate
from sklearn.model_selection import StratifiedKFold
VALIDATION_SPLIT = 0.10
BATCH_SIZE = 64
EPOCHS = 1000
# seed = 7
# np.random.seed(seed)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
# cvscores = []
input1 = Input(shape=(20, ))
dense11 = Dense(40, )(input1)
dense12 = Dense(20, )(dense11)

input2 = Input(shape=(44, ))
dense21 = Dense(44, )(input2)
dense22 = Dense(30, )(dense21)
#dense22 = Dense(2, )(dense21)

merged = concatenate([dense12, dense22])
dense1 = Dense(600, activation='relu')(merged)
drop = Dropout(0.1)(dense1)
output = Dense(20, activation='softmax')(drop)

model = Model([input1,input2], output)    

adam = Adam(lr = 0.001)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer= "Adagrad",
    metrics=['accuracy']
)
r = model.fit(
    [X_imp,X_nimp],
    y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=0
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()
print(model.summary())
pred = model.predict([test_imp,test_nimp])
#np.argmax(pred, axis=1)
samp_sub = pd.read_csv("../input/sample_submission.csv")
samp_sub["Cover_Type"] = np.argmax(pred, axis=1)
samp_sub.to_csv("sub.csv",index = False)
