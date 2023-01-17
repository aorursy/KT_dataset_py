import pandas

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

import xgboost as xgb

from sklearn import ensemble,metrics,cross_validation

import os

import seaborn as sns
seed = 7

np.random.seed(seed)
data = pd.read_csv('../input/train.csv')

test_ = pd.read_csv('../input/test.csv')



data['val']='train'

test_['val']='test'
all_data=pd.concat([data,test_])

cat_cols=[i for i in all_data.columns if 'cat' in i]
all_data['ps_car_01_cat']=all_data['ps_car_01_cat'].apply(lambda x:11 if x==-1 else x)

all_data['ps_car_02_cat']=all_data['ps_car_02_cat'].apply(lambda x:1 if x==-1 else x)

all_data['ps_car_02_cat']=all_data['ps_car_07_cat'].apply(lambda x:1 if x==-1 else x)

all_data['ps_car_09_cat']=all_data['ps_car_09_cat'].apply(lambda x:2 if x==-1 else x)

all_data['ps_ind_02_cat']=all_data['ps_ind_02_cat'].apply(lambda x:1 if x==-1 else x)

all_data['ps_ind_04_cat']=all_data['ps_ind_04_cat'].apply(lambda x:0 if x==-1 else x)

all_data['ps_ind_05_cat']=all_data['ps_ind_05_cat'].apply(lambda x:0 if x==-1 else x)
cat_cols=[i for i in all_data.columns if 'cat' in i]

new_data=pd.get_dummies(all_data,columns=cat_cols,drop_first=True)

X=[i for i in new_data.columns if i not in ['id','target','val']]

Y=['target']

train_data=new_data[new_data['val']=='train']

test_data=new_data[new_data['val']=='test']

del all_data

del data

del new_data

print ('Data process Complete')
train_data.shape, test_data.shape
def gini(list_of_values):

#     print ('lala2')

    sorted_list = sorted(list(list_of_values))

    height, area = 0, 0

    for value in sorted_list:

        height += value

        area += height - value / 2.

    fair_area = height * len(list_of_values) / 2

    return (fair_area - area) / fair_area

  

def normalized_gini(y_pred, y):

#     print ('lala3')

    normalized_gini = gini(y_pred)/gini(y)

    return normalized_gini



def gini_xgb(preds, dtrain):

#     print ('lala1')

    labels = dtrain.get_label()

    gini_score = normalized_gini(labels, preds)

    return [('gini', gini_score)]

train_X,test_X,train_y,test_y=cross_validation.train_test_split(train_data[X],train_data[Y],test_size=.3,)

train_X.shape,test_X.shape,train_y.shape,test_y.shape
K = 5

kf = cross_validation.KFold(train_data.shape[0],n_folds= K, random_state = 3228, shuffle = True)
keras_preds=[]

for train_index, test_index in kf:

    print ('A')

    train_X, valid_X = train_data[X].loc[train_index], train_data[X].loc[test_index]

    train_y, valid_y = train_data[Y].loc[train_index], train_data[Y].loc[test_index]



    # params configuration also from anokas' kernel

    model = Sequential()

    model.add(Dense(250, input_dim=207, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    model.fit(train_X.values, train_y.values, epochs=3, batch_size=32,verbose=True,

                    validation_data=(valid_X.values, valid_y.values))

    pred=model.predict_proba(test_data[X].values)

    keras_preds.append(list(pred))
len(keras_preds)
preds=[]

for i in range(len(keras_preds[0])):

    sum=0

    for j in range(K):

        sum+=keras_preds[j][i]

    preds.append(sum / K)
sns.distplot(np.ravel(preds))
output = pd.DataFrame({'id': test_['id'], 'target': np.ravel(preds)})

output.to_csv("sol_sub2.csv".format(K), index=False)   

print ('process done')
output.shape