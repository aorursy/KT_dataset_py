# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
# データがどのくらい欠損しているか確認



def missing_table(x):

    # 特徴ごとに欠損値の数をカウントする

    null_num = x.isnull().sum()  # pandas.Series

    missing_table = pd.DataFrame(null_num) # pandas.DataFrame

    missing_table = missing_table.rename(columns={0: '欠損値'})

    return missing_table



missing_table(train)
missing_table(test)
# trainのEmbarkedが欠損値のデータを削除する



train = train[train['Embarked'].isnull() == False]
missing_table(train)
# カテゴリ変数を変換



train['Sex'][train['Sex'] == 'male'] = 0

train['Sex'][train['Sex'] == 'female'] = 1

test['Sex'][test['Sex'] == 'male'] = 0

test['Sex'][test['Sex'] == 'female'] = 1
train_for_xgb = train.copy()

test_for_xgb = test.copy()
train.head()
# Embarked（S, Q, Cという3カテゴリ）をダミー変数にする

# （Embarked列が消え、Embarked_S, Embarked_Q, Embarked_C列が追加される）

train = pd.get_dummies(train, columns=['Embarked'])
train
# Pclass（1, 2, 3という3カテゴリ）をダミー変数にする

# （Pclass列が消え、Pclass_1, Pclass_2, Pclass_3列が追加される）

# train = pd.get_dummies(train, columns=['Pclass'])
# train
train['Family'] = train['SibSp'] + train['Parch']

train
train['Family_size'] = 'Alone'

train['Family_size'][train['Family'] > 1] = 'Small'

train['Family_size'][train['Family'] > 4] = 'Big'

train.head(10)
train = pd.get_dummies(train, columns=['Family_size'])

train
import keras

from keras.models import Sequential

from keras.models import Model

from keras.layers import Dense,Flatten,Activation,Input,Dropout,BatchNormalization

from keras.optimizers import Adam,Nadam

from keras.layers.advanced_activations import PReLU



from keras.losses import binary_crossentropy



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import tensorflow as tf
# kerasの学習の再現



np.random.seed(7)

tf.random.set_seed(7)

def load_model():

    

    model = Sequential()

    model.add(Dense(12,input_dim = train_x.shape[1], kernel_regularizer=keras.regularizers.l2(0.001)))

    model.add(PReLU())

#     model.add(Dropout(0.2))

#     model.add(BatchNormalization())

    model.add(Dense(9, kernel_regularizer=keras.regularizers.l2(0.001)))

    model.add(PReLU())

    model.add(Dense(1,activation='sigmoid'))

    

    model.compile(loss=binary_crossentropy,optimizer=Nadam(lr=0.002),metrics=['accuracy'])

    

    return model





# def load_model(layers=[12,9]):

#     model = Sequential()

#     model.add(Dense(layers[0],input_dim = train_x.shape[1],kernel_regularizer=keras.regularizers.l2(0.001)))

#     model.add(PReLU())

    

#     for index in range(1,len(layers)):

#         model.add(Dense(layers[index],kernel_regularizer=keras.regularizers.l2(0.001)))

#         model.add(PReLU())

        

#     model.add(Dense(1,activation='sigmoid'))

#     model.compile(loss=binary_crossentropy,optimizer=Nadam(lr=0.001),metrics=['accuracy'])

    

#     return model
from sklearn.model_selection import KFold
train_x = train[['Pclass','Sex','Embarked_C','Embarked_Q','Embarked_S','Family_size_Alone','Family_size_Big','Family_size_Small']].values

train_x = train_x.astype('float')

train_y = train['Survived'].values
test = pd.get_dummies(test, columns=['Embarked'])

test['Family'] = test['SibSp'] + test['Parch']

test['Family_size'] = 'Alone'

test['Family_size'][test['Family'] > 1] = 'Small'

test['Family_size'][test['Family'] > 4] = 'Big'

test = pd.get_dummies(test, columns=['Family_size'])

test_feature = test[['Pclass','Sex','Embarked_C','Embarked_Q','Embarked_S','Family_size_Alone','Family_size_Big','Family_size_Small']].values

test_feature = test_feature.astype('float')
# crossValidation

# 学習データを4分割して、うち一つをバリデーションデータとする。これを繰り返す



# 各foldのスコアを保存する

scores_accuracy = []

scores_loss = []

count = 0

kf = KFold(n_splits=5,shuffle=True,random_state=71)

for tr_idx,val_idx in kf.split(train_x):

    #学習データを学習データとバリデーションデータに分ける

    tr_x,val_x = train_x[tr_idx],train_x[val_idx]

    tr_y,val_y = train_y[tr_idx],train_y[val_idx]

#     print(tr_x.shape,tr_y.shape,val_x.shape,val_y.shape)



    model = load_model()

    history = model.fit(tr_x,tr_y,

              batch_size=128,epochs=150,

              validation_data=(val_x,val_y))

    

    scores_accuracy.append(history.history['val_accuracy'][-1])

    scores_loss.append(history.history['val_loss'][-1])

    

    test['Survived'] = model.predict(test_feature)

    test['Survived'] =test['Survived'].apply(lambda x: round(x,0)).astype('int')



    solution = test[['PassengerId', 'Survived']]

    #solution.to_csv(f'neural_network_with_batchnorm__{count}.csv', index=False)

    

    count += 1
from statistics import mean

# 各foldのスコアの平均を出力

print(scores_accuracy)

loss = mean(scores_loss)

print(scores_loss)

accuracy = mean(scores_accuracy)

print(f'loss: {loss:.4f}, accuracy: {accuracy:.4f}')
from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
grid_model = KerasClassifier(build_fn=load_model,verbose=0)



batch_size = [32,64,128]

epochs = [50,100,150]

# layers = [[12],[14],[16],[18],[12,9],[14,9],[16,9],[18,9]]





param_grid = dict(batch_size=batch_size,epochs=epochs)



grid = GridSearchCV(estimator=grid_model,param_grid=param_grid,verbose=2)

grid_result = grid.fit(train_x,train_y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
model = load_model()

history = model.fit(train_x,train_y,batch_size=grid_result.best_params_['batch_size'],epochs=grid_result.best_params_['epochs'],validation_data=(val_x,val_y))
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# plot training & validation loss value

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
# validation dataもtrainに突っ込む

train_x,train_y = np.concatenate([train_x,val_x]), np.concatenate([train_y,val_y])

model.fit(train_x,train_y,batch_size=grid_result.best_params_['batch_size'],epochs=grid_result.best_params_['epochs'])
from xgboost import XGBClassifier

from sklearn.metrics import log_loss, accuracy_score
train_for_xgb['Embarked'][train_for_xgb['Embarked'] == 'S'] = 0

train_for_xgb['Embarked'][train_for_xgb['Embarked'] == 'Q'] = 1

train_for_xgb['Embarked'][train_for_xgb['Embarked'] == 'C'] = 2

test_for_xgb['Embarked'][test_for_xgb['Embarked'] == 'S'] = 0

test_for_xgb['Embarked'][test_for_xgb['Embarked'] == 'Q'] = 1

test_for_xgb['Embarked'][test_for_xgb['Embarked'] == 'C'] = 2



# train_for_xgb['Family'] = train_for_xgb['SibSp'] + train_for_xgb['Parch']

# test_for_xgb['Family'] = test_for_xgb['SibSp'] + test_for_xgb['Parch']



train_for_xgb['New_pclass'] = 3

train_for_xgb['New_pclass'][train_for_xgb['Pclass'] == 3] = 1

train_for_xgb['New_pclass'][train_for_xgb['Pclass'] == 2] = 2

test_for_xgb['New_pclass'] = 3

test_for_xgb['New_pclass'][test_for_xgb['Pclass'] == 3] = 1

test_for_xgb['New_pclass'][test_for_xgb['Pclass'] == 2] = 2



train_for_xgb
train_xgb_x = train_for_xgb[['Pclass','Sex','Embarked','New_pclass']].values

train_xgb_y = train_for_xgb['Survived'].values

test_feature_xgb = test_for_xgb[['Pclass','Sex','Embarked','New_pclass']].values



train_xgb_x
# 各foldのスコアを保存するリスト

scores_accuracy = []

scores_logloss = []
# crossValidation

# 学習データを4分割して、うち一つをバリデーションデータとする。これを繰り返す

count = 0

kf = KFold(n_splits=5,shuffle=True,random_state=71)

for tr_idx,val_idx in kf.split(train_xgb_x):

    #学習データを学習データとバリデーションデータに分ける

    tr_x,val_x = train_xgb_x[tr_idx],train_xgb_x[val_idx]

    tr_y,val_y = train_xgb_y[tr_idx],train_xgb_y[val_idx]

#     print(tr_x.shape,tr_y.shape,val_x.shape,val_y.shape)



    # モデルの学習を行う

    model_xgb = XGBClassifier(n_estimators=20, random_state=71)

    model_xgb.fit(tr_x, tr_y)



    # バリデーションデータの予測値を確率で出力する

    va_pred = model_xgb.predict_proba(val_x)[:, 1]



    # バリデーションデータでのスコアを計算する

    logloss = log_loss(val_y, va_pred)

    accuracy = accuracy_score(val_y, va_pred > 0.5)



    # そのfoldのスコアを保存する

    scores_logloss.append(logloss)

    scores_accuracy.append(accuracy)

    

    

    count += 1
from statistics import mean

# 各foldのスコアの平均を出力

print(scores_accuracy)

loss = mean(scores_loss)

print(scores_loss)

accuracy = mean(scores_accuracy)

print(f'loss: {loss:.4f}, accuracy: {accuracy:.4f}')




# モデルの学習を行う

model_xgb = XGBClassifier(n_estimators=20, random_state=71)

model_xgb.fit(train_xgb_x, train_xgb_y)



model_nn = load_model()

model_nn.fit(train_x,train_y,

          batch_size=128,epochs=150)





pred_xgb = model_xgb.predict_proba(test_feature_xgb)[:, 1]

pred_nn = model_nn.predict(test_feature)

pred_nn = pred_nn.flatten()



test['Survived'] = pred_nn * 0.5 + pred_xgb * 0.5

test['Survived'] = test['Survived'].apply(lambda x: round(x,0)).astype('int')





solution = test[['PassengerId', 'Survived']]

solution.to_csv(f'nn_and_xgb.csv', index=False)

kf = KFold(n_splits=5,shuffle=True,random_state=71)

for tr_idx,_ in kf.split(train_x):

    #学習データを学習データとバリデーションデータに分ける

    tr_x_xgb,tr_y_xgb = train_xgb_x[tr_idx],train_xgb_y[tr_idx]

    tr_x_nn,tr_y_nn = train_x[tr_idx],train_y[tr_idx]

    

    model_xgb = XGBClassifier(n_estimators=20, random_state=71)

    model_xgb.fit(tr_x_xgb, tr_y_xgb)



    model_nn = load_model()

    model_nn.fit(tr_x_nn,tr_y_nn,

              batch_size=128,epochs=150)





    pred_xgb = model_xgb.predict_proba(test_feature_xgb)[:, 1]

    pred_nn = model_nn.predict(test_feature)

    pred_nn = pred_nn.flatten()



    test['Survived'] = pred_nn * 0.5 + pred_xgb * 0.5

    test['Survived'] = test['Survived'].apply(lambda x: round(x,0)).astype('int')





    solution = test[['PassengerId', 'Survived']]

    solution.to_csv(f'nn_and_xgb_{count}.csv', index=False)



    

    

    count += 1
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")