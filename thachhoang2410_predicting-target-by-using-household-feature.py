# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from tqdm import tqdm
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# fill NaN values of v2a1 = 0 when house_house_status = 0
train['v2a1'][train['tipovivi1']==1] = 0
test['v2a1'][test['tipovivi1']==1] = 0
# fill NaN values of v2a1 = 0 when house_house_status = 0
train['v18q1'][train['v18q']==0] = 0
test['v18q1'][test['v18q']==0] = 0
train['eviv1'] = np.logical_and(np.array(train['eviv1']), np.logical_not(np.array(train['pisonotiene'])))*1
test['eviv1'] = np.logical_and(np.array(test['eviv1']), np.logical_not(np.array(test['pisonotiene'])))*1
train = train.replace('no', 0)
train = train.replace('yes', 1)
test = test.replace('no', 0)
test = test.replace('yes', 1)
print("Training shape: ", train.shape)
print("Training info: ")
train.info()
print("\n-----------------------------------------\n")
print("Test shape: ", test.shape)
print("Test info: ")
test.info()
print("Test/Train raito: ", test.shape[0]/float(train.shape[0]))
train[['r4t3', 'tamhog', 'hhsize', 'tamviv']].describe()
print("Data type of columns:")
train.describe()
print("Check NaN values in Train set:")
isnull = train.isnull().sum().reset_index()
#isnull[isnull>0]
isnull.columns = ['Feature', 'Total_null']
total_null = isnull[isnull['Total_null']>0]
total_null
print("Check NaN values in Test set:")
isnull = test.isnull().sum().reset_index()
#isnull[isnull>0]
isnull.columns = ['Feature', 'Total_null']
total_null = isnull[isnull['Total_null']>0]
total_null
test.describe()
"""
feature_used = ['hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'tamhog', 'paredblolad', 'paredzocalo',
                'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras' ,'paredother' ,'pisomoscer',
                'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 
                'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera',
                'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3',
                'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4',
                'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2',
                'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3']
"""
feature_used = ['cielorazo', 'v18q1', 'computer', 'television', 'qmobilephone', 'refrig', 'bedrooms', 'hacdor', 'overcrowding', 'rooms', 
                'hacapo', 'v14a','paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'paredblolad', 'paredzocalo',
               'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 
                'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 
                'elimbasu2', 'elimbasu3', 'elimbasu4','elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 
               'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'eviv1', 'eviv2', 'eviv3', 'pisonotiene', 'pisomoscer', 
                'pisocemento', 'pisoother', 'pisonatur', 'pisomadera', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6',
               'area1', 'area2', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
train_groupby = train.groupby(['idhogar']).mean().reset_index()
test_groupby = test.groupby(['idhogar']).mean().reset_index()

test_missing_rent = test_groupby[test_groupby['v2a1'].isnull()]
test_rent = test_groupby[~test_groupby['v2a1'].isnull()]

train_missing_rent = train_groupby[train_groupby['v2a1'].isnull()]
train_rent = train_groupby[~train_groupby['v2a1'].isnull()]

temp_frame = [train_rent, test_rent]
rent_training = pd.concat(temp_frame)

temp_frame = [train_missing_rent, test_missing_rent]
rent_test = pd.concat(temp_frame)

X = rent_training[feature_used]
Y = rent_training[['v2a1']]

X_test = rent_test[feature_used]
idhogar_test = rent_test['idhogar']

# normalize data
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X)
min_max_scaler.fit(X_test)
X_scale = min_max_scaler.transform(X)


# train on linear regression model
# split train/test set
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(0)
X_train, X_val, Y_train, Y_val = train_test_split(X_scale, Y, test_size=0.1, random_state=0)

model = LinearRegression().fit(X_train, Y_train)
print("Training score: ", model.score(X_train, Y_train))
print("Validation score: ", model.score(X_val, Y_val))

# predict rent for missing values
idhogar_test_df = pd.DataFrame(idhogar_test, columns=['idhogar'])
idhogar_test_df = idhogar_test_df.reset_index().drop(columns=['index'])

rent_predict = model.predict(X_test)
rent_predict_df = pd.DataFrame(rent_predict, columns=['v2a1'])
test_result_predict = pd.concat([idhogar_test_df, rent_predict_df], axis=1, join='inner')

# merge predicting values with original train/test dataframe
temp_train = pd.merge(train, test_result_predict, on=['idhogar', 'idhogar'], how='left')
temp_train['v2a1_x'].fillna(temp_train['v2a1_y'], inplace=True)
temp_train = temp_train.drop(columns='v2a1_y')
temp_train = temp_train.rename(index=str, columns={'v2a1_x':'v2a1'})

temp_test = pd.merge(test, test_result_predict, on=['idhogar', 'idhogar'], how='left')
temp_test['v2a1_x'].fillna(temp_test['v2a1_y'], inplace=True)
temp_test = temp_test.drop(columns='v2a1_y')
temp_test = temp_test.rename(index=str, columns={'v2a1_x':'v2a1'})
# store the original data
original_train = train
original_test = test

# set train/test data to the new one which using predicting monthly rent values
train = temp_train
test = temp_test
print("Check NaN values in Train set:")
isnull = train.isnull().sum().reset_index()
#isnull[isnull>0]
isnull.columns = ['Feature', 'Total_null']
total_null = isnull[isnull['Total_null']>0]
total_null
print("Check NaN values in Test set:")
isnull = test.isnull().sum().reset_index()
#isnull[isnull>0]
isnull.columns = ['Feature', 'Total_null']
total_null = isnull[isnull['Total_null']>0]
total_null
train.describe()
temp_train = train[['Id', 'idhogar', 'r4h3', 'r4m3', 'r4t3']]
temp_train.columns=['Id','idhogar','Total_male','Total_female', 'Total_person']
temp_train.head()
train.loc[train['idhogar'] == '2b58d945f']
# replace all NaN value to -1
#train.fillna(-1, inplace=True)
#test.fillna(-1, inplace=True)
train = train.fillna(train.mean())
test = test.fillna(test.mean())
train = train.replace('no', 0)
train = train.replace('yes', 1)
test = test.replace('no', 0)
test = test.replace('yes', 1)
def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['rent_to_bedrooms'] = df['v2a1']/df['bedrooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household
    df['tamhog_to_bedrooms'] = df['tamhog']/df['bedrooms']
    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household
    df['r4t3_to_rooms'] = df['r4t3']/df['rooms'] # r4t3 - Total persons in the household
    df['r4t3_to_bedrooms'] = df['r4t3']/df['bedrooms']
    df['rent_to_r4t3'] = df['v2a1']/df['r4t3']
    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1'])
    df['hhsize_to_rooms'] = df['hhsize']/df['rooms']
    df['hhsize_to_bedrooms'] = df['hhsize']/df['bedrooms']
    df['rent_to_hhsize'] = df['v2a1']/df['hhsize']
    df['qmobilephone_to_r4t3'] = df['qmobilephone']/df['r4t3']
    df['qmobilephone_to_v18q1'] = df['qmobilephone']/df['v18q1']
    
extract_features(train)
extract_features(test)
test.info()
individual_features = ['idhogar','dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 
                       'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1', 'parentesco2', 'parentesco3', 
                       'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 
                       'parentesco10', 'parentesco11', 'parentesco12', 'meaneduc', 'instlevel1', 'instlevel2', 
                       'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
                       'instlevel9', 'age']
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

# get list features which will be scaled
list_features = list(set(list(train)) - set(['Id', 'idhogar', 'Target']))

# create a temp set
scaled_train = train.copy()
scaled_test = test.copy()

# fit scaler
scaled_train[list_features] = min_max_scaler.fit_transform(train[list_features])
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaled_test[list_features] = min_max_scaler.fit_transform(test[list_features])

# transform
#scaled_train[list_features] = min_max_scaler.transform(train[list_features])
#scaled_test[list_features] = min_max_scaler.transform(test[list_features])

scaled_train.describe()
train.describe()
#scaled_train = scaled_train.drop(columns=['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq'])
#scaled_test = scaled_test.drop(columns=['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq'])
head_household_train = scaled_train[scaled_train['parentesco1']==1]
head_household_test = scaled_test[scaled_test['parentesco1']==1]
member_household_train = scaled_train[scaled_train['parentesco1']!=1][individual_features]
member_household_test = scaled_test[scaled_test['parentesco1']!=1][individual_features]
#member_household_train = scaled_train[scaled_train['parentesco1']!=1].drop(columns=['Id', 'Target'])
#member_household_test = scaled_test[scaled_test['parentesco1']!=1].drop(columns=['Id'])
def concatenate_features(head_household, member_household):
    """
    inputs are the dataframe
    """    
    list_idhogar = []
    features = -np.ones((head_household.shape[0], (head_household.shape[1]-2)+(member_household.shape[1]-1)*12))
    for i in tqdm(range(len(head_household))):
        idhogar = head_household.iloc[i]['idhogar']
        members = member_household[member_household['idhogar']==idhogar].sort_values(by=['age'])
        members = members.drop(columns=['idhogar'])
        list_idhogar.append(idhogar)
        temp_head_household = head_household[head_household['idhogar']==idhogar].drop(columns=['idhogar', 'Id'])
        current_index = temp_head_household.shape[1]
        features[i][:current_index] = np.array(temp_head_household)
        for j in range(len(members)):
            next_index = current_index + members.shape[1]
            features[i][current_index:next_index] = np.array(members.iloc[j])
            current_index = next_index
    return (features, list_idhogar)
            
train_target = head_household_train['Target']
head_household_train = head_household_train.drop(columns='Target')
train_features, train_idhogar = concatenate_features(head_household_train, member_household_train)
test_features, test_idhogar = concatenate_features(head_household_test, member_household_test)
Y_train = np.array(train_target)
X_train = train_features
X_test = test_features
"""
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
min_max_scaler.fit(train_features)
min_max_scaler.fit(test_features)
X_train = min_max_scaler.transform(train_features)
X_test = min_max_scaler.transform(test_features)

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("X_test shape: ", X_test.shape)
"""
# calculate class weigths because of imbalanced classes
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train.flatten()), Y_train.flatten())
dict_class_weights = dict(enumerate(class_weights))
print("Class weights: ", dict_class_weights)
# Transform Y_train to multi-class matrix
Y_train = np.array(Y_train, dtype=int)
lb = preprocessing.LabelBinarizer()
lb.fit(Y_train)
print("Class: ", lb.classes_)
Y_train = lb.transform(Y_train)
print(Y_train[0:4])
# import library
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
# second model:
# constructing model
np.random.seed(0)
model_neuron = Sequential()
#model.add(Dense(output_dim=2048, input_shape=(X_train.shape[1],),
#               W_regularizer=l2(1.0), activation='relu'))
#model.add(Dense(output_dim=512,activation='relu'))
#model.add(Dropout(.3))
#model.add(Dense(output_dim=256,activation='relu',input_shape=(X_train.shape[1],)))
#model.add(Dense(output_dim=128,activation='relu',input_shape=(X_train.shape[1],)))
#model.add(Dense(output_dim=64,activation='relu',input_shape=(X_train.shape[1],), W_regularizer=l2(1.0)))
model_neuron.add(Dense(output_dim=4, input_shape=(X_train.shape[1],), W_regularizer=l2(1.0)))
model_neuron.add(Activation('softmax'))
model_neuron.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
print(model_neuron.summary())
history = model_neuron.fit(X_train, Y_train, nb_epoch=200, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(patience=10)], class_weight=dict_class_weights)
import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
print(X_train.shape)
print(Y_train.shape)
from sklearn.metrics import confusion_matrix
y = [np.argmax(i)+1 for i in Y_train]
Y_predict = model_neuron.predict(X_train)
class_predict = [np.argmax(i)+1 for i in Y_predict]
confusion_matrix(y, class_predict, labels=[1,2,3,4])
# split train/test set
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
temp_class_weights = {}
for (class_, weight) in dict_class_weights.items():
    temp_class_weights[class_+1] = weight
# train on SVM
np.random.seed(1)
X = X_train
Y = y
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=0)
model = SVC(kernel='linear', C=1, class_weight=temp_class_weights).fit(X_train, Y_train)
print("Training score: ", model.score(X_train, Y_train))
print("Validation score: ", model.score(X_val, Y_val))
from sklearn.metrics import f1_score
y_pred = model.predict(X_train)
print("F1 score training: ", f1_score(Y_train, y_pred, average='macro'))
y_pred = model.predict(X_val)
print("F1 score validation: ", f1_score(Y_val, y_pred, average='macro'))
class_predict = model.predict(X_train)
confusion_matrix(Y_train, class_predict, labels=[1,2,3,4])
temp_test = test[['Id', 'idhogar', 'parentesco1']]
#temp_test = temp_test[temp_test['parentesco1']==1]
temp_test_matrix = temp_test.as_matrix()
class_predict = model.predict(X_test)
test_id = []
predict_target = []
for row_index in temp_test_matrix:
    try:
        idhogar_index = test_idhogar.index(row_index[1])
        predict_target.append(class_predict[idhogar_index])
    except ValueError:
        predict_target.append(4)
    test_id.append(row_index[0])
print(sum(np.array(class_predict)==1))
print(sum(np.array(class_predict)==2))
print(sum(np.array(class_predict)==3))
print(sum(np.array(class_predict)==4))
# transfer prediction
sub = pd.DataFrame({'Id':test_id,'Target':predict_target})
output = sub[['Id','Target']]
output.to_csv("output_linear.csv",index = False)
output
