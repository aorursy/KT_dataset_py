import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data_path = '../input'
train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
train_data.head(3)
tablets = []
for owns, num_tablets in zip(train_data['v18q'], train_data['v18q1']):
    if owns == 0:
        tablets += [0]
    else:
        tablets += [num_tablets]
        
train_data['v18q1'] = tablets
train_data.head(3)
tmp_educ = []
sq_tmp_educ = []

for efe, efa, meduc, sq in zip(train_data['edjefe'], train_data['edjefa'], train_data['meaneduc'], train_data['SQBmeaned']):
    new_educ = meduc
    if new_educ != new_educ:
        if efa == "no":
            if efe == "no":
                new_educ = 0.0
            else:
                new_educ = float(efe)
        else:
            if efe == "no":
                new_educ = float(efa)
            else:
                new_educ = float(efe) + float(efa)
    if meduc != new_educ:
        print(meduc, ",", new_educ)
    tmp_educ += [new_educ]

    sq_tmp_educ += [new_educ ** 2]
        
train_data['meaneduc'] = tmp_educ
train_data['SQBmeaned'] = sq_tmp_educ
from sklearn.preprocessing import Imputer

v2a1 = []
rez_esc = []
for rentpay, rez in zip(train_data['v2a1'], train_data['rez_esc']):
    if rentpay != rentpay:
        v2a1 += [0]
    else:
        v2a1 += [rentpay]

    if rez != rez:
        rez_esc += [0]
    else:
        rez_esc += [rez]

        
#train_data['v2a1'] = v2a1
train_data['rez_esc'] = rez_esc
train_data.info(verbose=True, null_counts=True)
depend = []
for dependency, children, olds, total in zip(train_data['dependency'], train_data['hogar_nin'], train_data['hogar_mayor'], train_data['hogar_total']):
    calc_depend = False
    if depend != depend:
        calc_depend = True
    elif (dependency == "yes" or dependency == "no"):
        calc_depend = True

    #Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
    if calc_depend:
        i = (children + olds) / (total - children - olds)
    else:
        i = float(dependency)

    depend += [i]

train_data['dependency'] = depend

chw = []
for nin, adul in zip(train_data['hogar_nin'], train_data['hogar_adul']):
    if adul == 0:
        chw += [nin * 2]
    else:
        chw += [nin / adul]

train_data['child_weight'] = (train_data['hogar_nin'] + train_data['hogar_mayor']) / train_data['hogar_total']
train_data['child_weight2'] = chw
train_data['child_weight3'] = train_data['r4t1'] / train_data['r4t3']
train_data['work_power'] = train_data['dependency'] * train_data['hogar_adul']
train_data['SQBworker'] = train_data['hogar_adul'] ** 2
train_data['rooms_per_person'] = train_data['rooms'] / (train_data['tamviv'])
train_data['bedrooms_per_room'] = train_data['bedrooms'] / train_data['rooms']
train_data['female_weight'] = train_data['r4m3'] / train_data['r4t3']
#Predict v2a1 for household.
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

v2a1_drop = ['edjefe', 'edjefa', 'v2a1', 'Id']
train_hh = train_data.query('parentesco1 == 1').drop('Target', axis=1)
v2a1_train_tmp = train_hh.query('v2a1 == v2a1')
v2a1_train = v2a1_train_tmp.drop(v2a1_drop, axis=1)
v2a1_train = v2a1_train.drop('idhogar', axis=1)
v2a1_train_target = v2a1_train_tmp['v2a1'].copy()
#std_scaler = StandardScaler()
#std_scaler.fit(v2a1_train)
#v2a1_train = std_scaler.transform(v2a1_train)

v2a1_test_tmp = train_hh.query('v2a1 != v2a1').drop(v2a1_drop, axis=1)
v2a1_test = v2a1_test_tmp.drop('idhogar', axis=1)
forest = GradientBoostingRegressor(n_estimators=400, learning_rate=0.2, max_depth=5, random_state=0)
forest.fit(v2a1_train, v2a1_train_target)
print("score: ", forest.score(v2a1_train, v2a1_train_target))
v2a1_train_pred = forest.predict(v2a1_train)
forest_mse = mean_squared_error(v2a1_train_target, v2a1_train_pred)
print("RMSE: ", np.sqrt(forest_mse))

v2a1_pred = pd.DataFrame({'ID': v2a1_test_tmp['idhogar'], 'v2a1_pred': forest.predict(v2a1_test)})

v2a1 = []
for rent, idhh in zip(train_data['v2a1'], train_data['idhogar']):
    if rent != rent:
        i = 0
        for rent_p, index in zip(v2a1_pred['v2a1_pred'], v2a1_pred['ID']):
            if index == idhh:
                i = rent_p
                break
        if i == 0:
            for rent_org, index in zip(v2a1_train_target, v2a1_train_tmp['idhogar']):
                if index == idhh:
                    i = rent_org
        if i < 0:
            i = 0
        v2a1 += [i]
    else:
        v2a1 += [rent]

train_data['v2a1'] = v2a1
train_data['hogar_total'].value_counts()
corr_matrix = train_data.corr()
corr_matrix['Target'].sort_values(ascending=False)
from sklearn.model_selection import train_test_split

X_train = train_data.drop('Target', axis=1)
y_train = train_data['Target'].copy()

#X_train, X_validate, y_train, y_validate = train_test_split(X, y, random_state=0)
X_train = X_train.drop(['Id', 'idhogar', 'edjefe', 'edjefa'], axis=1)
#data augmentation
from sklearn.utils import shuffle

X_train_da = pd.concat([X_train, X_train, X_train])
#X_train_da = pd.concat([X_train_da, X_train])

#X_train['Noise1'] = np.random.rand()
#X_train['Noise2'] = np.random.rand()
X_train_da['age'] += np.random.randint(2) + np.random.randint(2) - 2
X_train_da['SQBage'] = X_train_da['age'] ** 2
X_train_da['hogar_total'] += np.random.randint(3) - 1
X_train_da['SQBhogar_total'] = X_train['hogar_total'] ** 2
X_train_da['v2a1'] += np.random.randint(10) * 1000 - 5000

X_train_da = pd.concat([X_train, X_train_da])

y_train = pd.concat([y_train, y_train, y_train, y_train])
#y_train = pd.concat([y_train, y_train])

X_train_da, y_train = shuffle(X_train_da, y_train)
X_train_dummy = pd.get_dummies(X_train_da)
X_train_dummy.info(verbose=True, null_counts=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='mean')
imputer.fit(X_train_dummy)
X_train_dummy = imputer.transform(X_train_dummy)

scaler = MinMaxScaler()
scaler.fit(X_train_dummy)
X_train_scaled = scaler.transform(X_train_dummy)

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers

n_classes = 5

y_train_keras = keras.utils.to_categorical(y_train, n_classes)

(n_samples, n_features) = X_train_scaled.shape

model = Sequential()
model.add(Dense(units=500, activation="relu", input_shape=(n_features, )))
model.add(Dropout(0.2))
model.add(Dense(units=500, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=500, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=n_classes, activation="softmax"))
          
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train_keras, epochs=100, validation_split=0.1, batch_size=n_samples, verbose=0)
y_train_pred = model.predict_classes(X_train_scaled, verbose=0)
f1_score(y_train, y_train_pred, average='macro')
plt.ylim(0, max(np.r_[history.history['val_loss'], history.history['loss']]))
plt.plot(history.history['val_loss'], c='red')
plt.plot(history.history['loss'], c='green')
#X_validate = X_validate.drop(['Id', 'idhogar', 'edjefe', 'edjefa'], axis=1)
#X_validate['Noise1'] = 0
#X_validate['Noise2'] = 0
#X_validate_dummy = pd.get_dummies(X_validate)
#X_validate_dummy = imputer.transform(X_validate_dummy)
#X_validate_scaled = scaler.transform(X_validate_dummy)

#print("score: ", mlp.score(X_validate_scaled, y_validate))
#y_validate_pred = mlp.predict(X_validate_scaled)

#y_validate_pred = model.predict_classes(X_validate_scaled, verbose=0)

#f1_score(y_validate, y_validate_pred, average='macro')
test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))

tablets = []
for owns, num_tablets in zip(test_data['v18q'], test_data['v18q1']):
    if owns == 0:
        tablets += [0]
    else:
        tablets += [num_tablets]
        
test_data['v18q1'] = tablets

tmp_educ = []
sq_tmp_educ = []

for efe, efa, meduc, sq in zip(test_data['edjefe'], test_data['edjefa'], test_data['meaneduc'], test_data['SQBmeaned']):
    new_educ = meduc
    if new_educ != new_educ:
        if efa == "no":
            if efe == "no":
                new_educ = 0.0
            else:
                new_educ = float(efe)
        else:
            if efe == "no":
                new_educ = float(efa)
            else:
                new_educ = float(efe) + float(efa)
    tmp_educ += [new_educ]

    sq_tmp_educ += [new_educ ** 2]
        
test_data['meaneduc'] = tmp_educ
test_data['SQBmeaned'] = sq_tmp_educ

v2a1 = []
rez_esc = []
for rentpay, rez in zip(test_data['v2a1'], test_data['rez_esc']):
    if rentpay != rentpay:
        v2a1 += [0]
    else:
        v2a1 += [rentpay]

    if rez != rez:
        rez_esc += [0]
    else:
        rez_esc += [rez]

#test_data['v2a1'] = v2a1
test_data['rez_esc'] = rez_esc

depend = []
for dependency, children, olds, total in zip(test_data['dependency'], test_data['hogar_nin'], test_data['hogar_mayor'], test_data['hogar_total']):
    calc_depend = False
    if depend != depend:
        calc_depend = True
    elif (dependency == "yes" or dependency == "no"):
        calc_depend = True

    #Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
    if calc_depend:
        i = (children + olds) / (total - children - olds)
    else:
        i = float(dependency)

    depend += [i]

test_data['dependency'] = depend

chw = []
for nin, adul in zip(test_data['hogar_nin'], test_data['hogar_adul']):
    if adul == 0:
        chw += [nin * 2]
    else:
        chw += [nin/adul]

test_data['child_weight'] = (test_data['hogar_nin'] + test_data['hogar_mayor']) / test_data['hogar_total']
test_data['child_weight2'] = chw
test_data['child_weight3'] = test_data['r4t1'] / test_data['r4t3']
test_data['work_power'] = test_data['dependency'] * test_data['hogar_adul']
test_data['SQBworker'] = test_data['hogar_adul'] ** 2
test_data['rooms_per_person'] = test_data['rooms'] / (test_data['tamviv'])
test_data['bedrooms_per_room'] = test_data['bedrooms'] / test_data['rooms']
test_data['female_weight'] = test_data['r4m3'] / test_data['r4t3']

train_hh = test_data.query('parentesco1 == 1')
v2a1_test_tmp = train_hh.query('v2a1 != v2a1').drop(v2a1_drop, axis=1)
v2a1_test = v2a1_test_tmp.drop('idhogar', axis=1)

(num_target, num_col) = v2a1_test.shape

if (num_target > 0):
    v2a1_pred = pd.DataFrame({'ID': v2a1_test_tmp['idhogar'], 'v2a1_pred': forest.predict(v2a1_test)})

    v2a1 = []
    for rent, idhh in zip(test_data['v2a1'], test_data['idhogar']):
        if rent != rent:
            i = 0
            for rent_p, index in zip(v2a1_pred['v2a1_pred'], v2a1_pred['ID']):
                if index == idhh:
                    i = rent_p
                    break
            if i == 0:
                for rent_org, index in zip(v2a1_train_target, v2a1_train_tmp['idhogar']):
                    if index == idhh:
                        i = rent_org
            if i < 0:
                i = 0
            v2a1 += [i]
        else:
             v2a1 += [rent]

    test_data['v2a1'] = v2a1


test_data_drop = test_data.drop(['Id', 'idhogar', 'edjefe', 'edjefa'], axis=1)
#test_data_drop['Noise1'] = np.random.rand()
#test_data_drop['Noise2'] = 0
test_data_dummy = pd.get_dummies(test_data_drop)
test_data_dummy = imputer.transform(test_data_dummy)
test_data_scaled = scaler.transform(test_data_dummy)

#y_pred = mlp.predict(test_data_scaled)
y_pred = model.predict_classes(test_data_scaled, verbose=0)

result = pd.DataFrame({'Id':test_data['Id'], 'Target':y_pred})
result.to_csv('result1.csv', index=False)