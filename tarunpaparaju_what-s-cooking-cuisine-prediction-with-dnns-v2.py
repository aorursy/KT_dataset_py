import json
import numpy as np

with open('../input/train.json') as json_file:  
    train_data = json.load(json_file)
    
with open('../input/test.json') as json_file:  
    test_data = json.load(json_file)
    
cuisines = []

for i in range(0, len(train_data)):
    cuisines.append(train_data[i]['cuisine'])
    
cuisines = list(set(cuisines))

ingredients_train = []

for i in range(0, len(train_data)):
    for j in range(0, len(train_data[i]['ingredients'])):
        ingredients_train.append(train_data[i]['ingredients'][j])
        
ingredients_train = list(set(ingredients_train))

# ingredients_test = []

# for i in range(0, len(test_data)):
    # for j in range(0, len(test_data[i]['ingredients'])):
        # ingredients_test.append(test_data[i]['ingredients'][j])
        
# ingredients_test = list(set(ingredients_test))

ingredients = ingredients_train # list(set(ingredients_train) | set(ingredients_test))

for i in range(0, len(train_data)):
    train_data[i]['cuisine'] = list.index(cuisines, train_data[i]['cuisine']) + 1
    
    for j in range(0, len(train_data[i]['ingredients'])):
        train_data[i]['ingredients'][j] = list.index(ingredients, train_data[i]['ingredients'][j]) + 1
train_features = []
train_targets = []

for i in range(0, len(train_data)):
    train_features.append(train_data[i]['ingredients'])
    train_targets.append([train_data[i]['cuisine']])

oneHotFeatures = np.zeros((len(train_data), len(ingredients)))

for i in range(0, len(train_data)):
    for j in range(0, len(ingredients)):
        if j + 1 in train_data[i]['ingredients']:
            oneHotFeatures[i][j] = 1
        else:
            oneHotFeatures[i][j] = 0
            
oneHotTargets = np.zeros((len(train_data), len(cuisines)))
            
for i in range(0, len(train_data)):
    for j in range(0, len(cuisines)):
        if j + 1 == train_data[i]['cuisine']:
            oneHotTargets[i][j] = 1
        else:
            oneHotTargets[i][j] = 0
            
oneHotFeatures = np.int32(oneHotFeatures)
train_features = oneHotFeatures

oneHotTargets = np.int32(oneHotTargets) 
train_targets = oneHotTargets
# import sklearn
# from sklearn.ensemble import GradientBoostingClassifier
# import numpy as np

# import xgboost
# from xgboost import XGBClassifier

# model = XGBClassifier(n_estimators=10, max_depth=6)
# model.fit(train_features, train_targets)

import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

model = Sequential()

model.add(Dense(200, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(400, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.5)) 

model.add(Dense(300, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# model.add(Dense(200, activation='sigmoid'))
# model.add(Dropout(0.225))
# model.add(BatchNormalization()) 

# model.add(Dense(400, activation='sigmoid'))
# model.add(Dropout(0.225))   
# model.add(BatchNormalization())  

# model.add(Dense(300, activation='sigmoid'))
# model.add(Dropout(0.225))
# model.add(BatchNormalization()) 

model.add(Dense(20, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(train_features[0:36819], train_targets[0:36819], epochs=10)
test_features = []

for i in range(0, len(test_data)):
    test_features.append(test_data[i]['ingredients'])
    
oneHotFeatures = np.zeros((len(test_data), len(ingredients)))

for i in range(0, len(test_data)):
    for j in range(0, len(ingredients)):
        if ingredients[j] in test_data[i]['ingredients']:
            oneHotFeatures[i][j] = 1
        else:
            oneHotFeatures[i][j] = 0
            
oneHotFeatures = np.int32(oneHotFeatures)
test_features = oneHotFeatures
predictions = model.predict(test_features)

cuisinePredictions = []

for i in range(0, len(predictions)):
    cuisinePredictions.append(cuisines[list.index(list(predictions[i]), max(predictions[i]))])
    
ids = np.int32(np.array([test_data[i]['id'] for i in range(0, len(test_data))]))

import pandas as pd

submission = pd.DataFrame(ids, columns={"id"})
submission['cuisine'] = cuisinePredictions
submission = submission[["id", "cuisine"]]
submission.to_csv('cuisine-prediction-submission-6.csv', index=False)
predictions =  model.predict(test_features) # sess.run(y, feed_dict={x:test_features})

cuisinePredictions = []

for i in range(0, len(predictions)):
    cuisinePredictions.append(cuisines[list.index(list(predictions[i]), max(predictions[i]))])
    
ids = np.int32(np.array([test_data[i]['id'] for i in range(0, len(test_data))]))

import pandas as pd

submission = pd.DataFrame(ids, columns={"id"})
submission['cuisine'] = cuisinePredictions
submission = submission[["id", "cuisine"]]
submission.to_csv('cuisine-prediction-submission-0.csv', index=False)