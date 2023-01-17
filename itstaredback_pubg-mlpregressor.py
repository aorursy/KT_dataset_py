import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from keras import backend as K
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from scipy import stats
def cleanFeatures(df):
    #drop columns that likely don't affect the outcome of the match, the y values, and the dummy values (matchType)
    droppedColumns = df.drop(columns=['Id', 'groupId', 'matchId', 'winPlacePerc', 'matchType'], errors='ignore')

    #get the dummy values for the matchType
    matchTypeDummies = pd.get_dummies(columns=list(df['matchType']), data=df['matchType'].values)

    #standardize everything else from 0-1
    mms = MinMaxScaler()
    scaledDroppedColumns = mms.fit_transform(droppedColumns)

    #create one input tensor
    scaledDroppedColumnsDf = pd.DataFrame(data=scaledDroppedColumns, columns=list(droppedColumns))
    X = pd.concat([scaledDroppedColumnsDf, matchTypeDummies], axis=1)
    
    return X

#read in the CSV file
allColumns = pd.read_csv("../input/train_V2.csv")
X = cleanFeatures(allColumns)

#replace empty win percentages (the labels) with 0.0
allColumns['winPlacePerc'] = allColumns['winPlacePerc'].fillna(0.0)
#remove any other NaN rows as they will just cause problems in the NN
allColumns.dropna()

#get just the labels for win percentage
y = allColumns['winPlacePerc'].values

#split the data at an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
#create a heat map to see if there are any features that are highly correlated
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(allColumns.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
K.clear_session()

model = MLPRegressor(
    hidden_layer_sizes=(200,), 
    activation="relu", 
    solver="adam", 
    alpha=0.0001, 
    batch_size=256, 
    learning_rate="constant", 
    learning_rate_init=0.001, 
    power_t=0.5, 
    momentum=0.9, 
    verbose=1, 
    early_stopping=True) # allow the model to stop early when it is no longer progressing
model.fit(X_train, y_train)
print("Score for test split data: " + str(model.score(X_test, y_test)))
testColumns = pd.read_csv('../input/test_V2.csv') 
testFeatures = cleanFeatures(testColumns)

testFeatures.describe()
model.out_activation_ = 'relu'
placementPredictions = model.predict(testFeatures)
#to assume that above 1.00 is a win, just set those to 1.00
placementPredictions[placementPredictions > 1.0] = 1.0

stats.describe(placementPredictions)
submission = pd.read_csv('../input/sample_submission_V2.csv')
submission['winPlacePerc'] = placementPredictions

submission.head()
submission.to_csv("submission.csv", index=False)