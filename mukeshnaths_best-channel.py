import pandas as pd
import numpy as np
from numpy.random import choice, randint
from sklearn.metrics import classification_report
import tensorflow
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
Dataset = pd.read_pickle("./Dataset.pkl")
Dataset_slice = pd.DataFrame(Dataset.iloc[:,0:10])

Dataset_encoded = pd.get_dummies(Dataset_slice.iloc[:,2:])
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

Data_encoded = ohe.fit_transform(Dataset_encoded).toarray()

Data_encoded_df = pd.DataFrame(Data_encoded)

Dataset_fin = pd.concat([Dataset.iloc[:,0:2], Data_encoded_df, 
                         Dataset.iloc[:,10:]], axis = 1, sort = False)
Dataset_NN = Dataset_fin.copy()
Dataset_NN.drop(Dataset_NN.columns[[24,25,26,27,28,29]], axis=1, inplace=True)
X = Dataset_NN.iloc[:,2:-1].astype(float)
Y = Dataset_NN.iloc[:,-1]
encoder = LabelEncoder()
encoder.fit(Y)

encoded_Y = encoder.transform(Y)

y = np_utils.to_categorical(encoded_Y)
labels = encoder.classes_
labels
def baseline_model():
    model=Sequential()
    model.add(Dense(36, input_dim=24, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(12, activation = 'relu'))
    model.add(Dense(4, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=100, verbose=1)
kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_predict(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print(results)
#estimator.fit(X,y)
df_res = pd.DataFrame(results)
df_res
estimator.predict_proba()
Dataset_test = pd.read_pickle("./Dataset_test.pkl")
Dataset_slice_t = pd.DataFrame(Dataset_test.iloc[:,0:10])

Dataset_encoded_t = pd.get_dummies(Dataset_slice_t.iloc[:,2:])
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

Dataset_encoded_t = ohe.fit_transform(Dataset_encoded_t).toarray()

Data_encoded_df_t = pd.DataFrame(Dataset_encoded_t)

Dataset_fin_t = pd.concat([Dataset_test.iloc[:,0:2], Data_encoded_df_t, 
                         Dataset_test.iloc[:,10:]], axis = 1, sort = False)
Dataset_NN_t = Dataset_fin_t.copy()
Dataset_NN_t.drop(Dataset_NN_t.columns[[24,25,26,27,28,29]], axis=1, inplace=True)
X_t = Dataset_NN_t.iloc[:,2:-1].astype(float)
Y_t = Dataset_NN_t.iloc[:,-1]
encoder_t = LabelEncoder()
encoder_t.fit(Y_t)
encoded_Y_t = encoder.transform(Y_t)

y_t = np_utils.to_categorical(encoded_Y_t)
y_actual = []
y_pred = []

for yt in Y_t:
    y_actual.append(yt)
    
for index, xt in X_t.iterrows():
    next_ind = index+1
    y_pred.append(labels[int(estimator.predict_proba(X_t[index:next_ind]))])
y_actual
y_pred
report = classification_report(y_actual, y_pred, target_names = labels)
print(report)


encoder_t.classes_
results.predict(X_t[120:121])
df_pred

from keras.utils.vis_utils import plot_model
plot_model(estimator.model, to_file='model.png')
#plot_model(estimator.model)


