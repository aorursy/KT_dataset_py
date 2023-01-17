# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib .pyplot as plt
from sklearn import preprocessing
import itertools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def normalize(data):
    headers = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research', 'Chance_of_Admit']
    x = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)
    data.columns = headers
    
    return(data)
def visualize(data):
    data.hist()
    plt.show()
    #input("Enter to view density plot: ")
    data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
    plt.show()
    #input("Press enter to view correlation plot: ")
    
    corr = data.corr()
    names = list(data)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,8,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()
def split_data(data):
    msk = np.random.rand(len(data)) < 0.8
    train_data=data[msk] 
    test_data =data[~msk]    
    
    return (train_data, test_data)
def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=True):    
         return tf.estimator.inputs.pandas_input_fn(       
         x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),       
         y = pd.Series(data_set[LABEL].values),       
         batch_size=n_batch,          
         num_epochs=num_epochs,       
         shuffle=shuffle)
df = pd. read_csv("../input/Admission_Predict_Ver1.1.csv")
data = df.drop(columns="Serial No.")
visualize(data)
data
data = normalize(data)
data.columns
train, test = split_data(data)
print(train.shape, test.shape)
col_names = list(data.columns)
LABEL = col_names.pop()
FEATURES = col_names
print(FEATURES, LABEL)
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
estimator = tf.estimator.LinearRegressor(feature_columns=feature_cols, model_dir="train")
estimator.train(input_fn=get_input_fn(train,                                       
                                           num_epochs=None,                                      
                                           n_batch = 128,                                      
                                           shuffle=False),                                      
                                           steps=10000)
ev = estimator.evaluate(    
          input_fn=get_input_fn(test,                          
          num_epochs=1,                          
          n_batch = 128,                          
          shuffle=False))
train["Chance_of_Admit"].describe()
y = estimator.predict(    
         input_fn=get_input_fn(test,                          
         num_epochs=1,                          
         n_batch = 100,                          
         shuffle=False))
predictions = list(p["predictions"] for p in itertools.islice(y, len(test["Chance_of_Admit"])))
print("Predictions: {}".format(str(predictions)))
predict = np.array(predictions[:])
predict.shape
shaped = np.reshape(predict, (len(test["Chance_of_Admit"]),))
print(len(shaped))
print(len(test["Chance_of_Admit"]))
rmse = (np.square(test["Chance_of_Admit"] - shaped)).mean(axis=None)
print(rmse)
plt.scatter(test["Chance_of_Admit"], shaped)
error = test["Chance_of_Admit"].values - shaped
error
plt.hist(error, bins=20)

