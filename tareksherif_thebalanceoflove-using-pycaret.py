!pip install pycaret==2.0
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.classification import *
from sklearn.model_selection import train_test_split
sampleNumber=10000

d = {
    'x1': np.random.randint(0,3,sampleNumber) ,#Make sacrifices
    'x2': np.random.randint(0,3,sampleNumber) ,#Punctuality
    'x3': np.random.randint(0,2,sampleNumber) ,#Not feeling bored while you are together
    'x4': np.random.randint(0,3,sampleNumber) ,#You evaluate gifts
    'x5': np.random.randint(0,2,sampleNumber) ,#Take care of my problems
    'x6': np.random.randint(0,2,sampleNumber) ,#Rai respect 
    'y' : np.zeros(sampleNumber, dtype=bool)
    }
data  = pd.DataFrame(data=d)


 
data["y"]=(data['x1'] +   data['x2'] +    data['x3'] +   data['x4'] +   data['x5']+    data['x6'])>4


X=data[['x1','x2','x3','x4','x5','x6']]
y=data[["y"]]


X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=.20, random_state=42)
train_Data=X_train
train_Data["y"]=y_train
del X_train
del y_train

clf= setup(data = train_Data, target = "y")
compare_models()
lr = create_model('lr')
plot_model(lr,"confusion_matrix")
evaluate_model(lr)
lr_pred = predict_model(lr, data = X_test) #new_data is pd dataframe
lr_pred
y_test.reset_index(drop=True, inplace=True)
y_test
y_test=y_test.astype("int").values.T
correct_predictions = np.nonzero(lr_pred["Label"].values==y_test)[0]
incorrect_predictions = np.nonzero(lr_pred["Label"].values!=y_test)[0]
print(len(correct_predictions)," classified correctly")
print(len(incorrect_predictions)," classified incorrectly")