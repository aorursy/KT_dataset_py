import numpy as np 

import pandas as pd

import random



from keras import layers, optimizers, regularizers

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras.models import Sequential



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import preprocessing, model_selection 
data = pd.read_csv("../input/mushrooms.csv")

data.tail(5)
np.unique(data["class"].values, return_counts=True)
def injectNAN(df, ratio, columns):      

    for i in range(int(len(df)*len(data.columns.values)*ratio)):

        df.iloc[random.randint(0,8123)][random.randint(1,columns)] =np.nan

    return(df)
data = injectNAN(data, 0.5,22)

data.tail()
colList = list(data.columns.values)[1:]

data = pd.get_dummies(data, columns=colList)



def toNumeric(s): 

    if s == "e": 

        return(0)

    elif s == "p": 

        return(1)



data["class"] = data["class"].apply(toNumeric)

data.tail(5)
X = data.iloc[:,1:].values # first columns

Y = data.iloc[:,0:1].values # last columns



X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.03)



print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
shroomModel2 = Sequential()

# layer 1

shroomModel2.add(Dense(30, input_dim=117, activation='relu', name='fc0',kernel_regularizer=regularizers.l2(0.01)))



#layer 2

shroomModel2.add(Dense(1, name='fc2',bias_initializer='zeros'))

shroomModel2.add(Activation('sigmoid'))



shroomModel2.summary()
shroomModel2.compile(optimizer = "adam", loss = "logcosh", metrics = ["binary_accuracy"])
shroomModel2.fit(x = X_train, y = Y_train, epochs = 30,verbose=1, batch_size = 64,validation_data=(X_test, Y_test))
from scipy.stats import chi2_contingency
df = pd.read_csv("../input/mushrooms.csv")



factors_paired = [(i,j) for i in df.columns.values for j in df.columns.values] 



chi2, p_values =[], []



for f in factors_paired:

    if f[0] != f[1]:

        chitest = chi2_contingency(pd.crosstab(df[f[0]], df[f[1]])) # Chi2 test for every contingency table possible

        chi2.append(chitest[0])

        p_values.append(chitest[1])

    else:      # for same factor pair

        chi2.append(0)

        p_values.append(0)

    

chi2 = np.array(chi2).reshape((23,23)) # shape it as a matrix

chi2 = pd.DataFrame(chi2, index=df.columns.values, columns=df.columns.values) # then a df for convenience



p_values = np.array(p_values).reshape((23,23)) # shape it as a matrix

p_values = pd.DataFrame(p_values, index=df.columns.values, columns=df.columns.values) # then a df for convenience
chi2.head()
p_values[(p_values >= 0.05)]
sns.heatmap(chi2,vmax=4000, center=1,square=True,robust=False,xticklabels=True , yticklabels=True, cmap="YlGnBu", linewidths=.5)

plt.show()
dropped_variables= ["cap-shape", "cap-surface","cap-color", "gill-attachment", "gill-spacing", "stalk-shape", "veil-type","veil-color", "ring-number","habitat","population","stalk-surface-below-ring","stalk-color-above-ring"]

data = pd.read_csv("../input/mushrooms.csv").drop(dropped_variables,axis=1)

data.head()
colList = list(data.columns.values)[1:]

data = pd.get_dummies(data, columns=colList)



data["class"] = data["class"].apply(toNumeric)
X = data.iloc[:,1:].values # first columns

Y = data.iloc[:,0:1].values # last columns



X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.03)



print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
shroomModel3 = Sequential()

# layer 1

shroomModel3.add(Dense(30, input_dim=57, activation='relu', name='fc0',kernel_regularizer=regularizers.l2(0.01)))



#layer 2

shroomModel3.add(Dense(1, name='fc2',bias_initializer='zeros'))

shroomModel3.add(Activation('sigmoid'))



shroomModel3.summary()
shroomModel3.compile(optimizer = "adam", loss = "logcosh", metrics = ["binary_accuracy"])
shroomModel3.fit(x = X_train, y = Y_train, epochs = 40,verbose=1, batch_size = 128,validation_data=(X_test, Y_test))
preds = shroomModel3.evaluate(x = X_test, y = Y_test)

print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))