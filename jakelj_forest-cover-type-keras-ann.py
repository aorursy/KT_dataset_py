import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import re

import sklearn

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer, accuracy_score



from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV

import pandas_profiling

from datetime import datetime

from itertools import combinations, chain

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_df = pd.read_csv('../input/learn-together/train.csv')

test_df = pd.read_csv('../input/learn-together/test.csv')

ID = test_df['Id']




def main(train_df, test_df):

    # this is public leaderboard ratio

    start = datetime.now()

    type_ratio = np.array([0.37053, 0.49681, 0.05936, 0.00103, 0.01295, 0.02687, 0.03242])

    

    total_df = pd.concat([train_df.iloc[:, :-1], test_df])

    

    # Aspect

    total_df["Aspect_Sin"] = np.sin(np.pi*total_df["Aspect"]/180)

    total_df["Aspect_Cos"] = np.cos(np.pi*total_df["Aspect"]/180)

    print("Aspect", (datetime.now() - start).seconds)

    

    # Hillshade

    hillshade_col = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]

    for col1, col2 in combinations(hillshade_col, 2):

        total_df[col1 + "_add_" + col2] = total_df[col2] + total_df[col1]

        total_df[col1 + "_dif_" + col2] = total_df[col2] - total_df[col1]

        total_df[col1 + "_div_" + col2] = (total_df[col2]+0.01) / (total_df[col1]+0.01)

        total_df[col1 + "_abs_" + col2] = np.abs(total_df[col2] - total_df[col1])

    

    total_df["Hillshade_mean"] = total_df[hillshade_col].mean(axis=1)

    total_df["Hillshade_std"] = total_df[hillshade_col].std(axis=1)

    total_df["Hillshade_max"] = total_df[hillshade_col].max(axis=1)

    total_df["Hillshade_min"] = total_df[hillshade_col].min(axis=1)

    print("Hillshade", (datetime.now() - start).seconds)

    

    # Hydrology ** I forgot to add arctan

    total_df["Degree_to_Hydrology"] = ((total_df["Vertical_Distance_To_Hydrology"] + 0.001) /

                                       (total_df["Horizontal_Distance_To_Hydrology"] + 0.01))

    

    # Holizontal

    horizontal_col = ["Horizontal_Distance_To_Hydrology",

                      "Horizontal_Distance_To_Roadways",

                      "Horizontal_Distance_To_Fire_Points"]

    

    

    for col1, col2 in combinations(hillshade_col, 2):

        total_df[col1 + "_add_" + col2] = total_df[col2] + total_df[col1]

        total_df[col1 + "_dif_" + col2] = total_df[col2] - total_df[col1]

        total_df[col1 + "_div_" + col2] = (total_df[col2]+0.01) / (total_df[col1]+0.01)

        total_df[col1 + "_abs_" + col2] = np.abs(total_df[col2] - total_df[col1])

    print("Horizontal", (datetime.now() - start).seconds)

    

    

    def categorical_post_mean(x):

        p = (x.values)*type_ratio

        p = p/p.sum()*x.sum() + 10*type_ratio

        return p/p.sum()

    

    # Wilder

    wilder = pd.DataFrame([(train_df.iloc[:, 11:15] * np.arange(1, 5)).sum(axis=1),

                          train_df.Cover_Type]).T

    wilder.columns = ["Wilder_Type", "Cover_Type"]

    wilder["one"] = 1

    piv = wilder.pivot_table(values="one",

                             index="Wilder_Type",

                             columns="Cover_Type",

                             aggfunc="sum").fillna(0)

    

    tmp = pd.DataFrame(piv.apply(categorical_post_mean, axis=1).tolist()).reset_index()

    tmp["index"] = piv.sum(axis=1).index

    tmp.columns = ["Wilder_Type"] + ["Wilder_prob_ctype_{}".format(i) for i in range(1, 8)]

    tmp["Wilder_Type_count"] = piv.sum(axis=1).values

    

    total_df["Wilder_Type"] = (total_df.filter(regex="Wilder") * np.arange(1, 5)).sum(axis=1)

    total_df = total_df.merge(tmp, on="Wilder_Type", how="left")

    

    for i in range(7):

        total_df.loc[:, "Wilder_prob_ctype_{}".format(i+1)] = total_df.loc[:, "Wilder_prob_ctype_{}".format(i+1)].fillna(type_ratio[i])

    total_df.loc[:, "Wilder_Type_count"] = total_df.loc[:, "Wilder_Type_count"].fillna(0)

    print("Wilder_type", (datetime.now() - start).seconds)

    

    

    # Soil type

    soil = pd.DataFrame([(train_df.iloc[:, -41:-1] * np.arange(1, 41)).sum(axis=1),

                          train_df.Cover_Type]).T

    soil.columns = ["Soil_Type", "Cover_Type"]

    soil["one"] = 1

    piv = soil.pivot_table(values="one",

                           index="Soil_Type",

                           columns="Cover_Type",

                           aggfunc="sum").fillna(0)

    

    tmp = pd.DataFrame(piv.apply(categorical_post_mean, axis=1).tolist()).reset_index()

    tmp["index"] = piv.sum(axis=1).index

    tmp.columns = ["Soil_Type"] + ["Soil_prob_ctype_{}".format(i) for i in range(1, 8)]

    tmp["Soil_Type_count"] = piv.sum(axis=1).values

    

    total_df["Soil_Type"] = (total_df.filter(regex="Soil") * np.arange(1, 41)).sum(axis=1)

    total_df = total_df.merge(tmp, on="Soil_Type", how="left")

    

    for i in range(7):

        total_df.loc[:, "Soil_prob_ctype_{}".format(i+1)] = total_df.loc[:, "Soil_prob_ctype_{}".format(i+1)].fillna(type_ratio[i])

    total_df.loc[:, "Soil_Type_count"] = total_df.loc[:, "Soil_Type_count"].fillna(0)

    print("Soil_type", (datetime.now() - start).seconds)

    

    icol = total_df.select_dtypes(np.int64).columns

    fcol = total_df.select_dtypes(np.float64).columns

    total_df.loc[:, icol] = total_df.loc[:, icol].astype(np.int32)

    total_df.loc[:, fcol] = total_df.loc[:, fcol].astype(np.float32)

    return total_df



total_df = main(train_df, test_df)

one_col = total_df.filter(regex="(Type\d+)|(Area\d+)").columns

total_df = total_df.drop(one_col, axis=1)
y = train_df["Cover_Type"].values

X = total_df[total_df["Id"] <= 15120].drop("Id", axis=1)

test = total_df[total_df["Id"] > 15120].drop("Id", axis=1)
from sklearn.ensemble import ExtraTreesClassifier

import statistics 

scores = []



for i in range(10):

    x_train, x_test, y_train, y_test = train_test_split(X,y, train_size = 0.6, shuffle = True, random_state=1)



    model = ExtraTreesClassifier()

    model.fit(x_train,y_train)

    scores.append((accuracy_score(y_test,model.predict(x_test) )))  

etc_preds = model.predict(test)

print(round(statistics.mean(scores),4))
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

onehotencoder = OneHotEncoder()

y2 = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)

test = sc_X.transform(test)
optimizer = 'adam'

kernel_initializer= 'Orthogonal'



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 47, kernel_initializer=kernel_initializer, activation = 'relu', input_dim = 47))

classifier.add(Dropout(rate = 0.1))



# Adding the second hidden layer

classifier.add(Dense(units = 200, kernel_initializer= kernel_initializer, activation = 'relu'))

classifier.add(Dropout(rate = 0.1))





# Adding the third hidden layer

classifier.add(Dense(units = 50, kernel_initializer= kernel_initializer, activation = 'relu'))

classifier.add(Dropout(rate = 0.1))



# Adding the output layer

classifier.add(Dense(units = 7, kernel_initializer= kernel_initializer, activation = 'softmax'))



# Compiling the ANN

classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X,y2, batch_size = 1000, epochs = 10000, callbacks=[tf.keras.callbacks.TensorBoard('logs')])



#Predicting values for classes [classes from 1-7, values predicted from 0-6 so add 1]

preds = classifier.predict_classes(test) + 1


'''

if you're running this on your own computer you can use the code below to see the model and loss curve etc



# Load TENSORBOARD

%load_ext tensorboard

# Start TENSORBOARD

%tensorboard --logdir logs --host localhost



'''
# predcitions by NN

pd.Series(preds).value_counts()
# predictions by extra trees classifier

pd.Series(etc_preds).value_counts()  # not scaled
submission = pd.DataFrame({ 'Id': ID,

                            'Cover_Type': preds })

submission.to_csv("submission_ann1.csv", index=False)