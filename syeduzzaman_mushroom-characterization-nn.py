import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn import utils

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.preprocessing import LabelEncoder 



# Load the dataset

df = pd.read_csv("../input/mushrooms.csv")

df.head(10)
df=df.dropna()
lb_make = LabelEncoder()



df["bruises"] = lb_make.fit_transform(df["bruises"])



df["gill-spacing"] = lb_make.fit_transform(df["gill-spacing"])

df["gill-size"] = lb_make.fit_transform(df["gill-size"])

df["gill-color"] = lb_make.fit_transform(df["gill-color"])

df["stalk-shape"] = lb_make.fit_transform(df["stalk-shape"])

df["stalk-root"] = lb_make.fit_transform(df["stalk-root"])

df["stalk-surface-above-ring"] = lb_make.fit_transform(df["stalk-surface-above-ring"])

df["stalk-surface-below-ring"] = lb_make.fit_transform(df["stalk-surface-below-ring"])



df["stalk-color-above-ring"] = lb_make.fit_transform(df["stalk-color-above-ring"])

df["stalk-color-below-ring"] = lb_make.fit_transform(df["stalk-color-below-ring"])



df["veil-type"] = lb_make.fit_transform(df["veil-type"])

df["veil-color"] = lb_make.fit_transform(df["veil-color"])



df["ring-number"] = lb_make.fit_transform(df["ring-number"])

df["ring-type"] = lb_make.fit_transform(df["ring-type"])

df["spore-print-color"] = lb_make.fit_transform(df["spore-print-color"])

df["population"] = lb_make.fit_transform(df["population"])

df["habitat"] = lb_make.fit_transform(df["habitat"])



df["class"] = lb_make.fit_transform(df["class"])

df["odor"] = lb_make.fit_transform(df["odor"])

df["cap-shape"] = lb_make.fit_transform(df["cap-shape"])

df["cap-surface"] = lb_make.fit_transform(df["cap-surface"])

df["cap-color"] = lb_make.fit_transform(df["cap-color"])

df["gill-attachment"] = lb_make.fit_transform(df["gill-attachment"])

df.head(10)


correlations = df.corr()

correlations

y=df.iloc[:,0]

x=df.iloc[:,1:23]

xtrain, xtest, ytrain, ytest = train_test_split( 

        x, y, test_size = 0.25, random_state = 0)



scaler = StandardScaler()

xtrain = scaler.fit_transform(xtrain)

xtest = scaler.fit_transform(xtest)
from sklearn.neural_network import MLPClassifier

# 5 hidden layers, Neurons=2/3*input layer+output layer=16, learning rate=0.001, activation function= Relu, Solver= Stochastic gradient descent 



#model

mlp = MLPClassifier(activation='relu',solver='sgd', alpha=1e-5,learning_rate_init=0.001,hidden_layer_sizes=(16,16,16,16,16),max_iter=1000,random_state=100)

mlp.fit(xtrain,ytrain)
#prediction 

predictions = mlp.predict(xtest)

Model_accuracy = (mlp.score(xtest, ytest))*100

Model_accuracy
print(confusion_matrix(ytest,predictions))
print(classification_report(ytest,predictions))