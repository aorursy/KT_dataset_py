# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")
df = data.copy()

df.info()
df.columns
sns.heatmap(df.isnull())

plt.show()
import missingno as msno

msno.bar(df);
sns.heatmap(df.describe()[1:].transpose(), annot= True, fmt=".1f",

            linecolor="black", linewidths=0.3,cmap="Reds_r")

plt.title("Data Summary", fontsize=(15), color="red")

plt.show()
df.columns = ["mean_prof", "std_prof", "kurtosis_prof",

             "skewness_prof", "mean_curve","std_curve",

             "kurtosis_curve","skewness_curve","target"]
plt.figure(figsize=(15,15))



plt.subplot(221)

sns.violinplot(data=df, x="target", y="mean_prof", inner="quartile", palette="OrRd")

plt.xlabel("target", fontsize=12)

plt.ylabel("mean_prof", fontsize=12)



plt.subplot(222)

sns.violinplot(data=df, x="target", y="std_prof", inner="quartile", palette="OrRd")

plt.xlabel("target", fontsize=12)

plt.ylabel("std_prof", fontsize=12)



plt.subplot(223)

sns.violinplot(data=df, x="target", y="kurtosis_prof", inner="quartile", palette="OrRd")

plt.xlabel("target", fontsize=12)

plt.ylabel("kurtosis_prof", fontsize=12)



plt.subplot(224)

sns.violinplot(data=df, x="target", y="skewness_prof", inner="quartile", palette="OrRd")

plt.xlabel("target", fontsize=12)

plt.ylabel("taget", fontsize=12)
y = df.target.values

x = df.drop(["target"], axis = 1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense
def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units=16, activation="relu", input_dim=x_train.shape[1]))

    classifier.add(Dense(units=16, activation="relu"))

    classifier.add(Dense(units=1, activation="sigmoid"))

    classifier.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 30,batch_size=10)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))
history = classifier.fit(x_test, y_test, epochs= 100,validation_split=0.20,

                         batch_size=512)
history_dict = history.history

loss_values = history_dict["loss"]

val_loss_values = history_dict["val_loss"]



plt.plot(loss_values, "bo", label="Traning Loss")

plt.plot(val_loss_values, "b", label="Validation Loss")

plt.title("Traning and Validation Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
acc = history_dict["accuracy"]

val_acc = history_dict["val_accuracy"]

plt.plot(acc, "bo", label="Training accuracy")

plt.plot(val_acc, "b", label="Validation accuracy")

plt.title("Training and Validation accuracy")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()