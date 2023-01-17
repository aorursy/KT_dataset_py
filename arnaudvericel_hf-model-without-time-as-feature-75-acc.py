# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import visualisation libraries

import matplotlib.pyplot as plt

import seaborn as sns
# read data

data = pd.read_csv(os.path.join(dirname, filename))
# describe the data in general terms

data.describe()
# get some info

data.info()
# Let's start by looking at the distribution of alive vs. deceased.

n_alive = len(data[data["DEATH_EVENT"]==0])

n_deceased = len(data[data["DEATH_EVENT"]==1])

plt.figure(figsize=(10, 10))

plt.pie((n_alive, n_deceased), labels=("alive", "deceased"), autopct='%1.2f%%')

plt.show()
# Let's plot the number of deceased vs. alive people sorted by gender

sns.countplot(x="DEATH_EVENT", data=data, hue="sex")
# number of male vs. female in the dataset

sns.countplot(x="sex", data=data)
# Let's plot the correlation degree wrt the death event

data.corr()["DEATH_EVENT"].sort_values().plot(kind="bar")
# final DataFrame without uncorrelated features

dropped = ["sex", "time", "smoking", "diabetes"]

final_data = data.drop(dropped, axis=1)

final_data
# Let's look at the effet of high blood pressure more precisely in the case of the deceased population

deceased = final_data[final_data["DEATH_EVENT"]==1]

alive = final_data[final_data["DEATH_EVENT"]==0]

sns.boxplot(data=deceased, x="high_blood_pressure", y="age")

plt.show()
# Let's plot the age distribution for the deceased and alive populations

sns.distplot(alive["age"], label="alive")

sns.distplot(deceased["age"], label="deceased")

plt.legend()

plt.show()
# Finally, let's group by status and look at the mean values of the different features

final_data.groupby("DEATH_EVENT").mean()
# Let's first preprocess the data

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
# X is the features, y is the target class

X = final_data.drop("DEATH_EVENT", axis=1)

y = final_data["DEATH_EVENT"]
# We split the dataset into training (80%) and testing (20%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
# The features are scaled between 0 and 1

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# We'll first try with the kmeans algorithm

from sklearn.neighbors import KNeighborsClassifier

kNeighbors = KNeighborsClassifier()

kNeighbors.fit(X_train, y_train)

train_pred_kneigh = kNeighbors.predict(X_train)

test_pred_kneigh = kNeighbors.predict(X_test)
# Let's print some metrics

from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix

print(f"KneighborsClassifier:\nTest:\n{classification_report(y_test, test_pred_kneigh)}\nTrain:\n{classification_report(y_train, train_pred_kneigh)}")

plt.figure(figsize=(12, 12))

plot_confusion_matrix(kNeighbors, X_test, y_test, cmap="coolwarm")

plt.show()
from sklearn.linear_model import LogisticRegression

LogRegression = LogisticRegression()

LogRegression.fit(X_train, y_train)

train_pred_logreg = LogRegression.predict(X_train)

test_pred_logreg = LogRegression.predict(X_test)
# Let's print some metrics

from sklearn.metrics import classification_report, plot_confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

print(f"Logistic Regression:\nTest:\n{classification_report(y_test, test_pred_logreg)}\nTrain:\n{classification_report(y_train, train_pred_logreg)}")

plt.figure(figsize=(12, 12))

plot_confusion_matrix(LogRegression, X_test, y_test, cmap="coolwarm")

plt.show()
# we prediction probabilities now and manually change the threshold

probabilities = pd.DataFrame(LogRegression.predict_proba(X_test))

thresholds = np.arange(0.1, 0.91, 0.02)

accuracies = np.array([])

f1scores = np.array([])

precisions = np.array([])

recalls = np.array([])



for thresh in thresholds:

    pred_logreg_newthresh = np.empty(len(probabilities), dtype=float)

    for i in range(len(probabilities)):

        if probabilities.iloc[i,0] <= thresh:

            pred_logreg_newthresh[i] = 1

        else:

            pred_logreg_newthresh[i] = 0

    accuracies = np.append(accuracies, accuracy_score(y_test, pred_logreg_newthresh))

    f1scores = np.append(f1scores, f1_score(y_test, pred_logreg_newthresh))

    precisions = np.append(precisions, precision_score(y_test, pred_logreg_newthresh))

    recalls = np.append(recalls, recall_score(y_test, pred_logreg_newthresh))

# DataFrame containing accuracies, f1scores and precisions for these different thresholds

d = {"thresholds": thresholds, "accuracies": accuracies, "f1scores": f1scores, "precision": precisions, "recall": recalls}

df = pd.DataFrame(d)*100
# plot the data

plt.figure(figsize=(10, 6))

plt.scatter(df.thresholds, df.accuracies, color="blue", label="accuracy")

plt.scatter(df.thresholds, df.f1scores, color="orange", label="f1 score")

plt.scatter(df.thresholds, df.precision, color="green", label="precision")

plt.scatter(df.thresholds, df.recall, color="red", label="recall")

plt.xlabel("probability threshold [%]")

plt.ylabel("metric [%]")

plt.axvline(x=50, color="black", linestyle="--", linewidth=0.5, zorder=-34)

plt.legend()

plt.show()
probabilities = pd.DataFrame(LogRegression.predict_proba(X_test))

pred_logreg_bestthresh = np.empty(len(probabilities), dtype=float)

threshold = 0.55

for i in range(len(probabilities)):

    if probabilities.iloc[i,0] <= threshold:

        pred_logreg_bestthresh[i] = 1

    else:

        pred_logreg_bestthresh[i] = 0

print(confusion_matrix(y_test, pred_logreg_bestthresh))

print(f"Accuracy score: {accuracy_score(y_test, pred_logreg_bestthresh)*100:.2f}%")

print(f"f1 score: {f1_score(y_test, pred_logreg_bestthresh)*100:.2f}%")
# We use tensforflow with the keras API

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping
# Sequential neural network that performs classification

nn = Sequential()

nn.add(Dense(units=8, activation="relu"))

nn.add(Dropout(0.3))

nn.add(Dense(units=16, activation="relu"))

nn.add(Dropout(0.3))

nn.add(Dense(units=32, activation="relu"))

nn.add(Dropout(0.3))

nn.add(Dense(units=16, activation="relu"))

nn.add(Dropout(0.3))

nn.add(Dense(units=8, activation="relu"))

nn.add(Dropout(0.3))

nn.add(Dense(units=1, activation="sigmoid"))

nn.compile(optimizer="adam", metrics=["acc"], loss="binary_crossentropy")
# Early stop if accuracy does not improve over 10 epochs

early_stop = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=15)
nn.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stop])
# plot the metrics

metrics = pd.DataFrame(nn.history.history)

metrics.plot()

plt.show()