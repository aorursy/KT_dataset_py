import numpy as np

import itertools

import os

import pandas as pd

import seaborn as sea



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import RFE





import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
sea.set_style("darkgrid")
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/"

                   "diabetes.csv")



data.head(10)
data.shape
# disable SettingWithCopyWarning messages

pd.options.mode.chained_assignment = None



data_X = data.loc[:, data.columns != "Outcome"]

data_Y = data[["Outcome"]]



print("data_X info:\n")

data_X.info()

print("\ndata_Y info:\n")

data_Y.info()
data_Y["Outcome"].value_counts()

#ghsgedy
train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y,

                                                    test_size=0.2,

                                                    stratify=data_Y,

                                                    random_state=0)



train_X.reset_index(drop=True, inplace=True);

test_X.reset_index(drop=True, inplace=True);

train_Y.reset_index(drop=True, inplace=True);

test_Y.reset_index(drop=True, inplace=True);
def plots(feature):

    fig = plt.figure(constrained_layout = True, figsize=(10,3))

    gs = gridspec.GridSpec(nrows=1, ncols=4, figure=fig)



    ax1 = fig.add_subplot(gs[0,:3])    

    sea.distplot(train_X.loc[train_Y["Outcome"]==0,feature],

                 kde = False, color = "#004a4d", norm_hist=False,

                 hist_kws = dict(alpha=0.8), bins=40,

                 label="Not Diabetes", ax=ax1);

    sea.distplot(train_X.loc[train_Y["Outcome"]==1,feature],

                 kde = False, color = "#7d0101", norm_hist=False,

                 hist_kws = dict(alpha=0.6), bins=40,

                 label="Diabetes", ax=ax1);

    ax2 = fig.add_subplot(gs[0,3])    

    sea.boxplot(train_X[feature], orient="v", color = "#989100",

                width = 0.2, ax=ax2);

    

    ax1.legend(loc="upper right");
plots("Pregnancies")
Q1 = train_X["Pregnancies"].quantile(0.25)

Q3 = train_X["Pregnancies"].quantile(0.75)

q95th = train_X["Pregnancies"].quantile(0.95)

IQR = Q3 - Q1

UW = Q3 + 1.5*IQR



train_X["Pregnancies"] = np.where(train_X["Pregnancies"] > UW,

                                  q95th, train_X["Pregnancies"])
plots("Glucose")
med = train_X["Glucose"].median()

train_X["Glucose"] = np.where(train_X["Glucose"] == 0, med, train_X["Glucose"])
plots("BloodPressure")
med = train_X["BloodPressure"].median()

q5th = train_X["BloodPressure"].quantile(0.05)

q95th = train_X["BloodPressure"].quantile(0.95)

Q1 = train_X["BloodPressure"].quantile(0.25)

Q3 = train_X["BloodPressure"].quantile(0.75)

IQR = Q3 - Q1

LW = Q1 - 1.5*IQR

UW = Q3 + 1.5*IQR



train_X["BloodPressure"] = np.where(train_X["BloodPressure"] == 0,

                                    med, train_X["BloodPressure"])

train_X["BloodPressure"] = np.where(train_X["BloodPressure"] < LW,

                                    q5th, train_X["BloodPressure"])

train_X["BloodPressure"] = np.where(train_X["BloodPressure"] > UW,

                                    q95th, train_X["BloodPressure"])
plots("SkinThickness")
med = train_X["SkinThickness"].median()

q95th = train_X["SkinThickness"].quantile(0.95)

Q1 = train_X["SkinThickness"].quantile(0.25)

Q3 = train_X["SkinThickness"].quantile(0.75)

IQR = Q3 - Q1

UW = Q3 + 1.5*IQR



train_X["SkinThickness"] = np.where(train_X["SkinThickness"] == 0,

                                    med, train_X["SkinThickness"])

train_X["SkinThickness"] = np.where(train_X["SkinThickness"] > UW,

                                    q95th, train_X["SkinThickness"])
plots("Insulin")
q60th = train_X["Insulin"].quantile(0.60)

q95th = train_X["Insulin"].quantile(0.95)

Q1 = train_X["Insulin"].quantile(0.25)

Q3 = train_X["Insulin"].quantile(0.75)

IQR = Q3 - Q1

UW = Q3 + 1.5*IQR



train_X["Insulin"] = np.where(train_X["Insulin"] == 0,

                              q60th, train_X["Insulin"])

train_X["Insulin"] = np.where(train_X["Insulin"] > UW,

                              q95th, train_X["Insulin"])
plots("BMI")
med = train_X["BMI"].median()

q95th = train_X["BMI"].quantile(0.95)

Q1 = train_X["BMI"].quantile(0.25)

Q3 = train_X["BMI"].quantile(0.75)

IQR = Q3 - Q1

UW = Q3 + 1.5*IQR



train_X["BMI"] = np.where(train_X["BMI"] == 0,

                          med, train_X["BMI"])

train_X["BMI"] = np.where(train_X["BMI"] > UW,

                          q95th, train_X["BMI"])
plots("DiabetesPedigreeFunction")
q95th = train_X["DiabetesPedigreeFunction"].quantile(0.95)

Q1 = train_X["DiabetesPedigreeFunction"].quantile(0.25)

Q3 = train_X["DiabetesPedigreeFunction"].quantile(0.75)

IQR = Q3 - Q1

UW = Q3 + 1.5*IQR



train_X["DiabetesPedigreeFunction"] = np.where(

                        train_X["DiabetesPedigreeFunction"] > UW,

                        q95th, train_X["DiabetesPedigreeFunction"])
plots("Age")
q95th = train_X["Age"].quantile(0.95)

Q1 = train_X["Age"].quantile(0.25)

Q3 = train_X["Age"].quantile(0.75)

IQR = Q3 - Q1

UW = Q3 + 1.5*IQR



train_X["Age"] = np.where(train_X["Age"] > UW,

                          q95th, train_X["Age"])
feature_names = train_X.columns



scaler = StandardScaler()



# fit to train_X

scaler.fit(train_X)



# transform train_X

train_X = scaler.transform(train_X)

train_X = pd.DataFrame(train_X, columns = feature_names)



# transform test_X

test_X = scaler.transform(test_X)

test_X = pd.DataFrame(test_X, columns = feature_names)
corr_matrix = pd.concat([train_X, train_Y], axis=1).corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))



plt.figure(figsize=(10,8))

sea.heatmap(corr_matrix,annot=True, fmt=".4f",

            vmin=-1, vmax=1, linewidth = 1,

            center=0, mask=mask,cmap="RdBu_r");
train_X.drop("SkinThickness", axis=1, inplace=True)

test_X.drop("SkinThickness", axis=1, inplace=True)
clf = MLPClassifier(solver="adam", max_iter=5000, activation = "relu",

                    hidden_layer_sizes = (12),                      

                    alpha = 0.01,

                    batch_size = 64,

                    learning_rate_init = 0.001,

                    random_state=2)



clf.fit(train_X, train_Y.values.ravel());
print(classification_report(test_Y, clf.predict(test_X),

                            digits = 4,

                            target_names=["Not Diabetes",

                                          "Diabetes"]))
clf.predict(test_X)
import tensorflow as tf

from tensorflow import keras

model = tf.keras.Sequential([

    tf.keras.layers.Dense(8, activation='relu', input_shape=[len(train_X.keys())]),

    tf.keras.layers.Dense(4, activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

  ])



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
EPOCHS = 1000



history = model.fit(train_X,train_Y,epochs=EPOCHS, validation_split=0.2, verbose=2)
test_loss, test_acc = model.evaluate(test_X,test_Y)

print('Test accuracy:', test_acc)