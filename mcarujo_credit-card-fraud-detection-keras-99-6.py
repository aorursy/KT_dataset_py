!pip install nb_black -q 

%load_ext nb_black
import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

import plotly.figure_factory as ff

import keras



data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.info()
table = data.corr()

with sns.axes_style("white"):

    mask = np.zeros_like(table)

    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(20, 20))

    sns.heatmap(

        round(table, 2),

        cmap="Reds",

        mask=mask,

        vmax=0.5,

        vmin=table.min().min(),

        linewidths=0.5,

        annot=True,

        annot_kws={"size": 12},

    ).set_title("Dataset's Correlation Matrix")
class_corr = (

    pd.DataFrame(table["Class"].drop("Class"))

    .reset_index()

    .sort_values("Class", ascending=False)

    .round(3)

)



fig = px.bar(class_corr, x="index", y="Class", color="Class", text="Class")

fig.update_layout(

    title_text="Correlation between Target and Features",

    yaxis_title="Correlation",

    xaxis_title="Features",

)

fig.update_traces(textposition="outside")

fig.show()
fig = ff.create_table(data.describe().round(3).drop("count").T, index=True)

fig.show()
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

data.Amount = scaler.fit_transform(data.Amount.values.reshape(-1, 1))
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE



X_train, X_test, y_train, y_test = train_test_split(

    data.drop(["Time", "Class"], axis=1),

    data.Class.values,

    test_size=0.20,

    random_state=42,

)



smote = SMOTE()

X, y = smote.fit_sample(data.drop(["Time", "Class"], axis=1), data.Class.values.ravel())

X_train_os, X_test_os, y_train_os, y_test_os = train_test_split(

    X, y, test_size=0.20, random_state=42

)



X_train_os.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout



model = Sequential(

    [

        Dense(units=16, input_dim=29, activation="relu"),

        Dense(24, activation="relu"),

        Dropout(0.5),

        Dense(20, activation="relu"),

        Dense(24, activation="relu"),

        Dense(1, activation="sigmoid"),

    ]

)



model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train_os, y_train_os, batch_size=15, epochs=5)

model.evaluate(X_test_os, y_test_os)
from plotly import figure_factory as ff, graph_objects as go

from sklearn.metrics import (

    log_loss,

    accuracy_score,

    confusion_matrix,

    f1_score,

    precision_score,

    recall_score,

)





def metrics(y_true, y_pred, y_pred_class):

    ac = accuracy_score(y_true, y_pred)

    ll = log_loss(y_true, y_pred_class)

    f1 = f1_score(y_true, y_pred, zero_division=1)

    ps = precision_score(y_true, y_pred)

    mc = confusion_matrix(y_true, y_pred)

    rc = recall_score(y_true, y_pred)



    header = ["Metric", "Accuracy", "Loss(log)", "F1", "Precision", "Recall"]

    score = [

        "Score",

        round(ac, 3),

        round(ll, 3),

        round(f1, 3),

        round(ps, 3),

        round(rc, 3),

    ]



    x = ["Real 0", "Real 1"]

    y = ["Predict 0", "Predict 1"]



    fig = ff.create_table([header, score], height_constant=20)

    fig.show()



    fig = ff.create_annotated_heatmap(z=mc, x=x, y=y, colorscale="Blues")

    fig.show()
y_pred = model.predict(X_test).round()

y_pred_class = model.predict_proba(X_test)

metrics(y_test, y_pred, y_pred_class)