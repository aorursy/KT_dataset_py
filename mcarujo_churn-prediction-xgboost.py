!pip install nb_black -q
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import math
%load_ext nb_black
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
data = pd.read_csv("../input/churn-dataset/churn_data.csv")

data.shape
data.isna().sum()
data.drop(["rewards_earned", "credit_score", "zodiac_sign"], axis=1, inplace=True)

data.dropna(inplace=True)
data.isna().sum()
def plot_hists(df, labels):

    row = 1

    col = 1

    num_graphs = len(labels)

    rows = math.ceil(num_graphs / 2)

    fig = make_subplots(rows=rows, cols=2, subplot_titles=labels)



    index = []

    for row in range(1, rows + 1):

        for col in range(1, 3):

            index.append({"row": row, "col": col})



    graphs = []

    pos_g = 0

    for label in labels:

        local_data = df[label].value_counts()

        x = list(local_data.index)

        y = list(local_data)

        fig.add_trace(

            go.Histogram(x=df[label]), row=index[pos_g]["row"], col=index[pos_g]["col"],

        )

        pos_g = pos_g + 1



    fig.update_layout(

        autosize=False,

        height=300 * rows,

        margin=dict(l=50, r=50, b=100, t=100, pad=4),

        #         paper_bgcolor="LightSteelBlue",

    )



    fig.show()
col = list(data.columns)

col.remove("user")
plot_hists(data, col[:9])
plot_hists(data, col[9:18])
plot_hists(data, col[18:])
with sns.axes_style("white"):

    table = data.corr().round(2)

    mask = np.zeros_like(table)

    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(20, 20))

    sns.heatmap(

        table,

        cmap="Reds",

        mask=mask,

        center=0,

        linewidths=0.5,

        annot=True,

        annot_kws={"size": 10},

    )
def to_bin(serie):

    return [0 if value == 0 else 1 for value in serie]





to_bin_cols = [

    "deposits",

    "withdrawal",

    "purchases_partners",

    "purchases",

    "cc_taken",

    "cc_recommended",

    "cc_disliked",

    "cc_liked",

    "cc_application_begin",

]

for col in to_bin_cols:

    data[col] = to_bin(data[col])



data.head()
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler



enc = OneHotEncoder()

scaler = StandardScaler()

format_columns = [

    "reward_rate",

]



dummies_coulmns = ["housing", "registered_phones", "payment_type"]

X = np.concatenate(

    (

        ## OHE

        enc.fit_transform(data[dummies_coulmns]).toarray(),

        ## BIN

        data.drop(format_columns + dummies_coulmns + ["churn"], axis=1).values,

        ## FMT

        scaler.fit_transform(data[format_columns]),

    ),

    axis=1,

)



Y = data.churn.values

X.shape
from xgboost import XGBClassifier

from scipy import stats



model = XGBClassifier(objective="binary:logistic")

grid = {

    "n_estimators": stats.randint(150, 1000),

    "learning_rate": stats.uniform(0.01, 0.6),

    "subsample": stats.uniform(0.3, 0.9),

    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],

    "colsample_bytree": stats.uniform(0.5, 1.0),

    "min_child_weight": [1, 2, 3, 4, 5],

}
from sklearn.model_selection import RandomizedSearchCV



rscv = RandomizedSearchCV(

    estimator=model, param_distributions=grid, n_iter=100, cv=5, verbose=5, n_jobs=-1,

).fit(X, Y)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)



model = rscv.best_estimator_

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred_class = model.predict_proba(X_test)
metrics(y_test, y_pred, y_pred_class)