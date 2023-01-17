!pip install nb_black -q
%load_ext nb_black
import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.figure_factory as ff

import os

from sklearn.model_selection import train_test_split

import tensorflow as tf



data = pd.read_csv("/kaggle/input/churn-modeling-dataset/Churn_Modelling.csv").drop(

    ["RowNumber", "CustomerId", "Surname"], axis=1

)

data.head()
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder



enc = OneHotEncoder(handle_unknown="ignore")

minmax_scaler = MinMaxScaler()

label_encoder = LabelEncoder()



X = np.concatenate(

    (

        ## OneHotEncoder

        enc.fit_transform(data[["Geography"]]).toarray(),

        ## Stander Scaler

        minmax_scaler.fit_transform(

            data[

                [

                    "CreditScore",

                    "Age",

                    "Tenure",

                    "Balance",

                    "NumOfProducts",

                    "EstimatedSalary",

                ]

            ]

        ),

        ## LabelEncoder

        label_encoder.fit_transform(data[["Gender"]]).reshape(-1, 1),

        ## No formatation

        data[["HasCrCard", "IsActiveMember"]].values,

    ),

    axis=1,

)



y = data.Exited.values

X.shape



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.05, random_state=42, stratify=y

)
columns = (

    [el for el in enc.categories_[0]]

    + ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary",]

    + ["Gender"]

    + ["HasCrCard", "IsActiveMember"]

    + ["Exited"]

)
import seaborn as sns

import matplotlib.pyplot as plt



table = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))

table.columns = columns

table = table.corr()

with sns.axes_style("white"):

    mask = np.zeros_like(table)

    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(10, 10))

    sns.heatmap(

        round(table, 2),

        cmap="Reds",

        mask=mask,

        vmax=table.max().max(),

        vmin=table.min().min(),

        linewidths=0.5,

        annot=True,

        annot_kws={"size": 12},

    ).set_title("Correlation Matrix App behavior dataset")



columns.remove("Exited")
from sklearn.ensemble import RandomForestClassifier





model_rfc = RandomForestClassifier()

model_rfc.fit(X_train, y_train)
sample = pd.DataFrame(X_test[0]).T

sample.columns = columns

sample
import shap



shap.initjs()  # Just to create better graphs =D

explainer = shap.TreeExplainer(model_rfc)
prediction = model_rfc.predict_proba(sample)

print("Direct print:", prediction)

print(

    "Probability to be class 0:",

    prediction[0][0],

    "\nProbability to be class 1:",

    prediction[0][1],

)
print(explainer.expected_value)
print("How is the target balance?", y_train.mean())
shap_values = explainer.shap_values(sample.loc[0])

shap_values
pd.DataFrame(shap_values, columns=columns)
print("Direct prediction:", prediction)

aux = shap_values[0].sum() + explainer.expected_value[0]

print("Sum of Baseline + Feature Contribuitions:", aux)
shap.force_plot(explainer.expected_value[0], shap_values[0], sample)
shap.force_plot(explainer.expected_value[1], shap_values[1], sample)
import shap

import pandas as pd

import numpy as np





# Train your shap to understand your model

def explain_train(model):

    return shap.TreeExplainer(model)





# Here you pass the return of the last function and also the dataframe with the columns model

def explain_this(explainer, sample):

    columns = sample.columns

    shap_values = explainer.shap_values(sample.iloc[0])

    aux = pd.DataFrame(shap_values, columns=columns)

    aux["_BASELINE"] = explainer.expected_value

    aux["_CLASSES"] = explainer.expected_value

    return aux
sample
explainer = explain_train(model_rfc)
shap_values = explain_this(explainer, sample)

shap_values