import numpy as np

import pandas as pd

import sklearn

import os



import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing



print(os.listdir("../input"))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



adult = pd.read_csv("../input/train_data.csv",

        names=[

        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adult.shape