import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

print(df.shape)

df.head()
y = df.DEATH_EVENT

x = df.drop(columns=["DEATH_EVENT"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y.unique()
from statsmodels.genmod.generalized_linear_model import GLM

from statsmodels.genmod import families

import statsmodels.stats.tests.test_influence
res = GLM(y_train, x_train,

          family=families.Binomial()).fit(attach_wls=True, atol=1e-10)

print(res.summary())
pred = np.array(res.predict(x_test), dtype=float)

pred = [1 if v >= 0.5 else 0 for v in pred]

accuracy_score(y_test, pred)