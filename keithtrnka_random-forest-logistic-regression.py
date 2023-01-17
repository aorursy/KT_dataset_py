# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# TO DUMMIES: Gender, DayOfTheWeek, Status

# EXPLODE DATE: AppointmentRegistration, ApointmentData



data = pd.read_csv("../input/No-show-Issue-Comma-300k.csv")

data = pd.get_dummies(data.drop(["AppointmentRegistration", "ApointmentData"], axis=1)).drop(["Status_No-Show"], axis=1)

data.describe()



y = data["Status_Show-Up"].values

X = data.drop(["Status_Show-Up"], axis=1).values
from sklearn.dummy import DummyClassifier

from sklearn.cross_validation import cross_val_score



scores = cross_val_score(DummyClassifier(), X, y)

print("Predict majority: {:.1f}% +/- {:.1f}%".format(100 * scores.mean(), 100 * scores.std()))
from sklearn.ensemble import RandomForestClassifier



scores = cross_val_score(RandomForestClassifier(100), X, y)

print("Random forest: {:.1f}% +/- {:.1f}%".format(100 * scores.mean(), 100 * scores.std()))
from sklearn.linear_model import LogisticRegressionCV



scores = cross_val_score(LogisticRegressionCV(), X, y)

print("Logistic regression: {:.1f}% +/- {:.1f}%".format(100 * scores.mean(), 100 * scores.std()))
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



model = Pipeline([

    ("scaler", StandardScaler()),

    ("lr", LogisticRegressionCV())

])



scores = cross_val_score(model, X, y)

print("Scaled LR: {:.1f}% +/- {:.1f}%".format(100 * scores.mean(), 100 * scores.std()))
model = RandomForestClassifier(100).fit(X, y)



feature_importance = pd.Series(index=data.drop(["Status_Show-Up"], axis=1).columns,

                               data=model.feature_importances_)

feature_importance.sort_values(ascending=False)