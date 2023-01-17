# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Construct a logistic regression model that predicts the target using age, gender_F and time_since_last gift

predictors = ["age","gender_F","time_since_last_gift"]

X = basetable[predictors]

y = basetable[["target"]]

logreg = linear_model.LogisticRegression()

logreg.fit(X, y)



# Assign the coefficients to a list coef

coef = logreg.coef_

for p,c in zip(predictors,list(coef[0])):

    print(p + '\t' + str(c))

    

# Assign the intercept to the variable intercept

intercept = logreg.intercept_

print(intercept)

# Construct a logistic regression model that predicts the target using age, gender_F and time_since_last gift

predictors = ["age","gender_F","time_since_last_gift"]

X = basetable[predictors]

y = basetable[["target"]]

logreg = linear_model.LogisticRegression()

logreg.fit(X, y)



# Assign the coefficients to a list coef

coef = logreg.coef_

for p,c in zip(predictors,list(coef[0])):

    print(p + '\t' + str(c))

    

# Assign the intercept to the variable intercept

intercept = logreg.intercept_

print(intercept)
import pandas as pd

CAvideos = pd.read_csv("../input/youtube-new/CAvideos.csv")

DEvideos = pd.read_csv("../input/youtube-new/DEvideos.csv")

FRvideos = pd.read_csv("../input/youtube-new/FRvideos.csv")

GBvideos = pd.read_csv("../input/youtube-new/GBvideos.csv")

INvideos = pd.read_csv("../input/youtube-new/INvideos.csv")

JPvideos = pd.read_csv("../input/youtube-new/JPvideos.csv")

KRvideos = pd.read_csv("../input/youtube-new/KRvideos.csv")

MXvideos = pd.read_csv("../input/youtube-new/MXvideos.csv")

RUvideos = pd.read_csv("../input/youtube-new/RUvideos.csv")

USvideos = pd.read_csv("../input/youtube-new/USvideos.csv")



print(USvideos)
