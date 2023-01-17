# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def logit(x):
    return 1/(1+np.exp(-x))

def getSigmoid(coef, intercept, predictors):
    res = np.array([])
    predArray = predictors.to_numpy()
    for attempt in predArray:
        res = np.append(res, [logit(np.dot(attempt, coef.T) + intercept)])
    return res
data = pd.read_csv('/kaggle/input/the-ultimate-halloween-candy-power-ranking/candy-data.csv').sort_values('winpercent')
y = data.chocolate.values
data.drop("competitorname", inplace = True, axis=1)
predictors = data.drop(["chocolate"], axis = 1)
predictors = (predictors-np.min(predictors))/(np.max(predictors)-np.min(predictors))

x_train, x_test, y_train, y_test = train_test_split(predictors, y, test_size=0.3,
                                                    random_state=42)

model = LogisticRegression(solver='liblinear', random_state=0).fit(x_train, y_train)

sig = getSigmoid(model.coef_, model.intercept_, predictors)
sig.sort()
prediction = model.predict(predictors)

print(model.score(predictors, y))

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(predictors.winpercent.values, prediction, 'o')
ax.plot(predictors.winpercent.values, sig)