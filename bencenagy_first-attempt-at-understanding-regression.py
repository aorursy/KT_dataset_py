# https://stackoverflow.com/questions/56283294/importerror-cannot-import-name-factorial

!pip install statsmodels==0.10.0rc2
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train_V2.csv')

test_data = pd.read_csv('../input/test_V2.csv')
train_data.info()

train_data.head()
train_data.head(10)[["damageDealt", "winPlacePerc"]]
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



data_to_plot = (

    train_data.sample(n=200)[["damageDealt", "winPlacePerc"]]

)



sns.relplot(x="damageDealt", y="winPlacePerc", data=data_to_plot)
data_to_plot = (

    train_data[train_data.matchType.isin({"solo", "solo-fpp"})].sample(n=200)[["damageDealt", "winPlacePerc"]]

)



sns.relplot(x="damageDealt", y="winPlacePerc", data=data_to_plot)
from scipy import stats

print("pure random\n", stats.linregress(np.random.random(200), np.random.random(200)).rvalue ** 2)



solo_train_data = train_data[train_data.matchType.isin({"solo", "solo-fpp"})]

print("20 samples\n", stats.linregress(solo_train_data.sample(n=20)[["damageDealt", "winPlacePerc"]]).rvalue ** 2)

print("200 samples\n", stats.linregress(solo_train_data.sample(n=200)[["damageDealt", "winPlacePerc"]]).rvalue ** 2)

print("2000 samples\n", stats.linregress(solo_train_data.sample(n=2000)[["damageDealt", "winPlacePerc"]]).rvalue ** 2)

print("20000 samples\n", stats.linregress(solo_train_data.sample(n=20000)[["damageDealt", "winPlacePerc"]]).rvalue ** 2)
sns.regplot(x="damageDealt", y="winPlacePerc", x_jitter=5, data=solo_train_data.sample(n=500)[["damageDealt", "winPlacePerc"]])
sns.regplot(x="damageDealt", y="winPlacePerc", x_jitter=5, lowess=True, data=solo_train_data.sample(n=200)[["damageDealt", "winPlacePerc"]])
import statsmodels.api as sm

sample = solo_train_data.sample(n=10)

print(sample[["winPlacePerc", "damageDealt"]])

print(sm.nonparametric.lowess(sample["winPlacePerc"], sample["damageDealt"]))
sns.regplot(x="damageDealt", y="winPlacePerc", x_jitter=5, order=2, data=solo_train_data.sample(n=200)[["damageDealt", "winPlacePerc"]])
sns.regplot(x="damageDealt", y="winPlacePerc", x_jitter=5, logistic=True, data=solo_train_data.sample(n=2000)[["damageDealt", "winPlacePerc"]])
from sklearn.model_selection import train_test_split



clean_train_data = train_data[["damageDealt", "winPlacePerc"]].dropna()



X_train, X_test, y_train, y_test = train_test_split(clean_train_data, clean_train_data.winPlacePerc, test_size=0.2)



model = sm.Logit(y_train, X_train.damageDealt)

result = model.fit()
from sklearn.metrics import mean_absolute_error



predictions = result.predict(X_test.damageDealt)

sns.scatterplot(y_test[:200], predictions[:200])



print(mean_absolute_error(y_test, np.random.random(len(y_test))))

print(mean_absolute_error(y_test, predictions))
garbled_predictions = (predictions - 0.5) * 2.0



print(mean_absolute_error(y_test, garbled_predictions))

sns.scatterplot(y_test[:200], garbled_predictions[:200])
submission_predictions = result.predict(test_data.damageDealt)

submission = pd.DataFrame({"Id": test_data.Id, "winPlacePerc": submission_predictions})

submission.to_csv("submission.csv", index=False)
submission.head()