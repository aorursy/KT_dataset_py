!pip install dabl
import pandas as pd

import numpy as np

import dabl
# Read in the house price dataset

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
dabl.plot(df, 'SalePrice')
from sklearn.datasets import load_wine

from dabl.utils import data_df_from_bunch



wine_bunch = load_wine()

wine_df = data_df_from_bunch(wine_bunch)



dabl.plot(wine_df, 'target')
import matplotlib.pyplot as plt

%matplotlib inline

from dabl.datasets import load_adult

from dabl.plot import class_hists



data = load_adult()



# histograms of age for each gender

class_hists(data, "age", "gender", legend=True)

# plt.show()
titanic = pd.read_csv(dabl.datasets.data_path("titanic.csv"))
titanic_clean = dabl.clean(titanic, verbose=0)
fc = dabl.SimpleClassifier(random_state=0) #dabl Simple Classifier



# Divide data into X and y

X = titanic_clean.drop("survived", axis=1)

y = titanic_clean.survived
%%time



fc.fit(X, y)