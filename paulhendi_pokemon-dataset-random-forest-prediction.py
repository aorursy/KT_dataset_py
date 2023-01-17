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
folder = "../input/"

pokemon_df = pd.read_csv(folder + "pokemon.csv")
combats_df = pd.read_csv(folder + "combats.csv")
tests_df = pd.read_csv(folder + "tests.csv")
merged1 = combats_df.merge(pokemon_df, how = "left", left_on = "First_pokemon", right_on = "#")
merged2 = combats_df.merge(pokemon_df, how = "left", left_on = "Second_pokemon", right_on = "#")

merged2.drop(["Winner","First_pokemon","Second_pokemon", "Generation", "Name"], axis = 1, inplace = True)
merged1.drop(["Generation", "Name"], axis = 1, inplace = True)

for i in merged2.columns : 
    merged2.rename(columns = {i : i + "_2"}, inplace = True)

data_train = pd.concat([merged1,merged2], axis = 1)

data_train.sample(5)
for col in data_train.columns :
    if (data_train[col].dtype == "object") :
        data_train[col] = data_train[col].factorize()[0]  # Encode categorical variables
    if (data_train[col].dtype == "bool") :
        data_train[col] = data_train[col].astype(int)  # Just change bool to int

data_train.sample(5)
# This feature below will be our label
data_train["Winner_2"] = (data_train["Winner"] == data_train["First_pokemon"]).astype(int)

# Those two features are just an idea of what new features we could have
data_train["RatioHP_bins"] = pd.qcut(data_train["HP"] / data_train["HP_2"], 10).factorize()[0]   
data_train["RatioAtt_bins"] = pd.qcut(data_train["Attack"] / data_train["Attack_2"], 10).factorize()[0]


X = data_train.drop(["Winner","Winner_2", "First_pokemon", "Second_pokemon", "#", "#_2", "HP","HP_2", "Attack","Attack_2"], axis = 1)
y = data_train.Winner_2

X.head()
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(n_estimators = 50)

# Cross validation with 5 folds
print(np.mean(cross_val_score(clf, X, y, cv = 5))) 
clf.fit(X,y)
print(X.columns[np.argsort(clf.feature_importances_)]) # "Legendary" is an important feature