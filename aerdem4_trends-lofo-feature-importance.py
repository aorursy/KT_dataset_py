!pip install lofo-importance
import numpy as np

import pandas as pd



loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")



fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

df = fnc_df.merge(loading_df, on="Id")





labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")

labels_df["is_train"] = True



df = df.merge(labels_df, on="Id", how="left")



test_df = df[df["is_train"] != True].copy()

df = df[df["is_train"] == True].copy()



df.shape, test_df.shape
from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge

from lofo import LOFOImportance, Dataset, plot_importance





def get_lofo_importance(target):

    cv = KFold(n_splits=5, shuffle=True, random_state=0)



    dataset = Dataset(df=df[df[target].notnull()], target=target, features=loading_features, 

                      feature_groups={"fnc": df[df[target].notnull()][fnc_features].values/500})



    model = Ridge(0.01)

    lofo_imp = LOFOImportance(dataset, cv=cv, scoring="neg_mean_absolute_error", model=model)



    return lofo_imp.get_importance()
plot_importance(get_lofo_importance(target="age"), figsize=(16, 8))
plot_importance(get_lofo_importance(target="domain1_var1"), figsize=(16, 8))
plot_importance(get_lofo_importance(target="domain1_var2"), figsize=(16, 8))
plot_importance(get_lofo_importance(target="domain2_var1"), figsize=(16, 8))
plot_importance(get_lofo_importance(target="domain2_var2"), figsize=(16, 8))