# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/data.csv", index_col="shot_id")
def hometest(row):
    if "vs" in row["matchup"]:
        return(1)
    else:
        return(0)

df["home"] = df.apply(lambda row: hometest(row), axis = 1)
df["distance"] = df.apply(lambda row: 0.1*(row["loc_x"]**2 + row["loc_y"]**2)**0.5, axis = 1)
df["time_remaining"] = df["minutes_remaining"] * 60 + df["seconds_remaining"]

f = {'shot_made_flag':['mean','count'], 'loc_x':['mean'], 'loc_y':['mean'], 'time_remaining':['mean']}

df.groupby(["shot_zone_area"]).agg(f)
from sklearn import mixture
for numGaussians in [8]:
    
    gaussianMixtureModel = mixture.GaussianMixture(n_components=numGaussians, covariance_type='full', 
                                                   init_params='kmeans', n_init=50, 
                                                   verbose=0, random_state=5)
    cluster_df = df.loc[lambda row: row["combined_shot_type"] == "Jump Shot", :]

    gaussianMixtureModel.fit(cluster_df.loc[:,['loc_x','loc_y']])

    # add the GMM cluster as a field in the dataset

    coltext = 'ShotCluster'+str(numGaussians)
    print(coltext)
    df[coltext] = gaussianMixtureModel.predict(df.loc[:,['loc_x','loc_y']])
df.drop(["loc_x", "loc_y"], axis=1, inplace = True)
df.drop(["action_type", "distance", "shot_type"], axis=1, inplace=True)
df.drop(["game_event_id", "season", "shot_zone_area", "shot_zone_basic", "shot_zone_range", "game_date", "matchup", "opponent"], axis=1, inplace=True)
#df.drop(["ShotCluster2", "ShotCluster3", "ShotCluster5", "ShotCluster13"], axis=1, inplace=True)
df.drop(["team_id", "team_name", "game_id", "lat", "lon", "minutes_remaining", "seconds_remaining"], axis=1, inplace = True)
df2 = pd.get_dummies(df, columns = ["combined_shot_type", "ShotCluster8"])
df_train, df_test = [x for _, x in df2.groupby(df['shot_made_flag'].isnull())]
df_train_y = df_train["shot_made_flag"]
df_train.drop("shot_made_flag", axis=1, inplace = True)
df_test.drop("shot_made_flag", axis=1, inplace = True)
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_leaf=10)
clf.fit(df_train, df_train_y)
ypred_tree = clf.predict_proba(df_test)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(df_train, df_train_y)
ypred_lr = logreg.predict_proba(df_test)
output = pd.concat([pd.DataFrame(df_test.index), pd.DataFrame(ypred_tree)[0], pd.DataFrame(ypred_lr)[0]], axis = 1)
output.columns = ["shot_id", "tree", "logreg"]
output["average"] = output["tree"]/2 + output["logreg"]/2
output1 = output[["shot_id", "tree"]]
output2 = output[["shot_id", "logreg"]]
output3 = output[["shot_id", "average"]]

output1.columns = ["shot_id", "shot_made_flag"]
output2.columns = ["shot_id", "shot_made_flag"]
output3.columns = ["shot_id", "shot_made_flag"]
output1.to_csv('../working/output1.csv', index = False)
output2.to_csv('../working/output2.csv', index = False)
output3.to_csv('../working/output3.csv', index = False)

from subprocess import check_output
print(check_output(["ls", "../working"]).decode("utf8"))
