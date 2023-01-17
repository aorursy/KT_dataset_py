import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_auc_score

from scipy import stats

import shap

import os

import gc

import sys



gc.enable()



sns.set(rc={'figure.figsize':(8,6)}, style = "white")
pt_raw = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")
def preprocessing(pt_df):

    

    print ("Start preprocessing")

    #pt1 = pt_df.drop(["event"], axis = 1)

    #pt2 = pt_df.drop(["time", "event"], axis = 1)

    

    o_diff_arr = np.zeros(len(pt_df['o']), np.float32)

    o_diff_arr[1:] = np.abs(pt_df['o'][:-1].to_numpy() - pt_df['o'][1:].to_numpy())    

    pt_df['o_diff'] = o_diff_arr

    del o_diff_arr



    dir_diff_arr = np.zeros(len(pt_df['dir']), np.float32)

    dir_diff_arr[1:] = np.abs(pt_df['dir'][:-1].to_numpy() - pt_df['dir'][1:].to_numpy())

    pt_df['dir_diff'] = dir_diff_arr

    del dir_diff_arr

    gc.collect()

    

    print ("Start Player Track Summary")

    pt_df[['PlayKey', 'dir', 'o', 's', 'o_diff', 'dir_diff']].groupby("PlayKey").mean().to_csv("pt_mean.csv")

    pt_df.drop(["time", "event", "dis", "x", "y"], axis = 1).groupby("PlayKey").min().to_csv("pt_min.csv")

    pt_df.drop(["event", "dis", "x", "y"], axis = 1).groupby("PlayKey").max().to_csv("pt_max.csv")

    pt_df.drop(["time", "event", 'x', 'y', 'dis'], axis = 1).groupby("PlayKey").std().to_csv("pt_std.csv")

    pt_df[["PlayKey", "dis"]].groupby("PlayKey").sum().to_csv("pt_sum.csv")



    del pt_df

    gc.collect()

    

    final_df = pd.read_csv("pt_mean.csv").merge(pd.read_csv("pt_max.csv"), how = "left", 

                          on = "PlayKey", suffixes=('', '_playmax'))

    final_df = final_df.merge(pd.read_csv("pt_min.csv"), how = "left", 

                          on = "PlayKey", suffixes=('', '_playmin'))

    final_df = final_df.merge(pd.read_csv("pt_std.csv"), how = "left", 

                          on = "PlayKey", suffixes=('', '_playstd'))

    final_df = final_df.merge(pd.read_csv("pt_sum.csv"), how = "left", 

                          on = "PlayKey", suffixes=('', '_playsum'))

    

    print ("Start Play List Summary")

    #pl_df = pd.read_csv("PlayList.csv")

    pl_df = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")

    pl_new = pl_df.drop(["PlayerKey", "GameID", "PositionGroup", "Weather", "Position", "PlayerDay"], axis = 1)

    del pl_df

    gc.collect()

    pl_new.loc[pl_new["StadiumType"] == 'Outdoors', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Indoors', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Oudoor', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Open', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Closed Dome', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Domed, closed', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Domed', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Dome', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Retr. Roof-Closed', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Outdoor Retr Roof-Open', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Retractable Roof', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Ourdoor', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Indoor, Roof Closed', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Retr. Roof - Closed', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Bowl', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Outddors', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Retr. Roof-Open', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Dome, closed', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Indoor, Open Roof', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Domed, Open', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Domed, open', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Heinz Field', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Cloudy', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Retr. Roof - Open', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Retr. Roof Closed', "StadiumType"] = "Indoor"

    pl_new.loc[pl_new["StadiumType"] == 'Outdor', "StadiumType"] = "Outdoor"

    pl_new.loc[pl_new["StadiumType"] == 'Outside', "StadiumType"] = "Outdoor"

    #pl_new["Surface"] = 1

    #pl_new.loc[pl_new["FieldType"] == "Synthetic", "Surface"] = 0

    #pl_new = pl_new.drop(["FieldType"], axis = 1)

    temp_mean = int(pl_new.loc[pl_new["Temperature"]!= -999]["Temperature"].mean())

    pl_new.loc[pl_new["Temperature"]== -999, "Temperature"] = temp_mean

    

    pl_new = pl_new[pl_new["PlayType"] != '0']

    

    print ("Start Injury Data Summary")

    #in_df = pd.read_csv("InjuryRecord.csv")

    in_df = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

    in_pk = in_df[["PlayKey"]].copy()

    in_pk.loc[:, "Injury"] = 1

    

    print ("Merging")

    final_df = pl_new.merge(final_df, how = "left", on = "PlayKey").dropna()



    final_df = final_df.merge(in_pk, how = "left", on = "PlayKey")

    #final_df['Injury'].astypes('int32')

    final_df.set_index('PlayKey', inplace = True)

    final_df.Injury = final_df.Injury.fillna(0)



    final_df = final_df.astype({'Injury': 'int32'})

    print ("Finish preprocessing")

    

    return (final_df)
injury_data = preprocessing(pt_raw)
injury_raw = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")
body_injury_df = injury_raw.groupby("BodyPart").count()[["PlayerKey"]].reset_index().rename(columns = {"PlayerKey": "Count"})

bodypart_plt = sns.catplot(x = "Count", y = "BodyPart", kind = "bar", data = body_injury_df)
injury_raw["InjuryDuration"] = "1"

injury_raw.loc[injury_raw["DM_M1"]==1, "InjuryDuration"]="Less than 7 days"

injury_raw.loc[injury_raw["DM_M7"]==1, "InjuryDuration"]="7 to 27 days"

injury_raw.loc[injury_raw["DM_M28"]==1, "InjuryDuration"]="28 to 41 days"

injury_raw.loc[injury_raw["DM_M42"]==1, "InjuryDuration"]="Equal or more than 42 days"

duration_injury_df = injury_raw.groupby("InjuryDuration").count()[["PlayerKey"]].rename(columns = {"PlayerKey": "Count"}).reset_index()

duration_plt = sns.catplot(x = "Count", y = "InjuryDuration", kind = "bar", order = ["Less than 7 days", "7 to 27 days", "28 to 41 days", "Equal or more than 42 days"], data = duration_injury_df)
duration_body_injury_df = injury_raw.groupby(["BodyPart", "InjuryDuration"]).count()[["PlayerKey"]].reset_index().rename(columns = {"PlayerKey": "Count"})

body_dur_plt = sns.catplot(x = "Count", y = "BodyPart", hue = "InjuryDuration",kind = "bar", hue_order = ["Less than 7 days", "7 to 27 days", "28 to 41 days", "Equal or more than 42 days"], data = duration_body_injury_df)
surface_injury_df = injury_raw.groupby("Surface").count()[["PlayerKey"]].reset_index().rename(columns = {"PlayerKey": "Count"})

surface_plt = sns.catplot(x = "Count", y = "Surface", kind = "bar", data = surface_injury_df)
surface_body_injury_df = injury_raw.groupby(["Surface", "BodyPart"]).count()[["PlayerKey"]].reset_index().rename(columns = {"PlayerKey": "Count"})

body_sur_plt = sns.catplot(x = "Count", y = "BodyPart", hue = "Surface",kind = "bar", data = surface_body_injury_df)
trueInjury_data = injury_data[injury_data['Injury'] == 1]

trueInjury_data.shape
trueInjury_data.columns
trueinjury_pos_df = trueInjury_data.groupby("RosterPosition").count()[["Injury"]].reset_index().rename(columns = {"Injury": "Count"})

all_pos_df = injury_data.groupby("RosterPosition").count()[["Injury"]].reset_index().rename(columns = {"Injury": "Count"})

pos_injury_df = all_pos_df.merge(trueinjury_pos_df, on = "RosterPosition", how = "left", suffixes = ('_all', '_injury')).fillna(0)

pos_injury_df["injury_rate"] = pos_injury_df["Count_injury"]/pos_injury_df["Count_all"]

pos_injury_df
trueinjury_type_df = trueInjury_data.groupby("PlayType").count()[["Injury"]].reset_index().rename(columns = {"Injury": "Count"})

all_type_df = injury_data.groupby("PlayType").count()[["Injury"]].reset_index().rename(columns = {"Injury": "Count"})

type_injury_df = all_type_df.merge(trueinjury_type_df, on = "PlayType", how = "left", suffixes = ('_all', '_injury')).fillna(0)

type_injury_df["injury_rate"] = type_injury_df["Count_injury"]/type_injury_df["Count_all"]

type_injury_df
pl_raw = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")

pos_injury_df = injury_raw.merge(pl_raw, how = 'left', on = 'PlayKey')[['BodyPart', 'RosterPosition', 'PlayKey']].groupby(

    'RosterPosition').count().reset_index().rename(columns = {'PlayKey': 'Count'})

sns.catplot(x = "Count", y = 'RosterPosition', kind = "bar", data = pos_injury_df)
pos_body_injury_df = injury_raw.merge(pl_raw, how = 'left', on = 'PlayKey')[['BodyPart', 'RosterPosition', 'PlayKey']].groupby(

    ['BodyPart', 'RosterPosition']).count().reset_index().rename(columns = {'PlayKey': 'Count'})

sns.catplot(x = "Count", y = 'RosterPosition', hue = 'BodyPart', kind = "bar", data = pos_body_injury_df)
type_body_injury_df = injury_raw.merge(pl_raw, how = 'left', on = 'PlayKey')[['BodyPart', 'PlayType', 'PlayKey']].groupby(

    ['BodyPart', 'PlayType']).count().reset_index().rename(columns = {'PlayKey': 'Count'})

sns.catplot(x = "Count", y = 'PlayType', hue = 'BodyPart', kind = "bar", data = type_body_injury_df)
def lgb_ready(df_bf):

    

    obj_col = []

    

    n_col = df_bf.shape[1]

    for i in range(n_col):



        if (df_bf.iloc[:,i].dtypes == 'object'):

            n_uni = df_bf.iloc[:,i].nunique()

            l_uni = df_bf.iloc[:,i].unique()

            print (l_uni)

            uni_int = [k for k in range(n_uni)]

    

            df_bf.iloc[:,i].replace(l_uni, uni_int, inplace=True)

            obj_col.append(i)

        

    return (obj_col)
injury_obj = lgb_ready(injury_data)

injury_obj
injury_X = injury_data.drop(["Injury"], axis = 1)

injury_y = injury_data["Injury"]
X_train, X_valid, y_train, y_valid = train_test_split(

    injury_X, injury_y, test_size=0.25, random_state=42)
train_dataset = lgb.Dataset(X_train, label = y_train, categorical_feature = injury_obj)

valid_dataset = lgb.Dataset(X_valid, label = y_valid, categorical_feature = injury_obj)



params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance':True,

    'max_bin': 255,

    'learning_rate': 0.01,

    'num_leaves': 31,

    'max_depth': 12,

    #'feature_fraction': 0.9,

    #'bagging_fraction': 0.9,

    #'bagging_freq': 10,

    #'bagging_seed': 0,

    'min_data_in_leaf': 7,

    'num_threads': 1,

    'random_state': 3

}



bst = lgb.train(params, train_dataset, num_boost_round = 145,

                valid_sets=[train_dataset,valid_dataset], verbose_eval=5)
y_pred = bst.predict(X_valid).astype(int)

#print (y_pred)

#sum(y_pred>0.5)

print (sum(y_pred==y_valid)/len(y_valid))

confusion_matrix(y_valid, y_pred)
shap_values = shap.TreeExplainer(bst).shap_values(X_valid)
shap.summary_plot(shap_values[1], X_valid, plot_type = "bar")
shap.summary_plot(shap_values[1], X_valid)
shap.dependence_plot('PlayerGame', shap_values[1], X_valid, show = False)
shap.dependence_plot('dir_diff', shap_values[1], X_valid, show = False)
shap.dependence_plot('PlayerGamePlay', shap_values[1], X_valid, show = False)
shap.dependence_plot('s', shap_values[1], X_valid, show = False)
shap.dependence_plot('dir', shap_values[1], X_valid, show = False)
shap.dependence_plot('Temperature', shap_values[1], X_valid, show = False)
shap.dependence_plot('o_diff', shap_values[1], X_valid, show = False)
healthy_data = injury_data[injury_data['Injury']==0]
def surface_move(feature):

    

    natural_df = healthy_data[healthy_data['FieldType']==1][feature].to_numpy()

    synthetic_df = healthy_data[healthy_data['FieldType']==0][feature].to_numpy()

    

    n_m = np.mean(natural_df)

    s_m = np.mean(synthetic_df)

    n_std = np.std(natural_df)

    s_std = np.std(synthetic_df)

    

    plot1 = sns.kdeplot(natural_df, shade = True, label = "Natural", c = "blue")

    plt.axvline(np.mean(natural_df), c = "blue")

    sns.kdeplot(synthetic_df, shade = True, label = "Synthetic", c = "darkred")

    plt.axvline(np.mean(synthetic_df), c = "darkred");

    

    plt.figure()

    

    plot2 = sns.kdeplot(natural_df, shade = True, label = "Natural", c = "blue").set(xlim=(n_m - s_m/2, n_m + s_m/2))

    plt.axvline(np.mean(natural_df), c = "blue")

    sns.kdeplot(synthetic_df, shade = True, label = "Synthetic", c = "darkred")

    plt.axvline(np.mean(synthetic_df), c = "darkred");

    

    print ("Natural mean:", n_m)

    print ("Synthetic mean:", s_m)

    print ("Natural std:", n_std)

    print ("Synthetic std:", s_std)

    print (stats.ttest_ind(natural_df, synthetic_df, equal_var = False))

    

    return (plot1, plot2)
surface_move('dir_diff')
surface_move('s')
surface_move('o_diff')