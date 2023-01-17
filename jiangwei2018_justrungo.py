import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import gc

import lightgbm as lgb
# 减小内存

def reduce_men_usage(df):

    start_mem = df.memory_usage().sum() / 1024 ** 2

    print(f"Begin Memory usage of dataframe is {start_mem} MB")



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

            col_min = df[col].min()

            col_max = df[col].max()



            if str(col_type)[:3] == 'int':

                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif col_min < np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            elif str(col_type)[:5] == 'float':

                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024 ** 2



    print(f"End Memory usage of dataframe is {end_mem} MB")



    return df
#按照比赛id划分数据集

def split_train_validation(data, fraction):

    matchIds = data['matchId'].unique().reshape([-1])

    train_size = int(len(matchIds) * fraction)

    random_index = np.random.RandomState(seed=2).permutation(len(matchIds))



    train_matchIds = matchIds[random_index[:train_size]]

    validation_matchIds = matchIds[random_index[train_size:]]



    x_train = data.loc[data['matchId'].isin(train_matchIds)]

    x_validation = data.loc[data['matchId'].isin(validation_matchIds)]

    return x_train, x_validation
def feature_engineering(is_train, debug):

    if is_train:

        data = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

        data = reduce_men_usage(data)

        if debug:

            data = data[data['matchId'].isin(data['matchId'].unique()[:2000])]

    else:

        data = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')

        data = reduce_men_usage(data)



    initial = data[['matchId','groupId','Id']]



    data.drop(columns=['killPoints', 'rankPoints', 'winPoints', 'matchType',

                       'maxPlace', 'Id'], inplace=True)



    data['matchSize'] = data.groupby('matchId')['matchId'].transform('count')

    data['groupSize'] = data.groupby(['matchId', 'groupId'])['groupId'].transform('count')



    data['killPlacePer'] = data['killPlace'] / data['matchSize']

    data['killPerDamage'] = data['kills'] / (0.01 * data['damageDealt'])

    data['killPlacePerWalk'] = (10 * data['killPlacePer']) / (0.001 * data['walkDistance'])

    data['walkPerBoost'] = data['boosts'] / (0.001 * data['walkDistance'])

    data['walkPerWeapon'] = data['weaponsAcquired'] / (0.001 * data['walkDistance'])

    data['DBNOsPerDamage'] = data['DBNOs'] / (0.01 * data['damageDealt'])

    data['healthItem'] = data['boosts'] + data['heals']

    data['killsPerStreaks'] = data['kills'] / data['killStreaks']

    data['killsPerDBNOs'] = data['kills'] / data['DBNOs']

    data['longestPerkill'] = data['kills'] / (0.1 * data['longestKill'])

    data['placePerStreak'] = data['killStreaks'] / (0.1 * data['killPlace'])

    data[data == np.Inf] = np.NaN

    data[data == np.NINF] = np.NaN

    data.fillna(0, inplace=True)

    data = reduce_men_usage(data)



    print('新建特征完成...')



    team_features = {

        'walkDistance': [np.var, np.mean],

        'killPlacePer': [sum, min, max, np.var, np.mean],

        'boosts': [sum, np.var, np.mean],

        'weaponsAcquired': [np.mean],

        'damageDealt': [np.var, min, max, np.mean],

        'heals': [sum, np.var, np.mean],

        'kills': [sum, max, np.var, np.mean],

        'longestKill': [max, np.mean, np.var],

        'killStreaks': [max, np.var, np.mean],

        'assists': [sum, np.mean, np.size],

        'DBNOs': [np.var, max, np.mean],

        'headshotKills': [max, np.mean],

        'rideDistance': [sum, np.mean, np.var],

        'killPerDamage': [np.var, max, np.mean],

        'killPlacePerWalk': [np.mean],

        'walkPerBoost': [np.mean],

        'walkPerWeapon': [np.mean],

        'DBNOsPerDamage': [np.mean],

        'healthItem': [np.var, np.mean],

        'killsPerStreaks': [min, max, np.mean],

        'killsPerDBNOs': [min, max, np.mean],

        'longestPerkill': [min, max, np.mean],

        'placePerStreak': [min, max, np.mean],

        'revives': [sum],

        'vehicleDestroys': [sum],

        'swimDistance': [np.var],

        'roadKills': [sum],

        'teamKills': [sum],

        'matchDuration': [np.mean],

        'numGroups': [np.mean]

    }



    if is_train:

        team_features['winPlacePerc'] = max



    final_x = data.groupby(['matchId', 'groupId']).agg(team_features)

    final_x.fillna(0, inplace=True)

    final_x.replace(np.inf, 1000000, inplace=True)

    final_x = reduce_men_usage(final_x)



    del data

    gc.collect()



    final_y = np.NaN

    if is_train:

        final_y = pd.DataFrame(final_x[('winPlacePerc', 'max')])

        final_y.columns = [na1 for na1, na2 in final_y.columns]

        final_x.drop(columns=[('winPlacePerc', 'max')], inplace=True)



    final_rank = final_x.groupby('matchId').rank(pct=True)

    final_x = final_x.reset_index()[['matchId', 'groupId']].merge(final_rank,

                                                                  suffixes=['', '_rank'],

                                                                  how='left',

                                                                  on=['matchId', 'groupId'])



    final_x.columns = [na1 + '_' + na2 for na1, na2 in final_x.columns]



    final_x.rename(columns={ 'matchId_': 'matchId', 'groupId_': 'groupId' }, inplace=True)



    final_x = reduce_men_usage(final_x)

    print('特征操作完成...')



    features = final_x.columns.tolist()

    features.remove('matchId')

    features.remove('groupId')



    after = final_x[['matchId','groupId']]



    return final_x, final_y, initial, after, features
def LGB_model(x_train, y_train, x_validation, y_validation):

    params = {

        #回归问题

        "objective": "regression",

        #度量参数

        "metric": "mae",

        #一棵树上的叶子树

        "num_leaves": 100,

        #学习率

        "learning_rate": 0.03,

        #在不进行重采样的情况下随机选择部分数据

        "bagging_fraction": 0.9,

        "bagging_seed": 0,

        #线程数

        "num_threads": 4,

        #在建立树时对特征随机采样的比例

        "colsample_bytree": 0.5,

        #一个叶子上数据的最小数量. 可以用来处理过拟合

        'min_data_in_leaf': 500,

        #分裂的最小增益阈值

        'min_split_gain': 0.00011,

        #L2 正则

        'lambda_l2': 9

    }



    train_set = lgb.Dataset(x_train, label=y_train)

    validation_set = lgb.Dataset(x_validation, label=y_validation)



    model = lgb.train(params,

                      #训练集

                      train_set=train_set,

                      #迭代次数

                      num_boost_round=9400,

                      #早停

                      early_stopping_rounds=200,

                      #每200轮打印一次

                      verbose_eval=200,

                      valid_sets=[train_set, validation_set]

                      )

    

#     model.save_model('.../output/kaggle/working/model.txt')



    return model
X,Y,_,_,features = feature_engineering(True, False)

X = X.merge(Y, how='left', on=['matchId', 'groupId'])



'''划分训练集、验证集'''

x_train, x_validation = split_train_validation(X, 0.9)

y_train = x_train['winPlacePerc']



y_validation = x_validation['winPlacePerc']

x_train = x_train.drop(columns=['matchId', 'groupId', 'winPlacePerc'])

x_validation = x_validation.drop(columns=['matchId', 'groupId', 'winPlacePerc'])



x_train = np.array(x_train)

y_train = np.array(y_train)

x_validation = np.array(x_validation)

y_validation = np.array(y_validation)



del X,Y

gc.collect()



model = LGB_model(x_train, y_train, x_validation, y_validation)
# '''打印特征重要性'''

# featureImp = list(model.feature_importance())

# featureImp, features = zip(*sorted(zip(featureImp, features)))

# with open("FeatureImportance.txt", "w") as text_file:

#     for i in range(len(featureImp)):

#         print(f"{features[i]} =  {featureImp[i]}", file=text_file)



# print("特征重要性输出结束...")

# del featureImp, features

# gc.collect()



X,_,initial,after,_ = feature_engineering(False,False)

X.drop(['matchId','groupId'], axis = 1, inplace = True)

X = np.array(X)

pred = model.predict(X, num_iteration=model.best_iteration)



del X

gc.collect()
after['winPlacePerc'] = pred

result = initial.merge(after,how = 'left', on = ['matchId','groupId'])



# Any results you write to the current directory are saved as output.

df_sub = pd.DataFrame()

df_test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")

df_test = reduce_men_usage(df_test)



# Restore some columns

df_sub['Id'] = df_test['Id']

df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

df_sub = df_sub.merge(result,how = 'left', on = ['Id','matchId','groupId'])



# Sort, rank, and assign adjusted ratio

df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()

df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()

df_sub_group = df_sub_group.merge(

    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(),

    on="matchId", how="left")

df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)



df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")

df_sub["winPlacePerc"] = df_sub["adjusted_perc"]



# Deal with edge cases

df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0

df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1



subset = df_sub.loc[df_sub.maxPlace > 1]

gap = 1.0 / (subset.maxPlace.values - 1)

new_perc = np.around(subset.winPlacePerc.values / gap) * gap

df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc



# Edge case

df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0

assert df_sub["winPlacePerc"].isnull().sum() == 0
df_sub[["Id", "winPlacePerc"]]
df_sub[["Id", "winPlacePerc"]].to_csv('submission.csv', index=False)