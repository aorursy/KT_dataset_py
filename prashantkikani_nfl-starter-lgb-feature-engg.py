import os

import pandas as pd

from kaggle.competitions import nflrush

import numpy as np



from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn

import random

from sklearn.model_selection import KFold

import lightgbm as lgb

import tqdm, gc

from scipy.stats import norm
env = nflrush.make_env()
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
pd.set_option('display.max_columns', None)  

train_df.head()
# distribution of target label

seaborn.distplot(train_df['Yards'])

plt.show()
not_used = ["GameId","PlayId","Yards"]

unique_columns = []

for c in train_df.columns:

    if c not in not_used and len(set(train_df[c][:11]))!= 1:

        unique_columns.append(c)
def fe(df):

    df['X1'] = 120 - df['X']

    df['Y1'] = 53.3 - df['Y']

    df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']

    

    def give_me_WindSpeed(x):

        x = str(x)

        x = x.replace('mph', '').strip()

        if '-' in x:

            x = (int(x.split('-')[0]) + int(x.split('-')[1])) / 2

        try:

            return float(x)

        except:

            return -99

    

    df['WindSpeed'] = df['WindSpeed'].apply(lambda p: give_me_WindSpeed(p))

    

    def give_me_GameWeather(x):

        x = str(x).lower()

        if 'indoor' in x:

            return  'indoor'

        elif 'cloud' in x or 'coudy' in x or 'clouidy' in x:

            return 'cloudy'

        elif 'rain' in x or 'shower' in x:

            return 'rain'

        elif 'sunny' in x:

            return 'sunny'

        elif 'clear' in x:

            return 'clear'

        elif 'cold' in x or 'cool' in x:

            return 'cool'

        elif 'snow' in x:

            return 'snow'

        return x

    

    df['GameWeather'] = df['GameWeather'].apply(lambda p: give_me_GameWeather(p))

    

    # from https://www.kaggle.com/ryches/model-free-benchmark

    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']

    

    df['is_rusher'] = df['NflId'] == df['NflIdRusher']

    

    for c in df.columns:

        if c in not_used: continue

        elif c == 'TimeHandoff':

            df['TimeHandoff_min'] = pd.Series([int(x[-7:-5]) for x in df[c]])

            df['TimeHandoff_sec'] = pd.Series([int(x[-4:-2]) for x in df[c]])

            # '2017-09-08T00:44:05.000Z'to '00:44:05.000Z' as time matters more than date

            df[c] = pd.Series([x[11:] for x in df[c]])

        elif c == 'TimeSnap':

            df['TimeSnap_min'] = pd.Series([int(x[-7:-5]) for x in df[c]])

            df['TimeSnap_sec'] = pd.Series([int(x[-4:-2]) for x in df[c]])

            # '2017-09-08T00:44:05.000Z'to '00:44:05.000Z' as time matters more than date

            df[c] = pd.Series([x[11:] for x in df[c]])

        elif c == 'PlayerHeight':

            df['height_1'] = pd.Series([int(x[0]) for x in df[c]])

            df['height_2'] = pd.Series([int(x[2]) for x in df[c]])

            df['height_3'] = df['height_1'] * 12 + df['height_2']

            df['BMI'] = (df['PlayerWeight'] * 703) / ((df['height_1'] * 12 + df['height_2']) ** 2)

        elif c == "DefensePersonnel":

            arr = [[int(s[0]) for s in t.split(", ")] for t in df["DefensePersonnel"]]

            df["DL"] = pd.Series([a[0] for a in arr])

            df["LB"] = pd.Series([a[1] for a in arr])

            df["DB"] = pd.Series([a[2] for a in arr])

        elif c == "OffensePersonnel":

            arr = [[int(s[0]) for s in t.split(", ")] for t in df["OffensePersonnel"]]

            df["RB"] = pd.Series([a[0] for a in arr])

            df["TE"] = pd.Series([a[1] for a in arr])

            df["WR"] = pd.Series([a[2] for a in arr])

        elif c == "GameClock":

            arr = [[int(s[0]) for s in t.split(":")] for t in df["GameClock"]]

            df["GameHour"] = pd.Series([a[0] for a in arr])

            df["GameMinute"] = pd.Series([a[1] for a in arr])

        elif c == "PlayerBirthDate":

            df['Season'] = pd.Series([int(x) for x in df['Season']])

            df["BirthY"] = pd.Series([int(t.split('/')[2]) for t in df["PlayerBirthDate"]])

            df['age'] = df['Season'] - df['BirthY']

            df['Season'] = pd.Series([str(x) for x in df['Season']])

            

    df['handoff_snap_diff_min'] = df['TimeHandoff_min'] - df['TimeSnap_min']

    df['handoff_snap_diff_sec'] = df['handoff_snap_diff_min'] * 60 + df['TimeHandoff_sec'] - df['TimeSnap_sec']

    return df
train_df = fe(train_df)
lbl_dict = {}

for c in train_df.columns:

    if train_df[c].dtype=='object' and c not in not_used: 

        lbl = preprocessing.LabelEncoder()

        train_df[c] = lbl.fit_transform(list(train_df[c].values))

        lbl_dict[c] = lbl
train_df.head(10)
train_df.shape
all_columns = []

for c in train_df.columns:

    if c in not_used: continue

    all_columns.append(c)



for c in unique_columns:

    for i in range(22):

        all_columns.append(c+str(i))
len(all_columns)
# from https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm



train_data=np.zeros((509762//22,len(all_columns)))

for i in tqdm.tqdm(range(0,509762,22)):

    count=0

    for c in train_df.columns:

        if c in not_used: continue

        train_data[i//22][count] = train_df[c][i]

        count+=1

    for c in unique_columns:

        for j in range(22):

            train_data[i//22][count] = train_df[c][i+j]

            count+=1        
y_train_ = np.array([train_df["Yards"][i] for i in range(0,509762,22)])

X_train = pd.DataFrame(data=train_data,columns=all_columns)
X_train.shape, y_train_.shape
data = [0 for i in range(199)]

for y in y_train_:

    data[int(y+99)]+=1

plt.plot([i-99 for i in range(199)],data)
# from https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm



y_train = np.zeros(len(y_train_),dtype=np.float)

for i in range(len(y_train)):

    y_train[i]=(y_train_[i])



scaler = preprocessing.StandardScaler()

scaler.fit([[y] for y in y_train])

y_train = np.array([y[0] for y in scaler.transform([[y] for y in y_train])])

data = [0 for i in range(199)]

for y in y_train:

    data[int(y+99)]+=1

plt.plot([i-99 for i in range(199)],data)
folds = 10

seed = 1997

kf = KFold(n_splits = folds, shuffle = True, random_state=seed)

y_valid_pred = np.zeros(X_train.shape[0])

models = []



lgb_params = dict(

    objective='regression',

    n_estimators=1000, 

    learning_rate=0.01,

    metric='rmse',

    bagging_fraction = 0.8,

    feature_fraction = 0.8,

)



for tr_idx, val_idx in kf.split(X_train, y_train):

    tr_x, tr_y = X_train.iloc[tr_idx,:], y_train[tr_idx]

    vl_x, vl_y = X_train.iloc[val_idx,:], y_train[val_idx]



    tr_data = lgb.Dataset(tr_x, label=tr_y)

    vl_data = lgb.Dataset(vl_x, label=vl_y)  

    clf = lgb.LGBMRegressor(**lgb_params)

    clf.fit(tr_x, tr_y,

        eval_set=[(vl_x, vl_y)],

        early_stopping_rounds=50,

        verbose=100)

    try:

        plt.figure(figsize=(50, 30))

        ax = lgb.plot_importance(clf, max_num_features=20)

        plt.show()

    except Exception as e:

        print('fi error', e)

        pass

    y_valid_pred[val_idx] += clf.predict(vl_x, num_iteration=clf.best_iteration_)

    models.append(clf)



gc.collect()
# from https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm



y_pred = np.zeros((509762//22,199))

y_ans = np.zeros((509762//22,199))



for i,p in enumerate(np.round(scaler.inverse_transform(y_valid_pred))):

    for j in range(199):

        if j>=p+10:

            y_pred[i][j]=1.0

        elif j>=p-10:

            y_pred[i][j]=(j+10-p)*0.05



for i,p in enumerate(y_train):

    for j in range(199):

        if j>=p:

            y_ans[i][j]=1.0



print("validation score:",np.sum(np.power(y_pred-y_ans,2))/(199*(509762//22)))
index = 0

for (test_df, sample_prediction_df) in tqdm.tqdm(env.iter_test()):

    test_df = fe(test_df)

    for c in test_df.columns:

        if c in lbl_dict and test_df[c].dtype=='object' and c not in not_used and not pd.isnull(test_df[c]).any(): 

#             vals = test_df[c].values

#             test_col = []

#             for val in vals:

#                 try:

#                     test_col.append(lbl_dict[c].transform(list(val))[0])

#                 except:

#                     test_col.append(np.nan)

#             test_df[c] = test_col

            try:

                test_df[c] = lbl_dict[c].transform(list(test_df[c].values))

            except:

                test_df[c] = [np.nan for i in range(22)]

    

    count=0

    test_data = np.zeros((1,len(all_columns)))

    for c in test_df.columns:

        if c in not_used: continue

        test_data[0][count] = test_df[c][index]

        count+=1

    for c in unique_columns:

        for j in range(22):

            test_data[0][count] = test_df[c][index + j]

            count+=1        

    y_pred = np.zeros(199)        

    y_pred_p = np.sum(np.round(scaler.inverse_transform(

        [model.predict(test_data)[0] for model in models])))/folds

    y_pred_p += 99

    for j in range(199):

        if j>=y_pred_p+10:

            y_pred[j]=1.0

        elif j>=y_pred_p-10:

            y_pred[j]=(j+10-y_pred_p)*0.05

    env.predict(pd.DataFrame(data=[y_pred],columns=sample_prediction_df.columns))

    index += 22

env.write_submission_file()