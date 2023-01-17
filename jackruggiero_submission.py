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



# Here I just loaded all of the packages that I used when creating my model.

# The dataset is nflrush which was pulled from kaggle.competitions
env = nflrush.make_env()

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)



# Here i just made a new environment and set Python to be able to read the data. 
pd.set_option('display.max_columns', None)  

train_df.head()



# Here is an example of 5 players that includes all of the columns in the dataset. 
seaborn.distplot(train_df['Yards'])

plt.show()



# Here is a graph of the distribution of rushes. 
f,ax = plt.subplots(figsize=(12,10))

seaborn.heatmap(train_df.iloc[:,2:].corr(),annot=True, linewidths=.1, fmt='.1f', ax=ax)



plt.show()
import plotly.express as px
df_RB = train_df.loc[:,['NflId', 'DisplayName', 'PlayerBirthDate', 'PlayerWeight', 'PlayerHeight', 'PlayerCollegeName']].drop_duplicates()



df_RB["HeightFt"] = df_RB["PlayerHeight"].str.split('-', expand=True)[0].astype(int)

df_RB["HeightIn"] = df_RB["PlayerHeight"].str.split('-', expand=True)[1].astype(int)

df_RB["HeightCm"] = df_RB["HeightFt"]*30.48 + df_RB["HeightIn"]*2.54



df_RB["WeightKg"] = df_RB["PlayerWeight"]*0.45359237



df_height = df_RB.groupby(['PlayerHeight','HeightFt','HeightIn']).size().reset_index().sort_values(["HeightFt", "HeightIn"])



df_height.columns = ["PlayerHeight","HeightFt","HeightIn","Count"]
CTeam = df_RB["PlayerCollegeName"].value_counts()

df_CTeamCount = pd.DataFrame({'College Name':CTeam.index, 'Count':CTeam.values}).sort_values("Count", ascending = False).head(50)



fig = px.bar(df_CTeamCount, x='College Name', y='Count', title="The 50 Top Colleges With The Most Players", height=700, width=800)



fig.update_traces(marker_color='rgb(239, 117, 100)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=2, opacity=0.7)



fig.show()
speed_df = train_df.loc[:,['DisplayName', 'S']].groupby('DisplayName').mean()

speed_df.columns = ["Average Speed"]

speed_df = speed_df.sort_values("Average Speed", ascending = False)
fig = px.histogram(speed_df, x="Average Speed",

                   title='Average Speed Distribution of RBs',

                   opacity=0.8,

                   color_discrete_sequence=['indianred']

                   )



fig.update_layout(

    yaxis_title_text='Count',

    height=500, width=800)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.8

                 )



fig.show()
not_used = ["GameId","PlayId","Yards"]

unique_columns = []

for c in train_df.columns:

    if c not in not_used and len(set(train_df[c][:11]))!= 1:

        unique_columns.append(c)

print(unique_columns)



# I took out the columns i wasn't going to use and then appended the unique columns.

# I also decided to print out the unique columns just so I would have a visual. 
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

# Above, I adjusted the wind speed column and made adjustments to the outputs of GameWeather    



    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']

    

    df['is_rusher'] = df['NflId'] == df['NflIdRusher']

    

    for c in df.columns:

        if c in not_used: continue

        elif c == 'TimeHandoff':

            df['TimeHandoff_min'] = pd.Series([int(x[-7:-5]) for x in df[c]])

            df['TimeHandoff_sec'] = pd.Series([int(x[-4:-2]) for x in df[c]])

# '2017-09-08T00:44:05.000Z'to '00:44:05.000Z' because time matters more than date in this case

            df[c] = pd.Series([x[11:] for x in df[c]])

        elif c == 'TimeSnap':

            df['TimeSnap_min'] = pd.Series([int(x[-7:-5]) for x in df[c]])

            df['TimeSnap_sec'] = pd.Series([int(x[-4:-2]) for x in df[c]])



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

# For the code above, this adjusted the variables I wrote down in all of the columns above

            

    df['handoff_snap_diff_min'] = df['TimeHandoff_min'] - df['TimeSnap_min']

    df['handoff_snap_diff_sec'] = df['handoff_snap_diff_min'] * 60 + df['TimeHandoff_sec'] - df['TimeSnap_sec']

    return df

# The lines above are the equations to calculate handoff_snap_diff_min and handoff_snap_diff_sec
train_df = fe(train_df)



# I then set train_df equal to the function fe which then calls train_df
label_dict = {}

for c in train_df.columns:

    if train_df[c].dtype=='object' and c not in not_used: 

        label = preprocessing.LabelEncoder()

        train_df[c] = label.fit_transform(list(train_df[c].values))

        label_dict[c] = label

        

# These lines create the dictionary label_dict, which adds objects inside of the train_df variable
train_df.sample(10)



# Here is a sample of 10 outputs from the train_df variable
all_columns = []

for c in train_df.columns:

    if c in not_used: continue

    all_columns.append(c)



for c in unique_columns:

    for i in range(22):

        all_columns.append(c+str(i))

# I created a new list which appended all of the unused columns in train_df

        

len(all_columns)



# Counts the length of the appended dataset which was 401
print(all_columns)
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



# The above was used from from https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm

# This code loops through all of the attempts and counts the yardage gained on each play. 
y_train = np.array([train_df["Yards"][i] for i in range(0,509762,22)])

X_train = pd.DataFrame(data=train_data,columns=all_columns)



# Set the groundwork to plot the chart, by setting the labels for each axis
data = [0 for i in range(199)]

for y in y_train:

    data[int(y+99)]+=1

plt.plot([i-99 for i in range(199)],data)



# Plots the amount of yards gained on each attempt and the amount of times that rush occured in the dataset
folds = 10

seed = 222

kf = KFold(n_splits = folds, shuffle = True, random_state=seed)

y_valid_pred = np.zeros(X_train.shape[0])

models = []



for tr_idx, val_idx in kf.split(X_train, y_train):

    tr_x, tr_y = X_train.iloc[tr_idx,:], y_train[tr_idx]

    vl_x, vl_y = X_train.iloc[val_idx,:], y_train[val_idx]

            

    print(len(tr_x),len(vl_x))

    tr_data = lgb.Dataset(tr_x, label=tr_y)

    vl_data = lgb.Dataset(vl_x, label=vl_y)  

    clf = lgb.LGBMRegressor(n_estimators=200,learning_rate=0.01)

    clf.fit(tr_x, tr_y,

        eval_set=[(vl_x, vl_y)],

        early_stopping_rounds=20,

        verbose=False)

    y_valid_pred[val_idx] += clf.predict(vl_x, num_iteration=clf.best_iteration_)

    models.append(clf)



gc.collect()



# Not too familiar with LGBM regressor so borrowed the above code from https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm

# The above makes predictions on future plays and yards gained. 
scaler = preprocessing.StandardScaler()

scaler.fit(y_train.reshape(-1, 1))

y_train = scaler.transform(y_train.reshape(-1, 1)).flatten()



# Standardized the objective variable by using the scaler
y_pred = np.zeros((509762//22,199))

y_ans = np.zeros((509762//22,199))



for i,p in enumerate(np.round(scaler.inverse_transform(y_valid_pred))):

    p+=99

    for j in range(199):

        if j>=p+10:

            y_pred[i][j]=1.0

        elif j>=p-10:

            y_pred[i][j]=(j+10-p)*0.05



for i,p in enumerate(scaler.inverse_transform(y_train)):

    p+=99

    for j in range(199):

        if j>=p:

            y_ans[i][j]=1.0



print("validation score:",np.sum(np.power(y_pred-y_ans,2))/(199*(509762//22)))



# The above calculates my validation score as to how accurate my model can predict the outcome of any riven rushing play out of a range of 199 plays
index = 0

for (test_df, sample_prediction_df) in tqdm.tqdm(env.iter_test()):

    test_df = fe(test_df)

    for c in test_df.columns:

        if c in label_dict and test_df[c].dtype=='object' and c not in not_used and not pd.isnull(test_df[c]).any():

            try:

                test_df[c] = label_dict[c].transform(list(test_df[c].values))

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

# The above code then finally predicts the outcome of running plays given 199 random unique situations. 