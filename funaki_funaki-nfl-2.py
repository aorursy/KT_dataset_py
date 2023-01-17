from kaggle.competitions import nflrush

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from tqdm import tqdm



env = nflrush.make_env()
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

df
iter_test = env.iter_test()
def processing(df):

    df=pd.get_dummies(df,columns=['Team','PlayDirection','OffenseFormation','Position'])

    df=df[(df['StadiumType']!='Cloudy') & (df['StadiumType']!='Bowl')]

    df=df.drop('FieldPosition', axis=1)

    outdoor=['Outdoor', 'Outdoors','Open','Indoor, Open Roof','Outdoor Retr Roof-Open', 'Oudoor', 'Ourdoor','Retr. Roof-Open','Outdor',

       'Retr. Roof - Open', 'Domed, Open', 'Domed, open', 'Outside','Heinz Field']

    indoor=['Indoors', 'RetractableRoof', 'Indoor','Retr. Roof-Closed','Dome', 'Domed, closed','Indoor, Roof Closed', 'Retr. Roof Closed','Closed Dome','Dome, closed','Domed']

    df['stadiumtype']=(df['StadiumType'].isin(outdoor)*1)

    rain=['Light Rain', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.','Rain', 'Heavy lake effect snow','Snow', 'Cloudy, Rain','Rain shower','Rainy']

    df['weather']=(~df['GameWeather'].isin(rain)*1)

    df['PlayerHeight']= df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    df['gameclock']=[ pd.Timedelta(val).total_seconds() for val in df['GameClock']]

    return df
def feature(df):

    features=pd.DataFrame(df,columns=['X', 'Y', 'S', 'A', 'Dis','Dir','YardLine', 'Quarter',

       'gameclock', 'Down', 'Distance','HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'DefendersInTheBox','PlayerHeight',

       'PlayerWeight','Temperature', 'Humidity', 'Team_home', 'stadiumtype', 'weather','PlayDirection_right',

       'OffenseFormation_ACE',

       'OffenseFormation_I_FORM', 'OffenseFormation_JUMBO',

       'OffenseFormation_PISTOL', 'OffenseFormation_SHOTGUN',

       'OffenseFormation_SINGLEBACK', 'OffenseFormation_WILDCAT', 'Position_C',

       'Position_CB', 'Position_DB', 'Position_DE', 'Position_DL',

       'Position_DT', 'Position_FB', 'Position_FS', 'Position_G',

       'Position_HB', 'Position_ILB', 'Position_LB', 'Position_MLB',

       'Position_NT', 'Position_OG', 'Position_OLB', 'Position_OT',

       'Position_QB', 'Position_RB', 'Position_S', 'Position_SAF',

       'Position_SS', 'Position_T', 'Position_TE', 'Position_WR'])

    return features

    



                      
df=processing(df)
df=df.dropna()
features=feature(df)
train_mean=features.mean(axis=0)

train_mean
train_std=features.std(axis=0)

train_std
def normalization(features):

    scaler=StandardScaler()

    X=(features-train_mean)/train_std

    return X
def training_prediction(X,target):

    lr=LinearRegression()

    lr.fit(X,target)

    return lr

    
X=normalization(features)
target=pd.Series(df['Yards'])

lr=training_prediction(X,target)
train_df=df.iloc[:0,:]
yard=['Yards' + str(i) for i in range(-99,100)]
for (test_df, sample_prediction_df) in tqdm(iter_test):

    test_df=processing(test_df)

    test_df=pd.concat([train_df,test_df],sort=False)

    test_df=test_df.fillna(0)

    test_df_groupby=test_df.groupby('PlayId').mean()

    test_feature=feature(test_df_groupby)

    test_X=normalization(test_feature)

    pred_y=lr.predict(test_X)

    pred_y=np.round(pred_y)

    score=np.array([(i >= pred_y)*1 for i in range(-99,100)])

    sample_prediction_df.iloc[0,:]=score.T

    env.predict(sample_prediction_df)
sample_prediction_df
env.write_submission_file()
import os

print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])