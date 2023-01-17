from kaggle.competitions import nflrush

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from tqdm import tqdm



env = nflrush.make_env()
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

df.columns
iter_test = env.iter_test()
df['Position'].unique()
offense_position=['QB','RB','FB','HB','WR','TE','C','G','T']

diffense_position=['DL','DT','NT','LB','ILB','MLB','OLB','DB','CB','S','SS','SAF']

new_df=df.groupby(['PlayId','Position']).count()
position_count=new_df['GameId'].unstack().fillna(0).astype(int)

position_count
pd.merge(position_count,df[['PlayId','Yards']],on='PlayId').corr()['Yards'].sort_values(ascending=False)
def process(df):

    df=pd.get_dummies(df,columns=['Team','PlayDirection','OffenseFormation'])

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
df=process(df)
df=df.dropna()
df_position=pd.merge(df,position_count, on='PlayId')
df_position=df_position.rename(columns={'S_x':'S','S_y':'S_position'})
df_position.columns
df_position.isnull().sum().sum()
df_position.corr()['Yards'].sort_values(ascending=False).head(20)
def feature(df):

    features=pd.DataFrame(df,columns=['X', 'Y', 'S', 'A', 'Dis','Dir','YardLine', 'Quarter',

       'gameclock', 'Down', 'Distance','HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'DefendersInTheBox','PlayerHeight',

       'PlayerWeight','Temperature', 'Humidity', 'Team_home', 'stadiumtype', 'weather','PlayDirection_right',

       'OffenseFormation_ACE',

       'OffenseFormation_I_FORM', 'OffenseFormation_JUMBO',

       'OffenseFormation_PISTOL', 'OffenseFormation_SHOTGUN',

       'OffenseFormation_SINGLEBACK', 'OffenseFormation_WILDCAT','C', 'CB', 'DB',

       'DE', 'DL', 'DT', 'FB', 'FS', 'G', 'HB', 'ILB', 'LB', 'MLB', 'NT', 'OG',

       'OLB', 'OT', 'QB', 'RB', 'S_position', 'SAF', 'SS', 'T', 'TE', 'WR'])

    return features

    



                      
features=feature(df_position)
train_mean=features.mean(axis=0)

train_mean
train_std=features.std(axis=0)

train_std
def normalize(features):

    scaler=StandardScaler()

    X=(features-train_mean)/train_std

    return X
def train_predict(X,target):

    lr=LinearRegression()

    lr.fit(X,target)

    return lr

    
X=normalize(features)
target=pd.Series(df_position['Yards'])

lr=train_predict(X,target)
r=lr.score(X,target)

r
train_df=df_position.iloc[:0,:]
for (test_df, sample_prediction_df) in tqdm(iter_test):

    new_df=test_df.groupby(['PlayId','Position']).count()

    position_count=new_df['GameId'].unstack().fillna(0).astype(int)

    test_df=process(test_df)

    test_df=pd.merge(test_df,position_count, on='PlayId')

    test_df=test_df.rename(columns={'S_x':'S','S_y':'S_position'})

    test_df=pd.concat([train_df, test_df],sort=False)

    test_df=test_df.fillna(0)

    test_feature=feature(test_df)

    test_X=normalize(test_feature)

    pred_y=lr.predict(test_X)

    pred_y=np.round(pred_y)

    score=np.array([(i >= pred_y).mean()*1 for i in range(-99,100)])

    sample_prediction_df.iloc[0,:]=score.T

    env.predict(sample_prediction_df)
sample_prediction_df
env.write_submission_file()
import os

print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])