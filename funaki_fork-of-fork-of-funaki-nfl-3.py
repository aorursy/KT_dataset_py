from kaggle.competitions import nflrush

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor as RFR

from tqdm import tqdm



env = nflrush.make_env()
from sklearn.model_selection import GridSearchCV,train_test_split
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
iter_test = env.iter_test()
new_df=df.groupby(['PlayId','Position']).count()
position_count=new_df['GameId'].unstack().fillna(0).astype(int)

position_count
rusher_df=df[df['NflId']==df['NflIdRusher']]
rusher_df.columns
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
rusher_df=process(rusher_df)
rusher_df=rusher_df.dropna()
df_position=pd.merge(rusher_df,position_count, on='PlayId')
df_position=df_position.rename(columns={'S_x':'S','S_y':'S_position'})
def feature(df):

    features=pd.DataFrame(df,columns=['X', 'Y', 'S', 'A', 'Dis','Dir','YardLine', 'Quarter',

       'gameclock', 'Down', 'Distance','HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'DefendersInTheBox','PlayerHeight',

       'PlayerWeight','Temperature', 'Humidity', 'Team_home', 'stadiumtype', 'weather','PlayDirection_right',

       'OffenseFormation_ACE',

       'OffenseFormation_I_FORM', 'OffenseFormation_JUMBO',

       'OffenseFormation_PISTOL', 'OffenseFormation_SHOTGUN',

       'OffenseFormation_SINGLEBACK', 'OffenseFormation_WILDCAT', 'C', 'CB', 'DB',

       'DE', 'DL', 'DT', 'FB', 'FS', 'G', 'HB', 'ILB', 'LB', 'MLB', 'NT', 'OG',

       'OLB', 'OT', 'QB', 'RB', 'S_position', 'SAF', 'SS', 'T', 'TE', 'WR'])

    return features

    



                      
features=feature(df_position)
train_mean=features.mean(axis=0)




train_std=features.std(axis=0)
def normalize(features):

    X=(features-train_mean)/train_std

    return X
X=normalize(features)
target=pd.Series(df_position['Yards'])
target.describe()
train_X,test_X,train_y,test_y=train_test_split(X,target,test_size=0.2)
features.shape
search_params = {

    'n_estimators'      : [300],

    'max_features'      : [3,10,20,40,'auto'],

    #'random_state'      : [1],

    #'n_jobs'            : [1],

    #'min_samples_split' : [10, 20, 30],

    'max_depth'         : [3]

}

 

gsr = GridSearchCV(

    RFR(),

    search_params,

    cv = 3,

    n_jobs = -1,

    verbose=True

)

 

gsr.fit(train_X, train_y)
print(gsr.best_score_)

print(gsr.best_estimator_)

print(gsr.best_params_)
def train_predict(X,target):

    clf = RFR(bootstrap=True, criterion='mse', max_depth=3,

                      max_features='auto', max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=2,

                      min_weight_fraction_leaf=0.0, n_estimators=300,

                      n_jobs=None, oob_score=False, random_state=None,

                      verbose=0, warm_start=False)

    clf.fit(X, target)

    return clf

    
clf=train_predict(train_X,train_y)
score_test=np.array([(i >= test_y)*1 for i in range(-99,100)])
pred_y=clf.predict(test_X)
from scipy.stats import norm 
yard = np.arange(-99, 100) 

pred_prob = [norm.cdf(yard, loc=i, scale=target.std()) for i in pred_y] 

pred_prob
#score=np.array([(i >= pred_y)*1 for i in range(-99,100)])
c=((pred_prob - score_test.T)**2).sum().sum()/(199*len(pred_prob))

c
train_df=df_position.iloc[:0,:]
#yard=['Yards' + str(i) for i in range(-99,100)]
for (test_df, sample_prediction_df) in tqdm(iter_test):

    new_df=test_df.groupby(['PlayId','Position']).count()

    position_count=new_df['GameId'].unstack().fillna(0).astype(int)

    rusher_df=test_df[test_df['NflId']==test_df['NflIdRusher']]

    test_df=process(test_df)

    test_df=pd.merge(test_df,position_count, on='PlayId')

    test_df=test_df.fillna(0)

    test_df=test_df.rename(columns={'S_x':'S','S_y':'S_position'})

    test_df=pd.concat([train_df,test_df],sort=False)

    test_df_groupby=test_df.groupby('PlayId').mean()

    test_feature=feature(test_df_groupby)

    test_feature=test_feature.fillna(0)

    test_X=normalize(test_feature)

    pred_y=clf.predict(test_X)

    pred_y=np.round(pred_y)

    yard = np.arange(-99, 100) 

    pred_prob =norm.cdf(yard, loc=pred_y[0], scale=target.std()) 

    sample_prediction_df.iloc[0,:]=pred_prob

    env.predict(sample_prediction_df)
sample_prediction_df
env.write_submission_file()
import os

print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])