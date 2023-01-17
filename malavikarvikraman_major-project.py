import numpy as np
import pandas as pd
df=pd.read_csv('../input/ipl2017/ipl2017.csv')
df.shape
df.info()
df.columns
df.dtypes
df.describe()
df.head(125)
df.drop(['mid','bowler','batsman','striker','non-striker'],axis=1,inplace=True)
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df.head()
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
df.shape
df.info()
df.describe()
df['bat_team'].unique()
df['bowl_team'].unique()
df=df.replace("Delhi Daredevils","Delhi Capitals")
df['bowl_team'].unique()
df.head()
df['venue'].unique()
df=df.replace("Punjab Cricket Association IS Bindra Stadium, Mohali","Punjab Cricket Association Stadium, Mohali")
len(df['venue'].unique())
df['venue'].unique()

df=df[df['overs']>=5.0]
df.dtypes
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team','venue'])
encoded_df.columns
encoded_df.head()
encoded_df.columns
X=encoded_df.drop(['total','date'],axis=1)
y=encoded_df['total']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X_train[['overs','runs','wickets']].hist(figsize=(20,20))
X_train.shape

type(X_train)
y_train.shape
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train,y_train)


y_pred=model.predict(X_test)
import seaborn as sns
sns.distplot(y_test-y_pred)
model.score(X_train,y_train)
from sklearn import metrics
print('Mean Absolute Error :',(metrics.mean_absolute_error(y_test,y_pred)))
model.score(X_test,y_test)
X_train.columns
def predict_score(batting_team='Chennai Super Kings', bowling_team='Mumbai Indians', overs=5.1, runs=50, wickets=0, venue ="SuperSport Park",runs_in_prev_5=50, wickets_in_prev_5=0):
  temp_array = list()

  # Batting Team
  if batting_team == 'Chennai Super Kings':
    temp_array = temp_array + [1,0,0,0,0,0,0,0]
  elif batting_team == 'Delhi Capitals':
    temp_array = temp_array + [0,1,0,0,0,0,0,0]
  elif batting_team == 'Kings XI Punjab':
    temp_array = temp_array + [0,0,1,0,0,0,0,0]
  elif batting_team == 'Kolkata Knight Riders':
    temp_array = temp_array + [0,0,0,1,0,0,0,0]
  elif batting_team == 'Mumbai Indians':
    temp_array = temp_array + [0,0,0,0,1,0,0,0]
  elif batting_team == 'Rajasthan Royals':
    temp_array = temp_array + [0,0,0,0,0,1,0,0]
  elif batting_team == 'Royal Challengers Bangalore':
    temp_array = temp_array + [0,0,0,0,0,0,1,0]
  elif batting_team == 'Sunrisers Hyderabad':
    temp_array = temp_array + [0,0,0,0,0,0,0,1]

  # Bowling Team
  if bowling_team == 'Chennai Super Kings':
    temp_array = temp_array + [1,0,0,0,0,0,0,0]
  elif bowling_team == 'Delhi Capitals':
    temp_array = temp_array + [0,1,0,0,0,0,0,0]
  elif bowling_team == 'Kings XI Punjab':
    temp_array = temp_array + [0,0,1,0,0,0,0,0]
  elif bowling_team == 'Kolkata Knight Riders':
    temp_array = temp_array + [0,0,0,1,0,0,0,0]
  elif bowling_team == 'Mumbai Indians':
    temp_array = temp_array + [0,0,0,0,1,0,0,0]
  elif bowling_team == 'Rajasthan Royals':
    temp_array = temp_array + [0,0,0,0,0,1,0,0]
  elif bowling_team == 'Royal Challengers Bangalore':
    temp_array = temp_array + [0,0,0,0,0,0,1,0]
  elif bowling_team == 'Sunrisers Hyderabad':
    temp_array = temp_array + [0,0,0,0,0,0,0,1]
    
  if venue == 'Barabati Stadium':
    temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue =='Brabourne Stadium':
    temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Buffalo Park':
    temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'De Beers Diamond Oval':
    temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Dr DY Patil Sports Academy':
    temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':
    temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Dubai International Cricket Stadium':
    temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Eden Gardens':
    temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Feroz Shah Kotla':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Himachal Pradesh Cricket Association Stadium':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == "Holkar Cricket Stadium":
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'JSCA International Stadium Complex':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Kingsmead':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'M Chinnaswamy Stadium':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'MA Chidambaram Stadium, Chepauk':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Maharashtra Cricket Association Stadium':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'New Wanderers Stadium':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Newlands':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'OUTsurance Oval':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Punjab Cricket Association Stadium, Mohali':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
  elif venue == 'Rajiv Gandhi International Stadium, Uppal':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
  elif venue == 'Sardar Patel Stadium, Motera':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
  elif venue == 'Sawai Mansingh Stadium':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
  elif venue == 'Shaheed Veer Narayan Singh International Stadium':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
  elif venue == 'Sharjah Cricket Stadium':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
  elif venue == 'Sheikh Zayed Stadium' :
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
  elif venue == "St George's Park":
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
  elif venue == 'Subrata Roy Sahara Stadium':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
  elif venue == 'SuperSport Park':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
  elif venue == 'Wankhede Stadium':
    temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    

  temp_array = [runs, wickets, overs,runs_in_prev_5, wickets_in_prev_5] + temp_array 

  temp_array = np.array([temp_array])

    
  return int(model.predict(temp_array)[0])
predict_score('Mumbai Indians','Chennai Super Kings',11.5,99,3,"Sheikh Zayed Stadium",40,1)
predict_score('Royal Challengers Bangalore','Kolkata Knight Riders',9.1,66,2,"Eden Gardens")
predict_score(batting_team='Mumbai Indians', bowling_team='Kings XI Punjab', overs=14.1, runs=136, wickets=4, venue="Wankhede Stadium",runs_in_prev_5=60, wickets_in_prev_5=1)
predict_score(batting_team='Chennai Super Kings', bowling_team='Rajasthan Royals', overs=8.4, runs=44, wickets=3, venue="MA Chidambaram Stadium, Chepauk",runs_in_prev_5=26, wickets_in_prev_5=2) 



predict_score(batting_team='Sunrisers Hyderabad', bowling_team='Royal Challengers Bangalore',venue ="M Chinnaswamy Stadium",overs=10.5, runs=67, wickets=3, runs_in_prev_5=29, wickets_in_prev_5=1)
