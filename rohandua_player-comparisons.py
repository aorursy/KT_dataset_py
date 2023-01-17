import warnings
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
fifa = pd.read_csv('../input/fifa19/data.csv')
fifa.head()
fifa.columns
fifa.info()
fifa.columns[fifa.isna().any()].tolist()
fifa.drop(['Unnamed: 0','Real Face'],axis=1,inplace=True)
fifa['Club'].fillna('Free Agent', inplace = True)
fifa['Position'].fillna('Not Specified', inplace = True)
impute_mean = fifa.loc[:, ['Crossing', 'Finishing', 'HeadingAccuracy','ShortPassing', 'Volleys', 'Dribbling', 'Curve', 
                                 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed','FKAccuracy',
                                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',
                                 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
                                 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',
                                 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                                 'GKKicking', 'GKPositioning', 'GKReflexes']]
for i in impute_mean.columns:
    fifa[i].fillna(fifa[i].mean(), inplace = True)
impute_mode = fifa.loc[:, ['Body Type','International Reputation', 'Height', 'Weight', 'Preferred Foot','Jersey Number',
                           'Work Rate']]
for i in impute_mode.columns:
    fifa[i].fillna(fifa[i].mode()[0], inplace = True)
impute_median = fifa.loc[:, ['Weak Foot', 'Skill Moves']]
for i in impute_median.columns:
    fifa[i].fillna(fifa[i].median(), inplace = True)
fifa['Value'].fillna('€0M', inplace = True)
fifa['Wage'].fillna('€0K', inplace = True)
fifa['Release Clause'].fillna('€0M', inplace = True)
fifa.rename(columns={'Value':'Value in M', 'Wage': 'Wage in K', 'Release Clause': 'Release in M' },inplace=True)
fifa['Value in M'] = fifa['Value in M'].apply(lambda x : x.rstrip('M'))
fifa['Value in M'] = fifa['Value in M'].apply(lambda x : x.lstrip('€'))
fifa['Release in M'] = fifa['Release in M'].apply(lambda x : x.rstrip('M'))
fifa['Release in M'] = fifa['Release in M'].apply(lambda x : x.lstrip('€'))
fifa['Wage in K'] = fifa['Wage in K'].apply(lambda x : x.rstrip('K'))
fifa['Wage in K'] = fifa['Wage in K'].apply(lambda x : x.lstrip('€'))
fifa['Value in M'] = fifa['Value in M'].apply(lambda x : int(x.rstrip('K'))/1000 if x.endswith('K') else x)
fifa['Release in M'] = fifa['Release in M'].apply(lambda x : int(x.rstrip('K'))/1000 if x.endswith('K') else x)
fifa['Value in M'] = fifa['Value in M'].astype('float')
fifa['Release in M'] = fifa['Release in M'].astype('float')
fifa['Wage in K'] = fifa['Wage in K'].astype('int')
fifa.fillna(0, inplace = True)
fifa.drop(['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'],axis=1,inplace=True)
def categorize(x):
    if(x.Position in (['ST','CF','RF','LF','LS','RS'])):
        return 'Striker'
    elif(x.Position in (['CB','LB','RB','RCB','LCB'])):
        return 'Defender'
    elif(x.Position in (['CM','RM','LM','CAM','LAM','RAM','CDM','LCM','RCM','RDM','LDM'])):
        return 'MidField'
    elif(x.Position in (['LW','RW','RWB','LWB'])):
        return 'Winger'
    elif(x.Position in (['GK'])):
        return 'GoalKeeper'
    else:
        return 'Not Specified'

fifa['ActualPosition'] = fifa.apply(categorize, axis=1)
cols =['Marking', 'StandingTackle', 'Aggression', 'Interceptions', 'Positioning', 
'Vision','Composure','Crossing', 'ShortPassing', 'LongPassing','Acceleration', 'SprintSpeed', 
'Agility','Reactions','Balance', 'Jumping', 'Stamina',  'FKAccuracy', 'ShotPower','LongShots', 'Penalties',
'Strength','Potential', 'Overall','Finishing', 'Volleys','SlidingTackle','HeadingAccuracy', 'Dribbling', 
'Curve', 'BallControl']
fifa[cols] = fifa[cols].astype('float')
def defending(data):
    return int(round((data[['Marking', 'StandingTackle', 'SlidingTackle']].mean()).mean()))

def general(data):
    return int(round((data[['HeadingAccuracy', 'Dribbling', 'Curve', 'BallControl']].mean()).mean()))

def mental(data):
    return int(round((data[['Aggression', 'Interceptions', 'Positioning','Vision','Composure']].mean()).mean()))

def passing(data):
    return int(round((data[['Crossing', 'ShortPassing', 'LongPassing']].mean()).mean()))

def mobility(data):
    return int(round((data[['Acceleration', 'SprintSpeed', 'Agility','Reactions']].mean()).mean()))
def power(data):
    return int(round((data[['Balance', 'Jumping', 'Stamina', 'Strength']].mean()).mean()))

def rating(data):
    return int(round((data[['Potential', 'Overall']].mean()).mean()))

def shooting(data):
    return int(round((data[['Finishing', 'Volleys', 'FKAccuracy','ShotPower','LongShots', 'Penalties']].mean()).mean()))

fifa['Defending'] = fifa.apply(defending, axis = 1)
fifa['General'] = fifa.apply(general, axis = 1)
fifa['Mental'] = fifa.apply(mental, axis = 1)
fifa['Passing'] = fifa.apply(passing, axis = 1)
fifa['Mobility'] = fifa.apply(mobility, axis = 1)
fifa['Power'] = fifa.apply(power, axis = 1)
fifa['Rating'] = fifa.apply(rating, axis = 1)
fifa['Shooting'] = fifa.apply(shooting, axis = 1)
fifa.drop(['Marking', 'StandingTackle', 'Aggression', 'Interceptions', 'Positioning', 
'Vision','Composure','Crossing', 'ShortPassing', 'LongPassing','Acceleration', 'SprintSpeed', 
'Agility','Reactions','Balance', 'Jumping', 'Stamina',  'FKAccuracy', 'ShotPower','LongShots', 'Penalties',
'Strength','Potential', 'Overall','Finishing', 'Volleys','SlidingTackle','HeadingAccuracy', 'Dribbling', 
'Curve', 'BallControl'],axis=1,inplace=True)
fifa.info()
Characterstics=['Defending','General', 'Mental', 'Passing', 'Mobility','Power','Rating','Shooting']

fig = go.Figure()
    
for index, row in fifa.loc[fifa.ActualPosition=='Striker'].head(5).iterrows():
    fig.add_trace(go.Scatterpolar(
          r=[row[Characterstics[0]],row[Characterstics[1]],row[Characterstics[2]],row[Characterstics[3]],
             row[Characterstics[4]],row[Characterstics[5]],row[Characterstics[6]],row[Characterstics[7]]],
          theta=Characterstics,
          fill='toself',
          name=row['Name']
    ))


fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[10, 95]
    )),
  showlegend=True
)

fig.show()
Characterstics=['Defending','General', 'Mental', 'Passing', 'Mobility','Power','Rating','Shooting']

fig = go.Figure()
    
for index, row in fifa.loc[fifa.ActualPosition=='Winger'].head(5).iterrows():
    fig.add_trace(go.Scatterpolar(
          r=[row[Characterstics[0]],row[Characterstics[1]],row[Characterstics[2]],row[Characterstics[3]],
             row[Characterstics[4]],row[Characterstics[5]],row[Characterstics[6]],row[Characterstics[7]]],
          theta=Characterstics,
          fill='toself',
          name=row['Name']
    ))


fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[10, 95]
    )),
  showlegend=True
)

fig.show()
Characterstics=['Defending','General', 'Mental', 'Passing', 'Mobility','Power','Rating','Shooting']

fig = go.Figure()
    
for index, row in fifa.loc[fifa.ActualPosition=='Defender'].head(5).iterrows():
    fig.add_trace(go.Scatterpolar(
          r=[row[Characterstics[0]],row[Characterstics[1]],row[Characterstics[2]],row[Characterstics[3]],
             row[Characterstics[4]],row[Characterstics[5]],row[Characterstics[6]],row[Characterstics[7]]],
          theta=Characterstics,
          fill='toself',
          name=row['Name']
    ))


fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[10, 95]
    )),
  showlegend=True
)

fig.show()
Characterstics=['Defending','General', 'Mental', 'Passing', 'Mobility','Power','Rating','Shooting']

fig = go.Figure()
    
for index, row in fifa.loc[fifa.ActualPosition=='MidField'].head(5).iterrows():
    fig.add_trace(go.Scatterpolar(
          r=[row[Characterstics[0]],row[Characterstics[1]],row[Characterstics[2]],row[Characterstics[3]],
             row[Characterstics[4]],row[Characterstics[5]],row[Characterstics[6]],row[Characterstics[7]]],
          theta=Characterstics,
          fill='toself',
          name=row['Name']
    ))


fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[10, 95]
    )),
  showlegend=True
)

fig.show()
GoalKeepers = fifa[fifa.ActualPosition=='GoalKeeper'][['ID','Name', 'Nationality','Position','Overall', 'Club','GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes']]
cols = ['GKDiving','GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
GoalKeepers[cols] = GoalKeepers[cols].astype(int)
Characterstics=['GKDiving','GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

fig = go.Figure()
    
for index, row in GoalKeepers.head(5).iterrows():
    fig.add_trace(go.Scatterpolar(
          r=[row[Characterstics[0]],row[Characterstics[1]],row[Characterstics[2]],row[Characterstics[3]],row[Characterstics[4]]],
          theta=Characterstics,
          fill='toself',
          name=row['Name']
    ))


fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[60, 95]
    )),
  showlegend=True
)

fig.show()
