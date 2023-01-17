import pandas as pd
fifa = pd.read_csv("../input/FIFA 2018 Statistics.csv")
fifa.head()
partidosCroacia = fifa[ fifa['Team']=='Croatia']

partidosCroacia
partidosCroacia.sum()
partidosCroacia['Goal Scored']
partidosCroacia['Goal Scored'].sum()
partidosCroacia['Goal Scored'].count()
partidosCroacia['Goal Scored'].sum() / partidosCroacia['Goal Scored'].count()
partidosCroacia['Goal Scored'].mean()
fifa.head()
fifa.groupby("Team").sum()
fifa.groupby("Team")['Goal Scored','Corners'].sum()
fifa.groupby("Team")['Goal Scored','Corners'].mean()
maximos = fifa.groupby("Team")['Goal Scored','Corners', 'Ball Possession %'].max()
minimos = fifa.groupby("Team")['Goal Scored','Corners', 'Ball Possession %'].min()
maximos.head()
minimos.head()
minimos.columns
minimos.index
pd.merge(maximos, minimos, left_index=True, right_index=True, suffixes=['_max', '_min'])
maximos = maximos.reset_index()
minimos = minimos.reset_index()
maximos.head()
minimos.head()
pd.merge(maximos, minimos, how='outer', left_on='Team', right_on='Team', suffixes=['^^MAX','__min'])