import numpy as np
import pandas as pd
odi = pd.read_csv('https://bit.ly/2NDmwxt')
odi.head()
# data type.
num_colms = odi._get_numeric_data()
num_colms.head()
# modi = pd.read_csv('https://bit.ly/2JNsUkA')
# modi.head()
# Create new columns
# is_century, is_duck, is_fifty, is_missed_century
odi['is_century'] = odi['Runs'].apply(lambda v: 1 if v> 99 else 0)
odi['is_duck'] = odi['Runs'].apply(lambda v: 1 if v == 0 else 0)
odi['is_fifty'] = odi['Runs'].apply(lambda v: 1 if v> 49 else 0)
odi['is_missed_century'] = odi['Runs'].apply(lambda v: 1 if v> 90 and v < 100 else 0)
odi.head()
players_summary = odi.groupby('Player').agg({
    'Runs' : 'sum',
    'ScoreRate' : 'mean',
    'Country' : 'count',
    'is_century' : 'sum',
    'is_fifty' : 'sum',
    'is_duck' : 'sum',
    'is_missed_century' : 'sum'
})

cols_rename = { 'Country' : 'Total Matches',
              'is_century': 'Centuries',
               'is_fifty' : 'Fifties',
               'is_duck' : 'Ducks',
               'is_missed_century' : 'Missed Centuries'
              }

players_summary = players_summary.rename( columns = cols_rename)
players_summary = players_summary.sort_values('Runs' , ascending = False)
players_summary.to_csv('players_summary.csv')
# trending analysis

odi['date']=pd.to_datetime(odi['MatchDate'], format ='%m-%d-%Y')
odi['year'] = odi['date'].dt.year
# splitting
odi['year1']=odi['MatchDate'].apply(lambda v: v.split('-')[-1])
odi['year2']=odi['MatchDate'].str[-4:]
odi[['date','year','year1','year2']].head()
#pivot table creation
years_summary = odi.pivot_table(index='Player',
                                columns='year',
                                values='Runs',
                               aggfunc= 'sum')
years_summary.head()
years_summary.loc['Sachin R Tendulkar']
%matplotlib inline
years_summary.loc['Sachin R Tendulkar'].plot.line()
years_century = odi.pivot_table(index='Player',
                                columns='year',
                                values='is_century',
                               aggfunc= 'sum')
years_summary.head()
years_fifties = odi.pivot_table(index='Player',
                                columns='year',
                                values='is_fifty',
                               aggfunc= 'sum')
years_fifties.head()
player_name = 'Sachin R Tendulkar'
sachin_centuries = years_century.loc[player_name]
sachin_fifties = years_fifties.loc[player_name]
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1)
sachin_centuries.plot.line(ax=axs)
sachin_fifties.plot.line(ax=axs)
plt.legend(['Year wise no of Centuries',
            'Year wise no of fifties'])
# versus analysis
sachin_rows = odi[odi['Player']== 'Sachin R Tendulkar']
versus_runs = sachin_rows.groupby('Versus')['Runs'].sum()
versus_runs.sort_values(ascending= False).head(10).plot.bar()
top5_versus = versus_runs.sort_values(ascending = False).head(5).index
top5_versus
import seaborn as sns
sachin_versus_rows = sachin_rows[sachin_rows['Versus'].isin(top5_versus)]
sns.boxplot(data=sachin_versus_rows, y ='Runs',x='Versus')
sachin_versus_rows.boxplot('Runs',by='Versus',figsize=(14,5))