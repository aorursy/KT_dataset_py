import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

%matplotlib inline
# Load data
df_comp = pd.read_csv('../input/Competitions.csv', index_col='Id', parse_dates=['EnabledDate', 'DeadlineDate'])

df_sub = pd.read_csv('../input/Submissions.csv', index_col='Id')

df_team = pd.read_csv('../input/Teams.csv', index_col='Id')
# Add year of competition (we  will group using this field)
df_comp['Year'] = df_comp['EnabledDate'].dt.year
# Add CompetitionId to submission
df_sub['CompetitionId'] = df_sub['TeamId'].map(df_team['CompetitionId'])
# Filter competitions with reward
reward_ids = df_comp[df_comp['RewardQuantity'] > 0].index

df_sub = df_sub[df_sub['CompetitionId'].isin(reward_ids)]
# Remove submissions with missing 'PublicScoreFullPrecision' and convert this column to float
df_sub.dropna(subset=['PublicScoreFullPrecision'], inplace=True)
df_sub['PublicScoreFullPrecision'] = df_sub['PublicScoreFullPrecision'].astype(float)
# Sneak peek into the data
df_sub.head()
# Define quantiles to be analyzed
quantiles = np.linspace(0.0, 1, 100)
# Find quantiles for each competition
df_comp_quantiles = df_sub.groupby('CompetitionId')['PublicScoreFullPrecision'].quantile(quantiles)
# Create a map of rankings per year
comps_logs = defaultdict(list)

for comp_id in df_comp_quantiles.index.levels[0]:
    # Find information regarding the competition
    comp_name, comp_year, comp_metric_max = df_comp.loc[comp_id, ['Title', 'Year', 'EvaluationAlgorithmIsMax']]
    
    # Get ranking of competition over quantiles
    ranking = df_comp_quantiles.loc[comp_id].sort_index().values

    # Reverse ranking it metric is to be maximized
    if not comp_metric_max:
        ranking = -ranking[::-1]
        
    comps_logs[comp_year].append(ranking)
# Defina a color map to plot each year series
cmap = plt.get_cmap('viridis')
fig, ax = plt.subplots(figsize=(15, 10))
first_year, last_year = min(comps_logs.keys()), max(comps_logs.keys())
for i, (year, comp_log) in enumerate(comps_logs.items()):
    rankings = np.array(comp_log)
    rankings -= rankings.mean(axis=1, keepdims=True)
    rankings /= rankings.std(axis=1, keepdims=True)
    color = cmap((year-first_year) / (last_year-first_year))
    ax.plot(quantiles[:-1], np.nanmedian(rankings, axis=0)[:-1], label=str(year), c=color);
ax.set_xlim(0.5, 1);
ax.set_ylim(0, 1.);
ax.legend();
