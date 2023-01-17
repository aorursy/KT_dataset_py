import pandas as pd
import sqlite3
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set(color_codes=True)

con = sqlite3.connect('../input/database.sqlite')
team_results = pd.read_sql_query("""
SELECT u.Id UserId,
       u.DisplayName DisplayName,
       t.CompetitionId,
       c.CompetitionName,
       t.id AS TeamId,
       t.Ranking AS TeamRanking,
       (SELECT COUNT(Id) FROM TeamMemberships WHERE TeamId = t.Id) NumTeamMembers,
       TotalTeamsForCompetition,
       c.UserRankMultiplier,
       c.Deadline
FROM Teams t
INNER JOIN Competitions c ON c.Id=t.CompetitionId
INNER JOIN TeamMemberships tm on tm.TeamId=t.Id
INNER JOIN Users u ON tm.UserId=u.Id
INNER JOIN (
  SELECT
  CompetitionId,
  COUNT(Id) AS TotalTeamsForCompetition
  FROM Teams teamsInner
  WHERE teamsInner.Ranking IS NOT NULL
  GROUP BY CompetitionId) cInner ON cInner.CompetitionId = t.CompetitionId
WHERE (c.FinalLeaderboardHasBeenVerified = 1)
AND (t.Ranking IS NOT NULL)
AND (t.IsBenchmark = 0)
AND (c.UserRankMultiplier  > 0)
""", con)

team_results.to_csv('team_results.csv', index=False)
team_results['Deadline'] = team_results['Deadline'].apply(lambda s: pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S'))
print(team_results.shape)
# select user to display
user_display_name = 'Leustagos'
user_results = team_results[team_results['DisplayName'] == user_display_name].copy()
user_id = user_results.UserId.iloc[0]

start_time = user_results.Deadline.min() + dt.timedelta(days=7)
end_time = team_results.Deadline.max() + dt.timedelta(days=7)
weekly_ranks = []
for cutoff in pd.date_range(start=start_time, end=end_time, freq='W'):
    past_results =  team_results[team_results.Deadline <= cutoff].copy()
    past_results['DaysSinceCompetitionDeadline'] = (cutoff - past_results.Deadline).apply(lambda d: d.days)
    past_results['Score'] = 10000.0 * past_results.TeamRanking.values ** (-0.75) *  10. /  \
    np.sqrt(past_results.NumTeamMembers.values) * \
    np.log10(1 + np.log10(past_results.TotalTeamsForCompetition.values)) *\
    np.exp(-past_results.DaysSinceCompetitionDeadline.values / 500.0) *\
    past_results.UserRankMultiplier.values
    
    temporal_ranking = past_results.groupby('UserId')[['Score']].sum().sort_values(by='Score', ascending=False)
    temporal_ranking['Rank'] = np.arange(len(temporal_ranking)) + 1
    user_rank = temporal_ranking.ix[user_id].Rank
    weekly_ranks.append([cutoff, user_rank])

weekly_ranks_df = pd.DataFrame(weekly_ranks, columns=['Time', 'Rank'])

fig, ax = plt.subplots()
ax.semilogy(weekly_ranks_df.Time, weekly_ranks_df.Rank, label='Global User Rank History', linewidth=3)
# Depending on your rank you could add different ticks
yticks = [1, 2, 3, 5, 10, 25, 50, 75, 100, 250, 500, 1000]
ax.set_yticks(yticks)
ax.set_yticklabels(yticks)
plt.title(user_display_name)
plt.legend(loc=0)
fig.savefig('global_rank_history.png', dpi=300)