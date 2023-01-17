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
start_time = dt.datetime(2012, 1, 1)
end_time = dt.datetime.now()
weekly_points = []
ranks = [1, 5, 10, 50, 100, 200, 500, 1000]
for cutoff in pd.date_range(start=start_time, end=end_time, freq='W'):
    past_results =  team_results[team_results.Deadline <= cutoff].copy()
    past_results['DaysSinceCompetitionDeadline'] = (cutoff - past_results.Deadline).apply(lambda d: d.days)
    past_results['Score'] = 10000.0 * past_results.TeamRanking.values ** (-0.75) *  10. /  \
    np.sqrt(past_results.NumTeamMembers.values) * \
    np.log10(1 + np.log10(past_results.TotalTeamsForCompetition.values)) *\
    np.exp(-past_results.DaysSinceCompetitionDeadline.values / 500.0) *\
    past_results.UserRankMultiplier.values
    
    temporal_ranking = past_results.groupby('UserId')[['Score']].sum().sort_values(by='Score', ascending=False)
    points = [temporal_ranking['Score'].values[rank-1] for rank in ranks]
    weekly_points.append([cutoff] + points)

weekly_points_df = pd.DataFrame(weekly_points, columns=['Time'] +['PointsRank %i' % r for r in ranks])



with sns.color_palette('Greens_r'):
    fig, ax = plt.subplots()
    for r in [1, 5, 10, 50]:
        ax.plot(weekly_points_df.Time, weekly_points_df['PointsRank %i' % r].values, label='Rank %i' % r, linewidth=3)
    plt.ylabel('Required Points')
    plt.title('Points required for top ranking')
    xticks = [pd.datetime(year, 5, 1) for year in range(2012, 2017)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([x.strftime('%b %Y') for x in xticks])
    plt.legend(loc=0)
    fig.savefig('required_points1.png', dpi=300)
with sns.color_palette('Blues_r'):
    fig, ax = plt.subplots()
    for r in [100, 200, 500, 1000]:
        ax.plot(weekly_points_df.Time, weekly_points_df['PointsRank %i' % r].values, label='Rank %i' % r, linewidth=3)
    plt.ylabel('Required Points')
    plt.title('Points required for top ranking')
    xticks = [pd.datetime(year, 5, 1) for year in range(2012, 2017)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([x.strftime('%b %Y') for x in xticks])
    plt.legend(loc=0)
    fig.savefig('required_points100.png', dpi=300)