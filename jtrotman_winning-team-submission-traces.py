import gc, os, sys, time

import pandas as pd, numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from IPython.display import HTML, display



IN_DIR = '../input'



def read_csv_filtered(csv, col, values):

    dfs = [df.loc[df[col].isin(values)]

           for df in pd.read_csv(f'{IN_DIR}/{csv}', chunksize=100000, low_memory=False)]

    return pd.concat(dfs, axis=0)





comps = pd.read_csv(os.path.join(IN_DIR, 'Competitions.csv')).set_index('Id')

comps = comps.query("HostSegmentTitle != 'InClass'")

idx = comps.EvaluationAlgorithmName.isnull()

comps.loc[idx, 'EvaluationAlgorithmName'] = comps.loc[idx, 'EvaluationAlgorithmAbbreviation']



comps['EvaluationLabel'] = comps.EvaluationAlgorithmAbbreviation

idx = comps.EvaluationLabel.str.len() > 30

comps.loc[idx, 'EvaluationLabel'] = comps.loc[idx, 'EvaluationLabel'].str.replace(r'[^A-Z\d\-]', '')



comps['DeadlineDate'] = pd.to_datetime(comps.DeadlineDate)

comps['EnabledDate'] = pd.to_datetime(comps.EnabledDate)

comps['DeadlineDateText'] = comps.DeadlineDate.dt.strftime('%c')

comps['EnabledDateText'] = comps.EnabledDate.dt.strftime('%c')

comps['Year'] = comps.DeadlineDate.dt.year

comps['RewardQuantity'].fillna('', inplace=True)

comps['Days'] = (comps.DeadlineDate - comps.EnabledDate) / pd.Timedelta(1, 'd')

comps['FinalWeek'] = (comps.DeadlineDate - pd.Timedelta(1, 'w'))



teams = read_csv_filtered('Teams.csv', 'CompetitionId', comps.index).set_index('Id')

# Just the winning teams

teams = teams.query('PrivateLeaderboardRank==1').copy()



tmemb = read_csv_filtered('TeamMemberships.csv', 'TeamId', teams.index).set_index('Id')

users = read_csv_filtered('Users.csv', 'Id', tmemb.UserId)

tmemb = tmemb.merge(users, left_on='UserId', right_on='Id')



# Submissions

subs = read_csv_filtered('Submissions.csv', 'TeamId', tmemb.TeamId)

subs = subs.rename(columns={'PublicScoreFullPrecision': 'Public'})

subs = subs.rename(columns={'PrivateScoreFullPrecision': 'Private'})

subs['SubmissionDate'] = pd.to_datetime(subs.SubmissionDate)



asfloats = ['PublicScoreLeaderboardDisplay',

            'Public',

            'PrivateScoreLeaderboardDisplay',

            'Private',]



subs[asfloats] = subs[asfloats].astype(float)

# subs.IsAfterDeadline.mean()



subs = subs.query('not IsAfterDeadline').copy()

subs['CompetitionId'] = subs.TeamId.map(teams.CompetitionId)

# subs['CompetitionId'].nunique()



def comp_id_for_field(value, field='Slug'):

    idx = comps[field]==value

    if idx.sum() < 1:

        return -1

    return comps.loc[idx].index[0]



# values some competitions use as invalid scores

for bad in [99, 999999]:

    for c in asfloats:

        idx = (subs[c] == bad)

        subs.loc[idx, c] = subs.loc[idx, c].replace({bad: np.nan})



# Display order: most recent competitions first

subs = subs.sort_values(['CompetitionId', 'Id'], ascending=[False, True])
plt.rc("figure", figsize=(14, 6))

plt.rc("font", size=14)

plt.rc("axes", xmargin=0.01)

plt.rc("axes", edgecolor='#606060')





def find_range(scores):

    scores = sorted(scores)

    n = len(scores)

    max_i = n - 1

    for i in range(n // 2, n):

        best = scores[:i]

        if len(best):

            m = np.mean(best)

            s = np.std(best)

            if s != 0:

                z = (scores[i] - m) / s

                if abs(z) < 3:

                    max_i = i

    return scores[0], scores[max_i]





def get_range(df):

    comp_id = df.iloc[0].CompetitionId

    c = comps.loc[comp_id]



    mul = -1 if c.EvaluationAlgorithmIsMax else 1

    a, b = find_range(df.Public.dropna().values * mul)

    A, B = find_range(df.Private.dropna().values * mul)



    A = min(a, A) * mul

    B = max(b, B) * mul



    R = (B - A)

    B += R / 20

    A -= R / 20

    return min(A, B), max(A, B)
COLORS = dict(Public='blue', Private='red')



for i, (comp_id, subs_df) in enumerate(subs.groupby('CompetitionId', sort=False)):



    if subs_df.shape[0] < 3:

        continue

    if subs_df.Public.count() < 1:

        continue

    if subs_df.Private.count() < 1:

        continue

    

    c = comps.loc[comp_id]

    df = subs_df.sort_values('Id').reset_index()

    team_id = df.iloc[0].TeamId

    f = 'max' if c.EvaluationAlgorithmIsMax else 'min'

        

    mcols = ['UserName', 'RequestDate', 'DisplayName', 'SubCount',

             'RegisterDate', 'PerformanceTier']

    tcols = ['TeamName', 'ScoreFirstSubmittedDate', 'LastSubmissionDate',

             'PublicLeaderboardRank', 'PrivateLeaderboardRank']



    team = teams.query(f'CompetitionId=={c.name}').iloc[0]

    members = tmemb.query(f'TeamId=={team_id}').copy()

    members['SubCount'] = members.UserId.map(df.SubmittedUserId.value_counts()).fillna(0)

    members = members[mcols].set_index('UserName')

    members = members.T.dropna(how='all').T

    members.columns = members.columns.str.replace(r'([a-z])([A-Z])', r'\1<br/>\2')



    markup = (

        '<h1 id="{Slug}">{Title}</h1>'

        '<p>'

        'Type: {HostSegmentTitle} &mdash; <i>{Subtitle}</i>'

        '<br/>'

        '<a href="https://www.kaggle.com/c/{Slug}/leaderboard">Leaderboard</a>'

        '<br/>'

        'Dates: <b>{EnabledDateText}</b> &mdash; <b>{DeadlineDateText}</b>'

        '<br/>'

        '<b>{TotalTeams}</b> teams; <b>{TotalCompetitors}</b> competitors; '

        '<b>{TotalSubmissions}</b> submissions'

        '<br/>'

        'Leaderboard percentage: <b>{LeaderboardPercentage}</b>'

        '<br/>'

        'Evaluation: <a title="{EvaluationAlgorithmDescription}">{EvaluationAlgorithmName}</a>'

        '<br/>'

        'Reward: <b>{RewardType}</b> {RewardQuantity} [{NumPrizes} prizes]'

        '<br/>'

        ).format(**c)



    markup += f'<h3>Team Members</h3>'

    markup += members.to_html(index_names=False, notebook=True, escape=False, na_rep='')

    markup += f'<h3>Submissions</h3>'

    display(HTML(markup))

    

    title = c.Title

    title += (' "{TeamName}"'

              ' - [public {PublicLeaderboardRank:.0f} '

              '| private {PrivateLeaderboardRank:.0f}]').format(**team)

    

    for t in ['Public', 'Private']:

        ax = df[t].plot(legend=True, color=COLORS[t])



        ser = df.Id.isin(teams[f'{t}LeaderboardSubmissionId'])

        q = df.loc[ser]

        plt.scatter(np.where(ser)[0], q[t], color=COLORS[t])



        # dotted line of peak score

        xs = np.arange(df.shape[0])

        yb = np.ones(df.shape[0])

        plt.plot(xs, yb * df[t].apply(f), linestyle=':', color=COLORS[t])



    if c.Days > 7:

        last_week = (df['SubmissionDate'] >= c.FinalWeek)

        week_markers = np.where(last_week)[0]

        if len(week_markers):

            plt.axvspan(week_markers.min(), week_markers.max(), color='k', alpha=0.1)



    if df.shape[0] > 4:

        bottom, top = get_range(df)

        plt.ylim(bottom, top)

    plt.title(title)

    plt.ylabel(c.EvaluationLabel)

    plt.xlabel('Submission Index')

    plt.xlim(-1, df.shape[0])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    plt.show()