# Fork and change this - save a breakdown of points per competition for each user in this list

KAGGLERS = ['jtrotman']
%matplotlib inline

import numpy as np, pandas as pd



DISPLAY_ROWS = 100

BAR_COLOR = '#5fd65f'

pd.options.display.max_rows = DISPLAY_ROWS



def do_read_csv(name, dates=None):

    path = f'../input/meta-kaggle/{name}.csv'

    df = pd.read_csv(path, parse_dates=dates, low_memory=False)

    print (df.shape, path)

    return df



def add_true_points(users):

    ach = do_read_csv('UserAchievements').set_index('UserId')

    d = ach.loc[ach.AchievementType=="Competitions"].Points

    users['TruePoints'] = users.index.map(d).fillna(0).astype(int)

    users['TrueRank'] = users['TruePoints'].rank(ascending=False, method='first').astype(int)



def user_name_link(r):

    return f'<a href="https://www.kaggle.com/{r.UserName}">{r.DisplayName}</a>'



def comp_link(r):

    return f'<a href="https://www.kaggle.com/c/{r.Slug}" name="{r.Subtitle}">{r.Title}</a>'



def users_show(df, add_rank='TruePoints'):

    # adjustments for display

    df = df.head(DISPLAY_ROWS).copy()

    uid = df.apply(user_name_link, axis=1)

    df.pop('UserName')

    df.pop('DisplayName')

    df.insert(0, 'DisplayName', uid)

    df.insert(0, 'Rank', df[add_rank].rank(ascending=False, method='first').astype(int))

    df = df.set_index('Rank')

    # put newlines in column names for better display in Kaggle UI

    df.columns = df.columns.str.replace('([a-z])([A-Z])', r'\1\n\2')

    bar = [c for c in ['Points', 'True\nPoints'] if c in df.columns]

    return df.style.bar(subset=bar, vmin=0, color=BAR_COLOR)
competitions = do_read_csv('Competitions', dates=['DeadlineDate']).set_index('Id')

teams = do_read_csv('Teams').set_index('Id')

members = do_read_csv('TeamMemberships').set_index('Id')

users = do_read_csv('Users').set_index('Id')

add_true_points(users)
cshow = [

    'Competition',

    'Subtitle',

    'DeadlineDate',

    'TotalTeams',

]

competitions.assign(Competition=competitions.apply(comp_link, 1)).query(

    'HostSegmentTitle!="InClass"').sort_values(

        'DeadlineDate', ascending=False)[cshow].head().style
users.TruePoints.value_counts().head() # over 3MM have 0 points
users.TruePoints.plot.hist(bins=100,

                           log=True,

                           title='Log Count of Users with Points',

                           figsize=(14, 7))
users.query('TruePoints>=50000').shape
users.query('TruePoints>=50000').TruePoints.plot.hist(bins=20,

    title='Count of Users with >=50k Points', figsize=(14, 7))
users_show(users.sort_values('TruePoints', ascending=False).head(DISPLAY_ROWS))
def kaggle_points(rank=1, nteams=1, team_size=1, t=0.0, mult=1, teams_exp=0.5):

    return ((100000. / (team_size**teams_exp))  # team size factor

            * (rank**-0.75)                     # leaderboard position

            * (np.log10(1 + np.log10(nteams)))  # size of competition

            * (np.exp(-t / 500.))               # time decay (days since deadline)

            * (mult)                            # some comps are half points

            )
competitions = competitions.query('TotalTeams>0').copy()

competitions.shape
from_comps = ['TotalTeams', 'DeadlineDate', 'UserRankMultiplier']



teams = teams.merge(competitions[from_comps], left_on='CompetitionId', right_index=True)

teams = teams.dropna(subset=['TotalTeams'])
# number of people in a team

teamSize = members.TeamId.value_counts()

teams['TeamSize'] = teams.index.map(teamSize)
teams['FinalRank'] = teams['PrivateLeaderboardRank']

idx = teams.FinalRank.isnull()

teams.loc[idx, 'FinalRank'] = teams.loc[idx, 'PublicLeaderboardRank']

# No rank means entered but no submissions = no points

teams = teams.dropna(subset=['FinalRank'])
ref_date = teams.DeadlineDate.max()

TSTAMP = ref_date.strftime('%y%m%d') # Timestamp for filenames

teams['DaysSince'] = (ref_date - teams.DeadlineDate).dt.days
MISS = 'Whiff'
from_users = ['DisplayName', 'UserName', 'PerformanceTier']

members = members.merge(users[from_users], left_on='UserId', right_index=True)
membersTeamSize = members.TeamId.map(teamSize)

membersMedal = members.TeamId.map(teams.Medal)

members['Comps'] = 1  # sum this to count comps

members['Solo'] = (membersTeamSize==1).astype(int)

members['SoloGold'] = ((membersMedal==1) & (membersTeamSize==1)).astype(int)

members['Gold'] = (membersMedal==1).astype(int)

members['Silver'] = (membersMedal==2).astype(int)

members['Bronze'] = (membersMedal==3).astype(int)

members[MISS] = (membersMedal.isnull()).astype(int)
def render(team_exponent):

    teams['Points'] = kaggle_points(teams.FinalRank,

                                    teams.TotalTeams,

                                    teams.TeamSize,

                                    teams.DaysSince,

                                    teams.UserRankMultiplier,

                                    team_exponent)

    

    # each member gets points as computed for team

    members['Points'] = members.TeamId.map(teams.Points)

    

    aggs = {

        'Comps': 'sum',

        'Solo': 'mean',

        'SoloGold': 'sum',

        'Gold': 'sum',

        'Silver': 'sum',

        'Bronze': 'sum',

        MISS: 'sum',

        'Points': 'sum'

    }

    df = members.groupby('UserName').agg(aggs)

    df = df.sort_values('Points', ascending=False).reset_index()

    # round down points now that they're all summed

    df['Points'] = np.floor(df.Points).astype(int)

    df['Solo'] = np.round(df.Solo, 2)

    # add in user fields

    df = df.join(users.set_index('UserName'), on='UserName', how='left')

    return df



# Summarize points earned in each competition for a user

def summarize_user(user, fname):

    df = members.query(f'UserName=="{user}"')

    df = df.sort_values('Points', ascending=False)

    df = df.join(teams.drop(['Points'], 1), on='TeamId')

    cshow = ['CompetitionId', 'DeadlineDate', 'Title', 'FinalRank', 'TeamSize', 'TeamName', 'Points']

    df = df.join(competitions.drop(from_comps, 1), on='CompetitionId')

    df = df[cshow].dropna()

    df.to_csv(fname, float_format='%.0f')



def summarize_users(team_exponent):

    for user in KAGGLERS:

        fname = f'{user}_points_{TSTAMP}_team_exp_{team_exponent}.csv'

        summarize_user(user, fname)



def show(team_exponent):

    df = render(team_exponent)  # Computes Points

    fname = f'comp_ranks_{TSTAMP}_team_exp_{team_exponent}.csv'

    df.query('Points>0').to_csv(fname, index_label='Index')

    summarize_users(team_exponent)  # Uses computed Points

    return users_show(df, add_rank='Points')
show(team_exponent=0.5)
show(team_exponent=1)
show(team_exponent=2)
show(team_exponent=0.75)