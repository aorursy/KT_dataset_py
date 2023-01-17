import gc, os, sys, time

import pandas as pd, numpy as np

from itertools import combinations

from IPython.display import HTML, display

import matplotlib.pyplot as plt

import seaborn as sns



pd.options.display.max_rows = 200
plt.rc('figure', figsize=(12, 8))
IN_DIR = '../input/meta-kaggle'

if not os.path.isdir(IN_DIR):

    IN_DIR = '../input'

len(os.listdir(IN_DIR))
users = pd.read_csv(os.path.join(IN_DIR, 'Users.csv'), parse_dates=['RegisterDate'])

idx = users.DisplayName.str.len() <= 1

users.loc[idx, 'DisplayName'] = users.loc[idx, 'UserName']

users.UserName.fillna('', inplace=True)

users.DisplayName.fillna('[deleted user]', inplace=True)

users.DisplayName = users.DisplayName.str[:32]

users = users.set_index('Id')

users.shape
one_day = pd.Timedelta(1, 'd')

latest = users.RegisterDate.max() + one_day

latest
users['Age'] = ((latest - users.RegisterDate) / one_day).astype('int32')
EXCLUDE_USERS = [2080166] # Kaggle Kerneler - very high stats that distort the league tables!



users.loc[EXCLUDE_USERS].T
users = users.drop(EXCLUDE_USERS)

users.shape
users.head()
def columns(fn):

    df = pd.read_csv(fn, nrows=5)

    return df.columns



def user_columns(fn):

    return [c for c in columns(fn) if 'UserId' in c]
for f in sorted(os.listdir(IN_DIR)):

    if '.csv' in f:

        csv = os.path.join(IN_DIR, f)

        cols = user_columns(csv)

        if len(cols) < 1:

            continue

        table = f.replace('.csv', '')

        df = pd.read_csv(csv, usecols=['Id'] + cols)

        # ForumMessageVotes contents duplicated

        # https://www.kaggle.com/kaggle/meta-kaggle/discussion/181883

        # must use drop_duplicates

        df = df.drop_duplicates(subset=['Id'])

        for col in cols:

            tag = f'Count_{table}_{col}'

            print(tag)

            vc = df[col].value_counts()

            ser = users.index.map(vc)

            users[tag] = ser.fillna(0).astype('int32')
MEANINGS = {

    "Datasets_CreatorUserId": "create a dataset",

    "Datasets_OwnerUserId": "own a dataset",

    "DatasetVersions_CreatorUserId": "create a dataset version",

    "DatasetVotes_UserId": "vote for a dataset",

    "Datasources_CreatorUserId": "create a datasource",

    "ForumMessages_PostUserId": "post a forum message",

    "ForumMessageVotes_FromUserId": "vote for a forum message",

    "ForumMessageVotes_ToUserId": "receive a forum vote",

    "Kernels_AuthorUserId": "author a kernel",

    "KernelVersions_AuthorUserId": "run a new version of a Notebook",

    "KernelVotes_UserId": "vote for a Notebook",

    "Submissions_SubmittedUserId": "submit to a competition",

    "TeamMemberships_UserId": "enter a competition (agree to rules)",

    "UserAchievements_UserId": "reach a new achievement milestone",

    "UserFollowers_UserId": "follow a user",

    "UserFollowers_FollowingUserId": "get followed by a user",

    "UserOrganizations_UserId": "add an organization",

    "Total_Activities": "appear in any activity",

}



pd.Series(MEANINGS)
N_SHOW = 50
tier_names = np.asarray(['novice', 'contributor', 'expert', 'master', 'grandmaster', 'staff'])

tier_colors = np.asarray(["#2ECB99", "#00BFF9", "#9A5289", "#FF6337", "#DFA848", "#000000"])

tier_html = np.asarray([f'<font color={c}>{n}</font>' for c, n in zip(tier_colors, tier_names)])

bar_color = '#20beff'



def user_name_link(r):

    return f'<a href="https://www.kaggle.com/{r.UserName}">{r.DisplayName}</a>'





def setup_user(df):

    uid = df.apply(user_name_link, axis=1)

    df.pop('UserName')

    df.pop('DisplayName')

    df['Tier'] = tier_html[df.PerformanceTier]

    df['DisplayName'] = uid





def league_table(col, src_df=users):

    name = col.replace('_', ' ')

    h1 = f"<H1 id={col}>{name}</H1>"

    h2 = f"<P>How many times did user <i>{MEANINGS[col]}</i>?"

    display(HTML(h1+h2))

    #

    col = "Count_" + col

    df = src_df.sort_values(col, ascending=False).head(N_SHOW)

    setup_user(df)

    df['PerDay'] = (df[col] / df['Age']).round(2)

    df['Rank'] = df[col].rank(method='min', ascending=False).astype(int)

    use = ['Rank', 'DisplayName', 'Tier', col, 'PerDay']

    return df[use].style.bar(subset=[col, 'PerDay'], vmin=0, color=bar_color)





def ratio_league_table(a, b, src_df=users):

    df = src_df.sort_values(a, ascending=False).head(N_SHOW)

    setup_user(df)

    df['Ratio'] = (df[b] / df[a]).round(2)

    df['Rank'] = df[a].rank(method='min', ascending=False).astype(int)

    use = ['Rank', 'DisplayName', 'Tier', a, b, 'Ratio']

    return df[use].style.bar(subset=[a, b], vmin=0, color=bar_color)



# for c in activity_sums.index: print(f'league_table("{c}")')
league_table("TeamMemberships_UserId")
league_table("Submissions_SubmittedUserId")
league_table("KernelVersions_AuthorUserId")
league_table("Kernels_AuthorUserId")
league_table("KernelVotes_UserId")
league_table("ForumMessages_PostUserId")
league_table("ForumMessageVotes_ToUserId")
league_table("ForumMessageVotes_FromUserId")
league_table("DatasetVersions_CreatorUserId")
league_table("Datasets_CreatorUserId")
league_table("Datasources_CreatorUserId")
league_table("Datasets_OwnerUserId")
league_table("DatasetVotes_UserId")
league_table("UserFollowers_FollowingUserId")
league_table("UserFollowers_UserId")
league_table("UserOrganizations_UserId")
# ratio_league_table("Count_ForumMessageVotes_ToUserId", "Count_ForumMessageVotes_FromUserId")

ratio_league_table(

    "VotesReceived", "VotesGiven", 

    users.rename(

        columns={

            "Count_ForumMessageVotes_FromUserId": "VotesGiven",

            "Count_ForumMessageVotes_ToUserId": "VotesReceived"

        }))
ratio_league_table(

    "Followed", "Following", 

    users.rename(

        columns={

            "Count_UserFollowers_UserId": "Following",

            "Count_UserFollowers_FollowingUserId": "Followed"

        }))
all_col_counts = users.columns[users.columns.str.startswith('Count_')]

len(all_col_counts)
users.Count_UserAchievements_UserId.value_counts()
count_cols = [c for c in all_col_counts if c != 'Count_UserAchievements_UserId']

len(count_cols)
users[count_cols].sum(1).value_counts().head()
users.query('UserName=="jtrotman"').T
users['Sum_Activity_Flags'] = (users[count_cols]>0).sum(1)
users.Sum_Activity_Flags.value_counts().sort_index()
users.Sum_Activity_Flags.plot.hist(bins=17, log=True, title='Count of activity types over all Kaggle users');
(users.Sum_Activity_Flags==0).mean()
users.Sum_Activity_Flags.max()
counts_df = (users.query('Sum_Activity_Flags>0')[count_cols])

counts_df.shape
counts_df.head().T
counts_df.columns = counts_df.columns.str.replace('^Count_', '')

counts_df.columns = counts_df.columns.str.replace('_', '\n')
plt.figure(figsize=(14, 12))

sns.heatmap(counts_df.corr(method='spearman'), vmin=-1, cmap='RdBu', annot=True, linewidths=1)

plt.title('Kaggle User Activity Counts - Spearman Correlation');
idx = users.Sum_Activity_Flags==users.Sum_Activity_Flags.max()

idx.sum()
show = ['UserName', 'DisplayName', 'RegisterDate', 'PerformanceTier']
users[idx][show]
league_table('Total_Activities', users.assign(Count_Total_Activities=users[count_cols].sum(1)))
n_users = len(users)

n_users
activity_sums = (users[count_cols]>0).sum(0).to_frame("UserCount")

activity_sums["PercentageOfUsers"] = ((activity_sums["UserCount"] / n_users) * 100).round(2)

activity_sums.index = activity_sums.index.str.replace("^Count_", "").map(MEANINGS.get)

activity_sums.sort_values("UserCount", ascending=False)
def users_with_n_activities(n, min_count=1):

    bi_sum = users.Sum_Activity_Flags==n

    for cols in combinations(count_cols, n):

        idx = bi_sum

        for c in cols:

            idx = (idx & (users[c]>0))

            n = idx.sum()

            if n<min_count:

                break

        if n>=min_count:

            yield (n,) + cols



def users_with_n_activities_df(n, min_count=1):

    df = pd.DataFrame.from_records(

        users_with_n_activities(n, min_count),

        columns=['Count'] + list(range(n))

    )

    return df
def show_df(df):

    df = df.sort_values('Count', ascending=False)

    df = df.reset_index(drop=True)

    return df.style.bar(subset=['Count'], color=bar_color)
show_df(users_with_n_activities_df(1))
show_df(users_with_n_activities_df(2))
show_df(users_with_n_activities_df(3, min_count=2000))
users.shape
entered = users.Count_TeamMemberships_UserId>0

submitted = users.Count_Submissions_SubmittedUserId>0



idx = (

 (users.Sum_Activity_Flags==0)

 | 

 ((users.Sum_Activity_Flags==1) & (entered))

 | 

 ((users.Sum_Activity_Flags==2) & (entered) & (submitted))

)
idx.sum()
idx.mean()
len(users) - idx.sum()
plt.rc('font', size=14)
users.RegisterDate.value_counts().plot(title='Kaggle User Registrations over Time')

plt.grid();
users.RegisterDate.value_counts().sort_index().tail(365 * 8).plot(title='Kaggle User Registrations over Time - log scale', logy=True)

plt.grid();
users.RegisterDate.value_counts().sort_index().rolling(14).mean().tail(365 * 8).plot(title='Kaggle User Registrations over Time - log scale', logy=True)

plt.grid();
dormant = users.Sum_Activity_Flags == 0

reg_date = users.RegisterDate

max_date = reg_date.max()

# floor dates to 1 week resolution for smoothing

reg_week = reg_date - (reg_date.dt.dayofweek * one_day)

dormant.groupby(reg_week).mean().plot(title='Kaggle Rate of Dormant Accounts over Time')

plt.axvspan(max_date - (180 * one_day), max_date, color='k', alpha=0.2)

plt.grid();
users.shape
users.index.max()
uids = users[[]].reset_index()

uids = uids.sort_values('Id')

uids['Id'].diff().plot(title='Kaggle User ID Jumps', logy=True)

plt.grid();
uids['GapToNext'] = uids.Id - uids.Id.shift()
uids.query('GapToNext>1').plot.scatter('Id', 'GapToNext', logy=True, title='Kaggle User ID Gaps');
uids[['Id']].diff().describe().T
uids['Id'].diff().plot.hist(bins=50, logy=True, title='Kaggle User ID Gaps')

plt.grid();
users.PerformanceTier.value_counts()
vc = users.PerformanceTier.value_counts()
vc.index = tier_names[vc.index]
vc
ty = users.groupby([users.RegisterDate.dt.year, users.PerformanceTier]).size()

ty = ty.unstack()

ty = ty.fillna(0)

ty = ty.astype(int)

ty.columns = tier_names
ty.style.background_gradient(axis=0)
tier_sums = users.groupby([users.Sum_Activity_Flags, users.PerformanceTier]).size()

tier_sums = tier_sums.unstack()

tier_sums = tier_sums.fillna(0)

tier_sums = tier_sums.astype(int)

tier_sums.columns = tier_names
tier_sums.style.background_gradient(axis=0)
qstr = 'PerformanceTier==1 and Sum_Activity_Flags==0'

# users.query(qstr) # use this to see who they are

users.query(qstr).RegisterDate.dt.year.value_counts() # or this to just summarize
# GM with least activites!

# no need to point this out publicly

# try other queries if you like...



# users.query('PerformanceTier==4 and Sum_Activity_Flags<=4')
# Novices who've nevertheless done everything



# users.query('PerformanceTier==0 and Sum_Activity_Flags==16').T
tier_sums = users.groupby([

    users.Count_Submissions_SubmittedUserId > 0,

    users.Count_Kernels_AuthorUserId > 0,

    users.Count_ForumMessages_PostUserId > 0,

    users.PerformanceTier

]).size().unstack().fillna(0).astype(int)

tier_sums.columns = tier_names
tier_sums.index.names = ["Submit", "Kernel", "Post"]
tier_sums.style.background_gradient(axis=0)
active_users = users.loc[users.Sum_Activity_Flags > 0]
active_users.to_csv("ActiveUsers.csv")