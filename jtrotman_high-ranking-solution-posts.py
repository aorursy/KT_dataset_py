# Fork to try out other queries

# e.g. TeamMedal==1 would show only gold medal place teams

# (Top 50 was chosen because that is how many entries Kaggle shows by default.)

QUERY_STR = 'Rank<=50'
import numpy as np, pandas as pd

from collections import Counter

from IPython.core.display import HTML

from bs4 import BeautifulSoup

import html, os, re, sys

import warnings

warnings.filterwarnings("ignore")



DEADLINE_CUTOFF = pd.Timedelta(-15, unit='m')

MEDALS = [ '', '&#129351;', '&#129352;', '&#129353;' ]

TIERS = np.asarray(['Novice', 'Contributor', 'Expert', 'Master', 'GrandMaster', 'Staff', 'Deleted'])



IN_DIR = '../input/meta-kaggle'

if not os.path.isdir(IN_DIR):

    IN_DIR = '../input'



def read_csv_filtered(csv, col, values):

    dfs = [df.loc[df[col].isin(values)]

           for df in pd.read_csv(csv, chunksize=100000)]

    return pd.concat(dfs, axis=0)
# Read all competitions

comps = pd.read_csv(f'{IN_DIR}/Competitions.csv',

                    index_col='Id',

                    parse_dates=['DeadlineDate', 'ProhibitNewEntrantsDeadlineDate'])

comps = comps.query('HostSegmentTitle != "InClass"').copy()

comps['HostName'].fillna('', inplace=True)

# Counts:

# EvaluationAlgorithmAbbreviation     385

# EvaluationAlgorithmName             354

# EvaluationAlgorithmDescription      382

idx = comps.EvaluationAlgorithmName.isnull()

comps.loc[idx, 'EvaluationAlgorithmName'] = comps.loc[idx, 'EvaluationAlgorithmAbbreviation']

idx = comps.EvaluationAlgorithmDescription.isnull()

comps.loc[idx, 'EvaluationAlgorithmDescription'] = comps.loc[idx, 'EvaluationAlgorithmName']

    

# show posts from after entry deadline for some comps

slugs = [ 'nfl-big-data-bowl-2020', 'halite' ]

idx = comps.Slug.isin(slugs)

comps.loc[idx, 'DeadlineDate'] = comps.loc[idx, 'ProhibitNewEntrantsDeadlineDate']



comps['Type'] = comps['HostSegmentTitle']

comps['Deadline'] = comps['DeadlineDate'].dt.date

comps['Reward'] = comps['RewardType'].fillna('?')



curr = {'USD':'$', 'EUR':'€'}

idx = comps['Reward'].isin(curr)

comps.loc[idx, 'Reward'] = (comps.loc[idx, 'Reward'].map(curr)

                          + comps.loc[idx, 'RewardQuantity'].apply(lambda d: f'{d:,.0f}'))



comps = comps.query('TotalTeams>0')

comps = comps.sort_values('DeadlineDate', ascending=False)
# Read teams that took part

teams = read_csv_filtered(f'{IN_DIR}/Teams.csv', 'CompetitionId', comps.index.values)

teams = teams.rename(columns={'Medal': 'TeamMedal'})

teams['TeamMedal'] = teams['TeamMedal'].fillna(0).astype(int)

teams['Rank'] = teams.PrivateLeaderboardRank

teams.loc[teams.Rank.isnull(), 'Rank'] = teams.loc[teams.Rank.isnull(), 'PublicLeaderboardRank']

teams = teams.query(QUERY_STR, engine='python').copy()



# Read members of teams

tmemb = read_csv_filtered(f'{IN_DIR}/TeamMemberships.csv', 'TeamId', teams.Id.values)

teams = teams.set_index('Id')

team_cols = ['CompetitionId', 'Rank', 'TeamName', 'TeamMedal']

tmemb = tmemb.join(teams[team_cols], on='TeamId')

tmemb['TeamSize'] = tmemb.groupby('TeamId').TeamId.transform('count')



comp_forums = set(comps.ForumId.dropna().astype(int))



# Read forum topics

topics = read_csv_filtered(f'{IN_DIR}/ForumTopics.csv', 'ForumId', comp_forums).set_index('Id')



# Read forum messages

msgs = read_csv_filtered(f'{IN_DIR}/ForumMessages.csv', 'ForumTopicId', topics.index.values)

msgs.dropna(subset=['Message'], inplace=True)

msgs['Medal'] = msgs['Medal'].fillna(0).astype(int)

msgs = msgs.join(topics.add_prefix('Topic'), on='ForumTopicId')



# Add stats/counts

msgs['IsFirst'] = msgs.Id.isin(topics.FirstForumMessageId)

msgs['ParaCount'] = msgs['Message'].str.count('</[Pp]>')

msgs['GithubCount'] = msgs['Message'].str.count(r'github(usercontent)?\.(com|io)')

msgs['ImageCount'] = msgs['Message'].str.count(r'<img ')

msgs['LinkCount'] = msgs['Message'].str.count('</[Aa]>')
# Quick fix for COVID competitions that (uniquely), share a forum

# f2comp = comps.reset_index().dropna(subset=['ForumId']).set_index('ForumId').Id

f2comp = comps.reset_index().groupby('ForumId').Id.max()



msgs['CompetitionId'] = msgs.TopicForumId.map(f2comp)



msgs = msgs.merge(tmemb[['UserId', 'Rank', 'TeamName', 'TeamMedal', 'TeamSize', 'CompetitionId']],

                  left_on=['PostUserId', 'CompetitionId'],

                  right_on=['UserId', 'CompetitionId'],

                  how='left')



msgs.dropna(subset=['Rank'], inplace=True)

msgs['Rank'] = msgs['Rank'].astype(int)

msgs['DeadlineDate'] = msgs.CompetitionId.map(comps.DeadlineDate)

msgs['PostDate'] = pd.to_datetime(msgs.PostDate)

delta = (msgs['PostDate'] - msgs['DeadlineDate'])

msgs = msgs.query('(PostDate - DeadlineDate) > @DEADLINE_CUTOFF', engine='python')
msgvotes = read_csv_filtered(f'{IN_DIR}/ForumMessageVotes.csv', 'ForumMessageId', msgs.Id.values)

# Issue: ForumMessageVotes contents duplicated

# https://www.kaggle.com/kaggle/meta-kaggle/discussion/181883

msgvotes = msgvotes.drop_duplicates(subset=['Id'])

msgs['Votes'] = msgs.Id.map(msgvotes.ForumMessageId.value_counts())

msgs['Votes'] = msgs['Votes'].fillna(0).astype(int)



users = read_csv_filtered(f'{IN_DIR}/Users.csv', 'Id', msgs.PostUserId.values).set_index('Id')

idx = users.DisplayName.str.len() <= 1

users.loc[idx, 'DisplayName'] = users.loc[idx, 'UserName']

users.UserName.fillna('', inplace=True)

users.DisplayName.fillna('[deleted user]', inplace=True)

users.DisplayName = users.DisplayName.str[:32]



msgs = msgs.join(users, on='PostUserId')

msgs['PerformanceTier'] = msgs['PerformanceTier'].fillna(6).astype('int8')

# Fill in case Id's were missing from Users.csv

msgs.UserName.fillna('', inplace=True)

msgs.DisplayName.fillna('[deleted user]', inplace=True)
def title_fmt(row):

    return ('<a href="#{Slug}" title="{Subtitle}\n'

            'Host: {HostName}\n'

            'NumPrizes: {NumPrizes}\n'

            'Enabled: {EnabledDate}\n'

            'Public LB: {LeaderboardPercentage}%\n'

            'Evaluation: {EvaluationAlgorithmAbbreviation}\n'

            'UserRankMultiplier: {UserRankMultiplier}\n'

            'CanQualifyTiers: {CanQualifyTiers}\n'

            'TotalCompetitors: {TotalCompetitors}\n'

            'MaxDailySubmissions: {MaxDailySubmissions}\n'

            'NumScoredSubmissions: {NumScoredSubmissions}'

            '">'

            '{Title}'

            '</a>').format(**row)



cols = ['Type', 'Deadline', 'Reward', 'TotalTeams']

tmp = comps.assign(Title=comps.apply(title_fmt, axis=1)).set_index('Title')

tmp[cols].style
GITHUB_ICON = '<img src="https://github.com/favicon.ico" alt="Github" width=16 height=16>'

IMAGE_ICON = '📊'

LINK_ICON = '🔗'

MAX_PREVIEW = 80 * 25



# return a boolean indicator cell:

# - an icon if the content exists

# - blank if it is zero

def bool_td(row, src, label, txt):

    if row[src] < 1:

        return '<td>&nbsp;</td>'

    if row[src] > 1:

        label += 's'

    return f'<td title="{row[src]} {label}">{txt}</td>'





def preview(r):

    txt = BeautifulSoup(r.Message, 'html').get_text()

    txt = re.sub(r'\[quote.*\[/quote\]', ' ', txt, flags=re.S)

    txt = txt.strip()

    txt = (txt[:MAX_PREVIEW] + '…') if len(txt) > MAX_PREVIEW else txt

    txt = html.escape(txt, quote=True)

    return (r.PostDate.strftime('%c') + '\n\n' + txt)





def user_txt(r):

    surprise = '!' * int(min(10, max(0, r.TeamSize - 5)))

    memb = f'{r.TeamSize:.0f} members{surprise}' if r.TeamSize > 1 else 'solo'

    txt = (f'{r.UserName} [{TIERS[r.PerformanceTier]}]\n'

            f'Registered: {r.RegisterDate}\n'

            f'Team: "{r.TeamName}" ({memb})'

    )

    txt = html.escape(txt, quote=True)

    return txt
# Thanks to / Modified from:

# https://www.kaggle.com/jazivxt/top-private-leaderboard-kernels

# https://www.kaggle.com/shivamb/data-science-glossary-on-kaggle

# https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions

def comp_html(df):



    comp_id = df.CompetitionId.values[0]

    c = comps.loc[comp_id]

    

    comp_url = "https://www.kaggle.com/c/" + str(c['Slug'])

    comp_img1 = f'https://storage.googleapis.com/kaggle-competitions/kaggle/{comp_id}/logos/header.png'

    comp_img2 = 'https://www.kaggle.com/static/images/competition-noimage.png'

    hs = """<div style="border: 2px solid black; padding: 10px; height:100px; width:500; background-image: url('""" + comp_img1 + """'), url('""" + comp_img2 + """'); background-size: cover;">

                <h1 style='color:#ffffff; text-shadow: 2px 2px #000000;' id='""" + c['Slug'] + """'>""" + c['Title'] + """</h1>

            </div>"""

    hs += f"<p><b>{c.HostName}</b> [{c.HostSegmentTitle}] <a href='{comp_url}'>{c.Subtitle}</a><br/>"

    

    hs += ('Dates: <b>{EnabledDate}</b> &mdash; <b>{DeadlineDate}</b><br/>'

           '<b>{TotalTeams}</b> teams; <b>{TotalCompetitors}</b> competitors; <b>{TotalSubmissions}</b> submissions<br/>'

           'Public LB: <b>{LeaderboardPercentage}</b>%<br/>'

           'Evaluation: <a title="{EvaluationAlgorithmDescription}">{EvaluationAlgorithmName}</a><br/>'

           'Reward: <b>{Reward}</b> [{NumPrizes} prizes]<br/>').format(**c)



    hs += ( f'<table>'

            f'<tr>'

            f'<th title="Leaderboard Rank">LB</th>'

            f'<th>Title</th>'

            f'<th>Votes</th>'

            f'<th>Author</th>'

            f'<th title="Medal">M</th>'

            f'<th title="Number of paragraphs">&para;</th>'

            f'<th title="Github links" width=32> {GITHUB_ICON} </th>'

            f'<th title="Images">{IMAGE_ICON}</th>'

            f'<th title="Hyperlinks">{LINK_ICON}</th>'

            f'</tr>'

    )

    for i, row in df.iterrows():

        

        url = f"{comp_url}/discussion/{row.ForumTopicId}#{row.Id}"

        aurl = f"https://www.kaggle.com/{row.UserName}"

        isfirst = ' *' if row.IsFirst else ''

        hs += (

            f'<tr>'

            f'<td class="m{row.TeamMedal:.0f}">{row.Rank}</td>'

            f'<td><a href="{url}" title="{preview(row)}"><b>{row.TopicTitle + isfirst}</b></a></td>'

            f'<td><b>{row.Votes}</b></td>'

            f'<td><a href="{aurl}" title="{user_txt(row)}">{row.DisplayName}</a></td>'

            f'<td>{MEDALS[row.Medal]}</td>'

            f'<td>{row.ParaCount}</td>'

            f'{bool_td(row, "GithubCount", "github link", GITHUB_ICON)}'

            f'{bool_td(row, "ImageCount", "image", IMAGE_ICON)}'

            f'{bool_td(row, "LinkCount", "hyperlink", LINK_ICON)}'

            f'</tr>'

        )

    hs += "</table><hr/>"

    return hs
style = '''

<style>

.m0 { background-color:white; color:black; font-weight: bold; }

.m1 { background-color:gold; color:white; font-weight: bold; }

.m2 { background-color:silver; color:white; font-weight: bold; }

.m3 { background-color:chocolate; color:white; font-weight: bold; }

</style>

'''



display(HTML(style + '<h1>Competitions</h1>'))



# record stats per user

stats = {

    'Competitions': Counter(),

    'Topics': Counter(),

    'Posts': Counter(),

    'Shown': Counter(),

    'Votes': Counter(),

    'Gold': Counter(),

    'GithubCount': Counter(),

    'ImageCount': Counter(),

    'LinkCount': Counter(),

}



AGG_SUM = ['Votes', 'GithubCount', 'ImageCount', 'LinkCount']



# Display order: most recent competitions first

msgs = msgs.sort_values('CompetitionId', ascending=False)



for comp_ord, compdf in msgs.groupby('CompetitionId', sort=False):

    df = compdf.sort_values(['Rank', 'Votes'], ascending=[True, False])

    df['UserPostCount'] = df.groupby('PostUserId', sort=False).PostUserId.cumcount()

    stats['Competitions'].update(df.PostUserId.unique())

    gb = df.groupby('PostUserId')

    for c in AGG_SUM:

        stats[c].update(gb[c].sum().to_dict())

    stats['Posts'].update(df.PostUserId)

    stats['Topics'].update(df.query('IsFirst').PostUserId)

    stats['Gold'].update(df.query('Medal==1').PostUserId)

    df = df[(df.UserPostCount == 0) | (df.Medal == 1)]

    stats['Shown'].update(df.PostUserId)

    display(HTML(comp_html(df)))
display(HTML("<h1 id='league-table'>User League Table</h1>"

             "<p>Finally: writers of note. "

             "Who wrote most post-deadline posts? "

             "And who's notes got most votes?"))



def user_name_link(r):

    return f'<a href="https://www.kaggle.com/{r.UserName}">{r.DisplayName}</a>'



df = pd.DataFrame(stats)

df = df.fillna(0)

df = df.astype(int)

df = df.join(users[['UserName', 'DisplayName']])

df = df.sort_values(['Shown', 'Competitions', 'Votes'], ascending=False)

df.to_csv('UserSolutionPostStats.csv', index_label='UserId')



uid = df.apply(user_name_link, axis=1)

df.pop('UserName')

df.pop('DisplayName')

df.insert(0, 'User', uid)

df = df.set_index('User')

df.columns = df.columns.str.replace("Count$", "s") # hack!

df.head(50).style