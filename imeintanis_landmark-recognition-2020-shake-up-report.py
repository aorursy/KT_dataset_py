import os, sys, subprocess

import numpy as np

import pandas as pd

import gc

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.core.display import HTML, Image

from scipy.stats import skew, kurtosis



# import chart_studio.plotly as py

import plotly.graph_objs as go

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



from html.parser import HTMLParser



pd.set_option('display.max_rows', 150)



print('List of datasets:')

print(os.listdir("../input/"))

print('LB files:')

print(os.listdir("../input/lr-2020-publicpvt-lb/"))
# see: https://www.kaggle.com/pednoi/visualize-the-shakeups-of-10-recent-competitions/data?select=Elo+Merchant+Category+Recommendation+_+Kaggle.html



class RankParser(HTMLParser):



    def __init__(self):

        self.entered = False

        self.sign = ''

        self.data = []

        

        super().__init__()

    

    def handle_starttag(self, tag, attrs):

        if tag == 'td' and ('data-th', 'Change') in attrs:

            self.entered = True

            

        if self.entered and tag == 'span':

            if len(attrs) > 0 and len(attrs[0]) > 1 and attrs[0][1].startswith('position-change'):

                direction = attrs[0][1][len('position-change__'):]

                if direction == 'fallen':

                    self.sign = '-'                    



    def handle_endtag(self, tag):

        if self.entered and tag == 'td':

            self.entered = False

            self.sign = ''



    def handle_data(self, data):

        if self.entered:

            data = '0' if data == '—' else data

            self.data.append(int(self.sign+data.strip()))

    

    def get_data(self):

        return self.data



def read_html(file_path):

    content = open(file_path, encoding='utf-8').read()    

    parser = RankParser()

    parser.feed(content)

    return parser.get_data()





def do_read_csv(name):

    df = pd.read_csv(name, low_memory=False)

    print ('> Loaded:', name, df.shape)

    return df
def plot_hist(title, diff):

    stats = ""

    stats += "count = %d\n" % len(diff)

    stats += "mean = %.2f\n" % np.mean(diff) # always zero because the data are zero-sum

    stats += "std = %.4f\n" % np.std(diff)

    stats += "skew = %.4f\n" % skew(diff)

    stats += "kurtosis = %.4f\n" % kurtosis(diff)

    

    print("Mean shake-up       " ,np.mean(diff))

    print("\nMedian shake-up     " ,np.median(diff))

    print("\nMax shake-up        " ,np.max(diff))

    print("\nMin shake-down ;)   " ,np.min(diff))

    print("\nStd shake-up        " ,np.std(diff))

    

    fig = plt.figure(figsize=(16, 6))

    #     sns.distplot(diff, bins=100)

    plt.hist(diff, bins = 50, edgecolor = 'black', color = 'green')

    plt.text(0.05, 0.5, stats, transform=plt.gca().transAxes)

    plt.xlabel("Places Shake-up")

    plt.ylabel("Frequency")

    plt.title(title)

    plt.show()





def color_negative_red(val):

    """

    Takes a scalar and returns a string with

    the css property `'color: red'` for negative

    strings, black otherwise.

    """

    try:

        color = 'red' if val < 0 else 'black'

    except:

        color = 'black'

    return 'color: %s' % color





def plot_candle(title, diff):

    closes = np.array(range(len(diff)))+1

    opens = closes + np.array(diff)

    highs = np.where(np.array(diff)<0, closes, opens)

    lows =  np.where(np.array(diff)>=0, closes, opens)

    

    hovertext = ['private lb: '+str(c)+'<br>public lb: '+ str(o) +'<br>TeamName: '+str(pvt_lb.iloc[c-1]['TeamName'])  for o, c in zip(opens, closes)]



    trace = go.Ohlc(x=list(range(1, len(diff)+1)), open=opens, high=highs, low=lows, close=closes,

                    increasing=dict(line=dict(color='#800000')), # '#FF6699'

                    decreasing=dict(line=dict(color='#228B22')),          # '#66DD99'

                    text=hovertext, 

                    hoverinfo='text')

    

    layout = go.Layout(

        title = "<b>%s</b>" % title,

        xaxis = dict(

            title='Final ranks (Pvt LB)',

            rangeslider = dict(visible=False)

        ), 

        yaxis=dict(

            title='shakeups',

            autorange='reversed'

        ),

        width=800,

        height=600,

    )

    

    fig = go.Figure(data=[trace], layout=layout)    

    iplot(fig, filename='shakeup_candlestick')

    

    

def make_scatter_competitions(comps, teams):



    shakes = {}

    COLOR_DICT = {0: 'deepskyblue', 1: 'gold', 2: 'silver', 3: 'chocolate'}

    plt.rc('font', size=14)

    for i, df in teams.groupby('CompetitionId', sort=False):

        fname = comps.Slug[i]

        row = comps.loc[i]

        shakeup = df.eval('abs(PrivateLeaderboardRank-PublicLeaderboardRank)').mean() / df.shape[0]

        title = (f'{row.Title} — {row.TotalTeams} teams — '

                 f'{shakeup:.3f} shake-up — {row.DeadlineText}')

        shakes[i] = shakeup

        df = df.sort_values('PrivateLeaderboardRank', ascending=False)  # plot gold last

        ax = df.plot.scatter('PublicLeaderboardRank', 'PrivateLeaderboardRank', c=df.Medal.map(COLOR_DICT), figsize=(15, 15))

        plt.title(title, fontsize=16)

        l = np.arange(df.PrivateLeaderboardRank.max())

        ax.plot(l, l, linestyle='--', linewidth=1, color='Black', alpha=0.5)

        ax.set_xlabel('Public Leaderboard Rank')

        ax.set_ylabel('Private Leaderboard Rank')

        plt.tight_layout()

        plt.show()

    return shakes
CSV_DIR = "../input/lr-2020-publicpvt-lb"

HTML_DIR = "../input/lr-2020-publicpvt-lb"

META_DIR = "../input/meta-kaggle"
# read PUBLIC LB (csv)



public_lb = do_read_csv(f'{CSV_DIR}/LR-2020-public_LB.csv')  

print('Public LB (final @03/10/2020) shape before cleaning:', len(public_lb))



# create df for future use

public_scores = public_lb.groupby(['TeamId'])['Score'].agg('max').sort_values(ascending=False)



public_Scores = pd.DataFrame(public_scores)

public_Scores = public_Scores.reset_index()



public_Scores = public_Scores.merge(public_lb[['TeamId', 'TeamName']], on='TeamId', how='right').drop_duplicates()



print('Public LB (final @03/10/2020) shape after cleaning:', len(public_Scores))
# read PVT LB (html)



pvt_lb = pd.read_excel(f'{CSV_DIR}/LR-2020-pvt_LB.xlsx', header=[0])

pvt_lb = pvt_lb[['#', '△pub', 'Team Name','Score', 'Entries', 'Last']]

pvt_lb.columns = ['Pvt_rank','pos_change', 'TeamName', 'Pvt_score', 'no_submissions', 'Last']                

                



# modify shake-up column

pvt_lb.pos_change = pvt_lb.pos_change.replace(to_replace='—', value=0)

pvt_lb.pos_change = pvt_lb.pos_change.astype(int)







file = 'LR-2020-pvt-LB.htm'

shake_up = read_html(f'{HTML_DIR}/{file}')

pvt_lb['shake'] = np.array(shake_up)



print('No. of Teams in PVT LB (final @03/10/2020):', len(pvt_lb))
assert len(public_Scores)==len(pvt_lb), 'Not valid shapes!'
np.min(shake_up), np.max(shake_up)
plot_hist('Google Landmark Recognition 2020: Shake-Up', shake_up)
plot_candle('Google Landmark Recognition 2020: Shake-Up',  pvt_lb['shake'])
# modified from: https://www.kaggle.com/robikscube/ashrae-leaderboard-and-shake



df = pvt_lb[['Pvt_rank','shake','TeamName','Pvt_score','no_submissions']].copy()



df['medal'] = ''

df.loc[df['Pvt_rank'] <= 139, 'medal'] = '🥉'

df.loc[df['Pvt_rank'] <= 69, 'medal'] = '🥈'

df.loc[df['Pvt_rank'] <= 12, 'medal'] = '🥇'

df = df[['Pvt_rank','medal','shake', 'TeamName', 'Pvt_score']]   # 'public_rank''Score', 'no_submissions'



df.head(100).style.applymap(color_negative_red).hide_index()
# select 20 most recent competitions



teams = do_read_csv(f'{META_DIR}/Teams.csv')

comps = do_read_csv(f'{META_DIR}/Competitions.csv').set_index('Id')

comps['DeadlineText'] = comps.DeadlineDate.str.split().str[0]

comps['DeadlineDate'] = pd.to_datetime(comps.DeadlineDate)



selected_comps = comps[(comps.HostSegmentTitle=='Featured') | (comps.HostSegmentTitle=='Research')].copy()

selected_comps = selected_comps.sort_values('DeadlineDate')[-20:]



# select teams for those competitions

teams = teams.loc[teams.CompetitionId.isin(selected_comps.index)]

teams = teams.assign(Medal=teams.Medal.fillna(0).astype(int))

print(teams.shape)
shakes = make_scatter_competitions(selected_comps, teams)
def fmt_link(row):

    return f'<a target=_blank href="https://www.kaggle.com/c/{row.Slug}">{row.Title}</a>'





show_cols = ['Title', 'HostSegmentTitle', 'TotalTeams','DeadlineText', 'Shakeup']

bars = ['TotalTeams', 'Shakeup']



selected_comps['Shakeup'] = np.array([shakes[key] for key in shakes.keys()])



tmp = selected_comps.assign(Title=selected_comps.apply(fmt_link, 1))

tmp[show_cols].set_index('Title').head(20).style.bar(subset=bars)