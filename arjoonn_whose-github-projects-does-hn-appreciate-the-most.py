import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import seaborn as sns

%pylab inline

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/hacker_news_sample.csv')

df.info()
def get_github(df, limit=30):

    github = df.loc[df['url'].fillna('').str.contains('github')].copy()



    def get_user(link):

        ru = None

        try:

            if 'github.com' in link:

                ru = link.split('/')[3]

                if ru.strip() == '':

                    ru = link.split('/')[2].split('.')[0]

                    if ru.strip() == '':

                        print('empty username on 1', link, ru)

            elif 'github.io' in link:

                ru = link.split('/')[2].split('.')[0]

                if ru.strip() == '':

                    print('empty username on 2', link)

        except:

            pass

        return ru



    github['users'] = github['url'].apply(get_user)

    github = github.dropna(subset=['users'])

    github['users'] = github.users.str.lower()

    github = github.sort_values(by='score', ascending=False)

    top_people = github.groupby('users')['score']

    scoresum = top_people.sum()

    people = scoresum.sort_values(ascending=False)

    names, scores = list(people.index), list(people.values)

    names, scores = names[:limit], scores[:limit]

    return names, scores, github.copy()
names, scores, gh = get_github(df)

print('Usernames/Groupnames which are most appreciated by the HN community')

print('-'*70)

for n, s in zip(names, scores):

    print('{:15} {:10}'.format(n, s))
limit = 10

dates = pd.to_datetime(gh.timestamp)

gh['years'] = dates.dt.year

data = gh[['years', 'users', 'score']].dropna()

crstab = pd.crosstab(data.users, data.years, data.score, aggfunc=np.sum)

crstab = crstab.fillna(0)

crstab['total'] = crstab.sum(axis=1)

rankings = []

for year in list(crstab.columns):

    crstab.sort_values(by=year, ascending=False, inplace=True)

    rankings.append(list(crstab.index)[:limit])



ranks_over_time = pd.DataFrame(rankings, index=list(crstab.columns)).T
def colornames(n):

    cmap = {'facebook': '#3b5998',

            'google': '#dd4b39',

            'microsoft': '#7cbb00',

            'blog': 'black; font-size: 1.3em',

            'django': '#0C3C26',

            'swannodette': '#0000FF',

            'shadowsocks': '#1c9b47',

            'donnemartin': '#f05f40'

           }

    color = cmap.get(n)

    color = color if color is not None else 'black'

    return 'color: {}'.format(color)

ranks_over_time[list(reversed(crstab.columns))].style.applymap(colornames)