%matplotlib inline

import pandas as pd

import numpy as np

import os, re, sys

import matplotlib.pyplot as plt

from IPython.core.display import HTML, Image
YEAR = 2016

OUTPUT_ZIP = f'mm_heatmaps_{YEAR}.zip'

DIR  = f'../input/2016-march-ml-mania'

PDIR = f'{DIR}/predictions/predictions'

TDIR = f'../input/meta-march-mania'

PNG_DIR = '/kaggle/plots'
os.makedirs(PNG_DIR, exist_ok=True)
teams = pd.read_csv(f'{DIR}/Teams.csv', index_col=0)

teams.shape
seeds = pd.read_csv(f'{DIR}/TourneySeeds.csv', index_col=2).query(f'Season=={YEAR}')

seeds.shape
res = pd.read_csv(f'{TDIR}/GroundTruths.csv', index_col=0).query(f'Season=={YEAR}')

res.shape
ids = set(res.Low) | set(res.High)

nteams = len(ids)

nteams
seeds = seeds[seeds.index.isin(ids)] # 64 teams

seeds = seeds.join(teams)

', '.join(seeds.Team_Name) # Team_Name -> seed based position
inds = dict(zip(seeds.index, range(nteams)))

print(inds)  # TeamID -> seed based position
ICOLS = ['i1', 'i2']



def add_inds(df, col):

    parts = getattr(df, col).str.split('_')

    df = df.assign(**{ICOLS[i]:parts.str[i+1].astype(int).map(inds) for i in range(2)})

    df = df.dropna()

    df[ICOLS] = df[ICOLS].astype(int)

    return df



# return a submission in a standard form

def read_sub(name):

    df = pd.read_csv(name)

    df.columns = df.columns.str.lower()

    df = add_inds(df, 'id')

    return df[['id', 'pred'] + ICOLS].set_index('id')



def log_loss(df):

    p = np.where(df.Truth, df.pred, 1 - df.pred)

    # clip low predictions to avoid infinite loss

    p = p.clip(min=1e-15)

    return (-np.log(p)).mean()



def score(sub):

    df = sub.join(res, how='inner')

    return log_loss(df)



# return final score and the round 1 score from first 32 games

def score_multi(sub):

    df = sub.join(res, how='inner')

    return log_loss(df), log_loss(df.query('Round==1'))



def to_matrix(sub):

    m = np.ones((nteams, nteams)) * 0.5

    m[sub.i1, sub.i2] = sub.pred

    m[sub.i2, sub.i1] = 1 - sub.pred

    return m
plt.rc('figure', figsize=(14, 14))

plt.rc('font', size=12)
def save_heatmap(probs, filename, cmap=plt.cm.seismic):

    fig, ax = plt.subplots()

    heatmap = ax.pcolormesh(probs, vmin=0., vmax=1., cmap=cmap)



    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.spines['left'].set_visible(False)

    

    ax.invert_yaxis()

    ax.tick_params(direction='out')

    ax.xaxis.tick_top()

    ax.yaxis.tick_left()

    plt.xticks(rotation=90)

    

    team_labels = seeds.Team_Name.values

    # put the major ticks at the middle of each cell

    ax.set_xticks(np.arange(nteams)+0.5, minor=False)

    ax.set_yticks(np.arange(nteams)+0.5, minor=False)

    ax.set_xticklabels(team_labels)

    ax.set_yticklabels(team_labels)

    plt.savefig(filename, bbox_inches='tight')   
save_heatmap(to_matrix(add_inds(res, 'index').rename(columns={'Truth': 'pred'})), f'truth_{YEAR}', cmap=plt.cm.bwr)
lst = os.listdir(PDIR)

len(lst)
errs = []

sums = 0

count = 0

for name in lst:

    if name.lower().endswith('.csv'):

        try:

            sub = read_sub(f'{PDIR}/{name}')

            out = re.sub(r'\.csv$', '.png', name, re.IGNORECASE)

            s = score_multi(sub)

            m = to_matrix(sub)

            sums += sub[['pred']].clip(0, 1)

            count += 1

            save_heatmap(m, f'{PNG_DIR}/{s[0]:.6f}_{s[1]:.6f}_{out}')

            plt.close()

        except:

            errs.append(name)
len(errs)
errs
ensemble = (sums / count)

ensemble = add_inds(ensemble, 'index')

score_multi(ensemble)
save_heatmap(to_matrix(ensemble), f'ensemble_{YEAR}')
lst = sorted(os.listdir(PNG_DIR))

len(lst)
for i, f in enumerate(lst[:50], 1):

    (score, sr1, tag) = re.findall(r'([\d\.]+)_([\d\.]+)_(.+)\.png$', f)[0]

    name = tag.replace('_', ' ')

    display(HTML(

        f'<h1 id="{tag}">[#{i}] {name}</h1>'

        f'<p>Score: {score} <br/>'

        f'R1 Score: {sr1}'

    ))

    display(Image(f'{PNG_DIR}/{f}'))
lst = np.asarray(os.listdir(PNG_DIR))



np.random.seed(42)

np.random.shuffle(lst)



def read_row(names):

    return np.hstack([plt.imread(os.path.join(PNG_DIR, n)) for n in names])



def read_grid(names2d):

    return np.vstack([read_row(row) for row in names2d])



def pop_gallery(shape):

    global lst

    n = np.product(shape)

    g = read_grid(lst[:n].reshape(shape))

    lst = lst[n:]

    return g



def show_gallery(gal):

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(gal, interpolation='bilinear')

    ax.axis('off')

    plt.tight_layout(pad=0)

    plt.show()
show_gallery(pop_gallery((2,2)))
show_gallery(pop_gallery((3,3)))
show_gallery(pop_gallery((4,4)))
show_gallery(pop_gallery((5,5)))
show_gallery(pop_gallery((6,6)))
show_gallery(pop_gallery((7,7)))
show_gallery(pop_gallery((8,8)))
!7z a -bd -mmt4 {OUTPUT_ZIP} {PNG_DIR}/*.png