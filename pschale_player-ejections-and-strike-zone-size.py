import pandas as pd

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from scipy.optimize import minimize

from itertools import permutations



from time import time



from tqdm import tqdm



%matplotlib inline

sns.set(color_codes=True)
pitches = pd.read_csv('../input/pitches.csv')

atbats = pd.read_csv('../input/atbats.csv', index_col='ab_id')

ej = pd.read_csv('../input/ejections.csv')

games = pd.read_csv('../input/games.csv', index_col='g_id')
balls_strikes_ejs = ej[~(ej['BS'].isna())].groupby('g_id').head(n=1)

balls_strikes_ejs = balls_strikes_ejs.set_index('g_id')

balls_strikes_ejs.rename(columns={'event_num': 'ej_event_num', 

                                  'ab_id': 'ej_ab_id'}, inplace=True)

balls_strikes_ejs.head()



pitches.dropna(inplace=True)

called_pitches = pitches[pitches.code.isin(['C', 'B'])].copy()

called_pitches['k'] = (called_pitches.code=='C').astype(int)

called_pitches = called_pitches[(called_pitches.k == 0) | 

                                ((np.abs(called_pitches.px) < 2) & (called_pitches.pz > 0.2) &

                                 (called_pitches.pz < 4.5))]

called_pitches = called_pitches.join(atbats, on='ab_id')



ej_pitches = called_pitches[['pz', 'px', 'ab_id', 'code', 'k',

                             'event_num', 'top', 'g_id', 

                             'stand', 'p_throws']].join(balls_strikes_ejs, how='right', on='g_id')

ej_pitches['d'] = ej_pitches.ej_event_num - ej_pitches.event_num

ej_pitches['after_ej'] = ej_pitches.d < 0

ej_pitches['ej_team_batting'] = ~(ej_pitches.top == ej_pitches.is_home_team)
ej_pitches.head()
class multi_logistic():

    def __init__(self):

        self.bounds = [1.3, 3.5, -1, 1]

        self.coef = [8, 10, 10, 10]

        self.corner_coef = [1, 1, 1, 1]

        self.trained = False

        

    def predict(self, X, bounds=None, coef=None, corner_coef=None):

        if bounds is None:

            bounds = self.bounds

        if coef is None:

            coef = self.coef

        if corner_coef is None:

            corner_coef = self.corner_coef

        mid_x = (self.bounds[2] + self.bounds[3])/2

        mid_y = (self.bounds[0] + self.bounds[1])/2

        log_odds = [-((bounds[0] - X[:, 1])*coef[0] + ((X[:,0] - mid_x)**2)*corner_coef[0]) , 

                (bounds[1] - X[:, 1])*coef[1] - ((X[:, 0] - mid_x)**2)*corner_coef[1],

                -((bounds[2] - X[:, 0])*coef[2] + (X[:,1] - mid_y)**2)*corner_coef[2],

                (bounds[3] - X[:, 0])*coef[3] - ((X[:,1] - mid_y)**2)*corner_coef[3]]



        odds = [np.exp(l) for l in log_odds]

        probs = [o/(o + 1) for o in odds]

        return probs[0] * probs[1] * probs[2] * probs[3]

    

    def eval(self, X, y, bounds=None, coef=None, corner_coef=None): 



        if bounds is None:

            bounds = self.bounds

        if coef is None:

            coef = self.coef

        if corner_coef is None:

            corner_coef = self.corner_coef



        preds = self.predict(X, bounds=bounds, coef=coef, corner_coef=corner_coef)

        scores = -np.log(1 - np.abs(y - preds))

        return scores

        #return np.minimum(scores, np.ones(scores.shape)*100)

        

    def fit(self, X, y, solver_method='SLSQP'):

        func = lambda p: np.mean(self.eval(X, y, bounds=p[:4], coef=p[4:8], corner_coef=p[8:]))

        results = minimize(func, x0=self.bounds+self.coef+self.corner_coef, 

                           method=solver_method, options={'ftol': 0.0005})

        self.bounds = list(results.x[:4])

        self.coef = list(results.x[4:8])

        self.corner_coef = list(results.x[8:])

        self.trained = True

        

        return self

    

xlims = [-2, 2]

ylims = [1, 4]



def fit_kn(df, n=150):

    mdl = KNeighborsClassifier(n_neighbors=n)

    mdl.fit(df[['px', 'pz']], df.k.astype(int))

    return mdl



def fit_mdl(df, mtype, **kwargs):

    X, y = df[['px', 'pz']].values, df['k'].values

    if mtype == 'knn':

        n = kwargs.get('n', 150)

        return KNeighborsClassifier(n_neighbors=n).fit(X, y)

    elif mtype == 'logistic':

        return multi_logistic().fit(X, y)

    else:

        raise ValueError('mtype not recognized')



def find_zone_stats(mdl, res=0.05):

    xx, yy = np.meshgrid(np.arange(xlims[0], xlims[1], res),

                     np.arange(ylims[0], ylims[1], res))  

    try:

        preds = mdl.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]

    except:

        preds = mdl.predict(np.c_[xx.ravel(), yy.ravel()])

        

    zone_size = preds.sum()*(res**2)*144

    boundary_size = (((-(np.abs(preds - 0.5)) + 0.5)*2)**2).sum()*(res**2)*144

    return zone_size, boundary_size



def plot_zone(mdl, res=0.05):

    xx, yy = np.meshgrid(np.arange(xlims[0], xlims[1], res),

                     np.arange(ylims[0], ylims[1], res))  

    try:

        preds = mdl.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]

    except:

        preds = mdl.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.imshow(preds.reshape(xx.shape), extent=[xlims[0], xlims[1], ylims[0], ylims[1]], cmap='bwr', origin='lower')

    return preds.reshape(xx.shape), [xlims[0], xlims[1], ylims[0], ylims[1]]





def describe_plot_zone(df, res=0.05, n=150, mtype=None):

    if mtype=='knn':

        mdl = fit_kn(df, n=n)

    elif mtype=='logistic':

        mdl = multi_logistic()

        mdl.fit(df[['px', 'pz']].values, df['k'].values)

    img, ext = plot_zone(mdl, res=res)

    zone_size, boundary_size = find_zone_stats(mdl, res=res)

    if mtype=='logistic':

        x = mdl.bounds

        plt.plot([x[2], x[3]], [x[0], x[0]], '-k') 

        plt.plot([x[2], x[3]], [x[1], x[1]], '-k') 

        plt.plot([x[2], x[2]], [x[0], x[1]], '-k') 

        plt.plot([x[3], x[3]], [x[0], x[1]], '-k') 

    plt.show()

    print('Zone is {:6.1f} in^2, with weighted boundary of {:6.1f} in^2'.format(zone_size, boundary_size))

    return img, ext, mdl

    
after_ej_dict = {False: 'Before ejection', True: 'After ejection'}

ej_team_dict = {True: "Team ejected", False: "Other team"}

zs = []

zb = []

managers_team_batting = []

after_ejection = []

zbounds = []

coefs = []

corner_coefs = []

for g in ej_pitches[ej_pitches.stand=='R'].groupby(['after_ej', 'ej_team_batting']):

    print(after_ej_dict[g[0][0]],'\n', ej_team_dict[g[0][1]])

    print('Based on {} pitches'.format(len(g[1])))



    for i in tqdm(range(10000)):

        mdl = fit_mdl(g[1].sample(len(g[1]), replace=True), mtype='logistic', n=int(np.sqrt(len(g[1]))/4))

        zsize, zbound = find_zone_stats(mdl)

        zbounds.append(mdl.bounds)

        coefs.append(mdl.coef)

        corner_coefs.append(mdl.corner_coef)

        zs.append(zsize)

        zb.append(zbound)

        after_ejection.append(after_ej_dict[g[0][0]])

        managers_team_batting.append(ej_team_dict[g[0][1]])

        

zbounds = np.array(zbounds)

logistic_panel = pd.DataFrame(dict(zone_size=zs, zb=zb, after_ejection=after_ejection, 

                                   team_batting=managers_team_batting, zbot=zbounds[:,0],

                                   ztop=zbounds[:,1], zleft=zbounds[:,2], zright=zbounds[:,3]))
g = sns.catplot(data=logistic_panel, x='after_ejection', hue='team_batting', y='zone_size', 

                kind='box', orient='v', showfliers=False)

g.set(xlabel='')

g.set(ylabel=r'Zone Size (in^2)')

g._legend.set_title('Team Batting')

plt.savefig('logistic_boxplot_zonesize.png')
logistic_panel['i'] = np.tile(range(10000), 4)

logistic_panel['combined'] = logistic_panel.apply(lambda x: x.team_batting + ' ' + x.after_ejection, axis=1)

logistic_pivoted_size = logistic_panel.pivot(index='i', columns='combined', values='zone_size')



print('Comparing Zone Sizes')

for p in permutations(list(logistic_pivoted_size.columns), 2):

    data = logistic_pivoted_size[p[1]] - logistic_pivoted_size[p[0]]

    p_val = np.mean(data>0)

    if p_val <= 0.5:

        print(p[0], np.mean(logistic_pivoted_size[p[0]]), 

              p[1], np.mean(logistic_pivoted_size[p[1]]), p_val)
g = sns.catplot(data=logistic_panel, x='after_ejection', hue='team_batting', y='zb', 

                kind='box', orient='v', showfliers=False)

g.set(xlabel='')

g.set(ylabel=r'Zone Size (in^2)')

g._legend.set_title('Team Batting')

plt.savefig('logistic_boxplot_boundary.png')
logistic_pivoted_boundary = logistic_panel.pivot(index='i', columns='combined', values='zb')



print('Comparing Zone Sizes')

for p in permutations(list(logistic_pivoted_boundary.columns), 2):

    data = logistic_pivoted_boundary[p[1]] - logistic_pivoted_boundary[p[0]]

    p_val = np.mean(data>0)

    if p_val <= 0.5:

        print(p[0], np.mean(logistic_pivoted_boundary[p[0]]), 

              p[1], np.mean(logistic_pivoted_boundary[p[1]]), p_val)
after_ej_dict = {False: 'Before ejection', True: 'After ejection'}

ej_team_dict = {True: "Team ejected", False: "Other team"}

zs = []; zb = [];

managers_team_batting = []

after_ejection = []

for g in ej_pitches[ej_pitches.stand=='R'].groupby(['after_ej', 'ej_team_batting']):

    print(after_ej_dict[g[0][0]],'\n', ej_team_dict[g[0][1]])

    print('Based on {} pitches'.format(len(g[1])))



    for i in tqdm(range(10000)):

        mdl = fit_mdl(g[1].sample(len(g[1]), replace=True), mtype='knn', n=int(np.sqrt(len(g[1]))/4))

        zsize, zbound = find_zone_stats(mdl)

        zs.append(zsize)

        zb.append(zbound)

        after_ejection.append(after_ej_dict[g[0][0]])

        managers_team_batting.append(ej_team_dict[g[0][1]])

        

knn_panel = pd.DataFrame(dict(zone_size=zs, zb=zb, after_ejection=after_ejection, 

                              team_batting=managers_team_batting))
g = sns.catplot(data=knn_panel, x='after_ejection', hue='team_batting', y='zone_size', 

                kind='box', orient='v', showfliers=False)

g.set(xlabel='')

g.set(ylabel=r'Zone Size (in^2)')

g._legend.set_title('Team Batting')

plt.savefig('knn_boxplot_zonesize.png')
knn_panel['i'] = np.tile(range(10000), 4)

knn_panel['combined'] = knn_panel.apply(lambda x: x.team_batting + ' ' + x.after_ejection, axis=1)

knn_pivoted_size = knn_panel.pivot(index='i', columns='combined', values='zone_size')



print('Comparing Zone Sizes')

for p in permutations(list(knn_pivoted_size.columns), 2):

    data = knn_pivoted_size[p[1]] - knn_pivoted_size[p[0]]

    p_val = np.mean(data>0)

    if p_val <= 0.5:

        print(p[0], np.mean(knn_pivoted_size[p[0]]), p[1], 

              np.mean(knn_pivoted_size[p[1]]), p_val)
g = sns.catplot(data=knn_panel, x='after_ejection', hue='team_batting', y='zb', 

                kind='box', orient='v', showfliers=False)

g.set(xlabel='')

g.set(ylabel=r'Zone Boundary (in^2)')

g._legend.set_title('Team Batting')

plt.savefig('knn_boxplot_boundary.png')


knn_pivoted_boundary = knn_panel.pivot(index='i', columns='combined', values='zb')



print('Comparing Zone Boundary Sizes')

for p in permutations(list(knn_pivoted_boundary.columns), 2):

    data = knn_pivoted_boundary[p[1]] - knn_pivoted_boundary[p[0]]

    p_val = np.mean(data>0)

    if p_val <= 0.5:

        print(p[0], np.mean(knn_pivoted_boundary[p[0]]), 

              p[1], np.mean(knn_pivoted_boundary[p[1]]), p_val)