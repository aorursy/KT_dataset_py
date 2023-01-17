import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

warnings.warn("deprecated", DeprecationWarning)

pd.options.mode.chained_assignment = None



train = pd.read_csv('../input/train.csv')

if train['id'].nunique() == train.shape[0]:

    print("The unique id for every row so that 'id' varaible go into index")

    train.set_index('id', inplace = True)



train.rename(columns = {'number_of_total_atoms': 'tot_Atom', 'percent_atom_al':'pr_al', 'percent_atom_ga':'pr_ga',

                       'percent_atom_in' : 'pr_in', 'lattice_vector_1_ang' : 'l1',

                       'lattice_vector_2_ang' : 'l2', 'lattice_vector_3_ang' : 'l3',

                       'lattice_angle_alpha_degree' : 'alpha', 'lattice_angle_beta_degree' : 'beta',

                       'lattice_angle_gamma_degree' : 'gamma', 'formation_energy_ev_natom' : 'stability',

                       'bandgap_energy_ev' : 'power'}, inplace = True)

print('Rename Column :', train.columns)
import scipy.stats

from collections import OrderedDict



g = sns.JointGrid(x =  'stability', y ='power', data = train, size = 7, ratio = 3)

g = g.plot_joint(sns.regplot, color = 'g', scatter_kws = {'alpha' : 0.3})

g = g.plot_marginals(sns.distplot, kde = True, color = 'r')

rsquare = lambda a, b: scipy.stats.pearsonr(a, b)[0] ** 2

g = g.annotate(rsquare, template="{stat}: {val:.2f}", stat="$R^2$", loc="upper right", fontsize=12)

#plt.setp(g.ax_marg_x.get_yticklabels(), visible=True)

#plt.setp(g.ax_marg_y.get_xticklabels(), visible=True)

#plt.annote()

plt.show()

train[['stability', 'power']].apply(lambda r: pd.Series(OrderedDict({'mean': r.mean(), 'std' : r.std(), 'kurt' : r.kurtosis(),

                                                                    'normP-val' : scipy.stats.mstats.normaltest(r)[1]}))).T
f, ax = plt.subplots(1,3, figsize = (12,4))

train.iloc[:, :2].plot.hist(subplots = True, sharex = False, ax = [ax[0], ax[1]])

cntControl = train.groupby(['spacegroup', 'tot_Atom']).size().reset_index()

ax[2].scatter(x = cntControl.loc[:,'spacegroup'], y = cntControl.loc[:,'tot_Atom'], s = 5*cntControl.iloc[:,0], 

              color = sns.color_palette('Set2', cntControl.shape[0]))

ax[2].set_ylabel('tot_Atom')

ax[2].set_xlabel('spacegroup')

ax[2].set_title('The Size Circle Plot for the Control Variable', fontsize = 12)

plt.subplots_adjust(wspace = 0.3)

plt.suptitle('Constrained Variable Distribution', fontsize = 14)

plt.show()



train['group'] = '0'

groupOrder = []

for num, ((ix, ix2), group) in enumerate(train.groupby(['spacegroup', 'tot_Atom'])):

    groupname = 'sp' + str(ix) + '_' + 'at' + str(ix2)

    train.loc[(train['spacegroup'] == ix) & (train['tot_Atom'] == ix2),'group'] = groupname

    groupOrder += groupname,

print('groupping is done')
import pandas.tools.plotting

f, ax = plt.subplots(1,1, figsize = (12,6))

#colors = pandas.tools.plotting._get_standard_colors(10, color_type = 'random')

colors = np.random.uniform(0,1, size = (10,3))

groups = train.groupby('group')

ax.set_color_cycle(colors)

for name, group in groups:

    ax.plot(group['stability'], group['power'], marker = 'o', linestyle = '', label = name, alpha = 0.7)

ax.legend(loc = 'upper right')

plt.xlabel('stability')

plt.ylabel('power')

plt.title('Scatter Plot Per Group', fontsize = 14)

plt.show()

g = sns.FacetGrid(train[['group', 'stability', 'power']], col = 'group', col_order = groupOrder, col_wrap = 4)

g = g.map(plt.scatter, "stability", "power", alpha = 0.7)

plt.show()
from pandas.tools.plotting import parallel_coordinates



sa34 = train.loc[train['group'] == 'sp33_at40.0',:]

sa38 = train.loc[train['group'] == 'sp33_at80.0',:]

part_sa34 = sa34.loc[(sa34['stability'] < 0.07) & (sa34['power'] > 2), :]

otherVar = ['pr_al', 'pr_ga', 'pr_in', 'l1', 'l2', 'l3', 'Name']



f, ax = plt.subplots(1,2, figsize = (12,4))

ax[0].scatter(x = sa34['stability'], y = sa34['power'], c = 'r', alpha = 0.8, label = 'sa34')

ax[0].scatter(x = sa38['stability'], y = sa38['power'], c = 'b', alpha = 0.8, label = 'sa38')

ax[0].axvline(x = 0.07, color = 'g', linewidth = 1)

ax[0].axhline(y = 2, color = 'g', linewidth = 1)

X  = np.linspace(0,0.07,100)

Y, Y1 = 2, 5

ax[0].legend(loc = 'upper right')

ax[0].fill_between(X, Y, Y1, color = 'green', alpha = 0.3)

ax[0].set_title('The Unique Area Filled by Green')



part_sa34['Name'] = 'Green Area'

sa38['Name'] = 'The Left Area for Blue Points'

data = pd.concat([part_sa34, sa38])

data = data.loc[:, otherVar]

ax[1] = parallel_coordinates(data, 'Name')

ax[1].legend(loc = 'upper left')

ax[1].set_title('Varialbe Distribution w.r.t Area')

plt.suptitle('Relationship between sp33_at40 & sp33_at80', fontsize = 15)

plt.subplots_adjust(0, 0, 1, 0.8)

plt.show()



print('sa34 appearance : ', sa34.shape[0])

print('sa38 appearance : ',  sa38.shape[0])

print('Green Area of Red Dot Appearance : ', part_sa34.shape[0])
sa11 = train.loc[train['group'] == 'sp194_at10.0',:]

sa11['Name'] = np.where(sa11['stability'] > 0.5, '2nd', '1st')

sa28 = train.loc[train['group'] == 'sp206_at80.0',:]

sa28['Name'] = np.where(sa28['stability'] > 0.3,'outlier', 'inner')



f, ax = plt.subplots(1,2, figsize = (12,3))

sa11.plot.scatter(x = 'stability', y = 'power', ax = ax[0])

ax[0].axvline(x = 0.5, color = 'g', linewidth = 1)

ax[0].set_title('Sa11 Scatter')

ax[1] = parallel_coordinates(sa11.loc[:, otherVar], 'Name')

ax[1].set_title('Sa11 Cluster Variable Distribution')

plt.suptitle("Sa11 & Sa28 Outlier Detection", fontsize = 14)

plt.show()



f, ax = plt.subplots(1,2, figsize = (12,3))

sa28.plot.scatter(x = 'stability', y = 'power', ax = ax[0])

ax[0].set_title('Sa28 Scatter')

ax[0].axvline(x = 0.3, color = 'g', linewidth = 1)

ax[1] = parallel_coordinates(sa28.loc[:, otherVar], 'Name')

ax[1].set_title('Sa28 Cluster Variable Distribution')



plt.show()
def getVal(x):

    if x < 10: return x/100

    else: return x/1000

f, ax = plt.subplots(1,3, figsize = (12,4), subplot_kw={'projection': 'polar'})

angle = train[['alpha', 'beta', 'gamma']].copy()

colors = plt.cm.Dark2(np.random.uniform(0,1,3))

color = plt.cm.Pastel1(0.1)

for ix, col in enumerate(angle.columns):

    test = angle[col].value_counts()

    pi = (test.index / 180) * np.pi

    pi_val = test.values

    bars = ax[ix].bar(pi, pi_val, width = list(map(getVal, pi_val)), bottom = 0, color = colors[ix])

    ax[ix].set_rmax(20)

    ax[ix].set_rlabel_position(-22.5)

    ax[ix].set_facecolor(color)

    ax[ix].set_title(col + ' Density', fontsize = 12)

plt.show()