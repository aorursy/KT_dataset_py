# General

import pandas as pd

import numpy as np

from tqdm.notebook import tqdm

import random, pickle



# Visualisation

from mpl_toolkits.mplot3d.axes3d import Axes3D

import matplotlib.pyplot as plt

from matplotlib import cm

import seaborn as sns

from matplotlib import animation



# Algos

housing_datapath = '/kaggle/input/boston-dataset/boston_data.csv'

housing_testdatapath = '/kaggle/input/boston-dataset/boston_test_data.csv'



target_variable = 'medv'
housing_data = pd.read_csv(housing_datapath, index_col=0).reset_index()

housing_testdata = pd.read_csv(housing_testdatapath, index_col=0)
plt.rcParams['figure.dpi'] = 180

plt.rcParams['figure.figsize'] = (5,5)



class BootstrapData:

    

    def __init__(self):

        print('Initialising Testing Class...')

    

    def create_bootstrappeddata(self, df, targetcol, newsize, _features, std_scaling=0.1, mode='ot'):



        df = df[_features].dropna().reset_index().drop('index', axis=1)



        bootstrappeddata = pd.DataFrame()



        for _ in range(newsize):

            temp = df.sample(1)

            if mode == 'ot':

                topertub = [targetcol]

            elif mode == 'of':

                topertub = [k for k in _features if k!=targetcol]

            elif mode == 'all':

                topertub = _features



            for eachFeat in topertub:



                _std = df[eachFeat].dropna().std()*std_scaling

                origval = temp[eachFeat].values[0]

                possiblevals = np.linspace(origval-_std, origval+_std, 20).tolist()

                temp[eachFeat] = random.sample(possiblevals, k=1)

                bootstrappeddata = pd.concat([bootstrappeddata, temp])



        return bootstrappeddata



bdHandler = BootstrapData()

var_x = 'lstat'

var_y = 'indus'

var_z = target_variable

pltData = housing_data[[var_x, var_y, var_z]].dropna()

_mode = 'of'

datadict = {}



for eachmode in ['ot', 'of', 'all']:

    datadict[eachmode] = {}

    for eachstd in tqdm(np.linspace(0, 0.5, 101)):

        

        datadict[eachmode][eachstd] = bdHandler.create_bootstrappeddata(df = pltData, targetcol = target_variable,newsize = 1000,

                                                                        _features = pltData.columns.tolist(), std_scaling = eachstd,

                                                                        mode=eachmode)



df = pd.DataFrame()

for eachP1key in tqdm(datadict.keys()):

    for eachP2Key in datadict[eachP1key].keys():

        tempdf = datadict[eachP1key][eachP2Key].copy()

        tempdf['mode'] = eachP1key

        tempdf['stdscalig'] = eachP2Key

        df = pd.concat([df, tempdf])

df.to_csv(prepared_datapath+'prep_vizdata.csv')

unique_col_set = df.groupby(['mode', 'stdscalig']).size().index.tolist()
%matplotlib notebook

modedecode = {'ot':'Only Target', 'of':'Only Features', 'all':'Features & Target'}

def update_graph(num):

    _select = list(unique_col_set[num])

    data=df[df[['mode', 'stdscalig']].isin(_select).product(axis=1) == 1]

    f1corr = np.round(data[[var_x, var_y, var_z]].corr()[[var_z]].iloc[0].values[0], 3)

    f2corr = np.round(data[[var_x, var_y, var_z]].corr()[[var_z]].iloc[1].values[0], 3)

    graph.set_edgecolor('k')

    graph.set_color('white')

    graph._offsets3d = (data.lstat, data.indus, data.medv)

    title.set_text('Data Perturbation, Mode: {0}, \nStdScaling: {1}; \nF1 Corr : {2}, F1 Corr : {3}\n\n'.format(modedecode[_select[0]], 

                                                                                               np.round(_select[1],3),

                                                                                               f1corr,

                                                                                               f2corr))

    

    if _select[0] == 'all':

        if num > 15:

            ax2.view_init(elev=10, azim=num-35)

        else:

            ax2.view_init(elev=10, azim=0)

    elif _select[0] == 'of':

        if num > 115:

            ax2.view_init(elev=10, azim=num-135)

        else:

            ax2.view_init(elev=10, azim=0)

    elif _select[0] == 'ot':

        if num > 215:

            ax2.view_init(elev=10, azim=num-235)

        else:

            ax2.view_init(elev=10, azim=0)

    

    

fig = plt.figure(figsize=plt.figaspect(0.5), facecolor='lightgray')



data = pltData.copy()

ax1 = fig.add_subplot(1, 2, 1, projection='3d', facecolor='lightgray')

# ax1.view_init(elev=10, azim=0)

ax1.scatter(data.lstat, data.indus, data.medv, c='white', edgecolor='k')

ax1.set_title('Actual\n')

ax1.set_xlabel('Feature 1', rotation=90)

ax1.set_ylabel('Feature 2', rotation=90)

ax1.set_zlabel('Target Variable', rotation=90)

ax1.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))



ax2 = fig.add_subplot(1, 2, 2, projection='3d', facecolor='lightgray')

ax2.set_title(f'BootStrapped - Mode : all')     

data=df[df['mode']=='all'].copy()

data=data[data['stdscalig']==0.0]

graph = ax2.scatter(data.lstat, data.indus, data.medv, c='white', edgecolor='k')

title = ax2.set_title('Data Perturbation, Mode: 0')

ax2.set_xlabel('Feature 1', rotation=90)

ax2.set_ylabel('Feature 2', rotation=90)

ax2.set_zlabel('Target Variable', rotation=90)

ax2.view_init(elev=10, azim=0)

ani = animation.FuncAnimation(fig, update_graph, interval=500, blit=False,

                              repeat=True, save_count=303)

plt.show()
FFwriter = animation.FFMpegWriter(fps=3, codec="libx264")     

ani.save('basic_animation1.mp4', writer = FFwriter )