# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from scipy.stats import normaltest

import holoviews as hv

from holoviews import opts

hv.extension('bokeh')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.neighbors import KNeighborsRegressor 

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
class football:

    """  **football** is the class for exploratory data 

        analysis and machine learning in data player . 

        This class have 8 attributes that are given important:



    - multi_categorical_plot



    - distplot_multi



    - boxplot_multi



    - learner_selection



    - training_evaluate

    """

    

    def __init__(self, data=None, cols=None, name='potential'):

        

        self.name = name # target

        self.data = data # feature

        self.cols = cols # feature columns name

        self.listof_model = {'LinearRegression': LinearRegression(), 

                'KNeighborsRegression':KNeighborsRegressor(),

                'RandomForestRegression': RandomForestRegressor(),

               'GradientBoostingRegression': GradientBoostingRegressor(),

                'XGBoostRegression': XGBRegressor(),

                'adaboost':AdaBoostRegressor()} # list of different learner

        

    #Read csv file

    def read(self, file):

        return pd.read_csv(file,sep=';', index_col='player_id')

    

    def multi_categorical_plot(self, data):

    

        """ plot a categorical feature

        

            data: float64 array  n_observationxn_feature

        

        """

        # Find a feature that type is object

        string = []

        for i in data.columns:

            if data[i].dtypes == "object":

                string.append(i)

    

        fig = plt.figure(figsize=(20,20))

        fig.subplots_adjust(wspace=0.4, hspace = 0.3)

        for i in range(1,len(string)+1):

            ax = fig.add_subplot(3,1,i)

            sns.countplot(y=string[i-1], data=data, ax=ax)

            ax.set_title(f" {string[i-1]} countplot")

            

    def distplot_multi(self, data):

        """ plot multi distplot"""

    

        

        from scipy.stats import norm

        cols = []

        

        #Feature that is int64 or float64 type 

        for i in data.columns:

            if data[i].dtypes == "float64" or data[i].dtypes == 'int64':

                cols.append(i)

        

        gp = plt.figure(figsize=(20,20))

        gp.subplots_adjust(wspace=0.4, hspace=0.4)

        for i in range(1, len(cols)+1):

            ax = gp.add_subplot(2,3,i)

            sns.distplot(data[cols[i-1]], fit=norm, kde=False)

            ax.set_title('{} max. likelihood gaussian'.format(cols[i-1]))

            

    def boxplot_multi(self, data):

        

        """ plot multi box plot

            hue for plotting categorical data

        """

    

        cols = []

        for i in data.columns:

            if data[i].dtypes == "float64" or data[i].dtypes == 'int64':

                cols.append(i)

    

        gp = plt.figure(figsize=(20,20))

        gp.subplots_adjust(wspace=0.4, hspace=0.4)

        for i in range(1, len(cols)+1):

            ax = gp.add_subplot(2,2,i)

            sns.boxplot(x = cols[i-1], data=data)

            ax.set_title('Boxplot for {}'.format(cols[i-1]))

            

    def correlation_plot(self, data, vrs= 'price'):

    

        """

        This function plot only a variable that are correlated with a target  

        

            data: array m_observation x n_feature

            vrs:  target feature (n_observation, )

            cols: interested features

        """

        

        cols = []

        for i in data.columns:

            if data[i].dtypes == "float64" or data[i].dtypes == 'int64':

                cols.append(i)

                

        feat = list(set(cols) - set([vrs]))

    

        fig = plt.figure(figsize=(15,10))

        fig.subplots_adjust(wspace = 0.3, hspace = 0.25)

        for i in range(1,len(feat)+1):

        

            gp = data.groupby(feat[i-1]).agg('mean').reset_index()

        

            if len(feat) < 3:

                ax = fig.add_subplot(1,3,i)

            else:

                n = len(feat)//2 + 1

                ax = fig.add_subplot(2,n,i)

            

            ax.scatter(data[feat[i-1]], data[vrs], alpha=.25)

            ax.plot(gp[feat[i-1]], gp[vrs], 'r-', label='mean',  linewidth=1.5)

            ax.set_xlabel(feat[i-1])

            ax.set_ylabel(vrs)

            ax.set_title('Plotting data {0} vs {1}'.format(vrs, feat[i-1]))

            ax.legend(loc='best')

            

    def split_data(self):

        """

        This function splits data to train set and target set

        

        data: matrix feature n_observation x n_feature dimension

        name: target  (n_observation, )

        cols: interested feature

        

        return xtrain, xtest, ytrain, ytest

        """

    

        train = self.data[self.cols]

        target = self.data[self.name]

    

        return train_test_split(train, target, random_state=42, test_size=0.2, shuffle=True)

    

    def learner_selection(self):



        """

            This function compute differents score measure like cross validation,

            r2, root mean squared error and mean absolute error.

            listof_model: dictionary type containing different model algorithm.     

        """ 

    

        result = {}

        

        x, _, y, _ = self.split_data() # take only xtrain and ytrain

    

        for cm in list(self.listof_model.items()):

        

            name = cm[0]

            model = cm[1]

        

            cvs = cross_val_score(model, x, y, cv=10).mean()

            ypred = cross_val_predict(model, x, y, cv=10)

            r2 = r2_score(y, ypred)

            mse = mean_squared_error(y, ypred)

            mae = mean_absolute_error(y, ypred)

            rmse = np.sqrt(mse)

        

            result[name] = {'cross_val_score': cvs, 'rmse': rmse, 'mae': mae, 'r2': r2}

        

            print('{} model done !!!'.format(name))

        

        

        return pd.DataFrame(result)

    

    def training_evaluate(self, algorithm):

        

        """This function train and evaluate our model to find r2, rmse and mae"""

        

        result = {}

        xtrain, xtest, ytrain, ytest = self.split_data()

        

        learner = algorithm # learner selected in model_selection function

        

        model = learner.fit(xtrain, ytrain)

        ypred = model.predict(xtest)

        

        r2 = learner.score(xtest, ytest)

        rmse =  np.sqrt(mean_squared_error(ytest, ypred))

        mae = mean_absolute_error(ytest, ypred)

        

        result['potential'] = {'r2':round(r2, 3),  'rmse':round(rmse, 3), 'mae':round(mae, 3)}

        

        return  pd.DataFrame(result)
file ='/kaggle/input/fifa-2021-complete-player-data/FIFA-21 Complete.csv'
fifa = football()
data = fifa.read(file)
data.head()
data.info()
# find categorical feature

data['nationality'].value_counts()
plt.figure(dpi = 200, figsize=(10,20))

sns.countplot(y='nationality', data=data)

plt.show()
data.describe()
data.corr()
plt.figure(dpi=100, figsize=(15,5))

sns.regplot(x='overall', y='potential', data=data)

plt.xlabel('overall')

plt.ylabel('potential')

plt.title('relation between overall and potential')
fifa.distplot_multi(data)
fifa.boxplot_multi(data)
def top_team(team=None, data=None, n=6):

    """

        This function give a top team that are similar

    

    """

    #compute cosine similarity

    def cosine_similarity(a, b): 

        return a.dot(b.T)/(np.linalg.norm(a, 2)*np.linalg.norm (b, 2))

    

    df = {}

    

    data = data.groupby('team')[cols].agg('mean').reset_index()

    

    pyers = data[data.team == team][cols].values[0] #interested team

    

    all_team = list(set(data.team.values) - set(team)) 

    

    for u in all_team:

        

        xv = cosine_similarity(pyers, data[data.team == u][cols].values[0])

        tn = data[data.team == u].team.values[0]

    

        df[u] = {team: round(xv, 3), 'team':tn}

        

    xd =  pd.DataFrame(df).sort_values(by=team, axis=1, ascending=False)



    return xd.T[:n]
def mostSimilar(player = None, data = None, club = None,  n = 10):

    """

        This function give a player that are most similar with another player with similar team.

    

    """

    

    def cosine_similarity(a, b):

        

        return a.dot(b.T)/(np.linalg.norm(a, 2)*np.linalg.norm (b, 2))

    

    df = {}

    

    tm = club.team.values

    

    data = data[data.team.isin(tm)]

    

    pyers = data[data.name == player][cols].values[0] # interested team

    all_name = list(set(data.name.values) - set(player))

    

    for u in all_name:

        

        xv = cosine_similarity(pyers, data[data.name == u][cols].values[0])

        tn = data[data.name == u].team.values[0]

        pn = data[data.name == u].position.values[0]

        nt = data[data.name == u].nationality.values[0]

        

        df[u] = {player:round(xv, 3), 'team':tn, 'position': pn, 'nationality':nt}     

    

    xd =  pd.DataFrame(df).drop(columns=player).sort_values(by=player, axis=1, ascending=False)   

        

    return xd.iloc[:, :n]
cols = [cols  for cols in data.columns if data[cols].dtype != 'object']
cols
team = list(data.team.unique())
team[:10] # 10 teams
%%time

barcelone = top_team(team=team[0], data=data)
barcelone # team that are similar with FC Barcelone
juve = top_team(team[1], data=data)
juve
psg =  top_team(team[2], data=data)
psg
dteam = data[data.team == team[2]]
dteam = dteam.sort_values(by='potential', ascending=False)
dteam.head()
mostSimilar(player=dteam.name.iloc[0], data=data, club=psg).T # the players that are similar with Mbappé
league = data.pivot_table(index='team', columns='nationality', values='potential').reset_index()
league.tail()
corr = league.corr()
plt.figure(dpi=200, figsize=(10,10))

sns.heatmap(corr)

plt.show()
def topMatch(name=None, data=None, n=10):

    df = data[name].drop(index=name).sort_values(ascending=False)[:n]

    return df
def getRecommendation(team=None, data = league):

    """

        This function give a recommendation for different nationalities to team

    """

    

    bara = data[data.team == team] # take interested team

    

    #take items(nationality) that team have not seen

    ncol = list(bara.isnull().sum()[bara.isnull().sum()>0].index)

    

    # take items that team have seen

    col_taken = list(bara.isnull().sum()[bara.isnull().sum()==0].index) 

    

    # remove team not neccesary

    col_taken = list(set(col_taken) - set(['team'])) 

    

    #take correlation matrix for col_taken

    C = corr[corr.index.isin(col_taken)]

    

    mc = C[ncol] # take also correlation matrix for ncol

    mc = mc[mc>0] # positive coef. correlation

    

    # sum all coef. corr for each unknown nationality by interested team

    total = pd.DataFrame(mc.sum(axis=0), columns=['sum'])

    

    #take columns in total that have not null coef. corr.

    ntotal = total[total['sum'] > 0].T 

    

    # take data that match for product matrix

    cm = mc[ntotal.columns] # cm have not null value

    

    #take potential data

    potential = bara[col_taken]

    

    #sort index for potential index

    potential = potential.sort_index(axis=1)  

    

    #sort index for cm index

    cm = cm.sort_index() 

    

    # compute the weighted matrix for nationality that have not seen by interested team

    result = {}

    for u in cm.columns:

    

        cls = cm[u].dropna().index # remove a nan value after taking a index.

    

        result[u] = np.dot(potential[cls].values, cm[u].dropna().values)[0] # 

        

    

    # compute recommendation nationality for team

    recommendation = {}

    for u in result.keys():

        recommendation[u] = round(result[u] / ntotal[u].values[0], 3)

        

    print('Recommended players for {}:'.format(team))

        

    return pd.DataFrame({team: recommendation}).sort_values(by=team, ascending=False)[:10]
rc = getRecommendation(team='Manchester City ') 
recommended_players = data[data.nationality.isin(rc.index)]
recommended_players.sort_values(by='potential', ascending=False)[:10]
fifa21 = football(data=data, cols=['age', 'overall', 'hits'])
fifa21.learner_selection()
fifa21.training_evaluate(XGBRegressor())