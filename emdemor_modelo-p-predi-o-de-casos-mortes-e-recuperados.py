import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings





from datetime import datetime,timedelta

from scipy.integrate import odeint



!pip install xtlearn

from xtlearn.feature_selection import FeatureSelector

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score

from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.metrics import r2_score,mean_squared_log_error,mean_absolute_error



from sklearn.base import BaseEstimator,TransformerMixin
warnings.filterwarnings("ignore")



# Setting seaborn style

sns.set_style("darkgrid")

colors = ["#449353","#ff9999","#4f7cac", "#80e4ed","#f8f99e","#b5ddbd"]

sns.set_palette(sns.color_palette(colors))

plt.rcParams["figure.figsize"] = [8,5]



# Setting Pandas float format

pd.options.display.float_format = '{:,.1f}'.format



SEED = 42   #The Answer to the Ultimate Question of Life, The Universe, and Everything

np.random.seed(SEED)
class RollingMean(BaseEstimator,TransformerMixin):

    '''

    Description

    ----------

    Provide rolling window calculations.

   

    Arguments

    ----------

    window: int

        Size of the moving window. This is the number of observations used for calculating the statistic.

        

    min_periods: int, default None

        Minimum number of observations in window required to have a value 

        (otherwise result is NA). For a window that is specified by an offset, 

        min_periods will default to 1. Otherwise, min_periods will default 

        to the size of the window.

        

    center: bool, default False

        Set the labels at the center of the window.

        



    active: boolean

        This parameter controls if the selection will occour. This is useful in hyperparameters searchs to test the contribution

        in the final score

        

    '''

    

    def __init__(self,window,

                 min_periods = None,

                 center = False,

                 active=True,

                 columns = 'all'

                ):

        self.columns = columns

        self.active = active

        self.window = window

        self.min_periods = min_periods

        self.center = center



        

    def fit(self,X,y=None):

        return self

        

    def transform(self,X):

        if not self.active:

            return X

        else:

            return self.__transformation(X)



    def __transformation(self,X_in):

        X = X_in.copy()

        

        if type(self.columns) == str:

            if self.columns == 'all':

                self.columns = list(X.columns)

        

        for col in self.columns:  

            X[col] = X[col].fillna(0).rolling(window = self.window,

                                              min_periods = self.min_periods,

                                              center = self.center

                                             ).mean()

        return X.dropna()

        

    def inverse_transform(self,X):

        return X
class ApplyLog1p(BaseEstimator,TransformerMixin):

    '''

    Description

    ----------

    Apply numpy.log1p to specified features.

   

    Arguments

    ----------

        

    columns: list, default False

        Column names to apply numpy.log1p.

        



    active: boolean

        This parameter controls if the selection will occour. This is useful in hyperparameters searchs to test the contribution

        in the final score

        

    '''

    

    def __init__(self,active=True,columns = 'all'):

        self.columns = columns

        self.active = active

        

    def fit(self,X,y=None):

        return self

        

    def transform(self,X):

        if not self.active:

            return X

        else:

            return self.__transformation(X)



    def __transformation(self,X_in):

        X = X_in.copy()

        

        if type(self.columns) == str:

            if self.columns == 'all':

                self.columns = list(X.columns)

        

        for col in self.columns:  

            X[col] = np.log1p(X[col])

            

        return X

        

    def inverse_transform(self,X):

        if not self.active:

            return X

        else:

            return self.__inverse_transformation(X)



    def __inverse_transformation(self,X_in):

        X = X_in.copy()

        

        if type(self.columns) == str:

            if self.columns == 'all':

                self.columns = list(X.columns)

        

        for col in self.columns:  

            X[col] = np.expm1(X[col])

            

        return X
dataset = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19_macro.csv',parse_dates=['date']).drop(columns=['monitoring'])

first_day = dataset.iloc[0]['date']

dataset['days'] = (dataset['date']-first_day).dt.days



dataset['change_cases'] = (dataset['cases']-dataset['cases'].shift())

dataset['change_deaths'] = (dataset['deaths']-dataset['deaths'].shift())

dataset['change_recovered'] = (dataset['recovered']-dataset['recovered'].shift())

plt.scatter(dataset['date'],dataset['cases'],linewidth=1,s=5,label='Casos Confirmados')

plt.scatter(dataset['date'],dataset['deaths'],linewidth=1,s=5, label='Óbitos')

plt.scatter(dataset['date'],dataset['recovered'],linewidth=1,s=5, label='Casos Recuperados')

plt.xticks(rotation=45)

plt.title('Número de Casos de Covid-19')

plt.xlabel('Data')

plt.ylabel('Milhões de Casos')

plt.legend()

plt.show()
plt.plot(dataset['date'],dataset['change_cases']/1000,linewidth=1,label='Casos Diarios',alpha = 0.4,color = colors[0])

plt.plot(dataset['date'],0.001*dataset['change_cases'].rolling(window = 14,center=True).mean(),label='Média Móvel (14 dias)',color = colors[0])



plt.xticks(rotation=45)

plt.title('Casos Diários de Covid-19')

plt.xlabel('Data')

plt.ylabel('Milhares de Casos')

plt.legend(loc = 'upper left')

plt.show()
plt.plot(dataset['date'],dataset['change_deaths'],linewidth=1,label='Óbitos Diários',color = colors[1],alpha=0.4)

plt.plot(dataset['date'],dataset['change_deaths'].rolling(window = 14,center=True).mean(),label='Média Móvel (14 dias)',color = colors[1])





plt.xticks(rotation=45)

plt.title('Óbitos Diários por Covid-19')

plt.xlabel('Data')

plt.ylabel('Milhares de Casos')

plt.legend(loc = 'upper left')

plt.show()
plt.plot(dataset['date'],dataset['change_recovered'],linewidth=1,label='Recuperações Diárias',color = colors[2],alpha=0.4)

plt.plot(dataset['date'],dataset['change_recovered'].rolling(window = 14,center=True).mean(),label='Média Móvel (14 dias)',color = colors[2])





plt.xticks(rotation=45)

plt.title('Óbitos Diários por Covid-19')

plt.xlabel('Data')

plt.ylabel('Milhares de Casos')

plt.legend()

plt.show()
# Pipeline for preprocessing

preproc = Pipeline(steps = [

    ('rolling_mean',RollingMean(window = 21,columns = [

        'cases', 'deaths', 'recovered',

        'change_cases','change_deaths','change_recovered'],center = True)),

    

    ('select',FeatureSelector(features = ['days','cases', 'deaths',

        'recovered','change_cases','change_deaths','change_recovered'])

    ),

])



# Log Scalling

log_apply = ApplyLog1p(columns = 'all')
# Applying pre-processing pipeline

df = log_apply.transform(preproc.transform(dataset))



# full dataset

X = df[['days','cases', 'deaths', 'recovered']]

yc = df['change_cases']

yd = df['change_deaths']

yr = df['change_recovered']



train_size = 0.85

index_split = int(round(train_size*len(X)))



# training dataset

X_trn  = X.iloc[:index_split]

yc_trn = yc.iloc[:index_split]

yd_trn = yd.iloc[:index_split]

yr_trn = yr.iloc[:index_split]



#test dataset

X_tst  = X.iloc[index_split:]

yc_tst = yc.iloc[index_split:]

yd_tst = yd.iloc[index_split:]

yr_tst = yr.iloc[index_split:]
# Pipeline for regression

regression_c = Pipeline(steps = [

    ('polinomial',PolynomialFeatures(degree = 1)),

    ('regressor',LinearRegression()),

])

regression_d = Pipeline(steps = [

    ('polinomial',PolynomialFeatures(degree = 1)),

    ('regressor',LinearRegression()),

])

regression_r = Pipeline(steps = [

    ('polinomial',PolynomialFeatures(degree = 1)),

    ('regressor',LinearRegression()),

])
regression_c.fit(X_trn,yc_trn)

print('M.A.E. of cases (train)= %.4f'%mean_absolute_error(yc,regression_c.predict(X)))

print('M.A.E. of cases (test) = %.4f'%mean_absolute_error(yc_tst,regression_c.predict(X_tst)))



regression_d.fit(X_trn,yd_trn)

print('\nM.A.E. of deaths (train)= %.4f'%mean_absolute_error(yd,regression_d.predict(X)))

print('M.A.E. of deaths (test)= %.4f'%mean_absolute_error(yd_tst,regression_d.predict(X_tst)))



regression_r.fit(X_trn,yr_trn)

print('\nM.A.E. of recovered (train)= %.4f'%mean_absolute_error(yr,regression_r.predict(X)))

print('M.A.E. of deaths (test)= %.4f'%mean_absolute_error(yd_tst,regression_d.predict(X_tst)))
predictions = log_apply.inverse_transform(pd.concat([

    X.reset_index(drop=True),

    pd.DataFrame(regression_c.predict(X),columns = ['change_cases']),

    pd.DataFrame(regression_d.predict(X),columns = ['change_deaths']),

    pd.DataFrame(regression_r.predict(X),columns = ['change_recovered']),

],1))
plt.scatter(dataset['days'],dataset['change_cases']/1000,s=5,label='Casos Diarios',alpha = 0.3,color = colors[0])

plt.plot(predictions['days'],0.001*predictions['change_cases'].rolling(window = 14,center=True).mean(),

         linewidth=2,label='Regressão',color = colors[0])



# plt.xticks(rotation=45)

plt.title('Modelo para Casos Diários de Covid-19')

plt.xlabel('Dias após Primeiro Caso')

plt.ylabel('Milhares de Casos')

plt.legend(loc = 'upper left')

plt.show()
plt.scatter(dataset['days'],dataset['change_deaths']/1000,s=5,label='Óbitos Diarios',alpha = 0.3,color = colors[1])

plt.plot(predictions['days'],0.001*predictions['change_deaths'].rolling(window = 14,center=True).mean(),

         linewidth=2,label='Regressão',color = colors[1])



# plt.xticks(rotation=45)

plt.title('Modelo para Óbitos Diários de Covid-19')

plt.xlabel('Dias após Primeiro Caso')

plt.ylabel('Milhares de Casos')

plt.legend(loc = 'upper left')

plt.show()
plt.figure(figsize=(8,5))

plt.scatter(dataset['days'],dataset['change_recovered']/1000,s=5,label='Recuperações Diárias',alpha = 0.3,color = colors[2])

plt.plot(predictions['days'],0.001*predictions['change_recovered'].rolling(window = 14,center=True).mean(),

         linewidth=2,label='Regressão',color = colors[2])



# plt.xticks(rotation=45)

plt.title('Modelo para Recuperações Diárias de Covid-19')

plt.xlabel('Dias após Primeiro Caso')

plt.ylabel('Milhares de Casos')

plt.legend(loc = 'upper left')

plt.show()
regression_c.fit(X,yc)

regression_d.fit(X,yd)

regression_r.fit(X,yr)



predictions = log_apply.inverse_transform(pd.concat([

    X.reset_index(drop=True),

    pd.DataFrame(regression_c.predict(X),columns = ['change_cases']),

    pd.DataFrame(regression_d.predict(X),columns = ['change_deaths']),

    pd.DataFrame(regression_r.predict(X),columns = ['change_recovered']),

],1))
def diff_eq(x,t):

    """

    Function resturning the differential equations of the model



    """

    # setting the functions

    c,r,d = x

    lnt = np.log1p(t)

    lnx = np.log1p(x)

    

    

    # mathematical equations

    DiffC = np.expm1(regression_c.predict([[lnt]+list(lnx)]))[0]

    DiffD = np.expm1(regression_d.predict([[lnt]+list(lnx)]))[0]

    DiffR = np.expm1(regression_r.predict([[lnt]+list(lnx)]))[0]



    return np.array([DiffC,DiffD,DiffR])



def neg_diff_eq(x,t):

    return -diff_eq(x,-t)
# defining the limits

t_min = 20

t_max   = 500

n_points = 500



# initial conditions

t0,*x0 = np.expm1(list(X.iloc[-50]))



# counting the points

n_points_right = int(round(n_points*(t_max-t0) / (t_max-t_min)))

n_points_left = int(round(n_points*(t0-t_min) / (t_max-t_min)))



# right integrate

days_list = np.linspace(t0,t_max,n_points_right)

x = odeint(diff_eq,x0,days_list)



# left integrate

neg_days_list = np.linspace(-t0,-t_min,n_points_left)

neg_x = odeint(neg_diff_eq,x0,neg_days_list)



#joinning solution

t_full = np.concatenate((-neg_days_list[::-1], days_list))

x_full = np.concatenate((neg_x[::-1], x))



print('Total de mortes: %d' % x[-1,1])
plt.scatter(dataset['days'],0.000001*dataset['cases'],marker='.',s=80,alpha=0.3,color = colors[0],label='Casos Confirmados')

plt.plot(t_full,0.000001*x_full[:,0],color='black',linestyle='dashed',linewidth=1.3,label='Modelo')



plt.title('Modelo para Casos Confirmados de Covid-19')

plt.xlabel('Dias após Primeiro Caso')

plt.ylabel('Milhões de Casos')

plt.legend(loc = 'upper left')

plt.show()
plt.scatter(dataset['days'],0.001*dataset['deaths'],marker='.',s=80,alpha=0.3,color = colors[1],label='Óbitos')

plt.plot(t_full,0.001*x_full[:,1],color='black',linestyle='dashed',linewidth=1.3,label='Modelo')



plt.title('Modelo Óbitos por Covid-19')

plt.xlabel('Dias após Primeiro Caso')

plt.ylabel('Milhares de Casos')

plt.legend(loc = 'upper left')

plt.show()
plt.scatter(dataset['days'],0.000001*dataset['recovered'],marker='.',s=80,alpha=0.3,color = colors[2],label='Recuperações')

plt.plot(t_full,0.000001*x_full[:,2],color='black',linestyle='dashed',linewidth=1.3,label='Modelo')



plt.title('Modelo para Casos Recuperados de Covid-19')

plt.xlabel('Dias após Primeiro Caso')

plt.ylabel('Milhões de Casos')

plt.legend(loc = 'upper left')

plt.show()
class Covid19Regressor(BaseEstimator,TransformerMixin):

    '''

    Description

    ----------

    Arguments

    ----------

    active: boolean

        This parameter controls if the selection will occour. This is useful in hyperparameters searchs to test the contribution

        in the final score

        

    '''

    

    def __init__(self,

                 confirmed = 'cases', 

                 deaths = 'deaths',

                 recovered = 'recovered',

                 

                 confirmed_rate = 'change_cases', 

                 deaths_rate = 'change_deaths',

                 recovered_rate = 'change_recovered',

                 

                 time = 'days',

                 window = 7,

                 min_periods = None,

                 center = True,

                 polynomial_degree = 2,

                 regressor = LinearRegression,

                 regressor_parameters = {},

                 t_initial = 'last',

                 t_min = 20,

                 t_max = 500,

                 n_points = 500

                 

                ):

        

        self.confirmed = confirmed

        self.confirmed_rate = confirmed_rate

        self.deaths = deaths

        self.deaths_rate = deaths_rate

        self.recovered = recovered

        self.recovered_rate = recovered_rate

        self.time = time

        self.window = window

        self.min_periods = min_periods

        self.center = center

        self.polynomial_degree = polynomial_degree

        self.regressor = regressor

        self.regressor_parameters = regressor_parameters

        self.t_initial = t_initial

        self.t_min = t_min

        self.t_max = t_max

        self.n_points = 1+n_points

        

        

    def fit(self,X,y):

        

        # Receiving the data

        self.X = X[[self.time,self.confirmed,self.deaths,self.recovered]].copy()

        self.y = y[[self.confirmed_rate,self.deaths_rate,self.recovered_rate]].copy()

        

        # Evaluating the rolling mean for X

        for col in [self.confirmed,self.deaths,self.recovered]:  

            self.X[col] = self.X[col].fillna(0).rolling(window = self.window,

                                              min_periods = self.min_periods,

                                              center = self.center

                                             ).mean()

            

        # Evaluating the rolling mean for y    

        for col in [self.confirmed_rate,self.deaths_rate,self.recovered_rate]:  

            self.y[col] = self.y[col].fillna(0).rolling(window = self.window,

                                              min_periods = self.min_periods,

                                              center = self.center

                                             ).mean()

            

        # Applying the log scale

        self.X[self.time] = np.log1p(self.X[self.time])



        for col in [self.confirmed,self.deaths,self.recovered]: 

            self.X[col] = np.log1p(self.X[col])

            

        for col in [self.confirmed_rate,self.deaths_rate,self.recovered_rate]: 

            self.y[col] = np.log1p(self.y[col])

        

        # Dropping NaN

        temp = pd.concat([self.X,self.y],1).dropna()

        self.X = temp[[self.time,self.confirmed,self.deaths,self.recovered]]

        self.y = temp[[self.confirmed_rate,self.deaths_rate,self.recovered_rate]]

            

        # Pipeline for regression

        regression_c = Pipeline(steps = [

            ('polinomial',PolynomialFeatures(degree = self.polynomial_degree)),

            ('regressor',self.regressor(**self.regressor_parameters)),

        ])

        regression_d = Pipeline(steps = [

            ('polinomial',PolynomialFeatures(degree = self.polynomial_degree)),

            ('regressor',self.regressor(**self.regressor_parameters)),

        ])

        regression_r = Pipeline(steps = [

            ('polinomial',PolynomialFeatures(degree = self.polynomial_degree)),

            ('regressor',self.regressor(**self.regressor_parameters)),

        ])

        

        # Fitting model

        regression_c.fit(self.X,self.y[self.confirmed_rate])

        regression_d.fit(self.X,self.y[self.deaths_rate])

        regression_r.fit(self.X,self.y[self.recovered_rate])

        

        

        # Predicted Rates

        self.predicted_rate = pd.concat([

            self.X.reset_index(drop=True),

            pd.DataFrame(regression_c.predict(self.X),columns = ['pred_'+self.confirmed_rate]),

            pd.DataFrame(regression_d.predict(self.X),columns = ['pred_'+self.deaths_rate]),

            pd.DataFrame(regression_r.predict(self.X),columns = ['pred_'+self.recovered_rate]),

        ],1)

        

        for col in self.predicted_rate.columns:

            self.predicted_rate[col] = np.expm1(self.predicted_rate[col])

        

        

        # Defining the diferential equations

        def diff_eq(x,t):

            """

            Function resturning the differential equations of the model



            """

            # setting the functions

            c,r,d = x

            lnt = np.log1p(t)

            lnx = np.log1p(x)





            # mathematical equations

            DiffC = np.expm1(regression_c.predict([[lnt]+list(lnx)]))[0]

            DiffD = np.expm1(regression_d.predict([[lnt]+list(lnx)]))[0]

            DiffR = np.expm1(regression_r.predict([[lnt]+list(lnx)]))[0]



            return np.array([DiffC,DiffD,DiffR])



        def neg_diff_eq(x,t):

            return -diff_eq(x,-t)

        

        if type(self.t_initial) == str:

            if self.t_initial == 'last':

                t_initial = int(round(list(cov19.predicted_rate[self.time])[-1]))

        else:

            t_initial = self.t_initial

        

        

        ind_ref = self.predicted_rate[self.time][

            round(self.predicted_rate[self.time]).astype(int) == t_initial].index[0]

        

        # initial conditions

        t0,*x0 = np.expm1(list(self.X.iloc[ind_ref]))



        n_points_right = int(round(self.n_points*(self.t_max-t0) / (self.t_max-self.t_min)))

        n_points_left = int(round(self.n_points*(t0-self.t_min) / (self.t_max-self.t_min)))





        # right integrate

        days_list = np.linspace(t0,self.t_max,n_points_right)

        x = odeint(diff_eq,x0,days_list)



        # left integrate

        neg_days_list = np.linspace(-t0,-self.t_min,n_points_left)

        neg_x = odeint(neg_diff_eq,x0,neg_days_list)



        #joinning solution

        self.t_ode = np.concatenate((-neg_days_list[::-1], days_list))

        self.x_ode = np.concatenate((neg_x[::-1], x))

        

        self.predictions = pd.concat([

            pd.DataFrame(self.t_ode,columns = [self.time]),

            pd.DataFrame(self.x_ode,columns = [self.confirmed,self.deaths,self.recovered])]

        ,1)



        return self

        

    def transform(self,X):

        return X

    

    def predict(self,X):

        

       

        return np.array([

            np.interp(X, self.t_ode, self.x_ode[:,0]),

            np.interp(X, self.t_ode, self.x_ode[:,1]),

            np.interp(X, self.t_ode, self.x_ode[:,2]),

        ])

        
cov19 = Covid19Regressor(window = 21,polynomial_degree = 1,t_initial = 'last')

cov19.fit(dataset[['days','cases','deaths','recovered']],

    dataset[['change_cases','change_deaths','change_recovered']])



t_list = np.arange(20,500,1)

x_list = cov19.predict(t_list)



plt.figure(figsize=(8,5))

plt.scatter(dataset['days'],0.000001*dataset['cases'],marker='.',s=80,alpha=0.3,color = colors[0],label='Casos Confirmados')

plt.plot(t_list,0.000001*x_list[0],color=colors[0],linestyle='dashed',linewidth=1.3,label='Modelo - Casos')



plt.scatter(dataset['days'],0.000001*dataset['deaths'],marker='.',s=80,alpha=0.3,color = colors[1],label='Óbitos')

plt.plot(t_list,0.000001*x_list[1],color=colors[1],linestyle='dashed',linewidth=1.3,label='Modelo - Óbitos')



plt.scatter(dataset['days'],0.000001*dataset['recovered'],marker='.',s=80,alpha=0.3,color = colors[2],label='Recuperações')

plt.plot(t_list,0.000001*x_list[2],color=colors[2],linestyle='dashed',linewidth=1.3,label='Modelo - Recuperações')





plt.title('Modelo para Covid-19')

plt.xlabel('Dias após Primeiro Caso')

plt.ylabel('Milhões de Casos')

# plt.legend(loc = 'lower right')

plt.legend(bbox_to_anchor=(0.52, 0.17), loc=3, borderaxespad=0.)

plt.show()
from tqdm import tqdm

delta = 10

new_list = []

for final_index in tqdm(range(100,224,5)):

    X_train = dataset.fillna(0).iloc[:final_index][['days','cases','deaths','recovered']]

    y_train = dataset.fillna(0).iloc[:final_index][['change_cases','change_deaths','change_recovered']]

    X_test = dataset.fillna(0).iloc[final_index:][['days','cases','deaths','recovered']]

#     X_test = dataset.iloc[final_index:final_index+delta][['days','cases','deaths','recovered']]



    cov19 = Covid19Regressor(window = 14,polynomial_degree = 1)

    cov19.fit(X_train,y_train)



    pred_tst = cov19.predict(X_test['days'])

    

    

    new_list.append([100*final_index/(len(dataset)),

        mean_absolute_error(X_test.iloc[:,1].values,np.nan_to_num(pred_tst[0])),

        mean_absolute_error(X_test.iloc[:,2].values,np.nan_to_num(pred_tst[1])),

        mean_absolute_error(X_test.iloc[:,3].values,np.nan_to_num(pred_tst[2]))

    ])

    

np.array(new_list)[:,1].mean()
plt.plot(np.array(new_list)[:,0],np.array(new_list)[:,1],label = 'Casos Confirmados')

plt.plot(np.array(new_list)[:,0],np.array(new_list)[:,2],label = 'Óbitos')

plt.plot(np.array(new_list)[:,0],np.array(new_list)[:,3],label = 'Casos Recuperados')



plt.title('Erro Absoluto do Conjunto de Teste')

plt.xlabel('Tamanho do Conjunto de Treino (%)')

plt.ylabel('M.A.E. para o Conjunto de Teste')

plt.yscale("log")

plt.legend(loc = 'upper right')

plt.show()