# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

FOLDER = "../input/"
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import linear_model, ensemble, neighbors
import numpy as np
import sympy as sp
import pandas as pd
from pylab import rcParams
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import f_regression

plt.rcParams['figure.figsize'] = [20, 10]
%matplotlib inline
dataset = pd.read_csv(FOLDER+'train.csv')
dataset.head()
dataset.mean()
dataset.std()/dataset.mean()
# Ref: 
# https://datascience.stackexchange.com/questions/10459/calculation-and-visualization-of-correlation-matrix-with-pandas
# https://matplotlib.org/examples/color/colormaps_reference.html

def correlation_matrix(df):
 
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('seismic', 30)

    #label = df.columns[0]
    #series = df[label]
    #series = series.apply(lambda x: -x)
  
    #df = pd.concat([df, series], axis=1)

    data = df.corr()
    data.iloc[0][0] = -1
    cax = ax1.imshow(data, interpolation = "nearest", cmap=cmap)
    # ax1.grid(True)
    plt.title('Correlation matix')
    labels = df.columns
    # ax1.set_xticklabels(labels,fontsize=8, rotation=-45)
    # ax1.set_yticklabels(labels,fontsize=8)

    fig.colorbar(cax)
    plt.tight_layout()
    plt.show()

labels = dataset.columns
correlation_matrix(dataset[labels[1:]])
dataset[labels[1:]].corr()
dataset.plot(x='total_bedrooms', y='households', style = 'b.')
dataset.plot(x='median_income', y='median_house_value', style = '.')
geo_plot = dataset[['longitude', 'latitude', 'median_house_value']]
geo_plot = geo_plot.sort_values(by='median_house_value')
t = np.arange(geo_plot.shape[0])

plt.scatter(x=geo_plot['longitude'], y=geo_plot['latitude'], c=t, cmap = cm.RdBu_r, alpha=.2)
# Ref:
# https://gist.github.com/brentp/5355925
# Vários ajustes tiveram que ser feitos ao código contido na referência
# https://www3.nd.edu/~rwilliam/stats1/x91.pdf

from scipy import stats
from sklearn.metrics import r2_score

class MeineLinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def R(self, y, X):
        if X.shape[1] == 0:
            return 0
        
        reg = linear_model.LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        return r2_score(y_pred, y)

    def fit(self, X, y, sample_weight=None):
        self = super().fit(X, y, sample_weight)
        
        y_pred = self.predict(X)
        
        Ryh = r2_score(y, y_pred)
        self.r2 = Ryh
            
        X = np.array(X)
        y = np.array(y)
        
        self.rse = np.sqrt(np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1] - 1))   
             
        self.se = np.array([]) # Standard error for beta_0, not calculated
        
        for i, x in enumerate(X.T):
            n = x - x.mean()
            
            Rxg = self.R(x, np.delete(X, i, 1))
            
            Sbk = np.sqrt( (1 - Ryh)/(1 - Rxg)/(X.shape[0] - X.shape[1] - 1) )*np.std(y)/np.std(x)
            
            self.se = np.append(self.se, Sbk)
        
        self.t = self.coef_ / self.se
        self.p = [2 * (1 - stats.t.cdf(np.abs(t_stu), df = y.shape[0] - X.shape[1])) for t_stu in self.t]
        #return self
baseline = dataset.copy()
target = 'median_house_value'

result = []

np.seterr(all='ignore')

for label in baseline.columns[1:-1]:
    X = baseline[label]
    y = baseline[target]
    
    reg = MeineLinearRegression()
    reg.fit(X.values.reshape(-1, 1), y)
    
    F, p = f_regression(X.values.reshape(-1, 1), y)
    
    result.append({'Label': label, 
                   'Beta': '{0:.2f}'.format(reg.coef_[0]),
                   't':'{0:.2f}'.format(reg.t[0]), 
                   'p-value':'{0:.6f}'.format(p[0]), 
                   'R²':"{0:.2f}%".format(reg.r2 * 100),
                   'F-value':'{0:.2f}'.format(F[0]) })
    
pd.DataFrame(result, columns=['Label', 'Beta', 'R²', 't', 'F-value', 'p-value'])
X = baseline[baseline.columns[1:-1]]
y = baseline[target]
    
reg = MeineLinearRegression()
reg.fit(X, y)

result_2 = []

for n, label in enumerate(X.columns):
    
    result_2.append({'Label': label, 
                     'Beta': '{0:.2f}'.format(reg.coef_[n]),
                     't':'{0:.2f}'.format(reg.t[n]), 
                     'p-value':'{0:.6f}'.format(reg.p[n])})

print("R²: {0:.2f}%".format(reg.r2 * 100))
pd.DataFrame(result_2)
test_1 = dataset[dataset['median_house_value'] < 500000]

target = 'median_house_value'

X = test_1[test_1.columns[1:-1]]
y = test_1[target]
    
evaluation = linear_model.LinearRegression()
score = cross_val_score(evaluation, X, y, cv = 10, scoring='r2')
print('R² = {0:.3f} ± {1:.3f}'.format(np.mean(score), 2*np.std(score)))

reg = MeineLinearRegression()
reg.fit(X, y)

result_2 = []

for n, label in enumerate(X.columns):
    
    result_2.append({'Label': label, 
                     'Beta': '{0:.2f}'.format(reg.coef_[n]),
                     't':'{0:.2f}'.format(reg.t[n]), 
                     'p-value':'{0:.6f}'.format(reg.p[n])})

pd.DataFrame(result_2)
test_1.plot(x='median_income', y='median_house_value', style = '.')
non_lin = dataset.copy()
stat = []
target = 'median_house_value'

for label_1 in non_lin.columns[3:-1]:
    
    for label_2 in non_lin.columns[3:-1]:

        if label_2 != label_1:
            
            X = non_lin[label_1]/non_lin[label_2]

            reg = linear_model.LinearRegression()
            reg.fit(X.values.reshape(-1, 1), non_lin[target])

            F, p = f_regression(X.values.reshape(-1, 1), non_lin[target])
            y = reg.predict(X.values.reshape(-1, 1))

            r2 = r2_score(non_lin[target], y)
            
            stat.append({'Label': label_1+'/'+label_2, 
                           'Beta': '{0:.2f}'.format(reg.coef_[0]),
                           'F':'{0:.0f}'.format(F[0]), 
                           'p-value':'{0:.6f}'.format(p[0]), 
                           'R²':"{0:.2f}%".format(r2 * 100), 
                           'corr':'{0:.5f}'.format( np.corrcoef(X,non_lin[target])[0][1] ),
                           'raw_R²':r2})
            
output = pd.DataFrame(stat, columns = ['Label', 'Beta', 'R²', 'p-value', 'F', 'corr', 'raw_R²'] )
output = output[output['raw_R²'] > 0.01 ]

output.sort_values(by='raw_R²', ascending=False)
def add_features(dset):
    
    mean_houses = pd.Series(dset['households']/dset['population'], name = 'mean_households')
    rooms_ratio = pd.Series(dset['total_rooms']/dset['total_bedrooms'], name = 'ratio' )
    
    return pd.concat([mean_houses, rooms_ratio, dset], axis = 1)
stat = []
target = 'median_house_value'

for n, label_1 in enumerate(non_lin.columns[3:-1]):
    
    for label_2 in non_lin.columns[3+n:-1]:
    
        X = non_lin[label_1]*non_lin[label_2]

        reg = linear_model.LinearRegression()
        reg.fit(X.values.reshape(-1, 1), non_lin[target])

        F, p = f_regression(X.values.reshape(-1, 1), non_lin[target])
        y = reg.predict(X.values.reshape(-1, 1))

        r2 = r2_score(non_lin[target], y)

        stat.append({'Label': label_1+'*'+label_2, 
                       'Beta': '{0:.5f}'.format(reg.coef_[0]),
                       't':'{0:.2f}'.format(np.sqrt(F[0])), 
                       'p-value':'{0:.6f}'.format(p[0]), 
                       'R²':"{0:.2f}%".format(r2 * 100),
                       'corr':'{0:.5f}'.format( np.corrcoef(X,non_lin[target])[0][1] ),
                       'raw_R²':r2})
            
output = pd.DataFrame(stat, columns = ['Label', 'Beta', 'R²', 'p-value', 't', 'corr', 'raw_R²'] )
output = output[output['raw_R²'] > 0.01 ]

output.sort_values(by='raw_R²', ascending=False)
from geopy.distance import distance

def get_distance(point):
    '''
    Calcula a distância de um ponto até as 3 metrópoles da Califórnia,
    San Francisco, San Diego e Los Angeles
    '''
    CITIES = [( 37 + 46/60 + 46/3600, -(122 + 25/60 +  9/3600) ),  # San Francisco
              ( 32 + 46/60 + 46/3600, -(117 +  8/60 + 47/3600) ),  # San Diego
              ( 34 +  3/60 + 14/3600, -(118 + 14/60 + 42/3600) )]  # Los Angeles
    
    d = []
    p = (point['latitude'], point['longitude'])
    
    for city in CITIES:
        d.append(distance(city, p).km)
        
    return d

def add_dist(dset):
    '''
    Adiciona uma coluna correspondente à menor distância até uma das 3 metrópoles
    '''
    coords = dset[['longitude', 'latitude']]
    raw_dists = np.array( [get_distance(coords.loc[i]) for i in range(coords.shape[0]) ])
    dists = pd.DataFrame( [np.min(l_dist) for l_dist in raw_dists], columns=['least_dist'] )
    
    return pd.concat([dists, dset], axis = 1)
test = pd.read_csv(FOLDER+'test.csv')
test.head()
def submit_pred(dset, dtest, model = linear_model.LinearRegression(), name = 'submit'):

    target = 'median_house_value'
    labels = dset.columns.drop('Id').drop(target)

    X = dset[labels]
    y = dset[target]
    
    score = cross_val_score(model, X, y, cv = 10, scoring='r2')
    print('R² = {0:.3f} ± {1:.3f}'.format(np.mean(score), 2*np.std(score)))
    
    model.fit(X, y)
    
    y_test_raw = model.predict(dtest[labels])

    y_test = [np.max([entry, 0]) for entry in y_test_raw]    # Evitar velores negativos

    output = pd.concat( [test['Id'], pd.Series(y_test, name=target)], axis=1)
    output.to_csv('./{0}.csv'.format(name), index=False)
submit_pred(dataset, test, name = 'raw_linear_model')
submit_pred(add_features(dataset), add_features(test), name = 'improv_linear_model')
d_dset = add_dist(add_features(dataset))
d_test = add_dist(add_features(test))
submit_pred(d_dset,
            d_test, 
            name = 'linear_model_dists_gps')
submit_pred(d_dset.drop('latitude', axis = 1).drop('longitude', axis = 1),
            d_test.drop('latitude', axis = 1).drop('longitude', axis = 1), 
            name = 'linear_model_dists')
submit_pred(d_dset.drop('latitude', axis = 1),
            d_test.drop('latitude', axis = 1),
            model = linear_model.LassoLars(alpha=.1),
            name = 'linear_model_lasso')
submit_pred(d_dset.drop('latitude', axis = 1).drop('longitude', axis = 1),
            d_test.drop('latitude', axis = 1).drop('longitude', axis = 1),
            model = linear_model.LassoLars(alpha=.1),
            name = 'linear_model_lasso')
mean = []
stds = []

K = 1
M = 60

for i in range(K, M, 3):
    knn = neighbors.KNeighborsRegressor(n_neighbors = i)
    score = cross_val_score(knn, d_dset.drop(target, axis = 1), d_dset[target], scoring = 'r2', cv = 10)
    mean.append( np.mean(score) )
    stds.append( np.std(score) )
    
table = pd.DataFrame()
table['Mean'] = mean
table['Std'] = stds
table
plt.plot(range(K, M, 3), mean)
plt.xlabel('K neghibors')
plt.ylabel('R² score')
plt.show()
submit_pred(d_dset,
            d_test,
            model = linear_model.BayesianRidge(),
            name = 'linear_model_bayes_gps')
submit_pred(d_dset.drop('latitude', axis = 1),
            d_test.drop('latitude', axis = 1),
            model = linear_model.BayesianRidge(),
            name = 'linear_model_bayes')
submit_pred(d_dset,
            d_test,
            model = ensemble.RandomForestRegressor(n_estimators = 100),
            name = 'random_forest_gps')
submit_pred(d_dset.drop('latitude', axis = 1),
            d_test.drop('latitude', axis = 1),
            model = ensemble.RandomForestRegressor(n_estimators = 100),
            name = 'random_forest')
submit_pred(d_dset,
            d_test,
            model = ensemble.ExtraTreesRegressor(n_estimators = 100),
            name = 'extra_trees_gps')
submit_pred(d_dset.drop('latitude', axis = 1),
            d_test.drop('latitude', axis = 1),
            model = ensemble.ExtraTreesRegressor(n_estimators = 100),
            name = 'extra_trees')
submit_pred(d_dset,
            d_test,
            model = ensemble.AdaBoostRegressor(base_estimator = 
                            ensemble.RandomForestRegressor(max_depth=20, n_estimators=10), n_estimators=50),
            name = 'ada_boost_gps')
