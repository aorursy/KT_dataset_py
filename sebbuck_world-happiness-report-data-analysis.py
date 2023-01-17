import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.datasets import load_boston

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.graphics.gofplots import ProbPlot

from patsy import dmatrices
df = pd.read_csv('../input/world-happiness-report-2019.csv')

df.head()
df.shape
df.dtypes
plt.figure(1, figsize = (30, 40))

n = 0

for x in ['SD of Ladder', 'Positive affect', 'Negative affect', 'Social support', 'Freedom', 'Corruption', 'Generosity', 

          'Log of GDP\nper capita', 'Healthy life\nexpectancy']:

    n += 1

    plt.subplot(3, 3, n)

    plt.subplots_adjust(hspace = 0.2, wspace = 0.4)

    plt.scatter(df['Ladder'], df[x])

    plt.title('{} plot'.format(x))

plt.show()
df.isnull().sum()

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna()
plt.style.use('fivethirtyeight')
plt.figure(1, figsize = (30, 40))

n = 0

for x in ['SD of Ladder', 'Positive affect', 'Negative affect', 'Social support', 'Freedom', 'Corruption', 'Generosity', 

          'Log of GDP\nper capita', 'Healthy life\nexpectancy']:

    n += 1

    plt.subplot(3, 3, n)

    plt.subplots_adjust(hspace = 0.2, wspace = 0.4)

    sns.regplot(df['Ladder'], df[x])

    plt.title('{} plot'.format(x))

plt.show()
Xa = pd.DataFrame(df, columns = ['SD of Ladder'])

Xb = pd.DataFrame(df, columns = ['Positive affect'])

Xc = pd.DataFrame(df, columns = ['Negative affect'])

Xd = pd.DataFrame(df, columns = ['Social support'])

Xe = pd.DataFrame(df, columns = ['Freedom'])

Xf = pd.DataFrame(df, columns = ['Corruption'])

Xg = pd.DataFrame(df, columns = ['Generosity'])

Xh = pd.DataFrame(df, columns = ['Log of GDP\nper capita'])

Xi = pd.DataFrame(df, columns = ['Healthy life\nexpectancy'])

y = pd.DataFrame(df.Ladder)



model = sm.OLS(y, sm.add_constant(Xi))

model_fit = model.fit()

dataframe = pd.concat([Xi, y], axis = 1)
yhat = model_fit.fittedvalues

residuals = model_fit.resid

normal_residuals = model_fit.get_influence().resid_studentized_internal

abs_sqrt_residuals = np.sqrt(np.abs(normal_residuals))

model_abs_residuals = np.abs(residuals)

leverage = model_fit.get_influence().hat_matrix_diag

cooks_dis = model_fit.get_influence().cooks_distance[0]
plot_lm_1 = plt.figure(1, figsize = (15, 15))

plot_lm_1.axes[0] = sns.residplot(yhat, dataframe.columns[-1], data = dataframe, lowess = True, scatter_kws = {'alpha': 0.8},

                                 line_kws = {'color': 'red', 'lw': 1, 'alpha': 0.8})



n = 0

for x in ['SD of Ladder', 'Positive affect', 'Negative affect', 'Social support', 'Freedom', 'Corruption', 'Generosity', 

          'Log of GDP\nper capita', 'Healthy life\nexpectancy']:

    n += 1

    model = sm.OLS(y, sm.add_constant(pd.DataFrame(df, columns = [x])))

    model_fit = model.fit()

    dataframe = pd.concat([df[x], y], axis = 1)

    yhat = model_fit.fittedvalues

    residuals = model_fit.resid

    normal_residuals = model_fit.get_influence().resid_studentized_internal

    abs_sqrt_residuals = np.sqrt(np.abs(normal_residuals))

    model_abs_residuals = np.abs(residuals)

    leverage = model_fit.get_influence().hat_matrix_diag

    cooks_dis = model_fit.get_influence().cooks_distance[0]

    plt.subplot(3, 3, n)

    plt.subplots_adjust(hspace = 0.5, wspace = 1)

    sns.residplot(yhat, dataframe.columns[-1], data = dataframe, lowess = True, scatter_kws = {'alpha': 0.8},

                                 line_kws = {'color': 'red', 'lw': 1, 'alpha': 0.8})

    plt.title('{}: yhat vs residuals'.format(x))

    plt.xlabel('yhat')

    plt.ylabel('residuals')

plt.show()
plot_lm_1 = plt.figure(1, figsize = (15, 15))

plot_lm_1.axes[0] = sns.residplot(yhat, dataframe.columns[-1], data = dataframe, lowess = True, scatter_kws = {'alpha': 0.8},

                                 line_kws = {'color': 'red', 'lw': 1, 'alpha': 0.8})



n = 0

for x in ['SD of Ladder', 'Positive affect', 'Negative affect', 'Social support', 'Freedom', 'Corruption', 'Generosity', 

          'Log of GDP\nper capita', 'Healthy life\nexpectancy']:

    n += 1

    model = sm.OLS(y, sm.add_constant(pd.DataFrame(df, columns = [x])))

    model_fit = model.fit()

    dataframe = pd.concat([df[x], y], axis = 1)

    yhat = model_fit.fittedvalues

    residuals = model_fit.resid

    normal_residuals = model_fit.get_influence().resid_studentized_internal

    abs_sqrt_residuals = np.sqrt(np.abs(normal_residuals))

    model_abs_residuals = np.abs(residuals)

    leverage = model_fit.get_influence().hat_matrix_diag

    cooks_dis = model_fit.get_influence().cooks_distance[0]

    plt.subplot(3, 3, n)

    plt.subplots_adjust(hspace = 0.5, wspace = 1)

    

    plt.scatter(yhat, abs_sqrt_residuals, alpha = 0.5)

    sns.regplot(yhat, abs_sqrt_residuals, scatter = False, ci = False, lowess = True, 

                line_kws = {'color': 'red', 'lw':1, 'alpha': 0.8})

    plt.title('{}: yhat vs sqrt resids'.format(x))

    plt.xlabel('yhat')

    plt.ylabel('$\sqrt{Standardized Residuals}$')

plt.show()
plot_lm_4 = plt.figure(1, figsize = (15, 15))



n = 0

for x in ['SD of Ladder', 'Positive affect', 'Negative affect', 'Social support', 'Freedom', 'Corruption', 'Generosity', 

          'Log of GDP\nper capita', 'Healthy life\nexpectancy']:

    n += 1

    model = sm.OLS(y, sm.add_constant(pd.DataFrame(df, columns = [x])))

    model_fit = model.fit()

    dataframe = pd.concat([df[x], y], axis = 1)

    yhat = model_fit.fittedvalues

    residuals = model_fit.resid

    normal_residuals = model_fit.get_influence().resid_studentized_internal

    abs_sqrt_residuals = np.sqrt(np.abs(normal_residuals))

    model_abs_residuals = np.abs(residuals)

    leverage = model_fit.get_influence().hat_matrix_diag

    cooks_dis = model_fit.get_influence().cooks_distance[0]

    plt.subplot(3, 3, n)

    plt.subplots_adjust(hspace = 0.5, wspace = 1)

    

    plt.scatter(leverage, normal_residuals, alpha = 0.5)

    sns.regplot(leverage, normal_residuals, scatter = False, ci = False, lowess = True, line_kws = {'color': 'red', 'lw': 1,

                                                                                               'alpha': 0.8})

    

    plt.title('{}: Resids vs Levrg'.format(x))

    plt.xlabel('Leverage')

    plt.ylabel('standard residuals')

plt.show()
X1 = df[['Social support', 'Ladder']].iloc[:, :].values

X2 = df[['Freedom', 'Ladder']].iloc[:, :].values

X3 = df[['Corruption', 'Ladder']].iloc[:, :].values

X4 = df[['Generosity', 'Ladder']].iloc[:, :].values

X5 = df[['Log of GDP\nper capita', 'Ladder']].iloc[:, :].values

X6 = df[['Healthy life\nexpectancy', 'Ladder']].iloc[:, :].values

algorithm = (KMeans(n_clusters = 4, init = 'k-means++', n_init = 10, max_iter = 300,

            tol = 0.0001, random_state = 111, algorithm = 'elkan')

            )

algorithm.fit(X1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_
algorithm = (KMeans(n_clusters = 3, init='k-means++', n_init = 10 , max_iter=300, 

                        tol=0.0001,  random_state= 111, algorithm='elkan') )

algorithm.fit(X1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_
h = 0.02

x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1

y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
plt.figure(1 , figsize = (15 , 7) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'Ladder' ,y = 'Social support' , data = df , c = labels1 , 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Social support') , plt.xlabel('Ladder')

plt.show()
m = df[['Positive affect', 'Negative affect', 'Social support', 'Freedom', 'Corruption', 'Generosity',

        'Log of GDP\nper capita', 'Healthy life\nexpectancy']]

y = df['Ladder']

model = sm.OLS(y,m).fit()

predictions = model.predict(m)

model.summary()
n = df[['Positive affect', 'Negative affect', 'Social support', 'Freedom', 'Corruption',

        'Log of GDP\nper capita', 'Healthy life\nexpectancy']]

y = df['Ladder']

model = sm.OLS(y,n).fit()

predictions = model.predict(n)

model.summary()
o = df[['Positive affect', 'Negative affect', 'Social support', 'Freedom',

        'Log of GDP\nper capita', 'Healthy life\nexpectancy']]

y = df['Ladder']

model = sm.OLS(y,o).fit()

predictions = model.predict(o)

model.summary()
df_cor = o.corr()

pd.DataFrame(np.linalg.inv(o.corr().values), index = df_cor.index, columns=df_cor.columns)
p = df[['Positive affect', 'Negative affect', 'Social support', 'Freedom', 'Healthy life\nexpectancy']]

y = df['Ladder']

model = sm.OLS(y, p).fit()

predictions = model.predict(p)

model.summary()
q = df[['Positive affect', 'Negative affect', 'Social support', 'Healthy life\nexpectancy']]

y = df['Ladder']

model = sm.OLS(y, q).fit()

predictions = model.predict(q)

model.summary()
r = df[['Positive affect', 'Social support', 'Healthy life\nexpectancy']]

y = df['Ladder']

model = sm.OLS(y, r).fit()

predictions = model.predict(r)

model.summary()
df_cor = r.corr()

pd.DataFrame(np.linalg.inv(r.corr().values), index = df_cor.index, columns=df_cor.columns)
plt.figure(1, figsize = (20, 10))

n = 0

for x in ['Positive affect', 'Social support', 'Healthy life\nexpectancy']:

    n += 1

    plt.subplot(1, 3, n)

    plt.subplots_adjust(hspace = 1, wspace = 0.4)

    sns.regplot(df['Ladder'], df[x])

    plt.title('{} plot'.format(x))

plt.show()