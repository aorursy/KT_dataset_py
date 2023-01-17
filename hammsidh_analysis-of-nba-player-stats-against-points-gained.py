import numpy as np

import csv 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats



from sqlalchemy import create_engine

from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import KMeans

from sklearn.cluster import SpectralClustering

from sklearn.mixture import GaussianMixture as GMM



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/nba-players-stats-20142015/players_stats.csv')

df.head()
df.info()
duplicateRowsDF = df[df.duplicated()]

duplicateRowsDF
df.isnull().any()
dfc = df[pd.notnull(df['BMI'])]
dfnull = dfc[dfc['Height'].isnull()]

dfnull
MPG = df['MIN'] / df['Games Played']

TFG = df['FGM'] + df['FGA']
dfl = (dfc[["Name", "Team", "Games Played", "MIN", "EFF", "FGM", "FGA", "PTS"]].sort_values("PTS", ascending=False))

dfl.insert(4, 'PMPG', MPG)

dfl.insert(8, 'TFG', TFG)



dfl.head()
# dfl.isnull().any()
sns.pairplot(dfl)
dfl.plot.scatter("EFF", "PTS", alpha=0.5, color= "r", figsize=(13,5))

plt.xlabel('Effisiensi Bermain')

plt.ylabel('Point')
Boston = dfl.Team == 'BOS'

Detroit = dfl.Team == 'DET'

Cleveland = dfl.Team == 'CLE'

Orlando = dfl.Team == 'ORL'

Toronto = dfl.Team == 'TOR'

Washington = dfl.Team == 'WAS'

Philadelphia = dfl.Team == 'PHI'

Milwaukee = dfl.Team == 'MIL'

NewYork = dfl.Team == 'NYK'

Atlanta = dfl.Team == 'ATL'

Charlote = dfl.Team == 'CHA'

Chicago = dfl.Team == 'CHI'

Indiana = dfl.Team == 'IND'

Miami = dfl.Team == 'MIA'



dfeast = dfl[Boston | Detroit | Cleveland | Orlando | Toronto | Washington | Philadelphia | 

            Milwaukee | NewYork | Atlanta | Charlote | Chicago | Indiana | Miami]



# dfeast.sort_values("Team", ascending=True)
maxt = dfeast.groupby(['Team'])['PTS'].transform(max) == dfeast['PTS']

dfmax = dfeast[maxt]

dfmax.sort_values("EFF", ascending=False)
import plotly.offline as py

import plotly.graph_objs as go



tr1 = go.Bar(

                 x = dfmax['Name'],

                 y = dfmax['EFF'],

                 name = 'Effisiensi',

                 marker = dict(color='crimson',

                              line = dict(color='rgba(0,0,0)', width=0.5)),

                 text = dfmax.Team)



tr2 = go.Bar(

                 x = dfmax['Name'],

                 y = dfmax['PTS'],

                 name = 'Points',

                 marker = dict(color='rgba(0, 0, 255, 0.5)',

                              line = dict(color='rgba(0,0,0)', width=0.5)),

                 text = dfmax.Team)

dn = [tr1, tr2]

layoutnew = go.Layout(barmode='group', title='Effisiensi Tertinggi Pemain pada TIM Regional Timur Terhadap Points')

fig = go.Figure(data=dn, layout=layoutnew)

fig.update_layout(barmode='stack')

py.iplot(fig)
engine= create_engine('sqlite:///:memory:')



dfeast.to_sql('data_table', engine) 

dfeastrt= pd.read_sql_query('SELECT SUM("EFF"), Team FROM data_table group by Team', engine)

K = dfeastrt['SUM("EFF")']
engine= create_engine('sqlite:///:memory:')



dfeast.to_sql('data_table', engine) 

dfeastrp= pd.read_sql_query('SELECT SUM("PTS"), Team FROM data_table group by Team', engine)
dfeastrp.insert(1, 'SEFF', K)

dfeastrp.rename(

    columns={

        'SUM("PTS")': "SPTS"

    },

    inplace=True)
dfeastrp['Rank'] = dfeastrp['SPTS'].rank(method='dense', ascending=False)

dfeastrp.sort_values("SPTS", ascending=False)
fig = go.Figure()

fig.add_trace(go.Scatter(x=dfeastrp['Team'], y=dfeastrp['SEFF'], fill='tozeroy',name = 'Jumlah Effisiensi Tim'))

fig.add_trace(go.Scatter(x=dfeastrp['Team'], y=dfeastrp['SPTS'], fill='tonexty',name = 'Jumlah Point Tim'))



fig.show()
engine= create_engine('sqlite:///:memory:')



dfeast.to_sql('data_table', engine) 

gp= pd.read_sql_query('SELECT "Games Played" FROM data_table group by Team', engine)
dfeastrp.insert(3, 'GP', gp)
dfeastgpp = dfeastrp['SPTS'] / 82

dfeastgpp

# dfeastrp['GP'].max()
fig = {

        'data': [ 

             {

                'values' : dfeastgpp,

                'labels' : dfeastrp['Rank'],

                'domain' : {'x': [0, 1]},

                'name' : 'Points / Game',

                'hoverinfo' : 'label+percent+name',

                'hole' : 0.3,

                'type' : 'pie'

              },

             ],

         'layout' : {

                     'title' : 'Rata Rata Point per Game yang didapatkan Team Sesuai Ranking',

                     'annotations' : [

                                        { 'font' : {'size' : 20},

                                          'showarrow' : False,

                                          'text' : ' ',

                                          'x' : 0.20,

                                          'y' : 1

                                         },

                                      ]    

                     }

        }

py.iplot(fig)
dfatl = dfeast[dfeast['Team'] == "ATL"]

dfatl
dfl.plot.scatter("EFF", "PTS", alpha=0.5, color= "r", figsize=(13,5))

plt.xlabel('Efisiensi Pemain')

plt.ylabel('Point')
msk = np.random.rand(len(dfl)) < 0.8

train = dfl[msk]

test = dfl[~msk]



plt.figure(figsize=(13,5))

plt.scatter(train.EFF, train.PTS, alpha=0.5, color='blue')

plt.xlabel("Efisiensi Pemain")

plt.ylabel("Point")

plt.show()
regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['EFF']])

train_y = np.asanyarray(train[['PTS']])

regr.fit (train_x, train_y)



print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
plt.figure(figsize=(13,5))

plt.scatter(train.EFF, train.PTS, alpha=0.5, color='blue')

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')

plt.xlabel("Efisiensi Pemain")

plt.ylabel("Point")
testk = regr.intercept_ + regr.coef_ * 1475

testk
from sklearn.metrics import r2_score



test_x = np.asanyarray(test[['EFF']])

test_y = np.asanyarray(test[['PTS']])

test_y_ = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
plt.figure(figsize=(13,5))

x_data, y_data = (dfl["EFF"].values, dfl["PTS"].values)

plt.plot(x_data, y_data, 'mo' ,alpha=0.5)

plt.xlabel('Efisiensi Pemain')

plt.ylabel('Point')

plt.show()
def sigmoid(x, Beta_1, Beta_2):

     y = 1 / (1 + np.exp (-beta_1*(x-Beta_2)) )

     return y



def expo (x, Beta_0, Beta_1):

     y = Beta_0*np.exp(Beta_1*x)

     return y



def qubic (x, Beta_0, Beta_1, Beta_2, Beta_3):

     y = Beta_0+Beta_1*x+Beta_2*x**2+Beta_3*x**3

     return y
beta_1 = 1.0

beta_2 = 1

beta_3= 1

beta_4=0.1



Y_preds =sigmoid (x_data, beta_1, beta_2)

xdata =x_data/max(x_data)

ydata =y_data/max(y_data)
from scipy.optimize import curve_fit

popt1, pcov1 = curve_fit(sigmoid, xdata, ydata, maxfev = 10000)

popt2, pcov2 = curve_fit (expo, xdata, ydata, maxfev = 10000) 

popt3, pcov3 = curve_fit (qubic, xdata, ydata, maxfev = 10000) 



print(" Exponensial","B1 = %f, B2=%f"%(popt2[0], popt2[1]))

print(" Sigmoid","B1 = %f, B2=%f"%(popt1[0], popt1[1]))

print(" Qubic","B1 = %f, B2=%f"%(popt3[0], popt3[1]))
x = np.linspace(10, 200, 1000)

x = x/max(x)

plt.figure(figsize=(13,5))



y1 = expo(x, *popt1)

y2 = sigmoid(x, *popt2)

y3 = qubic(x, *popt3)



plt.plot(xdata, ydata, 'mo',alpha=0.5 ,label='data')

plt.plot(x,y1, linewidth=3.0, label='Eksponensial')

plt.plot(x,y2, linewidth=3.0, label='Sigmoid')

plt.plot(x,y3, linewidth=3.0, label='Qubic')

plt.legend(loc='best')

plt.xlabel('Efisiensi Pemain')

plt.ylabel('Points')

plt.show()
msk = np.random.rand(len(dfl)) < 0.8



train_x = xdata[msk]

test_x = xdata[~msk]

train_y = ydata[msk]

test_y = ydata[~msk]



# build the model using train set

popt1, pcov1 = curve_fit(sigmoid, train_x, train_y, maxfev = 100000)

popt2, pcov2 = curve_fit(expo, train_x, train_y, maxfev = 100000)

popt3, pcov3 = curve_fit(qubic, xdata, ydata, maxfev = 10000)



y_hat1 = sigmoid(test_x, *popt1)

y_hat2 = expo(test_x, *popt2)

y_hat3 = qubic(test_x, *popt3)



print ("Sigmoid")

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat1 - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat1 - test_y) ** 2))

from sklearn.metrics import r2_score

print("R2-score: %.2f" % r2_score(y_hat1 , test_y) )



print ("\nExponens")

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat2 - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat2 - test_y) ** 2))

from sklearn.metrics import r2_score

print("R2-score: %.2f" % r2_score(y_hat2 , test_y) )



print ("\nQubic")

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat3 - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat3 - test_y) ** 2))

from sklearn.metrics import r2_score

print("R2-score: %.2f" % r2_score(y_hat3 , test_y) )
x = np.linspace(10, 200, 1000)

x = x/max(x)

plt.figure(figsize=(13,5))



y3 = qubic(x, *popt3)



plt.plot(xdata, ydata, 'mo',alpha=0.5 ,label='data')

plt.plot(x,y3, linewidth=3.0, label='Qubic')

plt.legend(loc='best')

plt.xlabel('Menit Bermain')

plt.ylabel('Points')

plt.show()
print ("\nQubic")

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat3 - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat3 - test_y) ** 2))

from sklearn.metrics import r2_score

print("R2-score: %.2f" % r2_score(y_hat3 , test_y) )
dfdel1 = dfc.Name != 'Sim Bhullar'

dfdel2 = dfc.Name != 'Jerrelle Benimon'

dfcl = dfc[dfdel1 & dfdel2]
dfcl["TRB/MIN"] = dfcl["REB"]/dfcl["MIN"] 

dfcl["AST/MIN"] = dfcl["AST"]/dfcl["MIN"]



fig, ax = plt.subplots()



x_var="AST/MIN"

y_var="TRB/MIN"



colors = {'SG':'blue', 'PF':'red', 'PG':'green', 'C':'purple', 'SF':'orange'}



ax.scatter(dfcl[x_var], dfcl[y_var], c=dfcl['Pos'].apply(lambda x: colors[x]), s = 10)



# set a title and labels

ax.set_title('NBA Dataset')

ax.set_xlabel(x_var)

ax.set_ylabel(y_var)
dfn = dfcl[["AST/MIN","TRB/MIN"]]



kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(dfn)

print(kmeans.cluster_centers_)
d0=dfn[y_kmeans == 0]

d1=dfn[y_kmeans == 1]

d2=dfn[y_kmeans == 2]

d3=dfn[y_kmeans == 3]

d4=dfn[y_kmeans == 4]



plt.scatter(d0[x_var], d0[y_var], s = 10, c = 'blue', label = 'D0')

plt.scatter(d1[x_var], d1[y_var], s = 10, c = 'green', label = 'D1')

plt.scatter(d2[x_var], d2[y_var], s = 10, c = 'red', label = 'D2')

plt.scatter(d3[x_var], d3[y_var], s = 10, c = 'purple', label = 'D3')

plt.scatter(d4[x_var], d4[y_var], s = 10, c = 'orange', label = 'D4')



plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
d0[x_var]='SG'

d1[x_var]='PF'

d2[x_var]='PG'

d3[x_var]='C'

d4[x_var]='SF'



dflist = pd.concat([d0[x_var], d1[x_var], d2[x_var], d3[x_var], d4[x_var]])
dfcluster = (dfc[["Name", "Team", "Pos"]])

dfcluster



dfcl["TRB/MIN"]

dfcl["AST/MIN"]



dfcluster.insert(2, 'TRBMIN', dfcl["TRB/MIN"])

dfcluster.insert(3, 'ASTMIN', dfcl["AST/MIN"])

dfcluster.insert(5, 'Next Pos', dflist)

dfcluster