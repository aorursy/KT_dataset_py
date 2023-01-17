# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format ='retina'
df = pd.read_csv('../input/sea-air flux of methane and CO2 2014 -data-.csv')

df.head()

# date variables, lat-long variables, continous variables, a few discrete ones

# getting rid of first 3 columns

df = df.iloc[:,3:]
metadata = pd.read_csv('../input/sea-air flux of methane and CO2 2014 -dict-.csv', usecols=['attrlabl', 'attrdef'])

metadata.head()

# Variable descriptions for added aid
df.describe()

# Some variables have '999' hard-code for no value, need to clean that
# droping when 999 inside columns

#(df['airmar_lat'][df['airmar_lat']==999])



df2 = df.copy()

df2.iloc[:,0].count() # how many rows...

for col in df2:

    #print(col)

    df2 = df2[df2[col]!=999]

df2.iloc[:,0].count() # then how many rows now after not considering 999 values for columns
def roll_plots(data, col_init_pos):

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    for (i, j) in enumerate(range(col_init_pos, col_init_pos + 3)):

        #print(i)

        plot = ax[i].scatter(data.iloc[:,j], data.iloc[:,j+1], c=data.iloc[:,j+2], s=5, alpha=0.25, edgecolors='none', )

        ax[i].set_xlabel(data.iloc[:,j].name)

        ax[i].set_ylabel(data.iloc[:,j+1].name)

        #plt.legend(  )

        #plt.colorbar(plot)

        

roll_plots(df2, np.random.randint(1, high=df2.shape[1]-3) ) # adding some eandomness there...

# you can find interesting graphs here,

# call roll_plots() with different # of initial column and see



# would be nice to have this without the '(calculated)' variables...

# also just for the upper east sampled geographic section

    
# sub dataset for that geo lat-long region 

df2['near_west_margin'] = 0.0

df2['near_west_margin'][ (df2['ship_lat']>78) & (df2['ship_long']>9.5) ] = 1 

df2['near_west_margin'].describe()
# '(calculated)' variables

for (i, col) in enumerate(metadata['attrdef'].str.split(' ')):

    if col[0] == '(calculated)':

        print(i, metadata.iloc[i]['attrdef'], '\n')
# graph

fig, ax = plt.subplots(figsize=(8,6))

plot = ax.scatter(df2['ysiDOsat'], df2['ysifDOM'], c=df2['ship_hullwatT'], alpha=0.25)

ax.set_xlabel('ysiDOsat')

ax.set_ylabel('ysifDOM', rotation=360)

ax.get_yaxis().labelpad = 25

cbar = fig.colorbar(plot)

cbar.ax.get_yaxis().labelpad = 25

cbar.ax.set_ylabel('ship_hullwatT', rotation=360)

ax.set_title('2014 Western Svalbard: Seawater Measurements')
# comparing 

fig, ax = plt.subplots(figsize=(8,6))

plot = ax.scatter(df2['corrch4iso'], df2['corrco2iso'], alpha=0.25)

ax.set_title('corrco2iso vs corrch4iso')

# no clear pattern
# linear reg

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

col_names = df2.columns

df3 = pd.DataFrame(scaler.fit_transform(df2), columns=col_names)

#df3.describe()



xtr, xte, ytr, yte = train_test_split(

    df3[['ysiwatT', 'ysisal', 'ysipH', 'ysiDO', 'ysiDOsat', 'ysifDOM', 'near_west_margin', 'co2wat']], 

    df3[['ch4iso']], # near-surface seawater, others ch4watlo ch4wathi co2wat co2iso ch4iso

    train_size=0.75)

xtr.shape, ytr.shape, xte.shape, yte.shape





lr = LinearRegression()

lr.fit(xtr, ytr)

print('Train Score is: {:.2f}%'.format(lr.score(xtr, ytr) * 100))

print('Checking Score: {:.2f}%'.format(r2_score(ytr, lr.predict(xtr)) * 100))



# graph

yte_pred = lr.predict(xte)

print('Test Score is: {:.2f}%'.format(r2_score(yte, yte_pred) * 100))



res = yte - yte_pred
# Lasso

from sklearn.linear_model import Lasso



lasso = Lasso(alpha=0.001, random_state=123)

lasso.fit(xtr, ytr)

print('Train Score is: {:.2f}%'.format(lasso.score(xtr, ytr) * 100))

print('Checking Score: {:.2f}%'.format(r2_score(ytr, lasso.predict(xtr)) * 100))



# graph

yte_l_pred = lasso.predict(xte)

print('Test Score is: {:.2f}%'.format(r2_score(yte, yte_l_pred) * 100))



yte.shape

yte_l_pred.shape



res_l = yte - yte_l_pred.reshape(len(yte_l_pred), 1)



# Comparing coefs 

print("Linear Reg.coefs:")

lr.coef_

print("Lasso (sparse) coefs:")

lasso.coef_



print('Difference in coefs:')

print('(Positive is more impact)')

lasso.coef_ - lr.coef_



print('seawater temp. and dissolved O2 got zeroed!')
# Ridge

from sklearn.linear_model import Ridge



ridge = Ridge(alpha=100, random_state=123)

ridge.fit(xtr, ytr)

print('Train Score is: {:.2f}%'.format(ridge.score(xtr, ytr) * 100))

print('Checking Score: {:.2f}%'.format(r2_score(ytr, ridge.predict(xtr)) * 100))



yte_r_pred = ridge.predict(xte)

print('Test Score is: {:.2f}%'.format(r2_score(yte, yte_r_pred) * 100))



yte.shape

yte_r_pred.shape



res_r = yte - yte_r_pred.reshape(len(yte_l_pred), 1)



# Comparing coefs 

print("Linear Reg.coefs:")

lr.coef_

print("Ridge (multicol-robust) coefs:")

ridge.coef_



print('Difference in (abs) coefs:')

print('(Positive is more impact)')

ridge.coef_ - lr.coef_



print('4th and 5th got the most adjustment.')