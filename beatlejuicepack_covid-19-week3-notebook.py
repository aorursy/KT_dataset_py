%%capture

!pip install -upgrade pip

!pip install ipywidgets matplotlib==2.2.0 pyproj==1.9.6 numpy pandas proj pillow cython sklearn datetime seaborn pylab
import matplotlib as mpl

import matplotlib.pyplot as plt

import pandas as pd

#import pylab as pl

import numpy as np

from scipy import stats

import seaborn as sns

import datetime as dt



from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

#from pylab import rcParams



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score

from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



from scipy.optimize import curve_fit



mpl.style.use('ggplot') # optional: for ggplot-like style

print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0 # check for latest version of Matplotlib



%matplotlib inline
#fname = 'SYB62_1_201907_Population, Surface Area and Density.csv'

#site = 'https://data.un.org/_Docs/SYB/CSV/SYB62_1_201907_Population,%20Surface%20Area%20and%20Density.csv'

#!wget -O  'SYB62_1_201907_Population, Surface Area and Density.csv' 'https://data.un.org/_Docs/SYB/CSV/SYB62_1_201907_Population,%20Surface%20Area%20and%20Density.csv'

#!unzip -o -j 'SYB62_1_201907_Population, Surface Area and Density.csv'
path = '../input/covid19-global-forecasting-week-3/'

csv = 'train.csv'

filepath = path + csv



pdf = pd.read_csv(filepath, parse_dates=['Date'])



pdf['Year'] = pdf['Date'].dt.year

pdf['Month'] = pdf['Date'].dt.month

pdf['Week'] = pdf['Date'].dt.week

pdf['Day'] = pdf['Date'].dt.day

pdf['DateKey'] = (pdf['Date'] - pdf['Date'].min()).astype(int)



bkdf = pdf

today = pd.datetime.now()



print(' Filepath: ', filepath, '\n', 'Shape: ', pdf.shape, '\n', 'Date: ', today)

pdf.head()
#path2 = '../output/kaggle/working/'

csv2 = 'SYB62_1_201907_Population, Surface Area and Density.csv'

filepath2 = csv2 #path2 + csv2



asdf = pd.read_csv(filepath2, encoding='iso-8859-1', header=1)

asdf = asdf.pivot_table(values='Value', index=['Region/Country/Area', 'Unnamed: 1', 'Year'], columns='Series').reset_index()

asdf = asdf.rename(columns={'Unnamed: 1':'Country_Region'}, inplace=False)

asdf['Population mid-year Total'] = ((asdf['Population mid-year estimates for females (millions)'] 

                                    + asdf['Population mid-year estimates for males (millions)'])

                                    * 1000000)

asdf = asdf.drop(['Year'], axis=1, inplace=False)

asdf = asdf.dropna()



print(' Filepath: ', filepath2, '\n', 'Shape: ', asdf.shape, '\n', 'Date: ', today)

asdf.head()
tp = '../input/covid19-global-forecasting-week-3/'

tf = 'test.csv'

filepath3 = tp + tf



test = pd.read_csv(filepath3, parse_dates=['Date'])



print(' Filepath: ', filepath3, '\n', 'Shape: ', test.shape, '\n', 'Date: ', today)

test.head()
tes = pd.concat([pdf, test])

tes = tes.reset_index(drop=True)

tes = pd.merge(tes, asdf, on='Country_Region', how='left')



tes['Year'] = tes['Date'].dt.year

tes['Month'] = tes['Date'].dt.month

tes['Week'] = tes['Date'].dt.week

tes['Day'] = tes['Date'].dt.day

tes['DateKey'] = (tes['Date'] - tes['Date'].min()).astype(int)



tes['ConfirmedCases'] = tes['ConfirmedCases'].replace(np.nan, 0)

tes['ConfirmedCases'] = tes['ConfirmedCases'].dropna()



tes['Fatalities'] = tes['Fatalities'].replace(np.nan, 0)

tes['Fatalities'] = tes['Fatalities'].dropna()



tes['Cases per pop'] = (tes['Population mid-year Total'] / (tes['ConfirmedCases'].max()))

casp = (tes['Population mid-year Total'].mean() / (tes['ConfirmedCases'].max()))

tes['Cases per pop'] = tes['Cases per pop'].fillna(casp).astype(int)

tes['Case Density'] = tes['Cases per pop'] / tes['Population density']



print('Shape: ', tes.shape)

tes.tail()
res = pd.merge(pdf, asdf, on='Country_Region', how='left')

res = res.reset_index(drop=True)





res['Cases per pop'] = (res['Population mid-year Total'] / (res['ConfirmedCases'].max()))

casp2 = (res['Population mid-year Total'].mean() / (res['ConfirmedCases'].max()))

res['Cases per pop'] = res['Cases per pop'].fillna(casp).astype(int)

res['Case Density'] = res['Cases per pop'] / res['Population density']





print('Shape: ', res.shape)

res.tail()
tes['Country_Region'].value_counts()
res['Country_Region'].value_counts()
name = res

name.corr()[name.corr() >= 0.3]
name = res.dropna()



def calculate_pvalues(name):

    name = name._get_numeric_data()

    col = pd.DataFrame(columns=name.columns)

    pval = col.transpose().join(col, how='outer')

    for r in name.columns:

        for c in name.columns:

            pval[r][c] = round(pearsonr(name[r], name[c])[1], 4)

    return pval



# 0 SIGNIFICANT, >0.1 NO SIGNIFICANCE

calculate_pvalues(name)[calculate_pvalues(name) <= 0.2]
xydf = res



xx = 'Date'

yy = 'ConfirmedCases'



xxdf = xydf[xx]

yydf = xydf[yy]



plt.scatter(xxdf, yydf,  color='blue')

plt.xlabel("True Values: " + xx)

plt.ylabel("True Values: " + yy)

plt.show()
resch = res.Country_Region.str.contains('China')

resch = res[resch]

resch
hubeidf = resch.dropna()

hubeidf = hubeidf.reset_index(drop=True)

hubei2df = hubeidf.Province_State.str.contains('Hubei')

hubei2df = hubeidf[hubei2df]

hubei2df
y = hubei2df['ConfirmedCases']

Z = hubei2df.drop(['ConfirmedCases', 'Country_Region', 'Date', 'Province_State'], axis=1, inplace=False)





Input=[('scale',StandardScaler()),('model',LinearRegression())]



pipe=Pipeline(Input)



pipe.fit(Z,y)



ypipe=pipe.predict(Z)



hubei2df['Predicted Cases'] = ypipe

hubei2df['Predicted Cases'] = hubei2df['Predicted Cases'].astype(int)

r_squared = r2_score(y, ypipe)

print('The R-square value is: ', r_squared)

hubei2df.head()
y2 = hubei2df['Fatalities']

Z2 = hubei2df.drop(['Fatalities', 'Country_Region', 'Date', 'Province_State'], axis=1, inplace=False)





Input=[('scale',StandardScaler()),('model',LinearRegression())]



pipe=Pipeline(Input)



pipe.fit(Z2,y2)



y2pipe=pipe.predict(Z2)



hubei2df['Predicted Fatalities'] = y2pipe

hubei2df['Predicted Fatalities'] = hubei2df['Predicted Fatalities'].astype(int)

r_squared2 = r2_score(y2, y2pipe)

print('The R-square value is: ', r_squared2)

hubei2df.head()
x = hubei2df['DateKey']

cr = hubei2df['ConfirmedCases']



f = np.polyfit(cr, x, 16)

p = np.poly1d(f)

print(p)



r_squared = r2_score(x, p(cr))

mse = mean_squared_error(x, p(cr))



plt.title('Polynomial fit')

plt.xlabel('DateKey')

plt.ylabel('ConfirmedCases')

plt.plot(x, ypipe, '-', x, cr, '.')

plt.show()



print('The R-square value is: ', r_squared)

print('The MSE value is: ', mse)
def PlotPolly(model, independent_variable, dependent_variabble, Name):

    x_new = np.linspace(15, 55, 100)

    y_new = model(x_new)



    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')

    plt.title('Polynomial Fit with Matplotlib for Confirmed Cases ~ Predicted Cases')

    ax = plt.gca()

    ax.set_facecolor((0.898, 0.898, 0.898))

    fig = plt.gcf()

    plt.xlabel(Name)

    plt.ylabel('ConfirmedCases')

    plt.show()

    plt.close()
pc = hubei2df['Predicted Cases']

f = np.polyfit(y, pc, 16)

p = np.poly1d(f)

print(p)

PlotPolly(p, y, pc, 'Predicted Cases')

print(np.polyfit(y, pc, 16))
hubei2df.corr()[hubei2df.corr() >= 0.1 ]
def sigmoid(x, Beta_1, Beta_2):

     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))

     return y
xdata = hubei2df['DateKey'].values/max(hubei2df['DateKey'].values)

ydata = hubei2df['ConfirmedCases'].values/max(hubei2df['ConfirmedCases'].values)

popt, pcov = curve_fit(sigmoid, xdata, ydata)



print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))



x = np.linspace(0, 51, 52)

x = x/max(x)

plt.figure(figsize=(8,5))

y = sigmoid(x, *popt)



plt.plot(xdata, ydata, 'ro', label='data')

plt.plot(x,y, linewidth=3.0, label='fit')

plt.legend(loc='best')

plt.ylabel('ConfirmedCases')

plt.xlabel('Date')

plt.show()
# split data into train/test

msk = np.random.rand(len(hubei2df)) < 0.8

train_x = xdata[msk]

test_x = xdata[~msk]

train_y = ydata[msk]

test_y = ydata[~msk]



# build the model using train set

popt, pcov = curve_fit(sigmoid, train_x, train_y)



# predict using test set

y_hat = sigmoid(test_x, *popt)



# evaluation

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) **2))

from sklearn.metrics import r2_score

print("R2-Score: %.2f" % r2_score(y_hat, test_y))



#logistic function

Y_pred = sigmoid(xdata, popt[0], popt[1])



#plot initial prediction against datapoints

plt.plot(xdata, Y_pred)

plt.plot(xdata, ydata, 'ro')
xdata2 = tes['ConfirmedCases'].values/max(tes['ConfirmedCases'].values)

sigpred = sigmoid(xdata2, *popt)#[0], popt[1])

print(len(sigpred))

print(sigpred)

tes['Predicted Cases (Sig)'] = (sigpred * popt[0]).astype(int)
xdata2 = tes['Fatalities'].values/max(tes['Fatalities'].values)

sigpred = sigmoid(xdata2, *popt)#[0], popt[1])

print(len(sigpred))

print(sigpred)

tes['Predicted Fatalities (Sig)'] = (sigpred * popt[0]).astype(int)

tes [32300:32370]
sub = pd.read_csv(path + 'submission.csv')

sub['ConfirmedCases'] = tes['Predicted Cases (Sig)']

sub['Fatalities'] = tes['Predicted Fatalities (Sig)']

sub.to_csv('submission.csv', index=False)

sub.head()