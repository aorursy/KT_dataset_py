import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from IPython.display import display #, HTML
from patsy import dmatrices
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colorbar 
import os
import seaborn as sns
%matplotlib inline
#%matplotlib notebook
E10_price = 1.379
SP98_price = 1.459
df = pd.read_excel('../input/measurements2.xlsx')
display(df.tail(6))
%matplotlib inline
fig, axarr = plt.subplots(3,1)
fig.set_size_inches(w=13, h=15)
red_halo = (1., 0, 0, 0.07)
# plot of speed
axarr[0].scatter(df.speed.values, df.consume.values, color=red_halo, s=400, marker='o', linewidths=0)
axarr[0].scatter(df.speed.values, df.consume.values, color='#000000', s=0.5, marker='o', alpha=.9)
axarr[0].grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)
axarr[0].set_xlabel('Speed in km/h')
axarr[0].set_ylabel('consume in L/100km')

#plot of distances
axarr[1].scatter(df.distance.values, df.consume.values, color=red_halo, s=400, marker='o',linewidths=0)
axarr[1].scatter(df.distance.values, df.consume.values, color='#000000', s=0.5, marker='o', alpha=.9)
axarr[1].grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)
axarr[1].set_xlabel('Distance in km')
axarr[1].set_ylabel('consume in L/100km')

#plot of outside temperature
axarr[2].scatter(df.temp_outside.values, df.consume.values, color=red_halo, s=400, marker='o',linewidths=0)
axarr[2].scatter(df.temp_outside.values, df.consume.values, color='#000000', s=0.5, marker='o', alpha=.9)
axarr[2].grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)
axarr[2].set_xlabel('outside temperature in °C')
text = axarr[2].set_ylabel('consume in L/100km')

from mpl_toolkits.mplot3d import Axes3D
alpha = 0.1
fig = plt.figure()
fig.set_size_inches(w=12, h=9)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df.speed.values, df.distance.values,  df.consume.values,  
           color='r', s=400, marker='o', alpha=alpha, linewidths=0)
ax.scatter(df.speed.values, df.distance.values,  df.consume.values,   
           color='#000000', s=5, marker='o', alpha=1, linewidths=0)


ax.set_xlabel('Speed in km/h')
ax.set_ylabel('Distance in km')
ax.set_ylim(0, 60) # exclude the two outliers from the graphic
text = ax.set_zlabel('Consume in L/100km')
# indicator if the heating was not used at all
df['heating_off']=df['temp_inside'].isnull()
df['heating_off']=df['heating_off'].apply(float)
# if the heating was turned completely off, replace the inside temperature by the outside temperature
df['temp_inside'].fillna(df['temp_outside'], inplace=True)
# get the temperature difference
df['temp_diff'] = df['temp_inside'] - df['temp_outside']
df['temp_diff_square'] = df['temp_diff']**2
# add the square and cube of the speed to the frame
df['speedsquare'] = df['speed']**2  # 5% better accuracy
df['speedcube'] =  df['speed']**3  # 1% better accuracy

# translate the gas type to something machine readable
def gastype(in_string):
    '''gas type in, integer out'''
    if in_string == "E10":
        return 0
    else:
        return 1
df['gas_type_num']= df['gas_type'].apply(gastype)
print(df.groupby(by='gas_type')['consume'].mean().round(2))
# use interactive or passive graphic format
%matplotlib inline
# %matplotlib notebook
from ipywidgets import *
# slope get down rather fast.
# delay: after 7% of an hour, equals 5 minutes, cut off.
def sigmo(x, slope = 33.5, delay = 0.07):
    return 1 / (1 + np.exp( (x-delay)* slope))

df['time_h'] = (df['distance']/df['speed'])
df['startphase'] = sigmo(df['time_h'])

#check the additional worth of it
rgr = LinearRegression()
rgr.fit(df.startphase.values.reshape(-1, 1), df.consume.values)
regression_fit = 'R² of the line: {:.2f}'.format(rgr.score(df.startphase.values.reshape(-1, 1), df.consume.values))

fig, axarr = plt.subplots(2,1)
fig.set_size_inches(w=12, h=8)
fig.tight_layout()
alpha = 0.05
# plot of startphase
line0, = axarr[0].plot(df.startphase.values,  df.startphase.values*rgr.coef_[0]+rgr.intercept_, 
                       color='#000000', alpha=.3, linewidth=0.5)
text0 = axarr[0].text(0.15, 10, regression_fit)
halo0 = axarr[0].scatter(df.startphase.values, df.consume.values, color=red_halo, s=400, marker='o',linewidths=0)
poin0 = axarr[0].scatter(df.startphase.values, df.consume.values, color='#000000', s=1, marker='o', alpha=.9)
x = np.linspace(0,df.startphase.values.max())
sigm0, = axarr[0].plot(x, sigmo(x)*df.consume.values.max())
axarr[0].grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)
axarr[0].set_xlabel('hours')
axarr[0].set_ylabel('consume in L/100km')
axarr[0].text(0.0, 7, 'segregated blob')


# heating costs extra in the startphase, later not so much
df['start_heating'] = df['startphase'] * df['temp_diff'] 
#check the additional worth of it
rgr = LinearRegression()
rgr.fit(df.start_heating.values.reshape(-1, 1), df.consume.values)
regression_fit = 'R² of the line: {:.2f}'.format(rgr.score(df.start_heating.values.reshape(-1, 1), df.consume.values))

#plot of start_heating
line1, = axarr[1].plot(df.start_heating.values, df.start_heating.values*rgr.coef_[0]+rgr.intercept_, 
                       color='#000000', alpha=.3, linewidth=0.5)
text1 = axarr[1].text(2, 10, regression_fit)
halo1 = axarr[1].scatter(df.start_heating.values, df.consume.values, color=red_halo, s=400, marker='o',linewidths=0)
poin1 = axarr[1].scatter(df.start_heating.values, df.consume.values, color='#000000', s=1, marker='o', alpha=.9)
axarr[1].grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)
axarr[1].set_xlabel('Start: Heating')
axarr[1].text(0, 7, 'segregated blob')
text = axarr[1].set_ylabel('consume in L/100km')

def update(slope, delay):
    df['startphase'] = sigmo(df['time_h'], slope=slope, delay=delay)
    rgr = LinearRegression()
    rgr.fit(df.startphase.values.reshape(-1, 1), df.consume.values)
    score = rgr.score(df.startphase.values.reshape(-1, 1), df.consume.values)
    #text0.set_position(0.4, 10)
    text0.set_text('R² of the line: {:.2f}'.format(score))
    line0.set_data(df.startphase.values, df.startphase.values*rgr.coef_[0]+rgr.intercept_)
    halo0.set_offsets(np.c_[df.startphase.values, df.consume.values])
    poin0.set_offsets(np.c_[df.startphase.values, df.consume.values])
    
    x = np.linspace(0,df.startphase.values.max())
    sigm0.set_data(x, sigmo(x, slope=slope, delay=delay)*df.consume.values.max())

    df['start_heating'] = df['startphase'] * df['temp_diff'] 
    #check the additional worth of it
    rgr = LinearRegression()
    rgr.fit(df.start_heating.values.reshape(-1, 1), df.consume.values)
    score = rgr.score(df.start_heating.values.reshape(-1, 1), df.consume.values)

    text1.set_text('R² of the line: {:.2f}'.format(score))
    line1.set_data(df.start_heating.values, df.start_heating.values*rgr.coef_[0]+rgr.intercept_)
    halo1.set_offsets(np.c_[df.start_heating.values, df.consume.values])
    poin1.set_offsets(np.c_[df.start_heating.values, df.consume.values])
    
    fig.canvas.draw()

def plot_interactive():
    slope_w = widgets.FloatSlider(description='slope', value=33.5, min=0, max=200, step=0.01, width=600,
                                 layout = Layout(width='60%', height='40px'))
    delay_w = widgets.FloatSlider(description='delay', value=0.070, min=0, max=0.2, step=0.001,
                                 layout = Layout(width='60%', height='40px'))
    interact(update, slope=slope_w, delay=delay_w)

fig.canvas.draw()
plot_interactive()
%matplotlib inline
rgr = LinearRegression()
fig, ax = plt.subplots(1,4)
fig.set_size_inches(w=15, h=3)
X=np.array([0,1.])
# plot of speed
rgr.fit(df.heating_off.values.reshape(-1, 1), df.consume.values)
ax[0].plot(X,  X*rgr.coef_[0]+rgr.intercept_, color='#000000', alpha=1, linewidth=0.5)
ax[0].scatter(df.heating_off.values, df.consume.values, color=red_halo, s=200, marker='o', linewidths=0)
ax[0].scatter(df.heating_off.values, df.consume.values, color='#000000', s=0.5, marker='o', alpha=.9)
ax[0].grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)
ax[0].set_xlabel('heating on (0) or off (1)')
ax[0].set_ylabel('consume in L/100km')

rgr.fit(df.AC.values.reshape(-1, 1), df.consume.values)
ax[1].plot(X,  X*rgr.coef_[0]+rgr.intercept_, color='#000000', alpha=1, linewidth=0.5)
ax[1].scatter(df.AC.values, df.consume.values, color=red_halo, s=200, marker='o', linewidths=0)
ax[1].scatter(df.AC.values, df.consume.values, color='#000000', s=0.5, marker='o', alpha=.9)
ax[1].grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)
ax[1].set_xlabel('AC off (0) or on (1)')
ax[1].set_ylabel('consume in L/100km')

rgr.fit(df.rain.values.reshape(-1, 1), df.consume.values)
ax[2].plot(X,  X*rgr.coef_[0]+rgr.intercept_, color='#000000', alpha=1, linewidth=0.5)
ax[2].scatter(df.rain.values, df.consume.values, color=red_halo, s=200, marker='o', linewidths=0)
ax[2].scatter(df.rain.values, df.consume.values, color='#000000', s=0.5, marker='o', alpha=.9)
ax[2].grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)
ax[2].set_xlabel('rain (1) or not (0)')
text = ax[2].set_ylabel('consume in L/100km')

rgr.fit(df.sun.values.reshape(-1, 1), df.consume.values)
ax[3].plot(X,  X*rgr.coef_[0]+rgr.intercept_, color='#000000', alpha=1, linewidth=0.5)
ax[3].scatter(df.sun.values, df.consume.values, color=red_halo, s=200, marker='o', linewidths=0)
ax[3].scatter(df.sun.values, df.consume.values, color='#000000', s=0.5, marker='o', alpha=.9)
ax[3].grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)
ax[3].set_xlabel('no sun (0) or sun (1)')
text = ax[3].set_ylabel('consume in L/100km')
fig.tight_layout()
# make a list of chosen predictors
prediction_values = ['distance','start_heating', 'startphase', 'time_h',
                     'speed', 'speedsquare', 'speedcube', 
                     'temp_diff', 'temp_diff_square', 'temp_outside', 
                     'heating_off', 'AC', 'rain', 'sun']
##############################################################
# in theory, the regression needs scaled data. However, 
# using scaled data had no effect. So scaling is not used today.
# scaler = StandardScaler()
# X_scale = scaler.fit_transform(df[prediction_values].values) 

# make numpy arrays for sklearn
X = df[prediction_values].values
Y = df['consume'].values
Y_gas = df['gas_type_num'].values

print('The result after crossfitting two regressions to get the effect of gas sorts:')
def crossfit_regression(X, Y, Y_gas, estimator, estimator_kwargs={}):
    ''' If you use the same code three times, make a function.
    '''
    # apply regression
    rgr = estimator(**estimator_kwargs)
    rgr.fit(X, Y)

    # apply again, this time trained on gas type
    rgr_gas = estimator(**estimator_kwargs)
    rgr_gas.fit(X, Y_gas)

    # get the residuals (the not-yet-explained variance left in the data)
    Y_residuals = Y - rgr.predict(X)
    X_gas_residuals = Y - rgr_gas.predict(X)

    # fit the residuals to get the influence of the gas type
    # reshape(-1,1) is necessary since scikit 19 if you have a single feature
    rgr_inference = estimator(**estimator_kwargs)
    rgr_inference.fit(X_gas_residuals.reshape(-1,1), Y_residuals)
    if hasattr(rgr_inference, 'coef_'):
        difference = rgr_inference.coef_[0] # there is only one coef, but given as list of one. :-)
    if hasattr(rgr_inference, 'estimator_'):
        difference = rgr_inference.estimator_.coef_[0]
    score = rgr.score(X, Y)
    print('The difference in consumption between E10 and SP98 is {:.2f} liter.'.format(difference))
    print('And R² of the model: {:.3f}'.format(score))    
    return difference, rgr, score

result, rgr, score= crossfit_regression(X, Y, Y_gas, LinearRegression, estimator_kwargs={})
result_list = []
result_list.append({'estimator':'linear', 'difference':result, 'R²': score})
print('The importance of the other factors (F-Values)')
from sklearn.feature_selection import f_regression
F, pval = f_regression(X, Y)
predictors_df = pd.DataFrame(columns=prediction_values)
predictors_df.loc['F-value of predictor'] = F
print(predictors_df.round(2).transpose())
# prepare the first two steps in the old way... as i already have the code :-)
rgr = LinearRegression()
rgr.fit(X, Y)

# apply again, this time trained on gas type
rgr_gas = LinearRegression()
rgr_gas.fit(X, Y_gas)

# get the residuals (the not-yet-explained variance left in the data)
Y_residuals = Y - rgr.predict(X)
X_gas_residuals = Y - rgr_gas.predict(X)

# prepare dataframe for statsmodels
residuals = pd.DataFrame(Y_residuals, columns=['consume'])
residuals['E10']=X_gas_residuals

# fit regression in statsmodels format.
# it's like sklearn rgr.fit(E10, consume)
results = smf.ols('consume ~ E10', data=residuals).fit()

# get the result out of the vast array of available values
consume = results.conf_int().loc['E10']

# assuming the difference is the beta of E10: attention, this
# is true only if the factor E10 is completely independent
difference = results.params[1]

# results.bse contains the standard error - if you want other confidence intervals...
print("The car uses {:.2f} L/100km more with E10. The 95% interval is between {:.2f} and {:.2f}".format(
                    difference, consume[0], consume[1]))
from sklearn.linear_model import ElasticNet
# l1_ratio = 1 is like Lasso, l1_ratio = 0 is like ridge.
# I would like mostly ridge, however, to make sure that useless features disappear,
# I keep a little bit of Lasso.
# alpha determines the cutoff. The smaller alpha, the less coefficients are used.
kwargs = {'alpha':1., 'l1_ratio':0.05, 'max_iter':10000}

difference, rgr, score = crossfit_regression(X, Y, Y_gas, estimator=ElasticNet, estimator_kwargs=kwargs)
result_list.append({'estimator':'elastic net', 'difference':difference, 'R²': rgr.score(X, Y)})
for i, word in enumerate(prediction_values):
    print('influence of {:s}: {:.4f}'.format(word, rgr.coef_[i]))
#correlate
corr = df[['consume'] + prediction_values].corr()

#show correlation in colors
fig, ax = plt.subplots(1,1)
fig.set_size_inches(w=6, h=6)
ax.set_title('Correlations between features')
image = plt.imshow(corr, cmap = cm.RdBu)
cb = fig.colorbar(image)
pos = range(len(['consume'] + prediction_values))
plt.xticks(pos, ['consume'] + prediction_values, rotation=90)
catch = plt.yticks(pos, ['consume'] + prediction_values)
prediction_values = ['distance', 'startphase', 'time_h',
                     'speed', 'speedsquare', 
                     'temp_diff', 'temp_diff_square', 'temp_outside', 
                      'AC', 'rain', 'sun']
# replace X
X = df[prediction_values].values
difference, rgr, score = crossfit_regression(X, Y, Y_gas, LinearRegression)
result_list.append({'estimator':'linear with updated features', 'difference':difference, 'R²': score})
from sklearn.linear_model import (TheilSenRegressor, RANSACRegressor, HuberRegressor)
print('TheilSen')
difference, rgr, score = crossfit_regression(X, Y, Y_gas, TheilSenRegressor)
result_list.append({'estimator':'TheilSen', 'difference':difference, 'R²': score})
# residual_threshold is the border to classify data as "data" or "outlier"
# the standard deviation is a bit more open than the standard Mean Average Deviation
# min_samples tells the algorithm how many datapoints have to be taken 
# stop_score tells him he can stop searching when his R² is better than that.
kwargs = {'min_samples':0.75, 'residual_threshold':Y.std(), 'max_trials':1000}
print('RANSAC')
difference, rgr, score = crossfit_regression(X, Y, Y_gas, estimator=RANSACRegressor, estimator_kwargs=kwargs)
datapoint_mask = rgr.inlier_mask_ #to check the excluded points later
result_list.append({'estimator':'RANSAC', 'difference':difference, 'R²': score})
print('RANSAC made {} different fits.'.format(rgr.n_trials_))
# Huber does not ignore outliers, instead they get a smaller weight determined by alpha.
# epsilon says how far from the standard a value should be to be classified as outlier.
# small epsi makes for less points selected, big epsi includes more points.
kwargs = {'alpha':0.01, 'epsilon':1.35}
print('HuberRegressor')
difference, rgr, score = crossfit_regression(X, Y, Y_gas, estimator=HuberRegressor, estimator_kwargs=kwargs)
result_list.append({'estimator':'HuberRegressor', 'difference':difference, 'R²': score})
df.loc[~datapoint_mask, ['consume']+prediction_values]
fig, axarr = plt.subplots(1,1)
fig.set_size_inches(w=16, h=8)
red_halo = (1, 0.1, 0.1, 0.9)
blue_halo = (0.15, 0.15, 0.9, 0.3)
yellow = (0.85, 0.75, 0, 1)
outliers = df.loc[~datapoint_mask]
inliers = df.loc[datapoint_mask]
startphase = df.loc[df['startphase'].values>0.02]

# plot of speed
axarr = sns.kdeplot(inliers['distance'], inliers['speed'],
                  cmap=cm.Blues, alpha=0.9, shade=True, shade_lowest=False)
axarr.scatter(df.distance.values, df.speed.values, 
              color=blue_halo, s=(df.consume.values)**2.5, marker='o', linewidths=0)
axarr.scatter(startphase['distance'], startphase['speed'].values,
              color=yellow, s=(startphase.consume.values)**2.5, marker='o', linewidths=0)
axarr.scatter(outliers.distance.values, outliers.speed.values, 
              color=red_halo, s=(outliers.consume.values)**2.3, marker='x', linewidths=0)
axarr.grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)
#axarr.set_xlim(0, 85)
axarr.set_ylim(10, 90)

axarr.set_ylabel('Speed in km/h')
axarr.set_xlabel('distance in km')
text = plt.title('speed on distance density, counsume as size, inlier/outlier/startphase as color',y=1)
fig, axarr = plt.subplots(1,1)
fig.set_size_inches(w=16, h=8)

# plot of speed
axarr = sns.kdeplot(inliers['consume'], inliers['temp_diff'], 
                  cmap=cm.Blues, alpha=0.95, shade=True, shade_lowest=False)
axarr.scatter(df['consume'], df['temp_diff'].values,
              color=blue_halo, s=(df.consume.values)**2.5, marker='o', linewidths=0)
axarr.scatter(startphase['consume'], startphase['temp_diff'].values,
              color=yellow, s=(startphase.consume.values)**2.5, marker='o', linewidths=0)
axarr.scatter(outliers['consume'], outliers['temp_diff'].values, 
              color=red_halo, s=(outliers.consume.values)**2.3, marker='x', linewidths=0)
axarr.grid(color='#000000', linestyle='-', linewidth=1, alpha=0.08)

axarr.set_ylabel('temperature difference in °C')
axarr.set_xlabel('consume')
text = plt.title('temperature on consume density, counsume as size, inlier/outlier/startphase as color',y=1)
X = df.loc[datapoint_mask, prediction_values].values
Y = df.loc[datapoint_mask,'consume'].values
Y_gas = df.loc[datapoint_mask,'gas_type_num'].values

difference, rgr, score = crossfit_regression(X, Y, Y_gas, estimator=LinearRegression)
result_list.append({'estimator':'linear with updated features and cleaned data', 'difference':difference, 'R²': score})
pd.DataFrame(result_list).round(2)
