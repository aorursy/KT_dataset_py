import pandas as pd

import numpy as np



from matplotlib import pyplot as plt

from matplotlib.dates import date2num, num2date

from matplotlib import dates as mdates

from matplotlib import ticker

from matplotlib.colors import ListedColormap

from matplotlib.patches import Patch



from scipy import stats as sps

from scipy.interpolate import interp1d



from IPython.display import clear_output



%config InlineBackend.figure_format = 'retina'
#import data

cases_df = pd.read_csv('/kaggle/input/covid19-challenges/test_data_canada.csv')

cases_df['date'] = pd.to_datetime(cases_df['date'])

province_df = cases_df.groupby(['province', 'date'])['id'].count()

province_df.index.rename(['region', 'date'], inplace=True)

hr_df = cases_df.groupby(['region', 'date'])['id'].count()

canada_df = pd.concat((province_df, hr_df))
prov_name = 'Ontario'



def prepare_cases(cases):

    # modification - Isha Berry et al.'s data already come in daily

    #new_cases = cases.diff()

    new_cases = cases



    smoothed = new_cases.rolling(7,

        win_type='gaussian',

        min_periods=1,

        # Alf: switching to right-aligned instead of centred to prevent leakage of

        # information from the future

        #center=True).mean(std=2).round()

        center=False).mean(std=2).round()

    

    zeros = smoothed.index[smoothed.eq(0)]

    if len(zeros) == 0:

        idx_start = 0

    else:

        last_zero = zeros.max()

        idx_start = smoothed.index.get_loc(last_zero) + 1

    smoothed = smoothed.iloc[idx_start:]

    original = new_cases.loc[smoothed.index]

    

    return original, smoothed



cases = canada_df.xs(prov_name).rename(f"{prov_name} cases")



original, smoothed = prepare_cases(cases)





# We create an array for every possible value of Rt

R_T_MAX = 12

r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)



# Gamma is 1/serial interval

# https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article

GAMMA = 1/4



def get_posteriors(sr, window=7, min_periods=1):

    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))



    # Note: if you want to have a Uniform prior you can use the following line instead.

    # I chose the gamma distribution because of our prior knowledge of the likely value

    # of R_t.

    

    # prior0 = np.full(len(r_t_range), np.log(1/len(r_t_range)))

    prior0 = np.log(sps.gamma(a=3).pdf(r_t_range) + 1e-14)



    likelihoods = pd.DataFrame(

        # Short-hand way of concatenating the prior and likelihoods

        data = np.c_[prior0, sps.poisson.logpmf(sr[1:].values, lam)],

        index = r_t_range,

        columns = sr.index)



    # Perform a rolling sum of log likelihoods. This is the equivalent

    # of multiplying the original distributions. Exponentiate to move

    # out of log.

    posteriors = likelihoods.rolling(window,

                                     axis=1,

                                     min_periods=min_periods).sum()

    posteriors = np.exp(posteriors)



    # Normalize to 1.0

    posteriors = posteriors.div(posteriors.sum(axis=0), axis=1)

    

    return posteriors



posteriors = get_posteriors(smoothed)



def highest_density_interval(pmf, p=.95):

    

    # If we pass a DataFrame, just call this recursively on the columns

    if(isinstance(pmf, pd.DataFrame)):

        return pd.DataFrame([highest_density_interval(pmf[col]) for col in pmf],

                            index=pmf.columns)

    

    cumsum = np.cumsum(pmf.values)

    best = None

    for i, value in enumerate(cumsum):

        for j, high_value in enumerate(cumsum[i+1:]):

            if (high_value-value > p) and (not best or j<best[1]-best[0]):

                best = (i, i+j+1)

                break

            

    low = pmf.index[best[0]]

    high = pmf.index[best[1]]

    return pd.Series([low, high], index=['Low', 'High'])





hdis = highest_density_interval(posteriors)



most_likely = posteriors.idxmax().rename('ML')



# Look into why you shift -1

result = pd.concat([most_likely, hdis], axis=1)



#result.tail()





#from pandas.plotting import register_matplotlib_converters

#register_matplotlib_converters()



def plot_rt(result, ax, state_name):

    

    ax.set_title(f"{prov_name}")

    

    # Colors

    ABOVE = [1,0,0]

    MIDDLE = [1,1,1]

    BELOW = [0,0,0]

    cmap = ListedColormap(np.r_[

        np.linspace(BELOW,MIDDLE,25),

        np.linspace(MIDDLE,ABOVE,25)

    ])

    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5

    

    index = result['ML'].index.get_level_values('date')

    values = result['ML'].values

    

    # Plot dots and line

    ax.plot(index, values, c='k', zorder=1, alpha=.25)

    ax.scatter(index,

               values,

               s=40,

               lw=.5,

               c=cmap(color_mapped(values)),

               edgecolors='k', zorder=2)

    

    # Aesthetically, extrapolate credible interval by 1 day either side

    lowfn = interp1d(date2num(index),

                     result['Low'].values,

                     bounds_error=False,

                     fill_value='extrapolate')

    

    highfn = interp1d(date2num(index),

                      result['High'].values,

                      bounds_error=False,

                      fill_value='extrapolate')

    

    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),

                             end=index[-1]+pd.Timedelta(days=1))

    

    ax.fill_between(extended,

                    lowfn(date2num(extended)),

                    highfn(date2num(extended)),

                    color='k',

                    alpha=.1,

                    lw=0,

                    zorder=3)



    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);

    

    # Formatting

    ax.xaxis.set_major_locator(mdates.MonthLocator())

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.xaxis.set_minor_locator(mdates.DayLocator())

    

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax.yaxis.tick_right()

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.margins(0)

    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)

    ax.margins(0)

    ax.set_ylim(0.0,3.5)

    ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))

    fig.set_facecolor('w')

    

results = {}



provinces_to_process = canada_df.loc[['Waterloo']]



for prov_name, cases in provinces_to_process.groupby(level='region'):

    clear_output(wait=True)

    print(f'Processing {prov_name}')

    new, smoothed = prepare_cases(cases)

    print('\tGetting Posteriors')

    try:

        posteriors = get_posteriors(smoothed)

    except:

        display(cases)

    print('\tGetting HDIs')

    hdis = highest_density_interval(posteriors)

    print('\tGetting most likely values')

    most_likely = posteriors.idxmax().rename('ML')

    result = pd.concat([most_likely, hdis], axis=1)

    results[prov_name] = result.droplevel(0)

    

clear_output(wait=True)

print('Done.')
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))

fig, ax = plt.subplots(figsize=(600/72,400/72))



plot_rt(result, ax, prov_name)



fig.tight_layout()

fig.set_facecolor('w')

ax.set_title(f'Real-time $R_t$ for {prov_name}')

ax.set_ylim(.5,3.5)

ax.xaxis.set_major_locator(mdates.WeekdayLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
waterloo_df = cases_df[((cases_df['province']=='Ontario')&(cases_df['region']=='Waterloo'))]

waterloo_cases_df = waterloo_df.groupby(['date'])['id'].count().to_frame()

waterloo_cases_rt_df = pd.DataFrame({'ML':result['ML'].values,'Low':result['Low'].values,'High':result['High'].values,'new cases':waterloo_cases_df['id'].values},index=result['ML'].index.get_level_values('date'))



#create dataframe for Waterloo region based on R_t calculated (see plot above for R_t)

waterloo_cases_rt_df
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import TimeseriesGenerator



values = waterloo_cases_rt_df.values

values = values.astype(np.float32)



scaler = StandardScaler()

scaled = scaler.fit_transform(values)



epochs_to_predict = 3 #predict 3 days into the future

X = scaled[:][:-epochs_to_predict] # remove data from last 3 days

y = scaled[:,3][epochs_to_predict:] #target/label column, remove data from first 3 days



#split into train and test sets

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.10, random_state=42, shuffle = False)



#generate time series for training and testing datasets

train_generator = TimeseriesGenerator(trainX, trainY, length=epochs_to_predict, sampling_rate=1, batch_size=epochs_to_predict)

test_generator = TimeseriesGenerator(testX, testY, length=epochs_to_predict, sampling_rate=1, batch_size=epochs_to_predict)



train_X, train_y = train_generator[0]

test_X, test_y = test_generator[0]



train_samples = train_X.shape[0]*len(train_generator)

test_samples = test_X.shape[0]*len(test_generator)



X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))

X_test = np.reshape(testX, (testX.shape[0],testX.shape[1],1))
import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



#define the architecture of the model

def create_model():

    model = Sequential()

    model.add(LSTM(32,input_shape=(4,1),return_sequences=True,activation='relu'))

    model.add(LSTM(64,return_sequences=True))

    model.add(LSTM(128))

    model.add(Dense(1))

    model.summary()

    return model
model=create_model()

lr_reduce =tf.keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=0.3,min_lr=1e-3)

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
#train the model

model.fit(X_train,trainY, epochs=100, batch_size=1, validation_data=(X_test, testY),validation_freq=10,callbacks=[lr_reduce])
from math import floor

#predict

predicts=[waterloo_cases_rt_df['new cases'][-1]]

newdata=scaled.copy()



for i in range(epochs_to_predict):

  predict=model.predict(newdata[-4:,3].reshape(1,4,1))

  predicts.append(floor(abs(predict)))

  newdata = newdata[:-1,:]



#new predicted cases starting from the second index

print(predicts)  
import datetime

import numpy as np

#extend dates by 3 days

a = result['ML'].index.get_level_values('date')

a[-1]+ datetime.timedelta(days=1)

date_rng = pd.date_range(start=a[-1], end=a[-1]+datetime.timedelta(days=epochs_to_predict), freq='D')



#plot

fig, ax = plt.subplots(figsize=(800/72,600/72))

plt.plot(waterloo_cases_df)

plt.plot(date_rng,predicts)

plt.gcf().autofmt_xdate()

plt.legend(['Original', 'Prediction'], loc='best',fontsize=14)

ax.set_title(f'Predicted New Cases Over {epochs_to_predict} Days for {prov_name} Region',fontsize=14)

plt.xlabel('Days',fontsize=14)

plt.ylabel('# of New Cases',fontsize=14)