import pandas as pd

import numpy as np

import seaborn as sb; sb.set()

import matplotlib.pyplot as plt

import random

from scipy.optimize import curve_fit

import scipy.integrate as spi



from keras.models import Sequential

from keras.layers import Dense, Flatten, GlobalMaxPooling1D, Dropout, GRU, BatchNormalization

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D



# Some utility functions for data preparation

def upsample(country,df):

    """pad zeros before the onset of the outbreak for a given country"""

    df = df.loc[df.country==country]

    days = int((df.date.values[0]-start).astype('timedelta64[D]')/ np.timedelta64(1, 'D'))

    data = [[start+ np.timedelta64(i,'D'), country, 0, 0] for i in range(days)]

    return pd.DataFrame(data, columns=['date','country',target, 'duration_outbreak'])



def split_sequences(sequences, n_steps):

    """preparation of data with n_steps window predicting the next value"""

    X, y = list(), list()

    for i in range(len(sequences)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the dataset

        if end_ix > len(sequences)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]

    X.append(seq_x)

    y.append(seq_y)

    return np.array(X), np.array(y)
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')



#load a mapping file to attribute international 3 lettres standard code to communicate with external data

df_map = pd.read_csv('../input/coronavirus/countryMapping.csv')

df = df.merge(df_map, how='left', on = 'Country/Region')

#transform data and create extra features

df['date'] = pd.to_datetime(df.ObservationDate)

df = df.loc[df.Confirmed>0]

df['Actives'] = df.apply(lambda x: x.Confirmed - x.Deaths - x.Recovered, axis = 1)

df.rename(columns= {'Country Code': 'country'}, inplace =True)

start = df.groupby('country').min().reset_index().rename(columns={'date':'start_outbreak'})[['country', 'start_outbreak']]

df = df.merge(start, on = 'country')

df['duration_outbreak'] = df.apply(lambda x: (x.date-x.start_outbreak).days, axis=1)

df_li = df[['date', 'country', 'Confirmed', 'Deaths', 'Recovered','Actives', 'duration_outbreak']]

#sum over regions of a same contry

df_li = df_li.groupby(['date','country']).agg({'Confirmed':'sum', 'Deaths':'sum', 'Recovered':'sum', 'Actives':'sum', 'duration_outbreak':'max'})

df_li.reset_index(inplace=True)



#select contries with more than 15 days of outbreak

lastDate = df[['date','country']].groupby('country').max().values[0][0]

sel = df.loc[df['date']== lastDate].groupby('country').max()

sel = sel.loc[sel.duration_outbreak > 15]

df_li = df_li.loc[df_li.country.isin(sel.index)]

print('%i countries'%df_li.country.unique().shape[0])

#create new features after aggregation and filtering

df_li['lethality'] = df_li.apply(lambda x: 100*x.Deaths / x.Confirmed , axis = 1)

#Convert cumulative Confirmed to new cases

newCases=[]

for cc in df_li.country.unique():

    cumul = df_li.loc[df_li.country==cc].sort_values('duration_outbreak').Confirmed.values

    newCases.extend([(cc, 0, cumul[0])] + [(cc, ix+1, i- cumul[ix]) for ix,i in enumerate(cumul[1:])])

newCases = pd.DataFrame(newCases, columns = ['country','duration_outbreak','new_cases'])

df_li = df_li.merge(newCases, on=['country','duration_outbreak'])

#Calculate prevalence from world bank population data

dfp = pd.read_csv('../input/coronavirus/world.csv')

wb_pop=  dfp.loc[dfp['Series Code'] == 'SP.POP.TOTL',['Country Code','2018 [YR2018]']]

wb_pop['2018 [YR2018]'] = wb_pop['2018 [YR2018]'].apply(lambda x: eval(x) if x !='..' else np.nan)

wb_pop = wb_pop.rename(columns={'2018 [YR2018]': 'population'})

df_li = df_li.merge(wb_pop, left_on = 'country', right_on='Country Code')

del df_li['Country Code']

df_li['prevalence'] = df_li.apply(lambda x: 10000*x.Confirmed/x.population, axis = 1)

df_li.dropna(inplace=True)

#basic variable transform

df_li['confirmed_log'] = df_li.Confirmed.apply(lambda x: np.log(x))

df_li.head()
#create dataset for 1DCNN

target = 'confirmed_log'

df_tar =  df_li[['date','country']+[target]].dropna()



#upsample for missing values before the onset of outbreak in a given country by padding 0

start = df_tar.loc[df_tar.country=='CHN'].date.values[0]

for cc in df_tar.country.unique():

    nsamp=upsample(cc, df_tar)

    df_tar = pd.concat([df_tar,nsamp], axis = 0, sort=False)

#Select countries with a significant number of cases    

countries = df_li.loc[df_li.Confirmed>200,'country'].unique()

sel = df_tar.loc[df_tar.country.isin(countries)].sort_values(['date','country'])

#make dataset compliant for 1DCNN inputs

sel = sel.pivot(index='date', columns='country', values=target)

#split on historic samples

train = sel[sel.index < '2020-03-07']

test = sel[sel.index >= '2020-03-07']

#slice the convolution window on 11 days

n_steps = 11

X,y = split_sequences(train.values,n_steps)

n_features = X.shape[2]

sel.head()
# define model

model = Sequential()

model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))

model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(48, activation='relu'))

model.add(Dense(n_features))

model.compile(optimizer='adam', loss='mse')

model.summary()



#unsucessful tested layers:

#model.add(BatchNormalization())

#model.add(Dropout(0.2))

#model.add(GRU(64, dropout=0.1, recurrent_dropout=0.1))
# fit model

random.seed(3)

epochs=500

start_plot=0

history = model.fit(X, y, epochs=epochs, verbose=0)

loss = history.history['loss'][start_plot:]



#plot loss. Accuracy is not calculated because there is no validation set

start_plot=0

iepochs = range(start_plot,epochs)

plt.plot(iepochs, loss, 'bo', label='Training loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
def predict(n, previous=None):

    X_test = train.iloc[-n_steps+n:]

    if not previous is None:

        X_test = pd.concat([X_test, previous], axis = 0, sort=False)

    ref = test.iloc[n].values    

    X_test = X_test.values.reshape((1, n_steps, n_features))

    yhat = [int(round(i)) for i in model.predict(X_test, verbose=0)[0]]

    df =pd.DataFrame(list(zip(test.columns, yhat,ref)), columns = ['country','yhat','ref'])

    rmse = round(np.sqrt(np.mean((df.yhat.values-df.ref.values)**2)),2)

    del df['ref']

    df['date']=test.iloc[n:n+1].index.values[0]

    df = df.pivot(index='date', columns='country', values='yhat')

    if previous is None:

        previous = df

    else:

        previous = pd.concat([previous, df], axis=0, sort=False)

    return previous,rmse



rmses=[]

previous=None

for i in range(test.shape[0]): 

    previous,rmse = predict(i,previous)

    rmses.append(rmse)

                           

print ('rmse : %.2f'%np.mean(rmses))
pred = pd.concat([train.iloc[-1:], previous], axis=0, sort=False)

x = pred.index.values.astype('datetime64[D]')

x=['-'.join(str(i).split('-')[1:]) for i in x]

xb = sel.index.values.astype('datetime64[D]')

xb=['-'.join(str(i).split('-')[1:]) for i in xb][-15:]



fig, axes = plt.subplots(ncols=4, nrows=6, figsize=(16,16))

for cc, ax in zip(test.columns, axes.flat):

    yb=list(sel[cc].values)[-15:]

    y=list(pred[cc].values)

    sb.lineplot(x=xb, y=yb, ax=ax).set_title(cc)

    sb.lineplot(x=x, y=y, ax=ax)

fig.tight_layout(h_pad=1, w_pad=0)
def diff_eqs(init,t):  

    y=np.zeros((3))

    v = init   

    y[0] = - beta * v[0] * v[1]

    y[1] = beta * v[0] * v[1] - gamma * v[1]

    y[2] = gamma * v[1]

    return y  

    

def SIR(x, *p):

    beta, gamma, amp, s0 ,i0 = p

    init = (s0, i0, 0.0)  

    res = spi.odeint(diff_eqs,init,x)

    return res[:,1]*amp



def fitted(y, p_initial):

    duration= y.shape[0]

    y = y/ np.linalg.norm(y)

    x = np.linspace(0, duration,duration)

    popt, pcov = curve_fit(SIR, x, y, p0=p_initial)

    print(popt)

    return (SIR(x, *popt), popt, x, y)    
s0=1-1e-4      #initial p(succeptible)

i0=8e-3        #initial p(infectious)

beta = 0.3

gamma = 0.1

amp = 0.75

p_initial = [beta, gamma, amp,s0,i0]



sel = df_li.loc[df_li.country == 'CHN', 'Actives'].values  

yhat, popt, x, y = fitted(sel, p_initial)



fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))

ax[0].plot(x,y, '-r', label='Infectious') 

ax[0].plot(yhat, '-b', label='Predicted')



yhat = SIR(range(0,100), *popt)    

ax[1].plot(yhat, '-b', label='Infectious')
S0=1-1e-4      #initial p(succeptible)

I0=1e-5        #initial p(infectious)

INPUT = (S0, I0, 0.0)

beta = 0.43

gamma = 0.07

amp = 1.2

p_initial = [beta, gamma, amp,s0,i0]



sel = df_li.loc[df_li.country == 'ITA', 'Actives'].values  

yhat, popt, x, y = fitted(sel, p_initial)



fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))

ax[0].plot(x,y, '-r', label='Infectious') 

ax[0].plot(yhat, '-b', label='Predicted')



yhat = SIR(range(0,120), *popt)    

ax[1].plot(yhat, '-b', label='Infectious')