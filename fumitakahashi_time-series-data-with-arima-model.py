import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
%matplotlib inline
train = pd.read_csv('../input/web-traffic-time-series-forecasting/train_1.csv.zip')
train.head()
train.info()
# Missing data per day

days = [r for r in range(train.shape[1] - 1)]
fig, ax = plt.subplots(figsize = (10,7))
plt.xlabel('Day')
plt.ylabel('# of null data')
ax.axvline(x=123, c = 'red',  lw = 0.5)
plt.plot(days, train.iloc[:,1:].isnull().sum())
train.columns[123]
# Histogram of pages with their number of data missing. 

train.isnull().sum(axis = 1).hist()
train = train.fillna(0)
train.Page
import re
def split_page(page):
  w = re.split('_|\.', page)
  return ' '.join(w[:-5]), w[-5], w[-2], w[-1]

li = list(train.Page.apply(split_page))
df = pd.DataFrame(li)
df.columns = ['Title', 'Language', 'Access_type','Access_origin']
df = pd.concat([train, df], axis = 1)
del df['Page']
df.iloc[:, -4:]
df[df.Language == 'de'].iloc[:,-4:]
df.Language.value_counts().plot(kind = 'bar')
df.Access_type.value_counts().plot(kind = 'bar')
df.Access_origin.value_counts().plot(kind = 'bar', color = 'orange')
sum_all = df.iloc[:,:-4].sum(axis = 0)

days = list(r for r in range(sum_all.shape[0]))

fig = plt.figure(figsize = (10, 7))
plt.xlabel('Days')
plt.ylabel('Views')
plt.title('Page View of All Pages')
plt.plot(days, sum_all)

summap = {}
lang_list = ["en", "ja", "de", "fr", "zh", "ru", "es", "commons", "www"]
for l in lang_list:
  summap[l] = df[df.Language == l].iloc[:,:-4].sum(axis = 0)/df[df.Language == l].shape[0]

fig = plt.figure(figsize = (15, 7))
plt.xlabel('Days')
plt.ylabel('Views')
plt.title('Average Page View by Language')

for key in summap:
  plt.plot(days, summap[key], label = key)
plt.legend()
plt.show()



from scipy.fftpack import fft

#data = df.iloc[idx,0:-4]

fig, ax = plt.subplots(figsize = (15, 7))

fftmean = {}
fftxvals = {}

for key in summap:
  fftval = fft(df[df.Language == key].iloc[:, :-6])

#calculate magnitude
  fftmag = [np.sqrt(np.real(x)*np.real(x)+
                    np.imag(x)*np.imag(x)) for x in fftval]
  arr = np.array(fftmag)
#calculate mean
  fftmean[key] = np.mean(arr,axis=0)

  fftxvals[key] = [day/fftmean[key].shape[0] for day in range(fftmean[key].shape[0])]

  npts = len(fftxvals[key])//2 + 1
  fftmean[key] = fftmean[key][:npts]/fftmean[key].shape[0]
  fftxvals[key] = fftxvals[key][:npts]
  ax.plot(fftxvals[key][1:], fftmean[key][1:], label = key)

plt.axvline(x = 1/7, color = 'black', lw = 0.5)
plt.axvline(x = 2/7, color = 'black', lw = 0.5)
plt.axvline(x = 3/7, color = 'black', lw = 0.5)

plt.xlabel('Frequency')
plt.ylabel('Views')
plt.title('Fourier Transform of Average View by Language')

plt.legend()
plt.show()
sums = pd.concat([df.iloc[:,-4:], df.iloc[:,:-4].sum(axis = 1)], axis = 1)
sums.columns = ['Title', 'Language', 'Access_type', 'Access_origin', 'sumvalues']
max_list = {}
for l in lang_list:
  lang_sums = sums[sums.Language == l]
  max_list[l] = lang_sums.sumvalues.idxmax()
df[df.index.isin(max_list.values())].iloc[:,-4:]
import matplotlib as mpl
mpl.rcParams['font.family'] = 'AppleGothic'

def plot_trend(lang, idx):
    fig = plt.figure(1,figsize=(10,5))
    plt.plot(days, df.iloc[idx,:-4])
    plt.xlabel('day')
    plt.ylabel('views')
    plt.title('Most Viewed Pages ({})'.format(lang))  
    plt.show()
for key in max_list:
  plot_trend(key, max_list[key])
sums2 = sums.drop(labels = max_list.values(), axis = 0)
max_list2 = {}
for l in lang_list:
  lang_sums = sums2[sums2.Language == l]
  max_list2[l] = lang_sums.sumvalues.idxmax()
  
df[df.index.isin(max_list2.values())].iloc[:,-4:]
main_titles = dict(zip(list(df[df.index.isin(max_list.values())].Language), list(df[df.index.isin(max_list.values())].Title)))

all_access = {}
mobile_access = {}
desktop_access = {}

for l in lang_list:
  all_access[l] = df.index[(df.Language == l) & (df.Title == main_titles[l]) & (df.Access_type == 'all-access')]
  mobile_access[l] = df.index[(df.Language == l) & (df.Title == main_titles[l]) & (df.Access_type == 'mobile-web')]
  desktop_access[l] = df.index[(df.Language == l) & (df.Title == main_titles[l]) & (df.Access_type == 'desktop')]
def plot_trend_access_type(lang):

    plt.figure(figsize=(15,4))

    plt.subplot(1,3, 1)
    plt.plot(days, df.iloc[all_access[l][0],:-4])
    plt.title('All Access ({})'.format(lang))
    plt.subplot(1,3, 2)
    plt.plot(days, df.iloc[mobile_access[l][0],:-4])
    plt.title('Mobile-web Access ({})'.format(lang))
    plt.subplot(1,3, 3)
    plt.plot(days, df.iloc[desktop_access[l][0],:-4])
    plt.title('Desktop Access ({})'.format(lang))
    plt.show()
for l in lang_list:
  plot_trend_access_type(l)
def plot_trend_access_origin(lang):

    plt.figure(figsize=(10,3))

    plt.subplot(1,2, 1)
    plt.plot(days, df.iloc[all_access[l][0],:-4])
    plt.title('{} ({})'.format(df.iloc[all_access[l][0],:].Access_origin,lang))
    plt.subplot(1,2, 2)
    plt.plot(days, df.iloc[all_access[l][1],:-4])
    plt.title('{} ({})'.format(df.iloc[all_access[l][1],:].Access_origin, lang))
    plt.show()
  
for l in lang_list:
    plot_trend_access_origin(l)
# Split the data into train and test

series = df.iloc[:, 0:-4]

from sklearn.model_selection import train_test_split

X = series.iloc[:,:500]
y = series.iloc[:,500:]

X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, test_size=0.1, random_state=42)

from statsmodels.tsa.arima_model import ARIMA

train, test = X_train[86431], y_train[86431]
record = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	# fit model
	model = ARIMA(record, order=(4,1,0))
	model_fit = model.fit(disp=False)
	# forecast one step
	yhat = model_fit.forecast()[0]
	# store the result
	predictions.append(yhat)
	record.append(test[t])

from math import sqrt
from sklearn.metrics import mean_squared_error
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
fig = plt.subplots(figsize=(10,7))
plt.plot(test)
plt.plot(predictions, color='red')
plt.legend(['test', 'prediction'])
plt.title('ARIMA with Walk-foward validation')
plt.show()

# evaluate an ARIMA model for a given order (p,d,q) with MSE
def evaluate_arima_model(train, test, arima_order):
	# prepare training dataset
	record = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(record, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		record.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	aic_score= model_fit.aic
	return error, aic_score


import warnings
warnings.filterwarnings("ignore")


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(train, test, p_values, d_values, q_values):
	train, test = train.astype('float32'), test.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse, aic = evaluate_arima_model(train, test, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
						aic_out = aic
					print('ARIMA  ', order,    'MSE=%.3f   AIC=%.3f' % ( mse, aic))
				except:
					continue
	#print('Best ARIMA:    ', best_cfg,  'MSE=%.3f  AIC=%.3f' % (best_cfg, best_score))


p_values = [0, 5]
d_values = range(0, 2)
q_values = range(4, 8)
warnings.filterwarnings("ignore")
evaluate_models(X_train[86431], y_train[86431], p_values, d_values, q_values)
p_values = [0, 1]
d_values = [0,1]
q_values = [5,7]
warnings.filterwarnings("ignore")
evaluate_models(X_train[86431], y_train[86431], p_values, d_values, q_values)
train, test = X_train[86431], y_train[86431]
record = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	# fit model
	model = ARIMA(record, order=(1,0,5))
	model_fit = model.fit(disp=False)
	# forecast one step
	yhat = model_fit.forecast()[0]
	# store the result
	predictions.append(yhat)
	record.append(test[t])
 
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
fig = plt.subplots(figsize=(10,7))
plt.plot(test)
plt.plot(predictions, color='red')
plt.plot(predictions, color='red')
plt.legend(['test', 'prediction'])
plt.title('ARIMA with Walk-foward validation 2')
plt.show()