import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
import warnings  
warnings.filterwarnings('ignore')
pd.set_option("display.precision", 3)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_rows", 25)
covid_19_data = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
print(covid_19_data.columns)
confirmed_by_date = covid_19_data.groupby(['Date','Country_Region'])[['ConfirmedCases']].agg("sum")
deaths_by_date = covid_19_data.groupby(['Date','Country_Region'])[['Fatalities']].agg("sum")
confirmed_cols = confirmed_by_date['ConfirmedCases'].unstack()
deaths_cols = deaths_by_date['Fatalities'].unstack()
countries = set(list(covid_19_data['Country_Region']))
print(countries)
#countries = ['Australia','US','Italy','Korea, South','United Kingdom']
doubling_ts = {}
models = {}
for COL in countries:
    a = confirmed_cols[COL].copy()[:]
    a_reset = a.reset_index()
    ydf = a_reset[COL].fillna(value=0)
    x = ydf.index.values
    y = ydf.values  
    transformer = FunctionTransformer(np.log, validate=True)
    y_trans = transformer.fit_transform(y.reshape(-1,1) + 1) #y[:,np.newaxis]
    y_trans = np.nan_to_num(y_trans)
    x_in = x.reshape(-1,1)
    weights = np.exp(x_in)
    model = LinearRegression().fit(x_in, y_trans, sample_weight=weights.flatten())
    y_fit = model.predict(x_in)
    #y = A * exp(B*x) + C | log(y)  | log(y-C) = log(A) + B * x
    #A,B = np.log(model.coef_[0]),(model.intercept_[0])
    A,B = model.coef_[0],model.intercept_[0]
    #print('A=',model.coef_,'B=',model.intercept_)
    #print('A * exp{Bx} + C')
    x = x_in.flatten()
    yfit = np.exp(A*x + B)
    yfit2 = B * np.exp(A*x)
    # The gradient of the line is the exponential power
    ygrad = np.gradient(np.log(y))
    ygrad[np.isnan(ygrad)] = 0.000
    #ygrad[np.isinf(ygrad)] = 0.0001
    ydouble = np.log(2)/ygrad
    #print('Doubling:', np.log(2)/np.log(np.exp(A)), ',Cases:', confirmed_cols[COL].max())
    doubling_ts[COL] = ydouble
    models[COL] = model
    # Linearised:
    # y = ab^x = ae^{lnb * x}
    # y = B * exp{A}^{x}
    # for y = x_o * b^{t}
    # Tdouble = log(2)/log(b) - in this one Tdoube = log(2)/log(exp(A))
    doubling_ts[COL] = ydouble
    
DFD = pd.DataFrame(doubling_ts)
DFD = DFD.rolling(3, win_type='gaussian').mean(std=2)
DFD['date'] = covid_19_data['Date']
DFD = DFD.set_index('date')
CFD = confirmed_cols[countries]
CDFD = pd.concat([DFD,CFD],axis=1,keys=['Doubling','Cases']).swaplevel(0,1,axis=1).sort_index(axis=1)
#countries = ['Australia','US','Italy','Korea, South','United Kingdom']
deaths_doubling_ts = {}
deaths_models = {}
for COL in countries:
    a = deaths_cols[COL].copy()[:]
    a_reset = a.reset_index()
    ydf = a_reset[COL].fillna(value=0)
    x = ydf.index.values
    y = ydf.values  
    transformer = FunctionTransformer(np.log, validate=True)
    y_trans = transformer.fit_transform(y.reshape(-1,1) + 1) #y[:,np.newaxis]
    y_trans = np.nan_to_num(y_trans)
    x_in = x.reshape(-1,1)
    weights = np.exp(x_in)
    model = LinearRegression().fit(x_in, y_trans, sample_weight=weights.flatten())
    y_fit = model.predict(x_in)
    #y = A * exp(B*x) + C | log(y)  | log(y-C) = log(A) + B * x
    #A,B = np.log(model.coef_[0]),(model.intercept_[0])
    A,B = model.coef_[0],model.intercept_[0]
    #print('A=',model.coef_,'B=',model.intercept_)
    #print('A * exp{Bx} + C')
    x = x_in.flatten()
    yfit = np.exp(A*x + B)
    yfit2 = B * np.exp(A*x)
    # The gradient of the line is the exponential power
    ygrad = np.gradient(np.log(y))
    ygrad[np.isnan(ygrad)] = 0.000
    #ygrad[np.isinf(ygrad)] = 0.0001
    ydouble = np.log(2)/ygrad
    #print('Doubling:', np.log(2)/np.log(np.exp(A)), ',Cases:', deaths_cols[COL].max())
    #doubling_time[COL] = {'Doubling':np.log(2)/np.log(np.exp(A)), 'Cases':country_cols[COL].max()}
    doubling_ts[COL] = ydouble
    deaths_models[COL] = model
    doubling_ts[COL] = ydouble
dftest = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
dates = dftest.groupby(['Country_Region','Date']).agg('mean')
dates_new = dates.index.droplevel().values
countries = set(list(dftest['Country_Region']))
#print(dates.shape)
#print(dates.size * len(countries))

predictions = {}
i = 0
for COL in countries:
    df = dftest.loc[dftest['Country_Region']==COL]
    #a = dates[COL].copy()[:]
    #a_reset = a.reset_index()
    idx = df['ForecastId'].values.astype(int)
    #print(idx.shape)
    #print(dates_new.shape)
    x_new = np.arange(len(x),len(x)+len(idx))
    x_in = x_new.reshape(-1,1)
    y_fit = deaths_models[COL].predict(x_in)
    deaths_fit = deaths_models[COL].predict(x_in)
    if i==0:
        predictions = pd.DataFrame({'cases':y_fit.flatten(),'country':COL, 'ID':idx, 
                                    'deaths': deaths_fit.flatten()},index=x_new)
    else:
        predictions = pd.concat([predictions,pd.DataFrame({'cases':y_fit.flatten(),'country':COL, 
                                                           'ID':idx, 'deaths': deaths_fit.flatten()},index=x_new)])
    i+=1
print(predictions.columns)
print(predictions.shape)
pred = predictions[['ID','cases','deaths']]
pred_new = pred.rename(columns={'ID':'ForecastId','cases':'ConfirmedCases','deaths':'Fatalities'})
pred_new = pred_new.set_index(pred_new['ForecastId'])
pred_new = pred_new[['ConfirmedCases','Fatalities']]
submission = pred_new.to_csv('submission.csv')
