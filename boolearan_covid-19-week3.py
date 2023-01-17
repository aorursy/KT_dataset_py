import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from datetime import date
import math
# Read data
data_test = pd.read_csv('test.csv')
data_test_spain = data_test.loc[data_test.Country_Region == 'Spain']
#data_test_spain
# Read data
data = pd.read_csv('train.csv')
data_grouped = data.groupby(['Country_Region', 'Date'], as_index=False).sum()
data_china_dia = data_grouped
data_china_dia['GrowRateConf'] = (data_china_dia.ConfirmedCases.pct_change()) +1
data_china_dia['GrowRateFat'] = (data_china_dia.Fatalities.pct_change()) +1
data_china_dia = data_china_dia.loc[data_china_dia.Country_Region == 'China']
data_china_dia = data_china_dia.sort_values(['Date'])
data_china_dia['dia_contagio'] = range(53, len(data_china_dia)+53)
data_china_dia = data_china_dia.drop(2592)
data_china_dia.head(10)
data_spain_dia = data_grouped
data_spain_dia['GrowRateConf'] = (data_spain_dia.ConfirmedCases.pct_change()) +1
data_spain_dia['GrowRateFat'] = (data_spain_dia.Fatalities.pct_change()) +1
data_spain_dia = data_spain_dia.loc[data_spain_dia.Country_Region == 'Spain']
data_spain_dia = data_spain_dia.sort_values(['Date'])
data_spain_dia = data_spain_dia.loc[data_spain_dia.Date>='2020-02-01']
data_spain_dia['dia_contagio'] = range(0, len(data_spain_dia))
data_spain_dia
data_spain_54 = data_spain_dia.loc[data_spain_dia.dia_contagio>= 54]
data_concatenado = pd.merge(data_spain_54, data_china_dia[['GrowRateConf', 'GrowRateFat', 
                                                           'dia_contagio', 'Country_Region']], 
                            on='dia_contagio', how='right')
data_concatenado.head(20)
for i in range(8, len(data_concatenado)):
    data_concatenado.loc[i,'ConfirmedCases'] = data_concatenado.loc[i-1,'ConfirmedCases'] \
                                                * data_concatenado.loc[i,'GrowRateConf_y']
    data_concatenado.loc[i,'Fatalities'] = data_concatenado.loc[i-1,'Fatalities'] \
                                                * data_concatenado.loc[i,'GrowRateFat_y']
data_concatenado      
        
data_merge = pd.merge(data_concatenado[['Date', 'ConfirmedCases', 'Fatalities', 'dia_contagio', 'GrowRateConf_y']], 
                      data_test_spain[['Date']], on='Date', how='right')

data_merge['ConfirmedCases'] = data_concatenado.iloc[range(0, len(data_merge)), 3]
data_merge['Fatalities'] = data_concatenado.iloc[range(0, len(data_merge)), 4]
data_merge['GrowRateConf_y'] = data_concatenado.iloc[range(0, len(data_merge)), 8]
data_merge['dia_contagio'] = data_concatenado.iloc[range(0, len(data_merge)), 7]
data_merge.columns = ['Date', 'ConfirmedCases', 'Fatalities', 'Dia_contagio', 'Factor']
data_merge
data_concatenado.ConfirmedCases.plot()
data_concatenado.Fatalities.plot()
data_escogida = data_merge[['ConfirmedCases','Fatalities']]
data_escogida.plot(figsize=(20,5))
# Get dimensions
data_escogida.shape
# Plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(7,3))
for i, ax in enumerate(axes.flatten()):
    data = data_escogida[data_escogida.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(data_escogida.columns[i])
    ax.xaxis.set_ticks_position('none')
    #plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.set_xticks(data.index[::4])
    ax.set_xticklabels(data.index[::4], rotation=67)
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=5)

plt.tight_layout();
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix(data_escogida, variables = data_escogida.columns)
plt.matshow(data_escogida.corr())
lq = 12*(71/100)**(1/4)
def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,round(lq))
    d = {'0.9':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(data_escogida)
nobs = 10
data_escogida_train, data_escogida_test = data_escogida[0:-nobs], data_escogida[-nobs:]

# Check size
print(data_escogida_train.shape)
print(data_escogida_test.shape)
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")
for name, column in data_escogida_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')
def weird_log(d):
    return math.log(d) if d else 0

column_names = ["ConfirmedCases", "Fatalities"]
data_spain_train_log = pd.DataFrame(columns = column_names)

for i in data_spain_train.index:
    log_confirmedcases_aux = weird_log(data_spain_train.loc[i,'ConfirmedCases'])
    log_fatalities_aux = weird_log(data_spain_train.loc[i,'Fatalities'])
    data_spain_train_log = data_spain_train_log.append({'ConfirmedCases':log_confirmedcases_aux, 'Fatalities':log_fatalities_aux}, ignore_index = True)

data_spain_train_log.index = data_spain_train.index

#data_spain_train_log = data_spain_train_log.loc[data_spain_train_log.index>'2020-02-08', :]

data_spain_train_log.tail(50)
data_spain_train_log_differenced = data_spain_train_log.diff().dropna()
data_spain_train_log_differenced2 = data_spain_train_log_differenced.diff().dropna()
data_spain_train_log_differenced3 = data_spain_train_log_differenced2.diff().dropna()

data_spain_train_log_differenced3 = data_spain_train_log_differenced3.loc[data_spain_train_log_differenced3.index>'2020-02-24', :]

data_spain_train_log_differenced3.tail(50)
for name, column in data_spain_train_log_differenced3.iteritems():
    adfuller_test(column.loc[column>0], name=column.name)
    print('\n')
data_spain_train_log_differenced3.plot(figsize = (20, 5))
