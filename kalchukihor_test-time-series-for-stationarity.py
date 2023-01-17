def test_stationarity(timeseries, ADF_regression = 'c', ADF_autolag = 'AIC', KPSS_regression='c', skip_first = 0):
    
    # Assembled by Ihor Kalchuk, https://www.linkedin.com/in/ihor-k-803b0779/
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(13).mean()
    rolstd = pd.Series(timeseries).rolling(13).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform ADF (Augmented Dickey–Fuller) test:
    print('ADF (Augmented Dickey–Fuller) Stationarity Test.')
    
    print ('Null Hypothesis: The series has a unit root (value of a=1), time series is not stationary due to trend.')
    print ('ADF_regression value: ', ADF_regression)
    print ('ADF_autolag value: ', ADF_autolag)
    dftest = sm.tsa.adfuller(timeseries, regression = ADF_regression, autolag = ADF_autolag)
        # regression values: “c” : constant only (default).
        #                    “ct” : constant and trend.
        #                    “ctt” : constant, and linear and quadratic trend.
        #                    “nc” : no constant, no trend.
        # autolag values:    if None, then maxlag lags are used.
        #                    if “AIC” (default) or “BIC”, then the number of lags is chosen to minimize the corresponding information criterion.
        #                    “t-stat” based choice of maxlag. Starts with maxlag and drops a lag until the t-statistic on the last lag length is significant using a 5%-sized test.
        
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
        #Test for stationarity: If the test statistic is less than the critical value, 
        #we can reject the null hypothesis (aka the series is stationary). 
        #When the test statistic is greater than the critical value, 
        #we fail to reject the null hypothesis (which means the series is not stationary).
    if dftest[1]<=0.05:
        ADF_stat = 1
        print('p-value: ', dftest[1], '. Rejecting  Null Hypothesis.')
        print('Vote of Dickey-Fuller Test: Data IS Stationary.')
    else:
        ADF_stat = 0
        print('p-value: ', dftest[1], '. Accepting  Null Hypothesis.') 
        print('Vote of ADF Test: Data is NOT Stationary.')
        
    # Perform KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test
    print('')
    print('KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Stationarity Test.')
    print ('Null Hypothesis: The process is trend stationary.')
    print ('KPSS_regression value: ', KPSS_regression)
    print ('Results of KPSS Test:')
    from statsmodels.tsa.stattools import kpss
    kpsstest = kpss(timeseries, regression = KPSS_regression, nlags="auto") # 'c' : The data is stationary around a constant (default).
                                                                          # 'ct': The data is stationary around a trend.
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
        #Test for stationarity: If the test statistic is greater than the critical value, 
        #we reject the null hypothesis (series is not stationary). 
        #If the test statistic is less than the critical value, 
        #we fail to reject the null hypothesis (series is stationary).
    if kpsstest[1]<=0.05:
        KPSS_stat = 0
        print('p-value: ', kpsstest[1], '. Rejecting  Null Hypothesis.')
        print('Vote of KPSS Test: Data is NOT Stationary.')
    else:
        KPSS_stat = 1
        print('p-value: ', kpsstest[1], '. Accepting  Null Hypothesis.') 
        print('Vote of KPSS Test: Data IS Stationary.')
    
    # Conclusion on ADF and KPSS test results:
    #Case 1: Both tests conclude that the series is not stationary -> series is not stationary
    #Case 2: Both tests conclude that the series is stationary -> series is stationary
    #Case 3: KPSS = stationary and ADF = not stationary  -> trend stationary, remove the trend to make series strict stationary
    #Case 4: KPSS = not stationary and ADF = stationary -> difference stationary, use differencing to make series stationary

    print('') 
    print('Conclusion on ADF and KPSS test results:') 
    if ( ADF_stat == 0 and KPSS_stat == 0 ) :
        print('Analysis of both ADF and KPSS tests shows that ', 'series is NOT stationary.')
    if ( ADF_stat == 1 and KPSS_stat == 1 ) :
        print('Analysis of both ADF and KPSS tests shows that ', 'series IS stationary.')
    if ( ADF_stat == 0 and KPSS_stat == 1 ) :
        print('Analysis of both ADF and KPSS tests shows that ', 'series is trend stationary. Trend needs to be removed to make series strict stationary. Than the detrended series should be checked for stationarity again.')
    if ( ADF_stat == 1 and KPSS_stat == 0 ) :
        print('Analysis of both ADF and KPSS tests shows that ', 'series is linear or difference stationary. Differencing is to be used to make series stationary. Than the differenced series should be checked for stationarity again.')
        
    
    #Perform check by mean and variance after dividing the time series in 2 sections:
    print('')
    print('Check by mean and variance after dividing the time series in 2 sections:')
    print('skip_first value: ', skip_first)  
    #skip_first = 20 # to ignore fluctuations at the beginning of the series
    split = round((len(timeseries) - skip_first) / 2) + skip_first
    X1, X2 = timeseries[skip_first:split], timeseries[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    data = []
    data.append([mean1, var1])
    data.append([mean2, var2])
    df = pd.DataFrame(data, index=[0, 1], columns=['Means', 'Variances'])
    print(df)

    
     #Perform check by mean and variance after dividing the time series in 3 sections:
    print('')
    print('Check by mean and variance after dividing the time series in 3 sections:')
    one, two, three = np.split(
        timeseries.sample(
        frac=1), [int(.33*len(timeseries)),
        int(.67*len(timeseries))])
    mean1, mean2, mean3 = one.mean(), two.mean(), three.mean()
    var1, var2, var3 = one.var(), two.var(), three.var()
    data = []
    data.append([mean1, var1])
    data.append([mean2, var2])
    data.append([mean3, var3])
    df = pd.DataFrame(data, index=[0, 1, 2], columns=['Means', 'Variances'])
    print(df)
test_stationarity(DataFrame['FieldName'])