import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#read 10Y minus 2Y to df: T10Y2Y



T10Y2Y = pd.read_csv("../input/T10Y2Y.csv", index_col = 0, parse_dates = True, skip_blank_lines = True, dtype= {'T10Y2Y': np.float64}, na_values ='.')



#read USREC  to df: USREC

USREC = pd.read_csv("../input/USREC.csv", index_col = 0, parse_dates = True, skip_blank_lines = True, dtype= {'USREC': np.float64}, na_values ='.')



#Inner join T10Y2Y, USREC: merged

merged = T10Y2Y.join(USREC, how = 'inner')

merged.USREC =((merged.USREC)*(np.max(merged.T10Y2Y)-np.min(merged.T10Y2Y))+np.min(merged.T10Y2Y))



#creates axis at y=0

merged['axis'] = pd.Series([0 for x in range(len(merged.index))], index=merged.index)



#plot USREC and 10TY2Y

merged.T10Y2Y.plot(y='2Y Treasury Yields', title = 'Inversions as Leading Indicator of Recession', legend = True)

merged.USREC.plot(kind ='line', y= 'Recessions', legend = True)

merged.axis.plot(kind='line')

plt.legend()

plt.show()

#read real gdp (GDPC1) to df: gdp

gdp = pd.read_csv("../input/GDPC1.csv", index_col = 0, parse_dates = True, skip_blank_lines = True, dtype= {'GDPC1': np.float64}, na_values ='.')

#read 2Y treasury yields (DGS2) to df: Two

Two = pd.read_csv("../input/DGS2.csv", index_col = 0, parse_dates = True, skip_blank_lines = True, dtype= {'DGS2': np.float64}, na_values ='.')



#plot gdp

gdp.plot()

plt.ylabel('Real GDP')

plt.title(label = 'GDPC1')

plt.show()



#plot DGS2

Two.plot()

plt.ylabel('2Y Bond Yield %')

plt.title(label = 'DGS2')

plt.show()


#calculate percent change between quarters of GDPC1

gdp['GDPC1_pct_change'] = gdp['GDPC1'].pct_change()



#create empty numpy array to recieve quarterly means for 2Y treasury yields

mean_two = np.empty(len(gdp))



#convert 2Y treasury yields from daily values to quarterly values

for i in range(len(gdp)-1):

    #for each quarter in gdp return the mean value of the 2Y treasury yields 

    mask = ((Two.index > gdp.index[i]) & (Two.index <= gdp.index[i+1]))

    mean_two[i]  = np.nanmedian(Two.DGS2[mask])

    



features = pd.DataFrame({'DGS2':mean_two}, index = gdp.index)

    

#take first difference of treasury yields



features['DGS2_first_diff']= features['DGS2'].diff()

features = features[:-1]



#plot gdp

gdp['GDPC1_pct_change'].plot()

plt.ylabel('GDPC1_pct_change')

plt.title(label = 'GDPC1 %change')

plt.show()



#plot DGS2

features['DGS2_first_diff'].plot()

plt.ylabel('DGS2_first_diff')

plt.title('1st diff. of DGS2 Aggregated Quarterly')

plt.show()
#imports

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import kpss



#test GDPC1 for non-stationary trends using augmented dickie fuller test

X = gdp.GDPC1_pct_change.dropna().values



result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

	print('\t%s: %.3f' % (key, value))

    

result = kpss(X)

print('KPSS Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[3].items():

	print('\t%s: %.3f' % (key, value))



#test DGS2 for non-stationary trends using augmented dickie fuller test

X = features.DGS2_first_diff.dropna().values



result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

	print('\t%s: %.3f' % (key, value))

    

result = kpss(X)

print('KPSS Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[3].items():

	print('\t%s: %.3f' % (key, value))
#Take a second difference of both time-series

gdp['GDPC1_pct_change_diff']=gdp['GDPC1_pct_change'].diff()

features['DGS2_second_diff'] = features['DGS2_first_diff'].diff()



#plot gdp

gdp.GDPC1_pct_change_diff.plot()

plt.ylabel('GDPC1_pct_change_diff')

plt.title(label= '1st diff. of GDPC1 %change')

plt.show()



#plot DGS2

features.DGS2_second_diff.plot()

plt.ylabel('DGS2_second_diff')

plt.title(label ='2nd diff. of DGS2 Aggregated Quarterly')

plt.show()



#test GDPC1 for non-stationary trends using augmented dickie fuller test

X = gdp.GDPC1_pct_change_diff.dropna().values



result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

	print('\t%s: %.3f' % (key, value))

    

result = kpss(X)

print('KPSS Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[3].items():

	print('\t%s: %.3f' % (key, value))





#imports

from sklearn.linear_model import LinearRegression

from scipy.stats import pearsonr



column = 'GDPC1_pct_change_diff'

column1 = 'DGS2_second_diff'



#merge gdp with features, drop rows with nans

merged = pd.concat([gdp[column], features[column1].loc[:'2009']], axis =1).dropna()



#fit linear model to data

model = LinearRegression()

X = merged[column1].values.reshape(-1,1)

y = merged[column].values.reshape(-1,1)

model.fit(X, y)



#calculate trend line and correlation coefficient

trend = model.predict(X)

r= pearsonr(X, y)[0][0]



#create scatter plot of GDP and DGS2

plt.scatter(X, y)

plt.plot(X, trend)

plt.xlabel(column1)

plt.ylabel(column)

plt.annotate(s= 'r = '+ str(r)[0:6], xy = [5,5], xycoords = 'axes points')

plt.title('GDPC1 vs. DGS2')

plt.show()

print('r-squared : {}'.format(model.score(X,y)))
from statsmodels.tsa.arima_model import ARMA

column = 'GDPC1_pct_change_diff'

column1 = 'DGS2_second_diff'

X=merged[column].resample('Q').mean().dropna()



#Fit the data to an AR(p) for p = 0,...,6 , and save the BIC

BIC = np.zeros(10)

for p in range(10):

    mod = ARMA(X, order=(p,0), freq = 'Q')

    res = mod.fit()

    

# Save BIC for AR(p)    

    BIC[p] = res.bic

    

# Plot the BIC as a function of p

plt.plot(range(1,10), BIC[1:10], marker='o')

plt.xlabel('Order of AR Model')

plt.ylabel('Bayesian Information Criterion')

plt.title(label = 'BIC vs. AR Order (GDPC1)')

plt.show()



X=merged[column1].resample('Q').mean().dropna()



#Fit the data to an AR(p) for p = 0,...,6 , and save the BIC

BIC = np.zeros(10)

for p in range(10):

    mod = ARMA(X, order=(p,0),freq = 'Q')

    res = mod.fit()

    

# Save BIC for AR(p)    

    BIC[p] = res.bic

    

# Plot the BIC as a function of p

plt.plot(range(1,10), BIC[1:10], marker='o')

plt.xlabel('Order of AR Model')

plt.ylabel('Bayesian Information Criterion')

plt.title(label = 'BIC vs. AR Order (DGS2)')

plt.show()
from statsmodels.tsa.arima_process import ArmaProcess



#create ARMA models

X=merged[column].dropna().resample('Q').mean()

X1=merged[column1].dropna().resample('Q').mean()

mod = ARMA(X, order = (2,0))

results =mod.fit()

mod = ARMA(X1, order = (2,0))

results1 =mod.fit()



#create ARMA process objects for simulating data

AR_object = ArmaProcess().from_estimation(results)

AR_object1 = ArmaProcess().from_estimation(results1)



#create empty arrays for storing r-squared values

rs =np.empty(1000)

rg = np.empty(1000)

rt= np.empty(1000)



#simulate 1000 series with similar autoregressive properties to actual data

for i in range(1000):

    

    simulated_data = AR_object.generate_sample(nsample= len(merged))

    simulated_data1 = AR_object1.generate_sample(nsample= len(merged))

    r=pearsonr(simulated_data, simulated_data1)[0]

    rs[i] = r**2



    r =pearsonr(X, simulated_data1)[0]

    rg[i] = r**2

    

    r =pearsonr(X1, simulated_data)[0]

    rt[i] = r**2

    

#calculate p-values for r-squared values greater than observed value    

p = np.sum(rs >=.3948**2)/len(rs)

pg = np.sum(rg >=.3948**2)/len(rg)

pt = np.sum(rt >=.3948**2)/len(rt)

print('p-value of simulated data series having higher correlation than observed: {}'.format(str(p)))

print('p-value of simulated gdp vs real DGS2 havign higher correlation than observed: {}'.format(str(pg)))

print('p-value of simulated DGS2 vs real GDPC1 having higher correlation than observed: {}'.format(str(pt)))


#import TimeSeriesSplit

from sklearn.model_selection import TimeSeriesSplit





#save column names for use as y, X: column, column 1

column = 'GDPC1_pct_change_diff'

column1 = 'DGS2_second_diff'





#merge features with gdp: merged

merged = pd.concat([gdp, features.loc[:'2009']], axis =1).dropna()



#instantiate linear model: est

est = LinearRegression()



#instantiate TimeSeriesSplit cross validator with 5 splits: cv

n_splits=5

cv= TimeSeriesSplit(n_splits)



#define X and y 

y = merged[column]

X= merged[column1]



# instantiate numpy arrays to store actual and predicted ys, pearsonr for each split

ys =[]

preds =[]



rs = []

p1=plt.subplot(2,1,1)

p2 =plt.subplot(212)



#iterate through each data split, i is iterateion number, tr are indexes for training data, tst are indexes for test data

for i, (tr,tst) in enumerate(cv.split(X,y)):

    

    #plot timeseries split behavior

    

    p1.plot(tr,np.full(shape = len(tr), fill_value = i),lw=6, c= 'red')

    p1.plot(tst,np.full(shape = len(tst), fill_value = i),lw=6, c = 'blue')

    

    

    #fit Linear Regression model    

    est.fit(X.iloc[tr].values.reshape(-1,1), y.iloc[tr].values.reshape(-1,1))

    

    #predict unseen test data for first difference of %change

    y_pred = est.predict(X.iloc[tst].values.reshape(-1,1))

    

    #append pearson r on period to rs

    rs.append(pearsonr(y_pred.squeeze(), y.iloc[tst])[0]) 

    

    

    #append actual ys and predicted ys to ys, preds

    ys.append(y.iloc[tst])

    preds.append(y_pred)

    

    #plot predicted GDPC1 %change first difference vs. actual %change first difference

  

    p2.plot(X.index[tst], y_pred, c = 'red')

    p2.plot(X.index[tst], y.iloc[tst], c= 'blue', alpha=.5)

    

    

        

#labels and legends for first and second subplots

p1.set(xlabel ='Data Index')

p1.set(ylabel = 'CV Iteration')

p1.set(title = 'Time Series Split Behavior')

p2.set(ylabel = '%change diff')

p1.legend(labels =['Training', 'Validation'])

p2.legend(labels = ['predicted', 'actual'])

plt.xlabel('Data Index')

plt.show()



#reshape data to be fit for trend line, 

for i in range(n_splits):

    plt.scatter(preds[i], ys[i], label =i)    

preds= np.squeeze(preds)

ps =preds[0]

ac=ys[0]

for i in range(1,n_splits):

    ps = np.hstack((ps,preds[i]))

    ac= np.hstack((ac,ys[i]))     

preds =ps

ys =ac



#fit trend line to predicted vs. actual 

model = LinearRegression()

model.fit(X=preds.reshape(-1,1), y=ys.reshape(-1,1))

trend = model.predict(preds.reshape(-1,1))



#calculate pearson r value for predictions vs. actual

r= pearsonr(preds.reshape(-1,1), ys.reshape(-1,1))[0][0]



#set labels,legend

plt.xlabel(xlabel= '%change diff predicted')

plt.ylabel(ylabel = '%change diff actual')

plt.legend()



#plot trend line

plt.plot(preds.reshape(-1,1), trend)



#annotate plot with pearson r value

plt.annotate(s= 'r = '+ str(r)[0:6], xy = [5,5], xycoords = 'axes points')

plt.title(label ='Actual Values vs. Predicted Values')



#show plot

plt.show()



#print pearson r for each period individually

for i in range(len(rs)):

    print('r of '+str(i)+": {}".format(str(rs[i])))

#import TimeSeriesSplit

from sklearn.model_selection import TimeSeriesSplit





#create new column with percent change lags

gdp['GDPC1_pct_change_lagged'] = gdp['GDPC1_pct_change'].shift()





#save column names for use as y, X: column, column 1

column = 'GDPC1_pct_change_diff'

column1 = 'DGS2_second_diff'





#merge features with gdp: merged

merged = pd.concat([gdp, features.loc[:'2009']], axis =1).dropna()



#instantiate linear model: est

est = LinearRegression()



#instantiate TimeSeriesSplit cross validator with 5 splits: cv

n_splits=5

cv= TimeSeriesSplit(n_splits)



#define X and y 

y = merged[column]

X= merged[column1]



# instantiate numpy arrays to store actual and predicted ys for each split

y1s=[]

preds1=[]



rs = []



p1=plt.subplot(2,1,1)

p2 =plt.subplot(212)



#iterate through each data split, i is iterateion number, tr are indexes for training data, tst are indexes for test data

for i, (tr,tst) in enumerate(cv.split(X,y)):

    

    #plot timeseries split behavior

    

    p1.plot(tr,np.full(shape = len(tr), fill_value = i),lw=6, c= 'red')

    p1.plot(tst,np.full(shape = len(tst), fill_value = i),lw=6, c = 'blue')

    

    #fit Linear Regression model    

    est.fit(X.iloc[tr].values.reshape(-1,1), y.iloc[tr].values.reshape(-1,1))

    

    #predict unseen test data for first difference of %change

    y_pred = est.predict(X.iloc[tst].values.reshape(-1,1))

    

    #Use lagged %change and predictions for difference of %change to predict current %change

    y_pred1 = y_pred + merged.GDPC1_pct_change_lagged.iloc[tst].values.reshape(-1,1)

    

    #append actual ys and predicted ys to y1s, and preds1

    

    preds1.append(y_pred1)

    y1s.append(merged.GDPC1_pct_change.iloc[tst])

    

    #append pearson r on period to rs

    rs.append(pearsonr(y_pred1.squeeze(), merged.GDPC1_pct_change.iloc[tst])[0]) 

    

    

    

    #plot predicted GDPC1 %change vs. actual GDPC1 %change

    

    p2.plot(X.index[tst], y_pred1, c = 'red')

    p2.plot(X.index[tst], merged.GDPC1_pct_change.iloc[tst], c= 'blue', alpha=.5)

        

#labels and legends for first and second subplots

p1.set(xlabel ='Data Index')

p1.set(ylabel = 'CV Iteration')

p1.set(title='Time Series Split Behavior')

p2.set(ylabel = 'GDP_pct_change')

p1.legend(labels =['Training', 'Validation'])

p2.legend(labels = ['predicted', 'actual'])

plt.xlabel('Data Index')

plt.show()





#reshape data to be fit for trend line,

for i in range(n_splits):

    plt.scatter(preds1[i], y1s[i], label =i)  

preds1= np.squeeze(preds1)

ps =preds1[0]

ac=y1s[0]

for i in range(1,n_splits):

    ps = np.hstack((ps,preds1[i]))

    ac= np.hstack((ac,y1s[i]))   

preds =ps

ys =ac



#fit trend line to predicted vs. actual

model = LinearRegression()

model.fit(X=preds.reshape(-1,1), y=ys.reshape(-1,1))

trend = model.predict(preds.reshape(-1,1))



#calculate pearson r value for predictions vs. actual

r= pearsonr(preds.reshape(-1,1), ys.reshape(-1,1))[0][0]



#set labels, legend, annotate with pearson r, title

plt.title(label='Actual Values vs. Predicted Values')

plt.xlabel(xlabel= 'GDPC1_%change_predicted')

plt.ylabel(ylabel = 'GDPC1_%change_actual')

plt.annotate(s= 'r = '+ str(r)[0:6], xy = [5,5], xycoords = 'axes points')

plt.legend()



#plot trend line 

plt.plot(preds.reshape(-1,1), trend)



#show plot

plt.show()



#print pearson r for each period individually

for i in range(len(rs)):

    print('r of '+str(i)+": {}".format(str(rs[i])))



#plot data vs. lag(1) to see how well lags predict GDPC1 %change



#fit trend line

model.fit(X=merged.GDPC1_pct_change_lagged.values.reshape(-1,1), y=merged.GDPC1_pct_change.values.reshape(-1,1))

trend = model.predict(merged.GDPC1_pct_change_lagged.values.reshape(-1,1))



#calculate pearson r value

r= pearsonr(merged.GDPC1_pct_change_lagged.values.reshape(-1,1), merged.GDPC1_pct_change.values.reshape(-1,1))[0][0]



#plot GDPC1 %change vs. lagged GDPC1 %change

plt.scatter(merged.GDPC1_pct_change_lagged, merged.GDPC1_pct_change)



#plot trend line

plt.plot(merged.GDPC1_pct_change_lagged, trend)



#annotate plot with pearson r value, set axis labels, title

plt.annotate(s= 'r = '+ str(r)[0:6], xy = [5,5], xycoords = 'axes points')

plt.xlabel(xlabel= 'GDPC1_%change_lagged')

plt.ylabel(ylabel = 'GDPC1_%change')

plt.title(label= 'Lag 1 Correlation of GDPC1 %Change')



#show plot

plt.show()





#merge features with gdp: merged

merged = pd.concat([gdp, features.loc[:'2009']], axis =1).dropna()





#determine 75th percentile for real gdp percent change

percentile_val = np.percentile(a=merged.GDPC1_pct_change, q=[75])





#create empty array to store labels for LogisticRegression model

LR_labels = np.empty(shape = len(merged))



#assign values 0,1 base on whether real gdp %change is above or below 75th percentile value



for i in range(len(merged.index)):

    if merged.GDPC1_pct_change.iloc[i]<percentile_val[0]:

        LR_labels[i] = 0

    else:

        LR_labels[i] = 1





merged['LR_labels'] = LR_labels

#import modules

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score









cv= TimeSeriesSplit(n_splits=5,  max_train_size=None)



column = 'LR_labels'

column1 = 'DGS2_first_diff'



preds=[]

ys=[]

f1s=[]





#define X and y 

y = merged[column]

X= merged[column1]



#instantiate LogisticRegression model est1



est1 =LogisticRegression(class_weight = 'balanced', solver = 'lbfgs')



#iterate through each data split, i is iterateion number, tr are indexes for training data, tst are indexes for test data

for i, (tr,tst) in enumerate(cv.split(X,y)):





    #fit logistic regression model with predictions of est for training data

    est1.fit(X.iloc[tr].values.reshape(-1,1), y.iloc[tr].values.reshape(-1,1).ravel())

    

    

    #predict percentile with logistic regression

    y_pred = est1.predict(X.iloc[tst].values.reshape(-1,1))

    

    #calculate period f1 score: f1

    f1= f1_score(y.iloc[tst].values.reshape(-1,1),y_pred, average ='binary')

    

    #append predicted values, actual values and f1 scores to preds, ys and f1s

    preds.append(y_pred)

    ys.append(y.iloc[tst].values.reshape(-1,1))

    f1s.append(f1)

    

    #plot predictions

    #plt.scatter(y_pred1, (merged.percentiles.iloc[tst] + est1.predict(y_pred1))/2, label = i)

    plt.scatter(X.iloc[tst], (.25*y.iloc[tst] +.75*y_pred), label = i) 







npy = ys[0]

npyp =preds[0]



for i in range(1,len(ys)):

    npy=np.hstack((npy.ravel(), np.ravel(ys[i])))

    npyp = np.hstack((npyp.ravel(), np.ravel(preds[i])))



f1 = f1_score(npy,npyp, average ='binary')

    



#annotate plot with pearson r value, set y-ticks

plt.annotate(s= 'f1 = '+ str(f1)[0:6], xy = [270,5], xycoords = 'axes points')

plt.legend()

plt.yticks(ticks=[0,.25,.75,1], labels =['true neg.', 'false neg.','false pos.', 'true pos.'])

plt.title(label= 'Logistic Regression Performance')

plt.show()



#print f1 scores for each period individually

for i in range(len(f1s)):

    print('f1 of '+str(i)+": {}".format(str(f1s[i])))
#imports

from numpy.random import binomial



#create numpy array to store simulated f1 scores

f1_scores=np.empty(shape=1000)



# use for loop to simulate 1000 f1 scores using a binomial distribution

for i in range(1000):

    bs = binomial(n=1, p=.25, size=len(npy))

    f1_scores[i] = f1_score(npy, bs, average ='binary')



#calculate fraction of simulated f1 scores greater than that observed

p_val = sum(f1_scores>f1)/len(f1_scores)



#print results

print("fraction of simulated data that had f1_score greater than observed: {}".format(str(p_val)))

print("observed f1-score: {}".format(str(f1)))

print("average f1_score of simulations: {}".format(str(np.mean(f1_scores))))


#define columns to use

column = 'GDPC1_pct_change'

column1 = 'DGS2_1st_diff'

column2 = 'T10Y2Y_1st_diff'



#create list to store timespans over which to resample data

spans = ['Q', '2Q', 'Y', '2Y', '3Y']





#use for loop to create subplots for each timespan

for i in range(len(spans)):

    

    #concatenate and resample data for the given timespan in spans

    merged = pd.concat([gdp, Two, T10Y2Y], axis =1).resample(spans[i], label = 'left').mean().loc[:'2009']  

    

    #create column for real gdp growth rate

    merged[column] = merged.GDPC1.pct_change()

    

    #create column to store firt difference of DGS2

    merged[column1] = merged.DGS2.diff()

    

    #create column to store the first difference of T10Y2Y

    merged[column2] = merged.T10Y2Y.diff()

    

    #drop rows with nan values

    merged = merged.dropna()



    #instantiate linear model   

    model = LinearRegression()

    

    #process data for fitting of linear model

    X = merged[column1].values.reshape(-1,1)

    X1 = merged[column2].values.reshape(-1,1)

    y = merged[column].values.reshape(-1,1)

    

    #fit linear model to DGS2_1st_diff vs. GDPC1_pct_change

    model.fit(X, y)



    #calculate trend line and correlation coefficient for the data

    trend = model.predict(X)

    r= pearsonr(X, y)[0][0]

    

    #fit linear model to T10Y2Y_1st_diff vs. GDPC1_pct_change

    model.fit(X1, y)

    trend1 = model.predict(X1)

    r1= pearsonr(X1,y)[0][0]

    



    #create subplot plot of DGS2_1st_diff vs. GDPC1_pct_change

    p1 = plt.subplot(1,2,1)

    plt.scatter(X, y)

    plt.plot(X, trend)

    plt.xlabel(column1)

    plt.ylabel(column)

    plt.annotate(s= 'r = '+ str(r)[0:6], xy = [5,5], xycoords = 'axes points')

    plt.annotate(s= 'r-sq = '+ str(r**2)[0:6], xy = [75, 5], xycoords = 'axes points')

    plt.title('GDPC1 vs. DGS2 by {}'.format(spans[i]))

    

    #create subplot plot of T10Y2Y_1st_diff vs. GDPC1_pct_change

    p2= plt.subplot(122, sharey=p1)

    plt.scatter(X1, y)

    plt.plot(X1, trend1)

    plt.xlabel(column2)

    plt.annotate(s= 'r = '+ str(r1)[0:6], xy = [5,5], xycoords = 'axes points')

    plt.annotate(s= 'r-sq = '+ str(r1**2)[0:6], xy = [75,5], xycoords = 'axes points')

    plt.title('GDPC1 vs. T10Y2Y by {}'.format(spans[i]))

    plt.show()



    
column = 'GDPC1_pct_change'

column1 = 'T10Y2Y_1st_diff'

rule ='2Q'



merged = pd.concat([gdp, T10Y2Y], axis =1).resample(rule =rule, label = 'left').mean().loc[:'2009']  

merged[column] = merged.GDPC1.pct_change()

merged[column1] = merged.T10Y2Y.diff()

merged = merged.dropna()



merged[column] = (merged[column] -merged[column].min())/(merged[column].max()- merged[column].min())

merged[column1]= (merged[column1]- merged[column1].min())/(merged[column1].max()- merged[column1].min())

merged[column].plot()

merged[column1].plot()

plt.ylabel('normalized movement')

plt.title('Aggregatd over {}'.format(rule))



plt.legend()

plt.show()


#define columns to use

column = 'GDPC1_pct_change'

column1 = 'T10Y2Y_first_diff'





#concatenate and resample data for the given timespan in spans

merged = pd.concat([gdp,  T10Y2Y], axis =1).resample('2Q', label = 'left').mean().loc[:'2009']  



#create column for real gdp growth rate

merged[column] = merged.GDPC1.pct_change()





#create column to store the first difference of T10Y2Y

merged[column1] = merged.T10Y2Y.diff()



#drop rows with nan values

merged = merged.dropna()





#create ARMA models

X=merged[column].dropna().resample('2Q').mean()

X1=merged[column1].dropna().resample('2Q').mean()

mod = ARMA(X, order = (6,0))

results =mod.fit()

mod = ARMA(X1, order = (1,0))

results1 =mod.fit()



#create ARMA process objects for simulating data

AR_object = ArmaProcess().from_estimation(results)

AR_object1 = ArmaProcess().from_estimation(results1)



#create empty arrays for storing r-squared values

rs =np.empty(10000)

rg = np.empty(10000)

rt= np.empty(10000)



#cacualte observed r

r_obs = pearsonr(X, X1)[0]



#simulate 1000 series with similar autoregressive properties to actual data

for i in range(10000):

    

    simulated_data = AR_object.generate_sample(nsample= len(X))

    simulated_data1 = AR_object1.generate_sample(nsample= len(X))

    

    r=pearsonr(simulated_data, simulated_data1)[0]

    rs[i] = r**2



    r =pearsonr(X, simulated_data1)[0]

    rg[i] = r**2

    

    r1 =pearsonr(X1, simulated_data)[0]

    rt[i] = r1**2

    

#calculate p-values for r-squared values greater than observed value    

p = np.sum(rs >=r_obs**2)/len(rs)

pg = np.sum(rg >=r_obs**2)/len(rg)

pt = np.sum(rt >=r_obs**2)/len(rt)

print('p-value of simulated data series having higher correlation than observed: {}'.format(str(p)))

print('p-value of simulated gdp vs real T10Y2Y havign higher correlation than observed: {}'.format(str(pg)))

print('p-value of simulated T10Y2Y vs real GDPC1 having higher correlation than observed: {}'.format(str(pt)))