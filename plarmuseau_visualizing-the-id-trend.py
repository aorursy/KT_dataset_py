import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#read data

train = pd.read_csv('../input/train.csv')

print(len(train))

train['counter']=0

train['group']=0

tel=0

uur=0

for xi in range(len(train)):

    tel+=1

    train.ix[xi,'counter']=tel

    train.ix[xi,'group']=uur

    if xi/175.375-uur>1:

        tel=0

        uur+=1



group_sum=pd.pivot_table(train, values='y', index='group', columns='counter', aggfunc='sum') 

X5_sum=pd.pivot_table(train, values='y', index='X5', columns='counter', aggfunc='sum') 

X0_sum=pd.pivot_table(train, values='y', index='X0', columns='counter', aggfunc='sum')  

X0_count=pd.pivot_table(train, values='y', index='X0', columns='counter', aggfunc='count')  

X1_sum=pd.pivot_table(train, values='y', index='X1', columns='counter', aggfunc='sum')  

X1_count=pd.pivot_table(train, values='y', index='X1', columns='counter', aggfunc='count')  

X2_sum=pd.pivot_table(train, values='y', index='X2', columns='counter', aggfunc='sum')  

X2_count=pd.pivot_table(train, values='y', index='X2', columns='counter', aggfunc='count')  

X3_sum=pd.pivot_table(train, values='y', index='X3', columns='counter', aggfunc='sum')  

X3_count=pd.pivot_table(train, values='y', index='X3', columns='counter', aggfunc='count')  

X4_sum=pd.pivot_table(train, values='y', index='X4', columns='counter', aggfunc='sum')  

X4_count=pd.pivot_table(train, values='y', index='X4', columns='counter', aggfunc='count')  

X6_sum=pd.pivot_table(train, values='y', index='X6', columns='counter', aggfunc='sum')  

X6_count=pd.pivot_table(train, values='y', index='X6', columns='counter', aggfunc='count')  

X8_sum=pd.pivot_table(train, values='y', index='X8', columns='counter', aggfunc='sum')  

X8_count=pd.pivot_table(train, values='y', index='X8', columns='counter', aggfunc='count')  

import matplotlib.pyplot as plt

gcs=pd.DataFrame( group_sum.sum(axis=1) )

gcs.plot()

plt.show()

print(gcs)

print(gcs/gcs.mean())

# time per 36 cars... approx 3600seconds or 1 hour... and here you see a nice trend..
print(len(X5_sum))

gcs=pd.DataFrame( X5_sum.sum(axis=1) )

gcs.plot()

plt.show()

print('hours/X5',gcs/3600)

print(gcs/gcs.mean())

# time per 33type of X5... max 23000seconds /3600 second/hour => 6.4 hours worked per X5 group suppose this is a person;..

# unexplainable there are people doing only 1 test... or they are doing a reengineering-reparation ?

# its also the only category that follows the ID... so probably this is a working shift of one person



# another theorie: since we have 33models, suppose there are 29 models made and balanced over the production.. Usually the new models are peaking so it would be bizar this is topping this way.

print(len(X0_sum))

gcs=pd.DataFrame( X0_sum.sum(axis=1) )

gcs2=pd.DataFrame( X0_count.sum(axis=1) )

timepertypeofcar=(gcs/gcs2)

timepertypeofcar.plot()

minutespertypeofcar2=(gcs/60)

minutespertypeofcar2.plot()

plt.show()

print('hours/X0',gcs/3600)

print(gcs/gcs2)

# time per X0 category... suppose this is a type of car... there is only one type of car taking 150seconds on average, the rest is 110seconds ore 90seconds

#this collides with the peaks we have in the forecast. 

#But there are only 33 type of cars and nothing has 47 types...  

#except if you count every engine  - model combination as a car. So probably this collides with the models

# the peaky behaviour of the number of time spend on the different models could collid with this.

# the dominant behaviour of XO in the stats could explain equally this relation with the models
print(len(X1_sum))

gcs=pd.DataFrame( X1_sum.sum(axis=1) )

gcs2=pd.DataFrame( X1_count.sum(axis=1) )

timepertypeofcar=(gcs/gcs2)

minutspertypeofcar=(gcs/60)

timepertypeofcar.plot()

minutspertypeofcar.plot()

plt.show()

print(gcs)

print(gcs/gcs2)

# time per X1 category... average time per type of category X1

# this could be very well be the models... and if so this should 'correlate with X0

# or this are the 'engines'
print(len(X2_sum))

gcs=pd.DataFrame( X2_sum.sum(axis=1) )

gcs2=pd.DataFrame( X2_count.sum(axis=1) )

timepertypeofcar=(gcs/gcs2)

timepertypeofcar.plot()

plt.show()

print(gcs)

print(gcs/gcs2)

# time per X2 category... average time per type of category X2
print(len(X3_sum))

gcs=pd.DataFrame( X3_sum.sum(axis=1) )

gcs2=pd.DataFrame( X3_count.sum(axis=1) )

timepertypeofcar=(gcs/gcs2)

minutespertypeofcar=(gcs/60)

timepertypeofcar.plot()

minutespertypeofcar.plot()

plt.show()

print(gcs)

print(gcs/gcs2)

# time per X3 category... average time per type of category X3

# 7 types of shift gears /4matic combination ? an AMG should gear faster then a manual gear...
print(len(X4_sum))

gcs=pd.DataFrame( X4_sum.sum(axis=1) )

gcs2=pd.DataFrame( X4_count.sum(axis=1) )

timepertypeofcar=(gcs/gcs2)

minutespertypeofcar=(gcs/60)

timepertypeofcar.plot()

minutespertypeofcar.plot()

plt.show()

print(gcs)

print(gcs/gcs2)

# this is an exceptional category, 4 cars in abc, and the rest is d... i don't know what this is, but its not important
print(len(X6_sum))

gcs=pd.DataFrame( X6_sum.sum(axis=1) )

gcs2=pd.DataFrame( X6_count.sum(axis=1) )

timepertypeofcar=(gcs/gcs2)

minutespertypeofcar=(gcs/60)

timepertypeofcar.plot()

minutespertypeofcar.plot()

plt.show()

print(gcs)

print(gcs/gcs2)

# time per X6 category... average time per type of category X6

# what has 12 types and 3 types running good ? engine cores ?
print(len(X8_sum))

gcs=pd.DataFrame( X8_sum.sum(axis=1) )

gcs2=pd.DataFrame( X8_count.sum(axis=1) )

timepertypeofcar=(gcs/gcs2)

minutespertypeofcar=(gcs/60)

timepertypeofcar.plot()

minutespertypeofcar.plot()

plt.show()

print(gcs)

print(gcs/gcs2)

# time per X8category... average time per type of category X8

# 25 types, but very bisar balanced and swiping between 400 - 200 minutes.
import statsmodels.api as sm

from statsmodels.tsa.arima_process import arma_generate_sample

import datetime

from scipy.stats import norm

import statsmodels.api as sm



gcs.columns=['cars']

gcs['time']=datetime.datetime.now()

day_time=datetime.datetime.now()



for xi in range(len(gcs)):

    seconde=gcs.ix[xi,'cars']

    #print(seconde)

    day_time=day_time + datetime.timedelta(hours=0, minutes=0, seconds=seconde)

    gcs.ix[xi,'time']=day_time



print(gcs.head())



y__ = pd.Series(gcs['cars'].values, index=gcs.time)

if len(y__)>0:

    dta_full = y__

    aic_full = pd.DataFrame(np.zeros((6,6), dtype=float))

    warnings.simplefilter('ignore')



    # Iterate over all ARMA(p,q) models with p,q in [0,6]

    for p in range(6):

        for q in range(6):

            if p == 0 and q == 0:

                continue

            

            # Estimate the model with no missing datapoints

            mod = sm.tsa.statespace.SARIMAX(dta_full, order=(p,0,q), enforce_invertibility=False)

            try:

                res = mod.fit(disp=False)

                aic_full.iloc[p,q] = res.aic

            except:

                aic_full.iloc[p,q] = np.nan

        

    print(aic_full)

    mod = sm.tsa.statespace.SARIMAX(dta_full, order=(1,0,1))

    res = mod.fit(disp=False)

    print(res.summary())
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

#create ID series

train_test = train.append(test)

y_m=pd.DataFrame(train_test[['y','ID']])



        

#labelencode        

for c in train_test.columns:

    if train_test[c].dtype == 'object':

        tempt = train_test[['y',c]]

        temp=tempt.groupby(c).median().sort('y')

        templ=temp.index

        print(templ)

        aant=len(templ)

        train_test[c].replace(to_replace=templ, value=[x for x in range(0,aant)], inplace=True, method='pad', axis=1)

        

train_test_ma = pd.rolling_mean(train_test,window=175)

print(train_test_ma[174:])

print(train['y'])
import statsmodels.formula.api as sm

res = sm.ols(formula="y ~ ID +X0+X1+X2+X3+X4+X5+X6+X8 +X47", data=train_test_ma[174:]).fit()

print(res.summary())

import statsmodels.formula.api as sm

res = sm.ols(formula="y ~ X0+X3+X5", data=train_test_ma[174:]).fit()

print(res.summary())



print('Predicted values: ', res.predict())

beta=res.params



beta_=pd.DataFrame(beta)

beta_T=pd.DataFrame(beta.T)

print(beta_T)



# forecast trainingdata

X_=train_test[train_test['y']>0 ]

X_['Intercept']=1.0

X_r=X_[['Intercept','X0','X3','X5']]

print(X_r.shape)

print(beta_T.shape)

train['y_pred']=X_r.dot(beta_T)

#print(train[['y','y_pred']])

# look at the scatterplot, still some unforecasted trouble but nicer already

plt.scatter(train['y'],train['y_pred'])

plt.show()



test  = pd.read_csv('../input/test.csv')

Xt_=train_test[train_test['y'].isnull() ]

Xt_['Intercept']=1.0

Xt_r=Xt_[['Intercept','X0','X3','X5']]

print(Xt_r.head())

test['y_pred']=Xt_r.dot(beta_T)

prediction=test[['y_pred','ID']]

prediction.to_csv('submission_Arima_PL_parameter.csv', index=False)
import statsmodels.formula.api as sm

res = sm.ols(formula="y ~ X0+X1+X2+X3+X4+X5+X6+X8 +X47", data=train_test_ma[174:]).fit()

print(res.summary())

import statsmodels.formula.api as sm

res = sm.ols(formula="y ~ ID+X0 +X1 +X5 +X6 + X8 +X47 +X2 +X3 +X77+X105 +X345 +X3 +X142 +X26 +X322+X46+X267+X151+X240+X342+X287+X152+X140+X65+X95+X70+X116+X265+X354+X177+X362+X52+X383+X273+X58+X157+X156+X64+X131+X355+X173+X73+X31+X338+X225+X271+X230+X71+X174+X27+X163+X141+X327+X127+X51+X292", data=train_test_ma[174:]).fit()

print(res.summary())



print('Predicted values: ', res.predict())

beta=res.params



beta_=pd.DataFrame(beta)

beta_T=pd.DataFrame(beta.T)

print(beta_T)



# forecast trainingdata

X_=train_test[train_test['y']>0 ]

X_['Intercept']=1.0

X_r=X_[['Intercept','ID','X0','X1','X5','X6','X8','X47','X2','X3','X77','X105','X345','X142','X26','X322','X46','X267','X151','X240','X342','X287','X152','X140','X65','X95','X70','X116','X265','X354','X177','X362','X52','X383','X273','X58','X157','X156','X64','X131','X355','X173','X73','X31','X338','X225','X271','X230','X71','X174','X27','X163','X141','X327','X127','X51','X292']]

print(X_r.shape)

print(beta_T.shape)

train['y_pred']=X_r.dot(beta_T)

#print(train[['y','y_pred']])

# look at the scatterplot, still some unforecasted trouble but nicer already

plt.scatter(train['y'],train['y_pred'])

plt.show()



test  = pd.read_csv('../input/test.csv')

Xt_=train_test[train_test['y'].isnull() ]

Xt_['Intercept']=1.0

Xt_r=Xt_[['Intercept','ID','X0','X1','X5','X6','X8','X47','X2','X3','X77','X105','X345','X142','X26','X322','X46','X267','X151','X240','X342','X287','X152','X140','X65','X95','X70','X116','X265','X354','X177','X362','X52','X383','X273','X58','X157','X156','X64','X131','X355','X173','X73','X31','X338','X225','X271','X230','X71','X174','X27','X163','X141','X327','X127','X51','X292']]

print(Xt_r.head())

test['y_pred']=Xt_r.dot(beta_T)

prediction=test[['y_pred','ID']]

prediction.to_csv('submission_Arima_PL.csv', index=False)

kolom=train_test.columns

kolom=[k for k in kolom if k not in ['y']]

formul="y ~ 1"

for xi in kolom:

    formul+="+"+xi

    

print(formul)

res = sm.ols(formula=formul, data=train_test_ma[174:]).fit()

print(res.summary())



beta=res.params



beta_=pd.DataFrame(beta)

beta_T=pd.DataFrame(beta.T)

print(beta_T)



# forecast trainingdata

X_=train_test[train_test['y']>0 ]

X_['Intercept']=1.0

X_r=X_[['Intercept']+kolom]

print(X_r.shape)

print(beta_T.shape)

train['y_pred']=X_r.dot(beta_T)

#print(train[['y','y_pred']])

# look at the scatterplot, still some unforecasted trouble but nicer already

plt.scatter(train['y'],train['y_pred'])

plt.show()



Xt_=train_test[train_test['y'].isnull() ]

Xt_['Intercept']=1.0

Xt_r=Xt_[['Intercept']+kolom]

print(Xt_r.head())

test['y_pred']=Xt_r.dot(beta_T)

prediction=test[['y_pred','ID']]

prediction.to_csv('submission_Arima_PL2.csv', index=False)