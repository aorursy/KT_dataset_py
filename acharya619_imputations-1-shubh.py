# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn import metrics

data = pd.read_csv('/kaggle/input/rk-puram-ambient-air/rk.csv')

data['PM25'] = data['PM2.5']

data = data.drop('PM2.5', axis=1)

data
data = data.replace(0,float('nan'))
data = data[16091:]

data.isnull().sum()
data = data.drop('Toluene', axis=1)

data.shape
data['datetime'] = pd.to_datetime(data['From Date'], format='%d/%m/%Y %H:%M')
data = data.drop(['From Date', 'To Date'], axis=1)
data = data.set_index('datetime')
data['Hour'] = data.index.hour

data['Year'] = data.index.year

data['Month'] = data.index.month

data['Weekday'] = data.index.weekday_name

data.head()
data = data.drop('VWS', axis=1)

data.isnull().sum()
data_AT = pd.DataFrame(data['AT'])

a = data_AT.AT.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_AT[start:stop-1].shape)

print(data_AT[start:stop-1].isnull().sum())

data_AT = data_AT[start:stop-1]

data_AT['masked'] = data_AT['AT']

np.random.seed(120)

data_AT['masked'] = data_AT['masked'].mask(np.random.random(data_AT['masked'].shape) < .30)

data_AT = data_AT.assign(missing=np.nan)

data_AT.missing[data_AT.masked.isna()] = data_AT.AT

data_AT[['AT','masked','missing']] = data_AT[['AT','masked','missing']].astype(float)

data_AT.info()
data_AT.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))
# imputing using mean and median

data_AT = data_AT.assign(FillMean=data_AT.masked.fillna(data_AT.masked.mean()))

data_AT = data_AT.assign(FillMedian=data_AT.masked.fillna(data_AT.masked.median()))

# imputing using the rolling average

data_AT = data_AT.assign(RollingMean=data_AT.masked.fillna(data_AT.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_AT = data_AT.assign(RollingMedian=data_AT.masked.fillna(data_AT.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_AT = data_AT.assign(InterpolateLinear=data_AT.masked.interpolate(method='linear'))

data_AT = data_AT.assign(InterpolateTime=data_AT.masked.interpolate(method='time'))

data_AT = data_AT.assign(InterpolateQuadratic=data_AT.masked.interpolate(method='quadratic'))

data_AT = data_AT.assign(InterpolateCubic=data_AT.masked.interpolate(method='cubic'))

data_AT = data_AT.assign(InterpolateSLinear=data_AT.masked.interpolate(method='slinear'))

data_AT = data_AT.assign(InterpolateAkima=data_AT.masked.interpolate(method='akima'))

data_AT = data_AT.assign(InterpolatePoly5=data_AT.masked.interpolate(method='polynomial', order=5)) 

data_AT = data_AT.assign(InterpolatePoly7=data_AT.masked.interpolate(method='polynomial', order=7))

data_AT = data_AT.assign(InterpolateSpline3=data_AT.masked.interpolate(method='spline', order=3))

data_AT = data_AT.assign(InterpolateSpline4=data_AT.masked.interpolate(method='spline', order=4))

data_AT = data_AT.assign(InterpolateSpline5=data_AT.masked.interpolate(method='spline', order=5))
#missing data % in AT

data['AT'].isnull().sum()/data['AT'].shape[0]
results = [(method, metrics.r2_score(data_AT.AT, data_AT[method])) for method in list(data_AT)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

results_df.sort_values(by='R_squared', ascending=False)
data_new = data_AT[['AT', 'InterpolateAkima', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_BP = pd.DataFrame(data['BP'])

a = data_BP.BP.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_BP[start:stop-1].shape)

print(data_BP[start:stop-1].isnull().sum())

data_BP = data_BP[start:stop-1]

data_BP['masked'] = data_BP['BP']

np.random.seed(120)

data_BP['masked'] = data_BP['masked'].mask(np.random.random(data_BP['masked'].shape) < .30)

data_BP = data_BP.assign(missing=np.nan)

data_BP.missing[data_BP.masked.isna()] = data_BP.BP

data_BP[['BP','masked','missing']] = data_BP[['BP','masked','missing']].astype(float)

data_BP.info()
data_BP.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))
# imputing using mean and median

data_BP = data_BP.assign(FillMean=data_BP.masked.fillna(data_BP.masked.mean()))

data_BP = data_BP.assign(FillMedian=data_BP.masked.fillna(data_BP.masked.median()))

# imputing using the rolling average

data_BP = data_BP.assign(RollingMean=data_BP.masked.fillna(data_BP.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_BP = data_BP.assign(RollingMedian=data_BP.masked.fillna(data_BP.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_BP = data_BP.assign(InterpolateLinear=data_BP.masked.interpolate(method='linear'))

data_BP = data_BP.assign(InterpolateTime=data_BP.masked.interpolate(method='time'))

data_BP = data_BP.assign(InterpolateQuadratic=data_BP.masked.interpolate(method='quadratic'))

data_BP = data_BP.assign(InterpolateCubic=data_BP.masked.interpolate(method='cubic'))

data_BP = data_BP.assign(InterpolateSLinear=data_BP.masked.interpolate(method='slinear'))

data_BP = data_BP.assign(InterpolateAkima=data_BP.masked.interpolate(method='akima'))

data_BP = data_BP.assign(InterpolatePoly5=data_BP.masked.interpolate(method='polynomial', order=5)) 

data_BP = data_BP.assign(InterpolatePoly7=data_BP.masked.interpolate(method='polynomial', order=7))

data_BP = data_BP.assign(InterpolateSpline3=data_BP.masked.interpolate(method='spline', order=3))

data_BP = data_BP.assign(InterpolateSpline4=data_BP.masked.interpolate(method='spline', order=4))

data_BP = data_BP.assign(InterpolateSpline5=data_BP.masked.interpolate(method='spline', order=5))
#missing data % in BP(big)

print(data['BP'].isnull().sum()/data['BP'].shape[0])

results = [(method, metrics.r2_score(data_BP.BP, data_BP[method])) for method in list(data_BP)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

results_df.sort_values(by='R_squared', ascending=False)
#missing data % in BP(small)

print(data['BP'].isnull().sum()/data['BP'].shape[0])

results = [(method, metrics.r2_score(data_BP.BP, data_BP[method])) for method in list(data_BP)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

results_df.sort_values(by='R_squared', ascending=False)
data_new = data_BP[['BP', 'InterpolateSLinear', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_RH = pd.DataFrame(data['RH'])

a = data_RH.RH.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_RH[start:stop].shape)

print(data_RH[start:stop].isnull().sum())

data_RH = data_RH[start:stop]

data_RH['masked'] = data_RH['RH']

np.random.seed(120)

data_RH['masked'] = data_RH['masked'].mask(np.random.random(data_RH['masked'].shape) < .30)

data_RH = data_RH.assign(missing=np.nan)

data_RH.missing[data_RH.masked.isna()] = data_RH.RH

data_RH[['RH','masked','missing']] = data_RH[['RH','masked','missing']].astype(float)

data_RH.info()

data_RH.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))
# imputing using mean and median

data_RH = data_RH.assign(FillMean=data_RH.masked.fillna(data_RH.masked.mean()))

data_RH = data_RH.assign(FillMedian=data_RH.masked.fillna(data_RH.masked.median()))

# imputing using the rolling average

data_RH = data_RH.assign(RollingMean=data_RH.masked.fillna(data_RH.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_RH = data_RH.assign(RollingMedian=data_RH.masked.fillna(data_RH.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_RH = data_RH.assign(InterpolateLinear=data_RH.masked.interpolate(method='linear'))

data_RH = data_RH.assign(InterpolateTime=data_RH.masked.interpolate(method='time'))

data_RH = data_RH.assign(InterpolateQuadratic=data_RH.masked.interpolate(method='quadratic', limit_area='inside'))

data_RH = data_RH.assign(InterpolateCubic=data_RH.masked.interpolate(method='cubic', limit_area='inside'))

data_RH = data_RH.assign(InterpolateSLinear=data_RH.masked.interpolate(method='slinear', limit_area='inside'))

data_RH = data_RH.assign(InterpolateAkima=data_RH.masked.interpolate(method='akima', limit_area='inside'))

data_RH = data_RH.assign(InterpolatePoly5=data_RH.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_RH = data_RH.assign(InterpolatePoly7=data_RH.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_RH = data_RH.assign(InterpolateSpline3=data_RH.masked.interpolate(method='spline', order=3))

data_RH = data_RH.assign(InterpolateSpline4=data_RH.masked.interpolate(method='spline', order=4))

data_RH = data_RH.assign(InterpolateSpline5=data_RH.masked.interpolate(method='spline', order=5))
#missing data % in RH

print(data['RH'].isnull().sum()/data['RH'].shape[0])

results = [(method, metrics.r2_score(data_RH.RH, data_RH[method])) for method in list(data_RH)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

results_df.sort_values(by='R_squared', ascending=False)
#missing data % in RH

print(data['RH'].isnull().sum()/data['RH'].shape[0])

results = [(method, metrics.r2_score(data_RH.RH, data_RH[method])) for method in list(data_RH)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

results_df.sort_values(by='R_squared', ascending=False)
data_new = data_RH[['RH', 'InterpolateSpline4', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_SR = pd.DataFrame(data['SR'])

a = data_SR.SR.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_SR[start:stop].shape)

print(data_SR[start:stop].isnull().sum())

data_SR = data_SR[start:stop]

data_SR['masked'] = data_SR['SR']

np.random.seed(120)

data_SR['masked'] = data_SR['masked'].mask(np.random.random(data_SR['masked'].shape) < .30)

data_SR = data_SR.assign(missing=np.nan)

data_SR.missing[data_SR.masked.isna()] = data_SR.SR

data_SR[['SR','masked','missing']] = data_SR[['SR','masked','missing']].astype(float)

data_SR.info()

data_SR.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_SR = data_SR.assign(FillMean=data_SR.masked.fillna(data_SR.masked.mean()))

data_SR = data_SR.assign(FillMedian=data_SR.masked.fillna(data_SR.masked.median()))

# imputing using the rolling average

data_SR = data_SR.assign(RollingMean=data_SR.masked.fillna(data_SR.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_SR = data_SR.assign(RollingMedian=data_SR.masked.fillna(data_SR.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_SR = data_SR.assign(InterpolateLinear=data_SR.masked.interpolate(method='linear'))

data_SR = data_SR.assign(InterpolateTime=data_SR.masked.interpolate(method='time'))

data_SR = data_SR.assign(InterpolateQuadratic=data_SR.masked.interpolate(method='quadratic', limit_area='inside'))

data_SR = data_SR.assign(InterpolateCubic=data_SR.masked.interpolate(method='cubic', limit_area='inside'))

data_SR = data_SR.assign(InterpolateSLinear=data_SR.masked.interpolate(method='slinear', limit_area='inside'))

data_SR = data_SR.assign(InterpolateAkima=data_SR.masked.interpolate(method='akima', limit_area='inside'))

data_SR = data_SR.assign(InterpolatePoly5=data_SR.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_SR = data_SR.assign(InterpolatePoly7=data_SR.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_SR = data_SR.assign(InterpolateSpline3=data_SR.masked.interpolate(method='spline', order=3))

data_SR = data_SR.assign(InterpolateSpline4=data_SR.masked.interpolate(method='spline', order=4))

data_SR = data_SR.assign(InterpolateSpline5=data_SR.masked.interpolate(method='spline', order=5))

#missing data % in SR

print(data['SR'].isnull().sum()/data['SR'].shape[0])

results = [(method, metrics.r2_score(data_SR.SR, data_SR[method])) for method in list(data_SR)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_SR[['SR', 'InterpolateAkima', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_WD = pd.DataFrame(data['WD'])

a = data_WD.WD.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_WD[start:stop-1].shape)

print(data_WD[start:stop-1].isnull().sum())

data_WD = data_WD[start:stop-1]

data_WD['masked'] = data_WD['WD']

np.random.seed(120)

data_WD['masked'] = data_WD['masked'].mask(np.random.random(data_WD['masked'].shape) < .30)

data_WD = data_WD.assign(missing=np.nan)

data_WD.missing[data_WD.masked.isna()] = data_WD.WD

data_WD[['WD','masked','missing']] = data_WD[['WD','masked','missing']].astype(float)

data_WD.info()

data_WD.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_WD = data_WD.assign(FillMean=data_WD.masked.fillna(data_WD.masked.mean()))

data_WD = data_WD.assign(FillMedian=data_WD.masked.fillna(data_WD.masked.median()))

# imputing using the rolling average

data_WD = data_WD.assign(RollingMean=data_WD.masked.fillna(data_WD.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_WD = data_WD.assign(RollingMedian=data_WD.masked.fillna(data_WD.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_WD = data_WD.assign(InterpolateLinear=data_WD.masked.interpolate(method='linear'))

data_WD = data_WD.assign(InterpolateTime=data_WD.masked.interpolate(method='time'))

data_WD = data_WD.assign(InterpolateQuadratic=data_WD.masked.interpolate(method='quadratic', limit_area='inside'))

data_WD = data_WD.assign(InterpolateCubic=data_WD.masked.interpolate(method='cubic', limit_area='inside'))

data_WD = data_WD.assign(InterpolateSLinear=data_WD.masked.interpolate(method='slinear', limit_area='inside'))

data_WD = data_WD.assign(InterpolateAkima=data_WD.masked.interpolate(method='akima', limit_area='inside'))

data_WD = data_WD.assign(InterpolatePoly5=data_WD.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_WD = data_WD.assign(InterpolatePoly7=data_WD.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_WD = data_WD.assign(InterpolateSpline3=data_WD.masked.interpolate(method='spline', order=3))

data_WD = data_WD.assign(InterpolateSpline4=data_WD.masked.interpolate(method='spline', order=4))

data_WD = data_WD.assign(InterpolateSpline5=data_WD.masked.interpolate(method='spline', order=5))

#missing data % in WD

print(data['WD'].isnull().sum()/data['WD'].shape[0])

results = [(method, metrics.r2_score(data_WD.WD, data_WD[method])) for method in list(data_WD)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_WD[['WD', 'InterpolateLinear', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_WS = pd.DataFrame(data['WS'])

a = data_WS.WS.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_WS[start:stop-1].shape)

print(data_WS[start:stop-1].isnull().sum())

data_WS = data_WS[start:stop-1]

data_WS['masked'] = data_WS['WS']

np.random.seed(120)

data_WS['masked'] = data_WS['masked'].mask(np.random.random(data_WS['masked'].shape) < .30)

data_WS = data_WS.assign(missing=np.nan)

data_WS.missing[data_WS.masked.isna()] = data_WS.WS

data_WS[['WS','masked','missing']] = data_WS[['WS','masked','missing']].astype(float)

data_WS.info()

data_WS.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_WS = data_WS.assign(FillMean=data_WS.masked.fillna(data_WS.masked.mean()))

data_WS = data_WS.assign(FillMedian=data_WS.masked.fillna(data_WS.masked.median()))

# imputing using the rolling average

data_WS = data_WS.assign(RollingMean=data_WS.masked.fillna(data_WS.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_WS = data_WS.assign(RollingMedian=data_WS.masked.fillna(data_WS.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_WS = data_WS.assign(InterpolateLinear=data_WS.masked.interpolate(method='linear'))

data_WS = data_WS.assign(InterpolateTime=data_WS.masked.interpolate(method='time'))

data_WS = data_WS.assign(InterpolateQuadratic=data_WS.masked.interpolate(method='quadratic', limit_area='inside'))

data_WS = data_WS.assign(InterpolateCubic=data_WS.masked.interpolate(method='cubic', limit_area='inside'))

data_WS = data_WS.assign(InterpolateSLinear=data_WS.masked.interpolate(method='slinear', limit_area='inside'))

data_WS = data_WS.assign(InterpolateAkima=data_WS.masked.interpolate(method='akima', limit_area='inside'))

data_WS = data_WS.assign(InterpolatePoly5=data_WS.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_WS = data_WS.assign(InterpolatePoly7=data_WS.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_WS = data_WS.assign(InterpolateSpline3=data_WS.masked.interpolate(method='spline', order=3))

data_WS = data_WS.assign(InterpolateSpline4=data_WS.masked.interpolate(method='spline', order=4))

data_WS = data_WS.assign(InterpolateSpline5=data_WS.masked.interpolate(method='spline', order=5))

#missing data % in WS

print(data['WS'].isnull().sum()/data['WS'].shape[0])

results = [(method, metrics.r2_score(data_WS.WS, data_WS[method])) for method in list(data_WS)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_WS[['WS', 'InterpolateAkima', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_CO = pd.DataFrame(data['CO'])

a = data_CO.CO.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_CO[start:stop-1].shape)

print(data_CO[start:stop-1].isnull().sum())

data_CO = data_CO[start:stop-1]

data_CO['masked'] = data_CO['CO']

np.random.seed(120)

data_CO['masked'] = data_CO['masked'].mask(np.random.random(data_CO['masked'].shape) < .30)

data_CO = data_CO.assign(missing=np.nan)

data_CO.missing[data_CO.masked.isna()] = data_CO.CO

data_CO[['CO','masked','missing']] = data_CO[['CO','masked','missing']].astype(float)

data_CO.info()

data_CO.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_CO = data_CO.assign(FillMean=data_CO.masked.fillna(data_CO.masked.mean()))

data_CO = data_CO.assign(FillMedian=data_CO.masked.fillna(data_CO.masked.median()))

# imputing using the rolling average

data_CO = data_CO.assign(RollingMean=data_CO.masked.fillna(data_CO.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_CO = data_CO.assign(RollingMedian=data_CO.masked.fillna(data_CO.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_CO = data_CO.assign(InterpolateLinear=data_CO.masked.interpolate(method='linear'))

data_CO = data_CO.assign(InterpolateTime=data_CO.masked.interpolate(method='time'))

data_CO = data_CO.assign(InterpolateQuadratic=data_CO.masked.interpolate(method='quadratic', limit_area='inside'))

data_CO = data_CO.assign(InterpolateCubic=data_CO.masked.interpolate(method='cubic', limit_area='inside'))

data_CO = data_CO.assign(InterpolateSLinear=data_CO.masked.interpolate(method='slinear', limit_area='inside'))

data_CO = data_CO.assign(InterpolateAkima=data_CO.masked.interpolate(method='akima', limit_area='inside'))

data_CO = data_CO.assign(InterpolatePoly5=data_CO.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_CO = data_CO.assign(InterpolatePoly7=data_CO.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_CO = data_CO.assign(InterpolateSpline3=data_CO.masked.interpolate(method='spline', order=3))

data_CO = data_CO.assign(InterpolateSpline4=data_CO.masked.interpolate(method='spline', order=4))

data_CO = data_CO.assign(InterpolateSpline5=data_CO.masked.interpolate(method='spline', order=5))

#missing data % in CO

print(data_CO.isnull().sum())

print(data['CO'].isnull().sum()/data['CO'].shape[0])

results = [(method, metrics.r2_score(data_CO.CO, data_CO[method])) for method in list(data_CO)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_CO[['CO', 'InterpolateLinear', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_NH3 = pd.DataFrame(data['NH3'])

a = data_NH3.NH3.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_NH3[start:stop-2].shape)

print(data_NH3[start:stop-2].isnull().sum())

data_NH3 = data_NH3[start:stop-2]

data_NH3['masked'] = data_NH3['NH3']

np.random.seed(120)

data_NH3['masked'] = data_NH3['masked'].mask(np.random.random(data_NH3['masked'].shape) < .30)

data_NH3 = data_NH3.assign(missing=np.nan)

data_NH3.missing[data_NH3.masked.isna()] = data_NH3.NH3

data_NH3[['NH3','masked','missing']] = data_NH3[['NH3','masked','missing']].astype(float)

data_NH3.info()

data_NH3.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_NH3 = data_NH3.assign(FillMean=data_NH3.masked.fillna(data_NH3.masked.mean()))

data_NH3 = data_NH3.assign(FillMedian=data_NH3.masked.fillna(data_NH3.masked.median()))

# imputing using the rolling average

data_NH3 = data_NH3.assign(RollingMean=data_NH3.masked.fillna(data_NH3.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_NH3 = data_NH3.assign(RollingMedian=data_NH3.masked.fillna(data_NH3.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_NH3 = data_NH3.assign(InterpolateLinear=data_NH3.masked.interpolate(method='linear'))

data_NH3 = data_NH3.assign(InterpolateTime=data_NH3.masked.interpolate(method='time'))

data_NH3 = data_NH3.assign(InterpolateQuadratic=data_NH3.masked.interpolate(method='quadratic', limit_area='inside'))

data_NH3 = data_NH3.assign(InterpolateCubic=data_NH3.masked.interpolate(method='cubic', limit_area='inside'))

data_NH3 = data_NH3.assign(InterpolateSLinear=data_NH3.masked.interpolate(method='slinear', limit_area='inside'))

data_NH3 = data_NH3.assign(InterpolateAkima=data_NH3.masked.interpolate(method='akima', limit_area='inside'))

data_NH3 = data_NH3.assign(InterpolatePoly5=data_NH3.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_NH3 = data_NH3.assign(InterpolatePoly7=data_NH3.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_NH3 = data_NH3.assign(InterpolateSpline3=data_NH3.masked.interpolate(method='spline', order=3))

data_NH3 = data_NH3.assign(InterpolateSpline4=data_NH3.masked.interpolate(method='spline', order=4))

data_NH3 = data_NH3.assign(InterpolateSpline5=data_NH3.masked.interpolate(method='spline', order=5))

#missing data % in NH3

print(data['NH3'].isnull().sum()/data['NH3'].shape[0])

results = [(method, metrics.r2_score(data_NH3.NH3, data_NH3[method])) for method in list(data_NH3)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_NH3[['NH3', 'InterpolateLinear', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_NO = pd.DataFrame(data['NO'])

a = data_NO.NO.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_NO[start:stop-1].shape)

print(data_NO[start:stop-1].isnull().sum())

data_NO = data_NO[start:stop-1]

data_NO['masked'] = data_NO['NO']

np.random.seed(120)

data_NO['masked'] = data_NO['masked'].mask(np.random.random(data_NO['masked'].shape) < .30)

data_NO = data_NO.assign(missing=np.nan)

data_NO.missing[data_NO.masked.isna()] = data_NO.NO

data_NO[['NO','masked','missing']] = data_NO[['NO','masked','missing']].astype(float)

data_NO.info()

data_NO.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_NO = data_NO.assign(FillMean=data_NO.masked.fillna(data_NO.masked.mean()))

data_NO = data_NO.assign(FillMedian=data_NO.masked.fillna(data_NO.masked.median()))

# imputing using the rolling average

data_NO = data_NO.assign(RollingMean=data_NO.masked.fillna(data_NO.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_NO = data_NO.assign(RollingMedian=data_NO.masked.fillna(data_NO.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_NO = data_NO.assign(InterpolateLinear=data_NO.masked.interpolate(method='linear'))

data_NO = data_NO.assign(InterpolateTime=data_NO.masked.interpolate(method='time'))

data_NO = data_NO.assign(InterpolateQuadratic=data_NO.masked.interpolate(method='quadratic', limit_area='inside'))

data_NO = data_NO.assign(InterpolateCubic=data_NO.masked.interpolate(method='cubic', limit_area='inside'))

data_NO = data_NO.assign(InterpolateSLinear=data_NO.masked.interpolate(method='slinear', limit_area='inside'))

data_NO = data_NO.assign(InterpolateAkima=data_NO.masked.interpolate(method='akima', limit_area='inside'))

data_NO = data_NO.assign(InterpolatePoly5=data_NO.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_NO = data_NO.assign(InterpolatePoly7=data_NO.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_NO = data_NO.assign(InterpolateSpline3=data_NO.masked.interpolate(method='spline', order=3))

data_NO = data_NO.assign(InterpolateSpline4=data_NO.masked.interpolate(method='spline', order=4))

data_NO = data_NO.assign(InterpolateSpline5=data_NO.masked.interpolate(method='spline', order=5))

#missing data % in NO

print(data['NO'].isnull().sum()/data['NO'].shape[0])

results = [(method, metrics.r2_score(data_NO.NO, data_NO[method])) for method in list(data_NO)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_NO[['NO', 'InterpolateAkima', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_NO2 = pd.DataFrame(data['NO2'])

a = data_NO2.NO2.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_NO2[start:stop-1].shape)

print(data_NO2[start:stop-1].isnull().sum())

data_NO2 = data_NO2[start:stop-1]

data_NO2['masked'] = data_NO2['NO2']

np.random.seed(120)

data_NO2['masked'] = data_NO2['masked'].mask(np.random.random(data_NO2['masked'].shape) < .30)

data_NO2 = data_NO2.assign(missing=np.nan)

data_NO2.missing[data_NO2.masked.isna()] = data_NO2.NO2

data_NO2[['NO2','masked','missing']] = data_NO2[['NO2','masked','missing']].astype(float)

data_NO2.info()

data_NO2.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_NO2 = data_NO2.assign(FillMean=data_NO2.masked.fillna(data_NO2.masked.mean()))

data_NO2 = data_NO2.assign(FillMedian=data_NO2.masked.fillna(data_NO2.masked.median()))

# imputing using the rolling average

data_NO2 = data_NO2.assign(RollingMean=data_NO2.masked.fillna(data_NO2.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_NO2 = data_NO2.assign(RollingMedian=data_NO2.masked.fillna(data_NO2.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_NO2 = data_NO2.assign(InterpolateLinear=data_NO2.masked.interpolate(method='linear'))

data_NO2 = data_NO2.assign(InterpolateTime=data_NO2.masked.interpolate(method='time'))

data_NO2 = data_NO2.assign(InterpolateQuadratic=data_NO2.masked.interpolate(method='quadratic', limit_area='inside'))

data_NO2 = data_NO2.assign(InterpolateCubic=data_NO2.masked.interpolate(method='cubic', limit_area='inside'))

data_NO2 = data_NO2.assign(InterpolateSLinear=data_NO2.masked.interpolate(method='slinear', limit_area='inside'))

data_NO2 = data_NO2.assign(InterpolateAkima=data_NO2.masked.interpolate(method='akima', limit_area='inside'))

data_NO2 = data_NO2.assign(InterpolatePoly5=data_NO2.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_NO2 = data_NO2.assign(InterpolatePoly7=data_NO2.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_NO2 = data_NO2.assign(InterpolateSpline3=data_NO2.masked.interpolate(method='spline', order=3))

data_NO2 = data_NO2.assign(InterpolateSpline4=data_NO2.masked.interpolate(method='spline', order=4))

data_NO2 = data_NO2.assign(InterpolateSpline5=data_NO2.masked.interpolate(method='spline', order=5))

#missing data % in NO2

print(data['NO2'].isnull().sum()/data['NO2'].shape[0])

results = [(method, metrics.r2_score(data_NO2.NO2, data_NO2[method])) for method in list(data_NO2)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_NO2[['NO2', 'InterpolateCubic', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_NOx = pd.DataFrame(data['NOx'])

a = data_NOx.NOx.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_NOx[start:stop-1].shape)

print(data_NOx[start:stop-1].isnull().sum())

data_NOx = data_NOx[start:stop-1]

data_NOx['masked'] = data_NOx['NOx']

np.random.seed(120)

data_NOx['masked'] = data_NOx['masked'].mask(np.random.random(data_NOx['masked'].shape) < .30)

data_NOx = data_NOx.assign(missing=np.nan)

data_NOx.missing[data_NOx.masked.isna()] = data_NOx.NOx

data_NOx[['NOx','masked','missing']] = data_NOx[['NOx','masked','missing']].astype(float)

data_NOx.info()

data_NOx.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_NOx = data_NOx.assign(FillMean=data_NOx.masked.fillna(data_NOx.masked.mean()))

data_NOx = data_NOx.assign(FillMedian=data_NOx.masked.fillna(data_NOx.masked.median()))

# imputing using the rolling average

data_NOx = data_NOx.assign(RollingMean=data_NOx.masked.fillna(data_NOx.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_NOx = data_NOx.assign(RollingMedian=data_NOx.masked.fillna(data_NOx.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_NOx = data_NOx.assign(InterpolateLinear=data_NOx.masked.interpolate(method='linear'))

data_NOx = data_NOx.assign(InterpolateTime=data_NOx.masked.interpolate(method='time'))

data_NOx = data_NOx.assign(InterpolateQuadratic=data_NOx.masked.interpolate(method='quadratic', limit_area='inside'))

data_NOx = data_NOx.assign(InterpolateCubic=data_NOx.masked.interpolate(method='cubic', limit_area='inside'))

data_NOx = data_NOx.assign(InterpolateSLinear=data_NOx.masked.interpolate(method='slinear', limit_area='inside'))

data_NOx = data_NOx.assign(InterpolateAkima=data_NOx.masked.interpolate(method='akima', limit_area='inside'))

data_NOx = data_NOx.assign(InterpolatePoly5=data_NOx.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_NOx = data_NOx.assign(InterpolatePoly7=data_NOx.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_NOx = data_NOx.assign(InterpolateSpline3=data_NOx.masked.interpolate(method='spline', order=3))

data_NOx = data_NOx.assign(InterpolateSpline4=data_NOx.masked.interpolate(method='spline', order=4))

data_NOx = data_NOx.assign(InterpolateSpline5=data_NOx.masked.interpolate(method='spline', order=5))

#missing data % in NOx

print(data['NOx'].isnull().sum()/data['NOx'].shape[0])

results = [(method, metrics.r2_score(data_NOx.NOx, data_NOx[method])) for method in list(data_NOx)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_NOx[['NOx', 'InterpolateAkima', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_Ozone = pd.DataFrame(data['Ozone'])

a = data_Ozone.Ozone.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_Ozone[start:stop-1].shape)

print(data_Ozone[start:stop-1].isnull().sum())

data_Ozone = data_Ozone[start:stop-1]

data_Ozone['masked'] = data_Ozone['Ozone']

np.random.seed(120)

data_Ozone['masked'] = data_Ozone['masked'].mask(np.random.random(data_Ozone['masked'].shape) < .30)

data_Ozone = data_Ozone.assign(missing=np.nan)

data_Ozone.missing[data_Ozone.masked.isna()] = data_Ozone.Ozone

data_Ozone[['Ozone','masked','missing']] = data_Ozone[['Ozone','masked','missing']].astype(float)

data_Ozone.info()

data_Ozone.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_Ozone = data_Ozone.assign(FillMean=data_Ozone.masked.fillna(data_Ozone.masked.mean()))

data_Ozone = data_Ozone.assign(FillMedian=data_Ozone.masked.fillna(data_Ozone.masked.median()))

# imputing using the rolling average

data_Ozone = data_Ozone.assign(RollingMean=data_Ozone.masked.fillna(data_Ozone.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_Ozone = data_Ozone.assign(RollingMedian=data_Ozone.masked.fillna(data_Ozone.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_Ozone = data_Ozone.assign(InterpolateLinear=data_Ozone.masked.interpolate(method='linear'))

data_Ozone = data_Ozone.assign(InterpolateTime=data_Ozone.masked.interpolate(method='time'))

data_Ozone = data_Ozone.assign(InterpolateQuadratic=data_Ozone.masked.interpolate(method='quadratic', limit_area='inside'))

data_Ozone = data_Ozone.assign(InterpolateCubic=data_Ozone.masked.interpolate(method='cubic', limit_area='inside'))

data_Ozone = data_Ozone.assign(InterpolateSLinear=data_Ozone.masked.interpolate(method='slinear', limit_area='inside'))

data_Ozone = data_Ozone.assign(InterpolateAkima=data_Ozone.masked.interpolate(method='akima', limit_area='inside'))

data_Ozone = data_Ozone.assign(InterpolatePoly5=data_Ozone.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_Ozone = data_Ozone.assign(InterpolatePoly7=data_Ozone.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_Ozone = data_Ozone.assign(InterpolateSpline3=data_Ozone.masked.interpolate(method='spline', order=3))

data_Ozone = data_Ozone.assign(InterpolateSpline4=data_Ozone.masked.interpolate(method='spline', order=4))

data_Ozone = data_Ozone.assign(InterpolateSpline5=data_Ozone.masked.interpolate(method='spline', order=5))

#missing data % in Ozone

print(data['Ozone'].isnull().sum()/data['Ozone'].shape[0])

results = [(method, metrics.r2_score(data_Ozone.Ozone, data_Ozone[method])) for method in list(data_Ozone)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_Ozone[['Ozone', 'InterpolateQuadratic', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_PM10 = pd.DataFrame(data['PM10'])

a = data_PM10.PM10.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_PM10[start:stop-1].shape)

print(data_PM10[start:stop-1].isnull().sum())

data_PM10 = data_PM10[start:stop-1]

data_PM10['masked'] = data_PM10['PM10']

np.random.seed(120)

data_PM10['masked'] = data_PM10['masked'].mask(np.random.random(data_PM10['masked'].shape) < .30)

data_PM10 = data_PM10.assign(missing=np.nan)

data_PM10.missing[data_PM10.masked.isna()] = data_PM10.PM10

data_PM10[['PM10','masked','missing']] = data_PM10[['PM10','masked','missing']].astype(float)

data_PM10.info()

data_PM10.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_PM10 = data_PM10.assign(FillMean=data_PM10.masked.fillna(data_PM10.masked.mean()))

data_PM10 = data_PM10.assign(FillMedian=data_PM10.masked.fillna(data_PM10.masked.median()))

# imputing using the rolling average

data_PM10 = data_PM10.assign(RollingMean=data_PM10.masked.fillna(data_PM10.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_PM10 = data_PM10.assign(RollingMedian=data_PM10.masked.fillna(data_PM10.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_PM10 = data_PM10.assign(InterpolateLinear=data_PM10.masked.interpolate(method='linear'))

data_PM10 = data_PM10.assign(InterpolateTime=data_PM10.masked.interpolate(method='time'))

data_PM10 = data_PM10.assign(InterpolateQuadratic=data_PM10.masked.interpolate(method='quadratic', limit_area='inside'))

data_PM10 = data_PM10.assign(InterpolateCubic=data_PM10.masked.interpolate(method='cubic', limit_area='inside'))

data_PM10 = data_PM10.assign(InterpolateSLinear=data_PM10.masked.interpolate(method='slinear', limit_area='inside'))

data_PM10 = data_PM10.assign(InterpolateAkima=data_PM10.masked.interpolate(method='akima', limit_area='inside'))

data_PM10 = data_PM10.assign(InterpolatePoly5=data_PM10.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_PM10 = data_PM10.assign(InterpolatePoly7=data_PM10.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_PM10 = data_PM10.assign(InterpolateSpline3=data_PM10.masked.interpolate(method='spline', order=3))

data_PM10 = data_PM10.assign(InterpolateSpline4=data_PM10.masked.interpolate(method='spline', order=4))

data_PM10 = data_PM10.assign(InterpolateSpline5=data_PM10.masked.interpolate(method='spline', order=5))

#missing data % in PM10

print(data['PM10'].isnull().sum()/data['PM10'].shape[0])

results = [(method, metrics.r2_score(data_PM10.PM10, data_PM10[method])) for method in list(data_PM10)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_PM10[['PM10', 'InterpolateAkima', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_PM25 = pd.DataFrame(data['PM25'])

data_PM25 = data_PM25.astype(float)

a = data_PM25.PM25.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_PM25[start:stop-1].shape)

print(data_PM25[start:stop-1].isnull().sum())

data_PM25 = data_PM25[start:stop-1]

data_PM25['masked'] = data_PM25['PM25']

np.random.seed(120)

data_PM25['masked'] = data_PM25['masked'].mask(np.random.random(data_PM25['masked'].shape) < .30)

data_PM25 = data_PM25.assign(missing=np.nan)

data_PM25.missing[data_PM25.masked.isna()] = data_PM25.PM25

data_PM25[['PM25','masked','missing']] = data_PM25[['PM25','masked','missing']].astype(float)

data_PM25.info()

data_PM25.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_PM25 = data_PM25.assign(FillMean=data_PM25.masked.fillna(data_PM25.masked.mean()))

data_PM25 = data_PM25.assign(FillMedian=data_PM25.masked.fillna(data_PM25.masked.median()))

# imputing using the rolling average

data_PM25 = data_PM25.assign(RollingMean=data_PM25.masked.fillna(data_PM25.masked.rolling(24,min_periods=1,).mean()))

# imputing using the rolling median

data_PM25 = data_PM25.assign(RollingMedian=data_PM25.masked.fillna(data_PM25.masked.rolling(24,min_periods=1,).median()))# imputing using the median

data_PM25 = data_PM25.assign(InterpolateLinear=data_PM25.masked.interpolate(method='linear'))

data_PM25 = data_PM25.assign(InterpolateTime=data_PM25.masked.interpolate(method='time'))

data_PM25 = data_PM25.assign(InterpolateQuadratic=data_PM25.masked.interpolate(method='quadratic', limit_area='inside'))

data_PM25 = data_PM25.assign(InterpolateCubic=data_PM25.masked.interpolate(method='cubic', limit_area='inside'))

data_PM25 = data_PM25.assign(InterpolateSLinear=data_PM25.masked.interpolate(method='slinear', limit_area='inside'))

data_PM25 = data_PM25.assign(InterpolateAkima=data_PM25.masked.interpolate(method='akima', limit_area='inside'))

data_PM25 = data_PM25.assign(InterpolatePoly5=data_PM25.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_PM25 = data_PM25.assign(InterpolatePoly7=data_PM25.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_PM25 = data_PM25.assign(InterpolateSpline3=data_PM25.masked.interpolate(method='spline', order=3))

data_PM25 = data_PM25.assign(InterpolateSpline4=data_PM25.masked.interpolate(method='spline', order=4))

data_PM25 = data_PM25.assign(InterpolateSpline5=data_PM25.masked.interpolate(method='spline', order=5))

#missing data % in PM25

print(data['PM25'].isnull().sum()/data['PM25'].shape[0])

results = [(method, metrics.r2_score(data_PM25.PM25, data_PM25[method])) for method in list(data_PM25)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_PM25[['PM25', 'InterpolateSpline4', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))
data_PM25.isnull().sum()
data_SO2 = pd.DataFrame(data['SO2'])

a = data_SO2.SO2.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_SO2[start:stop-2].shape)

print(data_SO2[start:stop-2].isnull().sum())

data_SO2 = data_SO2[start:stop-2]

data_SO2['masked'] = data_SO2['SO2']

np.random.seed(120)

data_SO2['masked'] = data_SO2['masked'].mask(np.random.random(data_SO2['masked'].shape) < .06)

data_SO2 = data_SO2.assign(missing=np.nan)

data_SO2.missing[data_SO2.masked.isna()] = data_SO2.SO2

data_SO2[['SO2','masked','missing']] = data_SO2[['SO2','masked','missing']].astype(float)

data_SO2.info()

data_SO2.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))

# imputing using mean and median

data_SO2 = data_SO2.assign(FillMean=data_SO2.masked.fillna(data_SO2.masked.mean()))

data_SO2 = data_SO2.assign(FillMedian=data_SO2.masked.fillna(data_SO2.masked.median()))

# imputing using the rolling average

data_SO2 = data_SO2.assign(RollingMean=data_SO2.masked.fillna(data_SO2.masked.rolling(81,min_periods=1,).mean()))

# imputing using the rolling median

data_SO2 = data_SO2.assign(RollingMedian=data_SO2.masked.fillna(data_SO2.masked.rolling(81,min_periods=1,).median()))# imputing using the median

data_SO2 = data_SO2.assign(InterpolateLinear=data_SO2.masked.interpolate(method='linear'))

data_SO2 = data_SO2.assign(InterpolateTime=data_SO2.masked.interpolate(method='time'))

data_SO2 = data_SO2.assign(InterpolateQuadratic=data_SO2.masked.interpolate(method='quadratic', limit_area='inside'))

data_SO2 = data_SO2.assign(InterpolateCubic=data_SO2.masked.interpolate(method='cubic', limit_area='inside'))

data_SO2 = data_SO2.assign(InterpolateSLinear=data_SO2.masked.interpolate(method='slinear', limit_area='inside'))

data_SO2 = data_SO2.assign(InterpolateAkima=data_SO2.masked.interpolate(method='akima', limit_area='inside'))

data_SO2 = data_SO2.assign(InterpolatePoly5=data_SO2.masked.interpolate(method='polynomial', order=5, limit_area='inside')) 

data_SO2 = data_SO2.assign(InterpolatePoly7=data_SO2.masked.interpolate(method='polynomial', order=7, limit_area='inside'))

data_SO2 = data_SO2.assign(InterpolateSpline3=data_SO2.masked.interpolate(method='spline', order=3))

data_SO2 = data_SO2.assign(InterpolateSpline4=data_SO2.masked.interpolate(method='spline', order=4))

data_SO2 = data_SO2.assign(InterpolateSpline5=data_SO2.masked.interpolate(method='spline', order=5))

#missing data % in SO2

print(data['SO2'].isnull().sum()/data['SO2'].shape[0])

results = [(method, metrics.r2_score(data_SO2.SO2, data_SO2[method])) for method in list(data_SO2)[3:]]

results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])

print(results_df.sort_values(by='R_squared', ascending=False))

data_new = data_SO2[['SO2', 'RollingMedian', 'masked', 'missing']]

data_new.plot(style=['k--', 'yH', 'bo-', 'r*'], figsize=(20, 10))