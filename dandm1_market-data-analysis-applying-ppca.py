import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
market_train_df = env.get_training_data()[0]
market_train_df.head()
import os

import numpy as np
from scipy.linalg import orth


class PPCA():

    def __init__(self):

        self.raw = None
        self.data = None
        self.C = None
        self.means = None
        self.stds = None
        self.eig_vals = None

    def _standardize(self, X):

        if self.means is None or self.stds is None:
            raise RuntimeError("Fit model first")

        return (X - self.means) / self.stds

    def fit(self, data, d=None, tol=1e-4, min_obs=10, verbose=False):

        self.raw = data
        self.raw[np.isinf(self.raw)] = np.max(self.raw[np.isfinite(self.raw)])

        valid_series = np.sum(~np.isnan(self.raw), axis=0) >= min_obs

        data = self.raw[:, valid_series].copy()
        N = data.shape[0]
        D = data.shape[1]

        self.means = np.nanmean(data, axis=0)
        self.stds = np.nanstd(data, axis=0)

        data = self._standardize(data)
        observed = ~np.isnan(data)
        missing = np.sum(~observed)
        data[~observed] = 0

        # initial

        if d is None:
            d = data.shape[1]
        
        if self.C is None:
            C = np.random.randn(D, d)
        else:
            C = self.C
        CC = np.dot(C.T, C)
        X = np.dot(np.dot(data, C), np.linalg.inv(CC))
        recon = np.dot(X, C.T)
        recon[~observed] = 0
        ss = np.sum((recon - data)**2)/(N*D - missing)

        v0 = np.inf
        counter = 0

        while True:

            Sx = np.linalg.inv(np.eye(d) + CC/ss)

            # e-step
            ss0 = ss
            if missing > 0:
                proj = np.dot(X, C.T)
                data[~observed] = proj[~observed]
            X = np.dot(np.dot(data, C), Sx) / ss

            # m-step
            XX = np.dot(X.T, X)
            C = np.dot(np.dot(data.T, X), np.linalg.pinv(XX + N*Sx))
            CC = np.dot(C.T, C)
            recon = np.dot(X, C.T)
            recon[~observed] = 0
            ss = (np.sum((recon-data)**2) + N*np.sum(CC*Sx) + missing*ss0)/(N*D)

            # calc diff for convergence
            det = np.log(np.linalg.det(Sx))
            if np.isinf(det):
                det = abs(np.linalg.slogdet(Sx)[1])
            v1 = N*(D*np.log(ss) + np.trace(Sx) - det) \
                + np.trace(XX) - missing*np.log(ss0)
            diff = abs(v1/v0 - 1)
            if verbose:
                print('\rAt iteration {} the diff is {:8.6f} (target {})'.format(counter,diff,tol),end='')
            if (diff < tol) and (counter > 5):
                break

            counter += 1
            v0 = v1


        C = orth(C)
        vals, vecs = np.linalg.eig(np.cov(np.dot(data, C).T))
        order = np.flipud(np.argsort(vals))
        vecs = vecs[:, order]
        vals = vals[order]

        C = np.dot(C, vecs)

        # attach objects to class
        self.C = C
        self.data = data
        self.eig_vals = vals
        self._calc_var()

    def transform(self, data=None):

        if self.C is None:
            raise RuntimeError('Fit the data model first.')
        if data is None:
            return np.dot(self.data, self.C)
        missing = np.isnan(data)
        #Obtain mean of columns as you need, nanmean is just convenient.
        it = 0
        if np.sum(missing) > 0:
            col_mean = np.nanmean(data, axis=0)
            data[missing] = np.take(col_mean, np.where(missing)[0])
            change = 1
            while(change>1e-3):
                CC = np.dot(self.C.T, self.C)
                X = np.dot(np.dot(data, self.C), np.linalg.inv(CC))
                proj = np.dot(X, self.C.T)
                change = np.max(np.abs(data[missing]-proj[missing]))
                print('\rIteration {}. Change is {:6.4f}'.format(it,change),end='')
                data[missing] = proj[missing]
                it += 1
        return np.dot(data, self.C)

    def _calc_var(self):

        if self.data is None:
            raise RuntimeError('Fit the data model first.')

        data = self.data.T

        # variance calc
        var = np.nanvar(data, axis=1)
        total_var = var.sum()
        self.var_exp = self.eig_vals.cumsum() / total_var

    def save(self, fpath):

        np.save(fpath, self.C)
        
    def load(self, fpath):

        assert os.path.isfile(fpath)

        self.C = np.load(fpath)
market_train_df['assetCode'].count()
market_train_df.drop(market_train_df[(market_train_df['open']>market_train_df['close']*5) | (market_train_df['close']>market_train_df['open']*5)].index,inplace=True)
market_train_df.drop(market_train_df[(np.abs(market_train_df['returnsOpenPrevRaw1'])>0.5) | (np.abs(market_train_df['returnsClosePrevRaw1'])>0.5)].index,inplace=True)
market_train_df.drop(market_train_df[(np.abs(market_train_df['returnsOpenPrevRaw10'])>1.5) | (np.abs(market_train_df['returnsClosePrevRaw10'])>1.5)].index,inplace=True)
market_train_df['time']=market_train_df['time'].dt.floor('D')
market_train_df['assetCode'].count()
def DfToData(df,valueCol):
    data = df.pivot(index='time',columns='assetCode',values=valueCol).values
    asset_means = np.nanmean(data,axis=0)
    data -= asset_means
    return data

dataList = []
dataList.append(DfToData(market_train_df,'returnsClosePrevRaw10'))
dataList.append(DfToData(market_train_df,'returnsClosePrevMktres10'))
data = np.concatenate(dataList,axis=0)

data.shape
ppca = PPCA()

ppca.fit(data, d=128, verbose=True,min_obs=1)
ppca.save('ppca.np')
# A function to transform a column from the  using a PPCA which was trained on a different column
def PpcaTransform(df,column,ppca):
    return ppca.transform(DfToData(df,column))
print('Residual 1 day opening price returns')
Open1d = PpcaTransform(market_train_df,'returnsOpenPrevRaw1',ppca)
print('\n\nResidual 1 day closing price returns')
Close1d = PpcaTransform(market_train_df,'returnsClosePrevRaw1',ppca)
print('\n\nResidual 10 day opening price returns')
Open10d = PpcaTransform(market_train_df,'returnsOpenPrevRaw10',ppca)
print('\n\nResidual 10 day closing price returns')
Close10d = PpcaTransform(market_train_df,'returnsClosePrevRaw10',ppca)
print('\n\nDaily volumes')
Volume = PpcaTransform(market_train_df,'volume',ppca)
print('\n\nThe target forward-looking returns')
Targets = PpcaTransform(market_train_df,'returnsOpenNextMktres10',ppca)
Market_Features = np.concatenate([Open1d,Close1d,Open10d,Close10d,Volume],axis=1)
np.save('Features.np',Market_Features)
np.save('Targets.np',Targets)
fig,ax=plt.subplots(figsize=(30,10))
im=ax.imshow(ppca.C,aspect='auto',cmap='RdBu')

plt.colorbar(im)
plt.figure(figsize=(30, 15))
labels = range(ppca.C.shape[1])
for values,label in zip(Close10d.T,labels):
    plt.plot(values,label=label)
plt.legend()
plt.figure(figsize=(30, 15))
plt.plot(Volume)
asset = 2060

data = DfToData(market_train_df,'returnsClosePrevRaw10')
pca_history = Close10d

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10),sharey=True)
reconstructed = np.sum(ppca.C[asset]*ppca.stds[asset]*pca_history,axis=1)
ax1.plot(reconstructed)
ax1.plot(data[:,asset])
ax2.plot(data[:,asset]-reconstructed)
print('Data Std Dev       : {:6.4f}'.format(np.nanstd(data[:,asset])))
print('Reconstructed StDev: {:6.4f}'.format(np.nanstd(reconstructed)))
print('Noise StDev        : {:6.4f}'.format(np.nanstd(data[:,asset]-reconstructed)))
data2 = DfToData(market_train_df,'returnsOpenPrevRaw1')
pca_history = Open1d

ppca.eig_vals
asset = 2060
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10),sharey=True)
reconstructed2 = np.sum(ppca.C[asset]*ppca.stds[asset]*pca_history,axis=1)
ax1.plot(reconstructed)
ax1.plot(data2[:,asset])
ax2.plot(data2[:,asset]-reconstructed2)
print('Data Std Dev       : {:6.4f}'.format(np.nanstd(data2[:,asset])))
print('Reconstructed StDev: {:6.4f}'.format(np.nanstd(reconstructed2)))
print('Noise StDev        : {:6.4f}'.format(np.nanstd(data2[:,asset]-reconstructed2)))
noise = data[:,asset]-reconstructed
plt.hist(noise[~np.isnan(noise)],bins=25)
plt.scatter(data[:,asset],reconstructed)
plt.scatter(data2[:,asset],reconstructed2)
def DfWithErrorStats(data,pca_history):
    data_std = []
    recon_std = []
    noise_std = []
    count_nan = []
    correct_count = []
    wrong_count = []
    net_return = []
    for asset in range(data.shape[1]):
        reconstructed = np.sum(ppca.C[asset]*ppca.stds[asset]*pca_history,axis=1)
        data_std.append(np.nanstd(data[:,asset]))
        recon_std.append(np.nanstd(reconstructed))
        noise_std.append(np.nanstd(data[:,asset]-reconstructed))
        count_nan.append((~np.isnan(data[:,asset])).sum())
        correct = 0
        wrong = 0
        net = 0
        for orig,recon in zip(data[:,asset],reconstructed):
            if not np.isnan(orig):
                if ((orig > 0) and (recon > 0)) or ((orig < 0) and (recon < 0)):
                    correct += 1
                else:
                    wrong += 1
                net += np.sign(recon)*orig
        correct_count.append(correct)
        wrong_count.append(wrong)
        net_return.append(net)

    std_df = pd.DataFrame({'Count':count_nan,'DataStd':data_std,'ReconStd':recon_std,'NoiseStd':noise_std, 'Correct':correct_count, 'Wrong':wrong_count, 'Net':net_return})
    std_df['Ratio']=std_df['NoiseStd']/std_df['DataStd']
    std_df['Accuracy']=std_df['Correct']/(std_df['Correct']+std_df['Wrong'])
    del data_std
    del recon_std
    del noise_std
    return std_df
data = market_train_df.pivot(index='time',columns='assetCode',values='returnsClosePrevRaw10').values
asset_means = np.nanmean(data,axis=0)
data -= asset_means

pca_history = Close10d

stats_train_df = DfWithErrorStats(data,pca_history)
stats_train_df.sort_values(['Count'])
stats_train_df['Accuracy'].hist(bins=50)
stats_train_df['Correct'].sum()/(stats_train_df['Correct'].sum()+stats_train_df['Wrong'].sum())
data = market_train_df.pivot(index='time',columns='assetCode',values='returnsOpenPrevRaw1').values
asset_means = np.nanmean(data,axis=0)
data -= asset_means

pca_history = Open1d

stats_1day_df = DfWithErrorStats(data,pca_history)
stats_1day_df.sort_values(['Count'])
stats_1day_df['Accuracy'].hist(bins=50)
stats_1day_df['Correct'].sum()/(stats_train_df['Correct'].sum()+stats_train_df['Wrong'].sum())
data = market_train_df.pivot(index='time',columns='assetCode',values='volume').values
asset_means = np.nanmean(data,axis=0)
data -= asset_means

pca_history = Volume

stats_vol_df = DfWithErrorStats(data,pca_history)
stats_vol_df.sort_values(['Count'])
stats_vol_df['Accuracy'].hist(bins=50)
stats_vol_df['Correct'].sum()/(stats_train_df['Correct'].sum()+stats_train_df['Wrong'].sum())
data = market_train_df.pivot(index='time',columns='assetCode',values='returnsOpenNextMktres10').values
asset_means = np.nanmean(data,axis=0)
data -= asset_means

pca_history = Targets

stats_target_df = DfWithErrorStats(data,pca_history)
stats_target_df.sort_values(['Count'])
stats_target_df['Accuracy'].hist(bins=50)
stats_target_df['Correct'].sum()/(stats_train_df['Correct'].sum()+stats_train_df['Wrong'].sum())
