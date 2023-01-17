import numpy as np
import pandas as pd
import gc
from matplotlib import pyplot as plt
from scipy.signal import lombscargle
import math
from tqdm import tqdm
# some help functions
# angular frequency to period
def freq2Period(w):
    return 2 * math.pi / w
# period to angular frequency
def period2Freq(T):
    return 2 * math.pi / T
gc.enable()
train = pd.read_csv('../input/training_set.csv')
print(train['object_id'].unique())
gc.collect()
# get data and normalize bands
def processData(train, object_id):
    
    #load data for given object
    X = train.loc[train['object_id'] == object_id]
    x = np.array(X['mjd'].values)
    y = np.array(X['flux'].values)
    passband = np.array(X['passband'].values)
    
    # normalize bands
    for i in np.unique(passband):
        yy = y[np.where(passband==i)]
        mean = np.mean(yy)
        std = np.std(yy)
        y[np.where(passband==i)] = (yy - mean)/std
    
    return x, y, passband
x, y, passband = processData(train, 615)
plt.scatter(x, y, c=passband)
plt.xlabel('time (MJD)')
plt.ylabel('Normalized flux')
plt.show()
# calculate periodogram
def getPeriodogram(x, y, steps = 10000, minPeriod = None, maxPeriod = None):
    if not minPeriod:
        minPeriod = 0.1 # for now, let's ignore very short periodic objects
    if not maxPeriod:
        maxPeriod = (np.max(x) - np.min(x))/2 # you cannot detect P > half of your observation period

    maxFreq = np.log2(period2Freq(minPeriod))
    minFreq = np.log2(period2Freq(maxPeriod))
    f = np.power(2, np.linspace(minFreq,maxFreq, steps))
    p = lombscargle(x,y,f,normalize=True)
    return f, p
%%time
f,p = getPeriodogram(x, y, steps=20000)
plt.semilogx(freq2Period(f),p)
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.show()
def findBestPeaks(x, y, F, P, threshold=0.3, n=5):
    
    # find peaks above threshold
    indexes = np.where(P>threshold)[0]
    # if nothing found, look at the highest peaks anyway
    if len(indexes) == 0:
        q = np.quantile(P, 0.9995)
        indexes = np.where(P>q)[0]
    
    peaks = []
    start = 0
    end = 0
    for i in indexes:
        if i - end > 10:
            peaks.append((start, end))
            start = i
            end = i
        else:
            end = i
    
    peaks.append((start, end))
        
    
    # increase accuracy on the found peaks
    results = []
    for start, end in peaks:
        if end > 0:
            minPeriod = freq2Period(F[min(F.shape[0]-1, end+1)])
            maxPeriod = freq2Period(F[max(start-1, 0)])
            steps = int(100 * np.sqrt(end-start+1)) # the bigger the peak width, the more steps we want - but sensible (linear increase leads to long computation)
            f, p = getPeriodogram(x, y, steps = steps, minPeriod=minPeriod, maxPeriod=maxPeriod)
            results.append(np.array([freq2Period(f[np.argmax(p)]), np.max(p)]))

    # sort by normalized periodogram score and return first n results
    if results:
        results = np.array(results)
        results = results[np.flip(results[:,1].argsort())]
    else:
        results = np.array([freq2Period(F[np.argmax(P)]), np.max(P)]).reshape(1,2)
    return results[0:n]
%%time
results = findBestPeaks(x, y, f, p)
print('Period(days) Power')
print(results)
plt.figure(figsize=(20,25))

for i in range(results.shape[0]):
    plt.subplot(results.shape[0],2,i+1)
    phase = x/results[i][0] % 1
    plt.scatter(phase, y, c = passband, s=4)
    plt.xlabel('Phase')
    plt.ylabel('Normalized flux')
    plt.title('Period: {:.4f}, power: {:.2f}'.format(results[i][0], results[i][1]))

plt.show()
from multiprocessing import Pool
import multiprocessing as mp

CORES = mp.cpu_count() #4

def getFeatures(object_id):
    
    x, y, passband = processData(train, object_id)
    f,p = getPeriodogram(x, y)
    peaks = findBestPeaks(x, y, f, p)
    features = np.zeros((5,2))
    features[:peaks.shape[0],:peaks.shape[1]] = peaks
    
    return np.append(np.array([object_id]), features.reshape(5*2))
object_ids = train['object_id'].unique()[0:100]
%%time
features = []
for object_id in object_ids:
    results = getFeatures(object_id)
    features.append(results)
%%time
p = Pool(CORES)

results = p.map(getFeatures, object_ids)
object_ids = train['object_id'].unique()
columns = np.array(['id'])
for i in range(5):
    period_str = 'period_'+str(i+1)
    power_str = 'power_'+str(i+1)
    columns = np.append(columns, np.array([period_str, power_str]))

results = p.map(getFeatures, object_ids)

output = pd.DataFrame(results, columns=columns)
output['id'] = output['id'].astype(np.int32)
output.to_csv('./train-periods.csv')