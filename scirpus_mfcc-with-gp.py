# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

3

4

# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pywt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.



from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt



# train.csv is huge, so I implement csv_fragments() function

# which yields DataFrame of the specified length while scaning a csv file from start to end.



import builtins



random_seed = 4126



cast = {

    'acoustic_data': 'int',

    'time_to_failure': 'float'

}



def denoise_signal_simple(x, wavelet='db4', level=1):

    coeff = pywt.wavedec(x, wavelet, mode="per")

    #univeral threshold

    uthresh = 10

    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    # Reconstruct the signal using the thresholded coefficients

    return pywt.waverec(coeff, wavelet, mode='per')



def df_fragments(path, length, skip=1):

    with open(path, 'r') as f:

        m = {}

        cols = []

        count = 0

        index = 0

        for line in f:

            if len(cols) == 0:

                for col in line.strip("\n\r ").split(','):

                    cols.append(col)

                continue

            if count == 0:

                for col in cols:

                    m[col] = []

            if index % skip == 0:

                for j, cell in enumerate(line.strip("\n\r ").split(',')):

                    col = cols[j]

                    m[col].append(getattr(builtins, cast[col])(cell))

            count += 1

            if count == length:

                if index % skip == 0:

                    yield pd.DataFrame(m)

                index += 1

                count = 0



def count_rows(path):

    with open(path, 'r') as f:

        i = -1

        for _ in f:

            i += 1

        return i
import librosa, librosa.display
for df in df_fragments('../input/train.csv', 150000):

    mfcc = librosa.feature.mfcc(df['acoustic_data'].values.astype('float32'),sr=40000)

    plt.figure(figsize=(25, 5))

    librosa.display.specshow(mfcc, x_axis='time')

    plt.colorbar()

    break
for df in df_fragments('../input/train.csv', 150000):

    x = denoise_signal_simple(df['acoustic_data'].values.astype('float32'))

    mfcc = librosa.feature.mfcc(x,sr=40000)

    plt.figure(figsize=(25, 5))

    librosa.display.specshow(mfcc, x_axis='time')

    plt.colorbar()

    break
mfcc_ttf_map = {}

# you can reduce train data to process for some quick experiments

# skip = 10

# I have no idea what the actual sampling-rate is, but 1000 scores better than defaut

sr = 40000





for df in tqdm(pd.read_csv('../input/train.csv', header=None,chunksize=150000,skiprows=1)):

    df.columns = ['acoustic_data','time_to_failure']

    if(df.shape[0]==150000):

        d = df['acoustic_data'].values.astype('float32')

        d-=np.mean(d)

        x = denoise_signal_simple(d)

        mfcc = librosa.feature.mfcc(x, sr=sr)

        mfcc_mean = mfcc.mean(axis=1)

        for i, mfcc_mean_of_pitch in enumerate(mfcc_mean):

            key = 'mfcc_{}'.format(i)

            if key not in mfcc_ttf_map:

                mfcc_ttf_map[key] = []

            mfcc_ttf_map[key].append(mfcc_mean_of_pitch)

        

        key = 'time_to_failure'

        if key not in mfcc_ttf_map:

            mfcc_ttf_map[key] = []

        mfcc_ttf_map[key].append(df.iloc[-1][df.columns[1]])



mfcc_ttf_df = pd.DataFrame(mfcc_ttf_map)



        
import re



print('generating test features...')

test_dir = '../input/test'

test_map = {}

for fname in tqdm(os.listdir(test_dir)):

    path = test_dir + '/' + fname

    df = pd.read_csv(path)

    x = denoise_signal_simple(df['acoustic_data'].values.astype('float32'))

    mfcc = librosa.feature.mfcc(x, sr=sr)

    mfcc_mean = mfcc.mean(axis=1)

    for i, mfcc_mean_of_pitch in enumerate(mfcc_mean):

        key = 'mfcc_{}'.format(i)

        if key not in test_map:

            test_map[key] = []

        test_map[key].append(mfcc_mean_of_pitch)

    key = 'seg_id'

    if key not in test_map:

        test_map[key] = []

    test_map[key].append(re.sub('.csv$', '', fname))

test_df = pd.DataFrame(test_map)



    
from sklearn.preprocessing import StandardScaler
mfcc_ttf_df.shape
test_df.shape
mfcc_ttf_df.head()
test_df.head()
test_df['time_to_failure'] = -1
alldata = pd.concat([mfcc_ttf_df[mfcc_ttf_df.columns],test_df[mfcc_ttf_df.columns]])
alldata.head()
ss = StandardScaler()

alldata[alldata.columns[:-1]] = ss.fit_transform(alldata[alldata.columns[:-1]])
class GPLow:

    def __init__(self):

        self.classes = 2

        self.class_names = [ 'class_0',

                             'class_1']





    def GrabPredictions(self, data):

        oof_preds = np.zeros((len(data), len(self.class_names)))

        oof_preds[:,0] = self.GP_class_0(data)

        oof_preds[:,1] = self.GP_class_1(data)

       

        oof_df = pd.DataFrame(oof_preds, columns=self.class_names)

        oof_df =oof_df.div(oof_df.sum(axis=1), axis=0)

        return oof_df





    def Output(self,p):

        return 1.0/(1.0+np.exp(-p))



    def GP_class_0(self,data):

        return self.Output( 0.030800*np.tanh((((((14.12737751007080078)) + ((((data["mfcc_0"]) >= (-3.0))*1.)))) * (((((data["mfcc_0"]) + (-3.0))) + (data["mfcc_17"]))))) +

                            0.100000*np.tanh(((-3.0) + ((((12.75185775756835938)) * (((-3.0) + ((((12.75185775756835938)) * (((-3.0) + (((data["mfcc_17"]) - (data["mfcc_18"]))))))))))))) +

                            0.089200*np.tanh(((((((5.0)) < ((-1.0*(((5.0))))))*1.)) + (((((data["mfcc_1"]) + (((data["mfcc_4"]) + ((-1.0*(((5.0))))))))) * 2.0)))) +

                            0.071034*np.tanh(((((data["mfcc_4"]) - ((((9.0)) - (data["mfcc_4"]))))) - ((((data["mfcc_4"]) + ((((8.0)) * (data["mfcc_6"]))))/2.0)))) +

                            0.000004*np.tanh(((((((((data["mfcc_0"]) - (data["mfcc_2"]))) - (2.196743))) * ((14.96836853027343750)))) * (((2.196743) - (data["mfcc_0"]))))) )

    

    def GP_class_1(self,data):

        return self.Output( 0.030800*np.tanh((((5.0)) + ((((((((data["mfcc_18"]) + ((13.36389827728271484)))/2.0)) * ((((5.0)) + (((data["mfcc_18"]) * 2.0)))))) * ((((5.0)) * 2.0)))))) +

                            0.100000*np.tanh((((((((5.0)) * 2.0)) * ((((((data["mfcc_18"]) + (data["mfcc_18"]))/2.0)) + (((2.131311) - (data["mfcc_17"]))))))) + ((10.0)))) +

                            0.089200*np.tanh(((((2.0) + (data["mfcc_2"]))) * ((((((7.0)) * ((7.0)))) * ((((7.0)) - (((data["mfcc_10"]) * 2.0)))))))) +

                            0.071034*np.tanh((((((11.08603382110595703)) * (((2.131311) + ((((13.62188434600830078)) + ((11.95074272155761719)))))))) * (((2.196743) - (data["mfcc_0"]))))) +

                            0.000004*np.tanh((((11.63304042816162109)) + (((data["mfcc_6"]) + ((((10.84789657592773438)) - (((data["mfcc_0"]) * ((10.84789657592773438)))))))))) )
gp = GPLow()

preds = gp.GrabPredictions(alldata)

alldata.insert(0,'class_0',preds.class_0.values)
plt.figure(figsize=(15,15))

plt.plot(alldata[:410].time_to_failure)

plt.plot(preds[:410].class_0*10)
def GPI(data):

    return (6.044562 +

            0.100000*np.tanh((((13.93472957611083984)) * ((((((13.93472957611083984)) * ((((((((data["mfcc_1"]) + (data["mfcc_6"]))) + (data["mfcc_5"]))) + (((data["mfcc_1"]) + ((((((-1.0*((data["mfcc_13"])))) + ((((((((-1.0*((((data["mfcc_0"]) + (-0.291443)))))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)))))/2.0)))) * 2.0)))) +

            0.100000*np.tanh((((13.78113555908203125)) * ((((13.78113555908203125)) * (((data["mfcc_1"]) + (((((data["mfcc_0"]) + (((((((((((data["mfcc_1"]) - (((((((data["mfcc_0"]) * 2.0)) - ((((data["mfcc_10"]) + (((((8.0)) >= (data["mfcc_1"]))*1.)))/2.0)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) * 2.0)))))))) +

            0.100000*np.tanh((((((((((((((((((((((((((((data["mfcc_3"]) < (data["mfcc_8"]))*1.)) + (((data["mfcc_15"]) * 2.0)))) + (((data["mfcc_14"]) * 2.0)))) + (data["mfcc_6"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (data["class_0"]))) + ((((data["class_0"]) >= (data["mfcc_15"]))*1.)))) * 2.0)) +

            0.100000*np.tanh(((((((((((data["mfcc_5"]) + (((((((((((data["mfcc_5"]) + ((((((((data["mfcc_6"]) + (2.108835))/2.0)) - (((((data["mfcc_0"]) * 2.0)) * (((0.670573) * 2.0)))))) * 2.0)))) * 2.0)) - (data["mfcc_17"]))) * 2.0)) - (data["mfcc_6"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((((((((data["mfcc_11"]) + ((((((data["class_0"]) + (data["mfcc_14"]))/2.0)) + (((((data["mfcc_14"]) + (((((data["mfcc_6"]) + (data["mfcc_14"]))) - ((((data["mfcc_4"]) + (data["mfcc_3"]))/2.0)))))) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * ((((7.0)) - (data["mfcc_3"]))))) +

            0.100000*np.tanh((((-1.0*((((((((data["mfcc_18"]) - ((((-1.0*((((((data["mfcc_9"]) - ((((((((((-2.0) < (data["mfcc_12"]))*1.)) + (((((((data["mfcc_10"]) / 2.0)) - ((((5.97305536270141602)) * (data["mfcc_0"]))))) / 2.0)))) * 2.0)) * 2.0)))) * 2.0))))) * 2.0)))) * 2.0)) * 2.0))))) * 2.0)) +

            0.100000*np.tanh(((((((data["mfcc_6"]) + (data["mfcc_1"]))) * ((12.51284027099609375)))) + ((((((12.51284408569335938)) * (((data["mfcc_10"]) - (((((((-0.270785) + (data["mfcc_0"]))) * (((2.264782) + ((((-0.270785) >= (((data["mfcc_6"]) + (data["mfcc_6"]))))*1.)))))) * 2.0)))))) - (1.697380))))) +

            0.100000*np.tanh((((12.71692085266113281)) * (((((((((data["mfcc_1"]) * 2.0)) - (((((((2.0) + ((((data["mfcc_14"]) >= (data["mfcc_4"]))*1.)))) + ((((2.0) >= ((-1.0*((data["mfcc_4"])))))*1.)))) * (data["mfcc_11"]))))) + ((-1.0*((-3.0)))))) - (((data["mfcc_0"]) * ((12.71692085266113281)))))))) +

            0.100000*np.tanh(((((((((2.0) - (((((((data["mfcc_13"]) - ((((((data["mfcc_13"]) * 2.0)) < (data["mfcc_1"]))*1.)))) - (data["mfcc_1"]))) + ((((((8.57544231414794922)) + ((((((data["mfcc_15"]) < (((-2.0) / 2.0)))*1.)) * ((8.57544231414794922)))))) * (data["mfcc_0"]))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((((((((((((((((data["mfcc_1"]) >= (data["mfcc_9"]))*1.)) + (((2.549079) + ((((-1.0*((data["mfcc_0"])))) * 2.0)))))/2.0)) + (((((data["mfcc_10"]) + (data["mfcc_1"]))) + ((((((-1.0*((data["mfcc_0"])))) * 2.0)) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((data["mfcc_13"]) + ((((((((((data["mfcc_7"]) < ((-1.0*((data["mfcc_13"])))))*1.)) + (((data["mfcc_5"]) + ((((((((data["mfcc_7"]) * (data["mfcc_5"]))) < (2.549079))*1.)) - (((2.223052) * (((data["mfcc_0"]) * 2.0)))))))))) * 2.0)) * 2.0)))) * 2.0)) +

            0.100000*np.tanh(((((((data["mfcc_3"]) - (((((((((((((-1.0*((((data["mfcc_0"]) * 2.0))))) + (((((((data["mfcc_6"]) + (data["class_0"]))/2.0)) >= (((data["mfcc_9"]) - (((data["mfcc_5"]) + (data["class_0"]))))))*1.)))/2.0)) * (-3.0))) * 2.0)) * 2.0)) * 2.0)))) + (data["mfcc_1"]))) * 2.0)) +

            0.100000*np.tanh((((12.07313346862792969)) * ((((((((((7.0)) * (((((1.697380) - ((((((data["mfcc_1"]) < (data["mfcc_11"]))*1.)) - (((data["mfcc_0"]) * ((-1.0*(((7.0))))))))))) + (data["mfcc_7"]))))) + (data["mfcc_7"]))) * 2.0)) + (data["mfcc_4"]))))) +

            0.100000*np.tanh((((((((((((((data["mfcc_11"]) < (data["mfcc_0"]))*1.)) + (((((data["mfcc_1"]) * 2.0)) - (((-2.0) + (data["mfcc_0"]))))))) + (((((((data["mfcc_5"]) + (1.697380))) - (((((data["mfcc_0"]) + (((data["mfcc_0"]) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((((((data["mfcc_7"]) + (data["mfcc_8"]))) + (((((((((data["mfcc_7"]) + (data["mfcc_11"]))) + (((((((data["mfcc_7"]) + (data["mfcc_11"]))) + ((((((data["mfcc_11"]) / 2.0)) < (((data["mfcc_6"]) * 2.0)))*1.)))) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh((((((data["mfcc_8"]) < (-0.088246))*1.)) + ((((((((((data["mfcc_11"]) < ((((((data["mfcc_0"]) < (-0.088246))*1.)) * 2.0)))*1.)) - (((data["mfcc_13"]) + (((data["mfcc_7"]) + (data["mfcc_13"]))))))) * 2.0)) - (((data["mfcc_0"]) * ((12.66860198974609375)))))))) +

            0.100000*np.tanh((((10.62614631652832031)) * (((((data["mfcc_15"]) + ((((((((((data["mfcc_15"]) * 2.0)) + (((((((2.193272) + (data["mfcc_18"]))) + (data["mfcc_18"]))) + (((((data["mfcc_14"]) + (((data["mfcc_7"]) + (data["mfcc_18"]))))) * 2.0)))))/2.0)) * 2.0)) * 2.0)))) * 2.0)))) +

            0.100000*np.tanh((((((((((5.0)) * ((((((5.0)) * (data["mfcc_14"]))) - (((data["mfcc_10"]) * 2.0)))))) + ((((((((data["mfcc_19"]) >= (data["mfcc_10"]))*1.)) + ((((((((data["mfcc_14"]) >= (data["mfcc_19"]))*1.)) + (data["mfcc_19"]))) * 2.0)))) + (((data["mfcc_19"]) * 2.0)))))) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((data["mfcc_1"]) + (((((((2.193272) * ((((((((((data["mfcc_0"]) * (2.223052))) - ((((((2.364957) * (data["mfcc_1"]))) < (2.236772))*1.)))) >= (data["mfcc_8"]))*1.)) - (((data["mfcc_0"]) * (2.193272))))))) - (data["mfcc_11"]))) * 2.0)))) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((data["mfcc_5"]) * 2.0)) + (((data["mfcc_5"]) + ((((((((13.37955856323242188)) - (((((data["mfcc_19"]) + (data["mfcc_9"]))) * 2.0)))) - (data["mfcc_6"]))) - (((data["mfcc_0"]) * ((((((((13.37955856323242188)) - ((((data["mfcc_10"]) >= (data["mfcc_12"]))*1.)))) - (data["mfcc_1"]))) * 2.0)))))))))) +

            0.100000*np.tanh(((((((data["mfcc_1"]) + (((data["mfcc_13"]) * 2.0)))) + (data["mfcc_13"]))) - ((((13.09366035461425781)) * (((((data["mfcc_13"]) - ((((((((-1.0*((((((((((data["mfcc_0"]) * 2.0)) * 2.0)) + (data["mfcc_13"]))) * 2.0))))) + (data["mfcc_1"]))) * 2.0)) * 2.0)))) * 2.0)))))) +

            0.100000*np.tanh(((((((data["mfcc_5"]) - (((data["mfcc_17"]) - (((data["mfcc_6"]) - (((((data["mfcc_3"]) - (((((((0.614120) < (data["mfcc_3"]))*1.)) >= ((((data["mfcc_17"]) >= (0.614120))*1.)))*1.)))) + ((((data["mfcc_1"]) + ((((-1.0*((((data["mfcc_15"]) * 2.0))))) * 2.0)))/2.0)))))))))) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((((data["mfcc_7"]) + ((((((((((data["mfcc_15"]) < (((data["mfcc_0"]) + (((-1.0) - (((data["mfcc_14"]) + (data["mfcc_14"]))))))))*1.)) * 2.0)) - (((((data["mfcc_0"]) * (((data["mfcc_0"]) + (((2.236772) * 2.0)))))) / 2.0)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((((((data["mfcc_6"]) + (((data["mfcc_18"]) - (((data["mfcc_4"]) - (((data["mfcc_7"]) * ((3.51214265823364258)))))))))) + (((-2.0) * (((((((3.51214265823364258)) + (data["mfcc_4"]))/2.0)) * (((data["mfcc_4"]) * ((((data["mfcc_15"]) < (data["mfcc_1"]))*1.)))))))))) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((data["mfcc_10"]) - (((((((((((data["mfcc_0"]) * 2.0)) + ((((-3.0) + ((((((((((data["mfcc_1"]) - ((((((((data["mfcc_0"]) * 2.0)) < (data["mfcc_0"]))*1.)) + (data["mfcc_17"]))))) < (data["mfcc_13"]))*1.)) * 2.0)) * 2.0)))/2.0)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

            0.100000*np.tanh((((7.0)) * ((((((data["mfcc_5"]) + (((((((data["mfcc_3"]) >= (((data["mfcc_15"]) + (data["mfcc_14"]))))*1.)) >= (((data["mfcc_15"]) + (data["mfcc_14"]))))*1.)))/2.0)) + ((((((((data["mfcc_3"]) / 2.0)) >= (((data["mfcc_15"]) + (data["mfcc_14"]))))*1.)) - (((data["mfcc_0"]) * 2.0)))))))) +

            0.100000*np.tanh(((((((((data["mfcc_18"]) - ((((((((((((((data["mfcc_1"]) >= (data["mfcc_18"]))*1.)) < (data["mfcc_1"]))*1.)) - (data["mfcc_1"]))) < (data["mfcc_19"]))*1.)) + (data["mfcc_5"]))))) * 2.0)) + ((((((7.0)) * (((data["mfcc_7"]) - ((((data["mfcc_1"]) >= (data["mfcc_19"]))*1.)))))) * 2.0)))) * 2.0)) +

            0.100000*np.tanh((((((((((((-3.0) * (data["mfcc_8"]))) >= (data["mfcc_0"]))*1.)) + (((((((-2.0) + ((((((((((((-3.0) * (data["mfcc_0"]))) + (data["mfcc_8"]))/2.0)) + ((((((data["mfcc_0"]) >= (data["mfcc_14"]))*1.)) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((data["mfcc_0"]) + (((((((((((((((0.735529) + (data["mfcc_0"]))) + (((((((0.427750) + ((-1.0*(((((((data["mfcc_0"]) + ((((data["mfcc_13"]) >= (((0.302810) / 2.0)))*1.)))/2.0)) * 2.0))))))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) +

            0.100000*np.tanh(((((((data["mfcc_1"]) * 2.0)) * 2.0)) + (((data["mfcc_10"]) - ((((-1.0*(((((((0.427751) < (data["mfcc_1"]))*1.)) - (((data["mfcc_0"]) * 2.0))))))) * ((((4.40433835983276367)) + ((((4.40433835983276367)) * (((((((((data["mfcc_0"]) * 2.0)) >= (data["mfcc_1"]))*1.)) < (data["mfcc_1"]))*1.)))))))))))) +

            0.100000*np.tanh(((((((((((-1.0) + (((data["mfcc_6"]) + (((data["mfcc_14"]) * 2.0)))))) * 2.0)) * 2.0)) + ((((((data["mfcc_15"]) * 2.0)) >= (((data["mfcc_18"]) - (data["mfcc_14"]))))*1.)))) - (((-1.0) + (((data["mfcc_14"]) + (((((data["mfcc_15"]) * (data["mfcc_15"]))) * 2.0)))))))) +

            0.100000*np.tanh((((((((((((data["mfcc_0"]) < ((((((data["mfcc_8"]) * (((data["mfcc_8"]) * (data["mfcc_0"]))))) < ((((data["mfcc_1"]) >= (((data["mfcc_6"]) * ((((data["mfcc_7"]) >= (data["mfcc_14"]))*1.)))))*1.)))*1.)))*1.)) - ((((((data["mfcc_8"]) >= (data["mfcc_0"]))*1.)) + (data["mfcc_0"]))))) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((3.0) + ((((((((((((2.223052) >= (data["mfcc_18"]))*1.)) - (data["mfcc_9"]))) + (((3.0) * ((((-1.0*((data["mfcc_0"])))) - (((((((((data["mfcc_0"]) < (data["mfcc_0"]))*1.)) - (data["mfcc_14"]))) < (data["mfcc_18"]))*1.)))))))) * 2.0)) * 2.0)))) * 2.0)) +

            0.100000*np.tanh(((((((data["mfcc_14"]) + ((((((((((((((data["class_0"]) * (-1.0))) < (data["mfcc_14"]))*1.)) * 2.0)) < (data["mfcc_18"]))*1.)) >= ((((data["class_0"]) >= (((((data["mfcc_14"]) * 2.0)) * (data["mfcc_3"]))))*1.)))*1.)))) * 2.0)) - ((((((data["mfcc_18"]) < (((data["mfcc_3"]) * 2.0)))*1.)) * 2.0)))) +

            0.100000*np.tanh((((((((((data["mfcc_0"]) * 2.0)) + (((data["mfcc_5"]) + (((((((((0.735529) - (((data["mfcc_0"]) + (((((2.22410607337951660)) >= (((((((data["mfcc_0"]) * (2.030533))) + ((((data["mfcc_10"]) < (data["mfcc_5"]))*1.)))) * 2.0)))*1.)))))) * 2.0)) * 2.0)) * 2.0)))))/2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((data["mfcc_15"]) + ((((((((((((((((((data["mfcc_6"]) < (data["mfcc_1"]))*1.)) - (data["mfcc_0"]))) - (((((data["mfcc_2"]) * (data["mfcc_1"]))) * 2.0)))) * ((((data["mfcc_1"]) < ((((data["mfcc_6"]) + (((data["mfcc_2"]) * (data["mfcc_2"]))))/2.0)))*1.)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +

            0.100000*np.tanh((-1.0*((((((((((data["mfcc_0"]) >= (1.105127))*1.)) >= ((((data["mfcc_0"]) >= (((data["mfcc_14"]) * (((data["mfcc_0"]) - ((((data["mfcc_0"]) >= (data["mfcc_7"]))*1.)))))))*1.)))*1.)) - ((-1.0*((((data["mfcc_0"]) - ((((data["mfcc_0"]) >= (((data["mfcc_14"]) * (data["mfcc_0"]))))*1.)))))))))))) +

            0.100000*np.tanh(((data["mfcc_0"]) * ((((((data["mfcc_14"]) >= (data["mfcc_3"]))*1.)) - ((((((((data["mfcc_8"]) * (data["mfcc_8"]))) + (((((data["mfcc_2"]) + (((data["mfcc_15"]) + (data["mfcc_2"]))))) * (((data["mfcc_8"]) + (data["mfcc_2"]))))))) + (((((data["mfcc_3"]) + (data["mfcc_12"]))) * 2.0)))/2.0)))))) +

            0.100000*np.tanh((((((((((data["mfcc_1"]) * ((((((((data["mfcc_18"]) < (data["mfcc_2"]))*1.)) - (data["mfcc_2"]))) * 2.0)))) * 2.0)) + (data["mfcc_14"]))/2.0)) + (((data["mfcc_6"]) + (((data["mfcc_14"]) * (((data["mfcc_4"]) + ((((data["mfcc_14"]) < ((((data["mfcc_2"]) + (data["mfcc_4"]))/2.0)))*1.)))))))))) +

            0.100000*np.tanh(((((((((((((((((((data["mfcc_7"]) >= (data["mfcc_15"]))*1.)) < (((data["mfcc_8"]) * 2.0)))*1.)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * (((((data["mfcc_15"]) * (((((data["mfcc_8"]) * 2.0)) * (((data["mfcc_5"]) + ((((data["mfcc_18"]) + (0.570601))/2.0)))))))) + (-1.0))))) +

            0.100000*np.tanh(((((data["mfcc_1"]) * ((((-1.0*((((data["mfcc_2"]) + (((data["mfcc_2"]) + (data["mfcc_1"])))))))) * ((((((data["mfcc_0"]) - ((((((data["mfcc_0"]) / 2.0)) < (data["mfcc_0"]))*1.)))) >= (data["mfcc_1"]))*1.)))))) - (((data["mfcc_0"]) - ((((0.595298) < (data["mfcc_0"]))*1.)))))) +

            0.100000*np.tanh(((data["mfcc_1"]) - (((data["mfcc_0"]) * (((((data["mfcc_0"]) + (data["mfcc_0"]))) * ((-1.0*((((((data["mfcc_5"]) + (data["mfcc_1"]))) - ((-1.0*((((data["mfcc_2"]) - (((data["mfcc_0"]) * (((data["mfcc_0"]) * (data["mfcc_0"]))))))))))))))))))))) +

            0.100000*np.tanh(((data["mfcc_15"]) * (((data["mfcc_3"]) + ((((((data["mfcc_15"]) * (data["mfcc_7"]))) + ((((((((((data["mfcc_7"]) >= (data["class_0"]))*1.)) * 2.0)) * 2.0)) * (((((data["mfcc_7"]) + (((((data["mfcc_15"]) + (data["mfcc_1"]))) + (data["mfcc_1"]))))) + (((data["mfcc_2"]) * 2.0)))))))/2.0)))))) +

            0.100000*np.tanh(((((((3.0) * (((((data["mfcc_2"]) + (((data["mfcc_6"]) / 2.0)))) * (((((((data["mfcc_6"]) >= (((((((((0.388910) - ((((data["mfcc_14"]) < (((data["mfcc_19"]) / 2.0)))*1.)))) < (data["mfcc_8"]))*1.)) < (data["mfcc_14"]))*1.)))*1.)) < ((-1.0*((data["mfcc_0"])))))*1.)))))) * 2.0)) * 2.0)) +

            0.100000*np.tanh((-1.0*((((data["mfcc_1"]) * (((data["mfcc_18"]) + (((((((((data["mfcc_7"]) / 2.0)) + (data["mfcc_7"]))) / 2.0)) + (((data["mfcc_1"]) + (((data["mfcc_2"]) * ((((data["mfcc_7"]) >= ((((data["mfcc_3"]) < (((data["mfcc_1"]) + ((((data["mfcc_7"]) + (data["mfcc_18"]))/2.0)))))*1.)))*1.))))))))))))))) +

            0.100000*np.tanh((((((((data["mfcc_2"]) >= ((((data["mfcc_12"]) >= ((((data["mfcc_1"]) >= (data["mfcc_0"]))*1.)))*1.)))*1.)) * (((((((data["mfcc_3"]) >= ((((5.96877813339233398)) * (data["mfcc_3"]))))*1.)) + ((((5.96877813339233398)) * (data["mfcc_3"]))))/2.0)))) - ((((data["mfcc_12"]) >= ((((data["mfcc_1"]) >= (data["mfcc_0"]))*1.)))*1.)))) +

            0.100000*np.tanh((-1.0*(((((data["mfcc_17"]) + (((((((((data["mfcc_0"]) * 2.0)) * 2.0)) * 2.0)) * ((((data["mfcc_0"]) < (((((-0.474062) - ((((data["mfcc_0"]) < (((data["mfcc_12"]) - ((((data["mfcc_2"]) < (data["mfcc_17"]))*1.)))))*1.)))) + (((data["mfcc_2"]) * (data["mfcc_17"]))))))*1.)))))/2.0))))) +

            0.100000*np.tanh(((data["mfcc_15"]) - (((((((data["mfcc_13"]) + (data["mfcc_12"]))) - ((((((data["mfcc_7"]) * (((data["mfcc_19"]) * (data["mfcc_7"]))))) + (((data["mfcc_7"]) - (((data["mfcc_11"]) + (data["mfcc_17"]))))))/2.0)))) - (((data["mfcc_15"]) - (((data["mfcc_13"]) + (data["mfcc_12"]))))))))) +

            0.100000*np.tanh(((((data["mfcc_2"]) * (3.0))) * ((((data["mfcc_1"]) < ((-1.0*(((((((((((data["mfcc_2"]) * (data["mfcc_1"]))) * ((((data["mfcc_4"]) < ((((data["mfcc_1"]) < (data["mfcc_2"]))*1.)))*1.)))) * ((((-0.937055) < (data["mfcc_1"]))*1.)))) < (0.585258))*1.))))))*1.)))) +

            0.100000*np.tanh((((((((data["mfcc_1"]) - (data["mfcc_19"]))) + (((((((data["mfcc_13"]) - (data["mfcc_1"]))) * (data["mfcc_17"]))) * (data["mfcc_2"]))))/2.0)) + ((((((((((((((data["mfcc_0"]) < (data["mfcc_1"]))*1.)) * ((((data["mfcc_3"]) + (data["mfcc_19"]))/2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))))

         
def GPII(data):

    return (6.044562 +

            0.100000*np.tanh((((13.85196399688720703)) * (((0.670573) - ((-1.0*((((((((((data["mfcc_14"]) + ((((data["mfcc_13"]) < (2.0))*1.)))) - (((data["mfcc_0"]) * ((((((data["mfcc_9"]) >= ((((data["mfcc_11"]) >= (0.670573))*1.)))*1.)) - (-3.0))))))) * 2.0)) * 2.0))))))))) +

            0.100000*np.tanh((((10.91737270355224609)) * ((((10.91737270355224609)) * (((((((3.0) + (((((data["mfcc_1"]) + ((-1.0*((((data["mfcc_15"]) - ((-1.0*((((data["mfcc_15"]) - (((data["mfcc_14"]) * 2.0)))))))))))))) - (((data["mfcc_0"]) * ((((10.91737270355224609)) + (data["mfcc_1"]))))))))) * 2.0)) * 2.0)))))) +

            0.100000*np.tanh(((((((((((((data["mfcc_12"]) - (((((data["mfcc_11"]) - (((data["mfcc_1"]) + ((-1.0*((((((((((data["mfcc_0"]) * 2.0)) - (0.427750))) * 2.0)) * 2.0))))))))) * 2.0)))) * 2.0)) + ((((data["mfcc_1"]) < ((((data["mfcc_0"]) < (data["mfcc_1"]))*1.)))*1.)))) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh((((((((((((3.69547343254089355)) * ((((((((((((((3.69547343254089355)) >= (data["mfcc_0"]))*1.)) + ((((data["mfcc_1"]) + (((data["mfcc_5"]) + (data["mfcc_0"]))))/2.0)))/2.0)) - (((data["mfcc_0"]) * 2.0)))) * 2.0)) * 2.0)))) + (((-3.0) * (data["mfcc_11"]))))) + (data["mfcc_12"]))) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((3.0) * (((((((data["mfcc_1"]) + (((((((((data["mfcc_14"]) + ((((((((((((3.0) * ((((-1.0*((data["mfcc_0"])))) * 2.0)))) + (data["mfcc_5"]))/2.0)) * 2.0)) + (((2.236772) + (data["mfcc_1"]))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)))) +

            0.100000*np.tanh(((((((13.57778072357177734)) + (3.0))) + (((((((((-2.0) * 2.0)) * (data["mfcc_11"]))) - (-2.0))) - ((((((12.16242599487304688)) * (((data["mfcc_0"]) - ((((((data["mfcc_1"]) - (data["mfcc_13"]))) + (((data["mfcc_0"]) * (((-2.0) * 2.0)))))/2.0)))))) * 2.0)))))/2.0)) +

            0.100000*np.tanh((((((((((((((((((((-1.0*((((data["mfcc_0"]) * 2.0))))) + ((((data["mfcc_11"]) < ((((((data["mfcc_0"]) * 2.0)) < (((((data["mfcc_6"]) - (data["mfcc_0"]))) - (-0.270785))))*1.)))*1.)))) * 2.0)) * 2.0)) + (data["mfcc_6"]))) - (data["mfcc_0"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh((((12.47262859344482422)) * (((((((((-1.0*((data["mfcc_6"])))) >= (data["mfcc_0"]))*1.)) + (((data["mfcc_10"]) + ((((((1.0) * 2.0)) + ((((((((data["mfcc_6"]) + ((((((((-1.0*((data["mfcc_0"])))) * 2.0)) * 2.0)) + (data["mfcc_1"]))))/2.0)) * 2.0)) * 2.0)))/2.0)))))) * 2.0)))) +

            0.100000*np.tanh((((((((((((((-1.0*((data["mfcc_0"])))) * 2.0)) + ((((((((data["mfcc_12"]) < (((data["mfcc_7"]) * 2.0)))*1.)) + (((2.0) + (((((data["mfcc_10"]) * 2.0)) + ((((((((-1.0*((data["mfcc_0"])))) * 2.0)) * 2.0)) * 2.0)))))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((((((((((((((data["mfcc_18"]) + (((((((data["mfcc_6"]) + (data["mfcc_14"]))) + ((((data["mfcc_15"]) + ((((((data["mfcc_15"]) / 2.0)) + ((((((data["mfcc_15"]) / 2.0)) < (data["class_0"]))*1.)))/2.0)))/2.0)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh((((((13.99419975280761719)) * (((data["mfcc_0"]) + (((data["mfcc_5"]) + (((((((((((((((data["mfcc_5"]) + (((1.697380) - ((((data["mfcc_1"]) < (data["mfcc_0"]))*1.)))))/2.0)) >= (data["mfcc_0"]))*1.)) - (((data["mfcc_0"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))))))) * 2.0)) +

            0.100000*np.tanh(((((((((((((((data["mfcc_10"]) + (3.0))) - ((((8.0)) * (data["mfcc_0"]))))) * 2.0)) - (((((data["mfcc_3"]) * 2.0)) - ((((((2.0) + (((data["mfcc_8"]) * 2.0)))) >= (data["mfcc_0"]))*1.)))))) * 2.0)) * 2.0)) + ((((data["mfcc_8"]) >= (data["mfcc_3"]))*1.)))) +

            0.100000*np.tanh((((((((11.03212165832519531)) - ((((data["mfcc_0"]) >= (((data["mfcc_14"]) + (data["mfcc_14"]))))*1.)))) + (((2.0) * ((((10.0)) * (((((data["mfcc_14"]) - (((data["mfcc_14"]) * (data["mfcc_6"]))))) - (((((3.0) + (data["mfcc_6"]))) * (data["mfcc_0"]))))))))))) * 2.0)) +

            0.100000*np.tanh(((((((((((((-3.0) * (((((data["mfcc_13"]) - (((((-3.0) * (((((data["mfcc_0"]) - ((((((data["mfcc_13"]) < ((((((data["mfcc_0"]) * 2.0)) < (data["mfcc_1"]))*1.)))*1.)) / 2.0)))) * 2.0)))) * 2.0)))) * 2.0)))) + (data["mfcc_1"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((data["mfcc_14"]) + ((-1.0*((((((2.223052) - (((((((data["mfcc_14"]) < (((data["mfcc_4"]) * (((data["mfcc_15"]) * (data["mfcc_15"]))))))*1.)) < (((data["mfcc_14"]) * (data["mfcc_14"]))))*1.)))) * ((((data["mfcc_4"]) + ((-1.0*((data["mfcc_6"])))))/2.0))))))))) * 2.0)) * ((10.0)))) +

            0.100000*np.tanh((((((((((((((((((((((((2.549079) * ((((((data["mfcc_0"]) >= ((((data["mfcc_12"]) >= (data["mfcc_1"]))*1.)))*1.)) - (((data["mfcc_0"]) * 2.0)))))) >= (data["mfcc_11"]))*1.)) - (data["mfcc_0"]))) * 2.0)) * 2.0)) - (data["mfcc_11"]))) * 2.0)) * 2.0)) * 2.0)) - (data["mfcc_11"]))) * 2.0)) +

            0.100000*np.tanh(((((((((((((((((2.0) + (data["mfcc_1"]))) * ((((data["mfcc_11"]) < (((0.023605) * (2.236772))))*1.)))) * ((((data["mfcc_10"]) < (data["mfcc_0"]))*1.)))) + (((data["mfcc_10"]) - (((((data["mfcc_0"]) * 2.0)) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh((((8.0)) * ((((((((((((((((((data["mfcc_5"]) + ((((5.0)) / 2.0)))/2.0)) * (2.415960))) - (((data["mfcc_0"]) * ((5.06101179122924805)))))) * (2.415960))) - (data["mfcc_7"]))) * ((7.10873889923095703)))) - ((((data["mfcc_5"]) < (data["mfcc_7"]))*1.)))) - (data["mfcc_7"]))))) +

            0.100000*np.tanh((((14.67224884033203125)) * ((((((14.67224884033203125)) * (((((((((data["mfcc_8"]) + (((data["mfcc_15"]) + (data["mfcc_6"]))))) * 2.0)) * 2.0)) + ((((((((data["mfcc_18"]) * 2.0)) * 2.0)) < (data["mfcc_15"]))*1.)))))) + (((-3.0) + ((((data["mfcc_8"]) < (data["mfcc_6"]))*1.)))))))) +

            0.100000*np.tanh(((((data["mfcc_1"]) - (((((((data["mfcc_0"]) - (data["mfcc_11"]))) - (data["mfcc_11"]))) * (data["mfcc_1"]))))) - (((((((data["mfcc_11"]) - ((((((((data["mfcc_0"]) * (-3.0))) + (data["mfcc_1"]))/2.0)) * 2.0)))) * ((9.14199638366699219)))) - ((((data["mfcc_0"]) + ((9.14199638366699219)))/2.0)))))) +

            0.100000*np.tanh(((((((((((data["mfcc_0"]) * 2.0)) * 2.0)) + ((((((-1.0*((data["mfcc_13"])))) + ((((((((((data["mfcc_14"]) < ((-1.0*((data["mfcc_13"])))))*1.)) * 2.0)) - (((((data["mfcc_0"]) * 2.0)) * 2.0)))) * 2.0)))) * ((8.0)))))) + (((((data["mfcc_0"]) * 2.0)) * 2.0)))) * 2.0)) +

            0.100000*np.tanh((((((6.35433721542358398)) * ((((((6.35433721542358398)) * (((((((((((((data["mfcc_11"]) + (data["mfcc_7"]))) * 2.0)) * 2.0)) + (((((((((data["mfcc_11"]) + (data["mfcc_7"]))) * (data["mfcc_7"]))) + (data["mfcc_7"]))) * 2.0)))) * 2.0)) * 2.0)))) + (data["mfcc_7"]))))) + (data["mfcc_7"]))) +

            0.100000*np.tanh(((((((data["mfcc_8"]) + ((((((0.427750) + ((((data["mfcc_10"]) < (data["mfcc_8"]))*1.)))/2.0)) + ((((9.0)) * (((data["mfcc_15"]) + ((((((data["mfcc_14"]) + ((((((((data["mfcc_10"]) * 2.0)) + (data["mfcc_15"]))) < (data["mfcc_14"]))*1.)))/2.0)) * 2.0)))))))))) * 2.0)) * ((8.0)))) +

            0.100000*np.tanh((((((((((((((((((((((-1.0*((data["mfcc_0"])))) * 2.0)) + ((((((((((data["mfcc_14"]) + (data["class_0"]))/2.0)) + (data["class_0"]))/2.0)) < (data["mfcc_0"]))*1.)))) * 2.0)) + ((-1.0*((data["mfcc_9"])))))) * 2.0)) + (data["mfcc_0"]))) * 2.0)) + (data["mfcc_9"]))) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((-3.0) + (((((((data["mfcc_10"]) - (((((((data["mfcc_0"]) - ((((data["mfcc_0"]) >= (((data["mfcc_14"]) + ((((0.174589) + (((((((((0.174589) + (2.364957))/2.0)) * (data["mfcc_10"]))) < (data["mfcc_0"]))*1.)))/2.0)))))*1.)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)))) +

            0.100000*np.tanh((((((((-1.0*((data["mfcc_0"])))) + (((((((data["mfcc_7"]) + ((((-1.0*((data["mfcc_4"])))) - ((-1.0*(((((((((((-1.0*((data["mfcc_0"])))) * 2.0)) - ((((data["mfcc_11"]) >= ((-1.0*((data["mfcc_0"])))))*1.)))) * 2.0)) * 2.0))))))))) * 2.0)) - (data["mfcc_11"]))))) * 2.0)) * 2.0)) +

            0.100000*np.tanh((((13.40357017517089844)) * (((data["mfcc_8"]) + ((((((13.40357017517089844)) + (2.364957))) * (((data["mfcc_8"]) + (((((((2.364957) + (data["mfcc_7"]))) - (data["mfcc_11"]))) * (((((((1.697380) - (((data["mfcc_0"]) * (2.0))))) * 2.0)) * 2.0)))))))))))) +

            0.100000*np.tanh((((((((((-1.0*((((data["mfcc_9"]) * (((((5.0)) + (((data["mfcc_9"]) * (((((5.0)) + (((data["mfcc_0"]) / 2.0)))/2.0)))))/2.0))))))) + (((-3.0) + (((((data["mfcc_0"]) * (((((-3.0) * 2.0)) - (data["mfcc_6"]))))) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((data["mfcc_0"]) - (((data["mfcc_17"]) - (data["mfcc_7"]))))) + (((((((data["mfcc_0"]) + (((data["mfcc_7"]) + (((3.0) * (((((((0.764911) - ((((((data["mfcc_17"]) - (data["mfcc_19"]))) >= (data["mfcc_0"]))*1.)))) - (data["mfcc_0"]))) * 2.0)))))))) * 2.0)) * 2.0)))) +

            0.100000*np.tanh(((((((2.030533) - (((2.0) * (((data["mfcc_0"]) + (((data["mfcc_0"]) + (data["mfcc_3"]))))))))) * 2.0)) + (((data["mfcc_14"]) + ((((((data["mfcc_0"]) < ((((((1.654606) >= (1.654606))*1.)) - ((((data["mfcc_11"]) + (data["mfcc_0"]))/2.0)))))*1.)) * 2.0)))))) +

            0.100000*np.tanh((((((8.61900520324707031)) * ((((8.61900520324707031)) * (((((data["mfcc_6"]) + (((data["mfcc_7"]) - (((((((data["mfcc_11"]) < (0.570601))*1.)) + ((((data["mfcc_7"]) < ((((data["mfcc_6"]) < ((((data["mfcc_19"]) < (0.369547))*1.)))*1.)))*1.)))/2.0)))))) - (0.369547))))))) - (data["mfcc_7"]))) +

            0.100000*np.tanh(((((((data["mfcc_1"]) + (((((data["mfcc_1"]) + ((((((((((data["mfcc_6"]) < (data["mfcc_1"]))*1.)) - (((data["mfcc_9"]) + (((((((((((0.0) >= (data["mfcc_6"]))*1.)) < (data["mfcc_0"]))*1.)) + (data["mfcc_0"]))) * 2.0)))))) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((((((((((((data["mfcc_8"]) + (((data["mfcc_1"]) + (((((((data["mfcc_8"]) + ((((4.0)) - (((((data["mfcc_0"]) * 2.0)) * 2.0)))))) * 2.0)) * 2.0)))))) * 2.0)) - (data["mfcc_18"]))) * 2.0)) - (((data["mfcc_18"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((((((((((((((data["mfcc_15"]) - (((((((data["mfcc_15"]) >= (data["mfcc_1"]))*1.)) >= (data["mfcc_15"]))*1.)))) - (((data["mfcc_2"]) * (((3.0) * (data["mfcc_1"]))))))) + (data["mfcc_5"]))) + (((data["mfcc_15"]) - ((((0.724339) >= (data["mfcc_2"]))*1.)))))) * 2.0)) * 2.0)) * 2.0)) +

            0.100000*np.tanh(((data["mfcc_15"]) * (((((data["mfcc_3"]) + (((data["mfcc_6"]) * (2.549079))))) * (((((data["mfcc_15"]) + (((((data["mfcc_14"]) + (((((-1.0*((data["mfcc_14"])))) >= (data["mfcc_6"]))*1.)))) + (((((((data["mfcc_3"]) + (data["mfcc_15"]))/2.0)) >= (((data["mfcc_3"]) * 2.0)))*1.)))))) * 2.0)))))) +

            0.100000*np.tanh(((((((data["mfcc_18"]) + (((data["mfcc_2"]) - (((data["mfcc_0"]) * (data["mfcc_2"]))))))) + (data["mfcc_5"]))) + ((((((((data["mfcc_0"]) + ((((-1.0) < (((1.697380) * (data["mfcc_0"]))))*1.)))) / 2.0)) < ((((data["mfcc_2"]) >= (((data["mfcc_8"]) * 2.0)))*1.)))*1.)))) +

            0.100000*np.tanh(((((((data["mfcc_15"]) + (data["mfcc_14"]))) + ((((data["mfcc_9"]) < (data["mfcc_15"]))*1.)))) * (((data["mfcc_3"]) + (((((data["mfcc_4"]) + (((data["mfcc_8"]) * ((((data["mfcc_8"]) + (data["mfcc_15"]))/2.0)))))) - (((((((data["mfcc_15"]) < (data["mfcc_9"]))*1.)) + (((data["mfcc_15"]) / 2.0)))/2.0)))))))) +

            0.100000*np.tanh((((((((((((((((((data["mfcc_1"]) + ((((((data["mfcc_0"]) * (((((data["mfcc_0"]) * 2.0)) * 2.0)))) >= (0.570601))*1.)))/2.0)) >= (0.694219))*1.)) + ((-1.0*((data["mfcc_0"])))))/2.0)) * 2.0)) * 2.0)) * 2.0)) * (((data["mfcc_0"]) + (((((data["mfcc_0"]) * 2.0)) * (data["mfcc_0"]))))))) +

            0.100000*np.tanh(((((data["mfcc_0"]) * 2.0)) * ((-1.0*(((((-1.0*((data["mfcc_0"])))) * (((((1.697380) + (data["mfcc_15"]))) * (((data["mfcc_2"]) + ((((-1.0*((data["mfcc_0"])))) + ((((data["mfcc_1"]) + ((((data["mfcc_1"]) + ((((data["mfcc_3"]) < (data["mfcc_1"]))*1.)))/2.0)))/2.0))))))))))))))) +

            0.100000*np.tanh(((((((data["mfcc_15"]) * 2.0)) * (((((-1.0*(((((data["mfcc_2"]) >= ((((data["mfcc_14"]) >= (data["mfcc_10"]))*1.)))*1.))))) < ((((data["mfcc_15"]) >= ((((((((data["mfcc_4"]) / 2.0)) / 2.0)) < (data["mfcc_18"]))*1.)))*1.)))*1.)))) * (((data["mfcc_18"]) + (((((data["mfcc_2"]) * 2.0)) * 2.0)))))) +

            0.100000*np.tanh((((((-1.0*((data["mfcc_19"])))) + (((((((((data["mfcc_5"]) - (data["mfcc_4"]))) - (data["mfcc_0"]))) - (((data["mfcc_0"]) - (data["mfcc_12"]))))) * (((data["mfcc_6"]) * (((((data["mfcc_5"]) - (data["mfcc_12"]))) - (data["mfcc_0"]))))))))) + (((data["mfcc_5"]) - (data["mfcc_12"]))))) +

            0.100000*np.tanh(((3.0) * ((((((((data["mfcc_2"]) / 2.0)) >= ((((((((data["mfcc_0"]) < (0.710992))*1.)) + (((data["mfcc_0"]) - ((((data["mfcc_0"]) >= ((-1.0*((((-1.0) + (((((data["mfcc_0"]) + (((data["mfcc_2"]) / 2.0)))) / 2.0))))))))*1.)))))) * 2.0)))*1.)) * 2.0)))) +

            0.100000*np.tanh(((data["mfcc_15"]) * (((((((((((data["mfcc_11"]) >= (data["mfcc_16"]))*1.)) >= ((((((((-2.0) + (data["mfcc_0"]))) * 2.0)) < ((((((((((data["mfcc_2"]) + (-2.0))) >= (data["mfcc_1"]))*1.)) * 2.0)) - (data["mfcc_0"]))))*1.)))*1.)) * 2.0)) + (((data["mfcc_2"]) + (data["mfcc_11"]))))))) +

            0.100000*np.tanh((((((((((data["mfcc_7"]) >= ((((((((data["mfcc_7"]) * 2.0)) * 2.0)) < (data["mfcc_3"]))*1.)))*1.)) * 2.0)) * 2.0)) * (((((data["mfcc_5"]) - ((((((((((data["mfcc_14"]) < (1.153898))*1.)) * 2.0)) * 2.0)) * 2.0)))) * (((data["mfcc_3"]) - ((-1.0*((((data["mfcc_7"]) * 2.0))))))))))) +

            0.100000*np.tanh((((((((-1.0) >= (data["mfcc_13"]))*1.)) * 2.0)) + ((((((((((data["mfcc_18"]) * (data["mfcc_17"]))) >= (data["mfcc_18"]))*1.)) * (((((data["mfcc_15"]) - (data["mfcc_13"]))) + (((data["mfcc_18"]) * (data["mfcc_7"]))))))) - (((data["mfcc_17"]) - (((data["mfcc_3"]) + (data["mfcc_15"]))))))))) +

            0.100000*np.tanh(((data["mfcc_7"]) - (((data["mfcc_12"]) - (((data["mfcc_1"]) * (((data["mfcc_12"]) - ((((((data["mfcc_10"]) + (data["mfcc_12"]))/2.0)) + ((((((data["mfcc_2"]) + (((data["mfcc_5"]) * (data["mfcc_5"]))))/2.0)) + (((data["mfcc_1"]) * ((((data["mfcc_12"]) >= (data["mfcc_2"]))*1.)))))))))))))))) +

            0.100000*np.tanh(((((data["mfcc_0"]) + (((((((data["mfcc_5"]) < (data["mfcc_0"]))*1.)) < (data["mfcc_5"]))*1.)))) * ((-1.0*(((((data["mfcc_17"]) + ((((11.78159523010253906)) * (((((((data["mfcc_5"]) * 2.0)) * 2.0)) * ((((data["mfcc_0"]) < ((-1.0*((((((-1.0*((data["mfcc_0"])))) >= (data["mfcc_3"]))*1.))))))*1.)))))))/2.0))))))) +

            0.100000*np.tanh(((((((data["mfcc_17"]) + ((((((-0.925553) >= (data["mfcc_1"]))*1.)) * 2.0)))) * (((data["mfcc_2"]) * (((((((((-0.925553) >= (data["mfcc_1"]))*1.)) * 2.0)) + (((data["mfcc_17"]) + (((data["mfcc_3"]) * (((data["mfcc_2"]) + (((data["mfcc_17"]) + (data["mfcc_1"]))))))))))/2.0)))))) / 2.0)) +

            0.100000*np.tanh((((((((((((((data["mfcc_16"]) + (data["mfcc_4"]))/2.0)) * 2.0)) + (data["mfcc_7"]))) * (((data["mfcc_7"]) - ((((((((data["mfcc_0"]) + (1.523039))) >= ((((data["mfcc_7"]) < ((((data["mfcc_0"]) < (data["mfcc_1"]))*1.)))*1.)))*1.)) * 2.0)))))) * 2.0)) * ((((0.265262) < (data["mfcc_17"]))*1.)))) +

            0.100000*np.tanh(((data["mfcc_17"]) * ((((((data["mfcc_1"]) + ((-1.0*((((data["mfcc_0"]) + ((-1.0*(((-1.0*((((data["mfcc_0"]) + ((-1.0*(((((((data["mfcc_18"]) >= (((((data["mfcc_19"]) + (data["mfcc_0"]))) + (data["mfcc_16"]))))*1.)) - (data["mfcc_16"]))))))))))))))))))))/2.0)) * 2.0)))))

from sklearn.metrics import mean_absolute_error

print('GPI: ',mean_absolute_error(mfcc_ttf_df.time_to_failure,GPI(alldata[:mfcc_ttf_df.shape[0]])))

print('GPII: ',mean_absolute_error(mfcc_ttf_df.time_to_failure,GPII(alldata[:mfcc_ttf_df.shape[0]])))

print('GPMean: ',mean_absolute_error(mfcc_ttf_df.time_to_failure,.5*GPI(alldata[:mfcc_ttf_df.shape[0]])+.5*GPII(alldata[:mfcc_ttf_df.shape[0]])))
sub = pd.DataFrame()

sub['seg_id'] = test_df.seg_id.values

sub['time_to_failure'] = GPI(alldata[mfcc_ttf_df.shape[0]:]).values

sub.to_csv('gpmfccsubI.csv',index=False)

sub.head()
sub = pd.DataFrame()

sub['seg_id'] = test_df.seg_id.values

sub['time_to_failure'] = GPII(alldata[mfcc_ttf_df.shape[0]:]).values

sub.to_csv('gpmfccsubII.csv',index=False)

sub.head()
sub = pd.DataFrame()

sub['seg_id'] = test_df.seg_id.values

sub['time_to_failure'] = .5*GPI(alldata[mfcc_ttf_df.shape[0]:]).values+.5*GPII(alldata[mfcc_ttf_df.shape[0]:]).values

sub.to_csv('gpmfccsubmean.csv',index=False)

sub.head()