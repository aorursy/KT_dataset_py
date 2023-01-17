import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout
#Loading the data

DATAPATH = "/kaggle/input/wind-turbine-scada-dataset/T1.csv"

data = pd.read_csv(DATAPATH, index_col = ['Date/Time'])



#Parsing data dates

data['Date'] = pd.to_datetime(data.index, format="%d %m %Y %H:%M")

data.set_index('Date',inplace=True)



#Renaming columns

data = data.rename(columns={'LV ActivePower (kW)': 'Active_power','Wind Speed (m/s)':'Wind_speed','Theoretical_Power_Curve (KWh)': 'Theoretical_power','Wind Direction (°)':'Wind_direction'})

data = data.dropna()
data.info()
data['Gust'] = np.array([0] + list(data['Wind_speed'][1:].values - data['Wind_speed'][:-1].values))
data
datax = np.asarray(data.copy().drop(['Active_power','Theoretical_power'],axis=1).values)

datay = data.copy()['Active_power'].values

th_power = data['Theoretical_power'].values

#datax = datax.reshape(datax.shape[0],datax.shape[1],1)

#datay = datay.reshape(datay.shape[0],1)

#th_power = th_power.reshape(th_power.shape[0],1)
data_samples = datax.shape[0]

x_train = datax[:int(0.7 * data_samples)]

x_val = datax[int(0.7 * data_samples) : int(0.85 * data_samples)]

x_test = datax[int(0.85 * data_samples) :]



y_train = datay[:int(0.7 * data_samples)]

y_val = datay[int(0.7 * data_samples) : int(0.85 * data_samples)]

y_test = datay[int(0.85 * data_samples) :]
print("Training examples: " + str(x_train.shape[0]))

print("Validation examples: " + str(x_val.shape[0]))

print("Test examples: " + str(x_test.shape[0]))
fig = plt.figure(figsize = (15,15))



sns.heatmap(data.corr(), vmax = .8, square = True)

plt.show()
model = Sequential([

    Dense(10,kernel_initializer = 'normal',input_shape = (3,),activation = 'relu'),

    Dense(25,kernel_initializer = 'normal',activation = 'relu'),

    Dense(25,kernel_initializer = 'normal',activation = 'relu'),

    Dense(10,kernel_initializer = 'normal',activation = 'relu'),

    Dense(1,kernel_initializer = 'normal',activation = 'linear')

])



model.compile(

    loss = 'mean_absolute_error',optimizer='adam',metrics=['mean_absolute_error'])
model.summary()
from keras.callbacks import ModelCheckpoint
def display_training_curves(training,validation,title,subplot):

    if subplot%10 == 1:

        plt.subplots(figsize = (10,10),facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
display_training_curves(fit.history['loss'], fit.history['val_loss'], 'loss', 211) 
from keras.models import load_model

model = load_model("/kaggle/input/wind-neural-net/my_model.hdf5")
model.summary()
import types

import tempfile

import keras.models



def make_keras_picklable():

    def __getstate__(self):

        model_str = ""

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:

            keras.models.save_model(self, fd.name, overwrite=True)

            model_str = fd.read()

        d = { 'model_str': model_str }

        return d



    def __setstate__(self, state):

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:

            fd.write(state['model_str'])

            fd.flush()

            model = keras.models.load_model(fd.name)

        self.__dict__ = model.__dict__





    cls = keras.models.Model

    cls.__getstate__ = __getstate__

    cls.__setstate__ = __setstate__

    

make_keras_picklable()
# Import pickle Package



import pickle

# Save the Modle to file in the current working directory



Pkl_Filename = "pickle_model.pkl"  



with open(Pkl_Filename, 'wb') as file:  

    pickle.dump(model, file)
y_pred = model.predict(x_test)
y_pred[y_pred<0] = 0
nik =   [ 8.51521484e+02,  1.01623571e+03,  1.19089645e+03,  1.10533343e+03,

        1.42126503e+03,  1.39092575e+03,  1.41171925e+03,  1.82458728e+03,

        1.85975805e+03,  1.67113438e+03,  1.81877738e+03,  1.93867418e+03,

        2.34995526e+03,  2.44985588e+03,  2.01707775e+03,  3.24057570e+03,

        2.86248240e+03,  3.17835318e+03,  3.22255214e+03,  3.09699840e+03,

        3.49890318e+03,  3.12445214e+03,  3.18895054e+03,  3.08471571e+03,

        3.26691114e+03,  1.82171707e+03,  1.91544339e+03,  1.44944391e+03,

        1.54458367e+03,  1.82310455e+03,  1.46501644e+03,  1.69375105e+03,

        1.52780627e+03,  1.32659679e+03,  1.38886342e+03,  1.40053397e+03,

        1.87686883e+03,  1.53326245e+03,  1.42362900e+03,  1.02583679e+03,

        4.58639510e+02,  1.95637024e+02,  2.44516846e+02,  8.30089462e+02,

        9.79176848e+02,  9.41398620e+02,  8.09213764e+02,  1.99804222e+03,

        1.98620114e+03,  2.06384231e+03,  1.59407199e+03,  1.26586770e+03,

        8.76922306e+02,  1.24389006e+03,  7.50481531e+02,  6.98305033e+02,

        5.24714408e+02,  8.30321045e+01,  8.94052562e+01,  1.67664450e+02,

        4.33144801e+02, -2.47997683e+02,  2.16719896e+02,  2.88409002e+02,

       -1.07121132e+02, -2.49083081e+02, -7.96218860e+01, -2.10658983e+02,

        1.02015809e+02,  3.63323177e+02,  4.17509602e+02,  1.50050933e+02,

        2.11442759e+02, -1.63285010e+02, -4.06918452e+02, -3.79267873e+02,

       -1.03101782e+01, -5.89267347e+01, -3.68568317e+02, -3.05789541e+02,

       -3.86028263e+02,  3.82210325e+02,  2.51759418e+01, -1.06567418e+02,

       -3.80354949e+02, -2.92466634e+02, -2.71878321e+02,  6.86941191e+01,

       -1.03984169e+02,  3.56347937e+02,  2.57185535e+02,  5.38580362e+02,

        8.74398340e+02,  1.62759751e+03,  1.13121751e+03,  1.18955572e+03,

        1.21611999e+03,  1.37472170e+03,  1.37715886e+03,  1.42569937e+03,

        1.00776499e+03,  1.30461818e+03,  1.70371210e+03,  2.30239452e+03,

        2.32295385e+03,  2.74813330e+03,  2.78814670e+03,  2.38518276e+03,

        3.35873994e+03,  3.13122743e+03,  2.60118205e+03,  3.10602928e+03,

        3.16470123e+03,  2.76378542e+03,  2.76054005e+03,  2.74330241e+03,

        2.08901218e+03,  2.30237835e+03,  2.00655424e+03,  2.28623093e+03,

        2.32853974e+03,  1.86848595e+03,  1.66337034e+03,  1.54578593e+03,

        1.58895388e+03,  1.51492254e+03,  1.55838695e+03,  1.68895846e+03,

        1.54354346e+03,  1.22693357e+03,  1.10181194e+03,  1.00793420e+03,

        1.13239431e+03,  9.22975240e+02,  9.60059393e+02,  8.57524313e+02,

        9.26119227e+02,  8.08307941e+02,  1.06247241e+03,  9.32240919e+02,

        9.38194435e+02,  8.01207399e+02,  4.42662162e+02,  7.41969921e+01,

        1.27983205e+02,  4.83270241e+02,  5.04443264e+02,  6.67900312e+02,

        4.71723794e+02,  9.38194184e+02,  9.55824388e+02,  6.98600269e+02,

        2.06911348e+02,  5.78985687e+02, -1.72951145e+01,  4.33062038e+01,

        3.63866031e+02,  1.24061177e+02,  1.70151415e+02,  2.06254429e+02,

        3.00178896e+02,  4.03852646e+02,  3.30182953e+02,  1.26384730e+03,

       -1.99561652e+02, -1.48569702e+02,  3.59390232e+02,  6.11908042e+02,

        8.36419805e+02,  6.82489734e+02,  9.87034454e+02,  1.54904624e+03,

        1.55684058e+03,  7.55191836e+02,  1.36046778e+03,  1.64210252e+03,

        1.12227941e+03,  1.47522436e+03,  1.14164933e+03,  1.92266273e+03,

        1.30719009e+03,  1.44668277e+03,  2.15878830e+03,  1.97587614e+03,

        2.42199325e+03,  1.98940977e+03,  2.40846796e+03,  2.39761384e+03,

        2.51182015e+03,  2.32958413e+03,  2.66558333e+03,  2.29025056e+03,

        1.86765778e+03,  2.44742757e+03,  1.53849161e+03,  2.29843416e+03,

        2.22039106e+03,  2.76569998e+03,  2.68102327e+03,  2.64286281e+03,

        2.50058146e+03,  2.44778308e+03,  2.27505711e+03,  2.04347937e+03,

        2.16780014e+03,  2.54164189e+03,  2.65522627e+03,  2.61686600e+03,

        2.72346776e+03,  2.58030136e+03,  2.76729915e+03,  2.73418637e+03,

        2.72546543e+03,  2.58771867e+03,  2.35382768e+03,  2.03750691e+03,

        1.83985326e+03,  1.64838525e+03,  1.68508791e+03,  1.19513160e+03,

        1.69611728e+03,  2.02019172e+03,  1.25849269e+03,  1.62446472e+03,

        1.40366416e+03,  1.59177103e+03,  1.63028711e+03,  1.33288228e+03,

        1.10425744e+03,  9.83493526e+02,  9.46631049e+02,  1.16862614e+03,

        1.17677933e+03,  1.62978971e+03,  1.57202558e+03,  2.27421715e+03,

        2.25042913e+03,  3.41993428e+03,  3.65552558e+03,  3.38302847e+03,

        3.62129487e+03,  3.41909439e+03,  3.58667261e+03,  3.40415902e+03,

        4.20051722e+03,  4.20873206e+03,  4.37067194e+03,  4.55051574e+03,

        4.80788750e+03,  4.88616800e+03,  4.85157008e+03,  4.37706484e+03,

        3.99217682e+03,  3.96496857e+03,  3.26873625e+03,  2.18683745e+03,

        1.50026085e+03,  2.30167846e+03,  1.55981550e+03,  1.14248964e+03,

        2.06204507e+03,  1.68078370e+03,  1.86655747e+03,  1.77710059e+03,

        1.81385164e+03,  1.88589382e+03,  1.86817029e+03,  1.85686882e+03,

        1.82958581e+03,  1.77357430e+03,  1.76344603e+03,  2.04094265e+03,

        2.13016265e+03,  1.43835981e+03,  1.85547694e+03,  1.73458580e+03,

        1.75489293e+03,  1.69784543e+03,  2.23088999e+03,  1.82793575e+03,

        1.74234232e+03,  1.77183870e+03,  1.91201296e+03,  8.30838475e+02,

        1.19634886e+03,  9.33444208e+02,  9.35616904e+02,  1.43166615e+03,

        1.04570419e+03,  9.53554065e+02,  5.69438657e+02,  3.22069609e+02,

        9.70517531e+02,  4.93357401e+02,  4.66630225e+02,  5.69213298e+02,

        1.62113121e+02,  4.76129982e+02,  1.10759022e+02,  1.89148625e+01,

        4.58126108e+02,  1.57970933e+02,  7.54565814e+01,  2.94177894e+02,

        5.03899099e+02,  3.24763090e+02,  2.63366369e+01,  4.82054364e+02,

       -2.31515372e+02,  3.05829551e+02,  2.38352933e+02,  4.45735949e+02,

        2.94194470e+02,  3.33328909e+02,  1.91263596e+02,  9.99865895e+02,

        1.23415155e+03,  4.38083456e+02,  2.71174173e+02,  7.21448310e+02,

        5.01557076e+02,  3.41581053e+02,  1.07629964e+02,  1.55365981e+02,

        2.26440258e+01,  7.07974418e+01,  3.65146674e+02,  2.50183599e+02,

        1.69713267e+02,  1.91605759e+02,  2.16755738e+02,  2.14906525e+02,

        2.66815280e+02,  2.17965333e+02,  3.75335675e+02, -1.60995244e+02,

       -2.14698767e+02, -1.04760109e+02, -1.68449905e+02, -2.19145201e+02,

        1.12729000e+02,  3.68275957e+02,  5.83343644e+02,  9.31368552e+02,

        1.00182005e+03,  1.21546124e+03,  1.23092434e+03,  5.59600503e+02,

        6.55130193e+02,  7.73164068e+02,  8.87526181e+02,  9.75655261e+02,

        1.07593240e+03,  8.21135809e+02,  5.17212873e+02,  8.59357235e+02,

        7.21844702e+02,  1.18969258e+03,  1.38266660e+03,  1.26101321e+03,

        1.28439888e+03,  1.38932503e+03,  1.05467945e+03,  1.07286627e+03,

        6.80593604e+02,  4.89291783e+02,  2.48779036e+02,  3.56540367e+02,

        2.91870617e+02,  6.03955663e+02,  1.48236745e+03,  1.72274836e+03,

        1.16489353e+03,  1.06792792e+03,  1.23943228e+03,  1.03961320e+03,

        8.21326269e+02,  6.42312704e+02,  3.30650561e+02,  1.60095360e+02,

        1.32697952e+02, -2.81844145e+02,  1.95158667e+00,  1.19512335e+00,

        1.58879468e+02, -1.88845264e+02, -2.82409889e+02, -4.35130282e+02,

        1.61193943e+02, -1.07704884e+02,  3.70392471e+00,  8.34111657e+01,

        6.32761513e+01,  1.36570105e+02, -9.24061441e+01, -1.26863290e+02,

       -1.90574335e+02, -3.84971008e+02, -4.45835741e+02,  1.33243730e+02,

        1.45803784e+02, -2.97753211e+01, -3.18130568e+02, -3.08814111e+01,

       -4.18074762e+02, -4.86910824e+02, -3.24349350e+02, -3.44428847e+02,

        9.30608120e+01, -1.81537074e+02,  1.65755110e+02,  1.50283674e+02,

        4.97699635e+02,  8.81159344e+02,  1.32012942e+03,  6.57431647e+02,

        5.66315669e+02,  7.70975979e+02,  1.65560430e+03,  1.94039070e+03,

        2.09389839e+03,  2.62702796e+03]
plt.figure(figsize=(12,7))

plt.plot(y_pred[-1000:],'r',label='Neural Net')

plt.plot(y_test[-1000:],'b',label='Real Values')

#plt.plot(nik,'g',label='Time series')

plt.legend()

plt.show()
y_pred
y_test