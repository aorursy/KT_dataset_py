import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt 

import numpy as np 

import pandas as pd 

import seaborn as sns

sns.set_style('white')

sns.set(font_scale=2)
df_train = pd.read_excel("/kaggle/input/Concrete_Data.xls")

print(df_train.shape)

display(df_train.head())
print(df_train.info())
display(df_train.describe())
cols = df_train.columns

color = ['dimgray', 'khaki', 'mediumorchid','cornflowerblue', 'crimson','orangered', 'navy', 'salmon']

sns.set(font_scale=1)



sns.jointplot(data=df_train, x=cols[0], y=cols[-1]

                  ,kind='reg',color=color[0])

plt.show()
sns.jointplot(data=df_train, x=cols[1], y=cols[-1]

                  ,kind='kde',color=color[1])

plt.show()
sns.jointplot(data=df_train, x=cols[2], y=cols[-1]

                  ,kind='kde',color=color[2])

plt.show()
sns.jointplot(data=df_train, x=cols[3], y=cols[-1]

                  ,kind='reg',color=color[3])

plt.show()
sns.jointplot(data=df_train, x=cols[4], y=cols[-1]

                  ,kind='reg',color=color[4])

plt.show()
sns.jointplot(data=df_train, x=cols[5], y=cols[-1]

                  ,kind='kde',color=color[5])

plt.show()
sns.jointplot(data=df_train, x=cols[6], y=cols[-1]

                  ,kind='kde',color=color[6])

plt.show()
sns.jointplot(data=df_train, x=cols[7], y=cols[-1]

                  ,kind='kde',color=color[7])

plt.show()
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X = pd.DataFrame(columns=cols[:-1], data=sc.fit_transform(df_train.drop(cols[-1],axis=1)))

display(X.head(3))

y = df_train[cols[-1]]

display(y.head(3))
sns.set(font_scale=3)

cols = X.columns

n_row = len(cols)

n_col = 2

n_sub = 1

fig = plt.figure(figsize=(20,40))

for i in range(len(cols)):

    plt.subplots_adjust(left=-0.3, right=1.3, bottom=-0.3, top=1.3)

    plt.subplot(n_row, n_col, n_sub)

    sns.distplot(X[cols[i]],norm_hist=False,kde=False, color=color[i],

                 label=['mean '+str('{:.2f}'.format(X.iloc[:,i].mean()))

                        +'\n''std '+str('{:.2f}'.format(X.iloc[:,i].std()))

                        +'\n''min '+str('{:.2f}'.format(X.iloc[:,i].min()))

                        +'\n''max '+str('{:.2f}'.format(X.iloc[:,i].max()))])                                                        

    n_sub+=1

    plt.legend()

plt.show()
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score, mean_absolute_error

from time import time



def test_models(mlds):

    for i in range(len(mlds)):

        r2 = []

        mae = []

        model = mlds[i]

        n= 0 

        for tr, te in KFold(n_splits=5,random_state=42, shuffle=True).split(X, y):

            st_time = time()

            X_tr = X.iloc[tr, :]

            y_tr = y.iloc[tr]

            X_val = X.iloc[te, :]

            y_val = y.iloc[te]

            model.fit(X_tr, y_tr)

            y_preds = model.predict(X_val)

            r2.append(r2_score(y_val, y_preds))

            mae.append(mean_absolute_error(y_val, y_preds))

            en_time = time()

            print('Time:',str(en_time-st_time),'Fold:',str(n),'r2:',str(r2[n]),'mae:',str(mae[n]))

            n+=1

        print('mean_r2', np.mean(r2))

        print('-----------------------------')



    

seed = 42

models = [LinearRegression(),RandomForestRegressor(random_state=seed, n_jobs=-1),

          XGBRegressor(random_state=seed, n_jobs=-1),LGBMRegressor(random_state=seed,n_jobs=-1)]

test_models(models)
import tensorflow as tf

from tensorflow.keras import layers



def create_model(hid_layers,num_cols, drop_rate):

    inp = layers.Input(shape=(num_cols,))

    x = layers.BatchNormalization()(inp)

    for i, units in enumerate(hid_layers):

        x= layers.Dense(units, 'relu')(x)

        x = layers.Dropout(drop_rate)(x)

        x = layers.BatchNormalization()(x)



    output = layers.Dense(1, 'linear')(x)

    

    model = tf.keras.models.Model(inputs=inp,outputs=output)

    model.compile(optimizer='adam', loss='mae')

    return model



hid_layers = [300,200,100]

model = create_model(hid_layers, X.shape[1], 0.2)



from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping



def callbacks():

    rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 0, 

                                min_delta = 1e-4, min_lr = 1e-6, mode = 'min')

        

    ckp = ModelCheckpoint(f'bests_weights.hdf5', monitor = 'val_loss', verbose = 0, 

                              save_best_only = True, save_weights_only = True, mode = 'min')

        

    es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 15, mode = 'min', 

                           baseline = None, restore_best_weights = True, verbose = 0)

    return [rlr, ckp, es]
import tensorflow.keras.backend as K

loss_mae = []

r2_scores = []



fold = 0

for tr, te in KFold(n_splits=5, random_state=42, shuffle=True).split(X,y):

    X_tr = X.iloc[tr, :]

    X_val = X.iloc[te, :]

    y_tr = y.iloc[tr]

    y_val = y.iloc[te]

    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), callbacks=callbacks(),

                        epochs=300, verbose=0)

    model.load_weights('bests_weights.hdf5')

    y_preds = model.predict(X_val)

    loss_mae.append(mean_absolute_error(y_val, y_preds))

    r2_scores.append(r2_score(y_val, y_preds))

    print(f'fold',str(fold)+':','mae:',loss_mae[fold],'r2_score:',r2_scores[fold])

    K.clear_session()

    fold+=1

print("mae: %0.2f (+/- %0.2f)" % (np.mean(loss_mae), np.std(loss_mae) * 2),'mean_r2:', np.mean(r2_scores))


