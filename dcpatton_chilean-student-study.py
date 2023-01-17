import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 5000)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



seed = 51



import tensorflow as tf

import random



print('TensorFlow version = ' + tf.__version__)

tf.random.set_seed(seed)

random.seed(seed)
performance_summary = pd.read_csv('/kaggle/input/rendimiento-escolar-chile/20180214_Resumen_Rendimiento 2017_20180131.csv'

                                  , delimiter=';')

performance_summary['PROM_ASIS'] = [x.replace(',', '.') for x in performance_summary['PROM_ASIS']]



indices = []

for index, row in performance_summary.iterrows():

    try:

        (float(row.PROM_ASIS))

    except:

        indices.append(index)

        

performance_summary.drop(indices, axis='index', inplace=True)



performance_summary['PROM_ASIS'] = performance_summary['PROM_ASIS'].astype(float)

performance_summary.head()
cohort = performance_summary.copy()



# SI_MUJ_TO = Total number of wommen in the type of education, without information on their final situation

indices = cohort.index[cohort['SI_MUJ_TO'] >0].tolist()

cohort.drop(indices, axis='index', inplace=True)

# SI_HOM_TO = Total number of men in the type of education, without information on their final situation

indices = cohort.index[cohort['SI_HOM_TO'] >0].tolist()

cohort.drop(indices, axis='index', inplace=True)

# APR_SI_TO = Total number of approved students for whom there is no information of their sex, in the type of education

indices = cohort.index[cohort['APR_SI_TO'] >0].tolist()

cohort.drop(indices, axis='index', inplace=True)

# RET_SI_01 = Number of students who do not have information on their gender who dropped out of 1st grade

indices = cohort.index[cohort['RET_SI_01'] >0].tolist()

cohort.drop(indices, axis='index', inplace=True)



cohort.drop(['APR_SI_03', 'APR_SI_04', 'APR_SI_07', 'APR_SI_TO', 'RET_SI_01', 'SI_HOM_01', 'SI_HOM_02', 'SI_HOM_03',

             'SI_HOM_04', 'SI_HOM_05', 'SI_HOM_07', 'SI_HOM_TO', 'SI_MUJ_01', 'SI_MUJ_02', 'SI_MUJ_03', 'SI_MUJ_04',

             'SI_MUJ_05', 'SI_MUJ_07', 'SI_MUJ_TO', 'PROM_ASIS_APR_SI'], axis='columns', inplace=True)
cohort.drop(['PROM_ASIS_APR_HOM', 'PROM_ASIS_APR_MUJ', 'PROM_ASIS_APR', 'PROM_ASIS_REP_HOM', 'PROM_ASIS_REP_MUJ',

             'PROM_ASIS_REP'], axis='columns', inplace=True)
cohort.isna().sum()
cohort.sample(5)
# APR_HOM_TO = Total number of men who passed

# REP_HOM_TO = Total number of men who failed

# RET_HOM_TO = Total number of men who returned (drop-out)

# TRA_HOM_TO = Total number of men who changed location

cohort['total_males'] = cohort['APR_HOM_TO'] + cohort['REP_HOM_TO'] + cohort['RET_HOM_TO'] + cohort['TRA_HOM_TO']



# drop schools with no males

indices = cohort.index[cohort['total_males'] == 0].tolist()

cohort.drop(indices, axis='index', inplace=True)



# APR_MUJ_TO = Total number of women who passed

# REP_MUJ_TO = Total number of women who failed

# RET_MUJ_TO = Total number of women who returned (drop-out)

# TRA_MUJ_TO = Total number of women who changed location

cohort['total_females'] = cohort['APR_MUJ_TO'] + cohort['REP_MUJ_TO'] + cohort['RET_MUJ_TO']+ cohort['TRA_MUJ_TO']



cohort['target'] = cohort['APR_HOM_TO'] # how many male students pass

cohort['target'] = (cohort['target']/(cohort['total_males']))  # ratio of males that pass
cohort.drop([

'APR_HOM_01',

'APR_HOM_02',        

'APR_HOM_03',        

'APR_HOM_04',        

'APR_HOM_05',        

'APR_HOM_06',       

'APR_HOM_07',        

'APR_HOM_08',        

'APR_HOM_TO',        

'APR_MUJ_01',        

'APR_MUJ_02',        

'APR_MUJ_03',        

'APR_MUJ_04',        

'APR_MUJ_05',        

'APR_MUJ_06',        

'APR_MUJ_07',        

'APR_MUJ_08',        

'APR_MUJ_TO',        

'REP_HOM_01',        

'REP_HOM_02',        

'REP_HOM_03',        

'REP_HOM_04',        

'REP_HOM_05',        

'REP_HOM_06',        

'REP_HOM_07',        

'REP_HOM_08',        

'REP_HOM_TO',        

'REP_MUJ_01',        

'REP_MUJ_02',        

'REP_MUJ_03',        

'REP_MUJ_04',        

'REP_MUJ_05',        

'REP_MUJ_06',        

'REP_MUJ_07',        

'REP_MUJ_08',        

'REP_MUJ_TO',        

'RET_HOM_01',        

'RET_HOM_02',        

'RET_HOM_03',        

'RET_HOM_04',        

'RET_HOM_05',        

'RET_HOM_06',        

'RET_HOM_07',        

'RET_HOM_08',        

'RET_HOM_TO',        

'RET_MUJ_01',        

'RET_MUJ_02',        

'RET_MUJ_03',        

'RET_MUJ_04',        

'RET_MUJ_05',        

'RET_MUJ_06',        

'RET_MUJ_07',        

'RET_MUJ_08',        

'RET_MUJ_TO',        

'TRA_HOM_01',        

'TRA_HOM_02',        

'TRA_HOM_03',        

'TRA_HOM_04',        

'TRA_HOM_05',        

'TRA_HOM_06',        

'TRA_HOM_07',        

'TRA_HOM_08',        

'TRA_HOM_TO',        

'TRA_MUJ_01',        

'TRA_MUJ_02',        

'TRA_MUJ_03',        

'TRA_MUJ_04',        

'TRA_MUJ_05',        

'TRA_MUJ_06',        

'TRA_MUJ_07',        

'TRA_MUJ_08',        

'TRA_MUJ_TO'        

    ], axis='columns', inplace=True)
cohort.sample(5)
cohort.drop(['AGNO', 'RBD', 'DGV_RBD', 'NOM_RBD'], axis='columns', inplace=True)
cohort.info()
# Encode a numeric column as zscores

def encode_numeric_zscore(df, name, mean=None, sd=None):

    if mean is None:

        mean = df[name].mean()



    if sd is None:

        sd = df[name].std()



    df[name] = (df[name] - mean) / sd

    

# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)

def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}-{x}"

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)
cohort.drop(['NOM_COM_RBD', 'NOM_DEPROV_RBD'], axis='columns', inplace=True)
encode_text_dummy(cohort, 'COD_REG_RBD')

encode_text_dummy(cohort, 'COD_PRO_RBD')

encode_text_dummy(cohort, 'COD_COM_RBD')  

# encode_text_dummy(cohort, 'NOM_COM_RBD')  # redundant

encode_text_dummy(cohort, 'COD_DEPROV_RBD') 

# encode_text_dummy(cohort, 'NOM_DEPROV_RBD') # redundant

encode_text_dummy(cohort, 'COD_DEPE')

encode_text_dummy(cohort, 'COD_DEPE2')

encode_text_dummy(cohort, 'RURAL_RBD')

encode_text_dummy(cohort, 'ESTADO_ESTAB')

encode_text_dummy(cohort, 'COD_ENSE')

encode_text_dummy(cohort, 'COD_ENSE2')
cohort.shape
y = cohort['target']

x = cohort.drop(['target'], axis='columns')
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed)



print('Training size: ' + str(len(x_train)))

print('Validation size: ' + str(len(x_test)))
import tensorflow

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, ELU, Input, Dropout



input = Input(shape=x.shape[1])



m = Dense(1024)(input)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(1024)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(1024)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(1024)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(1024)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(1024)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(1024)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(1024)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(1024)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(1024)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(1024)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



output = Dense(1, activation='linear')(m)



model = Model(inputs=[input], outputs=[output])



model.summary()
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



model.compile(optimizer=Adam(lr=0.001, amsgrad=True), loss='mean_squared_error', metrics=['mae'])



es = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
%%time

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[es], verbose=2

                    , epochs=300, batch_size=256)
history = history.history



import matplotlib.pyplot as plt

import seaborn as sns



fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))



ax1.plot(history['loss'], label='Training')

ax1.plot(history['val_loss'], label='Validation')

ax1.legend(loc='best')

ax1.set_title('Loss')



ax2.plot(history['mae'], label='Training')

ax2.plot(history['val_mae'], label='Validation')

ax2.legend(loc='best')

ax2.set_title('Mean Absolute Error')



plt.xlabel('Epochs')

sns.despine()

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))



ax1.plot(history['loss'][10:], label='Training')

ax1.plot(history['val_loss'][10:], label='Validation')

ax1.legend(loc='best')

ax1.set_title('Loss')



ax2.plot(history['mae'][10:], label='Training')

ax2.plot(history['val_mae'][10:], label='Validation')

ax2.legend(loc='best')

ax2.set_title('Mean Absolute Error')



epochs = range(10, len(history['loss']))

plt.xlabel('Epochs')

sns.despine()

plt.show()
model.evaluate(x_test, y_test)