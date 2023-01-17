import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, LSTM

import seaborn as sns
# data load and checks

df = pd.read_csv("../input/shadow_robot_dataset.csv")

# columns include spaces. Following code removes them:

df.columns = df.columns.str.replace('\s+', '')

df['experiment_number'] = df['experiment_number'].str.replace('\s+', '')
# exclude positioning

no_pos = [x for x in df.columns if x[-3:] != "pos"]
# numbers of measurement per experiment vs good/bad grasp

df_lstm = df.copy()

df_lstm['robustness_init'] = df['robustness']

df_lstm.loc[df['robustness']<100, 'robustness'] = 0

df_lstm.loc[df['robustness']>=100, 'robustness'] = 1





# ensure each experiment has only one outcome

experiments_robustness = df_lstm[['experiment_number',

                                  'robustness']].groupby('experiment_number', 

                                                         as_index = False, sort = False).mean()



print ("Possible mean of outcome for each experiment")

print (experiments_robustness['robustness'].value_counts())

print ("\n")



# explore relation between number of measurements and robustness

experiments = df_lstm[['experiment_number',

                       'robustness', 'robustness_init',

                       'measurement_number']].groupby('experiment_number', 

                                                      as_index = False, sort = False).max()



print ("Distribution of successful and unsuccessful grasps")

print (experiments['robustness'].value_counts())



# exclude one experiment with robustness around 3000

experiments_ex_outlier = experiments.loc[experiments['robustness_init']<500,]



# plot relation between number of measurements and final outcome

g = sns.lmplot(x="measurement_number", y="robustness_init", hue="robustness",

               truncate=True, size=7, data=experiments_ex_outlier, fit_reg=False)
# define Y

Y_lstm = experiments['robustness'].values



# transform dataframe into 3D array for LSTM input layer

exp_array = df_lstm['experiment_number'].unique()

exp_29_each = np.repeat(exp_array, 30).tolist()



msrm_29 = list(range(0,30))

msrm_29 = msrm_29*exp_array.shape[0]



# transform initial dataframe for following merge

x_columns = [x for x in no_pos if x not in ['experiment_number','robustness','measurement_number']]

x_columns_ext = ['experiment_number','measurement_number'] + x_columns

df_lstm = df_lstm.loc[:,x_columns_ext]



# prepare dataframe that has 29 measurements per each experiment

df_lstm2 = pd.DataFrame({"experiment_number": exp_29_each, "measurement_number": msrm_29})



df_lstm2= df_lstm2.merge(df_lstm, how = 'left', 

                         left_on = ['experiment_number', 'measurement_number'],

                         right_on = ['experiment_number', 'measurement_number'])

df_lstm2.fillna(0.0, inplace = True)



# reshape

X_lstm = df_lstm2[x_columns].values.reshape(exp_array.shape[0], 30, len(x_columns))

print ("X_lstm shape:")

print (X_lstm.shape)
# split into train and validation datasets

seed = 32

np.random.seed(seed)

X_tr, X_val, Y_tr, Y_val = train_test_split(X_lstm, Y_lstm, test_size=0.20, random_state=seed)
# Keras model

model = Sequential()

model.add(LSTM(128,

               input_shape = (30, len(x_columns)),

               return_sequences=False,

               dropout=0.2, 

               recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))

          

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_tr, Y_tr, batch_size = 10, epochs = 5, 

          validation_data=(X_val, Y_val), verbose = 2)
score, acc = model.evaluate(X_val, Y_val)

print ("")

print ('Test accuracy:', acc)