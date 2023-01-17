import sklearn as sk

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split #Обращение к этой функции через алиас sk - не получается. Почему?

from subprocess import check_output

#_________________________ ЧТЕНИЕ ИЗ ФАЙЛА  ____________________________

data = pd.read_csv('../input/train.csv')

#X_Fin_test = pd.read_csv('../input/test.csv')

#ids = X_Fin_test['id']

#X_Fin_test.drop(['id'], axis=1)



#print(data.head())

print(data.shape)

print(data[:1])

#print(data.iloc[:3, 1])

print(data.describe()) # сводная инфа по таблице

#____________________ ОЦЕНКА ДУБЛЕЙ  _____________________

#http://www.ritchieng.com/pandas-removing-duplicate-rows/

#Double = data.loc[data.duplicated(keep=False,subset=['spacegroup', 'number_of_total_atoms', 'percent_atom_al', 'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang', 'lattice_angle_alpha_degree', 'lattice_angle_beta_degree', 'lattice_angle_gamma_degree']), :]

#print(Double.shape)

#print(Double.sort_values(['spacegroup','number_of_total_atoms','percent_atom_al','percent_atom_ga','percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang', 'lattice_angle_alpha_degree', 'lattice_angle_beta_degree', 'lattice_angle_gamma_degree']))

#Double = data.loc[data.duplicated(keep='last',subset=['spacegroup', 'number_of_total_atoms', 'percent_atom_al', 'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang', 'lattice_angle_alpha_degree', 'lattice_angle_beta_degree', 'lattice_angle_gamma_degree']), :]

#print(Double.shape)

#____________________ УДАЛЕНИЕ ДУБЛЕЙ  _____________________

#data.drop_duplicates(inplace=True,keep='last',subset=['spacegroup', 'number_of_total_atoms', 'percent_atom_al', 'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang', 'lattice_angle_alpha_degree', 'lattice_angle_beta_degree', 'lattice_angle_gamma_degree'])

#print(data.shape)

# predict a value for both formation_energy_ev_natom and bandgap_energy_ev

dataX = data.copy().iloc[:, 1: 12]  # без колонки id и последних двух

#dataX.drop(['spacegroup'], axis=1, inplace=True)

dataY = data.copy().iloc[:, 12: ]   # последний 1 (или 2) столбца

#____________________ ОЦЕНКА ГРАФИКАМИ  _____________________



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style= 'whitegrid', context ='notebook')

#cols = [ 'LSTAT', 'INDUS', 'NOX','RМ','MEDV']

#sns.pairplot(data.iloc[:, :], size=10)

#plt.show()
#sns.distplot(data.iloc[:, 12:13]);plt.show()

#sns.distplot(data.iloc[:, 13:]);plt.show()
# Show the results of a 

#sns.lmplot(x="id", y="formation_energy_ev_natom", data=data)
#sns.lmplot(x="id", y="bandgap_energy_ev", data=data)
#sns.lmplot(x="lattice_vector_1_ang", y="lattice_angle_beta_degree", data=data)
corr = np.corrcoef(data.values.T)

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.set()

hm = sns.heatmap(corr, mask=mask, cbar=True , annot=True , square=True,

fmt='.1f', annot_kws ={'size':7})#,,cmap="YlGnBu"

#yticklabels=cols,

#xticklabels=cols)
#____________________ УДАЛЯЕМ СТОЛБЦЫ ПО РЕЗУЛЬТАТАМ КОРРЕЛ. МАТРИЦЫ  _____________________

#print(type(dataX))

#dataX = dataX.drop(columns='number_of_total_atoms')

#dataX.drop(['number_of_total_atoms'], axis=1, inplace=True)

print(dataX[:1])



#X_Fin_test.drop(['number_of_total_atoms'], axis=1)

#____________________ РАЗБИВКА НА ТРЕНИРОВКУ И ТЕСТ  _____________________

#   

#                   входы    выходы

#

#                     X        Y

# 75%   train      X_train  Y_train   - тренировочный набор для обучения в fit()

# 25%   test       X_test   Y_test    - тестовый набор для оценки результатов 

#

#  после обучения делаем расчет по X_test и имеем выход Y_pred. Y_pred сравниваем с Y_test и имеем оценку точности модели

#

# X - вектор входов

# Y - вектор выходов

# 

#

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size = 0.25, shuffle = False) # а так не хочет: sk.model_selection.train_test_split , random_state=1



print("Входов: "+str(X_train.shape[1]))

print("Выходов: "+str(y_train.shape[1]))

print(y_test.describe()) # сводная инфа по таблице



#print(X_train)

# train_data = np.array(data[1:])

# print(data)



#df = pd.DataFrame(X_train)
#___________________________ ПОИСК ПУСТЫХ  _______________________________

#print(df.isnull().sum()) # проверяем, есть ли пустые значения в столбцах
#__________________________ НОРМИРОВКА 0 - 1 (sigm) _____________________________

# Деревья решения и Случайные леса не нуждаются в нормировке!

from sklearn.preprocessing import MinMaxScaler

sk_tr = MinMaxScaler(feature_range=(0.001, 0.999))

#X_train_norm = sk_tr.fit_transform(df)

#X_test_norm  = sk_tr.transform(X_test)

#print(X_test_norm)

#_____________________ СТАНДАРТИЗАЦИЯ -0.5 - 0 - 0.5 (tanh) ______________________

from sklearn.preprocessing import StandardScaler

#sk_tr = StandardScaler()

sk_tr.fit(X_train); X_train_std = sk_tr.transform(X_train); 

sk_tr.fit(X_test);  X_test_std  = sk_tr.transform(X_test); 

#X_train_std = X_train.copy(); X_test_std = X_test.copy()

#_________________________________________________________________________________

#print("Преобразованный вход:"); print(X_test_std[:3]); print(pd.DataFrame(X_test_std).describe()) # сводная инфа по таблице

y_train_std = y_train.copy(); y_test_std = y_test.copy() #Оставляем как есть, т.к. выход "linear"

#print(y_test_std.describe())

y_train_std = y_train_std.drop(['bandgap_energy_ev'], axis=1);y_test_std = y_test_std.drop(['bandgap_energy_ev'], axis=1)



y_train2_std = y_train.copy(); y_test2_std = y_test.copy() #Оставляем как есть, т.к. выход "linear"

y_train2_std = y_train2_std.drop(['formation_energy_ev_natom'], axis=1);y_test2_std = y_test2_std.drop(['formation_energy_ev_natom'], axis=1)



#sk_tr.fit(y_train); y_train_std = sk_tr.transform(y_train); 

#sk_tr.fit(y_test);  y_test_std  = sk_tr.transform(y_test)



y_train_std['formation_energy_ev_natom'] = np.log1p(y_train_std['formation_energy_ev_natom'].values);

y_train2_std['bandgap_energy_ev'] = np.log1p(y_train2_std['bandgap_energy_ev'].values)



#print(y_train_std.shape[1])



y_test_std['formation_energy_ev_natom'] = np.log1p(y_test_std['formation_energy_ev_natom'].values);  

y_test2_std['bandgap_energy_ev'] = np.log1p(y_test2_std['bandgap_energy_ev'].values)



print("Преобразованный вЫход:"); print(y_test_std[:1]); print(pd.DataFrame(y_test_std).describe()) # сводная инфа по таблице
#________________________ МОДЕЛЬ ___________________________

import theano

from keras.models       import Sequential 

from keras.layers.core  import Dense

from keras.optimizers   import SGD

from keras.layers       import Dense, Dropout, Activation



print("Входов: "+str(X_train.shape[1]))

#print("Выходов: "+str(y_train_std.shape[1]))



model1 = Sequential()

k_init = "random_uniform" #uniform random_normal random_uniform

actForm = "tanh" #"tanh" # "sigmoid" "relu" "softmax" linear

#Softmax используется для последнего слоя глубоких нейронных сетей для задач классификации

НейроновВнутри = 50

model1.add(Dense(units=НейроновВнутри, input_shape=(X_train.shape[1],), kernel_initializer=k_init, activation=actForm)) # входной слой

#model1.add(Dropout(0.1))

model1.add(Dense(units=НейроновВнутри, kernel_initializer=k_init, activation=actForm))               # промежуточный

#model1.add(Dropout(0.1))

#КолВоВыходов = y_train_std.shape[1]

КолВоВыходов = 1

model1.add(Dense(units=КолВоВыходов, kernel_initializer=k_init, activation="relu"))             # выходной слой

#model1.add(Dense(input_dim=50, units=50, kernel_initializer="uniform", activation=actForm))               # промежуточный

#model1.add(Dropout(0.5))

#model1.summary()

#sgd = SGD(lr=0.001, decay=1e-7, momentum=.9) #SGD()

#model1.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop', metrics=['mean_squared_error'])

model1.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop',metrics=['mean_squared_logarithmic_error']) #accuracy mean_squared_logarithmic_error categorical_crossentropy mean_squared_error mean_absolute_error

# СОБСТВЕННО ТРЕНИРОВКА

history1 = model1.fit(X_train_std,y_train_std,epochs=1200,batch_size=50,verbose=0,validation_split=0.1) #,show_accuracy=True

#model1.fit(X_train_std,y_train_std,epochs=20,batch_size=10,verbose=0,validation_split=0.1) #,show_accuracy=True

#________________________ ПРОВЕРКА 1 ___________________________

#y_train_predict = model1.predict(X_train_std)

import sklearn.metrics as metrics



#model1.compile(loss ='mean_squared_logarithmic_error', optimizer='rmsprop',metrics=['mean_squared_logarithmic_error']) #accuracy mean_squared_logarithmic_error categorical_crossentropy mean_squared_error mean_absolute_error

# СОБСТВЕННО ТРЕНИРОВКА

#history1 = model1.fit(X_train_std,y_train_std,epochs=1200,batch_size=50,verbose=0,validation_split=0.1) #,show_accuracy=True

# ПРОВЕРКА

y_test_predict = model1.predict(X_test_std)



def rmsle(h, y): 

    #Compute the Root Mean Squared Log Error for hypthesis h and targets y

    #Args:

    #    h - numpy array containing predictions with shape (n_samples, n_targets)

    #    y - numpy array containing targets with shape (n_samples, n_targets)

    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())



print("RMLSE      : " + str(rmsle(y_test_predict, y_test_std)))

print("MLSE       : " + str(metrics.mean_squared_log_error(y_test_std, y_test_predict)))

print("r2_score 1 : " + str(metrics.r2_score(y_test_std, y_test_predict)))





score = model1.evaluate(X_test_std, y_test_std, batch_size=10) #, batch_size=100



#y_test_predict = y_test_predict.astype(np.float32).reshape((-1,1))

print("1 Выход истинный:");

print(y_test_std[:10])

print("1 Выход расчитанный:");

print(y_test_predict[:10])

#print([y_test_std[:10],y_test_predict[:10],((y_test_predict[:10])*100/y_test_std[:10])])

err_per = ((y_test_predict[:10])*100/y_test_std[:10])

print("Ошибка в %:");

print(err_per)

print("Точность работы на тестовых данных: %.5f%%" % ((1-score[0])*100))

print("Test score:", score)



#print("mean_absolute_error: " + str(metrics.mean_absolute_error(y_test_std, y_test_predict)))

#print("mean_squared_error: " + str(metrics.mean_squared_error(y_test_std, y_test_predict)))

#print("r2_score: " + str(metrics.r2_score(y_test_std, y_test_predict)))

#print("mean_squared_log_error: " + str(metrics.mean_squared_log_error(y_test_std, y_test_predict)))

# list all data in history

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

print(history1.history.keys())

# summarize history for accuracy

plt.plot(history1.history['mean_squared_logarithmic_error'])

plt.plot(history1.history['val_mean_squared_logarithmic_error']) #val_acc

plt.title('model MSLE')

plt.ylabel('MSLE')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()

# summarize history for loss

plt.plot(history1.history['loss'])

plt.plot(history1.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
# ---- Final TEST ---------------------------------

X_Fin_test = pd.read_csv('../input/test.csv')

ids = X_Fin_test['id']

X_Fin_test.drop(['id'], axis=1, inplace=True)

#X_Fin_test.drop(['number_of_total_atoms'], axis=1, inplace=True)

#____________________ УДАЛЕНИЕ ДУБЛЕЙ  _____________________

# Удаление дублей здесь делать нельзя, тк. сравнение будет с их базой

#print(X_Fin_test.shape)

#X_Fin_test.drop_duplicates(inplace=True,keep='last',subset=['spacegroup', 'number_of_total_atoms', 'percent_atom_al', 'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang', 'lattice_angle_alpha_degree', 'lattice_angle_beta_degree', 'lattice_angle_gamma_degree'])

#print(X_Fin_test.shape)



sk_tr.fit(X_Fin_test);  X_Fin_test_std  = sk_tr.transform(X_Fin_test); 

print(X_test[:1])

print(X_Fin_test[:1])



y_Fin_test_predict = model1.predict(X_Fin_test_std)
# ______ ВТОРОЙ ВЫХОД _______

model = Sequential()

k_init = "random_uniform" #uniform random_normal random_uniform

actForm = "tanh" #"tanh" # "sigmoid" "relu" "softmax" linear

#Softmax используется для последнего слоя глубоких нейронных сетей для задач классификации

НейроновВнутри = 150

model.add(Dense(units=НейроновВнутри, input_shape=(X_train.shape[1],), kernel_initializer=k_init, activation=actForm)) # входной слой

model.add(Dense(units=НейроновВнутри, kernel_initializer=k_init, activation=actForm))               # промежуточный

model.add(Dense(units=50, kernel_initializer=k_init, activation=actForm))               # промежуточный

#КолВоВыходов = y_train_std.shape[1]

КолВоВыходов = 1

model.add(Dense(units=КолВоВыходов, kernel_initializer=k_init, activation="relu"))             # выходной слой



model.compile(loss ='mean_squared_logarithmic_error', optimizer='rmsprop',metrics=['mean_squared_logarithmic_error']) #accuracy mean_squared_logarithmic_error categorical_crossentropy mean_squared_error mean_absolute_error

history2 = model.fit(X_train_std,y_train2_std,epochs=800,batch_size=30,verbose=0,validation_split=0.1) #,show_accuracy=True

y_test2_predict = model.predict(X_test_std)

print("RMLSE      : " + str(rmsle(y_test2_predict, y_test2_std)))

print("mean_squared_log_error: " + str(metrics.mean_squared_log_error(y_test2_std, y_test2_predict)))

print("r2_score 2 : " + str(metrics.r2_score(y_test2_std, y_test2_predict)))

score = model.evaluate(X_test_std, y_test2_std, batch_size=10) #, batch_size=100



print("2 Выход истинный:");

print(y_test2_std[:10])

print("2 Выход расчитанный:");

print(y_test2_predict[:10])

err_per = ((y_test2_predict[:10])*100/y_test2_std[:10])

print("Ошибка в %:");

print(err_per)

print("Точность работы на тестовых данных: %.5f%%" % ((1-score[0])*100))

print("Test score 2:", score)



#print("mean_absolute_error: " + str(metrics.mean_absolute_error(y_test2_std, y_test2_predict)))

#print("mean_squared_error: " + str(metrics.mean_squared_error(y_test2_std, y_test2_predict)))



y_Fin_test_predict2 = model.predict(X_Fin_test_std)
# list all data in history

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

print(history2.history.keys())

# summarize history for accuracy

plt.plot(history2.history['mean_squared_logarithmic_error'])

plt.plot(history2.history['val_mean_squared_logarithmic_error']) #val_acc

plt.title('model MSLE')

plt.ylabel('MSLE')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()

# summarize history for loss

plt.plot(history2.history['loss'])

plt.plot(history2.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
# ---- output file compile ---------------------------------

#print(y_Fin_test_predict)

outDF = pd.DataFrame()

outDF['id']=ids

#outDF['formation_energy_ev_natom'] = y_Fin_test_predict

#outDF['bandgap_energy_ev']         = y_Fin_test_predict2

outDF['formation_energy_ev_natom'] = np.exp(y_Fin_test_predict)-1  #((np.exp(pred1)-1)*0.5 + (np.exp(pred1_)-1)*0.5) 

outDF['bandgap_energy_ev']         = np.exp(y_Fin_test_predict2)-1  #((np.exp(pred2)-1)*0.5 + (np.exp(pred2_)-1)*0.5)

print(outDF[:10])

outDF.to_csv('prediction.csv',index=False)