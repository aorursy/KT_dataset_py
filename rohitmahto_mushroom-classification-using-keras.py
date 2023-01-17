# Mushroom Classification using Keras



import pandas as pd

from keras.callbacks import EarlyStopping, TensorBoard

from keras.layers import Dense

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler



dataset = pd.read_csv('../input/mushrooms.csv')

y_train = dataset['class']

x_train = dataset.drop(labels =['class'],axis=1)



ohe = OneHotEncoder()

le = LabelEncoder()



# transform all columns in ohe as we all columns have categorical data  

cols = x_train.columns.values

for col in cols:

    x_train[col] = le.fit_transform(x_train[col])



y_train = le.fit_transform(y_train)



ohe = OneHotEncoder(categorical_features='all')

x_train = ohe.fit_transform(x_train).toarray()

sc = StandardScaler()

x_train = sc.fit_transform(x_train)





x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size = 0.30, random_state = 42)

x_valid, x_test, y_valid, y_test = train_test_split(x_test,y_test, test_size = 0.50, random_state = 42)







model = Sequential()

model.add(Dense(units = 1024, activation ='relu',kernel_initializer='uniform', input_dim = 117))

model.add(Dense(units = 256, activation ='relu',kernel_initializer='uniform'))

model.add(Dense(units = 128, activation ='relu',kernel_initializer='uniform'))

model.add(Dense(units = 1, activation ='relu',kernel_initializer='uniform'))

model.compile(optimizer ='adam', loss ='mse', metrics=['accuracy'])



early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

tensorboard = TensorBoard(log_dir="logs/{}".format('mesh_Dens'),write_graph=True,write_images=True,histogram_freq=0)





model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=1,validation_data=(x_valid,y_valid),shuffle=True,callbacks=[tensorboard,early_stop])



_,acc = model.evaluate(x_test, y_test, verbose = 2, batch_size = 32)

print("acc: %.2f" % (acc))
