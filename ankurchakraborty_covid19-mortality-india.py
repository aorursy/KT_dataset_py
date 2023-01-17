#push all the imports here

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import tensorflow as tf



import matplotlib.pyplot as plt



#analyze the data

df = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")



#plenty of null values are there

df.isnull()



#dropping some unimportant columns

df = df.drop(columns=["government_id", "detected_city", "detected_state", "nationality", "status_change_date", "notes", "diagnosed_date", "detected_district"], axis=1)



df = df.drop(df.index[924:])



#So drop pretty much all the NaN rows

df = df.dropna()





#>>>>>>>>>>>>>>>>>this set is unnecessary because the data is having huge number of NaN

# #convert the string dates into datetime values

# df["diagnosed_date"] = pd.to_datetime(df["diagnosed_date"],format='%d/%m/%Y')

# df["status_change_date"] = pd.to_datetime(df["status_change_date"],format='%d/%m/%Y')





# survived_for = df["status_change_date"] - df["diagnosed_date"]



# #add the survived for time

# df["survived_for"] = df.apply(lambda row: row["status_change_date"] - row["diagnosed_date"], axis=1)



#>>>>>>>>>>>>>>>>>>>>>>>unused data processing



#Recovered or Hospitalized is 0, deceased is 1

df['deceased'] = df.apply(lambda row: 0 if row.current_status == 'Recovered' or 'Hospitalized' else 1, axis = 1)





#Male is 0, Female 1

df['gender_binary'] = df.apply(lambda row: 0 if row.gender == 'M' else 1, axis = 1)



df['age'] = df.apply(lambda row: pd.to_numeric(row.age), axis=1)



df.count()



df.head()
#Split data

model_data = df[["age", "gender_binary"]]

model_target = df[["deceased"]]



x_train,x_test, y_train, y_test = train_test_split(model_data, model_target, test_size=0.33)



scaler = StandardScaler()



x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)



N, D = x_train.shape



plt.plot(y_train) #this is why this model will fail
model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=(D,)))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model.compile(optimizer = 'adam',

             loss = 'binary_crossentropy',

             metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_train, y_train), epochs = 500)

plt.plot(history.history['loss'], label='loss')

plt.plot(history.history['val_loss'], label='val_loss')

plt.legend()
new_data = [[10,1],

            [50,1],

            [80,1]]



model.predict(new_data)
# save the model

model.save('covid19_mortality_india_ANN_failed_model.h5')