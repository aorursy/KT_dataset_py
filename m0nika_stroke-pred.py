import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.utils import np_utils

import os

print(os.listdir("../input"))
data=pd.read_csv("../input/cardiovascular-disease-dataset/cardio_train.csv")

df=pd.read_csv("../input/cardiovascular-disease-dataset/cardio_train.csv", header=0, sep=";")

dfcol=df.columns

df.head()
df[["cardio","height"]].groupby("cardio").count()

sns.countplot(x="cardio", data=df, palette="Set1")


from sklearn import preprocessing

scaler=preprocessing.MinMaxScaler()

dfscale=scaler.fit_transform(df)

dfscale2=pd.DataFrame(dfscale, columns=dfcol)

dfscale2.head()
xdf=dfscale2.iloc[:,0:11]

#xdf["gender"]=np.where(xdf["gender"]==1,"0","1") #Cambiar el 2 por 1, el 1 por 0 (por orden)

#Aca vendria un posible drop de variables xdf=xdf.drop(["gender","gluc"], axis=1)

ydf=dfscale2.iloc[:,-1]

x_training, x_testing, y_training, y_testing = train_test_split(xdf, ydf, test_size = 0.2, random_state=123, stratify=ydf)
print(xdf.shape)
from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.optimizers import SGD

from keras.layers import Dropout

from keras.constraints import maxnorm



model = Sequential()

model.add(Dense(25, input_dim=11, activation='softsign', kernel_constraint=maxnorm(2)))

#model.add(Dropout(0))

model.add(Dense(5, activation='softsign'))

#model.add(Dropout(0))

model.add(Dense(3, activation='softsign'))

#model.add(Dropout(0))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss = 'binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])



model.summary()
model.fit(x_training, y_training, epochs=50, batch_size=50, verbose=0)

score = model.evaluate(x_training, y_training)

print("\n Training Accuracy:", score[1])

score = model.evaluate(x_testing, y_testing)

print("\n Testing Accuracy:", score[1])
res=model.predict(x_testing)

res

resdf=pd.DataFrame(res, index=x_testing.index)

resdf.columns=["Pr"]

resdf["ID"]=range(14000)

resdf["y"]=np.where(resdf["Pr"]>=0.5,"1", "0")

resdf

prediction=resdf.drop(["Pr","ID"], axis=1)

predictionarray=prediction.astype(np.float)

sns.distplot(resdf["Pr"],  color="red")
c1=resdf[['ID','y']].groupby('y').count()

c1
y_testingdf=pd.DataFrame(y_testing, index=y_testing.index)

y_testingdf["ID"]=range(14000)

y_test=y_testingdf.drop(["ID"], axis=1)

c2=y_testingdf[['ID','cardio']].groupby('cardio').count()

c2
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test.values, predictionarray)

cm
Accuracy=cm[0,0]/(cm[0,0]+cm[1,0])

print("The accuracy of the model is: "+ str(Accuracy*100) + " %")
#INSERT DATA#

###############################################################################



day= 25 # day of bith 

month= 9 # month of bith (in numbers)

year= 1998 # year of bith

gender= 1 # 0 for women, 1 for men

height= 183 # in cm

weight= 89 # in kilograms

systolicbloodpressure= 120 # Systolic blood pressure

diastolicbloodpressure= 80 # Diastolic blood pressure

cholesterol= 1 # 1: normal, 2: above normal, 3: well above normal

gluc= 1 # 1: normal, 2: above normal, 3: well above normal

smoke= 0 # 1 if you smoke, 0 if not

alco= 0 # 1 if you drink alcohol, 0 if not

active= 1 # 1 if you do physical activity, 0 if not



##############################################################################

from datetime import date

f_date = date(year,month,day)

l_date = date.today()

delta = l_date - f_date

agedays=delta.days



agedayscale=(agedays-df["age"].min())/(df["age"].max()-df["age"].min())

heightscale=(height-df["height"].min())/(df["height"].max()-df["height"].min())

weightscale=(weight-df["weight"].min())/(df["weight"].max()-df["weight"].min())

sbpscale=(systolicbloodpressure-df["ap_hi"].min())/(df["ap_hi"].max()-df["ap_hi"].min())

dbpscale=(diastolicbloodpressure-df["ap_lo"].min())/(df["ap_lo"].max()-df["ap_lo"].min())

cholesterolscale=(cholesterol-df["cholesterol"].min())/(df["cholesterol"].max()-df["cholesterol"].min())

glucscale=(gluc-df["gluc"].min())/(df["gluc"].max()-df["gluc"].min())



single=np.array([agedayscale, gender, heightscale, weightscale, sbpscale, dbpscale, cholesterolscale, glucscale, smoke, alco, active ])

singledf=pd.DataFrame(single)

final=singledf.transpose()

final

finalres=model.predict(final)

finalres

print("The probability of having or to have a Cardiovascular Disease is: "+ str(round(finalres[0,0]*100,2)) + "%")