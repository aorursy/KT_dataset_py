# -*- coding: utf-8 -*-

"""

Created on Tue Apr 18 03:03:27 2017



@author: chuang yun yen

National Taiwan University

precision 99%

5 fold validation 

"""



import csv

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Activation,Dropout

X=[]

Y=[]

with open('../input/Kaggle_Training_Dataset.csv') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

         S=[]

         #S.append(row['deck_risk'])

         S.append(row['forecast_3_month'])

         S.append(row['forecast_6_month'])

         S.append(row['forecast_9_month'])

         S.append(row['in_transit_qty'])

         S.append(row['lead_time'])

         S.append(row['local_bo_qty'])

         S.append(row['min_bank'])

         S.append(row['national_inv'])

         S.append(row['oe_constraint'])

         S.append(row['perf_12_month_avg'])

         S.append(row['perf_6_month_avg'])

         S.append(row['pieces_past_due'])

         S.append(row['potential_issue'])

         S.append(row['ppap_risk'])

         S.append(row['sales_1_month'])

         S.append(row['sales_3_month'])

         S.append(row['sales_6_month'])

         S.append(row['sales_9_month'])

         S.append(row['sku'])

         S.append(row['stop_auto_buy'])  

         X.append(S)

         Y.append([row['went_on_backorder']])

for i in X:

    for n,s in enumerate(i):

      if s=='No':

        i[n]=0

      if s=='Yes':

        i[n]=1

      if s=='':

          i[n]=0

for i in X:

     for n,s in enumerate(i):

        i[n]=float(s)



         

for i in Y:

    for n,s in enumerate(i):

      if s=='No':

        i[n]=0

      if s=='Yes':

        i[n]=1

'''for i in Y:

     for n,s in enumerate(i):

        i[n]=float(s)'''



        

X=np.array(X).astype(float)  

Y=np.array(Y)  

model = Sequential()

model.add(Dense(100, activation='relu', input_dim=20))

model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='nadam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])     

model.fit(X,Y,verbose=1,shuffle=True, nb_epoch=3,batch_size=100,validation_split=0.2)

X=[]

Y=[]

with open('../input/Kaggle_Test_Dataset.csv') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

         S=[]

         #S.append(row['deck_risk'])

         S.append(row['forecast_3_month'])

         S.append(row['forecast_6_month'])

         S.append(row['forecast_9_month'])

         S.append(row['in_transit_qty'])

         S.append(row['lead_time'])

         S.append(row['local_bo_qty'])

         S.append(row['min_bank'])

         S.append(row['national_inv'])

         S.append(row['oe_constraint'])

         S.append(row['perf_12_month_avg'])

         S.append(row['perf_6_month_avg'])

         S.append(row['pieces_past_due'])

         S.append(row['potential_issue'])

         S.append(row['ppap_risk'])

         S.append(row['sales_1_month'])

         S.append(row['sales_3_month'])

         S.append(row['sales_6_month'])

         S.append(row['sales_9_month'])

         S.append(row['sku'])

         S.append(row['stop_auto_buy'])  

         X.append(S)

         Y.append([row['went_on_backorder']])

for i in X:

    for n,s in enumerate(i):

      if s=='No':

        i[n]=0

      if s=='Yes':

        i[n]=1

      if s=='':

          i[n]=0

for i in X:

     for n,s in enumerate(i):

        i[n]=float(s)         

for i in Y:

    for n,s in enumerate(i):

      if s=='No':

        i[n]=0

      if s=='Yes':

        i[n]=1

 

for i in X:

     for n,s in enumerate(i):

        i[n]=float(s)

score = model.evaluate(X,Y, batch_size=16)

print("LOSS")

print(score[0])

print("precision")

print(score[1])