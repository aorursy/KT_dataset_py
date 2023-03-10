import numpy as np

import scipy as sp

import pandas as pd

import matplotlib as mpl

import seaborn as sns

import matplotlib.pyplot as plt





train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv' )

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')







print("shops \n ")

print(shops)



print(" \n \n \nitems \n ")

print(items)



print(" \n \n \n item_cats \n ")

print(item_cats)



print(" \n \n \n submission \n ")

print(submission)



print(" \n \n \n train \n ")

print(train)



print(" \n \n \n test \n " )

print(test)

traindata = pd.merge(train[['date','shop_id','item_id']], items, how='outer', on='item_id')

traindata=traindata.drop_duplicates(subset=['item_category_id', 'shop_id'], keep='first').sort_values(by=['item_category_id'])

traindata = traindata[['shop_id','item_category_id']].dropna()







testdata = pd.merge(test, items, how='inner', on='item_id')

testdata=testdata.drop_duplicates(subset=['item_category_id', 'shop_id'], keep='first').sort_values(by=['item_category_id'])



testdata = testdata[['shop_id','item_category_id']].dropna()









print(traindata.head(100))



print(testdata)

from sklearn import svm





trainx = traindata.iloc[:, :1]

trainy=traindata.iloc[:, 1:2]



testx=testdata.iloc[:, :1]

testy=testdata.iloc[:, 1:2]



clf = svm.SVC()

clf.fit(trainx, trainy)

pred=clf.predict(testx)



#print(testy.transpose())

#print(pred)





from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score





print(" \n\n accuracy score")

print(accuracy_score(testy, pred))

print(" \n\n f1 score")



f1_score(testy, pred, average =None)


mydateparser = lambda x: pd.datetime.strptime(x, "%d.%m.%Y")



train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv',parse_dates=['date'], date_parser=mydateparser )

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')



train=train.set_index('date')

print(train)





montly  =train.resample('1M').mean()



montly
train = train[train.item_id<50]

test = test[test.item_id<50]







trainx = train[['item_id','shop_id']].iloc[:, :1]

trainy=train[['item_id','shop_id']].iloc[:, 1:2]



testx=test[['item_id','shop_id']].iloc[:, :1]

testy=test[['item_id','shop_id']].iloc[:, 1:2]



clf = svm.SVC()

clf.fit(trainx, trainy)

pred=clf.predict(testx)



#print(testy.transpose())





from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score





print(" \n\n accuracy score")

print(accuracy_score(testy, pred))
mydateparser = lambda x: pd.datetime.strptime(x, "%d.%m.%Y")



train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv',parse_dates=['date'], date_parser=mydateparser )



train=train.set_index('date')

print(train)

df=train[train.shop_id==0]

sns.catplot(x="item_id",y="item_price",data=df)
df2=train.groupby(['shop_id']).mean()

df2['Y??ksekKar']=df2.item_price > 1000



df2



from sklearn import svm





trainx = df2[['item_cnt_day','Y??ksekKar']].iloc[:49, :1]

trainy = df2[['item_cnt_day','Y??ksekKar']].iloc[:49, 1:2]







testx = df2[['item_cnt_day','Y??ksekKar']].iloc[49:, :1]

testy = df2[['item_cnt_day','Y??ksekKar']].iloc[49:, 1:2]







clf = svm.SVC()

clf.fit(trainx, trainy)

pred=clf.predict(testx)



#print(testy.transpose())

#print(pred)





from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score





print(" \n\n accuracy score")

print(accuracy_score(testy, pred))


from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense







X = df2[['item_cnt_day','Y??ksekKar']].iloc[:, :1]

y = df2[['item_cnt_day','Y??ksekKar']].iloc[:, 1:2]



model = Sequential()

model.add(Dense(3, input_dim=1, activation='relu'))

model.add(Dense(2, activation='relu'))

model.add(Dense(1, activation='sigmoid'))





model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])





model.fit(X, y, epochs=100, batch_size=10)



_, accuracy = model.evaluate(X, y)

print('Accuracy: %.2f' % (accuracy*100))
df2[['item_cnt_day','Y??ksekKar']].plot()
mydateparser = lambda x: pd.datetime.strptime(x, "%d.%m.%Y")



train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv',parse_dates=['date'], date_parser=mydateparser )   #train tablosu date kolonu tarih format??nda olacak ??ekilde okunur

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')



train=train.set_index('date')  #date kolonu tarih format??nda okunduktan sonra index yap??l??r



train= train[train.item_cnt_day>0]  # yanl???? girilen g??r??lt??l?? item bilgileri temizlenir



#traindata = pd.merge(train , items, how='left', on='item_id')    #bu kod ile items tablosu ile train tablosu join yap??l??r  on k??sm??nda belirtilen parametre ??zerinden bu ger??ekle??tirilir. 

#Bu i??lem ek g??r??lt?? olu??turdu??undan performans?? d??????r??yor o y??zden kullan??lmay??p yorum sat??r??na al??nm????t??r.



#traindata=traindata.drop_duplicates(subset=['item_category_id', 'shop_id'], keep='first') # tekrar eden veri varsa silinir.

#traindata = traindata[['shop_id','item_category_id']].dropna()











from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense







X = train.iloc[:, :4]    # tablonun ilk 4 kolonu arad??????m??z sonucun bulunmas?? i??in verilecek datalar olarak ayarlan??r.

y = train.iloc[:, 4:5]  # tablonun 5. kolonu tahmin edilmesi istenen k??s??md??r. Yani item say??s??



model = Sequential()

model.add(Dense(4, input_dim=4, activation='relu'))   #4 input alan o 4 inputtan 4 d??????ml?? gizli katmana ge??en ilk gizli katman olu??turulur.

model.add(Dense(8, activation='relu'))                #1. gizli katmandan veri alan  8 d??????mden olu??an 2. gizli katman eklenir.

model.add(Dense(1, activation='sigmoid'))             # Gizli katmanlardan 1 d??????ml?? ????k???? katman??na gelindi??i k??s??m eklenir. 





model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   #model i??in kullan??lacak algoritma ve sonu??ta ????kar??lacak do??ruluk metri??i belirlenir.





model.fit(X, y, epochs=2, batch_size=100)     #modelin ka?? kez e??itilece??i ka?? epoch dan olu??aca???? ve verilerin ka??ar ka??ar verilece??i belirlenir.



_, accuracy = model.evaluate(X, y)

print('Accuracy: %.2f' % (accuracy*100))     # do??ruluk de??eri bast??r??l??r.






