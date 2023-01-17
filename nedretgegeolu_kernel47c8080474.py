# -*- coding: utf-8 -*-

"""

@author: Ned212

"""

from pandas import read_csv

from numpy import array

from collections import Counter

import numpy as np



from sklearn.linear_model import LogisticRegression 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler





#eğitim veri kümesini alır.

train_data = read_csv("../input/traintest/train3700.csv")



#test veri kümesini alır.

test_data = read_csv('../input/traintest/test.csv')



#test veri kümesi uzunluğunu yazdırır.

test_len = len(test_data)

print("{0} Data Has Been Loaded!".format(test_len))

print(test_data.columns)



#eğitim veri kümesi uzunluğunu yazdırır.

train_len = len(train_data)

print("{0} Data Has Been Loaded!".format(train_len))

print(train_data.columns)



#dizi kullanılacak algoritmaları içerir.

algoritms = ["Logistic Regression", "Decision Tree Classifier", "SVC", "KNeighbors Classifier","Random Forest"]



#dizi elde edilen skorları içerir.

scores = []



#dizi tahminleri içerir.

predictions_list = []



#Eğitim - değer veri kümesindeki sonuç (hedef) karşılığını içerir.

y = train_data["item_cnt_day"]



shopId=input("Enter Shop ID: ")

shopId=int(shopId)



itemId= input("Enter Item ID: ")

itemId=int(itemId)



price= input("Enter Item Price: ")

price=int(price)



testDizi=((shopId,itemId,price),(shopId,itemId,price))

yDataDizi=[["x"],["x"]]



#Test - değer veri kümesindeki sonuç (hedef) karşılığını içerir.

#y_data = test_data["item_cnt_day"]

y_data = np.array(yDataDizi)[0:1,0:1]



#date_block_num,shop_id,item_id,item_price

#dizi veri kümesindeki değişkenleri içerir.

parameters = ["date_block_num", "shop_id", "item_id", "item_price"]



#Eğitim - değer veri kümesindeki değişkenler dizisini içerir.

x = train_data[parameters]



#Test- değer veri kümesindeki değişkenler dizisini içerir.

x_data = test_data[parameters]



#le = preprocessing.LabelEncoder()

#x = x.apply(le.fit_transform)



#Eğitim veri ataması yapılır.

x_train, y_train = x,y



#Test veri ataması yapılır.

x_test, y_test = x_data, y_data





#Logistic Regression Fit

Logistic_Regression = LogisticRegression().fit(x_train,y_train)



#DecisionTreeClassifier Fit

Decision_Tree_Classifier = DecisionTreeClassifier().fit(x_train,y_train)



#SVC Fit

SVC = SVC().fit(x_train,y_train)



#RandomForestClassifier Fit

RFC = RandomForestClassifier(n_estimators = 10 , criterion = 'entropy').fit(x_train,y_train);



#KNeighborsClassifier Fit

K_Neighbors_Classifier = KNeighborsClassifier(3).fit(x_train,y_train)



#Her bir algoritmanın EĞİTİM sonucu diziye eklenir.

scores.append(Logistic_Regression.score(x_train,y_train))

scores.append(Decision_Tree_Classifier.score(x_train,y_train))

scores.append(SVC.score(x_train,y_train))

scores.append(RFC.score(x_train,y_train))



scores.append(K_Neighbors_Classifier.score(x_train,y_train))



#Her bir algoritmanın TEST sonucu diziye eklenir.

predictions_LR = Logistic_Regression.predict(x_test)

predictions_DTC = Decision_Tree_Classifier.predict(x_test)

predictions_SVC = SVC.predict(x_test)

predictions_RFC = RFC.predict(x_test)



predictions_KNC = K_Neighbors_Classifier.predict(x_test)



#Her bir algoritmanın TEST PUANI diziye eklenir.

predictions_list.append(accuracy_score(y_test, predictions_LR))

predictions_list.append(accuracy_score(y_test, predictions_DTC))

predictions_list.append(accuracy_score(y_test, predictions_SVC))

predictions_list.append(accuracy_score(y_test, predictions_RFC))



predictions_list.append(accuracy_score(y_test, predictions_KNC))





print("\n\nLogistic Regression: How many item sells today: ",predictions_LR)

print("Desicion Tree : How many item sells today: ",predictions_DTC)

print("Support Vector Machine : How many item sells today: ",predictions_SVC)

print("K- Nearest Neighbors : How many item sells today: ",predictions_KNC)

print("Random Forest Classifier : How many item sells today: ",predictions_RFC)





pre_list = []

result_counter = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for result in predictions_KNC:

       

    if result not in pre_list:

        pre_list.append(result)

        

    if result in pre_list:

        myindex = pre_list.index(result)

        result_counter[myindex] = result_counter[myindex] + 1

        





print("\nSale Prediction: \n")

i = 0

for train in pre_list:  

    print("How Many Item Sells Today: ",pre_list[i])

    i = i + 1



sayac = 0

eb = 0

for endResult in result_counter:

    if result_counter[sayac] > int(eb):

        eb = result_counter[sayac]        

    sayac = sayac + 1

    

        



print("********************************")