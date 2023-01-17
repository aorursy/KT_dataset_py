# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json as js

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



games = pd.read_csv("../input/league-of-legends/games.csv")

games
champ_data = pd.read_json("../input/league-of-legends/champion_info_2.json")

champions = pd.read_json((champ_data["data"]).to_json(), orient="index")

champions.set_index(["id"], inplace = True)

champions.tail(20)
champList = ["t1_champ1id","t1_champ2id","t1_champ3id","t1_champ4id","t1_champ5id",

             "t2_champ1id","t2_champ2id","t2_champ3id","t2_champ4id","t2_champ5id"]
data = pd.read_csv("../input/league-of-legends/games.csv")
def conversion(x):

    champ = champions["name"][x]

    return champ





for column in champList:

    data[column] = data[column].apply(lambda x: conversion(x))

   



banList = ["t1_ban1","t1_ban2", "t1_ban3", "t1_ban4", "t1_ban5",

           "t2_ban1", "t2_ban2","t2_ban3", "t2_ban4", "t2_ban5"]



for column in banList:

    data[column] = data[column].apply(lambda x : conversion(x))

   
summ_data = pd.read_json("../input/league-of-legends/summoner_spell_info.json")

summoners = pd.read_json((summ_data["data"]).to_json(), orient="index")



def summ_conversion(x):

    summoner = summoners["name"][x]

    return summoner



summList = ["t1_champ1_sum1","t1_champ1_sum2","t1_champ2_sum1","t1_champ2_sum2","t1_champ3_sum1",

                 "t1_champ3_sum2","t1_champ4_sum1","t1_champ4_sum2","t1_champ5_sum1","t1_champ5_sum2",

                 "t2_champ1_sum1","t2_champ1_sum2","t2_champ2_sum1","t2_champ2_sum2","t2_champ3_sum1",

                 "t2_champ3_sum2","t2_champ4_sum1","t2_champ4_sum2",

                 "t2_champ5_sum1","t2_champ5_sum2"]





for column in summList:

    data[column] = data[column].apply(lambda x : summ_conversion(x))





data
data.columns
sumPicks = pd.concat([data['t1_champ1id'],data['t1_champ2id'],data['t1_champ3id'],data['t1_champ4id'],data['t1_champ5id'],

                      data['t2_champ1id'],data['t2_champ2id'],data['t2_champ3id'],data['t2_champ4id'],data['t2_champ5id']],

                      ignore_index=True)

sortedPicks = sorted(sumPicks)

sumBans = pd.concat([data['t1_ban1'],data['t1_ban2'],data['t1_ban3'],data['t1_ban4'],data['t1_ban5'],

                     data['t2_ban1'],data['t2_ban2'],data['t2_ban3'],data['t2_ban4'],data['t2_ban5']],

                     ignore_index=True)

sortedBans = sorted(sumBans)
import matplotlib.pyplot as plt 

import seaborn as sns

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,30))

plt.xticks(rotation=90)

sns.countplot(y=sortedPicks, data=data, ax=ax1, order=sumPicks.value_counts().index)

sns.countplot(y=sortedBans, data=data, ax=ax2 , order=sumBans.value_counts().index )

ax1.set_title('Champion Picks')

ax2.set_title('Champion Bans')

plt.show()
Ndata = pd.DataFrame(sumPicks.value_counts())

Ndata["Bans"]= sumBans.value_counts()

Ndata
Ndata.rename(columns={0: "Picks"}, inplace = True)

Ndata
#ax = plt.gca()



#Ndata.plot(kind='bar',y='Bans',ax=ax ,figsize=(50,25) )

#Ndata.plot(kind='bar',y='Picks', color='red', ax=ax)

#plt.xticks(rotation=90)

#plt.show()





Ndata.plot(kind='bar',figsize=(50,25))
def create_label_encoder_dict(df):

    from sklearn.preprocessing import LabelEncoder

    

    label_encoder_dict = {}

    for column in df.columns:

        # Only create encoder for categorical data types

        if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':

            label_encoder_dict[column]= LabelEncoder().fit(df[column])

    return label_encoder_dict
label_encoders = create_label_encoder_dict(data)

print("Encoded Values for each Label")

print("="*32)

for column in label_encoders:

    print("="*32)

    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))

    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)
data2 = data.copy() # create copy of initial data set

for column in data2.columns:

    if column in label_encoders:

        data2[column] = label_encoders[column].transform(data2[column])



print("Transformed data set")

print("="*32)

data2.head(15)
X_data = data2[["t1_ban1","t1_ban2", "t1_ban3", "t1_ban4", "t1_ban5",

           "t2_ban1", "t2_ban2","t2_ban3", "t2_ban4", "t2_ban5"]]

Y_data = data2['t1_champ1id']
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
from sklearn.neural_network import MLPClassifier
# Create an instance of linear regression

reg = MLPClassifier()

#reg = MLPClassifier(hidden_layer_sizes=(8,120))
reg.fit(X_train,y_train)
reg.n_layers_ # Number of layers utilized
test_predicted = reg.predict(X_test)

test_predicted
data3 = X_test.copy()

data3['predicted_pick']=test_predicted

data3['predicted_pick_en']=label_encoders['t1_champ1id'].inverse_transform(test_predicted)

data3['pick']=data3['t1_champ1id']=y_test

data3['pick_en']=label_encoders['t1_champ1id'].inverse_transform(y_test)

data3.head(40)


k=(reg.predict(X_test) == y_test)
k.value_counts()
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, reg.predict(X_test), labels=y_test.unique())

cm
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    import itertools

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
plt.figure(figsize=(45,45),dpi = 300)

plot_confusion_matrix(cm,data['t1_champ1id'].unique())
X_data2 = data2[["t1_ban1","t1_ban2", "t1_ban3", "t1_ban4", "t1_ban5",

           "t2_ban1", "t2_ban2","t2_ban3", "t2_ban4", "t2_ban5","t1_champ1id","t1_champ2id","t1_champ3id","t2_champ1id","t2_champ2id"]]

Y_data2 = data2['t2_champ3id']
# use new data for  train/test split

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_data2, Y_data2, test_size=0.30)
reg2= MLPClassifier()

reg2.fit(X_train2,y_train2)
reg2.n_layers_ # Number of layers utilized
test_predicted2 = reg2.predict(X_test2)

test_predicted2
data4 = X_test2.copy()

data4['predicted_pick']=test_predicted2

data4['predicted_pick_en']=label_encoders['t2_champ3id'].inverse_transform(test_predicted2)

data4['pick']=data3['t2_champ3id']=y_test2

data4['pick_en']=label_encoders['t2_champ3id'].inverse_transform(y_test2)

data4.head(40)
k2=(reg2.predict(X_test2) == y_test2)
k2.value_counts()
cm2=confusion_matrix(y_test2, reg2.predict(X_test2), labels=y_test2.unique())

cm2
plt.figure(figsize=(45,45),dpi = 300)

plot_confusion_matrix(cm2,data['t2_champ3id'].unique())
X_data3 = data2[["t1_ban1","t1_ban2", "t1_ban3", "t1_ban4", "t1_ban5",

           "t2_ban1", "t2_ban2","t2_ban3", "t2_ban4", "t2_ban5","t1_champ1id","t1_champ2id","t1_champ3id","t1_champ4id","t1_champ5id",

             "t2_champ1id","t2_champ2id","t2_champ3id","t2_champ4id"]]

Y_data3 = data2['t2_champ5id']
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_data3, Y_data3, test_size=0.30)
reg3 = MLPClassifier()

reg3.fit(X_train3,y_train3)
reg3.n_layers_ # Number of layers utilized
test_predicted3 = reg3.predict(X_test3)

test_predicted3
data5 = X_test3.copy()

data5['predicted_pick']=test_predicted3

data5['predicted_pick_en']=label_encoders['t2_champ5id'].inverse_transform(test_predicted3)

data5['pick']=data3['t2_champ5id']=y_test3

data5['pick_en']=label_encoders['t2_champ5id'].inverse_transform(y_test3)

data5.head(40)
k3=(reg3.predict(X_test3) == y_test3)
k3.value_counts()
cm3=confusion_matrix(y_test3, reg3.predict(X_test3), labels=y_test3.unique())

cm3
plt.figure(figsize=(45,45),dpi = 300 )

plot_confusion_matrix(cm3,data['t2_champ5id'].unique())
Xnew = [[99, 55,76,11,23,24,16,19,15,88]]

Ntest_predicted = reg.predict(Xnew)

print("X=%s, Predicted=%s" % (Xnew, label_encoders['t1_champ1id'].inverse_transform(Ntest_predicted)))