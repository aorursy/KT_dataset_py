# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/diabetes.csv')

data.head()
data.info()
plt.figure(figsize=(10,10))

sns.heatmap(data.corr(),annot = True)
data.Insulin.unique()
data.head()
x = data[data.Insulin==0].index

data.drop(x,inplace=True)

data.dropna(inplace=True)
data['Agepart'] = ' '
for j,each in enumerate(data.Age):

    if int(each/5)==4:

        data.iloc[j,9]= 0

    elif int(each/5)==5:

        data.iloc[j,9] = 1

    elif int(each/5)==6:

        data.iloc[j,9] = 2

    elif int(each/5)==7:

        data.iloc[j,9] = 3

    elif int(each/5)==8:

        data.iloc[j,9] = 4

    elif int(each/5)==9:

        data.iloc[j,9] = 5

    elif int(each/5)==10:

        data.iloc[j,9] = 6

    elif int(each/5)==11:

        data.iloc[j,9] = 7

    elif int(each/5)==12:

        data.iloc[j,9] = 8

    elif int(each/5)==13:

        data.iloc[j,9] = 9

    elif int(each/5)==14:

        data.iloc[j,9] = 10

    elif int(each/5)==15:

        data.iloc[j,9] = 11

    elif int(each/5)==16:

        data.iloc[j,9] = 12

    elif int(each/5)==17:

        data.iloc[j,9] = 13

    

        

data.head()
# 

from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential,load_model

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop,Adam



model = Sequential()

#

model.add(Dense(256, activation = "relu",input_dim=7))

model.add(Dropout(0.5))

model.add(Dense(64, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(14, activation = "softmax"))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
data.head()
from keras.callbacks import ModelCheckpoint ##this helps to save our model so that

x = data[data.Outcome==1].drop(['Age','Agepart','Outcome'],axis=1)

x = (x-np.min(x))/(np.max(x)-np.min(x))

y = data[data.Outcome==1].Agepart



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical





#define the model checkpoint callback -> this will keep on saving the model as a physical file

model_checkpoint = ModelCheckpoint('fas_mnist_1.h5', verbose=1, save_best_only=True)
k_fold = 5

histor = []

for each in range(k_fold):

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=np.random.randint(1,1000,1)[0])

    y_train = to_categorical(y_train,num_classes=14)

    history = model.fit(x_train,y_train,epochs=1000,batch_size=50,validation_split=0.1,callbacks=[ model_checkpoint], 

              verbose=1)

    histor.append(history)


plt.figure(figsize=(10,5))



plt.plot(histor[0].history['val_acc'],color='red')

plt.plot(histor[0].history['acc'],color='blue')

plt.figure(figsize=(10,5))



plt.plot(histor[1].history['val_acc'],color='red')

plt.plot(histor[1].history['acc'],color='blue')

plt.figure(figsize=(10,5))

plt.plot(histor[2].history['val_acc'],color='red')

plt.plot(histor[2].history['acc'],color='blue')

plt.figure(figsize=(10,5))

plt.plot(histor[3].history['val_acc'],color='red')

plt.plot(histor[3].history['acc'],color='blue')

plt.figure(figsize=(10,5))

plt.plot(histor[4].history['val_acc'],color='red')

plt.plot(histor[4].history['acc'],color='blue')



x = load_model('fas_mnist_1.h5')##loading saved model
x.evaluate(x_train,y_train)
x_test = data[data.Outcome==0].drop(['Age','Agepart','Outcome'],axis=1)

x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
y_head = x.predict(x_test)
y_head
y_head=np.argmax(y_head,axis=1)
y_head
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=1000,random_state=1)

x = data.drop(['Age','Outcome'],axis=1)

x = (x-np.min(x))/(np.max(x)-np.min(x))

x_train,x_test,y_train,y_test=train_test_split(x,data.Outcome,test_size=0.1,random_state=1)

rf.fit(x_train,y_train)

rf.score(x_test,y_test)
a = data.Outcome==0

x=data[a].drop(['Outcome'],axis=1)

x['Agepart']=y_head

x = (x-np.min(x))/(np.max(x)-np.min(x))

y_head1 = rf.predict(x.drop(['Age'],axis=1))
y_head1
len(y_head1[y_head1==1])
x['Outcome'] = y_head1
x['Agenow']=data[data.Outcome==0].Agepart # real age 

x['Agepart']=y_head # predicted diabetes age
x.head()
trace1 = go.Bar(

    x = x.Agenow[x.Outcome==1].index,

    y = x.Agenow[x.Outcome==1].values,

    marker = dict(color = 'red'),

    opacity = 0.7

    )

trace2 = go.Bar(

    x = x.Agepart[x.Outcome==1].index,

    y = x.Agepart[x.Outcome==1].values,

    marker = dict(color = 'blue'),

     opacity = 0.3

    )

data = [trace1,trace2]

layout = dict(barmode='overlay')

fig = dict(data=data,layout=layout)

iplot(fig)