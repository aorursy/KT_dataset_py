# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os #operating system 

for dirname, _, filenames in os.walk('/kaggle/input'):   

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
for a in os.walk('/kaggle/input'):

    print(a)  #extracts a tuple 
for a, b, c in os.walk('/kaggle/input'):

    print(a,b,c) #extracts what is inside the tuple. 
for a, b, c in os.walk('/kaggle/input'):

    print(c)
for a,b,c in os.walk('/kaggle/input'):

    for d in c: #for loop in a for loop 

        print(d)
#profversion

for a, b, c in os.walk('/kaggle/input'):

    print(a,b,c)

    for d in c:

        print(d)

df=pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv',low_memory=False)
df

#various PlayId in one GameId. 
df.head()
df.tail(10)
df.info()
df['PlayerHeight']
df[['PlayerHeight','PlayerWeight']]
df.iloc[0:20]
df.loc[0] #the first row 
df.isnull()
df.isnull().sum()
#grouping

g=df.groupby('GameId')
g.size() #we have 512 games and we have number of plays in each game.
g['PlayId']
import matplotlib.pylab as plt
df.groupby('PlayId').first()['Yards'].plot(kind='hist',

            figsize=(15,5),

            bins=100,

            title ='Distributions of Yards Gained (Target)')

plt.show()
train=df.select_dtypes(include='number') #only the numeric data
train #now we only have 25 columns 
Y=train.pop('Yards') #popping the 'Yards' column from the train data
Y
train #'Yards'is popped out so now we have 24 columns this will be the shape of the input later on. 
import tensorflow as tf
my_model=tf.keras.Sequential([

    tf.keras.layers.Dense(1, input_shape=[24])

])

#we made a very/extremely simple model with a dense layer with one node
my_model.compile(

    loss='mse',

    optimizer='adam'

)
my_model.fit(train,Y,epochs=1) 

#loss:nan there is something wrong! 

#we had a lot of missing data it is simply impossible to fit a model with missing data
train=train.dropna()
Y = train.pop('Yards')

df
train = df.select_dtypes(include='number')
train.info()
train=train.dropna()
Y = train.pop('Yards')
train.info()
my_model.fit(train, Y, epochs=1)

train.pop('GameId')

train.pop('PlayId')

my_model.fit(train, Y, epochs = 1)

#we popped out 2 columns input shape must be edited
my_model = tf.keras.Sequential([

    tf.keras.layers.Dense(1, input_shape=[22])

])
my_model.compile(

    loss = 'mse',

    optimizer = 'adam'

)
my_model.fit(train, Y, epochs = 1)

my_model_2 = tf.keras.Sequential([

    tf.keras.layers.Dense(512, input_shape=[22], activation='relu'),

    tf.keras.layers.Dense(1)

])



my_model_2.compile(

    loss = 'mse',

    optimizer = 'adam'

)
my_model_2.fit(train, Y, epochs = 10)