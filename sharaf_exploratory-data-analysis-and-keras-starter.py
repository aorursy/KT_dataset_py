# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras 

from keras.models import Sequential



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Let's read in the data 





# We have to label the columns ourselves. Credit to Billy Strub's kernel

columns = ['Age','Workclass','fnlgwt','Education','Education Num','Marital Status',

           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',

           'Hours/Week','Country','income']



test   = pd.read_csv('../input/adult-test.csv',names = columns)

train = pd.read_csv('../input/adult-training.csv',names = columns)

# merge the datasets 

df = test.append(train)







#Let's take a look at the data

df= df.drop(0)

df.head()
df.info()
y = df['income']
y = df['income']

df = df.drop('income',axis =1 )

#Function to convert to categorical 

for label in df.columns:

    if df[label].dtype == object:

        df[label] = df[label].astype('category')

df =  pd.get_dummies(df)
# Let's see what our dataframe looks like after preprocessing 

df.head()
y = y.astype('category').cat.codes
# Let's build  a neural network!

from keras.layers import Dense

model = Sequential()

ncols = df.shape[1]



model.add(Dense(100, input_shape = (ncols,),activation = 'relu'))

model.add(Dense(100,activation = 'relu'))

model.add(Dense(1,activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy',metrics = ['accuracy'],optimizer ='adam')

df.head()
model.fit(np.array(df),np.array(y),epochs=12)
y = test['income']

test = test.drop('income',axis =1 )

#Function to convert to categorical 

for label in test.columns:

    if test[label].dtype == object:

        test[label] = test[label].astype('category')

y = y.astype('category').cat.codes

test = pd.get_dummies(test)
test.info()