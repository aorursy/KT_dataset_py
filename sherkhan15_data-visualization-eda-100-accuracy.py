# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.python.data import Dataset

import keras

from keras.utils import to_categorical

import seaborn as sns

import sklearn

from sklearn.preprocessing import StandardScaler

from keras import models

from keras import layers
df=pd.read_csv('/kaggle/input/forest-cover-type-dataset/covtype.csv',index_col=0)
df.head()
df.info()

# We can see all columns have diffrent data types , float64(47), int64(7).
df.describe()
print(df.shape)



# We can see that there are 154340 instances having 55 attributes
# Statistical description



pd.set_option('display.max_columns', None)

print(df.describe())



# Learning :

# No attribute is missing as count is 581012 for all attributes. Hence, all rows can be used

# Negative value(s) present in Vertical_Distance_To_Hydrology. Hence, some tests such as chi-sq cant be used.

# Wilderness_Area and Soil_Type are one hot encoded. Hence, they could be converted back for some analysis

# Scales are not the same for all. Hence, rescaling and standardization may be necessary for some algos
# Skewness of the distribution



print(df.skew())



# Values close to 0 show less skew

# Several attributes in Soil_Type show a large skew. Hence, some algos may benefit if skew is corrected
# Number of instances belonging to each class



df.groupby('Cover_Type').size()





# We see that all classes not have an equal presence. So, class re-balancing is necessary
a=df['Cover_Type']

sns.countplot(a)
g = df.groupby('Cover_Type')

g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))

g.head(8)
import matplotlib.pyplot as plt
import numpy







#sets the number of features considered

size = 10 



#create a dataframe with only 'size' features

data=df.iloc[:,:size] 



#get the names of all the columns

cols=data.columns 



# Calculates pearson co-efficient for all combinations

data_corr = data.corr()



# Set the threshold to select only only highly correlated attributes

threshold = 0.5



# List of pairs along with correlation above threshold

corr_list = []



#Search for the highly correlated pairs

for i in range(0,size): #for 'size' features

    for j in range(i+1,size): #avoid repetition

        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):

            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index



#Sort to show higher ones first            

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))



#Print correlations and column names

for v,i,j in s_corr_list:

    print ("%s and %s = %.2f" % (cols[i],cols[j],v))



# Strong correlation is observed between the following pairs

# This represents an opportunity to reduce the feature set through transformations such as PCA
for v,i,j in s_corr_list:

    sns.pairplot(df, hue="Cover_Type", height=6, x_vars=cols[i],y_vars=cols[j] )

    plt.show()



col_list = df.columns

col_list = [col for col in col_list if not col[0:4]=='Soil']

fig, ax = plt.subplots(figsize=(10,10))  

sns.heatmap(df[col_list].corr(),square=True,linewidths=1)

plt.title('Correlation of Variables')



plt.figure(figsize=(10,10))

sns.boxplot(y='Slope',x='Cover_Type', data= df )

plt.title('slope vs Cover_Type')





sns.pairplot( df, hue='Cover_Type',vars=['Aspect','Slope','Hillshade_9am','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Fire_Points'],diag_kind="kde")

plt.show()


sns.lmplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', data=df, hue='Soil_Type2',fit_reg=False)
sns.lmplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', data=df, hue='Wilderness_Area1',fit_reg=False)




#names of all the attributes 

cols = df.columns



#number of attributes (exclude target)

size = len(cols)-1



#x-axis has target attribute to distinguish between classes

x = cols[size]



#y-axis shows values of an attribute

y = cols[0:size]



#Plot violin for all attributes

for i in range(0,size):

    sns.violinplot(data=df,x=x,y=y[i])  

    plt.show()



#Elevation is has a separate distribution for most classes. Highly correlated with the target and hence an important attribute

#Aspect contains a couple of normal distribution for several classes

#Horizontal distance to road and hydrology have similar distribution

#Hillshade 9am and 12pm display left skew

#Hillshade 3pm is normal

#Lots of 0s in vertical distance to hydrology

#Wilderness_Area3 gives no class distinction. As values are not present, others gives some scope to distinguish

#Soil_Type, 1,5,8,9,12,14,18-22, 25-30 and 35-40 offer class distinction as values are not present for many classes
#Removal list initialize

rem = []



#Add constant columns as they don't help in prediction process

for c in df.columns:

    if df[c].std() == 0: #standard deviation is zero

        rem.append(c)



#drop the columns        

df.drop(rem,axis=1,inplace=True)



print(rem)
from sklearn import preprocessing



x = df[df.columns[:55]]

y = df.Cover_Type

x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)


train_norm = x_train[x_train.columns[0:10]]

test_norm = x_test[x_test.columns[0:10]]
std_scale = preprocessing.StandardScaler().fit(train_norm)

x_train_norm = std_scale.transform(train_norm)

training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 

x_train.update(training_norm_col)

print (x_train.head())
x_test_norm = std_scale.transform(test_norm)

testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 

x_test.update(testing_norm_col)

print (x_train.head())

model = keras.Sequential([

 keras.layers.Dense(64, activation=tf.nn.relu,                  

 input_shape=(x_train.shape[1],)),

 keras.layers.Dense(64, activation=tf.nn.relu),

 keras.layers.Dense(8, activation=  'softmax')

 ])



model.compile(optimizer=tf.optimizers.Adam(),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

history2 = model.fit(

 x_train, y_train,

 epochs=5, batch_size = 60,

 validation_data = (x_test, y_test))


from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt

import numpy



history = model.fit(x_train, y_train, epochs=5,validation_split=0.7, shuffle=True)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()