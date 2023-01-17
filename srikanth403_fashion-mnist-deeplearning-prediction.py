#Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

#Labels



#Each training and test example is assigned to one of the following labels:



#0 T-shirt/top 1 Trouser 2 Pullover 3 Dress 4 Coat 5 Sandal 6 Shirt 7 Sneaker 8 Bag 9 Ankle boot



#TL;DR



#Each row is a separate image Column 1 is the class label. Remaining columns are pixel numbers (784 total). Each value is the darkness of the pixel (1 to 255)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Model building

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator



from time import time

# Input data files are available in the "../input/" directory.



import os

print(os.listdir("../input"))
#Read the Data

train=pd.read_csv('../input/fashion-mnist_train.csv')

test=pd.read_csv('../input/fashion-mnist_test.csv')
train.shape
#Explore the data

train.head(4)

train.describe()

testing=np.array(test,dtype = 'float32')

x_test= testing[:,1:]/255

y_test=testing[:,0]
train.columns.values

train.dtypes.value_counts()
xyz =np.array(train,dtype ='float32')
import random

i=random.randint(1,60000)

plt.imshow(xyz[i,1:].reshape(28,28))

label=xyz[i,0]

label
X_columns = train.columns[1:786]  

y_columns = train.columns[0]

y_columns
h2o.init()
train = h2o.import_file("../input/fashion-mnist_train.csv", destination_frame="train")

test = h2o.import_file("../input/fashion-mnist_test.csv", destination_frame="test")
X_columns = train.columns[1:785]

y_columns = train.columns[0]

y_columns
train["label"]=train["label"].asfactor()
train['label'].levels()
model = H2ODeepLearningEstimator(

                             distribution="multinomial",

                             activation = "RectifierWithDropout",

                             hidden = [32,32,32],

                             input_dropout_ratio=0.2,  

                             standardize=True,

                             epochs = 500

                             )


start = time()

model.train(X_columns,

               y_columns,

               training_frame = train)

end = time()

(end - start)/60
result = model.predict(test[: , 1:785])
result.as_data_frame().head(10) 
ab = test['label'].as_data_frame()

ab['result'] = result[0].as_data_frame()

ab.head()

ab.columns
#Accuracy

out = (ab['result'] == ab['label'])

np.sum(out)/out.size