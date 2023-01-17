'''
Author: Ritwik Biswas
Description: Using a Keras Sequential Neural Network to predict whether a mushroom is edible or poisonous
'''
import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
%matplotlib inline
import os


#random seed for reproducibility
np.random.seed(7)
df = pd.read_csv('../input/mushrooms.csv')
df.head()
total_size = df['class'].count()
print("Size: "+ str(total_size))

print("First entry sample:")
print(df.iloc[0])
class_list = []
feature_list = []

#Hash table for numerical encoding of features
num_lookup = {'a': 1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,
              'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z':26}
class_training = {16: 0, 5:1} #for binary classes to be assigned from 0 to 1
class_lookup = {0: 'poisonus', 1: 'edible'}
def encode(vec):
    '''
    Takes an shroom feature vector and encodes it to a numerical feature space
    '''
    encoded_temp = []
    for i in vec:
        try:
            val= num_lookup[i]
        except:
            val = 0
        encoded_temp.append(val)
    return encoded_temp
#Encode discrete features to numerical feature space and split 
for row in df.iterrows():
    index, data = row
    temp = encode(data.tolist())
    class_list.append(class_training[temp[0]])
    feature_list.append(temp[1:])
    
print("One data point:")
print(class_list[8000])
print(feature_list[8000])
training_size = int(0.9*total_size)
train_class = np.array(class_list[:training_size])
train_features = np.array(feature_list[:training_size])
test_class = np.array(class_list[training_size:])
test_features = np.array(feature_list[training_size:])
print("Training Length: " + str(len(train_features)))
print("Testing Length: " + str(len(test_features)))
# 4 layer network [20->10->5->1]
model = Sequential()
model.add(Dense(20, input_dim=22, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Define loss and optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_features, train_class, epochs=100, batch_size=10)
# Evaluate on Testing set
scores = model.evaluate(test_features, test_class)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#shrooms to test
test_shrooms = [['x', 'y', 'w', 't', 'p', 'f', 'c', 'n', 'n', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'u'], # poisonous
                ['x', 's', 'y', 't', 'a', 'f', 'c', 'b', 'k', 'e', 'c', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'n', 'n', 'g'], # edible
                ['x', 's', 'g', 'f', 'n', 'f', 'w', 'b', 'k', 't', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'e', 'n', 'a', 'g']] # edible
#encode shrooms
encoded_shrooms = []
for shroom in test_shrooms:
    encoded_shrooms.append(encode(shroom))
encoded_shrooms = np.array(encoded_shrooms)

#prediction
predictions = model.predict(encoded_shrooms)
print("raw_predictions: ")
print(predictions)

#decode predictions
print("\ndecoded_predictions: ")
for result in predictions:
    print(class_lookup[int(round(result[0]))])