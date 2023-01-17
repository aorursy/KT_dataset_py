'''
Author: Ritwik Biswas
Description: Using a Keras Sequential Neural Network to predict whether a titanic passenger survives
'''
import numpy as np 
import pandas as pd 
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
%matplotlib inline
import os

df = pd.read_csv('../input/train.csv')
df.head()
total_size = df['Survived'].count()
print("Size: \n"+ str(total_size))

feature_list = []
class_list = []

#lookup tables
gender_lookup = {'male': 1, 'female': 0}
embarked_lookup = {'C' : 0, 'Q': 1, 'S': 2}

#populate feature/class lists
for row in df.iterrows():
    index, data = row
    temp = data.tolist()
    
    #append survivor class
    class_list.append(temp[1])
    
    #process remaining data into numerical encoding
    temp = temp[2:]
    temp.pop(1) #remove name
    temp.pop(5) #remove ticket
    temp.pop(6) # remove cabin
    
    #encode vars
    temp[1] = gender_lookup[temp[1]]
    try:
        temp[6] = embarked_lookup[temp[6]]
    except:
        temp[6] = 0
    if math.isnan(temp[2]):
        temp[2] = 0
#     print(temp)
    feature_list.append(temp)
print(feature_list[:2])
print(class_list[:2])
training_size = int(0.9*total_size)
train_class = np.array(class_list[:training_size])
train_features = np.array(feature_list[:training_size])
test_class = np.array(class_list[training_size:])
test_features = np.array(feature_list[training_size:])
print("Training Length: " + str(len(train_features)))
print(train_class[:4])
print("Testing Length: " + str(len(test_features)))
# 3 layer network [7->3->1] 62.11%
# model = Sequential()
# model.add(Dense(7, input_dim=7, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# 5 layer network [7->14->14->7->1] 
model = Sequential()
model.add(Dense(7, input_dim=7, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Define loss and optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_features, train_class, epochs=200, batch_size=30)
# Evaluate on Testing set
scores = model.evaluate(test_features, test_class)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
test_df = pd.read_csv('../input/test.csv')
test_df.head()
feature_test = []
id_list = []
#populate feature/class lists
count = 0
for row in test_df.iterrows():
    index, data = row
    temp = data.tolist()
    id_list.append(temp[0])
    #process data into numerical encoding
    temp = temp[1:]
    
    temp.pop(1) #remove name
    temp.pop(5) #remove ticket
    temp.pop(6) # remove cabin
    
    #encode vars
    temp[1] = gender_lookup[temp[1]]
    try:
        temp[6] = embarked_lookup[temp[6]]
    except:
        temp[6] = 0
    if math.isnan(temp[2]):
        temp[2] = 0

    feature_test.append(temp)
    count += 1
print(len(feature_test))
print(len(id_list))
print(feature_test[:2])
print(id_list[:2])
test_features = np.array(feature_test)
predictions = model.predict(test_features)
# print(str(int(round(predictions[0][0]))))
with open('prediction.csv', 'w') as the_file:
    the_file.write('PassengerId,Survived\n')
    for i in range(0,len(predictions)):
        p = round(predictions[i][0])
        if math.isnan(p):
            p = 0
        out_str = str(id_list[i]) + "," + str(int(p)) + "\n"
        print(out_str)
        the_file.write(out_str)