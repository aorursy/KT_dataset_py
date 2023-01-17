# importing libraries 
import pandas as pd
import numpy
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# library to save the model
import pickle
# to read the file from the location and process it to save it in a 2D list, with removing excess symbols like (';')
# each row in the list comprises of a separate data instance
def process(path_to_folder):
    train = []
    for root, dirs, files in os.walk(path_to_folder):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    temp = text.split(';\n')
                    final = []
                    for i in range (len(temp)):
                        a = temp[i].split(',')
                        final.append(a)
                
                    train = train[:] + final
                    
    return train
    
# processing the data

trainphoneaccel = process('../input/prithviai-activitydetection/data/train/phone/accel')
trainphonegyro = process('../input/prithviai-activitydetection/data/train/phone/gyro')
trainwatchaccel = process('../input/prithviai-activitydetection/data/train/watch/accel')
trainwatchgyro = process('../input/prithviai-activitydetection/data/train/watch/gyro')
trainphoneaccel[:10]
train = trainphoneaccel + trainphonegyro + trainwatchaccel + trainwatchgyro
len(train)
def transform(data):
    data = data[:-1]
    data = pd.DataFrame(data, columns = ['Subject-id', 'Activity Label', 'Timestamp', 'x', 'y', 'z'])
    return data
train = transform(train)
train.shape
# convert the elements of the dataframe from string to numeric
train = train.convert_objects(convert_numeric=True)
train.head()
# removing all the null values from the dataframe
train = train.dropna(subset = ['Subject-id','Timestamp', 'Activity Label','x', 'y', 'z'])
label = train['Activity Label'].unique()
l={}
n=0
for i in label:
    l[i] = n+1
    n+=1

train['Activity Label'] = train['Activity Label'].apply(lambda x: l[x])
train.head()
testphoneaccel = process('../input/prithviai-activitydetection/data/test/phone/accel')
testphonegyro = process('../input/prithviai-activitydetection/data/test/phone/gyro')
testwatchaccel = process('../input/prithviai-activitydetection/data/test/watch/accel')
testwatchgyro = process('../input/prithviai-activitydetection/data/test/watch/gyro')
test = testphoneaccel + testphonegyro + testwatchaccel + testwatchgyro
test = transform(test)
test = test.convert_objects(convert_numeric=True)
test = test.dropna(subset = ['Subject-id','Timestamp', 'Activity Label','x', 'y', 'z'])
l={}
n=0
for i in label:
    l[i] = n+1
    n+=1

test['Activity Label'] = test['Activity Label'].apply(lambda x: l[x])
train['Timestamp'] = train['Timestamp'].apply(lambda x: x//1000000)
train['Timestamp'] = train['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))
test['Timestamp'] = test['Timestamp'].apply(lambda x: x//1000000)
test['Timestamp'] = test['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))
train.drop(columns="Subject-id",inplace=True)
test.drop(columns="Subject-id",inplace=True)
for time in ('year','month','week','day','hour','minute','second'):
    train[time] = getattr(train['Timestamp'].dt,time)
train.drop(columns="Timestamp",inplace=True)

for time in ('year','month','week','day','hour','minute','second'):
    test[time] = getattr(test['Timestamp'].dt,time)
test.drop(columns="Timestamp",inplace=True)
train.head()
train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)
data=pd.DataFrame()
data=pd.concat([train,test])
y=data["Activity Label"]
x=data.drop(columns="Activity Label")
x_train, x_test, y_train, y_test = train_test_split(x,y , train_size = 0.7, random_state =  42)
model = RandomForestClassifier()
model.fit(x_train, y_train)

y= model.predict(x_test)
acc = accuracy_score(y_test, y)
acc
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
df = pd.DataFrame(y)
df.to_csv('answer.csv')
