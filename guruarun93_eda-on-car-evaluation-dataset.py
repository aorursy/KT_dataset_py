import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

from sklearn.model_selection import train_test_split

import numpy as np

import os

print(os.listdir('../input'))



%matplotlib inline
data = pd.read_csv("../input/car_evaluation.csv")

data.head()
# assigning column names 

data.columns = ["buying","maint","doors","persons","lug_boot","safety","value"]

data.head()
# to view class distribution

data.value.value_counts().plot(kind='bar', title='Count (target)');
# Class count

class_count = data.value.value_counts()

# for oversampling getting the max count

max_class = max(class_count)



# Divide DataFrame by class

df_class_0 = data[data['value'] == "acc"]

df_class_1 = data[data['value'] == "good"]

df_class_2 = data[data['value'] == "unacc"]

df_class_3 = data[data['value'] == "vgood"]



#Oversampling

df_class_0_over = df_class_0.sample(max_class,replace = True)

df_class_1_over = df_class_1.sample(max_class,replace = True)

# df_class_2_over = df_class_2.sample(max_class) # not using maximum class

df_class_3_over = df_class_3.sample(max_class,replace = True)



data_os = pd.concat([df_class_0_over,df_class_1_over,df_class_3_over,df_class_2], axis = 0)

data_os.value.value_counts().plot(kind='bar', title='Count (target)');
# data cleansing

data_os.doors = data_os.doors.replace({"5more": 5}) 

# data_os.doors = data_os.doors.replace({"3":2,"5":4,"2":2,"4":4,5:4})

data_os.persons = data_os.persons.replace({"more": 5})

data_os.head()
# label encoding

map1 = {"low" : 1, "med":2,"high":3, "vhigh": 4}

map2 = {"small" : 1, "med":2,"big":3}

data_os["buying"] = data_os["buying"].map(map1)

data_os["maint"] = data_os["maint"].map(map1)

data_os["safety"] = data_os["safety"].map(map1)

data_os["lug_boot"] = data_os["lug_boot"].map(map2)

data_os.head()
data_os["doors"]  = pd.to_numeric(data_os["doors"])

data_os["persons"] = pd.to_numeric(data_os["persons"])

data_os["car_type"] = data_os["doors"]+data_os["persons"] # created feature

type_dict = {4:"Coupe",

             5:"Coupe",

            6:"GT",

            7:"Sedan",

            8:"Hatchback",

            9:"SUV",

            10:"SUV"}

# data_os["car_type"] = data_os["car_type"].map(type_dict)



# set(data_os["car_type"].values.tolist())

data_os["car_type"] = data_os["car_type"].astype('category')
target = ['value']

reject = target

features = [x for x in data_os.columns if x not in reject]

x = data_os[features]

y = data_os[target]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25, random_state = 0)

print(xTrain.shape)

print(xTest.shape)
import sklearn

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_jobs=-1,random_state=51)



model.fit(xTrain,yTrain)

print(model.score(xTest,yTest))

print(sklearn.metrics.f1_score(yTest,model.predict(xTest),average='macro'))