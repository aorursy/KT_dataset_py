#Import necessary packages

import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import math

import random as rand



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,GridSearchCV



import lightgbm as lgb
filepath="../input/"

train_path = filepath+"train.csv"

training = pd.read_csv(train_path)

training.describe()
#Based on this, the picture size will be 28x28

size=int(math.sqrt(training.shape[1]-1))

size
training.head()
#Pull a random integer between 0 and number of rows in data set to test

rand_num =rand.randint(0,training.shape[0])



#Create array based on pulling row corresponding to random int

number=np.array(training.loc[rand_num][1:],dtype='uint8')



color_cutoff = 180

number[number < color_cutoff] = 0

number[number >= color_cutoff] = 254





#Create 2-D array based on image size

number=number.reshape((size,size))



print("Labeled value is",str(training.loc[rand_num][0]))

plt.imshow(number,cmap='Greys')

plt.show()
#drop_list=training.columns.to_series()['pixel0':'pixel27']+training.columns.to_series()['pixel755':'pixel783']
#Labels are fairly evenly distributed

plt.hist(training['label'])
training_labels=training['label']

training_without_labels=training.drop(labels='label',axis=1)

#added

drop_list=training.columns.to_series()['pixel0':'pixel27']+training.columns.to_series()['pixel755':'pixel783']

training=training.drop(columns=drop_list.index,axis=1)



train_data, test_data, train_labels, test_labels = train_test_split(training_without_labels, training_labels, test_size=0.7)
d_train = lgb.Dataset(train_data, label=train_labels)

params = {}

#params['learning_rate'] = 0.01

params['boosting_type'] = 'gbdt'

params['objective'] = 'multiclass'

params['metric'] = 'multi_logloss'

params['num_class'] = 10

params['num_iterations'] = 500

#params['max_bin']=50

#params['num_leaves'] = 10



#params['min_data'] = 50

#params['max_depth'] = 10

clf = lgb.train(params, d_train)
#Prediction

predict=clf.predict(test_data)
predictions = []



for x in predict:

    predictions.append(np.argmax(x))
#Accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(test_labels,predictions)
accuracy
check_results = pd.DataFrame({'actual':test_labels,'predict':predictions})

check_results['combo'] = check_results['actual'].astype('str')+" "+check_results['predict'].astype('str')

check_results.head()
check_results[check_results.actual != check_results.predict]['combo'].value_counts().sort_values(ascending=False).head(10)
final_train = lgb.Dataset(training_without_labels, label=training_labels)



final_clf = lgb.train(params, final_train)
test_path = filepath+"test.csv"

testing = pd.read_csv(test_path)

#testing=testing.drop(columns=drop_list.index,axis=1)



solutions=final_clf.predict(testing)



solution_labels = []



for x in solutions:

    solution_labels.append(np.argmax(x))

    

submission=pd.DataFrame(solution_labels,columns=['Label'])

submission.index.name = "ImageID"

submission.index += 1



submission.to_csv('submission.csv')