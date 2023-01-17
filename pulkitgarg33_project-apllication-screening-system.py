# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/train.csv')
resource = pd.read_csv('../input/resources.csv')
test = pd.read_csv('../input/test.csv')
 
resource = resource.drop('description' , 1)
resource = resource.set_index('id').sum(level = 0).reset_index()
dataset = dataset.merge(resource , how= 'left' , on = 'id')
test = test.merge(resource , how= 'left' , on = 'id')

dataset['cost'] = dataset['quantity'] + dataset['price']
test['cost'] = test['quantity'] + test['price']
dataset.isna().sum()
test.isna().sum()

dataset = dataset.drop(['quantity' , 'price'] ,1)
dataset = dataset.drop(['project_essay_3', 'project_essay_4'] ,1)

test = test.drop(['quantity' , 'price'] ,1)
test = test.drop(['project_essay_3', 'project_essay_4'] ,1)

dataset
sns.catplot(x='school_state', y='project_is_approved',  kind='bar', data=dataset)
sns.catplot(x='project_grade_category', y='project_is_approved',  kind='bar', data=dataset)
dataset.drop(['project_grade_category' , 'school_state'] , axis =1 , inplace = True)
test.drop(['project_grade_category' , 'school_state'] , axis =1 , inplace = True)
print(dataset[['project_subject_categories', 'project_is_approved']].groupby(['project_subject_categories']).mean())
sns.catplot(x='project_subject_categories', y='project_is_approved',  kind='bar', data=dataset)
dataset.columns
print(dataset[['teacher_prefix', 'project_is_approved']].groupby(['teacher_prefix']).mean())
sns.catplot(x='teacher_prefix', y='project_is_approved',  kind='bar', data=dataset)
print(dataset[['project_subject_subcategories', 'project_is_approved']].groupby(['project_subject_subcategories']).mean())
sns.catplot(x='project_subject_subcategories', y='project_is_approved',  kind='bar', data=dataset)
dataset['cost'].hist()
print(len(dataset))
print(dataset['project_is_approved'].sum())
group = pd.cut(dataset.cost, [0,1000,20000])
piv_fare = dataset.pivot_table(index=group, columns='project_is_approved', values = 'cost', aggfunc='count')
piv_fare.plot(kind='bar')
dataset['cost'][dataset['cost'] ==0].shape
dataset
group = pd.cut(dataset.teacher_number_of_previously_posted_projects, [0,5 , 10 , 20 , 30 ,40 , 50 , 60])
piv_fare = dataset.pivot_table(index=group, columns='project_is_approved', values = 'teacher_number_of_previously_posted_projects', aggfunc='count')
piv_fare.plot(kind='bar')


print(dataset[['teacher_number_of_previously_posted_projects', 'project_is_approved']].groupby(pd.cut(dataset.teacher_number_of_previously_posted_projects, [0, 100 , 500 ])).mean())
topic_prob = dataset[['project_subject_subcategories', 'project_subject_categories', 'project_is_approved']].groupby(['project_subject_categories' , 'project_subject_subcategories']).mean()

dataset = dataset.merge(topic_prob , how= 'left' , on = ['project_subject_categories' , 'project_subject_subcategories'])
test = test.merge(topic_prob , how= 'left' , on = ['project_subject_categories' , 'project_subject_subcategories'])

dataset.drop(['project_subject_categories' , 'project_subject_subcategories'] , axis =1 , inplace = True)
test.drop(['project_subject_categories' , 'project_subject_subcategories'] , axis =1 , inplace = True)
dataset['project_essay_1'] = dataset['project_essay_1'] + dataset['project_essay_2']
test['project_essay_1'] = test['project_essay_1'] + test['project_essay_2']
dataset.drop('project_essay_2' , axis =1 , inplace = True)
test.drop('project_essay_2' , axis =1 , inplace = True)
"""
import datetime
def dow(date):
    days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dayNumber=date.weekday()
    return days[dayNumber]
def preprocess_price(price1):
    for i in range(len(price1)): 
       price1.loc[i,'Day'] = int(price1.loc[i,'project_submitted_datetime'].split(' ')[0].split('-')[2])    
       price1.loc[i,'Month'] = int(price1.loc[i,'project_submitted_datetime'].split(' ')[0].split('-')[1])    
       price1.loc[i,'Year'] = int(price1.loc[i,'project_submitted_datetime'].split(' ')[0].split('-')[0])    
       price1.loc[i , 'weekday'] = dow(datetime.date(int(price1.loc[i,'Year']) , int(price1.loc[i,'Month']) , int(price1.loc[i,'Day'])))
    price1 = price1.drop('project_submitted_datetime' , axis =1)
    return price1
"""
#new_data = preprocess_price(dataset)
dataset
dataset['topic_prob'] = dataset['project_is_approved_y']
test['topic_prob'] = test['project_is_approved']
dataset.drop('project_is_approved_y' , axis = 1 , inplace =True)

test.drop('project_is_approved' , axis = 1 , inplace =True)
dataset
dataset.drop(['teacher_id' , 'project_resource_summary' , 'project_essay_1' , 'project_title'] , axis = 1 , inplace = True)
test.drop(['teacher_id' , 'project_resource_summary' , 'project_essay_1' , 'project_title'] , axis = 1 , inplace = True)
dataset
dataset['teacher_prefix'] = dataset['teacher_prefix'].replace('Mr.', 'general').replace('Mrs.', 'general').replace('Ms.', 'general').replace('Dr.', 'professional').replace('Teacher', 'professional')
test['teacher_prefix'] = test['teacher_prefix'].replace('Mr.', 'general').replace('Mrs.', 'general').replace('Ms.', 'general').replace('Dr.', 'professional').replace('Teacher', 'professional')
dataset.drop('project_submitted_datetime' , axis = 1 , inplace =True)
test.drop('project_submitted_datetime' , axis = 1 , inplace =True)
dataset
[dataset] = [pd.get_dummies(data = df, columns = ['teacher_prefix']) for df in [dataset]]
[test] = [pd.get_dummies(data = df, columns = ['teacher_prefix']) for df in [test]]
y = np.array(dataset['project_is_approved_x']).reshape( (len(dataset) , 1) )
dataset.drop(['id', 'project_is_approved_x'] , axis = 1, inplace = True)
id = test['id']
test.drop(['id'] , axis = 1, inplace = True)
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
dataset.iloc[: , :3] = normalizer.fit_transform(dataset.iloc[: , :3])

test.fillna(test['topic_prob'].mean() , inplace = True , axis =1)
test.iloc[: , :3] = normalizer.transform(test.iloc[: , :3])
from sklearn.model_selection import train_test_split
x_train , x_val , y_train , y_val = train_test_split(dataset , y , test_size = 0.2) 
x_train.shape
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 100, epochs = 10)

# Part 3 - Making pred
y_pred_ann = classifier.predict(x_val)
y_pred_ann_threshold = classifier.predict(x_train)
len(y_pred_ann_threshold[y_train == 0])
y_pred_ann_threshold[y_train == 1][:22154]
y_train_thresh = np.concatenate(( y_pred_ann_threshold[y_train == 0] , y_pred_ann_threshold[y_train == 1][:28000]) , axis =0)
y_y_thresh = np.concatenate(( y_train[y_train == 0] , y_train[y_train == 1][:28000]) , axis =0)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(y_train_thresh.reshape((len(y_train_thresh) , 1)) , y_y_thresh.reshape((len(y_y_thresh) , 1)))
y_pred = log_reg.predict(y_pred_ann)

from sklearn.metrics import confusion_matrix , accuracy_score
print(confusion_matrix(y_pred , y_val))
print(accuracy_score(y_pred , y_val))

print(len(y_val))
print(y_val.sum())
result = classifier.predict(test)
result = log_reg.predict(result)
final = pd.DataFrame()
final['id'] = id
final['project_is_approved'] = result
final.to_csv('result_1.csv' , index = False)