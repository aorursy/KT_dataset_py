from datetime import datetime 

start = datetime.now()

#Importing libraries

import pandas as pd

import numpy as np

import scipy as sci

import seaborn as sns

import matplotlib.pyplot as plt

import multiprocessing

%matplotlib inline
train = pd.read_csv("../input/train.tsv", sep='\t')

test = pd.read_csv("../input/test.tsv", sep='\t')
#Getting rid of outliers

train['bigger_than_200'] = train['price'].map(lambda x: 1 if x >200 else 0)

train = train[train['bigger_than_200'] ==0]

del train['bigger_than_200']
print(train.shape)

print(test.shape)
#Checking any missing values,

import missingno as msno

msno.bar(train,sort=True,figsize=(10,5))

msno.bar(test,sort=True,figsize=(10,5))
#Getting the length of item description

train['length'] = train['item_description'].map(lambda x: len(str(x)))

test['length'] = test['item_description'].map(lambda x: len(str(x)))



np.mean(train['length'])

np.mean(test['length'])
train.head()
#Merging data

data = pd.concat([train,test])

#Defining a variable

data['train_or_not'] = data['train_id'].map(lambda x: 1 if x.is_integer() else 0)
#lowering letters

data['brand_name'] = data['brand_name'].map(lambda x: str(x).lower())

data['category_name'] = data['category_name'].map(lambda x: str(x).lower())

data['item_description'] = data['item_description'].map(lambda x: str(x).lower())

data['name'] = data['name'].map(lambda x: str(x).lower())
data['no_of_words'] = data['item_description'].map(lambda x: len(str(x).split()))



np.mean(data['no_of_words'])
##Brand names

#Number of unique brand names

print(len(set(data['brand_name'])))

print('brand_name in train',len(set(train['brand_name'])))

print('brand_name in test',len(set(test['brand_name'])))
train_cat_names= list(set(train['brand_name']))

test_cat_names= list(set(test['brand_name']))



in_test_not_in_train = [x for x in test_cat_names if x not in train_cat_names]

print(len(in_test_not_in_train))



in_train_not_in_test = [x for x in train_cat_names if x not in test_cat_names]

print(len(in_train_not_in_test))

#category

data['categories'] = data['category_name'].map(lambda x: list(str(x).split('/')))
#no descriptions

data['no_description'] = data['item_description'].map(lambda x: 1 if str(x) =='no description yet' else 0)

print(len(data[data['no_description']==1]))
print('brand_name = nan & no description',len(data[(data['brand_name']=='nan') & (data['no_description'] ==1)]))

#No brand name and no desc

no_desc_no_brand = data[(data['brand_name']=='nan') & (data['no_description'] ==1)]

no_desc_no_brand['test'] = no_desc_no_brand['test_id'].map(lambda x: 1 if x.is_integer() else 0)

no_desc_no_brand = no_desc_no_brand[no_desc_no_brand['test'] ==0]
plt.style.use('fivethirtyeight')

plt.subplots(figsize=(10,5))

no_desc_no_brand['price'].hist(bins=150,edgecolor='black',grid=False)

plt.xticks(list(range(0,100,5)))

plt.title('Price vs no brand&no_description')

plt.show() 
#No of rows whose price is bigger than 100

print("No of rows whose price is bigger than hundred in no_brand&no_description",len(no_desc_no_brand[no_desc_no_brand['price'] >200]))



no_desc_no_brand['price'].describe()

del no_desc_no_brand
from ggplot import *

p = ggplot(aes(x='price'), data=train[train['price']<200]) + geom_histogram(binwidth=10)+ theme_bw() + ggtitle('Histogram of price in train data')

print(p)
data['price'].describe().apply(lambda x: format(x, 'f'))
#Length of categories

data['len_categories'] = data['categories'].map(lambda x: len(x))
#Value_counts for item_condition_id

temp1=data['item_condition_id'].value_counts()[:5].to_frame()

sns.barplot(temp1.index,temp1['item_condition_id'],palette='inferno')

plt.title('Item condition id')

plt.xlabel('')

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()
#Making binary 'item_condition_id'

ic_list = list(set(data['item_condition_id']))



for i in ic_list:

    data['item_condition_id'+str(i)] = data['item_condition_id'].map(lambda x: 1 if x==i else 0)



del data['item_condition_id']
#Correlation between no_of_words and price

corr = data[['no_of_words','price','shipping','len_categories','length']].corr()



# Set up the matplot figure

f,ax = plt.subplots(figsize=(12,9))



#Draw the heatmap using seaborn

sns.heatmap(corr, cmap='inferno', annot=True)
##Name

import nltk

import collections as co

stopWords =co.Counter( nltk.corpus.stopwords.words() )

words = list(data['name'])

#Merging in a big string

big_string=" ".join(words)

#Splitting them via blank

name_list = big_string.split()

#Omitting splitwords

name_list = [x for x in name_list if x not in stopWords]

#Getting unique words

unique_names = list(set(name_list))

#Counting them

c = co.Counter(name_list)

most_common_100 = c.most_common(60)

most_common_100_2 = [x[0] for x in most_common_100]

#Making them a column

for i in most_common_100_2:

    data['name_'+str(i)] = data['name'].map(lambda x: 1 if i in x else 0)



print("name completed")
##Description

words1 = list(data['item_description'])

big_string1=" ".join(words1)

name_list1 = big_string1.split()



name_list1 = [x for x in name_list1 if x not in stopWords]



unique_names1 = list(set(name_list1))



c = co.Counter(name_list1)

most_common_100_desc = c.most_common(60)

most_common_100_2_desc = [x[0] for x in most_common_100_desc]

for i in most_common_100_2_desc:

    data['item_description_'+str(i)] = data['item_description'].map(lambda x: 1 if i in x else 0)



print("description completed")
##First common 200 brands

most_common_brands = data['brand_name'].value_counts().sort_values(ascending=False)[:100]



most_common_brands = list(most_common_brands.index)

#If a brand not in common brands, it was labeled as other_brand

other_brand = "other_brand"

data['brand_name'] = data['brand_name'].map(lambda x: x if x in most_common_brands else other_brand)



empty_df = pd.get_dummies(data['brand_name'])

emp_list = list(empty_df.columns.values)

emp_list = ['brand_' + str(x) for x in emp_list]

empty_df.columns = emp_list

        

data2 = pd.concat([data,empty_df],axis=1)

data = data2

del data2,empty_df

del name_list,name_list1,words,words1,big_string,big_string1

print("brand completed")
#categories

data['categories']= data['categories'].map(lambda x: list(x)+[0,0,0,0])

data['cat1']=data['categories'].map(lambda x: x[0])

data['cat2']=data['categories'].map(lambda x: x[1])

data['cat3']=data['categories'].map(lambda x: x[2])

data['cat4']=data['categories'].map(lambda x: x[3])

data['cat5']=data['categories'].map(lambda x: x[4])

most_common_cat1=data['cat1'].value_counts().sort_values(ascending=False)[:11]

most_common_cat2=data['cat2'].value_counts().sort_values(ascending=False)[:35]

most_common_cat3=data['cat3'].value_counts().sort_values(ascending=False)[:50]

most_common_cat4=data['cat4'].value_counts().sort_values(ascending=False)[:100]

most_common_cat5=data['cat5'].value_counts().sort_values(ascending=False)[:100]





#Categories, we fill focus on first 3 categories

cat1_list = list(most_common_cat1.index)

cat2_list = list(most_common_cat2.index)

cat3_list = list(most_common_cat3.index)

#If a category not in cat1, it was labeled as 'cat1_other'

cat1_other = "cat1_other"

data['cat1'] = data['cat1'].map(lambda x: x if x in cat1_list else cat1_other)

#If a category not in cat2, it was labeled as 'cat2_other'

cat2_other = "cat2_other"

data['cat2'] = data['cat2'].map(lambda x: x if x in cat2_list else cat2_other)

#If a category not in cat3, it was labeled as 'cat3_other'

cat3_other = "cat3_other"

data['cat3'] = data['cat3'].map(lambda x: x if x in cat3_list else cat3_other)
#Making binary for cat1

empty_df1 = pd.get_dummies(data['cat1'])

emp_list1 = list(empty_df1.columns.values)

emp_list1 = ['cat1_' + str(x) for x in emp_list1]

empty_df1.columns = emp_list1

#Making binary for cat2

empty_df2 = pd.get_dummies(data['cat2'])

emp_list2 = list(empty_df2.columns.values)

emp_list2 = ['cat2_' + str(x) for x in emp_list2]

empty_df2.columns = emp_list2

#Making binary for cat3

empty_df3 = pd.get_dummies(data['cat3'])

emp_list3 = list(empty_df3.columns.values)

emp_list3 = ['cat3_' + str(x) for x in emp_list3]

empty_df3.columns = emp_list3

#Merging them

data2 = pd.concat([data,empty_df1,empty_df2,empty_df3],axis=1)

data = data2

#Deleting unnecessary things

del data2,empty_df1,empty_df2,empty_df3

del data['cat1'],data['cat2'],data['cat3'],data['cat4'],data['cat5'],data['item_description'],data['name'],data['categories'],data['category_name'],data['brand_name']
print("category completed")

stop = datetime.now()

execution_time = stop-start 

print(execution_time)
test_id = data['test_id']

train_id = data['train_id']

del data['train_id'],data['test_id']

data_head = data.head()

#Separating the merged data into train and test

training = data[data['train_or_not'] ==1]

testing = data[data['train_or_not'] ==0]
del training['train_or_not']

del testing['train_or_not']
y = training['price'].values

#Deleting unnecessary columns

del training['price']

del testing['price']

train_size = len(list(training.columns.values))

train_names = list(training.columns.values)

"""

training = training.values

testing = testing.values

start = datetime.now()

import xgboost as xgb

model = xgb.XGBRegressor(n_estimators=50)

model.fit(training,y)

ending = datetime.now()

print(ending-start)

print (model)





from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(20, 15))

plot_importance(model, ax=ax)



training = pd.DataFrame(training)

testing= pd.DataFrame(testing)



temp = pd.DataFrame(model.feature_importances_)

temp2 = list(temp[temp[0]>0].index)

"""
#The numbers below represent which columns are important.We obtained them via the code above

temp2 =[0, 1, 4, 5, 8, 10, 11, 14, 17, 21, 23, 27, 50, 52, 53, 63, 65, 70, 71, 72, 73, 74, 84, 89, 91, 102, 104, 109, 110, 115, 124, 125, 131, 133, 139, 157, 158, 162, 173, 174, 178, 180, 185, 188, 194, 196, 205, 208, 218, 220, 231, 232, 235, 236, 247, 248, 251, 259, 265, 268, 272, 277, 281, 284, 288, 289, 303, 306, 308, 310, 312]

#Getting the names of important features via indexing

temp3 = [train_names[x] for x in temp2]

print("some important features are ",temp3[:20])
#Preparing model for ANN

testing.columns = train_names

training.columns = train_names

#Getting important columns

training_last = training[temp3]

testing_last = testing[temp3]

print(training_last.shape)

print(testing_last.shape)
input_node = len(list(training_last.columns.values))

print("there are ",input_node," nodes in input layer")

#Makin ndarray

training_last = training_last.values

testing_last = testing_last.values
#part 2 :Let'S make ANN

# importing the keras library

import keras

# required to initialize NN

from keras.models import Sequential

#Required to build layers of NN

from keras.layers import Dense

from keras.layers import Dropout

#Initializing the ANN

classifier = Sequential()
#adding the input layer and first hidden layer (71 nodes on Input layer, 71 nodes on Hidden Layer 1) and RELU

classifier.add(Dense(output_dim = 100 , init ='he_normal', activation ='relu',input_dim = input_node))

classifier.add(Dropout(p=0.15))

#Adding the second layer(71 nodes on Hidden layer 1, 60 nodes on Hidden Layer 2) and RELU

classifier.add(Dense(output_dim = 40 , init ='glorot_uniform', activation ='tanh'))

classifier.add(Dropout(p=0.07))

#adding the output layer- 

classifier.add(Dense(output_dim = 1 , init ='uniform'))

#compiling ANN- optimizer for weights on ANN , adam = storchastik gradient descentlerden birisi

classifier.compile( optimizer='adam' , loss='mean_squared_logarithmic_error', metrics = ['mae']  )
start = datetime.now()

classifier.fit(training_last, y ,batch_size=64,nb_epoch=8)

stop = datetime.now()

execution_time = stop-start 

print(execution_time)
#Preparing the submission file

our_pred = classifier.predict(testing_last)

our_pred = pd.DataFrame(our_pred)

ourpred = pd.DataFrame(our_pred).rename(columns={0:'price'})



test_id = test_id[len(train):len(data)]

test_id = test_id.map(lambda x: int(x))

test_id = test_id.reset_index(drop=True)

test_id = pd.DataFrame(test_id)
output_file = pd.concat([test_id,ourpred],axis=1)



print("average of test predictions = ",np.mean(output_file['price']))



output_file.to_csv('16-01-2018-mercari-ANN3.csv',index=False)