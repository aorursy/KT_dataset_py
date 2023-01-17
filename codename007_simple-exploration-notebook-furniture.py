import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pandas import DataFrame

%matplotlib inline 
with open("../input/train.json") as datafile1: #first check if it's a valid json file or not
    data1 = json.load(datafile1)
with open("../input/test.json") as datafile2: #first check if it's a valid json file or not
    data2 = json.load(datafile2)
with open("../input/validation.json") as datafile3: #first check if it's a valid json file or not
    data3 = json.load(datafile3)
#test = pd.DataFrame(data2)    
#test.shape
# for training data
my_dic_data = data1
keys= my_dic_data.keys()
dict_you_want1={'my_items1':my_dic_data['annotations']for key in keys}
dict_you_want2={'my_items2':my_dic_data['images']for key in keys}
df=pd.DataFrame(dict_you_want1)
fd = pd.DataFrame(dict_you_want2)
df2=df['my_items1'].apply(pd.Series)
#print ("df2",df2.head())
fd2=fd['my_items2'].apply(pd.Series)
#print ("fd2",fd2.head())
train_data = pd.merge(df2, fd2, on='image_id', how='outer')

# for validation data
my_dic_data = data3
keys= my_dic_data.keys()
dict_you_want1={'my_items1':my_dic_data['annotations']for key in keys}
dict_you_want2={'my_items2':my_dic_data['images']for key in keys}
df=pd.DataFrame(dict_you_want1)
fd = pd.DataFrame(dict_you_want2)
df2=df['my_items1'].apply(pd.Series)
#print ("df2",df2.head())
fd2=fd['my_items2'].apply(pd.Series)
#print ("fd2",fd2.head())
validation_data = pd.merge(df2, fd2, on='image_id', how='outer')

# for test data
my_dic_data = data2
keys= my_dic_data.keys()
dict_you_want2={'my_items2':my_dic_data['images']for key in keys}
fd = pd.DataFrame(dict_you_want2)
test_data=fd['my_items2'].apply(pd.Series)
train_data['url'] = train_data['url'].apply(lambda x:str(x[0]))
test_data['url'] = test_data['url'].apply(lambda x:str(x[0]))
validation_data['url'] = validation_data['url'].apply(lambda x:str(x[0]))

train_data.head()
validation_data.head()
test_data.head()
print("size of training data", train_data.shape)
print("size of validation data", validation_data.shape)
print("size of test data", test_data.shape)
# missing data in training data set
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()
# missing data in validation data set
total = validation_data.isnull().sum().sort_values(ascending = False)
percent = (validation_data.isnull().sum()/validation_data.isnull().count()).sort_values(ascending = False)
missing_validation_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_validation_data.head()
# missing data in test data 
total = test_data.isnull().sum().sort_values(ascending = False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending = False)
missing_test_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_test_data.head()
# now open the URL
temp = 4
print('image_id', train_data['image_id'][temp])
print('url:', train_data['url'][temp])
from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])

    display(HTML(images_list))
urls = train_data['url'][15:30]
display_category(urls, "")
urls = test_data['url'][15:30]
display_category(urls, "")
urls = validation_data['url'][15:30]
display_category(urls, "")
# Unique URL's
train_data.nunique()
#Class distribution
plt.figure(figsize = (10, 8))
plt.title('Category Distribuition')
sns.distplot(train_data['label_id'])

plt.show()
# Occurance of label_id in decreasing order(Top categories)
temp = pd.DataFrame(train_data.label_id.value_counts().head(8))
temp.reset_index(inplace=True)
temp.columns = ['label_id','count']
temp
plt.figure(figsize=(15,8))
count = train_data['label_id'].value_counts().head(30)
sns.barplot(count.index,  count.values,)
plt.xlabel('label id', fontsize=12)
plt.ylabel('Cou', fontsize=12)
plt.title("Distribution of label ids", fontsize=16)
# Extract website_name for train data
temp_list = list()
for path in train_data['url']:
    temp_list.append((path.split('//', 1)[1]).split('/', 1)[0])
train_data['website_name'] = temp_list
# Extract website_name for test data
temp_list = list()
for path in test_data['url']:
    temp_list.append((path.split('//', 1)[1]).split('/', 1)[0])
test_data['website_name'] = temp_list
# Extract website_name for validation data
temp_list = list()
for path in validation_data['url']:
    temp_list.append((path.split('//', 1)[1]).split('/', 1)[0])
validation_data['website_name'] = temp_list
print("Training data size",train_data.shape)
print("test data size",test_data.shape)
print("validation data size",validation_data.shape)

train_data.head()
test_data.head()
validation_data.head()
print("Total unique websites : ",len(train_data.website_name.value_counts()))
plt.figure(figsize=(15,8))
count = train_data.website_name.value_counts().head(10)
sns.barplot(count.values, count.index)
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.xlabel('Count', fontsize=12)
plt.ylabel('websites name', fontsize=12)
plt.title("websites names with their occurances", fontsize=16)
print("Total unique websites : ",len(test_data.website_name.value_counts()))
plt.figure(figsize=(15,8))
count = test_data.website_name.value_counts().head(10)
sns.barplot(count.values, count.index)
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Website name', fontsize=12)
plt.title("Website names with their occurances", fontsize=16)
print("Total unique websites : ",len(validation_data.website_name.value_counts()))
plt.figure(figsize=(15,8))
count = validation_data.website_name.value_counts().head(10)
sns.barplot(count.values, count.index)
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Website name', fontsize=12)
plt.title("Website names with their occurances", fontsize=16)