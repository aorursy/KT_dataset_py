import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")
train_data.head()
test_data.head()
#missing data in training dataset
missing = train_data.isnull().sum()
all_val = train_data.count()

missing_train_df = pd.concat([missing, all_val], axis=1, keys=['Missing', 'All'])
missing_train_df
#missing data in test dataset
missing = test_data.isnull().sum()
all_val = test_data.count()

missing_test_df = pd.concat([missing, all_val], axis=1, keys=['Missing', 'All'])
missing_test_df
temp = pd.DataFrame(train_data.landmark_id.value_counts().head(8))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
temp
sns.barplot(temp.index, temp.landmark_id)
plt.figure(figsize=(10,7))
plt.title('Top 8 landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,
            label="Count")
plt.show()
temp1 = pd.DataFrame(train_data.landmark_id.value_counts().tail(8))
temp1.reset_index(inplace=True)
temp1.columns = ['landmark_id','count']
temp1
plt.figure(figsize = (10, 7))
plt.title('Least frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp1,
            label="Count")
plt.show()
train_data.nunique()
plt.figure(figsize = (10, 8))
plt.title('Category Distribuition')
sns.distplot(train_data['landmark_id'],color='grey', kde=True,bins=100)
plt.show()
from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(10).iteritems()])

    display(HTML(images_list))
    
category = train_data['landmark_id'].value_counts().keys()[2]
urls = train_data[train_data['landmark_id'] == category]['url']
display_category(urls, "")
# Extract site_names for train data
temp_list = list()
for path in train_data['url']:
    temp_list.append((path.split('//', 1)[1]).split('/', 1)[0])
train_data['site_name'] = temp_list
# Extract site_names for test data
temp_list = list()
for path in test_data['url']:
    temp_list.append((path.split('//', 1)[1]).split('/', 1)[0])
test_data['site_name'] = temp_list
train_data.head(8)
test_data.head()
# Occurance of site in decreasing order(Top categories)
temp = pd.DataFrame(train_data.site_name.value_counts())
temp.reset_index(inplace=True)
temp.columns = ['site_name','count']
temp
# Plot the Sites with their count
plt.figure(figsize = (10, 7))
plt.title('Sites by count')
sns.set_color_codes("pastel")
sns.barplot(x="site_name", y="count", data=temp,
            label="Count")
locs, labels = plt.xticks()
plt.setp(labels, rotation=85)
plt.show()
