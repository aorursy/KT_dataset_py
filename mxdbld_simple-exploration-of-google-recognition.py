import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
# opening train/test csv files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(5)
# 10 for 1 ration
print(f"Train shape : {train.shape}")
print(f"Test shape : {test.shape}")
# Dataset organization
train.nunique()
plt.figure(figsize = (8, 5))
plt.title('Category Distribuition')
sns.distplot(train['landmark_id'])

plt.show()
# Top categories
print(train['landmark_id'].value_counts().head(7))
print(f"Median number : {train['landmark_id'].value_counts().median()}")
print(f"Mean number : {train['landmark_id'].value_counts().mean()}")
# More exhaustive description
train['landmark_id'].value_counts().describe()
f"Number of classes under 10 occurences : {(train['landmark_id'].value_counts() <= 10).sum()}/{len(train['landmark_id'].unique())}"
from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])

    display(HTML(images_list))

category = train['landmark_id'].value_counts().keys()[0]
urls = train[train['landmark_id'] == category]['url']
display_category(urls, "")
    

category = train['landmark_id'].value_counts().keys()[1]
urls = train[train['landmark_id'] == category]['url']
display_category(urls, "")
category = train['landmark_id'].value_counts().keys()[2]
urls = train[train['landmark_id'] == category]['url']
display_category(urls, "")
