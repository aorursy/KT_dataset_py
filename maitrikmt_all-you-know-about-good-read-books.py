# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("../input/goodreadsbooks/books.csv",error_bad_lines=False)
df.head(3)
# check null values in dataset
count_missing=df.isnull().sum()
percent_missing=count_missing*100/df.shape[0]
missing_value=pd.DataFrame({'Count_Missing':count_missing,
                            'percent_missing':percent_missing})
missing_value
df.drop(['bookID','isbn','isbn13'],inplace=True,axis=1)
df.rename({'  num_pages':'total_pages'},inplace=True,axis=1)
df['publication_date']=pd.to_datetime(df['publication_date'],format='%m/%d/%Y',errors='coerce')
df['Year']=df["publication_date"].dt.year
df.info()
df['language_code'].value_counts().head(5).plot(kind='pie',autopct='%1.1f%%',figsize=(8,8)).legend()
plt.figure(figsize=(20,10))
plt.title("Top 10 Languages of Books",fontsize=20)
plt.xlabel("Languages",fontsize=18)
plt.ylabel("No. of Books ",fontsize=18)
 
data=df.language_code.value_counts().head(10)
ax= sns.barplot(x=data.index,y=data.values)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=15,ha='center',va='bottom')
    

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
plt.figure(figsize=(20,10))
plt.title("Top 10 Books author",fontsize=20)
plt.xlabel("Author",fontsize=18)
plt.ylabel("No. of Books",fontsize=18)
 
data=df.authors.value_counts().head(10)
ax= sns.barplot(x=data.index,y=data.values,palette='deep')

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=15,ha='center',va='bottom')
    

plt.xticks(fontsize=18,rotation=90)
plt.yticks(fontsize=18)
plt.show()
plt.figure(figsize=(20,10))
plt.title("Top 10 Publisher",fontsize=20)
plt.xlabel("Publishers",fontsize=18)
plt.ylabel("No. of Books",fontsize=18)
 
data=df.publisher.value_counts().head(10)
ax= sns.barplot(x=data.index,y=data.values)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=15,ha='center',va='bottom')
    

plt.xticks(fontsize=18,rotation=20)
plt.yticks(fontsize=18)
plt.show()
plt.figure(figsize=(20,10))
plt.title("No of books rating wise",fontsize=20)
plt.xlabel("Ratings",fontsize=18)
plt.ylabel("No. of Books",fontsize=18)
 
data=df.average_rating.value_counts().head(10)
ax= sns.barplot(x=data.index,y=data.values)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=15,ha='center',va='bottom')
    

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

plt.figure(figsize=(20,10))
plt.title("Top 10 Text Reviews",fontsize=20)

 
data=df.nlargest(10,'text_reviews_count')
ax= sns.barplot(x=data['title'],y=data['text_reviews_count'])

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=15,ha='center',va='bottom')
plt.xlabel("Book Name",fontsize=18)
plt.ylabel("Text Reviews",fontsize=18)
    

plt.xticks(fontsize=18,rotation=90)
plt.yticks(fontsize=18)
plt.show()
plt.figure(figsize=(20,10))

 
data=df.nlargest(10,'ratings_count')
ax= sns.barplot(x=data['title'],y=data['ratings_count'])

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=15,ha='center',va='bottom')
    
plt.title("Top 10 no of ratings",fontsize=20)
plt.xlabel("Book Name",fontsize=18)
plt.ylabel("No. of rating",fontsize=18)
plt.xticks(fontsize=18,rotation=90)
plt.yticks(fontsize=18)
plt.show()
plt.title("Ratings",fontsize=20)
plt.xlabel("Rating",fontsize=18)
plt.ylabel("Frequency",fontsize=18)
df.average_rating.plot(kind='hist',bins=100,figsize=(10,7))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
plt.figure(figsize=(20,10))
plt.title("Every year no. of book publish",fontsize=20)
plt.xlabel("Year",fontsize=18)
plt.ylabel("No. of Book",fontsize=18)
 
data=df[(df.Year<2021) & (df.Year>2009)]
ax= sns.countplot(data.Year,data=data)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=15,ha='center',va='bottom')
    

plt.xticks(fontsize=18,rotation=90)
plt.yticks(fontsize=18)
plt.show()