# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading the Udemy-Courses dataset
course_dataset=pd.read_csv('../input/udemy-courses/udemy_courses.csv')
course_dataset.shape
#Identifying the columns present in the dataset
course_dataset.columns
#checking the first five rows of the dataset 
course_dataset.head()
#This displays general information about the dataset with informations like the column names their data types 
#and the count of non-null values for every column.
course_dataset.info()
#dropping the columns which are not useful for the analysis
course_dataset.drop(['course_id','course_title','url','published_timestamp'],inplace = True, axis = 1)
#checking if there are any columns which contain null values
course_dataset.isnull().sum()
#This displays information about the quantitive/numerical columns, information like count, mean, standard deviation, minimum value, maximum value 
#and the quartiles are displayed 
course_dataset.describe()
#this will help in knowing the number of categories present in each categorical variable
course_dataset.select_dtypes(['object','bool']).nunique()
print("Categories present in 'is_paid' variable:",course_dataset['is_paid'].unique())
print("Categories present in 'level' variable:",course_dataset['level'].unique())
print("Categories present in 'subject' variable:",course_dataset['subject'].unique())
#this displays the count of paid and free courses 
course_dataset['is_paid'].value_counts()
plt.figure(figsize=(20,5))
plt.style.use('ggplot')
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,wspace=0.5, hspace=0.2)
plt.subplot(141)
plt.title('Paid/Free Courses',fontsize = 15)
plt.pie(course_dataset['is_paid'].value_counts(),labels=['Paid','Free'],autopct="%1.1f%%")

plt.subplot(142)
plt.title('Count of Paid Vs. Free Courses',fontsize = 15)
sns.countplot(course_dataset['is_paid'])
#this displays the number of subscribers for paid vs. free courses:
sns.barplot(y=course_dataset['num_subscribers'],x=course_dataset['is_paid'],palette='prism')
plt.style.use('ggplot')
plt.xlabel('Paid?')
plt.ylabel('Number of subscribers')
course_dataset['level'].value_counts()
plt.figure(figsize=(20,5))
plt.style.use('ggplot')
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,wspace=0.5, hspace=0.2)
plt.subplot(141)
plt.title('Level of Courses',fontsize = 15)
plt.pie(course_dataset['level'].value_counts(),labels=['All','Beginner','Intermediate','Expert'],autopct="%1.1f%%")

plt.subplot(142)
plt.title('Count of Various Level of Courses',fontsize = 15)
sns.countplot(course_dataset['level'])
plt.xticks(rotation = 45)
plt.tight_layout()
sns.barplot(y=course_dataset['num_subscribers'],x=course_dataset['level'],palette='Reds')
plt.style.use('ggplot')
plt.xticks(rotation=45)
plt.xlabel('Course Level')
plt.ylabel('Number of subscribers')
course_dataset['subject'].value_counts()
plt.figure(figsize=(20,5))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,wspace=0.5, hspace=0.2)
plt.style.use('ggplot')
plt.subplot(141)
plt.title('Subject of Courses',fontsize = 15)
plt.pie(course_dataset['subject'].value_counts(),labels=['Web Development','Business Finance','Musical Instruments','Graphic Design'],autopct="%1.1f%%")

plt.subplot(142)
plt.title('Count of Various Subject of Courses',fontsize = 15)
sns.countplot(course_dataset['subject'])
plt.xticks(rotation = 45)
plt.tight_layout()
sns.barplot(y=course_dataset['num_subscribers'],x=course_dataset['subject'],palette='Blues')
plt.style.use('ggplot')
plt.xticks(rotation=45)
plt.xlabel('Course Subject')
plt.ylabel('Number of subscribers')
#this displays the distribution of the number of subscribers
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
sns.distplot(course_dataset['num_subscribers'],kde=False)
plt.xlabel('Number of Subscribers')
plt.title('Distribution of Number of Subscribers')
plt.figure(figsize=(20,5))
plt.style.use('ggplot')
sns.boxplot(course_dataset['num_subscribers'])
plt.xlabel('Number of Subscribers')
plt.title('Box-plot representing the distribution of number of subscribers')
#this displays the distribution of the number of subscribers
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
sns.distplot(np.log(course_dataset['num_subscribers']+1))
plt.xlabel('Number of Subscribers')
plt.title('Normal distribution of the number of subscribers')
#this displays the Corr. between Number of Subscribers & Number of Reviews
plt.figure(figsize=(10,8))
plt.title('Corr. between Number of Subscribers & Number of Reviews',fontsize = 20)
plt.style.use('ggplot')
sns.regplot(y=course_dataset['num_subscribers'],x=course_dataset['num_reviews'])
#this displays the Corr. between Number of Subscribers & price
plt.figure(figsize=(10,8))
plt.title('Corr. between Number of Subscribers and Price',fontsize = 20)
sns.regplot(y=course_dataset['num_subscribers'],x=course_dataset['price'])
plt.style.use('ggplot')
plt.xticks(rotation = 90)
#this displays the Corr. between Number of Subscribers & number of lectures
plt.figure(figsize=(10,8))
plt.title('Number of Subscribers-Number of Lectures',fontsize = 20)
plt.style.use('ggplot')
sns.regplot(y=course_dataset['num_subscribers'],x=course_dataset['num_lectures'])
#this displays the Corr. between Number of Subscribers & content duration
plt.figure(figsize=(10,8))
plt.title('Corr. between Number of Subscribers & Content duration',fontsize = 20)
plt.style.use('ggplot')
sns.regplot(y=course_dataset['num_subscribers'],x=course_dataset['content_duration'])
features=course_dataset[['num_subscribers','num_reviews','num_lectures','price','content_duration']]
## correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(features.corr(), linewidths=.5, annot=True, cmap='Reds', cbar=True)
plt.title('Correlation of Numerical Variables', fontsize=14, weight='bold')
plt.style.use('ggplot')
plt.show()
#replacing the 'True' with 1 and the 'False' with 0 in the 'is_paid' variable 
course_dataset['is_paid']=course_dataset['is_paid'].replace({True:1,False:0})
#label encoding the 'level' and 'subject' columns
lb = LabelEncoder()
X_level = course_dataset['level']
X_subject = course_dataset['subject']
course_dataset['level']=lb.fit_transform(X_level)
course_dataset['subject']=lb.fit_transform(X_subject)
#dataset after all the transformation of the categorical columns
course_dataset.head()
X=course_dataset[['is_paid','num_reviews','level','content_duration','subject']]
#target 
y=course_dataset['num_subscribers']
#splitting the dataset into training and testing subsets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 45)
from sklearn.preprocessing import MinMaxScaler

# creating a scaler
mm = MinMaxScaler()

# feeding the independent variable into the scaler
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)


#using a Random Forest model for predicting the number of subscribers 
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
#using mean square error to evaluate the performance of the model
from sklearn.metrics import mean_squared_error
mae= mean_squared_error(y_test,y_pred)
mae