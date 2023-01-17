import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Data = pd.read_csv("../input/udemy-courses/udemy_courses.csv")



Data.info()

Data[0:10]
cnt_pro = Data['is_paid'].value_counts()

plt.figure(figsize=(6,4))

sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)

plt.ylabel('Number of Data', fontsize=12)

plt.xlabel('is_paid', fontsize=12)

plt.xticks(rotation=80)

plt.show();
cnt_pro = Data['subject'].value_counts()

plt.figure(figsize=(6,4))

sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)

plt.ylabel('Number of subject', fontsize=12)

plt.xlabel('Subject', fontsize=12)

plt.xticks(rotation=80)

plt.show();
num_reviews= Data[Data['level']=='All Levels'].groupby(['is_paid']).agg({'num_reviews':['sum']})

num_lectures = Data[Data['level']=='All Levels'].groupby(['is_paid']).agg({'num_lectures':['sum']})

total= num_reviews.join(num_lectures)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total.plot(ax=plt.gca(), title='All Levels')
num_reviews= Data[Data['level']=='Beginner Level'].groupby(['is_paid']).agg({'num_reviews':['sum']})

num_lectures = Data[Data['level']=='Beginner Level'].groupby(['is_paid']).agg({'num_lectures':['sum']})

total= num_reviews.join(num_lectures)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total.plot(ax=plt.gca(), title='Beginner Level')
num_reviews= Data[Data['level']=='Intermediate Level'].groupby(['is_paid']).agg({'num_reviews':['sum']})

num_lectures = Data[Data['level']=='Intermediate Level'].groupby(['is_paid']).agg({'num_lectures':['sum']})

total= num_reviews.join(num_lectures)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total.plot(ax=plt.gca(), title='Intermediate Level')
num_reviews= Data[Data['level']=='Expert Level'].groupby(['is_paid']).agg({'num_reviews':['sum']})

num_lectures = Data[Data['level']=='Expert Level'].groupby(['is_paid']).agg({'num_lectures':['sum']})

total= num_reviews.join(num_lectures)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total.plot(ax=plt.gca(), title='Expert Level')
#Top 30 Most Popular Courses by num_subscribers

top_course = Data.sort_values(by='num_subscribers', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_course.course_title, x=top_course.num_subscribers)

plt.xticks()

plt.xlabel('num_subscribers')

plt.ylabel('Course_title')

plt.title('The Most Popular Courses')

plt.show()
#Top 30 Most Popular Courses Reviews by num_reviews

top_course = Data.sort_values(by='num_reviews', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_course.course_title, x=top_course.num_reviews)

plt.xticks()

plt.xlabel('num_reviews')

plt.ylabel('Course_title')

plt.title('The Most Popular Courses Reviews')

plt.show()
#Top 30 Num_lectures by num_lectures

top_course = Data.sort_values(by='num_lectures', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_course.course_title, x=top_course.num_lectures)

plt.xticks()

plt.xlabel('num_lectures')

plt.ylabel('Course_title')

plt.title('Num_lectures Courses')

plt.show()
Data1= Data[['is_paid','price','num_subscribers','num_reviews','num_lectures','content_duration','subject']] #Subsetting the data

cor = Data.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
Data1.head()
#Frequency distribution of classes"

train_outcome = pd.crosstab(index=Data["is_paid"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
#Select feature column names and target variable we are going to use for training

subject = {'Web Development': 1 ,'Business Finance': 2, 'Musical Instruments': 3, 'Graphic Design': 4} 

Data1.subject = [subject[item] for item in Data1.subject] 

print(Data1)
print("Any missing sample in test set:",Data1.isnull().values.any(), "\n")
from sklearn.model_selection import train_test_split

Y = Data1['is_paid']

X = Data1.drop(columns=['is_paid'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=9)
print('X train shape: ', X_train.shape)

print('Y train shape: ', Y_train.shape)

print('X test shape: ', X_test.shape)

print('Y test shape: ', Y_test.shape)
X_test
Y_test
# We define the number of trees in the forest in 100. 



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix



# We define the model

rfcla = RandomForestClassifier(n_estimators=100,random_state=49,n_jobs=-1)



# We train model

rfcla.fit(X_train, Y_train)



# We predict target values

Y_predict5 = rfcla.predict(X_test)
Y_predict5
test_acc_rfcla = round(rfcla.fit(X_train,Y_train).score(X_test, Y_test)* 100, 2)

train_acc_rfcla = round(rfcla.fit(X_train, Y_train).score(X_train, Y_train)* 100, 2)
# The confusion matrix

rfcla_cm = confusion_matrix(Y_test, Y_predict5)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(rfcla_cm, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="Greens")

plt.title('Random Forest Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
model1 = pd.DataFrame({

    'Model': ['Random Forest'],

    'Train Score': [train_acc_rfcla],

    'Test Score': [test_acc_rfcla]

})

model1.sort_values(by='Test Score', ascending=False)
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict5)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt



disp = plot_precision_recall_curve(rfcla,X_train, Y_train)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))