!pip install dabl
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
data
data.isnull().sum()
import dabl

plt.rcParams['figure.figsize'] = (18, 6)

plt.style.use('fivethirtyeight')

dabl.plot(data, target_col = 'math score')
plt.rcParams['figure.figsize'] = (18, 6)

plt.style.use('fivethirtyeight')

dabl.plot(data, target_col = 'reading score')
plt.rcParams['figure.figsize'] = (18, 6)

plt.style.use('fivethirtyeight')

dabl.plot(data, target_col = 'writing score')
total = data.shape[0]

math_greater_fifty = 0

for i in data['math score']:

    if i > 50:

        math_greater_fifty += 1

probability_math_greater_fifty = math_greater_fifty/total

print(probability_math_greater_fifty)
total = data.shape[0]

reading_greater_fifty = 0

for i in data['reading score']:

    if i > 50:

        reading_greater_fifty += 1

probability_reading_greater_fifty = reading_greater_fifty/total

print(probability_reading_greater_fifty)
total = data.shape[0]

writing_greater_fifty = 0

for i in data['writing score']:

    if i > 50:

        writing_greater_fifty += 1

probability_writing_greater_fifty = writing_greater_fifty/total

print(probability_writing_greater_fifty)
total_students = data.shape[0]

number_of_students_passing_in_all_subjects = data[(data['math score'] > 40) &

                                                  (data['writing score'] > 40) & 

                                                  (data['reading score'] > 40)].shape[0]

probability_of_students_passing_in_all_the_subjects = (number_of_students_passing_in_all_subjects/total_students)*100

print("The Probability of Students Passing in all the Subjects is {0:.2f} %".format(probability_of_students_passing_in_all_the_subjects))
plt.subplot(1, 3, 1)

sns.distplot(data['math score'])



plt.subplot(1, 3, 2)

sns.distplot(data['reading score'])



plt.subplot(1, 3, 3)

sns.distplot(data['writing score'])



plt.suptitle('Checking for Skewness', fontsize = 18)

plt.show()
data[(data['gender'] == 'female') &

     (data['math score'] > 90) & 

     (data['writing score'] > 90) &

     (data['reading score'] > 90)]
data['total_score'] = data['math score'] + data['reading score'] + data['writing score']
from math import * 

import warnings

warnings.filterwarnings('ignore')



data['percentage'] = data['total_score']/3



for i in range(0, 1000):

    data['percentage'][i] = ceil(data['percentage'][i])



plt.rcParams['figure.figsize'] = (15, 9)

sns.distplot(data['percentage'], color = 'orange')



plt.title('Comparison of percentage scored by all the students', fontweight = 30, fontsize = 20)

plt.xlabel('Percentage scored')

plt.ylabel('Count')

plt.show()
def getgrade(percentage):

  

  if(percentage >= 90):

    return 'O'

  if(percentage >= 80):

    return 'A'

  if(percentage >= 70):

    return 'B'

  if(percentage >= 60):

    return 'C'

  if(percentage >= 40):

    return 'D'

  else :

    return 'E'



data['grades'] = data.apply(lambda x: getgrade(x['percentage']), axis = 1 )



data['grades'].value_counts()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['lunch'] = le.fit_transform(data['lunch'])

data['test preparation course'] = le.fit_transform(data['test preparation course'])

# we have to map values to each of the categories

data['race/ethnicity'] = data['race/ethnicity'].replace('group A', 1)

data['race/ethnicity'] = data['race/ethnicity'].replace('group B', 2)

data['race/ethnicity'] = data['race/ethnicity'].replace('group C', 3)

data['race/ethnicity'] = data['race/ethnicity'].replace('group D', 4)

data['race/ethnicity'] = data['race/ethnicity'].replace('group E', 5)



data['parental level of education'] = le.fit_transform(data['parental level of education'])

data['gender'] = le.fit_transform(data['gender'])
data.shape
data
x = data.iloc[:,:10]

y = data.iloc[:,10]
x
y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

x_train = mm.fit_transform(x_train)

x_test = mm.transform(x_test)
from sklearn.linear_model import LogisticRegression

model =  LogisticRegression()

model.fit(x_train,y_train)



y_pred = model.predict(x_test)

# calculating the classification accuracies

print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy :", model.score(x_test, y_test))
y_pred.shape
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,y_test)



plt.rcParams['figure.figsize'] = (8, 8)

sns.heatmap(cm, annot = True, cmap = 'Greens')

plt.title('Confusion Matrix for Logistic Regression', fontweight = 30, fontsize = 20)

plt.show()
192/250
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()



model.fit(x_train,y_train)



y_pred = model.predict(x_test)



print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy :", model.score(x_test, y_test))
from sklearn.metrics import confusion_matrix



# creating a confusion matrix

cm = confusion_matrix(y_test, y_pred)



# printing the confusion matrix

plt.rcParams['figure.figsize'] = (8, 8)

sns.heatmap(cm, annot = True, cmap = 'Reds')

plt.title('Confusion Matrix for Random Forest', fontweight = 30, fontsize = 20)

plt.show()
y_pred
import csv

with open('grades.csv', 'w', newline='') as csv_file:

        wr = csv.writer(csv_file)

        wr.writerows(y_pred)

        csv_file.close()