# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read dataset
df = pd.read_csv('../input/xAPI-Edu-Data.csv')

#print number of features (columns)
print(df.shape[1])

#print number of records (rows)
print(df.shape[0])

#print first 5 rows
df.head()
#print counts of unique values
print(df.gender.value_counts())

df.gender.replace(['M','F'], [1,0] , inplace = True)
df.gender
#educational level student belongs
df.StageID.replace(['HighSchool','lowerlevel','MiddleSchool'],[1,2,3], inplace = True)


#Section ID- classroom student belongs (nominal:’A’,’B’,’C’)
df.SectionID.replace(['A','B','C'], [1,2,3], inplace = True)


#  Semester- school year semester (nominal:’ First’,’ Second’)

df.Semester.replace(['S','F'], [1,2], inplace = True)


#  Parent responsible for student (nominal:’mom’,’father’)


df.Relation.replace(['Mum','Father'],[1,2], inplace = True)


# Parent Answering Survey- parent answered the surveys which are provided from school or not (nominal:’Yes’,’No’)

df.ParentAnsweringSurvey.replace(['Yes','No'], [0,1], inplace = True)

# Parent School Satisfaction- the Degree of parent satisfaction from school(nominal:’Yes’,’No’)
df.ParentschoolSatisfaction.replace(['Bad','Good'], [0,1], inplace = True)

#Student Absence Days-the number of absence days for each student (nominal: above-7, under-7)
df.StudentAbsenceDays.replace(['Under-7','Above-7'], [0,1], inplace = True)


#Class - student level
# Low-Level: interval includes values from 0 to 69,
# Middle-Level: interval includes values from 70 to 89,
# High-Level: interval includes values from 90-100.


df.Class.replace(['L','M','H'],[0,1,2], inplace = True)


df.PlaceofBirth.replace(['KuwaIT','lebanon','Egypt','SaudiArabia','USA','Jordan','venzuela',
                         'Iran','Tunis','Morocco','Syria','Palestine','Iraq','Lybia'],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                         inplace = True)



df.Topic.replace(['English','Spanish','French','Arabic','IT','Math','Chemistry',
                  'Biology','Science', 'History','Quran','Geology'],
                 [1,2,3,4,5,6,7,8,9,10,11,12], inplace = True)



df.drop(['StageID','Topic','NationalITy', 'SectionID',  'Semester', 'GradeID', 'Relation', 'ParentAnsweringSurvey'], axis = 1 , inplace = True)
df.head()
df.Class.value_counts()
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

x = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_neighbors=19)
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

# Sample Prediction 

example_measures = np.array([1,5,40,50,12,50,0,0])
example_measures = example_measures.reshape(1, -1)

prediction = clf.predict(example_measures)
print(prediction)





