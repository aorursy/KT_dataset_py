# -*- coding: utf-8 -*-

"""

Created on Tue Jul 18 23:49:32 2017



@author: Mahsa Hassankashi

"""



import numpy as np

import csv as csv

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.cross_validation import StratifiedKFold # Add important libs



train=[]

test=[]     

gender_submission=[] #Array Definition



#Enter your path in the below 

path1 =  r'D:\titanic\train.csv'    #Address Definition

path2 =  r'D:\titanic\test.csv'



with open(path1, 'r') as f1:    #Open File as read by 'r'

    reader = csv.reader(f1)     

    next(reader, None)          #Skip header because file header is not needed

    for row in reader:          #fill array by file info by for loop

        train.append(row)

    train = np.array(train)       	

	

with open(path2, 'r') as f2:

    reader2 = csv.reader(f2)

    next(reader2, None)  

    for row2 in  reader2:

        test.append(row2)

    test = np.array(test)

    

train = np.delete(train,[0],1)  #delete first column of which is passengerid

test = np.delete(test,[0],1)





train[train[0::,3] == 'male', 3] = 1         #replacement gender with number

train[train[0::,3] == 'female', 3] = 0

test[test[0::,2] == 'male',2] = 1

test[test[0::,2] == 'female',2] = 0



for row in  train:  

        Title = row[2].split(',')[1].split('.')[0].strip() #Extracting Name in order to gain title

        row[2] = Title





for row in train:           #Fill empty cell or age column by the below logic

	if (row[4]==''):

		if (row[1]=='1' and row[2]=='Miss' and row[3]=='0'):

			row[4] =30

		if (row[1]=='1' and row[2]=='Mrs' and row[3]=='0'):

			row[4] =40

		if (row[1]=='1' and row[2]=='Officer' and row[3]=='0'):

			row[4] =49

		if (row[1]=='1' and row[2]=='Officer' and row[3]=='0'):

			row[4] =40.5

		if (row[1]=='2' and row[2]=='Miss' and row[3]=='0'):

			row[4] =24

		if (row[1]=='2' and row[2]=='Mrs' and row[3]=='0'):

			row[4] =31.5

		if (row[1]=='3' and row[2]=='Miss' and row[3]=='0'):

			row[4] =18

		if (row[1]=='3' and row[2]=='Mrs' and row[3]=='0'):

			row[4] =31

		if (row[1]=='1' and row[2]=='Master' and row[3]=='1'):

			row[4] =4

		if (row[1]=='1' and row[2]=='Mr' and row[3]=='1'):

			row[4] =40

		if (row[1]=='1' and row[2]=='Officer' and row[3]=='1'):

			row[4] =51

		if (row[1]=='1' and row[2]=='Royalty' and row[3]=='1'):

			row[4] =40

		if (row[1]=='1' and row[2]=='Dr' and row[3]=='1'):

			row[4] =40.4

		if (row[1]=='2' and row[2]=='Master' and row[3]=='1'):

			row[4] =1

		if (row[1]=='2' and row[2]=='Mr' and row[3]=='1'):

			row[4] =31

		if (row[1]=='2' and row[2]=='Officer' and row[3]=='1'):

			row[4] =46.5

		if (row[1]=='3' and row[2]=='Master' and row[3]=='1'):

			row[4] =4

		if (row[1]=='3' and row[2]=='Mr' and row[3]=='1'):

			row[4] =26



			

for row in  test:

    Title = row[1].split(',')[1].split('.')[0].strip()

    row[1] = Title





for row in test:

	if (row[3]==''):

		if (row[0]=='1' and row[1]=='Miss' and row[2]=='0'):

			row[3] =32

		if (row[0]=='1' and row[1]=='Mrs' and row[2]=='0'):

			row[3] =48

		if (row[0]=='1' and row[1]=='Royalty' and row[2]=='0'):

			row[3] =39

		if (row[0]=='2' and row[1]=='Miss' and row[2]=='0'):

			row[3] =19.5

		if (row[0]=='2' and row[1]=='Mrs' and row[2]=='0'):

			row[3] =29

		if (row[0]=='3' and row[1]=='Miss' and row[2]=='0'):

			row[3] =22

		if (row[0]=='3' and row[1]=='Mrs' and row[2]=='0'):

			row[3] =28

		if (row[0]=='3' and row[1]=='Ms' and row[2]=='0'):

			row[3] =22

		if (row[0]=='1' and row[1]=='Master' and row[2]=='1'):

			row[3] =9.5

		if (row[0]=='1' and row[1]=='Mr' and row[2]=='1'):

			row[3] =42

		if (row[0]=='1' and row[1]=='Officer' and row[2]=='1'):

			row[3] =53

		if (row[0]=='2' and row[1]=='Master' and row[2]=='1'):

			row[3] =5

		if (row[0]=='2' and row[1]=='Mr' and row[2]=='1'):

			row[3] =28

		if (row[0]=='2' and row[1]=='Officer' and row[2]=='1'):

			row[3] =35.5

		if (row[0]=='3' and row[1]=='Master' and row[2]=='1'):

			row[3] =7

		if (row[0]=='3' and row[1]=='Mr' and row[2]=='1'):

			row[3] =25



			

train[train[0::,10] == 'C', 10] = 0

train[train[0::,10] == 'S', 10] = 1

train[train[0::,10] == 'Q', 10] = 2

train[train[0::,10] == '',10] = np.median(train[train[0::,10]!= '',10].astype(np.int))

train[train[0::,8] == '',8] = np.round(np.mean(train[train[0::,8]!= '',8].astype(np.float)))

train = np.delete(train,[2,7,9],1)

test[test[0::,9] == 'C', 9] = 0

test[test[0::,9] == 'C', 9] = 0

test[test[0::,9] == 'S', 9] = 1

test[test[0::,9] == 'Q', 9] = 2

test[test[0::,9] == '',9] = np.median(test[test[0::,9]!= '',9].astype(np.int))

for i in range(np.size(test[0::,0])):

    if test[i,7] == '':

        test[i,7] = np.median(test[(test[0::,7] != '') & (test[0::,0] == test[i,0]),7].astype(np.float))



	

test = np.delete(test,[1,6,8],1)  #Delete name column because I did not use it here





parameter_gridsearch = {

                 'max_depth' : [3, 4],  #depth of each decision tree

                 'n_estimators': [50, 20],  #count of decision tree

                 'max_features': ['sqrt', 'auto', 'log2'],      

                 'min_samples_split': [2],      

                 'min_samples_leaf': [1, 3, 4],

                 'bootstrap': [True, False],

                 }



randomforest = RandomForestClassifier()

crossvalidation = StratifiedKFold(train[0::,0] , n_folds=5)

    

gridsearch = GridSearchCV(randomforest,             #grid search for algorithm optimization

                                   scoring='accuracy',

                                   param_grid=parameter_gridsearch,

                                   cv=crossvalidation)

    

    

gridsearch.fit(train[0::,1::], train[0::,0])    #train[0::,0] is as target

model = gridsearch

parameters = gridsearch.best_params_



print('Best Score: {}'.format(gridsearch.best_score_))



path3 =  r'D:\titanic\gender_submission.csv'

output = gridsearch.predict(test)



with open(path3, 'w',  newline='') as f3, open(path2, 'r') as f4: # write output and other column from test

    forest_Csv = csv.writer(f3)

    forest_Csv.writerow(["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"])

    test_file_object = csv.reader(f4)

    next(test_file_object, None)

    i = 0

    for row in  test_file_object:

        row.insert(1,output[i].astype(np.uint8))

        forest_Csv.writerow(row)

        i += 1

    