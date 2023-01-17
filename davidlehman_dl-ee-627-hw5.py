# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))







# Any results you write to the current directory are saved as output.
#!/usr/bin/env python



import numpy



dataDir='/Users/davidlehman/Documents/EE627/HW5/data_in_matrixForm/'

file_name_test=dataDir + 'testTrack_hierarchy.txt'

file_name_train=dataDir + 'trainIdx2_matrix.txt'

output_file= dataDir + 'output1.txt'



fTest= open('../input/testTrack_hierarchy.txt', 'r')

fTrain=open('/kaggle/input/trainIdx2_matrix.txt', 'r')

Trainline= fTrain.readline()

fOut = open('output1.txt', 'w')



trackID_vec=[0]*6

albumID_vec=[0]*6

artistID_vec=[0]*6

lastUserID=-1



user_rating_inTrain=numpy.zeros(shape=(6,3))



#testInt = 0



for line in fTest:

	arr_test=line.strip().split('|')

	userID= arr_test[0]

	trackID= arr_test[1]

	albumID= arr_test[2]

	artistID=arr_test[3]



	if userID!= lastUserID:

		ii=0

		user_rating_inTrain=numpy.zeros(shape=(6,3))



	trackID_vec[ii]=trackID

	albumID_vec[ii]=albumID

	artistID_vec[ii]=artistID

	ii=ii+1

	lastUserID=userID



	if ii==6 : 

		while (Trainline):

		# for Trainline in fTrain:

			arr_train = Trainline.strip().split('|')

			trainUserID=arr_train[0]

			trainItemID=arr_train[1]

			trainRating=arr_train[2]

			Trainline=fTrain.readline()		



			if trainUserID< userID:

				continue

			if trainUserID== userID:				

				for nn in range(0, 6):

					if trainItemID==albumID_vec[nn]:

						user_rating_inTrain[nn, 0]=trainRating

					if trainItemID==artistID_vec[nn]:

						user_rating_inTrain[nn, 1]=trainRating

				#testInt = testInt + 1



			if trainUserID> userID:

				for nn in range(0, 6):

					outStr=str(userID) + '|' + str(trackID_vec[nn])+ '|' + str(user_rating_inTrain[nn,0]) + '|' + str(user_rating_inTrain[nn, 1])

					fOut.write(outStr + '\n')

				break





#print(testInt)



fTest.close()

fTrain.close()



#In order to find 3 tracks with "1", and 3 with "0", you need to find the split preferences in order to get the extremes on both ends. Also if you want to guarantee

#a "1", then you can match a user testing data with the training data (for artist, album, genre... if they do in fact both exist).



#Also, the reason that there are so many 0's is because if you look at the "testTrack_hierarchy.txt" and "trainIdx2_matrix.txt" files, you will see that 

#None of the user ID match with one another. Therefore, the for loop in line 53 will never execute and user_rating_inTrain will never get populated.