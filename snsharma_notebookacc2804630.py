# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import csv

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from sklearn.preprocessing import OneHotEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# PREPARING TRAINING DATA

# Features (survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked)

# Embarked 1 C = Cherbourg

#          2 Q = Queenstown

#          3 S = Southampton

# Sex 0 = Male

#     1 = Female

# For normalization converting each value between o-1

# For missing data: 

# a. Taking the avg in case of Age and Fare

# b. Taking sex as female if survived or else male

# c. Taking embarked as 3

# d. Taking survived as 0

# e. Taking Pclass as 3

# f. Taking SibSp and Parch as 0



def prepare_train_data():

    with open("../input/train.csv") as f:

        reader = csv.reader(f)

        rows=[]

        for row in reader:

            if row[0] == 'PassengerId':

                continue # rejecting header row

            rows.append([row[1], row[2], row[4], row[5], row[6], row[7], row[9], row[11]])

        # print(rows[0:5])

        # correcting data, str to int, fixing missing data

        for row in rows:

            # survived

            if row[0] != '':

                row[0] = int(row[0])

            else:

                row[0] = 0

            # Pclass

            if row[1] != '':

                row[1] = int(row[1])

            else:

                row[1] = 3

            # Sex

            if row[2].lower() == "female":

                row[2] = 1

            else:

                row[2] = 0

            # Age avg = 30

            if row[3] == '':

                row[3] = 30.0

            else:

                row[3] = float(row[3])

            # SibSp

            if row[4] == '':

                row[4] = 0

            else:

                row[4] = int(row[4])

            # Parch

            if row[5] == '':

                row[5] = 0

            else:

                row[5] = int(row[5])

            # Fare avg = 32.204

            if row[6] == '':

                row[6] = 32.204

            else:

                row[6] = float(row[4])

            # Embarked

            if row[7] == 'C':

                row[7] = 1

            elif row[7] == 'Q':

                row[7] = 2

            else:

                row[7] = 3

        # print(rows[0:5])

        # normalizing data and separating input: train_X and output: train_y

        train_X = []

        train_y = [row[0] for row in rows]

        train_Y = np_utils.to_categorical(train_y, 2)

        pClass = [row[1] for row in rows]

        sex = [row[2] for row in rows]

        age = [row[3] for row in rows]

        sibSp = [row[4] for row in rows]

        parch = [row[5] for row in rows]

        fare = [row[6] for row in rows]

        embarked = [row[7] for row in rows]



        min_val = float(min(pClass))

        max_val = float(max(pClass))

        pClass = [(float(x)-min_val)/(max_val-min_val) for x in pClass]



        min_val = float(min(age))

        max_val = float(max(age))

        age = [(float(x)-min_val)/(max_val-min_val) for x in age]



        min_val = float(min(sibSp))

        max_val = float(max(sibSp))

        sibSp = [(float(x)-min_val)/(max_val-min_val) for x in sibSp]



        min_val = float(min(parch))

        max_val = float(max(parch))

        parch = [(float(x)-min_val)/(max_val-min_val) for x in parch]



        min_val = float(min(fare))

        max_val = float(max(fare))

        fare = [(float(x)-min_val)/(max_val-min_val) for x in fare]



        min_val = float(min(embarked))

        max_val = float(max(embarked))

        embarked = [(float(x)-min_val)/(max_val-min_val) for x in embarked]



        for (pc, s, a, sib, par, f, em) in zip(pClass, sex, age, sibSp, parch, fare, embarked):

            train_X.append([pc, s, a, sib, par, f, em])

        return (train_X, train_Y)

    

    

    
# PREPARING TEST DATA

def prepare_test_data():

    with open("../input/test.csv") as f:

        reader = csv.reader(f)

        rows=[]

        for row in reader:

            if row[0] == 'PassengerId':

                continue # rejecting header row

            rows.append([row[1], row[3], row[4], row[5], row[6], row[8], row[10]])

        # print(rows[0:5])

        # correcting data, str to int

        for row in rows:

            # Pclass

            if row[0] != '':

                row[0] = int(row[0])

            else:

                row[0] = 3

            # Sex

            if row[1].lower() == "female":

                row[1] = 1

            else:

                row[1] = 0

            # Age avg = 30

            if row[2] == '':

                row[2] = 30.0

            else:

                row[2] = float(row[2])

            # SibSp

            if row[3] == '':

                row[3] = 0

            else:

                row[3] = int(row[3])

            # Parch

            if row[4] == '':

                row[4] = 0

            else:

                row[4] = int(row[4])

            # Fare avg = 32.204

            if row[5] == '':

                row[5] = 32.204

            else:

                row[5] = float(row[5])

            # Embarked

            if row[6] == 'C':

                row[6] = 1

            elif row[6] == 'Q':

                row[6] = 2

            else:

                row[6] = 3

        # print(rows[0:5])

        # normalizing data and separating input: train_X and output: train_y

        test_X = []

        pClass = [row[0] for row in rows]

        sex = [row[1] for row in rows]

        age = [row[2] for row in rows]

        sibSp = [row[3] for row in rows]

        parch = [row[4] for row in rows]

        fare = [row[5] for row in rows]

        embarked = [row[6] for row in rows]



        min_val = float(min(pClass))

        max_val = float(max(pClass))

        pClass = [(float(x)-min_val)/(max_val-min_val) for x in pClass]



        min_val = float(min(age))

        max_val = float(max(age))

        age = [(float(x)-min_val)/(max_val-min_val) for x in age]



        min_val = float(min(sibSp))

        max_val = float(max(sibSp))

        sibSp = [(float(x)-min_val)/(max_val-min_val) for x in sibSp]



        min_val = float(min(parch))

        max_val = float(max(parch))

        parch = [(float(x)-min_val)/(max_val-min_val) for x in parch]



        min_val = float(min(fare))

        max_val = float(max(fare))

        fare = [(float(x)-min_val)/(max_val-min_val) for x in fare]



        min_val = float(min(embarked))

        max_val = float(max(embarked))

        embarked = [(float(x)-min_val)/(max_val-min_val) for x in embarked]



        for (pc, s, a, sib, par, f, em) in zip(pClass, sex, age, sibSp, parch, fare, embarked):

            test_X.append([pc, s, a, sib, par, f, em])

        

        return test_X
