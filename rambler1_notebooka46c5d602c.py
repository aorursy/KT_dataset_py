# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

from sklearn.ensemble import RandomForestClassifier as rf

from sklearn import preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

features=[]

classes=[]

with open('../input/train.csv','r') as mycsvfile:

    data = csv.reader(mycsvfile)

    ch=0

    for row in data:

        if ch==0:

            ch=1

            continue

        gend=0 if row[4]=='male' else 1

        features.append([row[2],gend,row[5],float(row[9])])

        classes.append(int(row[1]))

  

farray=np.array(features)

carray=np.array(classes)

trees=rf(n_estimators=50)

le=preprocessing.LabelEncoder()

for i in range(0,3):

    farray[:,i]=le.fit_transform(farray[:,i])

trees=trees.fit(farray,classes)

test=[]

cl2=[]

d=dict()

with open('../input/genderclassmodel.csv','r') as classfile:

    c1=csv.reader(classfile)

    ch=1

    for row in c1:

        if ch==1:

            ch=0

            continue

        d[row[0]]=row[1]



with open('../input/test.csv','r') as mytestfile:

    testdata=csv.reader(mytestfile)

    ch=0

    for row in testdata:

        if ch==0:

            ch=1

            continue

        if row[3]!='male' and row[3]!='female':

           

            continue

        gend=0 if row[3]=='male' else 1

        if row[1]!='1' and row[1]!='2' and row[1]!='3':

           

            continue

        if row[4] is None or row[4]=='':

            row[4]=0           

           

        if row[0] not in d:

  

            continue

        if row[8] is None or row[8]=='':

  

            continue

        test.append([row[1],gend,float(row[4]),row[8]])

        

        cl2.append(int(d[row[0]]))

tarray=np.array(test)

cl=np.array(cl2)



for i in range(0,3):

    tarray[:,i]=le.fit_transform(tarray[:,i])

trees.predict(tarray)

trees.score(tarray,cl)