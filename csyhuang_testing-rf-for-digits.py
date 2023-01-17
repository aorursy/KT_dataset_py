# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

X_test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")

y_train = train['label']

X_train = train.drop(['label'],axis=1)



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



#output = forest.predict(test_data).astype(int)

df = pd.DataFrame(y_pred)

df.index+=1

df.index.name='ImageId'

df.columns=['Label']

df.to_csv('results.csv', header=True)    
#f = open('digit_RF2.csv', 'w')

#f.write("ImageID,Label\n")

#for ii,jj in zip(np.arange(output.shape[0]), output):

#    f.write(str(ii)+','+str(jj)+'\n')

#f.close()    
#import csv

#from sklearn.ensemble import RandomForestClassifier



#train_data = train.values

#test_data = test.values

#train_y_data = train_y.values



#forest = RandomForestClassifier(n_estimators=100)

#forest = forest.fit( train_data, train_y_data )

#output = forest.predict(test_data).astype(int)



#for i in range(10):

#    print(output[i])

    