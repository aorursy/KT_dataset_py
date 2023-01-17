# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn import svm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
File_object = open("../input/test.csv","r")

co =0

for line in File_object:

    if(co>4):

        break

    print(line)

    co =co+1

#print(File_object.read(([3])))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.columns)

#File_object.read()



#print(File_object["var_0"][:6])

#plt.plot([1,2,3,4,5,6],File_object["var_0"][:6],'ro')

#plt.show()


print(train["var_0"][:6])

plt.plot([1,2,3,4,5,6],train["var_0"][:6],'ro')

#tra =np.array(train)

plt.show()

train.describe()

fig =plt.figure()

count =1

for i in train.columns[1:]:

    fig.add_subplot(700,4000,count)

    plt.title(count)

    count =count+1

    plt.hist(train[i],25)

    plt.show()

    


#p =math.log(train["var_0"])

plt.hist(train["var_0"],50,alpha =0.75,normed =1)



#print(train.columns([1,2]))

y_train =np.array([train["target"]])

#print(X_train)



#print(y_train)

y_train =y_train.reshape(-1)

#y_train = np.array(Y_train.iloc[0:,[0,1]])

print(y_train)

X_train =train.iloc[0:,[i for i in range (2,(len(train.columns)))]]

X_test =test.iloc[0:,[i for i in range (1,(len(test.columns)))]]

#X_train =train.drop(["ID_code"],axis =1)

print(X_train)

print("*********************")

print(np.array(X_train[2:3]))



counter =0

c=0

for i in y_train:

    if i ==0:

        counter =counter+1

    else:

        c =c+1



print("no. of zeroes in the data set",counter,"percentage",counter/len(train)*100)

print("no. of ones in the data set",c,"percentage",c/len(train)*100)

#X_train =np.array(X_train)

print(X_train)

#X_train =X_train.drop(X_train.columns)

#X_train =X_train.iloc[2:,[i for i in range (0,len(X_train.columns))]]

#print(X_train)

x_train =[]

x_test =[]

for i in range(len(X_train)):

    #print(i)

    x_train.append(np.array(X_train[i:i+1]))

#print(x_train)

for i in range(len(X_test)):

    #print(i)

    x_test.append(np.array(X_test[i:i+1]))

#print(x_train)

print(np.shape(x_train))

x_train =np.array(x_train)

x_test =np.array(x_test)

print(np.shape(x_train))

x_train =x_train.reshape((len(X_train),200))

x_test =x_test.reshape((len(X_test),200))

print(np.shape(x_train))

print(np.shape(y_train))
#X_train = preprocessing.scale(X_train)

print(np.shape(x_train))

scaler = preprocessing.MinMaxScaler()



# Fit your data on the scaler object

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)

#x_train = pd.DataFrame(x_train)

print(np.shape(x_train))

#print(x_train)

#X_train = preprocessing.MinMaxScaler(x_train)

#print(X_train)

#print(np.size(X_train))

clf =RandomForestClassifier(n_estimators=5,max_features =100)

#clf =svm.SVC(kernel='linear')

x_train =np.around(x_train)

x_test =np.around(x_test)

X = pd.DataFrame(x_train)

Y = pd.DataFrame(y_train)



x_test =pd.DataFrame(x_test)

print(X)

print(x_test)

print(x_train.shape)

print(y_train.shape)

print(Y.shape)

#Y =Y.reshape(-1)

#y_train =y_train.reshape((-1,1))

#clf.fit(x_train,y_train)

#X_test = test.drop(["ID_code"][:,6],axis =1)

#X = preprocessing.scale(X_test)

#ans =[]

#ans =clf.predict(x_test)

#pre =np.array([clf.predict(X_test[:,6])])

#pre = pre.reshape(-1)

#print(np.shape(pre))

#print(pre)

#np.savetxt('answer.csv',ans)
print(1)

q =x_train[0:200000]

w =y_train[0:200000]

clf.fit(q,w)

#clf.fit(q,w)

#X_test = test.drop(["ID_code"][:,6],axis =1)

#X = preprocessing.scale(X_test)

print(1)



accuracy =clf.score(q,w)

ans =[]

print(accuracy)

#ans =clf.predict(x_test)

#print(ans)



#pre =np.array([clf.predict(X_test[:,6])])

#pre = pre.reshape(-1)

#print(np.shape(pre))

#print(pre)

#np.savetxt('answer.csv',ans)
import pickle

saved_model = pickle.dumps(clf) 

#clf.save("mymodel")

model =pickle.loads(saved_model)

ans =model.predict(x_test)

print(ans)



ANS =pd.DataFrame(ans)

print(ANS)

co =0

cv =0

for i in ans:

    if i ==1:

        co =co+1

    else:

        cv =cv+1

print(cv,co)

submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = ANS

submission.to_csv('lgb_starter_submission.csv', index=False)

print(submission)



"""

df1 =pd.read_csv('../input/sample_submission.csv')

df1.drop(["target"],axis=1,inplace=True)

print(df1)

#ANS.columns="target"

df1 =df1.join(ANS)

print(df1)

np.savetxt('answer.csv',df1)



#os.path.real_path(ans.csv)

"""


