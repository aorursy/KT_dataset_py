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
#importing the dataset

dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

X = dataset.iloc[:, [2,4,5,6,7,9]].values 

y = dataset.iloc[:, 1].values







# Taking care of missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 2:3])

X[:, 2:3] = imputer.transform(X[:, 2:3])

print(X)

#pclass1, #pclass2, #pclass3, sex(1-male), age, sib, parch, fare



# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

#print(X[:,1]);

X[:,1]=labelencoder_X_1.fit_transform(X[:, 1])

#print(X[:,1]);



X[:,0]=labelencoder_X_1.fit_transform(X[:, 0])



print(X);





# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #test on 2000 observations and train on 8000



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test) #for test set we dont need to fit

#as y is just categorical no need for scaling, if range is large then need to perform scaling

#pre-processing completed



print(X_train);







#importing libraries

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout #Dropout to reduce overfitting (line 131)





#initializing the ANN

classifier= Sequential()



#running the input layer and the first hidden layer with dropout

classifier.add(Dense(output_dim=3, init='uniform', activation='relu', input_dim=6)) #average of no. of nodes in i/p and nodes in o/p

classifier.add(Dropout(rate=0.1));



#weight assigned uniform distribution

#relu corresponds to Rectifier funtion

#input_dim is no. of nodes in input layer

#increase rate if overfitting still there but do not go >0.5 because then there will be no neurons to learn



classifier.add(Dense(output_dim=3, init='uniform', activation='relu')) #2nd layer

classifier.add(Dropout(rate=0.1));



#classifier.add(Dense(output_dim=2, init='uniform', activation='relu')) #adding 3rd layer

#classifier.add(Dropout(rate=0.1));



#adding o/p Layer

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) #output node is 1 as binary outcome will exit the bank / wont exit

#sigmoid function for o/p , use softmax if o/p is >1



#compiling the ANN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # adam algorithm: adaptive moment estimation, to update weights iteratively

#adam algo comes under stochastic gradient descent

#binary crossentropy as binary outcome



# Fitting classifier to the Training set

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=300)
Y_pred = classifier.predict(X_test)

Y_final=[];

for i in range(len(Y_pred)):

    Y_final.append(list(Y_pred[i])[0]);

for i in range(len(Y_final)):

    if(Y_final[i]<0.5):

        Y_final[i]=0;

    else:

        Y_final[i]=1;

print(type(Y_final));

print(y_test)



accu=0;

corr=0;

for i in range(len(Y_final)):

    if(Y_final[i]==y_test[i]):

        corr+=1;

accu=corr/len(Y_final);

print(accu)

print(corr);
#importing the test dataset

test_dataset = pd.read_csv('/kaggle/input/titanic/test.csv')

Xt = test_dataset.iloc[:, [0,1,3,4,5,6,8]].values 

print(type(Xt));

for i in range(len(Xt)):

    print(Xt[i]);

    

    

# Taking care of missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(Xt[:, 3:4])

Xt[:, 3:4] = imputer.transform(Xt[:, 3:4])

new_prediction=[];

ans=[];



for i in range(len(Xt)):

    l=list(Xt[i]);

    nl=[];

    ans1=[]

    print(l);

    if(l[1]==3):

        nl.append(2);

    elif(l[1]==2):

        nl.append(1);

    elif(l[1]==1):

        nl.append(0);

        

    if(l[2]=='male'):

        nl.append(1);

    else:

        nl.append(0);

    nl.append(l[3]);

    nl.append(l[4]);

    nl.append(l[5]);

    nl.append(l[6]);

    new_prediction=classifier.predict(sc.transform(np.array([nl[0:6]])));

    print(new_prediction)

    if(new_prediction<0.5):

        new_prediction=0;

    else:

        new_prediction=1;

    ans1.append(l[0]);

    ans1.append(new_prediction);

    ans.append(ans1);

print(ans);

    



ans=np.array(ans);

print(ans);
submission = pd.DataFrame({"PassengerId": ans[:,0], 

                           "Survived": ans[:,1]})

print(submission.head())



submission.to_csv("submission.csv",index=False)