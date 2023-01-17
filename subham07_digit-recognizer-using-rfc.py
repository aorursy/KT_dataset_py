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

dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')



target = dataset['label'];

inputs=dataset.drop(['label'], axis='columns')



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.2, random_state = 0) #test on 2000 observations and train on 8000



#Training the Model



from sklearn.ensemble import RandomForestClassifier



model=RandomForestClassifier(n_estimators=100);

print(model)

model.fit(X_train,y_train);







model.score(X_test,y_test)

#importing the test dataset and convert into numpy array

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



test_ip=test;

tip=test_ip.values



#Calculating Final Predictions

ans=[]



for i in range(28000):

    x=np.array([tip[i]]);

    res=model.predict(x);

    res=res[0];

    ans1=[];

    ans1.append(i+1);

    ans1.append(res);

    ans.append(ans1);



ans=np.array(ans);

submission = pd.DataFrame({"ImageId": ans[:,0], 

                           "Label": ans[:,1]})

print(submission.head())



submission.to_csv("submission.csv",index=False)