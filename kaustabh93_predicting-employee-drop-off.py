# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read the file

input_file = "../input/HR_comma_sep.csv"

df = pd.read_csv(input_file, header = 0)
#drop the sales column

df =df.drop('sales', axis=1)
#convert the salary column to numerical values

for i in range(len(df)):

    if df.iloc[i][8] == 'low':

        df.set_value(i,'salary',0)

    elif df.iloc[i][8] == 'medium':

       df.set_value(i,'salary',1)

    else:

        df.set_value(i,'salary',2)
#See the first 10 records

print(df[:10])
#extract the prediction labels

labels = df['left']

df =df.drop('left', axis=1)
#Split into training and test sets

X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)
#See 10 samples of each of the 4 sets obtained

print("Training Input")

print(X_train[:10])

print("\nTraining Labels")

print(y_train[:10])

print("\nTest Inputs")

print(X_test[:10])

print("\nTest Labels")

print(y_test[:10])
#Initialise the classifier

#I'd be using SVM



clf = svm.SVC()
#Train the SVM

clf.fit(X_train, y_train) 
#Predict Results

y_pred = clf.predict(X_test)
#Calculate the prediction accuracy

print(accuracy_score(y_test, y_pred))