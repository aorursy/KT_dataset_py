# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
studentInfo = pd.read_csv("../input/open-university-learning-analytics-dataset/anonymiseddata/studentInfo.csv") 

studentInfo.head()
# lets see all potential final results

studentInfo["final_result"].unique()
#create a new column to classify final results. classify studets with a pass or distinction as "1", the rest as "0"

studentInfo["result.class"] = 1



#studentInfo["result.class"] = studentInfo["final_result"].apply(lambda x: 0 if (x == 'Fail') | x == "Withdrawn") else 1)

studentInfo["result.class"].loc[(studentInfo["final_result"] == "Withdrawn") | (studentInfo["final_result"] == "Fail")] = 0



#and look at the dataset again

studentInfo.head()
Xfactors = studentInfo[["gender", "region", "highest_education", "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "disability"]]

X_noncat = pd.get_dummies(Xfactors)



X_noncat.head()
Youtcome = studentInfo["result.class"].values

Youtcome
#fit the model

X_train, X_test, y_train, y_test = train_test_split(X_noncat, Youtcome, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
#predict on a testset

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#check out the confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)