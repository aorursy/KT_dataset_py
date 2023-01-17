# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualization

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split 

from sklearn.metrics import confusion_matrix







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



hrdata = pd.read_csv('../input/HR_comma_sep.csv')

hrdata.describe()





# Any results you write to the current directory are saved as output.
hrdata.head(n=1)

#heatmap of correlation matrix

visual1=sns.heatmap(hrdata.corr())

 #examining correlation with 'left' column

hrdata.corr().left

#create dummy variables to handle categorical data

hrdatapre= pd.get_dummies(hrdata)


#split data set 

from sklearn.cross_validation import train_test_split

hrtrain, hrtest = train_test_split(hrdatapre, train_size = 0.8)



#Set up test and train data

train_y = hrtrain['left']

train_x = hrtrain.drop('left',axis=1)



ytrue= hrtest['left']

test_x = hrtest.drop('left',axis=1)



#RandomForest Classifier 

forest_hr = RandomForestClassifier()

forest_hr.fit(train_x, train_y)

testresults= forest_hr.predict(test_x)
#creating confusion matrix to examine results

confusion_matrix(ytrue, testresults)

forest_hr.score(test_x, ytrue)
