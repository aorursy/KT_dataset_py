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
# Pandas for data read in 

import pandas as pd

from pandas import Series, DataFrame



# numpy, matplotlib, seaborn for data analysis

import numpy as np



# ML 

from sklearn.svm import SVC

DF_Titanic = pd.read_csv("../input/train.csv")

DF_Test = pd.read_csv("../input/test.csv")
DF_Titanic = DF_Titanic.drop(["Name","Ticket","Embarked"],axis=1)

DF_Test = DF_Test.drop(["Name","Ticket","Embarked"],axis=1)

DF_Features = DF_Titanic.drop(["PassengerId","Cabin"],axis = 1)

DF_Features.Sex.replace(['male', 'female'],[1,0], inplace = True)

DF_Features.Fare.replace('NaN',0,inplace = True)

DF_Identi = DF_Features.Survived

DF_Features_no_Iden = DF_Features.drop(["Survived"],axis = 1)

DF_Features_no_Iden['Sex'].value_counts()

DF_Features_no_Iden.Age.replace('nan',0,inplace = True)
np_samp = np.array(DF_Features_no_Iden)

np_Iden = np.array(DF_Identi)

TT_svm = SVC()

TT_svm.fit(np_samp,np_Iden)
DF_Test_Result = DF_Test.PassengerId

DF_Test = DF_Test.drop(["PassengerId","Cabin"],axis=1)

DF_Test.Sex.replace(['male','female'],[1,0],inplace = 1)

DF_Test.Age.replace('nan',0,inplace = True)

DF_Test.Fare.replace('nan',0,inplace = True)

DF_Test.head()
survive_prediction = TT_svm.predict(DF_Test)

print(survive_prediction)
sub = pd.DataFrame({"PassengerId":DF_Test_Result, "Survived": survive_prediction})

sub.to_csv('titanic.csv', index = False)
