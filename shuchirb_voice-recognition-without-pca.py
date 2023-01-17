# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from pandas import read_csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

voiceData =read_csv('../input/voice.csv')



print("Voice Data Set::")

print(voiceData) 



features=voiceData[["meanfreq",	"sd","median",	"Q25",	"Q75",	"IQR",	"skew",	"kurt",	"sp.ent",	"sfm",	"mode",	"centroid", "meanfun",	"minfun",	"maxfun",	"meandom",	"mindom",	"maxdom",	"dfrange",	"modindx"]]



targetVariable=voiceData.label



featureTrain, featureTest, targetTrain, targetTest=train_test_split(features,targetVariable, test_size=.2)



print("Voice Train Data Set::")

print(featureTrain)



print("Voice Test Data Set::")

print(featureTest)



model=DecisionTreeClassifier()

fittedModel=model.fit(featureTrain, targetTrain)

predictions=fittedModel.predict(featureTest)



print("Prediction::")

print(predictions)



print("Confusion Matrix::")

print(confusion_matrix(targetTest,predictions))

#array([[320,   9],

#       [  6, 299]], dtype=int64)

print("Accuracy::")

print(accuracy_score(targetTest,predictions)) # 0.97634069400630918