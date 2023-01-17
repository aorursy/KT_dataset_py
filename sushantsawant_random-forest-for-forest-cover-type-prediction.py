# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train=pd.read_csv("../input/train.csv")
# non categorical features
features = ['Slope', 'Aspect',  'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Hydrology','Cover_Type'] # reducing features
selectedPoints  = 1000 #Number of sample set
df = train.ix[:selectedPoints, features]#indices
# 1 Pair plot
pairPlot=sns.pairplot(df)

#1 inference cover type 2 is in maximum in this dataset
#2 aspect is greator when near to hydrology
# 2 heatmap
plt.clf()
print(len(train.index))
dfAll = train.ix[:len(train.index), features]
corrmat = dfAll.corr()
heatMap=sns.heatmap(corrmat, vmax=.8, square=True)
heatMap.figure.savefig("heatMap.png")
#distance to hydrology and cover type is negatively co related 
#and quite intuitive Far from water, lesser vegitation

#3 Categorical Data Exploration
plt.clf()
categoricalFeatures = ["Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4","Cover_Type"]
train=pd.read_csv("../input/train.csv")
#dfAll = train[categoricalFeatures]
#print(dfAll.head())
#categoricalDf = train.ix[:selectedPoints,categoricalFeatures].copy()#indices
#for feature in categoricalDf:
 #   print(feature)
    #wildernessAreaPlot=sns.stripplot(x=feature, y="Cover_Type", data=categoricalDf, jitter=True);
    #wildernessAreaPlot.figure.savefig(feature+".png")

#Tried for loop will be fixing later
wildernessArea1Plot=sns.stripplot(x="Wilderness_Area1", y="Cover_Type", data=train, jitter=True);
wildernessArea1Plot.figure.savefig("wildernessArea1Plot.png")


#wilderness area 1 is type 1,2 and 5.

wildernessArea2Plot=sns.stripplot(x="Wilderness_Area2", y="Cover_Type", data=train, jitter=True);
wildernessArea2Plot.figure.savefig("wildernessArea2Plot.png")

wildernessArea3Plot=sns.stripplot(x="Wilderness_Area3", y="Cover_Type", data=train, jitter=True);
wildernessArea3Plot.figure.savefig("wildernessArea3Plot.png")
wildernessArea4Plot=sns.stripplot(x="Wilderness_Area4", y="Cover_Type", data=train, jitter=True);
wildernessArea4Plot.figure.savefig("wildernessArea4Plot.png")
#4 scatter plot for categorical data
# similar code for all wilderness and soil type
plt.clf()
scatterPlot=sns.swarmplot(x="Wilderness_Area1", y="Cover_Type", data=train)
scatterPlot.figure.savefig("scatterPlot.png")

#barplot
plt.clf()
barPlot=sns.barplot(x="Wilderness_Area1", y="Cover_Type", data=train)
barPlot.figure.savefig("barPlot.png")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

id=test['Id']
y=train['Cover_Type']

train=train.drop(['Id','Cover_Type'],1)
test=test.drop(['Id'],1)

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=44)

rfClassifier=RandomForestClassifier(n_estimators=16,class_weight='balanced',n_jobs=4,random_state=44,criterion='gini')
rfClassifier.fit(x_train,y_train)

accuracy=rfClassifier.score(x_test,y_test)
print(accuracy)
rfClassifier.fit(train,y)

prediction=rfClassifier.predict(test)
print(prediction)

output=pd.DataFrame(id)
output['Cover_Type']=prediction
print(output.head())
output.to_csv("output.csv",index=False)
