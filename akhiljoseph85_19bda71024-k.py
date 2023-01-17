#import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#reading train data
data=pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")
data.head()
#data information
data.info()
LABELS=["tensed","loose"]
data.isnull().sum()
sns.set_style("whitegrid")
sns.countplot(x="flag",data=data)
tensed = data[data['flag']==1]

loose = data[data['flag']==0]
print(tensed.shape,loose.shape)
tensed.timeindex.describe()
loose.timeindex.describe()
sns.heatmap(data.corr())
data.head()
sns.boxplot(data["refVelocityBack"])
data["refPositionBack"]=np.sqrt(data["refPositionBack"])
data["positionBack"]=np.sqrt(data["positionBack"])
data["refVelocityBack"]=np.sqrt(data["refVelocityBack"])
data["trackingDeviationBack"]=np.sqrt(data["trackingDeviationBack"])
data["velocityBack"]=np.sqrt(data["velocityBack"])
data["positionFront"]=np.sqrt(data["positionFront"])
data["refPositionFront"]=np.sqrt(data["refPositionFront"])
data["refVelocityFront"]=np.sqrt(data["refVelocityFront"])
data["trackingDeviationFront"]=np.sqrt(data["trackingDeviationFront"])
data["velocityFront"]=np.sqrt(data["velocityFront"])

data.head()
from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
F=data.drop(['timeindex','flag'], axis = 1) 
F=scaling.fit_transform(F)
F
train_n=pd.DataFrame(F)
train_n.head()
train_n
d=data.iloc[:, 0:2] 
d

train= pd.concat([d,train_n],axis=1)
train
#correlation plot
sns.heatmap(train.corr())
#Distribution graphs
import itertools

col = train.columns[:12]
plt.subplots(figsize = (20, 15))
length = len(col)

for i, j in itertools.zip_longest(col, range(length)):
    plt.subplot((length/2), 3, j + 1)
    plt.subplots_adjust(wspace = 0.1,hspace = 0.5)
    train[i].hist(bins = 20)
    plt.title(i)
plt.show()
#correlation plot
ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(train.corr(), annot = True)
plt.show()
#dropping deependent variable from training dataset
target=train.flag
inputs=train.drop('flag',axis='columns')
#reading test data
df=pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")
df
df.info()
#missing values
df.isnull().sum()
df
df.head()
from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
M=df.drop(['timeindex'], axis = 1) 
M=scaling.fit_transform(M)
text_n=pd.DataFrame(M)
text_n.head()
text_n
T=df.iloc[:, 0:1] 
T
test= pd.concat([T,text_n],axis=1)
test
#Distribution graphs
import itertools

col = test.columns[:12]
plt.subplots(figsize = (20, 15))
length = len(col)

for i, j in itertools.zip_longest(col, range(length)):
    plt.subplot((length/2), 3, j + 1)
    plt.subplots_adjust(wspace = 0.1,hspace = 0.5)
    test[i].hist(bins = 20)
    plt.title(i)
plt.show()
test.isnull().sum()
#correlation plot
ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(test.corr(), annot = True)
plt.show()
test.isnull().sum()
test.info()
#splitting train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.3,)
#model=randomforestclassifier
from sklearn.ensemble import RandomForestClassifier


w = 50 # The weight for the positive class

rand = RandomForestClassifier(class_weight={0: 1, 1: w})
rand.fit(x_train, y_train)
rand.score(x_train,y_train)*100
y_predi = rand.predict(x_test)
y_predi
from sklearn.metrics import f1_score
#F1score
f1_score(y_test,y_predi)
#confusion matrix
from sklearn.metrics import confusion_matrix
accuracy=confusion_matrix(y_test,y_predi)
accuracy
#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predi))
predicted=rand.predict(test)
predicted
sample=pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
sample["flag"]=predicted
import numpy as np
sample.to_csv("sub.csv",index=False)



