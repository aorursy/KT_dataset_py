import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
train=pd.read_csv("../input/human-activity-recognition-with-smartphones/train.csv")
train.head()
test=pd.read_csv("../input/human-activity-recognition-with-smartphones/test.csv")
test.head()
train.shape                                   # checking the number of rows and columns
test.shape                                     # checking the number of rows and columns
train.isnull().sum().any()                     # checking the number of missing values
test.isnull().sum().any()                       # checking the number of missing values


#Observation: There are no missing values in train and test data
train.dtypes.value_counts()                      # checking the datatype of the columns
test.dtypes.value_counts()                          # checking the datatype of the columns
train['Activity'].value_counts()                  # Checking the number of records for each labels


#How active are the participants
# we are going to count how many sensor measurments are there for each activity for each partcipant



pivoted=train.pivot_table(index="subject",columns="Activity",aggfunc='count').iloc[:,:6]

count_df=pd.DataFrame(pivoted.to_records())        #convert pivot table to dataframe

count_df=count_df.set_index ("subject")         # Change the index

count_df.columns =["LAYING","SITTING","STANDING","WALKING","WALKING_DOWNSTAIRS","WALKING_UPSTAIRS"]

count_df
# SAMPLING IS DONE AT 1.28 SECONDS,SO THE TOTAL DURATION FOR ALL THE ACTIONS WILL BE COUNT 8 1.28

# THIS IS A SAMPLE ,WHICH IS WHY THE DURATION IS QUIET LESS.IF A PERSON WEARS THE TRACKER FOR THE ACTIVITIES

# YOU CAN MEASURE HOW MANY HOURS HE WALKS,SLEEPS,SITS..ETC..



duration_df=count_df * 1.28

duration_df
# Feature Engineering : Creating new columns,active and passive

duration_df['active']= duration_df['WALKING']+duration_df['WALKING_UPSTAIRS']+duration_df['WALKING_DOWNSTAIRS']

duration_df['passive']=duration_df['LAYING']+duration_df['SITTING']+duration_df['STANDING']
duration_df
plt.figure(figsize=(14,4))

sns.barplot(x=duration_df.index,y=duration_df.LAYING)


#Observation: Person 1 and 14 sleep very little

#The missing numbers in x-axis are the people who are present in test data
x=duration_df.index

plt.figure(figsize=(14,4))

ax=plt.subplot(111)

ax.bar(x-0.4, duration_df.active, width=0.4, color='b', align='center')

ax.bar(x, duration_df.passive, width=0.4, color='g', align='center')

plt.show()
#Observation: Person 19 and 21 are highly inactive
plt.figure(figsize=(14,4))

sns.barplot(x=duration_df.index,y=duration_df.WALKING)


#Observation: person 1 and 25 are walking a lot
x_train=train.drop(['Activity','subject'],axis=1)

y_train=train['Activity']
x_train.shape
y_train.shape
x_test=test.drop(['Activity','subject'],axis=1)

y_test=test['Activity']
# lets reduce the dimensions from 561 columns to 2 columns

from sklearn.decomposition import PCA

pca=PCA(n_components=2)

principalComponents=pca.fit_transform(x_train)

principalDf = pd.DataFrame(principalComponents,columns=['pc1','pc2'])
principalDf.head()
finalDf=pd.concat([principalDf,y_train],axis=1)
import seaborn as sns

plt.figure(figsize=(12,8))

sns.scatterplot(x='pc1',y='pc2',data=principalDf,hue=y_train)
#Observations: The resting and ative states are clearly separable here 

#We can differentiate walking,walking-upstairs and walking-downstairs. 

#There is not much differentiation between sitting,laying and standing
from sklearn .decomposition import PCA

pca = PCA(n_components=3)

principalComponents=pca.fit_transform(x_train)

principalDf=pd.DataFrame(principalComponents,columns=['pc1','pc2','pc3'])
principalDf.head()
finalDf=pd.concat([principalDf,y_train],axis=1)
import plotly.express as px

fig=px.scatter_3d(finalDf,x="pc1",y="pc2",z="pc3",color="Activity",color_discrete_map={"pc1": "pc2","pc3":"green"})

fig.show()


#Observations: Here we are able to see all the different activities using 3 principal components.
from sklearn.tree import DecisionTreeClassifier

dt_clf=DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)               #fit the model in train 
y_pred=dt_clf.predict(x_test)           #make predictions on test data
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_pred,y_test)
print(cm)
cr=classification_report(y_pred,y_test)

print(cr)
print(accuracy_score(y_pred,y_test))
#Human  activity  recognition  has  broad  applications  in medical research and human survey system. 

#In this project, we  designed  a  smartphone-based  recognition  system  that recognizes five human activities: walking, limping, jogging, going upstairs and going downstairs. 

#The activity data were trained and tested using  PCA , Decision Tree algorithm and Confusion Matrix . 

#The best  classification  rate  in our experiment  accuracy_score was 0.8646080760095012
