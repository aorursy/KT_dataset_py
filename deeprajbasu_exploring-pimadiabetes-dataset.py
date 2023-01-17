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

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

print(df.shape)

df.head()
print(df.isnull().sum())#lookin if any missing values are present 

print('*'*75)

      

df.info()#some other useful info, such as datatypes of colums shape of our dataset etc
df.describe()
df.describe().T

#creating the transpose of the same information 
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]= df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)



print(df.isnull().sum())
df.hist(figsize = (20,20))
#replacing all the missing glucose values with the mean 



df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)

#same thing with bloodpressure and BMI

df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)

df['BMI'].fillna(df['BMI'].median(), inplace = True)

df.isna().sum()
#using the median value for skin thickness and insulin 

df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)

df['Insulin'].fillna(df['Insulin'].median(), inplace = True)

df.isna().sum()
df.hist(figsize = (20,20))
df.dtypes # lets check the datatypes 
df.Outcome.value_counts().plot(kind="bar")

# The number of non-diabetics is almost twice the number of diabetic patients
#The above graph shows that the data is biased towards datapoints having outcome value as 0 where it means that diabetes was not present actually. 

#The number of non-diabetics is almost twice the number of diabetic patients
sns.pairplot(df, hue = 'Outcome',height=5)
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
df.corr()
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

sns.heatmap(df.corr(), annot=True,cmap ='BuGn_r')  # seaborn has very simple solution for heatmap
plt.figure(figsize=(22,6))

sns.boxplot(data=df, orient="h", palette="Set2")
# scaling the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



df_scaled = scaler.fit_transform(df)

plt.figure(figsize=(22,6))

sns.boxplot(data=df_scaled, orient="h", palette="Set2")
df_scaled
#re arrangeing the scaled data as a dataframe

df_scaled =pd.DataFrame(df_scaled,columns=list(df.columns))

df_scaled
#Featus and lables 

X =  df_scaled.drop(["Outcome"],axis=1)

y = df.Outcome 
#importing train_test_split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
#import model

from sklearn.neighbors import KNeighborsClassifier



test_scores = []#will store testing scores 

train_scores = []#will score training scores

a = range(1,15) # stores a list between 1 and 15





for i in a:

    clasifier = KNeighborsClassifier(i)

    clasifier.fit(X_train,y_train)

    

    train_scores.append(clasifier.score(X_train,y_train))

    test_scores.append(clasifier.score(X_test,y_test))
print(train_scores)
print(test_scores)
for i,v in enumerate(train_scores):

    print(i+1,"   ",v)

#access the index position of given value in train_scores     

## score that comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i+1 for i, v in enumerate(train_scores) if v == max_train_score]



print('Max train score : ',max_train_score ,'\n',"k = ", train_scores_ind)
## score that comes from testing on the same datapoints that were used for training

max_test_score = max(test_scores)

test_scores_ind = [i+1 for i, v in enumerate(test_scores) if v == max_test_score]



print('Max test score : ',max_test_score ,'\n',"k = ", test_scores_ind)
plt.figure(figsize=(12,5))

sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
#Setup a knn classifier with 11 neighbors

clasifier = KNeighborsClassifier(11)



clasifier.fit(X_train,y_train)

clasifier.score(X_test,y_test)
#import confusion_matrix

from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above. Creating the confusion Matrix

preds = clasifier.predict(X_test)

preds
conf_matrix =pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted'], margins=False)



#conf_matrix=conf_matrix.drop(['All'],axis=1)

conf_matrix
c_matrix =pd.DataFrame(confusion_matrix(y_test,preds))

c_matrix
sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
#import classification_report

from sklearn.metrics import classification_report as report

print(report(y_test,preds))
##