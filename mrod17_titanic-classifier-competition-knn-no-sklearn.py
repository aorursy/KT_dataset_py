
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head(10)
        

#Looking at the numeric variables
train_data.describe()
#notice that the age variable has missing data while the other columns have no missing data
#the Name variable includes titles of the person. One approach to impute the age of missing data is to take
#the average age of each group's title

train_data["Name"] = train_data["Name"].str.split(',').str[1]
train_data["Name"] = train_data["Name"].str.split('.').str[0]
train_data["Name"] = train_data["Name"].str.strip()

print(train_data.head())

x = train_data.groupby('Name').agg(['count']).index.get_level_values('Name')
x
#The names are now transformed into titles.
#Taking average age of each group to fill missing age data
train_data["Age"] = train_data.groupby("Name").transform(lambda x: x.fillna(x.mean()))['Age']
#changing sex to be 0 or 1 for female & male
train_data['Sex'].replace({'female':0,'male':1},inplace=True)
train_data.head()

#Building a knn algorithm
#Need to normalize data (min-max normalization)
#selecting only numeric or categorical variable transformed to numeric variables
train_data_knn = train_data.iloc[:,[False,False,True, False,True,True,True,True,False,True,False,False]]
train_labels_knn = train_data.iloc[:,1]
normalized_data_train=(train_data_knn-train_data_knn.min())/(train_data_knn.max()-train_data_knn.min())
normalized_data_train.head()


#first need distance function (using Euclidean Distance)

def distance(df1,actual_labels,k):
    closest_lst = []
    labels = np.zeros(len(df1),dtype=int)
    for i in range(len(df1)):
        distance_df = ((df1.iloc[i,:] - df1.drop([i],axis=0))**2).sum(axis=1)
        k_closest = ((df1.iloc[i,:] - df1.drop([i],axis=0))**2).sum(axis=1).nsmallest(n=k, keep='first')
        closest_lst.append(k_closest)
       

    return closest_lst


def classify(big_list, labels,k):
    predictions = np.zeros(len(big_list),dtype=int)
    for i in range(len(big_list)):
        summation = 0
        for j in big_list[i].index:
            summation += labels[j]
        if summation/k > .5:
            predictions[i] = 1

    return predictions



###Try different K values on training set, see which is best

for j in range(1,15,2):
    preds = distance(normalized_data_train,train_labels_knn,j)
    predictions = classify(preds, train_labels_knn,j)
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == train_labels_knn[i]:
            accuracy +=1
    print("For K=", j,":")
    print("Accuracy",accuracy/len(predictions))

#Function for testing set
def distance(df1,df2,actual_labels,k):
    closest_lst = []
    labels = np.zeros(len(df1),dtype=int)
    for i in range(len(df1)):
        distance_df = ((df1.iloc[i,:] - df2)**2).sum(axis=1)
        k_closest = ((df1.iloc[i,:] - df2)**2).sum(axis=1).nsmallest(n=k, keep='first')
        closest_lst.append(k_closest)
       

    return closest_lst

preds = distance(normalized_data_train,normalized_data_train,train_labels_knn,5)

def classify(big_list, labels,k):
    predictions = np.zeros(len(big_list),dtype=int)
    for i in range(len(big_list)):
        summation = 0
        for j in big_list[i].index:
            summation += labels[j]
        if summation/k > .5:
            predictions[i] = 1

    return predictions
    
predictions = classify(preds, train_labels_knn,5)
#Get testing data ready:

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


test_data["Name"] = test_data["Name"].str.split(',').str[1]
test_data["Name"] = test_data["Name"].str.split('.').str[0]
test_data["Name"] = test_data["Name"].str.strip()
test_data['Sex'].replace({'female':0,'male':1},inplace=True)


x = test_data.groupby('Name').agg(['count']).index.get_level_values('Name')
test_data["Age"] = test_data.groupby("Name").transform(lambda x: x.fillna(x.mean()))['Age']


test_data_knn = test_data.iloc[:,[False,True,False,True,True,True,True,False,True,False,False]]
normalized_data_test=(test_data_knn-test_data_knn.min())/(test_data_knn.max()-test_data_knn.min())
preds = distance(normalized_data_test,normalized_data_train,train_labels_knn,5)
predictions = classify(preds, train_labels_knn,5)
data = {'PassengerId': test_data["PassengerId"].values, 'Survived':predictions} 
df_submission = pd.DataFrame(data)

df_submission.to_csv("submission5.csv",index=False)