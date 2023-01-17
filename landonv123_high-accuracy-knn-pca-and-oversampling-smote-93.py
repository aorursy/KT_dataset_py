#import the required modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#import our data file "glass.csv" into a pandas dataframe

glass_df = pd.read_csv("../input/glass/glass.csv")
#get information about the general values of each feature

glass_df.describe()
#see the first 10 values of the dataset

glass_df.head(10)
#get info about the datatypes of each feature

glass_df.info()
#get counts of null values for each feature

glass_df.isnull().sum()
#plot a histogram of each feature

glass_df.hist(figsize=(20,20))
# Checking to see how spread out our features values are

ax = sns.boxplot(data=glass_df)
#Get the distribution of the different classifications

glass_df.Type.value_counts().plot(kind="bar")
# This will be the dataframe containing the features used to train our model

X = pd.DataFrame(glass_df.drop(["Type"], axis = 1),

            columns=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'])



# This will be the dataframe containing the labels of each data point

y=glass_df.Type
#importing train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/5, random_state=5, stratify = y)
#importing the k classifier

from sklearn.neighbors import KNeighborsClassifier





test_scores = []

train_scores = []



# we will be testing 14 different values for k, starting with 1 and ending before 15

for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
# This score comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
# This score comes from testing on the datapoints that were split in the beginning to be used for testing solely

max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
#plotting the train and test score for each value of k we tested

plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()

X_minmax = min_max_scaler.fit_transform(X)

X_norm = pd.DataFrame(X_minmax)
# a peek at the new normalized features

X_norm.head()
#now all of our values fall between 0 and 1

ax = sns.boxplot(data=X_norm)
#now lets run the model on the normalized data and see if it has any effect on the accuracy

#importing train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_norm,y, test_size=1/5, random_state=5, stratify = y)



test_scores = []

train_scores = []



# we will be testing 14 different values for k, starting with 1 and ending before 15

for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
# This score comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
# This score comes from testing on the datapoints that were split in the beginning to be used for testing solely

max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
from sklearn.decomposition import PCA



scores = []



for i in range(9):

    pca = PCA(n_components=(i+1))

    principalComponents = pca.fit_transform(X_norm)

    principalDf = pd.DataFrame(data = principalComponents)

    

    X_train, X_test, y_train, y_test = train_test_split(principalDf,y, test_size=1/5, random_state=5, stratify = y)



    test_scores = []

    train_scores = []



    # we will be testing 14 different values for k, starting with 1 and ending before 15

    for i in range(1,15):



        knn = KNeighborsClassifier(i)

        knn.fit(X_train,y_train)

    

        train_scores.append(knn.score(X_train,y_train))

        test_scores.append(knn.score(X_test,y_test))

        

    # This score comes from testing on the datapoints that were split in the beginning to be used for testing solely

    max_test_score = max(test_scores)

    scores.append(max_test_score)

    

for i in range(len(scores)):

    print("With {} components, our accuracy was {}.".format(i+1,scores[i]))
plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,10),scores)

p.set(xlabel="Number of Principal Components", ylabel="Model Accuracy", title = "Model Accuracy per number of Principal Components")
print("Our model had a maximum accuracy score of {:.2f}% with {} principal components.".format(max(scores)*100,scores.index(max(scores))+1))
ax = y.value_counts().plot(kind="bar")

ax.set(xlabel="Type of Glass", ylabel="Count", title = "Before SMOTE")
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

x_res, y_res = sm.fit_resample(X_norm,y)

y_res_df = pd.DataFrame(y_res)

ax = y_res_df.Type.value_counts().plot(kind="bar")

ax.set(xlabel="Type of Glass", ylabel="Count", title = "After SMOTE")
scores = []



for i in range(1,3):

    pca = PCA(n_components=(i))

    principalComponents = pca.fit_transform(x_res)

    principalDf = pd.DataFrame(data = principalComponents)

    

    X_train, X_test, y_train, y_test = train_test_split(x_res,y_res, test_size=1/5, random_state=5, stratify = y_res)



    test_scores = []

    train_scores = []



    # we will be testing 14 different values for k, starting with 1 and ending before 15

    for i in range(1,15):



        knn = KNeighborsClassifier(i)

        knn.fit(X_train,y_train)

    

        test_scores.append(knn.score(X_test,y_test))

        

    # This score comes from testing on the datapoints that were split in the beginning to be used for testing solely

    max_test_score = max(test_scores)

    scores.append(max_test_score)

    

for i in range(len(scores)):

    print("With {} components, our accuracy was {:.2f}%.".format(i+1,scores[i]*100))
finalDf = pd.concat([principalDf, y_res], axis = 1)

color_dict = dict({1:'brown',

                  2:'green',

                  3: 'orange',

                  5: 'red',

                   6: 'dodgerblue',

                  7: 'purple'})

plt.figure(figsize=(10,10))

ax = sns.scatterplot(x=0,y=1,hue="Type",data=finalDf, palette=color_dict)

ax.set(xlabel="PC1",ylabel="PC2", title = "Our Final Dataset with 2 Principal Components")