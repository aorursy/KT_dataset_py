# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/titanic/train_and_test2.csv")
data.info()
data.corr()
# Multiple Linear Regression uygulamak için kategorik veriyi sayısal değere dönüştürmek için kolonu bir değişkene atadık

passengerid =  pd.DataFrame(data.iloc[:,0:1].values,index=range(len(data.iloc[:,0:1].values)),columns=['passengerid',])

# Multiple Linear Regression uygulamak için kategorik veriyi sayısal değere dönüştürmek için kolonu bir değişkene atadık.

age = pd.DataFrame(data.iloc[:,1:2].values,index=range(len(data.iloc[:,1:2].values)),columns=['age',])

# Tahmin ettireceğimiz kolonun değerlerini bir değişkene atadık.

sex = pd.DataFrame(data.iloc[:,2:3].values,index=range(len(data.iloc[:,2:3].values)),columns=['sex',])

embarked = pd.DataFrame(data.iloc[:,23:24].values,index=range(len(data.iloc[:,23:24].values)),columns=['embarked',])

sibsp = pd.DataFrame(data.iloc[:,4:5].values,index=range(len(data.iloc[:,4:5].values)),columns=['sibsp',])

survived = pd.DataFrame(data.iloc[:,26:27].values,index=range(len(data.iloc[:,26:27].values)),columns=['survived',])

type(passengerid)

print(passengerid )
preData = pd.concat([passengerid,age,sex,embarked,sibsp,survived],axis=1)
preData.head()
import seaborn as sns

sns.heatmap(preData.isnull(), yticklabels=False)
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(preData.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
preData[['sex', 'survived']].groupby(['sex'], as_index=False).agg(['mean', 'count', 'sum'])
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(preData,survived,test_size=0.33,random_state=0)

# sklearn kütüphanesini kullanarak LinearRegression sınıfını dahil ediyoruz.

from sklearn.linear_model import LinearRegression

# LinearRegression sınıfından bir nesne oluşturuyoruz.

lr = LinearRegression()

# Train veri kümelerini vererek makineyi eğitiyoruz.

lr.fit(x_train,y_train)

# test kümesini vererek eğittiğimiz makinenin tahmin üretmesini sağlıyoruz.

result = lr.predict(x_test)

print(result)
cv = KFold(n_splits=10)            # Desired number of Cross Validation folds

accuracies = list()

max_attributes = len(list(preData))

depth_range = range(1, max_attributes + 1)



# Testing max_depths from 1 to max attributes

# Uncomment prints for details about each Cross Validation pass

for depth in depth_range:

    fold_accuracy = []

    tree_model = tree.DecisionTreeClassifier(max_depth = depth)

    # print("Current max depth: ", depth, "\n")

    for train_fold, valid_fold in cv.split(preData):

        f_train = preData.loc[train_fold] # Extract train data with cv indices

        f_valid = preData.loc[valid_fold] # Extract valid data with cv indices



        model = tree_model.fit(X = f_train.drop(['survived'], axis=1), 

                               y = f_train["survived"]) # We fit the model with the fold train data

        valid_acc = model.score(X = f_valid.drop(['survived'], axis=1), 

                                y = f_valid["survived"])# We calculate accuracy with the fold validation data

        fold_accuracy.append(valid_acc)



    avg = sum(fold_accuracy)/len(fold_accuracy)

    accuracies.append(avg)

    # print("Accuracy per fold: ", fold_accuracy, "\n")

    # print("Average accuracy: ", avg)

    # print("\n")

    

# Just to show results conveniently

df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})

df = df[["Max Depth", "Average Accuracy"]]

print(df.to_string(index=False))
# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(preData.drop(['survived'],axis=1).values,preData['survived'],test_size=0.33,random_state=0)

# Create Decision Tree with max_depth = 3

decision_tree = tree.DecisionTreeClassifier(max_depth = 3)



decision_tree.fit(x_train, y_train)



# Predicting results for test dataset

y_pred = decision_tree.predict(x_test)

print(y_pred)



        
