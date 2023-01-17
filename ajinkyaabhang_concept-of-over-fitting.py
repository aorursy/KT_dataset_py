#importing library

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import tree, metrics

import seaborn as sns
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.shape
df.describe()
df.isnull().sum()
df.quality.unique()
# mapping the target variable

quality_map = {

    3:0,

    4:1,

    5:2,

    6:3,

    7:4,

    8:5

}
df['quality'] = df.quality.map(quality_map)
df.quality.head()
df.tail()
#shuffling and splitting data into train into test



df = df.sample(frac=1).reset_index(drop=True)
df.head()
#for training

df_train = df.head(1000)



#for testing

df_test = df.tail(599)
df_train_X = df_train.drop(['quality'],axis=1)
df_test_X = df_test.drop(['quality'],axis=1)
df_train_X.shape
df_test_X.shape
train_accuracies = []

test_accuracies = []
for i in range(1,30):

    clf = tree.DecisionTreeClassifier(max_depth=i)

    clf.fit(df_train_X, df_train['quality'])

    

    train_prediction = clf.predict(df_train_X)

    test_prediction = clf.predict(df_test_X)

    

    train_accuracy = metrics.accuracy_score(df_train['quality'], train_prediction)

    test_accuracy = metrics.accuracy_score(df_test['quality'], test_prediction)

    

    train_accuracies.append(train_accuracy)

    test_accuracies.append(test_accuracy)
plt.figure(figsize=(10, 5))  

sns.set_style("whitegrid")  

plt.plot(train_accuracies, label="train accuracy")  

plt.plot(test_accuracies, label="test accuracy")  

plt.legend(loc="upper left", prop={'size': 15})  

plt.xticks(range(0, 26, 5))  

plt.xlabel("max_depth", size=20)  

plt.ylabel("accuracy", size=20)  

plt.show() 