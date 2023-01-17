# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
data = pd.read_csv("../input/heart.csv")

data.head(5)
data.info()
print("number of rows in the dataset: " + str(data.shape[0]))

print("number of columns in the dataset: " + str(data.shape[1]))
data.isnull().sum()
data.describe()
#making piechart of the age feature : male/female



male = len( data[data["sex"] == 1] )

female = len( data[data["sex"] == 0] )



plt.figure(figsize=(8,6))



labels = "Male", "Female"

sizes = [male, female]

colors = ["yellow", "green"]



plt.pie(x=sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')

plt.axis("equal")

plt.show()
#making piechart of cp feature: chest pain type



size = [ len(data[data["cp"]==0]), len(data[data["cp"]==1]), len(data[data["cp"]==2]), len(data[data["cp"]==3]) ]

colors = ["red", "yellow", "green", "blue"]

labels = [ "chest pain 0", "chest pain 1", "chest pain 2", "chest pain 3" ]



plt.figure(figsize=(8,6))



plt.pie(x=size, colors=colors, labels=labels, autopct="%1.1f%%", startangle=180)

plt.axis("equal")

plt.show()
sns.set_style("whitegrid")
#plotting the correlation between features



plt.figure(figsize=(14,8))

sns.heatmap(data.corr(), annot=True, cmap="coolwarm")

plt.show()
#plotting the distribution of thalach: MAXIMUM HEART RATE ACHIEVED feature



sns.distplot(data["thalach"], kde=False, bins=30, color = "red")

plt.show()
#plotting the distribution of chol: SERUM CHOLESTROL IN MG/DL feature



sns.distplot(data["chol"], kde=False)

plt.show()
#plotting the number of people with heart disease based on age



plt.figure(figsize=(15,6))

sns.countplot(x="age", hue="target", data=data, palette="GnBu")

plt.show()
#scatterplot of thalach vs chol



plt.figure(figsize=(10,8))

sns.scatterplot(x="thalach", y="chol", hue="target", data=data)

plt.show()
#scatterplot between trestbps vs age



plt.figure(figsize=(10,8))

sns.scatterplot(x="age", y="trestbps", hue="target", data=data)

plt.show()
from sklearn.model_selection import train_test_split
#splitting the dataset



x = data.drop("target", axis=1)

y = data["target"]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#preprocessing the data: scaling



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.fit_transform(x_test)



x_train = pd.DataFrame( x_train_scaled )

x_test = pd.DataFrame( x_test_scaled )
x_train.shape
x_test.shape
#gridsearchcv to find best parameters

#kNN algorithm classification



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



knn = KNeighborsClassifier()

params = { "n_neighbors" : [ i for i in range(1,33,2) ] }
model = GridSearchCV(knn, params, cv=10)
model.fit(x_train, y_train)



#printing the best value for k in kNN

print(model.best_params_)
pred = model.predict(x_test)
#printing accuracy score



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



print("accuracy score is : ", accuracy_score(y_test,pred))



print("accuracy score using kNN is ", round(accuracy_score(y_test,pred), 5)*100, "%")
# creating confusion matrix



class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



confusion = confusion_matrix(y_test, pred)



#create a heat map

sns.heatmap(pd.DataFrame(confusion), annot = True, cmap = 'YlGnBu', fmt = 'g')



plt.title('Confusion matrix for k-Nearest Neighbors Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')



plt.show()
# printing the classification report



print(classification_report(y_test, pred))
#receiver operating characteristics ROC curve



from sklearn.metrics import roc_auc_score, roc_curve



#get prediction probabilities

y_prob = model.predict_proba(x_test)[:,1]



false_pos_rate_knn, true_pos_rate_knn, threshold_knn = roc_curve(y_test, y_prob)



plt.figure(figsize=(6,4))

plt.plot(false_pos_rate_knn, true_pos_rate_knn)

plt.plot([0,1],ls='--')



plt.xlabel("false positive rate")

plt.ylabel("true positive rate")

plt.title("receiver operating characteristic curve")



plt.show()
# area under the roc curve



print(roc_auc_score(y_test, pred))
# classification using logistic regression



from sklearn.linear_model import LogisticRegression

log = LogisticRegression()



# finding best parameters from gridsearchcv



params = { "penalty":["l1","l2"], "C":[0.01,0.1,1,10,100], "class_weight":["balanced",None] }

log_model = GridSearchCV(log, param_grid=params, cv=10)

log_model.fit(x_train,y_train)

print(log_model.best_params_)
pred_log = log_model.predict(x_test)
print("the accuracy score using logistic regression is : ", round(accuracy_score(y_test, pred_log),5)*100, "%")
print(classification_report(y_test,pred_log))
# creating the confusion matrix



class_names = [0,1]

fig,ax = plt.subplots()

tickmarks = np.arange(len(class_names))

plt.xticks(tickmarks, class_names)

plt.yticks(tickmarks, class_names)



confusion = confusion_matrix(y_test, pred_log)



sns.heatmap(pd.DataFrame(confusion), annot = True)



plt.title('Confusion matrix for Logisitic Regression Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
target_prob_log = log_model.predict_proba(x_test)[:,1]



false_pos_rate_log, true_pos_rate_log, threshold_log = roc_curve(y_test, target_prob_log)



plt.figure(figsize=(6,4))

plt.plot(false_pos_rate_log, true_pos_rate_log)

plt.plot([0,1], ls="--")



plt.title("roc curve for logistic regression")

plt.xlabel("false positive rate")

plt.ylabel("true positive rate")



plt.show()
# area under the roc curve



print(roc_auc_score(y_test, pred_log))
# comparing the roc curve from knn and logistic regression method



plt.figure(figsize=(8,6))

plt.plot(false_pos_rate_log, true_pos_rate_log, label="logistic regression")

plt.plot(false_pos_rate_knn, true_pos_rate_knn, label="KNN")

plt.plot([0,1], ls="--")



plt.title("roc curve for logistic regression vs kNN")

plt.xlabel("false positive rate")

plt.ylabel("true positive rate")

plt.legend()

plt.show()