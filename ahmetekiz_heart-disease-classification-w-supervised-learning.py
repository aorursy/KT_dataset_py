# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")  # patients data frame
df.info()
df.head()
# libraries for Visualization



import seaborn as sns

import matplotlib.pyplot as plt
labels = ['M' if i == 1 else 'F' for i in df.sex.value_counts().index]

colors = ['gray','red']

explode = [0,0.1]

sizes = [df.sex.value_counts().values]



# visual

plt.figure(figsize = (8,8))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')   # matplot methodu , 'autopct' ile 1 tane ondalik kismini gostermek icin

plt.title('Sex Ratio of Patients',color = 'blue',fontsize = 15)

plt.show()
plt.figure(figsize=(15,10))

sns.countplot(df.age)

plt.title("Age Distribution",color = 'black',fontsize=20)

plt.show()


heart_disease = ['Has Heart Disease' if i == 0 else 'No Heart Disease' for i in df.target]



df_1 =  pd.DataFrame({'target':heart_disease})



plt.figure(figsize=(10,7))



sns.countplot(x = df_1.target)

plt.ylabel('Count')

plt.xlabel('Has Heart Disease or No Heart Disease')

plt.title('Heart Disease Count',color = 'blue',fontsize=15)

plt.show()
y = df.target.values

x_data = df.drop(["target"],axis=1)



# normalization 

x = ( x_data - np.min(x_data) ) / ( np.max(x_data) - np.min(x_data) ).values
# %% split data

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42) # test data = 0.2 data
accuracies = {} # Create a dictionary to save algorithms accuracies.



def save_score(name,score):

    """

    Parameters

    ----------

    name : Algorithm name

    score : Algorithm test score    



    Returns

    -------

    None.

    """

    accuracies[name] = score*100

    

    return None
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()



lr.fit(x_train,y_train)



print("Test Accuracy {}".format(lr.score(x_test,y_test)))



save_score( "Logistic Regression Classification",lr.score(x_test,y_test) )
# knn model



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k



knn.fit(x_train,y_train)



prediction = knn.predict(x_test)



print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))



save_score( "K-Nearest Neighbour (KNN) Classification",knn.score(x_test,y_test) )
# find k value



score_list = []



for each in range(1,50):

    

    knn2 = KNeighborsClassifier(n_neighbors = each)

    

    knn2.fit(x_train,y_train)

    

    score_list.append(knn2.score(x_test,y_test))

    

plt.figure(figsize=(30,6))   

plt.plot(range(1,50),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
# SVM



from sklearn.svm import SVC



svm = SVC(random_state = 1)



svm.fit(x_train,y_train)



print("print accuracy of svm algo: ",svm.score(x_test,y_test))



save_score( "Support Vector Machine (SVM) Classification",svm.score(x_test,y_test) )
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()



nb.fit(x_train,y_train)



print("Accuracy of Naive Bayes Algo: ",nb.score(x_test,y_test))



save_score( "Naive Bayes Classification",nb.score(x_test,y_test) )
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()



dt.fit(x_train,y_train)



print("score: ", dt.score(x_test,y_test))



save_score( "Decision Tree Classification",dt.score(x_test,y_test) )
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100,random_state = 1)  # n_estimater = kac agac olacak



rf.fit(x_train,y_train)



print("random forest algo result: ",rf.score(x_test,y_test))



accuracies["Random Forest Classification"] = [rf.score(x_test,y_test)*100]



save_score( "Random Forest Classification",rf.score(x_test,y_test) )
plt.figure(figsize=(30,8))

sns.barplot(x = list( accuracies.keys() ), y = list( accuracies.values() ) )

plt.show()
# Predictions



y_lr_pred = lr.predict(x_test)

y_knn_pred = knn.predict(x_test)

y_svm_pred = svm.predict(x_test)

y_nb_pred = nb.predict(x_test)

y_dt_pred = dt.predict(x_test)

y_rf_pred = rf.predict(x_test)



y_true = y_test
# confusion matrix



from sklearn.metrics import confusion_matrix



cm_lr = confusion_matrix(y_true,y_lr_pred)

cm_knn = confusion_matrix(y_true,y_knn_pred)

cm_svm = confusion_matrix(y_true,y_svm_pred)

cm_nb = confusion_matrix(y_true,y_nb_pred)

cm_dt = confusion_matrix(y_true,y_dt_pred)

cm_rf = confusion_matrix(y_true,y_rf_pred)
# cm visualization



plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,4)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,5)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dt,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,6)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()