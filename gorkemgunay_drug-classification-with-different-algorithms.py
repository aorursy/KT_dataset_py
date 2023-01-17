# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/drug-classification/drug200.csv")
df.head()
df.describe()
df.isnull().sum()
df.isna().sum()
df.info()
print("Max Age:", df.Age.max())

print("Min Age:", df.Age.min())
# Age distribution

plt.figure(figsize = (9,5))

sns.distplot(df.Age)

plt.show()
df.Sex.value_counts()
# Sex Distribution

plt.figure(figsize=(9,5))

sns.countplot(x = df.Sex)

plt.show()
df.BP.value_counts()
plt.figure(figsize = (9,5))

sns.countplot(df.BP)

plt.show()
df.Cholesterol.value_counts()
plt.figure(figsize = (9,5))

sns.countplot(df.Cholesterol)

plt.show()
print("Max Na_to_K:",df.Na_to_K.max())

print("Min Na_to_K:",df.Na_to_K.min())

print("Mean Na_to_K:",df.Na_to_K.mean())
plt.figure(figsize = (9,5))

sns.distplot(df.Na_to_K)

plt.show()
df.Drug.value_counts()
plt.figure(figsize = (9,5))

sns.countplot(df.Drug)

plt.show()
plt.figure(figsize = (9,5))

sns.swarmplot(x = "Drug", y = "Age",data = df)

plt.legend(df.Drug.value_counts().index)

plt.title("Age -- Drug")

plt.show()
print("Minimum Age of DrugB",df.Age[df.Drug == "drugB"].min())

print("Maximum Age of DrugA",df.Age[df.Drug == "drugA"].max())
df_Sex_Drug = df.groupby(["Drug","Sex"]).size().reset_index(name = "Count")

df_Sex_Drug


plt.figure(figsize = (9,5))

sns.barplot(x = "Drug",y="Count", hue = "Sex",data = df_Sex_Drug)

plt.title("Sex -- Drug")

plt.show()
df_BP_Drug = df.groupby(["Drug","BP"]).size().reset_index(name = "Count")

df_BP_Drug
plt.figure(figsize = (9,5))

sns.barplot(x = "Drug",y="Count", hue = "BP",data = df_BP_Drug)

plt.title("BP -- Drug")

plt.show()
plt.figure(figsize = (9,5))

sns.swarmplot(x = "Drug", y = "Na_to_K",data = df)

plt.title("Na_to_K -- Drug")

plt.show()
print("Minimum Na_to_K for DrugY:",df.Na_to_K[df.Drug == "DrugY"].min())
df_CH_Drug = df.groupby(["Drug","Cholesterol"]).size().reset_index(name = "Count")

df_CH_Drug
plt.figure(figsize = (9,5))

sns.barplot(x = "Drug",y="Count", hue = "Cholesterol",data = df_CH_Drug)

plt.title("Cholesterol -- Drug")

plt.show()
plt.figure(figsize = (9,5))

sns.swarmplot(x = "Drug", y = "Na_to_K",hue="BP",data = df)

plt.legend()

plt.title("Na_to_K -- BP -- Drug")

plt.show()
df['Na_to_K_Bigger_Than_15'] = [1 if i >=15.015 else 0 for i in df.Na_to_K]

df.head()
df_NaK15 = df.groupby(["Drug","Na_to_K_Bigger_Than_15"]).size().reset_index(name = "Count")

df_NaK15
plt.figure(figsize = (9,5))

sns.barplot(x = "Drug",y="Count", hue = "Na_to_K_Bigger_Than_15",data = df_NaK15)

plt.title("Na_to_K_Bigger_Than_15 -- Drug")

plt.show()
from sklearn.preprocessing import LabelEncoder



def label_encoder(y):

    le = LabelEncoder()

    df[y] = le.fit_transform(df[y])
label_list = ["Sex","BP","Cholesterol","Na_to_K","Na_to_K_Bigger_Than_15","Drug"]



for l in label_list:

    label_encoder(l)
df.head()
from sklearn.model_selection import train_test_split



x = df.drop(["Drug"],axis=1)

y = df.Drug



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42, shuffle = True)



y_train = y_train.values.reshape(-1,1)

y_test = y_test.values.reshape(-1,1)



print("x_train shape:",x_train.shape)

print("x_test shape:",x_test.shape)

print("y_train shape:",y_train.shape)

print("y_test shape:",y_test.shape)
# To store results of models

result_dict_train = {}

result_dict_test = {}
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

accuracies = cross_val_score(knn, x_train, y_train, cv=5)

knn.fit(x_train,y_train)



print("Train Score:",np.mean(accuracies))

print("Test Score:",knn.score(x_test,y_test))
result_dict_train["KNN Default Train Score"] = np.mean(accuracies)

result_dict_test["KNN Default Test Score"] = knn.score(x_test,y_test)
grid = {'n_neighbors':np.arange(1,120),

        'p':np.arange(1,3),

        'weights':['uniform','distance']

       }



knn = KNeighborsClassifier(algorithm = "auto")

knn_cv = GridSearchCV(knn,grid,cv=5)

knn_cv.fit(x_train,y_train)



print("Hyperparameters:",knn_cv.best_params_)

print("Train Score:",knn_cv.best_score_)

print("Test Score:",knn_cv.score(x_test,y_test))
result_dict_train["KNN GridSearch Train Score"] = knn_cv.best_score_

result_dict_test["KNN GridSearch Test Score"] = knn_cv.score(x_test,y_test)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(random_state = 42)

accuracies = cross_val_score(rfc, x_train, y_train, cv=5)

rfc.fit(x_train,y_train)



print("Train Score:",np.mean(accuracies))

print("Test Score:",rfc.score(x_test,y_test))
result_dict_train["Random Forest Default Train Score"] = np.mean(accuracies)

result_dict_test["Random Forest Default Test Score"] = rfc.score(x_test,y_test)
grid = {'n_estimators':np.arange(100,1000,100),

        'criterion':['gini','entropy']

       }



rf = RandomForestClassifier(random_state = 42)

rf_cv = GridSearchCV(rf,grid,cv=5)

rf_cv.fit(x_train,y_train)



print("Hyperparameters:",rf_cv.best_params_)

print("Train Score:",rf_cv.best_score_)

print("Test Score:",rf_cv.score(x_test,y_test))
result_dict_train["Random Forest GridSearch Train Score"] = rf_cv.best_score_

result_dict_test["Random Forest GridSearch Test Score"] = rf_cv.score(x_test,y_test)
from sklearn.svm import SVC

svc = SVC(random_state = 42)

accuracies = cross_val_score(svc, x_train, y_train, cv=5)

svc.fit(x_train,y_train)



print("Train Score:",np.mean(accuracies))

print("Test Score:",svc.score(x_test,y_test))
result_dict_train["SVM Default Train Score"] = np.mean(accuracies)

result_dict_test["SVM Default Test Score"] = svc.score(x_test,y_test)
grid = {

    'C':[0.01,0.1,1,10],

    'kernel' : ["linear","poly","rbf","sigmoid"],

    'degree' : [1,3,5,7],

    'gamma' : [0.01,1]

}



svm  = SVC ();

svm_cv = GridSearchCV(svm, grid, cv = 5)

svm_cv.fit(x_train,y_train)

print("Best Parameters:",svm_cv.best_params_)

print("Train Score:",svm_cv.best_score_)

print("Test Score:",svm_cv.score(x_test,y_test))

result_dict_train["SVM GridSearch Train Score"] = svm_cv.best_score_

result_dict_test["SVM GridSearch Test Score"] = svm_cv.score(x_test,y_test)
df_result_train = pd.DataFrame.from_dict(result_dict_train,orient = "index",columns=["Score"])

df_result_train
df_result_test = pd.DataFrame.from_dict(result_dict_test,orient = "index",columns=["Score"])

df_result_test
fig,ax = plt.subplots(1,2,figsize=(20,5))

sns.barplot(x = df_result_train.index,y = df_result_train.Score,ax = ax[0])

sns.barplot(x = df_result_test.index,y = df_result_test.Score,ax = ax[1])

ax[0].set_xticklabels(df_result_train.index,rotation = 75)

ax[1].set_xticklabels(df_result_test.index,rotation = 75)

plt.show()