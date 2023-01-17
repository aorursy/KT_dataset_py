# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #seaborn

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart.csv")
df.info()
df.columns
df.head()
df.describe()
df.dropna(inplace = True)
#Correlation Map

f,ax  = plt.subplots( figsize = (12,10))

sns.heatmap(df.corr() , annot = True , linewidth = 10 , linecolor = 'black' , fmt = '.1f' , ax = ax)

plt.show()
sns.countplot(df.target)

plt.title("Number of Diseases")

plt.xlabel("NOT DISEASE                               DISEASE")

plt.ylabel("Count")



df_disease  = df[df['target'] == 1]



sns.countplot(df_disease.sex,palette="Set3")

plt.title("Hearth disease occurence according to sex")

plt.xlabel("FEMALE                                      MALE")
#%% Box Plot

 # Classifaction  

 #sex M = 1 F = 0

 #choloesterol

 #hue = Target 1 is disease , 0 is not disease

 #outlayer can be seen 

 

sns.boxplot(x ="sex", y = "chol" , hue = "target" , data  = df , palette = "PRGn")

plt.show()





#%% Swarmn Plot 

#FOR CLASSIFICATION

sns.swarmplot (x= "sex", y ="cp" , hue = "target" ,data = df )

#FOR CLASSIFICATION

sns.swarmplot (x= "sex", y ="chol" , hue = "target" ,data = df )
#FOR CLASSIFICATION

sns.swarmplot (x= "sex", y ="trestbps" , hue = "target" ,data = df )
sns.swarmplot(x = "sex" , y ="fbs", hue ="target", data = df )
sns.swarmplot(x = "sex" , y ="thalach", hue ="target", data = df )
df_male = df[df["sex"] == 1]

df_female =df[df["sex"] == 0]
# Pair Plot

plt.figure(figsize=(10,8), dpi= 80)

sns.pairplot(df, kind="scatter", hue="target", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))

plt.show()
#%% Kde Plot # 



sns.kdeplot(df.target ,df.cp , shade =True ,cut = 1)

df.head(5)

df_FV = df_disease[["trestbps","chol","thalach"]]
#%% Violin Plot  



#Distribution of trestbps , chol , thalach in patients who HAVE Heart disease



# Show each distribution with both violins and points

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=df_FV, palette=pal, inner="points") 



plt.show()

data2 = df_disease[["restecg","exang","oldpeak","thal","slope"]]
ax = sns.violinplot(data=data2, palette="Set2", inner="points",scale="width",split = True) 



plt.show()
df_2 = df.drop(['age','sex'],axis =1)
color_list = ['red' if i== 1 else 'green' for i in df_2.loc[:,'target']]

pd.plotting.scatter_matrix(df_2.loc[:, df_2.columns != 'target'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
x_data = df.drop(['target'],axis = 1)
y = df.target.values
# %% normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

print("test accuracy of LogisticRegression Model is  {}".format(lr.score(x_test,y_test)))



# KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

#x,y = df.loc[:,df.columns != 'target'], df.loc[:,'target']

knn.fit(x,y)

prediction = knn.predict(x)

print('Prediction: {}'.format(prediction))

# train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)



#%% KNN Classifier 

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=5) accuracy is: ',knn.score(x_test,y_test)) # accuracy
# Model complexity

neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

#%% decision tree



from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)



print("decision tree score: ", dt.score(x_test,y_test))



#%%  random forest

# %%

# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)





from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100,random_state = 1)

rf.fit(x_train,y_train)

print("random forest algorithm result: ",rf.score(x_test,y_test))
#%%SVC Code and accuracy 

# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=42)

 # %% SVM

 

from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

 

# %% test

print("print accuracy of svm algo: ",svm.score(x_test,y_test))



# %%Naive Bayes Application

# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)



from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

 

# %% test

print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))

 
# %%

# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)





#%%  random forest

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100,random_state = 42)

rf.fit(x_train,y_train)

print("random forest algo result: ",rf.score(x_test,y_test))



##CONFUSION MATRIX FOR RANDOM FOREST MODEL

y_pred = rf.predict(x_test)

y_true = y_test

#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)





# %% cm visualization

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.3)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
#K fold CV K = 10

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = knn , X = x_train , y = y_train , cv = 10)

print("avarege accuracy : " ,np.mean(accuracies))

print("average std" ,np.std(accuracies))
knn.fit(x_train,y_train)

print("test accuracy: ",knn.score(x_test,y_test))
from sklearn.model_selection import GridSearchCV
# %% grid search cross validation for knn



from sklearn.model_selection import GridSearchCV



grid = {"n_neighbors":np.arange(1,50)}

knn= KNeighborsClassifier()



knn_cv = GridSearchCV(knn, grid, cv = 10)  # GridSearchCV

knn_cv.fit(x,y)



#%% print hyperparameter KNN algoritmasindaki K degeri

print("tuned hyperparameter K: ",knn_cv.best_params_)

print("tuned best score: ",knn_cv.best_score_)
# STARTING FROM BEGINNING

x_data = df.drop(['target'],axis = 1) # We drop target for train and test data

y = df.target.values # y is our target where there is disease and not 



#NORMALIZATION of x 

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values



#We split our data as test and train

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)



#We fit our data

knn.fit(x_train,y_train)

lr.fit(x_train,y_train)

dt.fit(x_train,y_train)

rf.fit(x_train,y_train)

svm.fit(x_train,y_train)
x_axis  = ['logistic regression', 'knn' , 'decision_tree' , 'random_forest' , 'support_vector_machine', ]

y_results =[lr.score(x_test,y_test),knn.score(x_test,y_test),dt.score(x_test,y_test),rf.score(x_test,y_test),svm.score(x_test,y_test)]
plt.figure(figsize=(11,10))

plt.xlabel("Machine Learning Algorithms")

plt.ylabel("ML SCORES")

plt.title("ML Algorithms vs Test Scores")

plt.plot(x_axis ,y_results)

y_results