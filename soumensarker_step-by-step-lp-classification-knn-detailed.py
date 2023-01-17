# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import sklearn

from sklearn import preprocessing

from scipy.stats import pearsonr



from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import accuracy_score



# visualization and plotting

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/indian_liver_patient.csv')

df.shape
#show the first five rows of data

df.head()
df.info(verbose=True)
df.describe()
# let's look on target variable - classes imbalanced?

df.rename(columns={'Dataset':'target'},inplace=True)

df.head()
# let's look on target variable - classes imbalanced?

df['target'].value_counts()
print(df.isnull().sum())
p = df.hist(figsize = (20,20))
df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)
le = preprocessing.LabelEncoder()

le.fit(df.Gender.unique())

df['Gender_Encoded'] = le.transform(df.Gender)

df.drop(['Gender'], axis=1, inplace=True)
## checking the balance of the data by plotting the count of outcomes by their value

color_wheel = {1: "#0392cf", 

               2: "#7bc043"}

colors = df["target"].map(lambda x: color_wheel.get(x + 1))

print(df.target.value_counts())

p=df.target.value_counts().plot(kind="bar")
p=sns.pairplot(df, hue = 'target')
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(df.drop(["target"],axis = 1),),

        columns=['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',

        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 

        'Albumin', 'Albumin_and_Globulin_Ratio','Gender_Encoded'])
X.head()
y = df.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42, stratify=y)
test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
## score that comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely

max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
knn = KNeighborsClassifier(12)



knn.fit(X_train,y_train)

knn.score(X_test,y_test)
#import confusion_matrix

from sklearn.metrics import confusion_matrix

#let us get the predictions using the classifier we had fit above

y_pred = knn.predict(X_test)

confusion_matrix(y_test,y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
y_pred = knn.predict(X_test)

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
#import classification_report

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
#import GridSearchCV

from sklearn.model_selection import GridSearchCV

#In case of classifier like knn the parameter to be tuned is n_neighbors

param_grid = {'n_neighbors':np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv= GridSearchCV(knn,param_grid,cv=5)

knn_cv.fit(X,y)



print("Best Score:" + str(knn_cv.best_score_))

print("Best Parameters: " + str(knn_cv.best_params_))
best_clf = KNeighborsClassifier(34)

best_clf.fit(X_train, y_train)

best_clf.score(X_test,y_test)
from yellowbrick.classifier import ROCAUC
fig, ax=plt.subplots(1,1,figsize=(12,8))

roc_auc=ROCAUC(best_clf, ax=ax)

roc_auc.fit(X_train, y_train)

roc_auc.score(X_test, y_test)



roc_auc.poof()