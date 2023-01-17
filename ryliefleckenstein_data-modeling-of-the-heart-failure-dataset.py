import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import RFE

from sklearn.feature_selection import RFECV
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
heart_dat = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

heart_dat.head()
heart_dat.columns
# y includes our labels and x includes our features

dlist = ['DEATH_EVENT']

x = heart_dat.drop(dlist, axis=1)

y = heart_dat['DEATH_EVENT']

x.head()
ax = sns.countplot(y,label="Count")       # Deaths = 96, Alive = 203

N, Y = y.value_counts()                   # 1 - Deaths, 0 - Alive

print('Number of Deaths: ',Y)

print('Number Alive : ',N)
x.describe()
heart_dat_dia = y

heart_dat_n_2 = (x - x.mean()) / (x.std())              # standardization

data = pd.concat([y,heart_dat_n_2.iloc[:,0:12]],axis=1)

data = pd.melt(data,id_vars="DEATH_EVENT",

                    var_name="features",

                    value_name='value') 

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="DEATH_EVENT", data=data,split=True, inner="quart")

plt.xticks(rotation=90)
plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="DEATH_EVENT", data=data)

plt.xticks(rotation=90)
plot_1 = heart_dat.drop(['anaemia',

       'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'age', 'creatinine_phosphokinase','ejection_fraction'],axis=1)

pd.plotting.scatter_matrix(plot_1, c=y, figsize=(15, 15),

 marker='o', hist_kwds={'bins': 20}, s=60,

 alpha=.8)
plot_2 = heart_dat.drop(['anaemia',

       'diabetes', 'high_blood_pressure', 'sex', 'smoking','platelets', 'serum_creatinine', 'serum_sodium','time'],axis=1)

pd.plotting.scatter_matrix(plot_2, c=y, figsize=(15, 15),

 marker='o', hist_kwds={'bins': 20}, s=60,

 alpha=.8)
# Creating a Random Forest Classifier with all features



X_train, X_test, y_train, y_test = train_test_split(

 x, y, test_size=0.2, random_state=10110)



#random forest classifier with n_estimators=10

clf_rf = RandomForestClassifier(random_state=10110)      

clr_rf = clf_rf.fit(X_train,y_train)



ac = accuracy_score(y_test,clf_rf.predict(X_test))

print('Accuracy is: ',ac)

cm = confusion_matrix(y_test,clf_rf.predict(X_test))

sns.heatmap(cm,annot=True,fmt="d")
# The "accuracy" scoring is proportional to the number of correct classifications

clf_rf_2 = RandomForestClassifier(random_state=10110, n_estimators=30) 

rfecv = RFECV(estimator=clf_rf_2, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation

rfecv = rfecv.fit(X_train, y_train)



print('Optimal number of features :', rfecv.n_features_)

print('Best features :', X_train.columns[rfecv.support_])
# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score of number of selected features")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
# Creating a Random Forest Classifier with the 4 best features and n_estimators = 30



# y includes our labels and x includes our features

dlist = ['DEATH_EVENT', 'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',

        'high_blood_pressure','serum_sodium', 'sex', 'smoking']

x_2 = heart_dat.drop(dlist,axis=1)

y_2 = heart_dat['DEATH_EVENT']



X_train, X_test, y_train, y_test = train_test_split(

 x_2, y_2, test_size=0.2, random_state=10110)



#random forest classifier with n_estimators=30

clf_rf_3 = RandomForestClassifier(random_state=10110, n_estimators=30)      

clr_rf_3 = clf_rf_3.fit(X_train,y_train)



ac = accuracy_score(y_test,clf_rf_3.predict(X_test))

print('Accuracy is: ',ac)

cm = confusion_matrix(y_test,clf_rf_3.predict(X_test))

sns.heatmap(cm,annot=True,fmt="d")
# Creating the final model, a Random Forest Classifier with the 4 best features and n_estimators = 30



# y includes our labels and x includes our features

dlist = ['DEATH_EVENT', 'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',

        'high_blood_pressure','serum_sodium', 'sex', 'smoking']

x_2 = heart_dat.drop(dlist,axis=1)

y_2 = heart_dat['DEATH_EVENT']





#random forest classifier with n_estimators=30

clf_rf_final = RandomForestClassifier(random_state=10110, n_estimators=30)      

clr_rf_final = clf_rf_final.fit(x_2,y_2)