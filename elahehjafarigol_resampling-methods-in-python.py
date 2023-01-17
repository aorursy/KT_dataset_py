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
#Import libraries

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt



%matplotlib inline
#Load spreadsheet

file = pd.read_csv('../input/heartdisease/framingham_heart_disease.csv')



#Exploratory data analysis

print (file.shape)

print (file.info())
data = pd.DataFrame (file)



fill_feat = ["glucose", "education", "BPMeds", "totChol", "cigsPerDay", "BMI", "heartRate"]

for i in fill_feat: 

    data[i].fillna(np.mean(data[i]),inplace=True)
#Normalization using Sklearn

Data = np.asarray(data[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',

       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',

       'diaBP', 'BMI', 'heartRate', 'glucose']])

Target = data['TenYearCHD']



#Scaling the data before training

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler





Data = preprocessing.StandardScaler().fit(Data).transform(Data)

Data = pd.DataFrame (Data)

Data.columns = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',

       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',

       'diaBP', 'BMI', 'heartRate', 'glucose']

#print (Data.head())



Target = pd.DataFrame (Target)

Target.columns = ["label"]

#print(Target.head())



droplist = ['BPMeds', 'prevalentHyp','diabetes']

Data = Data.drop(droplist, axis = 1 )



X = Data

y = Target
target_count = Target.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')





count_class_N, count_class_P =Target.label.value_counts()



# Divide by class

Class_N = Target == 0

Class_P = Target == 1



#Plot the target

sb.countplot (x = Target.label, data = Data, palette = "bwr")

plt.show()
#Random under-sampling

from imblearn.under_sampling import RandomUnderSampler



rus = RandomUnderSampler()

X_rus, y_rus = rus.fit_resample(X, y)

X_res = pd.DataFrame(X_rus)

y_res = pd.DataFrame(y_rus)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Under-sampling based on the Condensed Nearest Neighbor

from imblearn.under_sampling import CondensedNearestNeighbour



con = CondensedNearestNeighbour()

X_con, y_con = con.fit_resample(X, y)

X_res = pd.DataFrame(X_con)

y_res = pd.DataFrame(y_con)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Under-sampling based on the Edited Nearest Neighbor

from imblearn.under_sampling import EditedNearestNeighbours



enn = EditedNearestNeighbours()

X_enn, y_enn = enn.fit_resample(X, y)

X_res = pd.DataFrame(X_enn)

y_res = pd.DataFrame(y_enn)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Under-sampling based on the Repeated Edited Nearest Neighbor

from imblearn.under_sampling import RepeatedEditedNearestNeighbours



renn = RepeatedEditedNearestNeighbours()

X_renn, y_renn = renn.fit_resample(X, y)

X_res = pd.DataFrame(X_renn)

y_res = pd.DataFrame(y_renn)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Under-sampling using the cluster centroids

from imblearn.under_sampling import ClusterCentroids



cc = ClusterCentroids(sampling_strategy = 'majority')

X_cc, y_cc = cc.fit_sample(X, y)

X_res = pd.DataFrame(X_cc)

y_res = pd.DataFrame(y_cc)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Under-sampling based on one-sided selection

from imblearn.under_sampling import OneSidedSelection



os = OneSidedSelection()

X_os, y_os = os.fit_sample(X, y)

X_res = pd.DataFrame(X_os)

y_res = pd.DataFrame(y_os)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Under-sampling based on removing the Tomek links

from imblearn.under_sampling import TomekLinks



tomek = TomekLinks()

X_tl, y_tl = os.fit_sample(X, y)

X_res = pd.DataFrame(X_tl)

y_res = pd.DataFrame(y_tl)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn import model_selection

from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics

from sklearn.metrics import classification_report



from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier as KNN





#Create a list of sampling methods

models = []

models.append(("Random Under-sampling", rus))

models.append(("Condensed Nearest Neighbor", con))

models.append(("Edited Nearest Neighbor", enn))

models.append(("Repeated Edited Nearest Neighbor", renn))

models.append(("Cluster Centroids", cc))

models.append(("One-sided Selection", os))

models.append(("Tomek Links", tomek))



#Evaluate each model in turn



def Evaluate(MODEL):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

    knn = KNN()

    knn.fit(X_train, y_train.values.ravel()) 

    #Making predictions

    y_pred = knn.predict(X_test) 

    #print(classification_report(y_test, y_pred))

    Accuracy = metrics.accuracy_score(y_test, y_pred)

    print("Accuracy:", Accuracy)



for i in models:

    print ("Model:")  

    print (i)

    Evaluate(i)

    print ("******************************************")  
#Random over-samplong 

from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X, y)

X_res = pd.DataFrame(X_ros)

y_res = pd.DataFrame(y_ros)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Over-samplong using Adaptive Synthetic (ADASYN) sampling approach

from imblearn.over_sampling import ADASYN



ad = ADASYN()

X_ad, y_ad = ad.fit_sample(X, y)

X_res = pd.DataFrame(X_ad)

y_res = pd.DataFrame(y_ad)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Over-samplong using Synthetic Minority Over-sampling Technique (SMOTE)

from imblearn.over_sampling import SMOTE



sm = SMOTE()

X_sm, y_sm = smote.fit_sample(X, y)

X_res = pd.DataFrame(X_sm)

y_res = pd.DataFrame(y_sm)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Over-samplong using Synthetic Minority Over-sampling Technique (SMOTE) with SVM algorithm

from imblearn.over_sampling import SVMSMOTE



svmsm = SVMSMOTE()

X_sm, y_sm = svmsm.fit_sample(X, y)

X_res = pd.DataFrame(X_sm)

y_res = pd.DataFrame(y_sm)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Create a list of sampling methods

models = []

models.append(("Random Over-sampling", ros))

models.append(("Adaptive Synthetic (ADASYN)", ad))

models.append(("Synthetic Minority Technique (SMOTE)", sm))

models.append(("SMOTE with SVM", svmsm))



#Evaluate each model in turn



def Evaluate(MODEL):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

    knn = KNN()

    knn.fit(X_train, y_train.values.ravel()) 

    #Making predictions

    y_pred = knn.predict(X_test) 

    #print(classification_report(y_test, y_pred))

    Accuracy = metrics.accuracy_score(y_test, y_pred)

    print("Accuracy:", Accuracy)



for i in models:

    print ("Model:")  

    print (i)

    Evaluate(i)

    print ("******************************************")  
#Combination of ver-sampling and under-sampling using SMOTE and cleaning using Edited Nearest Neighbor.

from imblearn.combine import SMOTEENN



smenn = SMOTEENN()

X_smenn, y_smenn = smenn.fit_sample(X, y)

X_res = pd.DataFrame(X_sm)

y_res = pd.DataFrame(y_sm)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Combination of ver-sampling and under-sampling using SMOTE and cleaning using Tomek links.

from imblearn.combine import SMOTETomek



smtomek = SMOTETomek()

X_smtomek, y_smtomek = smtomek.fit_sample(X, y)

X_res = pd.DataFrame(X_sm)

y_res = pd.DataFrame(y_sm)



New_data = pd.concat([X_res, y_res], axis=0)



target_count = New_data.label.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



count_class_N, count_class_P = New_data.label.value_counts()



# Divide by class

Class_N = New_data.label == 0

Class_P = New_data.label == 1



#Plot the target

sb.countplot (x = New_data.label, data = New_data, palette = "bwr")

plt.show()
#Create a list of sampling methods

models = []

models.append(("SMOTE and cleaning using Edited Nearest Neighbor", smenn))

models.append(("SMOTE and cleaning using Tomek links", smtomek))



#Evaluate each model in turn



def Evaluate(MODEL):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

    knn = KNN()

    knn.fit(X_train, y_train.values.ravel()) 

    #Making predictions

    y_pred = knn.predict(X_test) 

    #print(classification_report(y_test, y_pred))

    Accuracy = metrics.accuracy_score(y_test, y_pred)

    print("Accuracy:", Accuracy)



for i in models:

    print ("Model:")  

    print (i)

    Evaluate(i)

    print ("******************************************") 