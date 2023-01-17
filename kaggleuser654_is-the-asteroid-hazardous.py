# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break

    



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#load data

asteroids_data = pd.read_csv('/kaggle/input/nasa-asteroids-classification/nasa.csv')
asteroids_data.head()
#binarising Hazardous

asteroids_data['Hazardous'].replace({True: 1, False: 0}, inplace = True)
asteroids_data['Hazardous'].value_counts()
#list of features

features = asteroids_data.columns.tolist()

features = features[0:-1]

#check

print(features)
asteroids_data.info()
asteroids_data[['Close Approach Date', 'Orbiting Body','Orbit Determination Date', 'Equinox']].head()
#unique values

print('Orbiting Body unique values: {}'.format(asteroids_data['Orbiting Body'].unique()))

print('Equinox unique values: {}'.format(asteroids_data['Equinox'].unique()))
for f in ['Neo Reference ID', 'Name', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 

          'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Close Approach Date', 'Epoch Date Close Approach', 'Relative Velocity km per sec', 

          'Miles per hour', 'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 'Miss Dist.(miles)', 'Orbiting Body', 'Orbit ID', 'Orbit Determination Date', 'Epoch Osculation', 'Equinox']:

    features.remove(f)
print('There are now {} features.'.format(len(features)))
from sklearn.model_selection import train_test_split 



#split data into train, cross validation, and test sets

asteroid_set, asteroid_test = train_test_split(asteroids_data, test_size=0.2, random_state=7)

asteroid_train, asteroid_cv = train_test_split(asteroid_set, test_size=0.25, random_state=5)
#pairplot to see relationship between 'Minimum Orbit Intersection', 'Absolute Magnitude', and 'Hazardous'

sns.pairplot(asteroid_train[['Minimum Orbit Intersection', 'Absolute Magnitude', 'Hazardous' ]], diag_kind = 'hist', hue='Hazardous', palette = {1:'red', 0:'blue'})

plt.figure(figsize=(10,10))

sns.scatterplot(asteroid_train['Minimum Orbit Intersection'], asteroid_train['Absolute Magnitude'], hue = asteroid_train['Hazardous'], palette={1: 'red', 0:'blue'})

plt.show()
asteroid_train[['Minimum Orbit Intersection','Absolute Magnitude']].describe()
from sklearn.preprocessing import robust_scale



for col in ['Absolute Magnitude', 'Minimum Orbit Intersection']:

    asteroid_train[col + "_scaled"] = robust_scale(asteroid_train[col])

    asteroid_cv[col + "_scaled"] = robust_scale(asteroid_cv[col])

    asteroid_test[col + "_scaled"] = robust_scale(asteroid_test[col])

    features.append(col + '_scaled')

    features.remove(col)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import scikitplot as skplt
lr = LogisticRegression()

lr.fit(asteroid_train[['Absolute Magnitude_scaled','Minimum Orbit Intersection_scaled']], asteroid_train['Hazardous'])

predictions = lr.predict(asteroid_cv[['Absolute Magnitude_scaled','Minimum Orbit Intersection_scaled']])
from sklearn.metrics import confusion_matrix

import scikitplot as skplt

from sklearn.metrics import accuracy_score





def evaluation(test_y, predictions):

    

    #accuracy score

    accuracy = accuracy_score(test_y, predictions)

    print("The classification accuracy is {:.2f} %." .format(accuracy*100))

    

  

    y_test_mean = test_y.mean()

    #null accuracy

    null_accuracy = max(y_test_mean, 1-y_test_mean)

    print('The null accuracy is {:.2f} %.'.format(null_accuracy*100))

    

    #confusion matrix

    skplt.metrics.plot_confusion_matrix(test_y, predictions)

    

    conf_matrix = confusion_matrix(test_y, predictions)

    

    TN = conf_matrix[0,0] #true negatives

    FP = conf_matrix[0,1] #false positives

    FN = conf_matrix[1,0] #false negatives

    TP = conf_matrix[1,1] #true positives

    

    #precision

    precision = TP/(TP+FP)*100

    print('The precision is {:.2f} %.'.format(precision))

    #sensitivity/ recall

    recall = TP/(FN+TP)*100

    print('The sensitivity/recall is {:.2f} %.'.format(recall))

    #specificity

    specificity = TN/(FP+TN)*100

    print('The specificity is {:.2f} %.'.format(specificity))

    #F_score

    F_score = (2*precision*recall)/(precision + recall)

    print('The F score is {:.2f} %.'.format(F_score))

    

    return None
evaluation(asteroid_cv['Hazardous'], predictions)


from sklearn.svm import SVC



svm = SVC(C=1.5, kernel='rbf')

svm.fit(asteroid_train[['Absolute Magnitude_scaled','Minimum Orbit Intersection_scaled']], asteroid_train['Hazardous'])

svm_pred = svm.predict(asteroid_cv[['Absolute Magnitude_scaled','Minimum Orbit Intersection_scaled']])
evaluation(asteroid_cv['Hazardous'], svm_pred)
#test set predictions

svm_pred = svm.predict(asteroid_test[['Absolute Magnitude_scaled','Minimum Orbit Intersection_scaled']])

evaluation(asteroid_test['Hazardous'], svm_pred)
X1 = asteroid_test['Absolute Magnitude_scaled']

X2 = asteroid_test['Minimum Orbit Intersection_scaled']

#create meshgrid for contour plot

x1, x2 = np.meshgrid(np.arange(X1.min() - 0.25, X1.max() + 0.25, 0.01), 

                     np.arange(X2.min() - 0.25, X2.max() + 0.25, 0.01))
L = lr.predict(np.c_[x1.ravel(), x2.ravel()])

L = L.reshape(x1.shape)

#plot boundary and points

plt.figure(figsize=(12,10))

plt.contourf(x2, x1, L, cmap=plt.cm.twilight, alpha=0.8)

#plt.scatter(asteroid_test['Minimum Orbit Intersection_scaled'],asteroid_test['Absolute Magnitude_scaled'], 

          # cmap=plt.cm.coolwarm, s=20, edgecolors='k')

sns.scatterplot(asteroid_test['Minimum Orbit Intersection_scaled'],asteroid_test['Absolute Magnitude_scaled'], hue=asteroid_test['Hazardous'],  palette = {1:'red', 0:'blue'})

plt.show()
Z = svm.predict(np.c_[x1.ravel(), x2.ravel()])

Z = Z.reshape(x1.shape)

#plot boundary and points

plt.figure(figsize=(12,10))

plt.contourf(x2, x1, Z, cmap=plt.cm.twilight, alpha=0.8)

#plt.scatter(asteroid_test['Minimum Orbit Intersection_scaled'],asteroid_test['Absolute Magnitude_scaled'], 

          # cmap=plt.cm.coolwarm, s=20, edgecolors='k')

sns.scatterplot(asteroid_test['Minimum Orbit Intersection_scaled'],asteroid_test['Absolute Magnitude_scaled'], hue=asteroid_test['Hazardous'],  palette = {1:'red', 0:'blue'})

plt.show()
#train and test Logistic regression model with polynomial features



from sklearn.preprocessing import PolynomialFeatures



for i in [2,3]:

    poly = PolynomialFeatures(i)

    X = poly.fit_transform(asteroid_train[['Absolute Magnitude_scaled','Minimum Orbit Intersection_scaled']])

    

    lr_poly = LogisticRegression()

    lr_poly.fit(X, asteroid_train['Hazardous'])

    X_cv = poly.fit_transform(asteroid_cv[['Absolute Magnitude_scaled','Minimum Orbit Intersection_scaled']])

    pred_poly = lr_poly.predict(X_cv)

    

    print('Logistic Regression with polynomial features degree {} scores: \n'.format(i))

    evaluation(asteroid_cv['Hazardous'], pred_poly)

    print('\n')
