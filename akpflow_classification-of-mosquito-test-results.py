# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as st

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from mlxtend.plotting import plot_confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score



pd.set_option('display.max_rows', 10)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
wnvData = pd.read_csv("../input/west-nile-virus-wnv-mosquito-test-results.csv")
wnvData.dtypes
wnvData.head()
wnvData.shape
wnvData.describe()
wnvData.info()
wnvData.isnull().values.any()

wnvData.isnull().sum()
wnvData.columns.values
s = (wnvData.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)




wnvData.drop( wnvData.columns [[3,12, 4, 6]], axis=1, inplace=True)



#wnvData.drop( wnvData.columns ["BLOCK, "LOCATION", 'TRAP', 'TEST DATE'], axis=1, inplace=True)
s = (wnvData.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
wnvData.head()
from sklearn import preprocessing 

le = preprocessing.LabelEncoder()



le.fit(wnvData['RESULT'])

wnvData['RESULT'] = le.transform(wnvData['RESULT'])



le.fit(wnvData['TRAP_TYPE'])

wnvData['TRAP_TYPE'] = le.transform(wnvData['TRAP_TYPE'])



le.fit(wnvData['SPECIES'])

wnvData['SPECIES'] = le.transform(wnvData['SPECIES'])

def plot_corr (wnvData, size =14):

    corr =wnvData.corr()

    fig, ax= plt.subplots(figsize =(size,size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns)     #draw x tick marks

    plt.yticks(range(len(corr.columns)), corr.columns) 
plot_corr(wnvData)
wnvData.drop( wnvData.columns [[2]], axis=1, inplace=True)
plot_corr(wnvData)
feature_col_name = [ 'SEASON YEAR', 'WEEK', 'SPECIES', 'TRAP_TYPE',

       'NUMBER OF MOSQUITOES', 'Wards', 'Census Tracts', 

       'Community Areas', 'Historical Wards 2003-2015']

predicted_class_name= [ 'RESULT']



X = wnvData[feature_col_name].values

y = wnvData[predicted_class_name].values

split_test_size = 0.1



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split_test_size, random_state=42)
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

X_train = my_imputer.fit_transform(X_train)

X_test = my_imputer.transform(X_test)
wnvData.head()
wnvData['RESULT'].value_counts()
# Writing a function to automate fitting the classifiers and evaluate the algorithms



def classifier(model,train_independent,train_dependent,test_independent,true):

    model.fit(train_independent,train_dependent)

    prediction = model.predict(X_test)

    print(classification_report(true,prediction))

    

    # Confusion Matrix plot

    

    cm = confusion_matrix(y_test,prediction)

    fig= plot_confusion_matrix(conf_mat=cm,figsize=(4,4),cmap=plt.cm.Reds,hide_spines=True)

    plt.title('Confusion Matrix',fontsize=14)

    plt.xlabel('Predicted Values')

    plt.ylabel('Actual Values')

    plt.grid('off')

    plt.show()





    # 10-fold Cross Validation

    accuracies = cross_val_score(estimator= model,X= X_train,y=y_train,cv=10)

    print("The average model accuracy score is : %s" % "{0:.2%}".format(accuracies.mean()))

    print("The average accuracy score standard deviation is : %s" % "{0:.3%}".format(accuracies.std()))

    

    # Values of the ROC Curve as a probabilistic approach to classification

    roc_predict = model.predict_proba(X_test)

    roc_predict = [p[1] for p in roc_predict]

    area = roc_auc_score(y_test,roc_predict)

    float(area)

    print ("The area under the Reciver Operating Characteristic curve is: ", (round(area,2)))


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

classifier(log_reg,X_train,y_train,X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

classifier (dtree,X_train,y_train,X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50)

classifier(rfc,X_train,y_train,X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=20)

classifier(knn, X_train,y_train,X_test,y_test)


# Importing the libraries for the XGBoost algorithm



from xgboost import XGBClassifier

xgb_classifier = XGBClassifier()

classifier(xgb_classifier,X_train,y_train,X_test,y_test)


Model_Scores ={'Logistic Regression':{'10 Fold Cross Validation Score':

                                      "{0:.2%}".format((cross_val_score(estimator= log_reg,X= X_train,y=y_train,cv=10)).mean()), 

                                      'Standard Deviation':"{0:.2%}".format((cross_val_score(estimator= log_reg,X= X_train,y=y_train,cv=10)).std())},

                'Decision Trees':{'10 Fold Cross Validation Score':"{0:.2%}".format((cross_val_score(estimator= dtree,X= X_train,y=y_train,cv=10)).mean()),

                                     'Standard Deviation':"{0:.2%}".format(((cross_val_score(estimator= dtree,X= X_train,y=y_train,cv=10)).std()))},

               'Random Forest':{'10 Fold Cross Validation Score':"74.86%", 

                                 'Standard Deviation':"5.85%"},

               'K Nearest Neighbors':{'10 Fold Cross Validation Score':"{0:.2%}".format((cross_val_score(estimator= knn,X= X_train,y=y_train,cv=10)).mean()), 

                                       'Standard Deviation':"{0:.2%}".format(((cross_val_score(estimator= knn,X= X_train,y=y_train,cv=10)).std()))},

               'XGBoost':{'10 Fold Cross Validation Score':"{0:.2%}".format((cross_val_score(estimator= xgb_classifier,X= X_train,y=y_train,cv=10)).mean()), 

                           'Standard Deviation':"{0:.2%}".format(((cross_val_score(estimator= xgb_classifier,X= X_train,y=y_train,cv=10)).std()))}

              }

Model_Scores= pd.DataFrame(Model_Scores)

Model_Scores