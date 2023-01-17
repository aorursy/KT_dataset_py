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



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
train_data=np.load('/kaggle/input/eval-lab-4-f464/train.npy', allow_pickle=True)

print(train_data)
labels = train_data[:, 0]

images = train_data[:, 1]
trial_train_l=pd.DataFrame(labels)

trial_train_images=pd.DataFrame(images)

print(trial_train_l.head())

print(trial_train_images.head())
#print(images)

print(labels)
flatten_train = images.copy()

for i in range(len(flatten_train)):

    flatten_train[i] = flatten_train[i].flatten()
print(flatten_train)

print(flatten_train[0])
print(np.unique(labels))
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

l2 = labelencoder.fit_transform(labels)

print(len(flatten_train))

print(len(labels))

image_train_without_pca=np.vstack(flatten_train)



number_of_components = 120



from sklearn.decomposition import PCA

pca = PCA(n_components=number_of_components, svd_solver='randomized',whiten=True).fit(image_train_without_pca)

X_train = pca.transform(image_train_without_pca)



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC

#from xgboost.sklearn import XGBClassifier

#TODO

clf = SVC()       #Initialize the classifier object

#train_x = X_train.values

#parameters = {'n_estimators':[10,50,100,150,200]}    #Dictionary of parameters

#parameters = {'n_estimators':[1000,1500,2000,2500,3000,3500,4000],'min_samples_leaf':[1,2],'max_features':['sqrt'],'min_samples_split':[2],'max_depth':[25,30],'bootstrap':[False]}

#scorer = make_scorer(accuracy_score, greater_is_better = True)#Initialize the scorer using make_scorer

parameters = {'C': [10,20],  

              'gamma': [0.005], 'class_weight': [None,'balanced'],

              'kernel': ['rbf'],'degree':[3]}

grid_obj = GridSearchCV(clf,parameters)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,labels)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(best_clf)

#unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

#ptimized_predictions = best_clf.predict(img_flt)        #Same, but use the best estimator



#acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

#acc_opd = accuracy_score(labels, optimized_predictions)*100         #Calculate accuracy for optimized model



#print("Accuracy score on unoptimized model:{}".format(acc_unop))

#print("Accuracy score on optimized model:{}".format(acc_opd))

sv_best=best_clf
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBClassifier

from xgboost.sklearn import XGBClassifier

#TODO

clf = XGBClassifier()       #Initialize the classifier object

#train_x = X_train.values

#parameters = {'n_estimators':[10,50,100,150,200]}    #Dictionary of parameters

parameters = {

         'colsample_bytree': [0.8,1.0],

         'max_depth': [3],

         'n_estimators':[100]

         }

scorer = make_scorer(accuracy_score, greater_is_better = True)#Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,labels)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(best_clf)

# #unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

# #optimized_predictions = best_clf.predict(X_train)        #Same, but use the best estimator



# #acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

# #acc_opd = accuracy_score(y_train, optimized_predictions)*100         #Calculate accuracy for optimized model



# #print("Accuracy score on unoptimized model:{}".format(acc_unop))

# #print("Accuracy score on optimized model:{}".format(acc_opd))

xg_best=best_clf
# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import make_scorer

# from sklearn.metrics import accuracy_score

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import ExtraTreesClassifier

# #from xgboost.sklearn import XGBClassifier

# #TODO

# clf = RandomForestClassifier()       #Initialize the classifier object

# #train_x = X_train.values

# #parameters = {'n_estimators':[10,50,100,150,200]}    #Dictionary of parameters

# parameters = {'n_estimators':[800,950,1000,1100,1200],'min_samples_leaf':[1,2],'max_features':['sqrt'],'min_samples_split':[2],'bootstrap':[False,True]}

# scorer = make_scorer(accuracy_score, greater_is_better = True)#Initialize the scorer using make_scorer



# grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



# grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



# best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

# print(best_clf)

# #unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

# optimized_predictions = best_clf.predict(X_train)        #Same, but use the best estimator



# #acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

# acc_opd = accuracy_score(y_train, optimized_predictions)*100         #Calculate accuracy for optimized model



# #print("Accuracy score on unoptimized model:{}".format(acc_unop))

# #print("Accuracy score on optimized model:{}".format(acc_opd))

# # rf_best=best_clf
# from sklearn.ensemble import VotingClassifier, RandomForestClassifier

# estimators=[('xgb', xg_best),('rf',rf_best),('et',sv_best)]

# eclf1 =VotingClassifier(estimators,voting='soft')

# eclf1=eclf1.fit(X_train,y_train)
test_data=np.load('/kaggle/input/eval-lab-4-f464/test.npy', allow_pickle=True)
#test_ids = img_test[:, 0]

test_id = test_data[:, 0]

test_images = test_data[:, 1]
print(test_id)
flatten_images_test = test_images.copy()

for i in range(len(flatten_images_test)):

    flatten_images_test[i] = flatten_images_test[i].flatten()
image_test_without_pca=np.vstack(flatten_images_test)
X_test = pca.transform(image_test_without_pca)

#print(X_test)
y1=sv_best.predict(X_test)

y2=xg_best.predict(X_test)
df1 = pd.DataFrame({"ImageId": test_id, "Celebrity": y1})

df1.to_csv('l4_1.csv',index=False)

df2 = pd.DataFrame({"ImageId": test_id, "Celebrity": y2})

df2.to_csv('l4_2.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data_x.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe





# create a link to download the dataframe

create_download_link(df1)

#create_download_link(df2)
