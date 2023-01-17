# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import sklearn.decomposition as skde

import sklearn.model_selection as ms

from sklearn import linear_model

import sklearn.metrics as sklm

import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test_dataset = pd.read_csv("../input/test.csv")

train_dataset = pd.read_csv("../input/train.csv")

# train_dataset.corr()

#test_dataset.corr()

# test_dataset.info()

train_dataset.isnull().values.any() #Check if there are null values in dataset

train_dataset["target"].value_counts() #Test to check for class imbalance

sns.countplot(train_dataset["target"]) #visualize class distribution

#split into input and label

labels = train_dataset["target"]

new_train_dataset = train_dataset.drop(["target","ID_code"],axis =1)

new_train_dataset.head()



#select important features

from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators=100)

selected_features = SelectFromModel(clf)

selected_features.fit(new_train_dataset,labels.values.ravel())

list_sel= new_train_dataset.columns[(selected_features.get_support())]

new_train = pd.DataFrame(data = new_train_dataset,columns = list_sel)

new_train.head()

# check class distribution in percentage

count_0 = len(train_dataset[train_dataset["target"] == 0])

count_1 = len(train_dataset[train_dataset["target"] == 1])

percentage_count_0 = ((count_0)/(count_0+count_1)) * 100

percentage_count_1 = 100-percentage_count_0

print("{}{}{}{}{}".format("Percentage of 0 class is ",percentage_count_0,"\n","Percentage of 1 class is ",percentage_count_1))

#use SMOTE technique to take care of the class imbalance

os = SMOTE(random_state=0) #   We are using SMOTE as the function for oversampling

os_data_X,os_data_y=os.fit_sample(new_train,labels)

os_data_X = pd.DataFrame(data=os_data_X,columns=new_train.columns)

os_data_y= pd.DataFrame(data=os_data_y,columns=["target"])





print("length of oversampled data is ",len(os_data_X))

print("Number of 0 class in oversampled data",len(os_data_y[os_data_y["target"]==0]))

print("Number of 1 class in oversampled data",len(os_data_y[os_data_y["target"]==1]))

print("Proportion of 0 class in oversampled data is ",len(os_data_y[os_data_y["target"]==0])/len(os_data_X))

print("Proportion of 1 class in oversampled data is ",len(os_data_y[os_data_y["target"]==1])/len(os_data_X))



os_data_X.shape,os_data_y.shape

#check if therea are highly correlated features

os_data_X.corr()
#scale dataset

# scaler = preprocessing.StandardScaler()

# scaled_data = scaler.fit_transform(os_data_X)

# scaled_data = pd.DataFrame(data = scaled_data,columns = os_data_X.columns)

# scaled_data.head()

os_data_X.head()
#split data into test and train set

x_train, x_test, y_train, y_test = train_test_split(os_data_X, os_data_y, test_size = 0.25, random_state = 0)

x_train.head()
#convert dataset to lightgbm format for lightgbm training

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

train_set = lgb.Dataset(x_train, label=y_train)

train_eval = lgb.Dataset(x_test, y_test, reference=train_set)

params = {}

params['learning_rate'] = 0.001

params['boosting_type'] = 'rf'

params['objective'] = 'binary'

params['metric'] = 'binary_error'

params['sub_feature'] = 0.5

params['min_data'] = 50

params['max_depth'] = 8

params['bagging_freq'] =  5

params['bagging_fraction'] = 0.4

params['feature_fraction'] = 0.05

params['num_leaves'] = 256

params['task'] = 'train'

params['min_data_in_leaf'] = 100

params['max_bin'] = 120

params['num_iteration']=150

params['verbose'] = 1

# clf = lgb.train(params,train_set, early_stopping_rounds=15,valid_sets=[train_set,train_eval],

#             valid_names=['train', 'eval'],)



# #Prediction

# y_pred=clf.predict(x_test)

# #convert into binary values

# for i in range(0,89951):

#     if y_pred[i]>=.5:       # setting threshold to .5

#        y_pred[i]=1

#     else:  

#        y_pred[i]=0



folds = StratifiedKFold(n_splits=10)

oof_preds = np.zeros(x_train.shape[0])

sub_preds = np.zeros(x_test.shape[0])



model = lgb.LGBMRegressor(**params, n_estimators=100000)

model.fit(x_train, y_train, eval_set=[(x_test,y_test)], early_stopping_rounds=3000, verbose=1000)



# oof_preds[val_] = model.predict(x_test,num_iteration=model.best_iteration_)

sub_preds += model.predict(x_test, num_iteration=model.best_iteration_)

np.max(sub_preds)

for i in range(0,89951):

    if sub_preds[i]>=.5:       # setting threshold to .5

       sub_preds[i]=1

    else:  

        sub_preds[i]=0
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, sub_preds)

cm

from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report

accuracy=accuracy_score(sub_preds,y_test)

precision = precision_score(sub_preds,y_test)

recall = recall_score(sub_preds,y_test)

accuracy,precision,recall
scaler = preprocessing.StandardScaler()

ID_code = test_dataset["ID_code"]

TEST = pd.DataFrame(data = test_dataset,columns = list_sel)



scaled_TEST = scaler.fit_transform(TEST)

scaled_TEST = pd.DataFrame(data = scaled_TEST,columns = list_sel)

prediction = np.zeros(scaled_TEST.shape[0])

print(scaled_TEST.head())

prediction += model.predict(scaled_TEST,num_iteration=model.best_iteration_)

#submission

submission = pd.DataFrame({'ID_code' : ID_code,

                            'target' : prediction})

submission['target'] = submission['target'].apply(lambda x : 1 if x > 0.5 else 0)

submission.to_csv('./version2.csv', index=False)

sub = pd.read_csv('./version2.csv')

sub['target'].unique()
