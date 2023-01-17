# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importing mobile price train dataset

df = pd.read_csv('../input/train.csv')
#Data Exploration

print("Dataframe size: ", df.shape)
df.head()
#This is classification problem as mobile phone are seperated as per price range

#Calculating unique values of price_range class

df.price_range.unique()
#Checking if train data has any null value

if(df.isnull().sum().sum() == 0):

    print('Dataframe does not have any null values')
#Considering all available features for decision tree classifier

features = ['battery_power','blue','clock_speed','dual_sim','fc','four_g','int_memory','m_dep','mobile_wt','n_cores',

                   'pc','px_height','px_width','ram','sc_h','sc_w','talk_time','three_g','touch_screen','wifi']

X = df[features]

y = df['price_range']
# Decision tree classifier and model evaluation using kFold cross validation



from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold



kfold_mae_train=0

kfold_mae_test=0

kfold_f_imp_dic = 0



no_of_folds = 5



kf = KFold(no_of_folds,True,1)



for train_index, test_index in kf.split(X):

    

    X_train,X_test = X.iloc[train_index],X.iloc[test_index]

    y_train,y_test = y.iloc[train_index],y.iloc[test_index]

    

    dt_classifier = DecisionTreeClassifier(random_state=1)

    dt_classifier.fit(X_train,y_train)

    

    mae_train = mean_absolute_error(dt_classifier.predict(X_train),y_train)

    kfold_mae_train=(kfold_mae_train+mae_train)

    

    mae_test = mean_absolute_error(dt_classifier.predict(X_test),y_test)

    kfold_dt_mae_test = (kfold_mae_test+mae_test)

    

    kfold_f_imp_dic = kfold_f_imp_dic + dt_classifier.feature_importances_

    

print('Decision Tree Regressor train set mean absolute error =',kfold_mae_train/no_of_folds)

print('Decision Tree Regressor test set mean absolute error  =',kfold_dt_mae_test/no_of_folds)



# Feature importance in decision treee

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure



f_importance_dic = dict(zip(features,kfold_f_imp_dic/no_of_folds))

df_imp_features = pd.DataFrame(list(f_importance_dic.items()),columns=['feature','score'])



plt.figure(figsize=(30,10))

plt.bar(df_imp_features['feature'], df_imp_features['score'],color='green',align='center', alpha=0.5)

plt.xlabel('Mobile features', fontsize=20)

plt.ylabel('Relative feature score',fontsize=20)

plt.title('Relative Feature importance in determining price',fontsize=30)
#Prediction on test data

df_test = pd.read_csv('../input/test.csv')

df_test['p_price_range'] = dt_classifier.predict(df_test[features])

df_test.head()