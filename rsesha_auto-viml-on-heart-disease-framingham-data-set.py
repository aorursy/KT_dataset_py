#!pip install autoviml
### You can get the latest version from the Github
!pip install git+https://github.com/AutoViML/Auto_ViML.git
from autoviml.Auto_ViML import Auto_ViML
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
# read data
df = pd.read_csv('/kaggle/input/framingham-heart-study-dataset/framingham.csv')

# first glimpse at data
df.head(20)

# data shape
df.shape

# data types
df.dtypes
target = 'TenYearCHD'
# clarify what is y and what is x label
y = df[target]
X = df.drop([target], axis = 1)
from sklearn.model_selection import train_test_split
# divide train test: 80 % - 20 %
train, test = train_test_split(df, test_size = 0.2, random_state=29)
print(len(train))
len(test)
m, feats, trainm, testm = Auto_ViML(train, target, test,
                            sample_submission='',
                            scoring_parameter='', KMeans_Featurizer=False,
                            hyper_param='RS',feature_reduction=True,
                             Boosting_Flag=False, Binning_Flag=False,
                            Add_Poly=0, Stacking_Flag=False,Imbalanced_Flag=True,
                            verbose=1)
### Test the Auto_ViML Model
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
model_name = 'Auto_ViML'
# prediction = knn.predict(x_test)
normalized_df_knn_pred = testm[target+'_predictions']
y_test = test[target].values

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_knn_pred)
print(f"The accuracy score for {model_name} is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(y_test, normalized_df_knn_pred)
print(f"The f1 score for {model_name} is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_knn_pred)
print(f"The precision score for {model_name} is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_knn_pred)
print(f"The recall score for {model_name} is: {round(recall,3)*100}%")
# Check overfit of the Auto_ViML Random Forests model
# accuracy test and train
X_test = testm[feats].values
y_train = trainm[target].values
X_train = trainm[feats].values
acc_test = m.score(X_test, y_test)
print("The accuracy score of the test data is: ",acc_test*100,"%")
acc_train = m.score(X_train, y_train)
print("The accuracy score of the training data is: ",round(acc_train*100,2),"%")


