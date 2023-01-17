# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
glass_df = pd.read_csv('/kaggle/input/glass/glass.csv')

print(glass_df.info())
glass_types = glass_df['Type'].unique()

print(glass_types)
print(glass_df['Type'].value_counts())

print(glass_df['Type'].value_counts(normalize=True) * 100)
glass_df_na = glass_df.isna()

print(len(glass_df))

for col in glass_df.columns:

    print(glass_df_na[col].value_counts())
X_1 = glass_df[glass_df.columns[:-1]]

y_1 = glass_df['Type']
from sklearn.model_selection import train_test_split

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=123)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
log_reg_model_1 = log_reg.fit(X_train_1, y_train_1)
log_reg_model_1.score(X_test_1, y_test_1)
y_pred_1 = log_reg_model_1.predict(X_test_1)
from sklearn.metrics import confusion_matrix

cf_matrix_1 = confusion_matrix(y_test_1, y_pred_1)

print(cf_matrix_1)
df_cm = pd.DataFrame(cf_matrix_1, index = glass_types,

                  columns = glass_types)

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)
X_2 = glass_df[glass_df.columns[:-1]]

y_2 = glass_df['Type'].apply(lambda x : 1 if x == 7 else 0)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.3, random_state=123)
print(y_train_2.value_counts())

print(y_test_2.value_counts())
log_reg_bin = LogisticRegression()

log_reg_binary_model = log_reg_bin.fit(X_train_2, y_train_2)
from sklearn.metrics import recall_score, precision_score, accuracy_score

y_train_pred = log_reg_binary_model.predict(X_train_2)

y_test_pred = log_reg_binary_model.predict(X_test_2)





# print("The Confusion matrix for Training for class :", i) 

# Confusion matrix 

confusion = confusion_matrix(y_train_2,y_train_pred )

print('Confusion matrix: \n',confusion)

print('Accuracy_Score:',accuracy_score(y_train_2,y_train_pred))

print('Recall_Score:',recall_score(y_train_2,y_train_pred))

print('Precision_Score:',precision_score(y_train_2,y_train_pred))






# print("The Confusion matrix for Training for class :", i) 

# Confusion matrix 

confusion = confusion_matrix(y_test_2,y_test_pred )

print('Confusion matrix: \n',confusion)

print('Accuracy_Score:',accuracy_score(y_test_2,y_test_pred))

print('Recall_Score:',recall_score(y_test_2,y_test_pred))

print('Precision_Score:',precision_score(y_test_2,y_test_pred))
y_pred_2 = log_reg_binary_model.predict(X_test_2)

cf_matrix_2 = confusion_matrix(y_test_2, y_pred_2)
df_cm_2 = pd.DataFrame(cf_matrix_2, index = [0,1],

                  columns = [0, 1])

plt.figure(figsize = (10,7))

sns.heatmap(df_cm_2, annot=True)