# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print ("Dataset in your input directory:")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



def OneHotEncoding(df, cols):

    for col in cols:

        col_value_unique = df[col].unique()

        for col_value in col_value_unique:

            df[col+'_'+col_value] = np.where(df[col]==col_value, 1, 0)

    return df

print ("\nOneHotEncoding Available")
# Read input file

# display first 5 rows

linkname = '../input/bank-marketing/bank-additional-full.csv'

dataset_full = pd.read_csv(linkname, sep = ';')



dataset_full.head()
# Take a look at features

# category features: job, marital, education, default, housing, loan, month, day_of_week, poutcome

# numeric features: age, default, campaign, previous, euribor3m

# classification: y = yes/no



# ==================Input column name of the fearue you want to check ==================================



feature_name = 'marital'



# ======================================================================================================



pos_counts = dataset_full.loc[dataset_full.y.values == 'yes', feature_name].value_counts() 

neg_counts = dataset_full.loc[dataset_full.y.values == 'no', feature_name].value_counts()

    

all_counts = list(set(list(pos_counts.index) + list(neg_counts.index)))

    

#Counts of how often each outcome was recorded.

freq_pos = (dataset_full.y.values == 'yes').sum()

freq_neg = (dataset_full.y.values == 'no').sum()

    

pos_counts = pos_counts.to_dict()

neg_counts = neg_counts.to_dict()

    

all_index = list(all_counts)

all_counts = [pos_counts.get(k, 0) / freq_pos - neg_counts.get(k, 0) / freq_neg for k in all_counts]



sns.barplot(all_counts, all_index)

plt.title(feature_name)

plt.tight_layout()
# Choose and map features using OneHotCoding





# ==================Input column name of the fearue you selected ==================================



selected_ctg_feature = ['job', 'marital']

selected_num_feature = ['age']



# =================================================================================================



# Get full list of selected features and y

select_col = selected_ctg_feature

select_col = select_col + selected_num_feature

select_col.append("y")



# print(select_col)



# Transform categorial features using OneHotCoding

df_select = dataset_full.loc[:,select_col]

df_select = OneHotEncoding(df_select, selected_ctg_feature)

df_select = df_select.drop(columns = selected_ctg_feature)



# df_select.head()



df_select["y"] = df_select["y"].map({'yes': 1, 'no': 0})

df_label = df_select["y"]

df_feature = df_select.drop(columns = 'y')

label = df_label.astype('float64').values

feature = df_feature.astype('float64').values



print(label[0:5])

print(feature[0:5,:])

# Model selection and training



# ==================   1. SVM   ==================================



from sklearn import svm



# Train model

clf_svm = svm.SVC(gamma='scale')

clf_svm.fit(feature, label) 



# Prediction

predict_train = clf_svm.predict(feature)

predict_test = clf_svm.predict(feature)



# Accuracy output

accuracy_svm_train = np.mean(predict_train == label)

accuracy_svm_test = np.mean(predict_test == label)



print ("SVM Training Accuracy: ", accuracy_svm_train)

print ("SVM Testing Accuracy: ", accuracy_svm_test)



# Feature coefficient check