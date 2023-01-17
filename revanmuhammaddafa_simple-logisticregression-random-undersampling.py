import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

warnings.filterwarnings("ignore")





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/creditcardfraud/creditcard.csv',sep=',')
df.head()
df.describe()
##mengetahui jumlah value dari tiap label pada kelas di kolom

df["Class"].value_counts()
df.hist(bins=50, figsize=(20,15))

plt.show()
count_classes = pd.value_counts(df['Class'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
##using standard scaler

from sklearn.preprocessing import StandardScaler

df['norm_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df.head()
##Menghapus kolom pada tabel

df= df.drop(columns="Amount")
df.head()
corr = df.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(corr,vmax=0.8,square = True)

plt.show()


# Lets shuffle data



df = df.sample(frac=1)



# get random data on that class

ct_df = df.loc[df['Class'] == 1]

cf_df = df.loc[df['Class'] == 0][:492]



#make it in one dataframe

normal_distributed_df = pd.concat([ct_df, cf_df])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



new_df.head()
##mengetahui jumlah value dari tiap label pada kelas di kolom

new_df["Class"].value_counts()
# New_df is from the random undersample data (fewer instances)

X = new_df.drop('Class', axis=1)

y = new_df['Class']
from sklearn.model_selection import train_test_split

# Whole dataset

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values.ravel()

y_test = y_test.values.ravel()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

clf = LogisticRegression()

clf.fit(X_train, y_train)



label_prediction = clf.predict(X_test)

mse = mean_squared_error(y_test, label_prediction)

rmse = np.sqrt(mse)
print("mse = ",mse)
print("rmse = ",rmse)
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, label_prediction)

print(cnf_matrix)
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))