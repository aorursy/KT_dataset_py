import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Dimension of train :",train_df.shape)
print("Dimension of test :",test_df.shape)
train_df.head()
test_df.head()
train_df.info()
test_df.info()
train_df.isnull().values.any()
test_df.isnull().values.any()
def check_sparsity(df):
    non_zeros = (df.ne(0).sum(axis=1)).sum()
    total = df.shape[1]*df.shape[0]
    zeros = total - non_zeros
    sparsity = round(zeros / total * 100,2)
    density = round(non_zeros / total * 100,2)

    print(" Total:",total,"\n Zeros:", zeros, "\n Sparsity [%]: ", sparsity, "\n Density [%]: ", density)

check_sparsity(train_df)
check_sparsity(test_df)
train_df['target'].describe()
#Distribution plot of target variable
plt.figure(figsize=(8,5))
sns.distplot(train_df['target'])
plt.figure(figsize=(8,5))
sns.distplot(np.log(train_df['target']), kde='False')
X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)
X_train.shape,X_test.shape
drop_cols=[]
for cols in X_train.columns:
    if X_train[cols].std()==0:
        drop_cols.append(cols)
print("Number of constant columns dropped: ", len(drop_cols))
print(drop_cols)
X_train.drop(drop_cols,axis=1, inplace = True)
X_test.drop(drop_cols, axis=1, inplace = True)
X_train.shape,X_test.shape
