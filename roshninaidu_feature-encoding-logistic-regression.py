import pandas as pd

import numpy as np

from sklearn import preprocessing

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import scipy

from sklearn import linear_model, datasets
train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv",  index_col='id')

test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col = 'id')
train.head()
test.head()
print(train.isnull().sum().sum())

print(test.isnull().sum().sum())
test.nunique()
sns.countplot(train['target'])

plt.title("Distribution of Target values")

plt.show()
train['bin_4'] = train['bin_4'].map({'Y': 1, 'N': 0})

train['bin_3'] = train['bin_3'].map({'T': 1, 'F': 0})

train['ord_1'] = train['ord_1'].map({'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3,'Grandmaster': 4})

train['ord_2'] = train['ord_2'].map({'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5})
test['bin_4'] = test['bin_4'].map({'Y': 1, 'N': 0})

test['bin_3'] = test['bin_3'].map({'T': 1, 'F': 0})

test['ord_1'] = test['ord_1'].map({'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3,'Grandmaster': 4})

test['ord_2'] = test['ord_2'].map({'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5})
l3 = train[train['target'] == 1]['ord_3'].value_counts() / train['ord_3'].value_counts() 

l4 = train[train['target'] == 1]['ord_4'].value_counts() / train['ord_4'].value_counts()

l5 = train[train['target'] == 1]['ord_5'].value_counts() / train['ord_5'].value_counts()
def sorted_ord(col_name,ratio):

    s_ratio = ratio.sort_values()

    keys = list(s_ratio.keys())

    train[col_name] = train[col_name].apply(lambda x : keys.index(x))

    test[col_name] = test[col_name].apply(lambda x : keys.index(x))



sorted_ord('ord_3',l3)

sorted_ord('ord_4',l4)

sorted_ord('ord_5',l5)
train.head()
nom_col = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4','nom_5','nom_6','nom_7','nom_8','nom_9','day','month']

nom_train = train[nom_col].astype(str)

nom_test = test[nom_col].astype(str)



ohc_test = pd.get_dummies(nom_test, sparse = True)

ohc_train = pd.get_dummies(nom_train, sparse = True)
ohc_test.dtypes
ohc_train
unique_test = list(set(ohc_test.columns) - set(ohc_train.columns))

unique_train = list(set(ohc_train.columns) - set(ohc_test.columns))



print("Unique test columns: ", len(unique_test))

print("Unique Train columns: ",len(unique_train))



#Drop all extra test cols, cant learn anything

print(len(ohc_test.columns))



ohc_test_final = ohc_test.drop(unique_test, axis = 1)



print(len(ohc_test_final.columns))



#Drop all extra train cols, doesnt matter for test

print(len(ohc_train.columns))



ohc_train_final = ohc_train.drop(unique_train, axis = 1)



print(len(ohc_train_final.columns))



dropped_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4','nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9','day','month','target']

final_df = train.drop(dropped_cols, axis =1)

target_train = train['target']



dropped_cols_test = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4','nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9','day','month']

test_df = test.drop(dropped_cols_test, axis=1)



import scipy

df = final_df.to_sparse()

df1 = test_df.to_sparse()



df

df=df.to_coo()

df=df.tocsr()



df1

df1=df1.to_coo()

df1=df1.tocsr()



final_dataset = scipy.sparse.hstack([df,ohc_train_final])

final_test = scipy.sparse.hstack([df1,ohc_test_final])

# Generation of C value taking too long, using the best value of C = 0.12

# C = np.arange(0.07, 0.13, 0.01)

C = [0.12]



lr = LogisticRegression()



max_iter = [10000]



hyperparameters = dict(C=C, max_iter = max_iter)



clf = GridSearchCV(lr, hyperparameters, cv=10, verbose=0)



best_model = clf.fit(final_dataset,target_train)
predictions = best_model.predict_proba(final_test)
predictions
samplesub = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv", index_col="id")



output = pd.DataFrame({'Id': samplesub.index, 'target': predictions[:,-1]})

output.to_csv('submission.csv', index=False)