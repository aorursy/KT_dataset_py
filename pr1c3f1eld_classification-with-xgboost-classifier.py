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
import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
train = pd.read_csv("../input/cat-in-the-dat/train.csv")

#train.head(5)

#train.tail(5)
#sns.countplot(train["nom_0"])

#sns.countplot(train["nom_1"]) -scaling

#sns.countplot(train["nom_2"]) -scaling

#sns.countplot(train["nom_3"])

#sns.countplot(train["nom_4"]) -scaling

#sns.countplot(train["nom_5"]) -drop

#sns.countplot(train["nom_6"]) -drop

#sns.countplot(x = "nom_7", data = train) -drop

#sns.countplot(train["nom_8"]) -drop

#sns.countplot(train["ord_0"])

#sns.countplot(train["ord_1"]) -scaling

#sns.countplot(train["ord_2"]) -scaling

#sns.countplot(train["ord_3"]) -scaling

#sns.countplot(train["ord_4"]) -scaling

#sns.countplot(train["ord_5"]) -drop

#sns.countplot(train["month"])
#train = train.drop(["ord_5","nom_5","nom_6","nom_7","nom_8","nom_9"], axis = 1)
train.tail(5)
train.drop("id", axis = 1, inplace = True)
train.head(5)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score



#bin_3

l_enc = LabelEncoder()

X = train["bin_3"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"bin_3":X})

train.drop("bin_3", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#bin_4

X = train["bin_4"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"bin_4":X})

train.drop("bin_4", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#nom_0

scaler = StandardScaler()

X = train["nom_0"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_0":X})

train.drop("nom_0", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#nom_1

scaler = StandardScaler()

X = train["nom_1"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_1":X})

train.drop("nom_1", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#nom_2

scaler = StandardScaler()

X = train["nom_2"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_2":X})

train.drop("nom_2", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#nom_3

scaler = StandardScaler()

X = train["nom_3"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_3":X})

train.drop("nom_3", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#nom_4

scaler = StandardScaler()

X = train["nom_4"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_4":X})

train.drop("nom_4", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#nom_5

scaler = StandardScaler()

X = train["nom_5"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_5":X})

train.drop("nom_5", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#nom_6

scaler = StandardScaler()

X = train["nom_6"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_6":X})

train.drop("nom_6", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#nom_7

scaler = StandardScaler()

X = train["nom_7"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_7":X})

train.drop("nom_7", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#nom_8

scaler = StandardScaler()

X = train["nom_8"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_8":X})

train.drop("nom_8", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#nom_9

scaler = StandardScaler()

X = train["nom_9"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_9":X})

train.drop("nom_9", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#ord_1

scaler = StandardScaler()

X = train["ord_1"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"ord_1":X})

train.drop("ord_1", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#ord_2

scaler = StandardScaler()

X = train["ord_2"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"ord_2":X})

train.drop("ord_2", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#ord_3

scaler = StandardScaler()

X = train["ord_3"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"ord_3":X})

train.drop("ord_3", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#ord_4

scaler = StandardScaler()

X = train["ord_4"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"ord_4":X})

train.drop("ord_4", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
#ord_5

scaler = StandardScaler()

X = train["ord_5"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"ord_5":X})

train.drop("ord_5", axis = 1, inplace = True)

train = pd.concat([train,X_df], axis = 1)

train.head(5)
X_t = train.drop("target", axis = 1).values

X_t = scaler.fit_transform(X_t)

#print(X_t)

Y_t = train["target"].values

#print(Y_t)
from xgboost import XGBClassifier



rs = 2

X_train, X_dev, Y_train, Y_dev = train_test_split(X_t,Y_t, test_size = 0.2, random_state = rs)
classifier = XGBClassifier(learning_rate=0.05,n_estimators=50000,seed=2019,reg_alpha=5,eval_metric='auc',tree_method='gpu_hist')

classifier.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_dev, Y_dev)],early_stopping_rounds=50, verbose=50)
results = classifier.evals_result()

epochs = len(results['validation_0']['auc'])

x_axis = range(0, epochs)



# plotting the loss

plt.figure(figsize=(15, 7))

plt.plot(x_axis, results['validation_0']['auc'], label='Train')

plt.plot(x_axis, results['validation_1']['auc'], label='Val')

plt.legend()

plt.ylabel('AUC')

plt.xlabel('# of iterations')

plt.title('XGBoost AUC')

plt.show()
Y_hat = classifier.predict(X_dev)

print(accuracy_score(Y_dev, Y_hat))
test_orig = pd.read_csv("../input/cat-in-the-dat/test.csv")
test = test_orig.copy()

test.drop("id", axis = 1, inplace = True)



#bin_3

l_enc = LabelEncoder()

X = test["bin_3"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"bin_3":X})

test.drop("bin_3", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#bin_4

l_enc = LabelEncoder()

X = test["bin_4"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"bin_4":X})

test.drop("bin_4", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#nom_0

scaler = StandardScaler()

X = test["nom_0"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_0":X})

test.drop("nom_0", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#nom_1

scaler = StandardScaler()

X = test["nom_1"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_1":X})

test.drop("nom_1", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#nom_2

scaler = StandardScaler()

X = test["nom_2"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_2":X})

test.drop("nom_2", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#nom_3

scaler = StandardScaler()

X = test["nom_3"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_3":X})

test.drop("nom_3", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#nom_4

scaler = StandardScaler()

X = test["nom_4"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_4":X})

test.drop("nom_4", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#nom_5

scaler = StandardScaler()

X = test["nom_5"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_5":X})

test.drop("nom_5", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#nom_6

scaler = StandardScaler()

X = test["nom_6"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_6":X})

test.drop("nom_6", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#nom_7

scaler = StandardScaler()

X = test["nom_7"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_7":X})

test.drop("nom_7", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#nom_8

scaler = StandardScaler()

X = test["nom_8"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_8":X})

test.drop("nom_8", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#nom_9

scaler = StandardScaler()

X = test["nom_9"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"nom_9":X})

test.drop("nom_9", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#ord_1

scaler = StandardScaler()

X = test["ord_1"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"ord_1":X})

test.drop("ord_1", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#ord_2

scaler = StandardScaler()

X = test["ord_2"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"ord_2":X})

test.drop("ord_2", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#ord_3

scaler = StandardScaler()

X = test["ord_3"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"ord_3":X})

test.drop("ord_3", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#ord_4

scaler = StandardScaler()

X = test["ord_4"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"ord_4":X})

test.drop("ord_4", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

#test.head(5)



#ord_5

scaler = StandardScaler()

X = test["ord_5"].values

X = l_enc.fit_transform(X)

X_df = pd.DataFrame(data = {"ord_5":X})

test.drop("ord_5", axis = 1, inplace = True)

test = pd.concat([test,X_df], axis = 1)

test.head(5)
X_test = test.values

#print(X_test)

X_test = scaler.fit_transform(X_test)

#print(X_test)
Y_test = classifier.predict_proba(X_test, ntree_limit=classifier.best_ntree_limit)[:, 1]

print(Y_test)
Y_df = pd.DataFrame(data = {'target':Y_test})

sub = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")

sub.drop("target", axis = 1, inplace = True)

test_final = pd.concat([sub, Y_df], axis = 1)

test_final.head(5)
test_final.to_csv("submission.csv", index = False)