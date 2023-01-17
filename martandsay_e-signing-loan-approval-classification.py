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

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/esigning-of-loan-based-on-financial-history/financial_data.csv")
df.head()
df.columns
df.info()
df.describe()
# Check missing values

df.isnull().any()
# Remove unwanted column

df_new = df.drop(columns=["entry_id", "e_signed", "pay_schedule"])
df.head()
# Histogram for every single column

plt.figure(figsize=(15, 12))

for i in range(df_new.shape[1]):

    plt.subplot(6, 3, i+1)

    f = plt.gca()

    f.set_title( df_new.columns.values[i])

    bins = len(df_new.iloc[:, i].unique())

    if bins >= 100:

        bins = 100

    plt.hist(df_new.iloc[:, i], bins=bins)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
df_new.corrwith(df.e_signed)
df_new.corrwith(df.e_signed).plot.bar(rot=60, figsize=(16, 13), title="Correlation with e-signed", grid=True

                                     , fontsize=15)
plt.figure(figsize=(15, 12))

sns.heatmap(df_new.corr(), annot=True)
df_new.head()
# personal_account_m & personal_account_y column can be changed to single column personal_Account_months



df["personal_account_month"] = df["personal_account_m"] + 12* df["personal_account_y"]
users = df["entry_id"]

response = df["e_signed"]

df.drop(columns=["entry_id", "months_employed", "e_signed","personal_account_m", "personal_account_y"], inplace=True)
df.head()
df = pd.get_dummies(df, drop_first=True)
df.columns
df.head()
X = df.values

y= response.values
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators =10, max_features=10, random_state=0,  criterion='entropy')

clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predict)

#sns.heatmap(cm, annot=True)

cm
# accuracy

clf.score(X_test, y_test) # we can see score is very bad. Lets apply gridsearchcv to fine tune our model
from sklearn.model_selection import GridSearchCV

params = {

    "n_estimators" : [10, 100, 200 ],  "criterion":["entropy", "gini"]

}

gs = GridSearchCV(estimator=clf, param_grid=params, cv=10)
result = gs.fit(X_train, y_train)
result.best_params_
result.best_score_
#So we can see our best accuracy is 63%. Actually acocrding to our data it is a good accuracy and our case study is not that sensitive so it may compromise with some %of accuracy.

# lets train and fit our final model with the best params.

clf = RandomForestClassifier(n_estimators=200, criterion="entropy")

clf.fit(X_train, y_train)

y_predict=clf.predict(X_test)

cm = confusion_matrix(y_test, y_predict)

cm
classification_report(y_test, y_predict)
import keras

from keras.layers import Dense, Dropout

from keras.models import Sequential
clf_n = Sequential([

    Dense(activation="relu", init="uniform", input_dim=19, output_dim=10),

    Dense(activation="relu", init="uniform", output_dim=10),

    Dropout(0.5),

    Dense(activation="relu", init="uniform", output_dim=10),

    Dense(activation="sigmoid", init="uniform", output_dim=1)

])
clf_n.summary()
clf_n.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy")
print(X_train.shape)

print(np.array(y_train.shape))
clf_n.fit((X_train), (y_train), batch_size=20, epochs=50)
# So here we can see our deep learning network accuracy is very less. It may be because of data. As deep learning requires a large datasets. So for our case we will settle with

# random forest classifier. You also try diffrent model to fine tune and let us know which one do you feel suits our case study.

# Thanks... UPVOTE IF YOU LIKE THE KERNEL