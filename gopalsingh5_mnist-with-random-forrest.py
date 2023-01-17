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
train_dir = "/kaggle/input/digit-recognizer/train.csv"

test_dir = "/kaggle/input/digit-recognizer/test.csv"

submission_dir = "/kaggle/input/digit-recognizer/submission.csv"
df_train = pd.read_csv(train_dir)
df_train.head()
import matplotlib.pyplot as plt



def plot_images(df, n):

    for i in range(n**2):

        random_value = np.random.randint(10)

        plt.subplot(n, n, i+1)

        X = df.iloc[random_value, 1:].values.reshape(28, 28)

        plt.imshow(X, cmap="binary")

        plt.title(str(df["label"][random_value]))

        plt.axis("off")

    plt.figure(figsize=(10, 10))

    plt.subplots_adjust(left=1, right=1.5, bottom=1.5, top=2.5)

    plt.show()
plot_images(df_train, 4)
X = df_train.iloc[:, 1:].values

y = df_train["label"].values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=123)
print("train size: {}\ntest size: {}".format(len(X_train), len(X_test)))
from sklearn.ensemble import RandomForestClassifier



rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

rnd_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score



preds = rnd_clf.predict(X_test)

accuracy_score(preds, y_test)
rnd_clf.get_params()
from sklearn.model_selection import GridSearchCV



params = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 150, 200]

}



grid_search = GridSearchCV(rnd_clf, params, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
grid_search.best_score_
final_model = grid_search.best_estimator_

preds = final_model.predict(X_test)

accuracy_score(preds, y_test)
from sklearn.metrics import confusion_matrix

import seaborn as sns



cnf_mtx = confusion_matrix(preds, y_test)
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(cnf_mtx, annot=True, fmt="d")
test_df = pd.read_csv(test_dir)

predictions = final_model.predict(test_df)

test_df["Label"] = predictions
# Checking how many times our data is getting confused with what number

# It turns out its 9 and 0 the most

temp = pd.read_csv(test_dir)

temp["label"] = predictions



plot_images(temp, 5)
test_df.head()
submission = pd.DataFrame({'ImageId': range(1,len(test_df)+1) ,'Label': predictions })

submission.head()
submission.to_csv("submission.csv",index=False)