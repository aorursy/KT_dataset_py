# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv(os.path.join(dirname, filename))
print(data["diagnosis"].value_counts())

print(f'Type of the diagnosis column: {data["diagnosis"].dtype}')
malign = pd.get_dummies(data["diagnosis"], drop_first=True, dtype=int)
# we replace diagnosis by the dummy variable

df = pd.concat((data.drop("diagnosis", axis=1), malign), axis=1)

df.rename(columns={'M':'malign'}, inplace=True)
import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px
fig = px.pie(df, names="malign", title='Distribution of malignant vs. benign tumor')

fig.show()
df.columns
dropped = ["id", "Unnamed: 32"]

df = df.drop(dropped, axis=1)
plt.figure(figsize=(20, 20))

sns.heatmap(df.corr(), cmap="coolwarm", annot=True)

plt.show()
px.bar(df.corr()["malign"].sort_values(), title="correlation degree", color=df.corr()["malign"].sort_values(), color_continuous_scale=px.colors.sequential.Jet)
last_dropped = ["symmetry_se", "texture_se", "fractal_dimension_mean"]

final_df = df.drop(last_dropped, axis=1)
print(f"Our model will be based upon {len(final_df.iloc[0,:])-1} features to predict if benign or malignant.")
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
X = final_df.drop("malign", axis=1)

y = final_df["malign"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train.shape
lg = LogisticRegression()

lg.fit(X_train, y_train)
test_pred = lg.predict(X_test)

train_pred = lg.predict(X_train)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

#print(classification_report(y_test, test_pred))

print(f"Confusion matrix on training set:\n{confusion_matrix(y_train, train_pred)}")

print(f"Confusion matrix on test set:\n{confusion_matrix(y_test, test_pred)}")

print(f"Accuracy score on the training set: {accuracy_score(y_train, train_pred)*100:.2f}%")

print(f"Accuracy score on the test set: {accuracy_score(y_test, test_pred)*100:.2f}%")

print(f"f1 score on the training set: {f1_score(y_train, train_pred)*100:.2f}%")

print(f"f1 score on the test set: {f1_score(y_test, test_pred)*100:.2f}%")