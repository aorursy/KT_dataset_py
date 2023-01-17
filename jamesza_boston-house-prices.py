import os

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelBinarizer





#root_path = r"D:\Files\Personal\Projects\Kaggle\HousePrices"

#os.chdir(root_path)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")

df_train.head()
print(df_train.info())

print(df_train.describe())
sns.set_style("white")

sns.distplot(df_train["SalePrice"])

sns.despine(left=True, bottom=True)

median = np.median(df_train["SalePrice"])

plt.axvline(df_train["SalePrice"].mean(), color="b")

plt.axvline(median, color="g")

plt.show()
df_train = pd.get_dummies(df_train)

df_train.head()
df_train = pd.get_dummies(df_train, columns=["MSSubClass"])

df_train.head()
cols = list(df_train.columns.values)

cols.pop(cols.index("SalePrice"))

df_train = df_train[cols+["SalePrice"]]

corr_matrix = df_train.corr()

fig, ax = plt.subplots(figsize=(60,60))

sns.heatmap(corr_matrix, linewidths=.5, ax=ax, square=True)

sns.set_context("notebook", font_scale=2)

plt.title("Correlation Heatmap")

plt.show()
cols = list(df_train.columns.values)

cols.pop(cols.index("SalePrice"))

df_train = df_train[cols+["SalePrice"]]

corr_matrix = df_train.corr()

corr_matrix = corr_matrix[(corr_matrix["SalePrice"] > 0.3) | (corr_matrix["SalePrice"] < -0.3)]

corr_matrix = corr_matrix[corr_matrix.index]

fig, ax = plt.subplots(figsize=(20,20))

sns.heatmap(corr_matrix, linewidths=.5, ax=ax, square=True)

sns.set_context("notebook", font_scale=2)

plt.title("Correlation Heatmap")

plt.show()
df_train.drop("GarageCars", 1, inplace=True)
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
df_train = df_train.drop((missing_data[missing_data["Total"] > 1]).index,1)

df_train.isnull().sum().max() 
corr_matrix = df_train.corr()

corr_matrix = corr_matrix[(corr_matrix["SalePrice"] > 0.3) | (corr_matrix["SalePrice"] < -0.3)]

corr_matrix = corr_matrix[corr_matrix.index]

t = 0

total_columns = len(corr_matrix.columns.values)

columns = corr_matrix.columns.values[0:(total_columns-1)]

current_cols = columns

columns_per_plot = 5

while t < total_columns:    

    if (total_columns - t) >= columns_per_plot:

        cols = columns[t:(t+columns_per_plot)]

        t += columns_per_plot

    else:

        cols = columns[t:t+(total_columns-t)]

        t += (total_columns-t)

    cols = np.append(cols, ["SalePrice"])

    sns.set()

    sns.pairplot(df_train[cols], size = 2.5)

    plt.show()
fig, ax = plt.subplots(figsize=(5,5))

sns.set()

sns.boxplot(y="GrLivArea", data=df_train, ax=ax)

plt.show()

sns.jointplot(x="GrLivArea", y="SalePrice", data=df_train)

plt.show()
df_train.drop(df_train[df_train["GrLivArea"] > 4000][df_train["SalePrice"] < 300000].index, inplace=True)

fig, ax = plt.subplots(figsize=(5,5))

sns.set()

sns.boxplot(y="GrLivArea", data=df_train, ax=ax)

plt.show()

sns.jointplot(x="GrLivArea", y="SalePrice", data=df_train)

plt.show()
t = 0

total_columns = len(corr_matrix.columns.values)

columns = corr_matrix.columns.values[0:(total_columns-1)]

current_cols = columns

columns_per_plot = 5

while t < total_columns:    

    if (total_columns - t) >= columns_per_plot:

        cols = columns[t:(t+columns_per_plot)]

        t += columns_per_plot

    else:

        cols = columns[t:t+(total_columns-t)]

        t += (total_columns-t)

    cols = np.append(cols, ["SalePrice"])

    sns.set()

    sns.pairplot(df_train[cols], size = 2.5)

    plt.show()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



y = df_train["SalePrice"]

X = df_train.drop(["SalePrice", "Id"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

clf = RandomForestRegressor(n_jobs=2, n_estimators=1000)

model = clf.fit(X_train, y_train)
feature_importance = pd.Series(data=model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

num_cols = feature_importance.index

features = 50

bar_widths = feature_importance.iloc[0:features].values

bar_positions = np.arange(features) + 0.9

tick_positions = range(1,features+1)

fig, ax = plt.subplots(figsize=(30,30))

ax.barh(bar_positions, bar_widths, 0.5)



ax.set_yticks(tick_positions)

ax.set_yticklabels(num_cols, fontsize=20)

ax.set_ylabel("Feature", fontsize=25)

ax.set_xlabel("Importance", fontsize=25)

ax.set_title("Importance of Features Determined By Random Forest", fontsize=40)

plt.show()
from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



predictions = model.predict(X_test)

 

for i in np.arange(0, len(predictions)):

    print("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_test)[i], predictions[i]))
plt.figure(figsize=(10, 5))

plt.scatter(y_test, predictions, s=20)

plt.title('Predicted vs. Actual')

plt.xlabel('Actual Sale Price')

plt.ylabel('Predicted Sale Price')



plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])

plt.tight_layout()

plt.show()
print("Train Accuracy: ", explained_variance_score(y_train, model.predict(X_train)))

print("Test Accuracy: ", explained_variance_score(y_test, predictions))



print("Mean absolute error: ", mean_absolute_error(y_test.astype(float, copy=False), predictions))

print("Mean squared error: ", mean_squared_error(y_test.astype(float, copy=False), predictions))

print("R2 score: ", r2_score(y_test.astype(float, copy=False), predictions))