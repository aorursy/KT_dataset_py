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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
udemy_data = pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')

udemy_data.head()
udemy_data.info()
udemy_data.columns
udemy_data.describe()
subject_col = udemy_data['subject'].value_counts()

subject_col
categorical_df = udemy_data.select_dtypes(include=["object"])
categorical_df.subject.unique()
categorical_df.level.unique()
subject_col.plot.barh(color='lightblue');
sns.distplot(udemy_data.price, bins=10, kde=False);
sns.barplot(x = "level", y = "price", hue = "subject", data = udemy_data,ci = None)

plt.show()
g = sns.catplot(x="price", aspect=3, data=udemy_data, kind="count")

g.fig.suptitle("price Counts", y=1.0)

plt.show()
sns.relplot(x="price", y="num_subscribers", data=udemy_data, kind="line", aspect=4, ci="sd")

plt.show()
sns.distplot(udemy_data.price, bins=10)
sns.kdeplot(udemy_data.price, shade = True)
sns.catplot(x = "subject", y = "price", kind="violin",aspect=3, data = udemy_data,);
# 3.cu icin hue

sns.catplot(x = "subject", y = "price", hue="level", kind="violin", aspect=3, data = udemy_data);
sns.catplot(x = "price", y = "level", hue = "subject", kind = "point", data = udemy_data);
sns.scatterplot(x = "price", y = "subject", hue = "level", data = udemy_data);
sns.relplot(x="num_reviews", y="num_subscribers",aspect=3, data=udemy_data, kind="scatter")

plt.show()
sns.set_palette("RdBu")

correlation=udemy_data.corr()

sns.heatmap(correlation, annot=True)

plt.show()


sns.lmplot(x = "price", y = "num_subscribers", hue = "level", data=udemy_data);
sns.boxplot(x = udemy_data["price"]);
sns.boxplot(x = "level", y = "price", data = udemy_data);
sns.boxplot(x = "level", y = "price", hue="subject", data = udemy_data)
fig, ax = plt.subplots()

ax.hist(udemy_data.level, label="price", bins=10)

ax.set_xlabel("level")

ax.set_ylabel("price")

plt.show()
sns.catplot(x="is_paid",data=udemy_data, kind="count")

plt.show()
sns.lmplot(x = "num_subscribers",y="num_reviews", hue = "subject", data=udemy_data);
sns.lineplot(x = "price", y = "num_subscribers", hue="subject", data = udemy_data);
# we dont have NaN value

df_for_model.isna().sum().sum()
df_for_model = udemy_data.copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_le = df_for_model.copy()

le.fit(df_le["level"])
list(le.classes_)
pd.get_dummies(df_for_model["level"])
X = df_for_model['num_subscribers'].values

y = df_for_model['num_reviews'].values
print(X.shape)

print(y.shape)
X = X.reshape(-1,1)

y = y.reshape(-1,1)
X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# !!! DO NOT FORGET TO LIBRARIES

reg = LinearRegression()



reg.fit(X_train,y_train)

preds = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,preds))

print(rmse)

print('Score',reg.score(X_test,y_test))
# !!! DO NOT FORGET TO LIBRARIES





# Instantiate model with 100 decision trees

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data

rf.fit(X_train, y_train);

# Use the forest's predict method on the test data

predictions = rf.predict(X_test)

# Calculate the absolute errors

errors = abs(predictions - y_test)



# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#Score

print("Score :",rf.score(X_test, y_test))