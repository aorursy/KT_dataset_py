# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df_kick = pd.read_csv("../input/ks-projects-201801.csv")
df_kick.describe()
df_kick.head(n=3)
# Any results you write to the current directory are saved as output.
# Remove confusing redundant columns; The usd_pledged_real and usd_goal_real are the only two that matter
df_trimmed = df_kick.drop(columns=['goal', 'pledged','usd pledged'])
print(df_trimmed.shape)
print(df_trimmed.info())

df_trimmed.head(n=3)

percentual_sucess = round(df_trimmed["state"].value_counts() / len(df_trimmed["state"]) * 100,2)

print("State Percentual in %: ")
print(percentual_sucess)

plt.figure(figsize = (8,6))

ax1 = sns.countplot(x="state", data=df_trimmed)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45)
ax1.set_title("Status Project Distribuition", fontsize=15)
ax1.set_xlabel("State Description", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)

plt.show()
# Divide the available dataframe into train and test categories
# Ignore the projects that are live, suspended, canceled or undefined
# Trim again
df_trimmed = df_trimmed[df_trimmed.state != 'canceled' ]
df_trimmed = df_trimmed[df_trimmed.state != 'live' ]
df_trimmed = df_trimmed[df_trimmed.state != 'undefined' ]
df_trimmed = df_trimmed[df_trimmed.state != 'suspended' ]#or df_trimmed.state !='live' or df_trimmed.state!= 'undefined' or df_trimmed.state!='suspended']
df_trimmed.head(10)


# Total number of projects in each main category and total successful overlapped bar plots
main_cats = df_trimmed["main_category"].value_counts()
main_cats_failed = df_trimmed[df_trimmed["state"] == "failed"]["main_category"].value_counts()
main_cats_success = df_trimmed[df_trimmed["state"] == "successful"]["main_category"].value_counts()

#main_cats_failed and main_cats_success are pandas series objects

main_cats_success=main_cats_success.sort_index(ascending=True)
main_cats_failed=main_cats_failed.sort_index(ascending=True)

main_cats_combined = (main_cats_success + main_cats_failed)

x1=main_cats_combined.index
sns.set_color_codes("pastel")
sns.barplot(y=main_cats_combined,x=x1,color="b",label="Total")

x2=main_cats_success.index
sns.set_color_codes("muted")
  
ax2 =sns.barplot(y=main_cats_success,x=x2,color="b",label="Success")
ax2.set_xticklabels(ax2.get_xticklabels(),rotation = 70)

plt.show()


#%% Success by country

country_wise = df_trimmed["country"].value_counts()
country_wise_failed = df_trimmed[df_trimmed["state"] == "failed"]["country"].value_counts()
country_wise_success = df_trimmed[df_trimmed["state"] == "successful"]["country"].value_counts()


country_wise_success=country_wise_success.sort_index(ascending=True)
country_wise_failed=country_wise_failed.sort_index(ascending=True)

country_wise_combined = (country_wise_success + country_wise_failed)

z1=country_wise_combined.index
sns.set_color_codes("pastel")
sns.barplot(y=country_wise_combined,x=z1,color="b",label="Total")

z2=country_wise_success.index
sns.set_color_codes("muted")
  
ax2 =sns.barplot(y=country_wise_success,x=z2,color="b",label="Success")
ax2.set_xticklabels(ax2.get_xticklabels(),rotation = 70)


plt.show()

df_trimmed["pledge_log"] = np.log(df_trimmed["usd_pledged_real"]+ 1)
df_trimmed["goal_log"] = np.log(df_trimmed["usd_goal_real"]+ 1)
g = sns.FacetGrid(df_trimmed, hue="state", col="main_category", margin_titles=True)
g=g.map(plt.hist, "goal_log",edgecolor="w",bins=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]).add_legend()

# Using tree ensenble method for prediction
# Do one-hot encoding on main category names
# One-hot encoding for Histogram bins for goal-log
# create a new array?

df_dummies = pd.get_dummies(df_trimmed['main_category'])
df_y_dummies = pd.get_dummies(df_trimmed['state'])
df_trimmed = df_trimmed.join(df_dummies)
df_trimmed = df_trimmed.join(df_y_dummies)


#df_dummies.head(3)
df_trimmed.head(3)                                  
                                                     

# Divide into train and test sets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import math

X = df_trimmed.iloc[:,11:27]
y = df_trimmed.iloc[:,27]

Xarr = X.values
yarr = y.values


# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(Xarr,yarr,
                                                    test_size=0.3)
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X_train, y_train)
print(scores.mean())

y_predict = clf.predict(X_test)
meanSquaredError=mean_squared_error(y_test, y_predict)
print("MSE:", meanSquaredError)
rootMeanSquaredError = math.sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)
                                                                    