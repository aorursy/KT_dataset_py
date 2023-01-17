import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
ks = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

ks.head()
count_category = ks.groupby('main_category')

size = count_category['main_category'].value_counts()





category = [category for category, df in ks.groupby('main_category')]





plt.figure(figsize=(5,3), dpi=200)

pie_chart = plt.pie(size,labels=category, textprops={'fontsize': 6}, autopct='%.f%%', pctdistance=0.8)





plt.title('Percentage of Kickstart Projects Grouped By Categories', fontsize=8)

plt.show()
ks.isna().sum()
ks.drop(['name', 'deadline', 'launched', 'usd pledged', 'goal', 'currency', 'pledged', 'ID'], axis='columns', inplace=True)
ks.head()
ks['country'].unique()
ks['main_category'].unique()
ks['state'].unique()
ks['backers'].describe()
ks.head()
ks['state'].value_counts()
ks['state'].replace({'canceled': 'failed', 'live': 'successful', 'suspended': 'failed'}, inplace=True)
ks.drop(ks[ks['state'] == 'undefined'].index, axis='rows', inplace=True)
ks['country'].value_counts()
# 'N,0"' is referencing Null Island so we can drop



ks.drop(ks[ks['country'] == 'N,0"'].index, axis='rows', inplace=True)
data = ks
data.head()
df = ks.groupby(['country'])
# Getting the total $ pledged for each country and without scientific notation



total_amount = df['usd_pledged_real'].sum().apply(lambda x: '%.0f' % x)



total_amount = [int(i)/1000000 for i in total_amount] 



print(total_amount)
countries = [country for country, df in ks.groupby('country')]



plt.bar(countries, total_amount)

plt.xticks(rotation='vertical', size=8)

plt.ylabel('Amount Spent by Country(in millions $)', size = 13)

plt.xlabel('Country Abbreviated', size = 13)



plt.show()



#
data.head()
data = ks
state_dummies = pd.get_dummies(data['state']).iloc[: , 1:]
main_cat_dummies = pd.get_dummies(data['main_category']).iloc[: , 1:]
category_dummies = pd.get_dummies(data['category']).iloc[: , 1:]
country_dummies = pd.get_dummies(data['country']).iloc[: , 1:]
numerical_data = ks.drop(columns = ['category', 'main_category', 'state', 'country'], axis='columns')
num_data = pd.concat([numerical_data], axis='columns', sort=False)
cat_data = pd.concat([state_dummies, main_cat_dummies, category_dummies, country_dummies], axis='columns', sort=False)
final_data = pd.concat([num_data, cat_data], axis='columns', sort=False)
final_data.head()
sns.heatmap(data.corr(), center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
X = final_data.drop(['successful'], axis='columns')
y = final_data['successful']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()



clf = lr.fit(X_train, y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

acc_score = accuracy_score(y_test, y_pred)

print(acc_score*100)



# Score of 87.819%
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree 



clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)

clf_entropy.fit(X_train, y_train)



DTC_y_pred = clf_entropy.predict(X_test)



DTC_acc_score = accuracy_score(y_test, DTC_y_pred)

print(DTC_acc_score * 100)



#Score of 98 % (BEST MODEL SO FAR)