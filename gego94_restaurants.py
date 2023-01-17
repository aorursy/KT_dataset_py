# import libraries



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier 

from sklearn import metrics 

import matplotlib.pyplot as plt

%matplotlib inline
# fetching data annd removing useless columns like name, url, ...

data = pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv').drop(['url','address','name','phone','location',

                          'reviews_list','menu_item','listed_in(city)'], axis=1)



# cleaning datam by removing null values and categorize other data

data.dropna(inplace=True)

data.drop(data[ data['rate'] == 'NEW'].index , inplace=True)

data.drop(data[data['rate'].str.len() < 2].index , inplace=True)

data.online_order = data.online_order.astype('category').cat.codes

data.book_table = data.book_table.astype('category').cat.codes

data.rest_type = data.rest_type.astype('category').cat.codes

data['listed_in(type)'] = data['listed_in(type)'].astype('category').cat.codes

data['approx_cost(for two people)'] = data['approx_cost(for two people)'].str.replace(',','',regex=True).astype(float)



# for these kind of data i took the number of possibilities they gave, because they're lists of elements

data.dish_liked = data.dish_liked.str.split(',').str.len().astype('category').cat.codes + 1

data['cuisines'] = data.cuisines.str.split(',').str.len().astype('category').cat.codes + 1



# representing the score as the float number *10, because the model needs integer values as targets

data['rate'] = (data.rate.str.slice(stop=-2).astype(float) * 10).astype(int)

data.head()
# most of the columns have not much possibile values, so decision trees are good.



data.nunique()
data.shape
# create train and test set and take 10% of data as test

x = data.drop('rate', axis=1).values

y = data['rate'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 2)
# Trying prediction with entropy model. Max depth of the tree is free.

res = pd.DataFrame(columns=['Accuracy'])

for i in range(2,28):

    entropy = DecisionTreeClassifier(criterion="entropy", max_depth=i)

    entropy = entropy.fit(X_train,y_train)

    y_pred = entropy.predict(X_test)

    res = res.append({'Accuracy': metrics.accuracy_score(y_test, y_pred)}, ignore_index=True)

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# print(entropy.get_depth())

plt.plot(range(2,28), res)
res
# Trying prediction with gini model. Max depth of the tree is free.

res1 = pd.DataFrame(columns=['Accuracy'])

for i in range(2,28):

    gini = DecisionTreeClassifier(criterion="gini", max_depth=i)

    gini = gini.fit(X_train,y_train)

    y_pred = gini.predict(X_test)

    res1 = res1.append({'Accuracy': metrics.accuracy_score(y_test, y_pred)}, ignore_index=True)

plt.plot(range(2,28), res1)
res1
# As we can see, decision trees performs really well in these kind of tasks, where there isn't a big variance in data values. (only in 1)

# Further more the differnce between the 2 methods is really small and depends principally on how train data is taken. Generally entropy 

# methods performs a bit better.