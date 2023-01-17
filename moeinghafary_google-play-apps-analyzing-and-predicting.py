
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
df.isnull().sum()
df = df[df.App != 'Life Made WI-Fi Touchscreen Photo Frame'] # this app had bad featurs so I droped it !
df["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in df["Installs"] ]
print('new form of installs column:')
df["Installs"]
def convert_str_to_numeric(value):

    if value.endswith("M"):
        return float(value.split("M")[0]) * 1000

    elif value.endswith("k"):
        return float(value.split("k")[0])


df['Size'] = df['Size'].apply(convert_str_to_numeric)
print('new form of size columns :')
df['Size']

df['Price'] = df['Price'].str.replace('$', '').astype(float)
data = df.sort_values(by='Installs', ascending=False)[['App', 'Installs']][:10]
data
df_ = df[df['Size'] != 'Varies with device']
df_ = df_.sort_values(by='Size', ascending=False)[['App', 'Size']][:10]
df_
sns.distplot(df['Rating'] , color='y')
df2 = df.copy()
df2['Reviews'] = df2['Reviews'].astype(float)
df2 = df2[df2['Reviews'] >= df2['Reviews'].mean()]
df2 = df2.sort_values(by='Rating', ascending=False)[['App', 'Rating', 'Reviews']][:10]
df2
uniq_category = df['Category'].value_counts()[:5]
uniq_category

sns.countplot(df['Category'])
plt.xticks(rotation=90)
plt.title('count of apps in each category',fontsize = 15)



print('count of most 10 genres :')
genres = df['Genres'].value_counts()[:10]
genres = genres.to_frame()
genres.columns = ['Genres count']
genres

a = df[(df['Price'] > 0) & (df['Price'] <= 1)]
b = df[(df['Price'] > 1) & (df['Price'] <= 2)]
c = df[(df['Price'] > 2) & (df['Price'] <= 3)]
d = df[(df['Price'] > 3) & (df['Price'] <= 4)]
e = df[(df['Price'] > 4) & (df['Price'] <= 5)]
f = df[df['Price'] > 5]

array = []
array_column_name = ['<1$', '1$<2$$', '2$<3$$', '3$<4$$', '4$<5$$', '5$<']
for i in [a, b, c, d, e, f]:
    array.append(i['App'].count())

df_ = pd.DataFrame({'lable' : array_column_name,'value' : array})

from matplotlib import cm
color = cm.inferno_r(np.linspace(.25, .8, 6))
df_.plot.bar(x='lable', y='value', rot=70
              ,color=color ,legend= False)
plt.title("*$* count of price ranges *$$*", fontsize = 15)
plt.show()

from sklearn import preprocessing

lb = preprocessing.LabelEncoder()

df['Genres'] = lb.fit_transform(df['Genres'])
df['Category'] = lb.fit_transform(df['Category'])
df['Content Rating'] = lb.fit_transform(df['Content Rating'])
df['Reviews'] = df['Reviews'].apply(lambda x: int(x))

sns.heatmap(df[['Category','Rating','Reviews','Installs', 'Size' ,'Genres','Price']].corr(),\
            annot=True, linewidths=0.5, fmt=".2f",cmap='winter')

from sklearn import preprocessing

lb = preprocessing.LabelEncoder()
ms = preprocessing.MinMaxScaler()

df['Genres'] = lb.fit_transform(df['Genres'])
df['Category'] = lb.fit_transform(df['Category'])
df['Type'] = df['Type'].map({'Free': 1, 'Paid': 0})
df['Content Rating'] = lb.fit_transform(df['Content Rating'])
df[['Size', 'Installs', 'Reviews']] = ms.fit_transform(df[['Size', 'Installs', 'Reviews']])
df = df.dropna()

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

reg = tree.DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=45)

y = df['Rating']
x = df[['Category', 'Rating', 'Reviews', 'Installs', 'Size', 'Genres', 'Price', 'Content Rating', 'Type']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=45)

reg.fit(x_train, y_train)
acc = reg.score(x_test, y_test)
pred = reg.predict(x_test)

print('Accuracy :{0:.7f}'.format(acc))
print('Mean Absolute Error:{0:.5f}'.format(metrics.mean_absolute_error(y_test, pred)))
print('Mean Squared Error:{0:.5f}'.format(metrics.mean_squared_error(y_test, pred)))
print('Root Mean Squared Error:{0:.5f}'.format(np.sqrt(metrics.mean_squared_error(y_test, pred))))