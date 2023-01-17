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
df_w = pd.read_csv(os.path.join(dirname, filename))
df_w
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
def basic_info(data):

    categorical = []

    numerical = []

    print("Size = ", data.size)

    print("Shape = ", data.shape)

    data.info()

    for i in data.columns:

        if data[i].dtype == object:

            categorical.append(i)

        else:

            numerical.append(i)

    return categorical, numerical
basic_info(df_w)
df_w['Date'] = pd.to_datetime(df_w['Date'])
categorical, numerical = basic_info(df_w)
categorical
numerical
df_w.isnull().sum()
df_w = df_w.dropna()
basic_info(df_w)
df_w['Rating'].value_counts()
plt.figure(figsize=(15,8))

sns.countplot(df_w['Rating'], label = "Rating")

plt.legend()

plt.show()
df_w['Category'].value_counts()
plt.figure(figsize=(15,8))

sns.countplot(df_w['Category'], label = "Category counts")

plt.xticks(rotation = -45)

plt.legend()

plt.show()
plt.figure(figsize=(30,10))

sns.countplot(df_w['Rating'] ,hue = df_w['Category'])
df_w['Price'].value_counts()
def change_price(x):

    if x == "Free":

        #print(x)

        x = 0.0

        return x

    else:

        #print(x)

        x = x[2:]

        x = x.replace(",", "")

        x = float(x)

        return x
df_w['Price'] = df_w['Price'].apply(lambda x: change_price(x))
df_w['Price'].dtype
cat, num = basic_info(df_w)
cat
num
plt.figure(figsize=(20,13))

plt.style.use('seaborn-white')

ax = plt.subplot(221)

sns.boxplot(df_w['Rating'])

ax = plt.subplot(222)

sns.boxplot(df_w['Price'])

ax = plt.subplot(223)

sns.boxplot(df_w['No of people Rated'])
def free_or_not(x):

    if x == 0.0:

        return 0.0

    else:

        return 1.0
df_w['Price'] = df_w['Price'].apply(lambda x: free_or_not(x))
df_w['Price'].value_counts()
def making_new_df(data, columnlist):

    for i in columnlist:

        dummy = pd.get_dummies(data[i])

        #print(dummy)

        del dummy[dummy.columns[-1]]

        data = pd.concat([data, dummy], axis = 1)

    return data
df_w2 = making_new_df(df_w, ['Category'])
df_w2
df_w2 = df_w2.drop(['Name', "Category", "Date", "No of people Rated"], axis = 1)
df_w2
from sklearn.model_selection import train_test_split
X = df_w2.loc[:, df_w2.columns != 'Rating']

y = df_w2['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X_train, y_train)
y_predict = model.predict(X_test)
predict_dataframe = pd.DataFrame(data={"Predicted": y_predict, "Actual": y_test})
predict_dataframe
model.score(X_test, y_test)
plt.plot(predict_dataframe["Predicted"][:20], "*")

plt.plot(predict_dataframe['Actual'][:20], "^")

plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring = 'neg_mean_squared_error', cv =5)
modelR = ridge_regressor.fit(X_train, y_train)
y_predict_R = modelR.predict(X_test)
rigid_df = pd.DataFrame(data = {"Predicted": y_predict_R, "Actual": y_test})
rigid_df
modelR.score(X_test, y_test)
from sklearn.svm import SVR
svr = SVR()
model_svr = svr.fit(X_train, y_train)
y_predict_svr = model_svr.predict(X_test)
svr_df = pd.DataFrame(data = {"Predicted": y_predict_svr, "Actual": y_test})
svr_df
model_svr.score(X_test, y_test)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
modef_rfr = rfr.fit(X_train, y_train)
y_predict_rfr = modef_rfr.predict(X_test)
rfr_df = pd.DataFrame(data={"Predicted": y_predict_rfr, "Actual": y_test})
rfr_df
modef_rfr.score(X_test, y_test)
plt.plot(rfr_df["Predicted"][:20], "*")

plt.plot(rfr_df['Actual'][:20], "^")

plt.show()
print("Linear Regression score is: ", model.score(X_test, y_test))

print("Ridge Regression score is: ", modelR.score(X_test, y_test))

print("Support Vector Regression score is: ", model_svr.score(X_test, y_test))

print("Random Forest Regression score is: ", modef_rfr.score(X_test, y_test))
fig, ax=plt.subplots()

plt.rcParams['text.color'] = 'black'

labels = ['Free', "Paid"]

sizes = df_w['Price'].value_counts()

percent = 100*np.array(sizes)/np.array(sizes).sum() #to show % of every category

patches, texts = ax.pie(sizes, shadow=True, startangle=90)

labels = ['{0} - {1:1.2f}%'.format(i,j) for i, j in zip(labels,percent)]

sort_legend = False

ax.axis('equal')



plt.legend(patches, labels, loc= 'best', bbox_to_anchor=(-0.1, 1.), fontsize = 10)

ax.axis('equal')

plt.show()
df_w['Date'].dt.year.unique() # use dt to use attributes such as year, month and more
unique_dates = df_w['Date'].dt.year.unique()

#df_new = df_w[df_w['Date'].dt.year == 2014]

#df_new

unique_dates.sort()

print(unique_dates)

sum_array = list()

for i in unique_dates:

    df_new = df_w[df_w['Date'].dt.year == i]

    sum_array.append(sum(df_new['No of people Rated']))



print(sum_array)



plt.figure(figsize = (20,8))

plt.style.use('seaborn-darkgrid')



plt.plot(unique_dates, sum_array)

plt.xlabel("Years")

plt.ylabel("No of People Rated (k)")

for x, y in zip(unique_dates, sum_array):

    plt.text(x, y, str(y))