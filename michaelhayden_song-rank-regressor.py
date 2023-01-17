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
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression





# had to add the encoding paramarater beccause there was an issue with an uft-8

import pandas as pd

df = pd.read_csv("../input/top50spotify2019/top50.csv", encoding='ISO-8859-1')
#look at what we have

df.head()
#change name of unamed to rank so we can build a model to see how well it can guess the rank of the test set.

df.rename(columns={'Unnamed: 0':'Rank'}, inplace=True)
#make sure we changed unnamed to Rank

df.columns
corr = df.corr()

plt.figure(figsize= (16,12))

sns.heatmap(corr,

           xticklabels=corr.columns.values,

           yticklabels=corr.columns.values)
#converting categorical info into binary info

#df_numerial is the numerical data types

df_numerical = df.select_dtypes(include=['int64','float64'])

df_categorical = df.select_dtypes(exclude=['int64','float64'])
#creating dummy variables for the categorical columns. Look at the head to see what we have

categorical_list = df_categorical.columns.tolist()

df_dummies=pd.get_dummies(df_categorical,columns=categorical_list)

df_dummies.head()
result = pd.concat([df_numerical.merge(df_dummies, left_index=True, right_index=True)])
result.head()
#root mean squared error

df_X_transformed = result.copy()

print(df_X_transformed)
df_y = df.Rank
df_X_transformed = df_X_transformed.drop("Rank", axis=1)
#split between train and test set

X_train, X_test, y_train, y_test = train_test_split(df_X_transformed, df_y, test_size=.10, random_state=20)
#lets fit this model

linreg = LinearRegression()

linreg.fit(X_train,y_train)
#building the models

X_train.shape, X_test.shape
X_train.columns
#build and train the model to see if the songs fall at the rank they deserve

rf_model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=42)

rf_model.fit(X_train, y_train)
rf_test_pred = rf_model.predict(X_test)
rf_model.score(X_test, y_test)
plt.figure(figsize =(10,10))

plt.scatter(rf_test_pred, y_test, alpha=.1, c='blue')

plt.plot(np.linspace(0,50, 5), np.linspace(0, 50, 5), 'r-');

plt.xlabel("Predicted")

plt.ylabel("Actual")
# Test set RMSE. A score of .228 is not great. I believe the factors that lead to this include small test size as well as the small sampole size.

rmse_test = np.sqrt(metrics.mean_squared_error(y_test, linreg.predict(X_test)))

rmse_test = rmse_test/(max(y_test) - min(y_test))

print ("Test set RMSE: ", rmse_test)


df['Genre'].value_counts().plot.bar()

plt.title('Count by Genre')

plt.ylabel('quanity')

plt.xlabel('Genre')

plt.show()
#each in denotes 5 integers

ax = df['Beats.Per.Minute'].plot.hist(bins=20, alpha=0.5)

plt.title("Histogram of Beats per minute")

plt.xlabel('Beat Count')

plt.ylabel('Total')

#plt.show()
# Analysing the relationship between energy and loudness

fig=plt.subplots(figsize=(10,10))

sns.regplot(x='Energy',y='Loudness..dB..',data=df,color='black')
# Analysing the relationship between energy and loudness

fig=plt.subplots(figsize=(10,10))

sns.regplot(x='Speechiness.',y='Beats.Per.Minute',data=df,color='black')
df.head()
submisison = pd.DataFrame(df, columns=['Rank'])

result = pd.concat([])