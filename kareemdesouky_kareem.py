# import first

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

# change the style from the very beging

plt.style.use('ggplot')

%matplotlib inline
df = pd.read_csv('../input/1-predict-if-a-customer-will-leave-the-company/churn_train.csv')

df.head()
df.info()
df.describe().transpose()
df.isna().sum()
df.dtypes
# check how many unique values each feature has:

for column in df.columns:

    print(column, len(df[column].unique()))
corr = df.corr()['Churn Status']

corr
corr = df.corr()

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr,annot=True)
df.drop('Customer ID' , inplace=True , axis=1)

df
df=pd.get_dummies(df,columns=['Most Loved Competitor network in in Month 1','Most Loved Competitor network in in Month 2',

                                     'Network type subscription in Month 1','Network type subscription in Month 2'] )
x = df.drop('Churn Status',axis=1)

x
y = df.iloc[:,-1]

y
# x['Total Data Consumption'] = x['Total Data Consumption'].apply(lambda x: np.log(x))

# x['Total Spend in Months 1 and 2 of 2017'] = x['Total Spend in Months 1 and 2 of 2017'].apply(lambda x: np.log(x))

# x
# standrize the values

from sklearn.preprocessing import StandardScaler , MinMaxScaler

scaler = StandardScaler()

x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 ,random_state=0)
# from sklearn.linear_model import LogisticRegression

# cl = LogisticRegression(random_state = 0)

# # cl.fit(x_train,y_train)

# from sklearn.naive_bayes import GaussianNB , BernoulliNB , MultinomialNB

# cl = GaussianNB()

# cl.fit(x_train,y_train)
from sklearn.neighbors import KNeighborsClassifier 

cl = KNeighborsClassifier(n_neighbors = 22)

cl.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, cl.predict(x_test))

pd.DataFrame(cm)
cl.score(x_test,y_test)