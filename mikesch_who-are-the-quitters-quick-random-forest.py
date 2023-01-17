import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')



# machine learning

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
df1 = pd.read_csv('../input/HR_comma_sep.csv', header=0)
#check for missing values

df1.info()
#quick look at data for features that will be used in algo

df1.iloc[:,:5].describe()
#create x & y variables for algo

cols  = df1.columns.tolist()

order = [0,1,2,3,4,5,7,8,9,6]

cols  = [cols[i] for i in order]

df1   = df1[cols]

x     = df1.iloc[:,:5]

y     = df1.iloc[:,-1]
#normalize data & split into train, test groups

x_scale = preprocessing.StandardScaler().fit(x)

x_norm = x_scale.transform(x)



x_train, x_test, y_train, y_test = train_test_split(x_norm,y, test_size=.4, random_state=42)
#logistic regression (to compare against random forest)



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred_lr = logreg.predict(x_test)



logreg.score(x_train, y_train)
#how many did predictions did we get wrong

print('# of wrong predictions:', sum(y_test - y_pred_lr))

print('% of wrong predictions: {:.2f}%'.format((sum(y_test - y_pred_lr)/len(y_test)*100)))
#random forest

random_forest = RandomForestClassifier(n_estimators=100, oob_score=True)

random_forest.fit(x_train, y_train)

y_pred_rf = random_forest.predict(x_test)



random_forest.score(x_train, y_train)
#how many did predictions did we get wrong

print('# of wrong predictions:', sum(y_test - y_pred_rf))

print('% of wrong predictions: {:.2f}%'.format((sum(y_test - y_pred_rf)/len(y_test)*100)))
#plot feature importance of random forest

column_list = []

for i in range(0,5):

    column_list.append(df1.columns[i])  



feature_df = pd.DataFrame(random_forest.feature_importances_, index=column_list).sort_values(by=0, ascending=False)



sns.barplot(x=feature_df.index, y=feature_df.iloc[:,0])

plt.ylabel('Feature Importance')

plt.xticks(size=10, rotation=20)