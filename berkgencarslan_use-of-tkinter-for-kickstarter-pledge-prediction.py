import tkinter as tk

import pandas as pd



def compute():

    a = e1.get()

    a2 = e2.get()

    a3 = e3.get()

    a4 = e4.get()

    a6 = e6.get()

    a7 = e7.get()

    global c

    c = pd.DataFrame({'name' : [a],'category': [a2],'main_category': [a3],'backers': [a4],'state': 1,'launched':[a6],'deadline':[a7]})

    print(c)

    return





master = tk.Tk()

master.wm_title("Kickstarter Campaign Adviser")



tk.Label(master, text="Name").grid(row=0)

tk.Label(master, text="Category").grid(row=1)

tk.Label(master, text="Main Category").grid(row=2)

tk.Label(master, text="Virality").grid(row=3)

tk.Label(master, text="Launched").grid(row=4)

tk.Label(master, text="Deadline").grid(row=5)



e1 = tk.Entry(master)

e2 = tk.Entry(master)

e3 = tk.Entry(master)

e4 = tk.Entry(master)

e6 = tk.Entry(master)

e7 = tk.Entry(master)



e1.grid(row=0, column=1)

e2.grid(row=1, column=1)

e3.grid(row=2, column=1)

e4.grid(row=3, column=1)

e6.grid(row=4, column=1)

e7.grid(row=5, column=1)



tk.Button(master, text='Quit', command=master.quit).grid(row=6, column=0, sticky=tk.W, pady=4)

tk.Button(master, text='Compute', command=compute).grid(row=6, column=1, sticky=tk.W, pady=4)





tk.mainloop()



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats



from sklearn.preprocessing import minmax_scale

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_validate



dataset = pd.read_csv("ks-projects-201801.csv")

dataset = dataset.dropna()

dataset = dataset.loc[dataset['state'].isin(['failed','successful'])]

targ_dict = {'failed': 0,'successful': 1 }

dataset['state'] = dataset['state'].map(targ_dict)
dataset = dataset.reset_index()

dataset.drop("index",axis= 1 ,inplace =True)



dataset.drop(dataset[dataset.usd_pledged_real < 100].index, inplace=True)

dataset.drop(dataset[dataset.usd_pledged_real > 1000000].index, inplace=True)



dataset = dataset.reset_index()

dataset.drop("index",axis= 1 ,inplace =True)
print(dataset["usd_pledged_real"].mean())

print(dataset["usd_pledged_real"].median())



print(dataset["backers"].mean())

print(dataset["backers"].median())
dataset.loc[dataset['backers'] < 100, 'backers'] = 1

dataset.loc[dataset['backers'] > 4500, 'backers'] = 4

dataset.loc[(dataset['backers'] >= 1500) & (dataset['backers'] <= 4500),'backers'] = 3

dataset.loc[(dataset['backers'] >= 100) & (dataset['backers'] <= 1500),'backers'] = 2
y = dataset['usd_pledged_real']

X = dataset.drop('usd_pledged_real',axis=1)
X = X.append(c, ignore_index=True)



X["num_words"]        = X["name"].apply(lambda x: len(x.split()))

X["num_chars"]        = X["name"].apply(lambda x: len(x.replace(" ","")))

X['launched'] = pd.to_datetime(X['launched'])

X['launched_date'] = X['launched'].dt.date

X["launched_week"]    = X["launched"].dt.week

X['launched'] = pd.to_datetime(X['launched'])

X['launched_date'] = X['launched'].dt.date

X['deadline'] = pd.to_datetime(X['deadline'])

X['deadline_date'] = X['deadline'].dt.date

X["launched_day"]     = X["launched"].dt.weekday

X["is_weekend"]       = X["launched_day"].apply(lambda x: 1 if x > 4 else 0)

X['time_campaign_d'] = (X['deadline_date'] - X['launched_date']).dt.days

X['time_campaign_d'] = X['time_campaign_d'].astype(int)

X = X[X['time_campaign_d'] != 14867]





to_drop = ['ID', 'name','goal','deadline', 'pledged', 'currency', 'launched','country','usd_goal_real','usd pledged','launched_date', 'deadline_date']



X.drop(to_drop, axis=1, inplace=True)

#removing outlier value





X = pd.get_dummies(X, columns=['category', 'main_category'],\

                          prefix=['cat', 'main_cat'], drop_first=True)





c2 = X.iloc[232441, :]

X.drop(232441, inplace=True)
import statsmodels.api as sm



results = sm.OLS(y, X.astype(float)).fit()

print(results.summary())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from math import sqrt

from sklearn.metrics import r2_score

from sklearn import metrics

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)

print("Linear Regression")

print(lin_reg.predict(X_test))

print("Linear Regression r2")

print(r2_score(y_test,lin_reg.predict(X_test)))



op = lin_reg.predict(X_test)

print(metrics.mean_absolute_error(y_test, op))
op2 = lin_reg.predict(sc.transform(np.array([c2])))

print(op2)
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(X_train,y_train)

Z = X_test + 0.5

K = X_test - 0.4

print("Decision Tree")

print(r_dt.predict(X_test))

print("Decision Tree R2 degeri:")

print(r2_score(y_test, r_dt.predict(X_test)))

dh = r_dt.predict(X_test)

print(metrics.mean_absolute_error(y_test, dh))
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)

rf_reg.fit(X_train,y_train)

print("Random Forest")

print(rf_reg.predict(X_test))

print("Random Forest R2 degeri:")

print(r2_score(y_test, rf_reg.predict(X_test)) )

print(r2_score(y_test, rf_reg.predict(K)) )

print(r2_score(y_test, rf_reg.predict(Z)) )

rf =  rf_reg.predict(X_test)

print(metrics.mean_absolute_error(y_test, rf))
datas = pd.DataFrame({'y_test': y_test[:,], 'rf': rf[:,]})

print(datas)
sns.jointplot(x="rf",y="y_test",data=datas,color='c')
sns.lmplot(x="rf",y="y_test",data=datas)
op3 = rf_reg.predict(sc.transform(np.array([c2])))

print(op3)