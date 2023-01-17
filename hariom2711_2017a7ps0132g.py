import pandas as pd

import seaborn as sns

import sklearn as skl



df = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

df.head()



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(['old','new'])



df['type'] = le.transform(df['type'])



df = df.dropna()



X = df.iloc[:, 1:13]

y = df['rating']



from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=500)
clf.fit(x_train, y_train)
from sklearn.metrics import mean_squared_error

y_pred = clf.predict(x_val)

# #y_pred_2 = clf2.predict(x_val)



for i in range(0, len(y_pred)):

    if y_pred[i] < 1:

        y_pred[i] = 1

    elif y_pred[i] > 6:

        y_pred[i] = 6

acc1 = mean_squared_error(y_pred,y_val)

print("Accuracy score of clf1: {}".format(acc1))

df_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')



le = preprocessing.LabelEncoder()

le.fit(['old','new'])



df_test['type'] = le.transform(df_test['type'])



#df['feature3']= df['feature3'].fillna(df['feature3'].mean())

#df.isnull().sum()

df_test = df_test.fillna(df_test.mean())

xTest = df_test.iloc[:, 1:]

y_test_p = clf.predict(xTest)



for i in range(0, len(y_test_p)):

    if y_test_p[i] < 1:

        y_test_p[i] = 1

    elif y_test_p[i] > 6:

        y_test_p[i] = 6

    else:

        y_test_p[i] = round(y_test_p[i])

y_test_p
y_pd=pd.Series(y_test_p)

y_pd

y_pd1=df_test['id']

ans = pd.DataFrame()

ans['id'] = y_pd1

ans['rating'] = y_pd

ans.to_csv('soln.csv', index=False)