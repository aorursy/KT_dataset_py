import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df_train = pd.read_csv('../input/av-janata-hack-payment-default-prediction/train_20D8GL3.csv')
df_test = pd.read_csv('../input/av-janata-hack-payment-default-prediction/test_O6kKpvt.csv')
df_train.head()
df_test.head()
df_train.drop(['SEX','EDUCATION','MARRIAGE'], axis=1, inplace=True)
df_test.drop(['SEX','EDUCATION','MARRIAGE'], axis=1, inplace=True)
df_train.drop(['ID', 'default_payment_next_month'], axis=1).describe()
df_train.corr().style.background_gradient()
scaler = StandardScaler()
aux = df_train.loc[:, df_train.columns[9:-1]]
aux['LIMIT_BAL'] = df_train['LIMIT_BAL']
x = scaler.fit_transform(aux)
df_train.loc[:, df_train.columns[9:-1]] = x[:,0:-1]
df_train['LIMIT_BAL'] = x[:,-1]
df_train.head()
X_train, x_test, Y_train, y_test = train_test_split(df_train.loc[:, df_train.columns[1:-1]].values, df_train.loc[:, df_train.columns[-1]].values, test_size=0.3, random_state=42)
score = []
for value in np.arange(1, 50):
    knn = KNeighborsClassifier(n_neighbors=value)
    knn.fit(X_train,Y_train)
    score.append(knn.score(x_test, y_test))
plt.plot([index for index in np.arange(len(score))], score)
plt.scatter([index for index in np.arange(len(score))], score)
plt.show()
print('N = {}'.format(score.index(max(score))))
print('Score = {}'.format(max(score)))
knn = KNeighborsClassifier(n_neighbors=score.index(max(score)))
knn.fit(X_train,Y_train)
aux = df_test.loc[:, df_test.columns[9:]]
aux['LIMIT_BAL'] = df_test['LIMIT_BAL']
x = scaler.fit_transform(aux)
df_test.loc[:, df_test.columns[9:]] = x[:,0:-1]
df_test['LIMIT_BAL'] = x[:,-1]
predict = knn.predict_proba(df_test.loc[:, df_test.columns[1:]].values)
predict = pd.DataFrame(predict)
df_test['default_payment_next_month'] = predict[1]
df_test = df_test[['ID', 'default_payment_next_month']]
df_test.to_csv('output.csv', index=False, header=True)