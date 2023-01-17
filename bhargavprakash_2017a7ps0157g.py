import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



pd.set_option('display.max_columns', 100)
df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
df.shape
df.head()
df.isnull().sum()
df.dropna(axis=0, inplace=True)

df.isnull().sum()
X = df[["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "type"]].copy()

y = df["rating"].copy()
X_encoded =  pd.get_dummies(X)

X_encoded.head()
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_scaled = scaler.fit_transform(X_encoded) 

X_scaled

font = {'fontname':'Arial', 'size':'14'}

title_font = { 'weight' : 'bold','size':'16'}

plt.hist(df['rating'], bins=20)

plt.title("ratings")

plt.show()
corr = df.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(12, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
sns.regplot(x='feature5', y='rating', data=df)
sns.regplot(x='feature6', y='rating', data=df)
sns.regplot(x='feature7', y='rating', data=df)
sns.boxplot(x='type', y='rating', data=df)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics



model = GaussianNB()

model.fit(X_train,y_train)

y_pred = model.predict(X_test) 



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_train)



print('Train accuracy score:',accuracy_score(y_train,y_pred))

print('Test accuracy score:', accuracy_score(y_test,logreg.predict(X_test)))
from sklearn.ensemble import RandomForestClassifier as RFC



rfc_b = RFC(n_estimators=250, min_samples_split=6)

rfc_b.fit(X_train,y_train)

y_pred = rfc_b.predict(X_train)



print('Train accuracy score:',accuracy_score(y_train,y_pred))

print('Test accuracy score:', accuracy_score(y_test,rfc_b.predict(X_test)))
from sklearn.svm import SVC 



clf = SVC(kernel='rbf', gamma=100)  

clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)



print('Test accuracy score:', accuracy_score(y_test,y_pred))
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

params_knn = {'n_neighbors': np.arange(1, 25)}

knn_gs = GridSearchCV(knn, params_knn, cv=5)

knn_gs.fit(X_train, y_train)

knn_best = knn_gs.best_estimator_

print(knn_gs.best_params_)



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

params_rf = {'n_estimators': [50, 100, 200]}

rf_gs = GridSearchCV(rf, params_rf, cv=5)

rf_gs.fit(X_train, y_train)

rf_best = rf_gs.best_estimator_

print(rf_gs.best_params_)



from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)



print('knn: {}'.format(knn_best.score(X_test, y_test)))

print('rf: {}'.format(rf_best.score(X_test, y_test)))

print('log_reg: {}'.format(log_reg.score(X_test, y_test)))



from sklearn.ensemble import VotingClassifier

estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]

ensemble = VotingClassifier(estimators, voting='hard')



ensemble.fit(X_train, y_train)

ensemble.score(X_test, y_test)

df1 = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
df1.shape
numeric_features = ['feature3', 'feature4', 'feature5', 'feature8', 'feature9', 'feature10', 'feature11']

for col in numeric_features:

    df1[col].fillna(df1[col].mean(), inplace=True)

df1.isnull().sum()
X_pred = df1[["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "type"]].copy()
X_pred_encoded =  pd.get_dummies(X_pred)

X_pred_encoded.head()
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_pred_scaled = scaler.fit_transform(X_pred_encoded) 



X_pred_scaled
X_pred_encoded.shape
predictions = rfc_b.predict(X_pred_scaled)
predictions.mean()
predictions.std()
submission = pd.DataFrame({'id':df1['id'],'rating':predictions})

submission.shape
filename = 'rating predictions14.csv'



submission.to_csv(filename,index=False)
predictions2 = ensemble.predict(X_pred_encoded)
predictions2.mean()
predictions2.std()
submission = pd.DataFrame({'id':df1['id'],'rating':predictions2})

submission.shape
filename = 'rating predictions15.csv'



submission.to_csv(filename,index=False)