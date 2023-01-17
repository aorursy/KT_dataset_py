import pandas as pd
train=pd.read_csv("train.csv")
train.head()
features=train.drop(["Survived","Name","Ticket","Cabin"],axis=1)
labels=train.Survived
from category_encoders import OrdinalEncoder

oe=OrdinalEncoder()
features_encoded=oe.fit_transform(features)

features_encoded.head()

features_encoded.Fare.describe()

features_encoded.isnull
from sklearn.preprocessing import Imputer

im=Imputer()

features_encoded_imputed=im.fit_transform(features_encoded)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

features_encoded_imputed_scaled=sc.fit_transform(features_encoded_imputed)
from sklearn.decomposition import PCA

pc=PCA(n_components=4)

features_encoded_imputed_scaled_pca=pc.fit_transform(features_encoded_imputed_scaled)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(features_encoded_imputed,labels.values,test_size=0.3)

x_trainn,x_testt,y_trainn,y_testt=train_test_split(features_encoded_imputed_scaled,labels.values,test_size=0.3)

x_trainnn,x_testtt,y_trainnn,y_testtt=train_test_split(features_encoded_imputed_scaled_pca,labels.values,test_size=0.3)
from sklearn.naive_bayes import GaussianNB

NaiBay=GaussianNB()
from sklearn.ensemble import RandomForestClassifier

RanFor=RandomForestClassifier(n_estimators=100,random_state=1)
from sklearn.ensemble import GradientBoostingClassifier

GraBoo=GradientBoostingClassifier(n_estimators=1000,random_state=1)
from sklearn.metrics import f1_score

NaiBay.fit(x_train,y_train)

f1_score(y_test,NaiBay.predict(x_test))
NaiBay.fit(x_trainn,y_trainn)

f1_score(y_testt,NaiBay.predict(x_testt))
NaiBay.fit(x_trainnn,y_trainnn)

f1_score(y_testtt,NaiBay.predict(x_testtt))
RanFor.fit(x_train,y_train)

f1_score(y_test,RanFor.predict(x_test))
RanFor.fit(x_trainn,y_trainn)

f1_score(y_testt,RanFor.predict(x_testt))
RanFor.fit(x_trainnn,y_trainnn)

f1_score(y_testtt,RanFor.predict(x_testtt))
from sklearn.neural_network import MLPClassifier

NeuNet=MLPClassifier(random_state=1)
NeuNet.fit(x_train,y_train)

f1_score(y_test,NeuNet.predict(x_test))
NeuNet.fit(x_trainn,y_trainn)

f1_score(y_testt,NeuNet.predict(x_testt))
NeuNet.fit(x_trainnn,y_trainnn)

f1_score(y_testtt,NeuNet.predict(x_testtt))
param_grid = {'n_estimators': [10, 100, 500, 1000, 1500, 2000]}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(RanFor, param_grid=param_grid, cv=5)
grid.fit(x_trainn, y_trainn)
grid.best_params_
grid.best_score_
RanFor=RandomForestClassifier(n_estimators=1500,random_state=1)
RanFor.fit(x_trainn,y_trainn)
f1_score(y_testt,RanFor.predict(x_testt))