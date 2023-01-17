import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score, KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

import lightgbm as lgb

from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import VotingClassifier

import time

from contextlib import contextmanager
churn_data = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
churn_data.info() # NAN value bulunmamaktadir.
churn_data.head()
churn_data.drop(labels=['RowNumber','CustomerId','Surname'],

                axis=1,

                inplace=True)
churn_data.head()
categorical_features = ["Geography","Gender","NumOfProducts","HasCrCard","IsActiveMember"]



numerical_features = ["CreditScore","Age","Tenure","Balance","EstimatedSalary"]



target = "Exited"
churn_data[numerical_features].describe()
churn_data[numerical_features].hist(bins=30, figsize=(10, 10));
fig, ax = plt.subplots(1, 5, figsize=(30, 5))

churn_data[churn_data.Exited == 0][numerical_features].hist(bins=30, color="blue", alpha=0.5, ax=ax);

churn_data[churn_data.Exited == 1][numerical_features].hist(bins=30, color="red", alpha=0.5, ax=ax);
g = sns.pairplot(churn_data,hue = 'Exited',corner=True)
# Import label encoder 

from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column 'species'. 

churn_data["Geography"] = label_encoder.fit_transform(churn_data["Geography"])

churn_data["Gender"] = label_encoder.fit_transform(churn_data["Gender"])
churn_data.head()
churn_data[numerical_features].describe()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 

data_scaled = scaler.fit_transform(churn_data[numerical_features])
churn_data[numerical_features] = data_scaled
churn_data.head()
df = churn_data

X = df.drop(['Exited'], axis=1)

y = df["Exited"]

    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Veri kümesi oluşturalım.

lgb_train = lgb.Dataset(data=X_train, label=y_train,  free_raw_data=False)

# Değerlendirme veri kümesini oluşturuyoruz.

lgb_eval = lgb.Dataset(data=X_test, label=y_test, reference=lgb_train,  free_raw_data=False)



# Eğitim parametrelerini belirleyelim

params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'num_leaves': 31,

    'learning_rate': 0.05,

    'verbose': -1

}



# Kategorik değişkenlerin indeksleri

kategorik_indeks = [1,2,6,7,8]

print('Kategorik değişkenler: ' + str(churn_data.columns[kategorik_indeks].values))



print('Eğitim...')

# Modeli eğitelim

evals_result={}

gbm = lgb.train(params,

                lgb_train,

                valid_sets = lgb_eval,

                categorical_feature = kategorik_indeks,

                num_boost_round= 150,

                early_stopping_rounds= 25,

                evals_result=evals_result)

print('Eğitim bitti...')



# Tahmin ve değerlendirme

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)



print('En iyi sonucu veren iterasyon: ', gbm.best_iteration)

print('Eğri altı alan değeri:', roc_auc_score(y_test, y_pred))

print('İsabetlilik değeri:', accuracy_score(y_test, ( y_pred>= 0.5)*1))
print('Eğri altı alan...')

ax = lgb.plot_metric(evals_result, metric='auc')

ax.set_title('Eğri Altı Alanın İterasyona Göre Değişimi')

ax.set_xlabel('İterasyon')

ax.set_ylabel('Eğri Altı Alan Değeri')

ax.legend_.remove()

plt.show()
ax = lgb.plot_importance(gbm, max_num_features=10)

ax.set_title('')

ax.set_xlabel('Özniteliklerin Önemi')

ax.set_ylabel('Öznitelikler')

plt.show()
y_pred