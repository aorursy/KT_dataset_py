import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        sep=',',
        engine='python',
        index_col=['Id'],
        na_values="?",)
data.info()
data.describe()
data["capital.gain"].value_counts().plot(kind="bar", logy=True)
data["capital.loss"].value_counts().plot(kind="bar", logy=True)
data['capital.change'] = data['capital.gain'] - data['capital.loss']
data["capital.change"].value_counts().plot(kind="bar", logy=True)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data['income'] = encoder.fit_transform(data['income'])
data['sex'].value_counts()
data['race'].value_counts()
data = pd.get_dummies(data, columns=["sex","race"])
sns.heatmap(data.corr(), vmin=0., vmax=1., cmap = plt.cm.RdYlGn_r, annot=True )
data["native.country"].value_counts()
data = pd.get_dummies(data, columns=["workclass"])
data.drop_duplicates(keep="first", inplace=True)

data = data.drop(["fnlwgt",'education','relationship','occupation' ,'marital.status',"native.country","capital.loss","capital.gain"], axis=1)
data.head()
Y_train = data.pop('income')

X_train = data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

categorical_pipeline = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='if_binary'))
])
from sklearn.preprocessing import StandardScaler

numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])
from sklearn.preprocessing import RobustScaler

sparse_pipeline = Pipeline(steps=[
    ('scaler', RobustScaler())
])
from sklearn.compose import ColumnTransformer

cols_numericas = ['age','education.num','hours.per.week']
cols_escala = ['capital.change']
cols_cat = ['sex_Female','sex_Male','race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','workclass_Federal-gov','workclass_Local-gov','workclass_Never-worked','workclass_Private','workclass_Self-emp-inc','workclass_Self-emp-not-inc','workclass_State-gov','workclass_Without-pay']
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, cols_numericas),
    ('spr', sparse_pipeline, cols_escala),
    ('cat', categorical_pipeline,cols_cat)
])

X_train = preprocessor.fit_transform(X_train)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Vizinhos que serão testados:
neighbors = [5,10,15,20,25,30]

for n in neighbors:
    score = cross_val_score(KNeighborsClassifier(n_neighbors=n), X_train, Y_train, cv=13, scoring="accuracy").mean()
    
    print("N: ", n)
    print("Score: ", score)
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, Y_train)
test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?",
        index_col=['Id'])

test_data['capital.change'] = test_data['capital.gain'] - test_data['capital.loss']
test_data = test_data.drop(["fnlwgt",'education','relationship','occupation' ,'marital.status',"native.country","capital.loss","capital.gain"], axis=1)
test_data = pd.get_dummies(test_data, columns=["sex","race","workclass"])
col_names_test = ['age','education.num','hours.per.week','capital.change','sex_Female','sex_Male','race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White','workclass_Federal-gov','workclass_Local-gov','workclass_Never-worked','workclass_Private','workclass_Self-emp-inc','workclass_Self-emp-not-inc','workclass_State-gov','workclass_Without-pay']

test_data.columns = col_names_test 
X_test = test_data
X_test = preprocessor.transform(X_test)
prediction = knn.predict(X_test)
prediction
subs_map = {0: '<=50K', 1: '>50K'}
prediction_str = np.array([subs_map[i] for i in prediction], dtype=object)
submission = pd.DataFrame()
submission[0] = test_data.index
submission[1] = prediction_str
submission.columns = ['Id', 'Income']
submission.to_csv('submission.csv', index=False)