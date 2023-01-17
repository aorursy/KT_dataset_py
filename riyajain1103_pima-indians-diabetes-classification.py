# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.columns
df.isnull().sum()
X=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
X.head()
y=df['Outcome']
y.head()
X.info()
X.shape
corr=df.corr()
sns.heatmap(corr,annot=True)
sns.distplot(X['Pregnancies'])
sns.distplot(X['Pregnancies'],kde=False)
X['Pregnancies'].max()
X['Pregnancies'].min()
X['Pregnancies'].mode()
X['Pregnancies'].mean()
a=df['Pregnancies']==17
b=df[a]
b.shape
b.head()
c=df['Pregnancies']==df['Pregnancies'].min()
d=df[c]
d.head()
sns.catplot(x='Outcome',y='Pregnancies',data=df)
sns.catplot(x='Outcome',y='Pregnancies',kind='box',data=df)
c=df['Glucose']==df['Glucose'].min()
d=df[c]
d.head()
c=df['Glucose']==df['Glucose'].max()
d=df[c]
d.head()
c=df['Glucose']==df['Glucose'].median()
d=df[c]
d.head()
df['Glucose'].mean()
c=df['Glucose']>df['Glucose'].mean()
d=df[c]
d.head(700)
d['Outcome'].mode()
d['Outcome'].value_counts()
c=df['Glucose']<df['Glucose'].mean()
d=df[c]
d.head(700)
d['Outcome'].mode()
d['Outcome'].count()
d['Outcome'].value_counts()
sns.distplot(X['Glucose'])
sns.distplot(X['Glucose'],kde=False)
sns.catplot(x='Outcome',y='Glucose',data=df)
df['BloodPressure'].mean()
df['BloodPressure'].max()
df['BloodPressure'].min()
df['BloodPressure'].mode()
df['BloodPressure'].value_counts()
c=df['BloodPressure']<df['BloodPressure'].mean()
d=df[c]
d.head(700)

d['Outcome'].mode()
d['Outcome'].count()
d['Outcome'].value_counts()
c=df['BloodPressure']>df['BloodPressure'].mean()
d=df[c]
d.head(700)

d['Outcome'].mode()
d['Outcome'].count()
d['Outcome'].value_counts()
sns.distplot(X['BloodPressure'],kde=False)
sns.catplot(x='Outcome',y='BloodPressure',data=df)
df['SkinThickness'].mean()
df['SkinThickness'].max()
df['SkinThickness'].min()
sns.catplot(x='Outcome',y='SkinThickness',data=df)
c=df['SkinThickness']<df['SkinThickness'].mean()
d=df[c]
d.head(700)
d['SkinThickness'].value_counts()
d['Outcome'].mode()
d['Outcome'].count()
d['Outcome'].value_counts()

105/(105+246)
c=df['SkinThickness']>df['SkinThickness'].mean()
d=df[c]
d.head(700)
d['SkinThickness'].value_counts()
d['Outcome'].value_counts()

163/(163+254)
#We saw percentage of diabetic patient is more when skin thickness> than its mean value.
sns.distplot(X['SkinThickness'],kde=False)
sns.heatmap(corr,annot=True)
# correlation of skin thickness and blood pressure is comparatively more.
df.plot(x='SkinThickness',y='BloodPressure',style='o')
sns.lineplot(x=df['SkinThickness'],y=df['BloodPressure'])
df['Insulin'].describe()
c=df['Insulin']==846
d=df[c]
d.head()
c=df['Insulin']==0
d=df[c]
d.head()
sns.catplot(x='Outcome',y='Insulin',data=df)
sns.distplot(X['Insulin'],kde=False)
c=df['Insulin']>df['Insulin'].mean()
d=df[c]
d.head(700)
d['Insulin'].value_counts()
d['Outcome'].value_counts()
121/(121+168)
c=df['Insulin']<df['Insulin'].mean()
d=df[c]
d.head(700)
d['Insulin'].value_counts()
d['Outcome'].value_counts()
147/(147+332)
sns.heatmap(corr,annot=True)
df.plot(x='Insulin',y='BloodPressure',style='o')
sns.lineplot(x=df['Insulin'],y=df['BloodPressure'])
sns.lineplot(x=df['Insulin'],y=df['SkinThickness'])
sns.relplot(x="Insulin", y="BloodPressure", hue="SkinThickness", data=df);

df['BMI'].describe()
c=df['BMI']<df['BMI'].mean()
d=df[c]
d.head(700)

d['Outcome'].value_counts()
84/(84+289)
d['BMI'].value_counts()
c=df['BMI']>df['BMI'].mean()
d=df[c]
d.head(700)
d['Outcome'].value_counts()
184/(184+211)
sns.distplot(X['BMI'],kde=False,color='brown')
sns.catplot(x='Outcome',y='BMI',data=df)
sns.lineplot(x=df['BMI'],y=df['Glucose'],color='orange')
sns.lineplot(x=df['BMI'],y=df['BloodPressure'],color='red')
sns.lineplot(x=df['BMI'],y=df['Insulin'])
sns.relplot(x="SkinThickness", y="BloodPressure", hue="BMI", data=df);

sns.relplot(x="SkinThickness", y="BloodPressure", hue="Outcome", data=df,color='purple');

sns.relplot(x="SkinThickness", y="Insulin", hue="Outcome", data=df,color='purple');

sns.relplot(x="SkinThickness", y="Pregnancies", hue="Outcome", data=df,color='purple');

df['DiabetesPedigreeFunction'].describe()
sns.distplot(X['DiabetesPedigreeFunction'],kde=False)
sns.catplot(x='Outcome',y='DiabetesPedigreeFunction',data=df)
c=df['DiabetesPedigreeFunction']<df['DiabetesPedigreeFunction'].mean()
d=df[c]
d.head(700)

d['Outcome'].value_counts()
139/(139+334)
c=df['DiabetesPedigreeFunction']>df['DiabetesPedigreeFunction'].mean()
d=df[c]
d.head(700)
d['Outcome'].value_counts()
129/(129+166)
sns.lineplot(x=df['DiabetesPedigreeFunction'],y=df['Insulin'],color='pink')
sns.lineplot(x=df['DiabetesPedigreeFunction'],y=df['Glucose'],color='purple')
sns.lineplot(x=df['DiabetesPedigreeFunction'],y=df['SkinThickness'],color='green')
df['Age'].describe()
sns.boxplot(df['Age'],color='silver')
sns.distplot(df['Age'],kde=False,color='pink')
sns.catplot(x='Outcome',y='Age',data=df)
sns.relplot(x="Age", y="Pregnancies", hue="Outcome", data=df,color='purple');

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X = sc_X.fit_transform(X)

from sklearn import preprocessing
# from sklearn.preprocessing import Normalizer
X = preprocessing.normalize(X, norm='l2')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



X # Normalised X
#Fitting Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from yellowbrick.model_selection import LearningCurve
visualizer = LearningCurve(
    classifier, cv=10, scoring='f1_weighted'
)

visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show()   
classifier.get_params()
from sklearn.model_selection import GridSearchCV
grid={"C":np.logspace(-3,3,4,5,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
grid_cv=GridSearchCV(classifier,grid,cv=10)
grid_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",grid_cv.best_params_)
print("accuracy :",grid_cv.best_score_)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv =2)
accuracies.std()
accuracies.mean()
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)
cm
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
y_prob = classifier.predict_proba(X_test)
y_prob
y_pred1= []
for i in range(len(y_prob)):
    if y_prob[i][0]>0.7: 
        y_pred1.append(1)
    else:
        y_pred1.append(0)
y_pred1
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred1)
from sklearn.metrics import roc_curve
roc_curve(y_test,y_pred)
# # Fitting XGBoost to the Training set
# from xgboost import XGBClassifier
# classifier = XGBClassifier(n_estimators=500,learning_rate=0.1,verbosity=1)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

# # Applying k-Fold Cross Validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv =2)
# accuracies.std()
# accuracies.mean()

# print(cm)
# from sklearn.model_selection import GridSearchCV
# params = {
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5]
#         }
# xgb=XGBClassifier()
# clf = GridSearchCV(xgb, params)
# clf.fit(X_train, y_train)
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
# import lightgbm as lgbm
# from sklearn.ensemble import VotingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import roc_curve,auc
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_predict
# from yellowbrick.classifier import DiscriminationThreshold

# # Stats
# import scipy.stats as ss
# from scipy import interp
# from scipy.stats import randint as sp_randint
# from scipy.stats import uniform as sp_uniform

# #


# random_state=42

# fit_params = {"early_stopping_rounds" : 100, 
#              "eval_metric" : 'auc', 
#              "eval_set" : [(X,y)],
#              'eval_names': ['valid'],
#              'verbose': 0,
#              'categorical_feature': 'auto'}

# param_test = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
#               'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],
#               'num_leaves': sp_randint(6, 50), 
#               'min_child_samples': sp_randint(100, 500), 
#               'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
#               'subsample': sp_uniform(loc=0.2, scale=0.8), 
#               'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
#               'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
#               'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
#               'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

# #number of combinations
# n_iter = 300

# #intialize lgbm and lunch the search
# lgbm_clf = lgbm.LGBMClassifier(random_state=random_state, silent=True, metric='None', n_jobs=4)
# grid_search = RandomizedSearchCV(
#     estimator=lgbm_clf, param_distributions=param_test, 
#     n_iter=n_iter,
#     scoring='accuracy',
#     cv=5,
#     refit=True,
#     random_state=random_state,
#     verbose=True)

# grid_search.fit(X_train, y_train, **fit_params)
# opt_parameters =  grid_search.best_params_
# lgbm_clf = lgbm.LGBMClassifier(**opt_parameters)

# lgbm_clf.fit(X_train,y_train)
# y_pred = lgbm_clf.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

# # Applying k-Fold Cross Validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = lgbm_clf, X = X_train, y = y_train, cv =2)
# accuracies.std()
# accuracies.mean()