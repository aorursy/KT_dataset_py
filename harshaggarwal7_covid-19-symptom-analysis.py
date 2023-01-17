import pandas as pd

import numpy as np

import missingno as msno

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler , Normalizer

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from scipy.stats import norm

from scipy import stats

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv("../input/covid19-symptoms-checker/Cleaned-Data.csv")



pd.pandas.set_option('display.max_columns',None)
display("Peeking into Data", df)
display("Shape of dataset")

print("Rows:",df.shape[0],"\nColumns:",df.shape[1])
display("NULL Values", df.isnull().sum())
display("Description",df.describe())
df.info()
#df = df.drop('Country',axis=1)

sns.distplot(df.drop('Country',axis=1))
for i in df.columns:

    print("\nColumn Name:",i,"-->",df[i].unique(),"-->Unique Count",len(df[i].unique()))
severity_columns = df.filter(like='Severity_').columns
df['Severity_None'].replace({1:'None',0:'No'},inplace =True)

df['Severity_Mild'].replace({1:'Mild',0:'No'},inplace =True)

df['Severity_Moderate'].replace({1:'Moderate',0:'No'},inplace =True)

df['Severity_Severe'].replace({1:'Severe',0:'No'},inplace =True)
df['Condition']=df[severity_columns].values.tolist()
def removing(list1):

    list1 = set(list1) 

    list1.discard("No")

    a = ''.join(list1)

    return a
df['Condition'] = df['Condition'].apply(removing)
age_columns = df.filter(like='Age_').columns

gender_columns = df.filter(like='Gender_').columns

contact_columns = df.filter(like='Contact_').columns
No_risk_age = df.groupby(['Severity_None'])[age_columns].sum()

No_risk_gender = df.groupby(['Severity_None'])[gender_columns].sum()

No_risk_contact = df.groupby(['Severity_None'])[contact_columns].sum()
Low_risk_age = df.groupby(['Severity_Mild'])[age_columns].sum()

Low_risk_gender = df.groupby(['Severity_Mild'])[gender_columns].sum()

Low_risk_contact = df.groupby(['Severity_Mild'])[contact_columns].sum()
Moderate_risk_age = df.groupby(['Severity_Moderate'])[age_columns].sum()

Moderate_risk_gender = df.groupby(['Severity_Moderate'])[gender_columns].sum()

Moderate_risk_contact = df.groupby(['Severity_Moderate'])[contact_columns].sum()
Severe_risk_age = df.groupby(['Severity_Severe'])[age_columns].sum()

Severe_risk_gender = df.groupby(['Severity_Severe'])[gender_columns].sum()

Severe_risk_contact = df.groupby(['Severity_Severe'])[contact_columns].sum()
sns.countplot(df['Condition'])
df.drop("Country",axis=1,inplace=True)
df.drop(severity_columns,axis=1,inplace=True)
df['Symptoms_Score'] = df.iloc[:,:5].sum(axis=1) + df.iloc[:,6:10].sum(axis=1)
df.shape
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['Condition'] = le.fit_transform(df['Condition'])
df
from pylab import rcParams

rcParams['figure.figsize'] = 13, 18

corrmat = df.corr()

k = 22

cols = corrmat.nlargest(k, 'Condition')['Condition'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
X= df.drop(['Condition'],axis=1)

y= df['Condition']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
'''from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'n_estimators': [100, 200, 300, 500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)'''
'''# Fit the grid search to the data

grid_search.fit(X_train, y_train)'''
'''print('Best Parameters',grid_search.best_params_)

best_grid = grid_search.best_estimator_

print('\n Best Estimator',best_grid)'''
"""Best Parameters {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 100}

Best Estimator RandomForestClassifier(max_depth=4, max_features='sqrt')"""
from sklearn.ensemble import RandomForestClassifier

rfc1=RandomForestClassifier(criterion= 'gini', max_depth= 4, max_features= 'sqrt', n_estimators= 100)
rfc1.fit(X_train, y_train)
pred=rfc1.predict(X_test)
pred
from sklearn.metrics import accuracy_score

print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,pred)
!pip install catboost
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=200)
categorical_var = np.where(X_train.dtypes != np.float)[0]

print('\nCategorical Variables indices : ',categorical_var)
model.fit(X_train,y_train,cat_features = categorical_var,plot=False)
predict_train = model.predict(X_train)

print('\nTarget on train data',predict_train)
accuracy_train = accuracy_score(y_train,predict_train)

print('\naccuracy_score on train dataset : ', accuracy_train)
predict_test = model.predict(X_test)

print('\nTarget on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(y_test,predict_test)

print('\naccuracy_score on test dataset : ', accuracy_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver = 'lbfgs')

model.fit(X_train, y_train)
# use the model to make predictions with the test data

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
'''from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train, y_train)'''
'''y_pred_knn = knn.predict(X_test)'''
'''from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred_knn)'''
'''from sklearn.svm import SVC



svm = SVC(kernel='linear',C=0.025, random_state=101)



svm.fit(X_train, y_train)'''
'''y_pred_svc = svc.predict(X_test)'''
'''from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred_svc)'''
from sklearn.naive_bayes import MultinomialNB



mb = MultinomialNB()



mb.fit(X_train, y_train)
y_pred_mb = mb.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred_mb)
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = 4)

y_train.shape
from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout

from keras.models import Sequential,Model

from keras.optimizers import SGD

from keras.callbacks import ModelCheckpoint,LearningRateScheduler

import keras

from keras import backend as K
model=keras.models.Sequential()

#model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128,activation='relu'))

model.add(keras.layers.Dense(128,activation='relu'))

model.add(keras.layers.Dense(128,activation='relu'))

model.add(keras.layers.Dense(128,activation='relu'))

model.add(keras.layers.Dense(128,activation='relu'))

model.add(keras.layers.Dense(4,activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

                   

model.fit(X_train, y_train,epochs=10, batch_size=32, verbose=1)
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis=1)
y_pred