import pandas as pd
import numpy as np
import seaborn as sns
data_train=pd.read_csv('../input/PL_XSELL.csv')
data_train.shape
data_train.head().T
data_train=data_train.drop(['random','CUST_ID'],axis=1)
import matplotlib.pyplot as plt
plt.pie(data_train['GENDER'].value_counts(),labels=data_train['GENDER'].value_counts().index,autopct='%1.1f%%')
plt.subplots(figsize=(16,8))
sns.heatmap(data_train.corr()[data_train.corr().abs()>0.1],annot=True)
# convert factors to labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_train['GENDER'] = le.fit_transform(data_train['GENDER'])
data_train['AGE_BKT'] = le.fit_transform(data_train['AGE_BKT'])
data_train['OCCUPATION'] = le.fit_transform(data_train['OCCUPATION'])
data_train['ACC_TYPE'] = le.fit_transform(data_train['ACC_TYPE'])
data_train['OCCUPATION'] = le.fit_transform(data_train['OCCUPATION'])
data_train.info()
data_train['ACC_OP_DATE']=pd.to_datetime(data_train['ACC_OP_DATE'])
data_train['ACC_OP_YR']=data_train['ACC_OP_DATE'].dt.year
from sklearn.preprocessing import StandardScaler
x=data_train.drop(['ACC_OP_DATE'],axis=1)
scaler=StandardScaler().fit(x)
y=pd.DataFrame(scaler.transform(x),columns=x.columns)
y.boxplot(vert=False,figsize=(15,10))
data_train.describe().T
# Load libraries
from matplotlib import pyplot
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
X=data_train.drop(['AGE','TARGET','ACC_OP_DATE'],axis=1)
Y=data_train[['TARGET']]
from statsmodels.stats.outliers_influence import variance_inflation_factor
x_features = list(X)

x_features.remove('TOT_NO_OF_L_TXNS')
x_features.remove('ACC_OP_YR')


data_mat = X[x_features].as_matrix()
data_mat.shape
vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
vif_factors = pd.DataFrame()
vif_factors['column'] = list(x_features)
vif_factors['vif'] = vif
print(vif_factors)
X=data_train[x_features]
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'
# Spot-Check Algorithms
models = []
models.append(('Logistic', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('GB',GradientBoostingClassifier()))
# evaluate each model in turn
results = []
names = []
model_comp=pd.DataFrame(columns=['Model','Test Accuracy','Std.Dev'])
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    model_comp=model_comp.append([{'Model':name, 'Test Accuracy':cv_results.mean(), 'Std.Dev':cv_results.std()}],ignore_index=True)
    
model_comp
model=DecisionTreeClassifier(max_depth=15)
model=model.fit(X_train,Y_train)
model.score(X_train,Y_train)
model.score(X_validation,Y_validation)
model=DecisionTreeClassifier(max_depth=5)
model=model.fit(X_train,Y_train)
model.score(X_train,Y_train)
model.score(X_validation,Y_validation)
from IPython.display import Image  
from sklearn import tree
from os import system

train_char_label = ['yes', 'no']
Loan_campaign_File = open('Loan_campaign_tree.dot','w')
dot_data = tree.export_graphviz(model, out_file=Loan_campaign_File, feature_names = list(X), class_names = list(train_char_label))

Loan_campaign_File.close()


# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print (pd.DataFrame(model.feature_importances_, columns = ["Imp"], index = X.columns).sort_values(by='Imp',ascending=False))
