# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# loading data
df = pd.read_csv(os.path.join(dirname, filename))
df.shape
df.head()
df.info()
#checking the categories for object data types
print(df['international plan'].unique())
print(df['voice mail plan'].unique())
df.describe(include='all')

#check for missing values
df.isnull().sum()

# check for duplicated rows
df.duplicated().sum()
# check for outliers
df.skew()

df['number vmail messages'].describe()
df['number vmail messages'][df['number vmail messages']>0].describe()
# New categorical feature = 
                # if numofvmailmessage <1 = No VM plan
                # if numofvmailmessage >1 and <38 = Normal users
                # if numofvmailmessage >38 and <53= High Frequency users
df['vmail_messages'] = pd.cut(df['number vmail messages'],bins=[0,1,38,52],
                             labels=['No VM plan','Normal Users','High Frequency users'],
                             include_lowest=True)
df.head(20)
cor = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(cor.round(3),annot=True,cmap='coolwarm')
plt.show()
df.columns
numerics =['account length','number vmail messages',
       'total day minutes', 'total day calls', 'total day charge',
       'total eve minutes', 'total eve calls', 'total eve charge',
       'total night minutes', 'total night calls', 'total night charge',
       'total intl minutes', 'total intl calls', 'total intl charge',
       'customer service calls']
xnum = df[numerics]
y = df['churn']
from sklearn.feature_selection import f_classif
fval,pval = f_classif(xnum,y)
for i in range(len(numerics)):print(numerics[i],pval[i])

categories = ['state','area code','phone number', 'international plan',
       'voice mail plan','vmail_messages']

y = df['churn']
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
for col in categories:
    xcat = LabelEncoder().fit_transform(df[col]).reshape(-1,1)
    cval,pval = chi2(xcat,y)
    print(col,pval)
#selecting important features based on previous analysis
x = df[['international plan','vmail_messages','total day minutes','total eve minutes',
     'total night minutes','total intl minutes','customer service calls']]
y = df['churn']
x.head()
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
preprocessor = ColumnTransformer([('ohe',OneHotEncoder(),[1]),
                                ('ode',OrdinalEncoder(),[0]),
                                 ('sc',StandardScaler(),[2,3,4,5,6])],remainder='passthrough')
x_new = preprocessor.fit_transform(x)
pd.DataFrame(x_new).head()
# train test split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x_new,y,test_size=0.2,random_state=5)
print(x.shape)
print(xtrain.shape)
print(xtest.shape)
print(y.shape)
print(ytrain.shape)
print(ytest.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
model.fit(xtrain,ytrain)

# performance analysis
from sklearn import metrics
ypred = model.predict(xtest)
print("Accuracy : ",metrics.accuracy_score(ytest,ypred))
print("Recall : ",metrics.recall_score(ytest,ypred))
print("F1 score : ",metrics.f1_score(ytest,ypred))
print("Precision : ",metrics.precision_score(ytest,ypred))
# performance analysis on train data
ypred2 = model.predict(xtrain)
print("Accuracy : ",metrics.accuracy_score(ytrain,ypred2))
print("Recall : ",metrics.recall_score(ytrain,ypred2))
print("F1 score : ",metrics.f1_score(ytrain,ypred2))
print("Precision : ",metrics.precision_score(ytrain,ypred2))
# preprocessing pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
preprocessor = ColumnTransformer([('ohe',OneHotEncoder(),[1]),
                                ('ode',OrdinalEncoder(),[0])],
                                remainder="passthrough")
x_new = preprocessor.fit_transform(x)
pd.DataFrame(x_new)
# train test split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x_new,y,test_size=0.2,random_state=5)
print(x.shape)
print(xtrain.shape)
print(xtest.shape)
print(y.shape)
print(ytrain.shape)
print(ytest.shape)
# decision tree
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(random_state=5,class_weight={0:0.5,1:0.5})
model2.fit(xtrain,ytrain)
import graphviz
from sklearn import tree

fname = ['International plan', 'vmail_NO_Plan','vmail_Normal','vmail_HF', 'Total day minutes',
       'Total eve minutes', 'Total night minutes', 'Total intl minutes',
       'Customer service calls']
cname = ['Not Leaving','Leaving']
graphdata = tree.export_graphviz(model2,feature_names=fname,class_names=cname,
                                filled=True,rounded=True)
graph = graphviz.Source(graphdata)
graph

# performance analysis
ypred2 = model2.predict(xtest)
print("Accuracy : ",metrics.accuracy_score(ytest,ypred2))
print("Recall : ",metrics.recall_score(ytest,ypred2))
print("F1 score : ",metrics.f1_score(ytest,ypred2))
print("Precision : ",metrics.precision_score(ytest,ypred2))

# performance analysis on train data
ypred2 = model2.predict(xtrain)
print("Accuracy : ",metrics.accuracy_score(ytrain,ypred2))
print("Recall : ",metrics.recall_score(ytrain,ypred2))
print("F1 score : ",metrics.f1_score(ytrain,ypred2))
print("Precision : ",metrics.precision_score(ytrain,ypred2))
param_grid = {"max_depth":np.arange(3,25,2),
              "min_samples_leaf":np.arange(3,50,2),
              "min_samples_split":np.arange(10,120,5)}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=5),
                          param_grid=param_grid,n_jobs=-1,
                          scoring='recall',verbose=True,cv=5)
grid_search.fit(x_new,y)
print(grid_search.best_score_)
print(grid_search.best_params_)
# Controlling overfitting
model2 = DecisionTreeClassifier(criterion='gini',random_state=5,
                               max_depth=8,min_samples_leaf=5,min_samples_split=20)
model2.fit(xtrain,ytrain)

# performance analysis On test data
ypred2 = model2.predict(xtest)
print("Accuracy : ",metrics.accuracy_score(ytest,ypred2))
print("Recall : ",metrics.recall_score(ytest,ypred2))
print("F1 score : ",metrics.f1_score(ytest,ypred2))
print("Precision : ",metrics.precision_score(ytest,ypred2))
# performance analysis on train data
ypred2 = model2.predict(xtrain)
print("Accuracy : ",metrics.accuracy_score(ytrain,ypred2))
print("Recall : ",metrics.recall_score(ytrain,ypred2))
print("F1 score : ",metrics.f1_score(ytrain,ypred2))
print("Precision : ",metrics.precision_score(ytrain,ypred2))
model2.feature_importances_
for i in range(len(fname)):print(fname[i],model2.feature_importances_[i])
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=100,random_state=5,
                               max_depth=8,oob_score=True)
#train the model
model4.fit(xtrain,ytrain)
# performance analysis On test data
ypred2 = model4.predict(xtest)
print("Accuracy : ",metrics.accuracy_score(ytest,ypred2))
print("Recall : ",metrics.recall_score(ytest,ypred2))
print("F1 score : ",metrics.f1_score(ytest,ypred2))
print("Precision : ",metrics.precision_score(ytest,ypred2))
# performance analysis on train data
ypred2 = model4.predict(xtrain)
print("Accuracy : ",metrics.accuracy_score(ytrain,ypred2))
print("Recall : ",metrics.recall_score(ytrain,ypred2))
print("F1 score : ",metrics.f1_score(ytrain,ypred2))
print("Precision : ",metrics.precision_score(ytrain,ypred2))
#check OOB (out of bag) score
model4.oob_score_
from sklearn.ensemble import AdaBoostClassifier
model5 = AdaBoostClassifier(n_estimators=120,random_state=5,learning_rate=0.2)
model5.fit(xtrain,ytrain)

# performance analysis On test data
ypred2 = model5.predict(xtest)
print("Accuracy : ",metrics.accuracy_score(ytest,ypred2))
print("Recall : ",metrics.recall_score(ytest,ypred2))
print("F1 score : ",metrics.f1_score(ytest,ypred2))
print("Precision : ",metrics.precision_score(ytest,ypred2))

# performance analysis on train data
ypred2 = model5.predict(xtrain)
print("Accuracy : ",metrics.accuracy_score(ytrain,ypred2))
print("Recall : ",metrics.recall_score(ytrain,ypred2))
print("F1 score : ",metrics.f1_score(ytrain,ypred2))
print("Precision : ",metrics.precision_score(ytrain,ypred2))

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
model6 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=150,random_state=5)
model6.fit(xtrain,ytrain)


# performance analysis On test data
ypred2 = model6.predict(xtest)
print("Accuracy : ",metrics.accuracy_score(ytest,ypred2))
print("Recall : ",metrics.recall_score(ytest,ypred2))
print("F1 score : ",metrics.f1_score(ytest,ypred2))
print("Precision : ",metrics.precision_score(ytest,ypred2))
# performance analysis on train data
ypred2 = model6.predict(xtrain)
print("Accuracy : ",metrics.accuracy_score(ytrain,ypred2))
print("Recall : ",metrics.recall_score(ytrain,ypred2))
print("F1 score : ",metrics.f1_score(ytrain,ypred2))
print("Precision : ",metrics.precision_score(ytrain,ypred2))

from xgboost import XGBClassifier
model7 = XGBClassifier(learning_rate=0.005,n_estimators=120,max_depth=8)
model7.fit(xtrain,ytrain)
# performance analysis On test data
ypred2 = model7.predict(xtest)
print("Accuracy : ",metrics.accuracy_score(ytest,ypred2))
print("Recall : ",metrics.recall_score(ytest,ypred2))
print("F1 score : ",metrics.f1_score(ytest,ypred2))
print("Precision : ",metrics.precision_score(ytest,ypred2))
# performance analysis on train data
ypred2 = model7.predict(xtrain)
print("Accuracy : ",metrics.accuracy_score(ytrain,ypred2))
print("Recall : ",metrics.recall_score(ytrain,ypred2))
print("F1 score : ",metrics.f1_score(ytrain,ypred2))
print("Precision : ",metrics.precision_score(ytrain,ypred2))
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

log_model = LogisticRegression()
dt_model = DecisionTreeClassifier(random_state=5,max_depth=10)
rf_model = RandomForestClassifier(n_estimators=100,random_state=5,max_depth=10)
gb_model = GradientBoostingClassifier(learning_rate=0.01,n_estimators=120,random_state=5)


model8 = StackingClassifier(classifiers=[dt_model,rf_model,gb_model],
                           meta_classifier=log_model)
model8.fit(xtrain,ytrain)
# performance analysis On test data
ypred2 = model8.predict(xtest)
print("Accuracy : ",metrics.accuracy_score(ytest,ypred2))
print("Recall : ",metrics.recall_score(ytest,ypred2))
print("F1 score : ",metrics.f1_score(ytest,ypred2))
print("Precision : ",metrics.precision_score(ytest,ypred2))
# performance analysis on train data
ypred2 = model8.predict(xtrain)
print("Accuracy : ",metrics.accuracy_score(ytrain,ypred2))
print("Recall : ",metrics.recall_score(ytrain,ypred2))
print("F1 score : ",metrics.f1_score(ytrain,ypred2))
print("Precision : ",metrics.precision_score(ytrain,ypred2))
