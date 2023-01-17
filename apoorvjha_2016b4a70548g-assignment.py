import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import matplotlib.pylab as py

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn import preprocessing, model_selection, metrics

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score
data_orig = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')

data = data_orig
test_orig = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')
data.info()
id_test=test_orig['ID']

test=test_orig.copy()
df=data.copy()
%matplotlib inline

data.hist(figsize = (20, 20))
df.Class.value_counts().plot(kind = "bar", rot = 0)
df.col56.value_counts().plot(kind = "bar", rot = 0)
df.col37.value_counts().plot(kind = "bar", rot = 0)
df.col44.value_counts().plot(kind = "bar", rot = 0)
df.col2.value_counts().plot(kind = "bar", rot = 0)
df.col11.value_counts().plot(kind = "bar", rot = 0)
df=df.drop(['ID'],axis=1)

test=test.drop(['ID'],axis=1)
test.shape
x=[]

columns=df.columns

for i in columns:

    if(df[i].dtype == 'float'):## to access columns in a loop

        x.append(i)
for i in range(9):

    df=df.drop([x[i]],axis=1)

    test=test.drop([x[i]],axis=1)### dropping float values
df=df.drop(['col2','col11','col37','col44','col56'],axis=1)

test=test.drop(['col2','col11','col37','col44','col56'],axis=1)### srting columns
df.head()
corr = df.corr()

(corr['Class']).sort_values()
x_train=df.drop(['Class'],axis=1)

y_train = df.loc[ : , "Class"]
corr_matrix = x_train.corr().abs()



#Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



#Find features with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]



# Drop features 

x_train.drop(to_drop, axis=1, inplace=True)

test.drop(to_drop, axis=1, inplace=True)
#x_train.insert(0,'col11',data['col11'],True)

#x_train.insert(1,'col37',data['col37'],True)

#x_train.insert(2,'col44',data['col44'],True)

#test.insert(0,'col11',test_orig['col11'],True)

#test.insert(1,'col37',test_orig['col37'],True)

#test.insert(2,'col44',test_orig['col44'],True)
#x.remove('col2')

#x.remove('col56')
#for i in x:

##    x_train=pd.get_dummies(x_train,columns=[i],prefix=[i])

#for i in x:

#    test=pd.get_dummies(test,columns=[i],prefix=[i])
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(x_train)

x_train = pd.DataFrame(np_scaled)

x_train.head()
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(test)

test = pd.DataFrame(np_scaled)

test.head()
print(x_train.shape)

test.shape
from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state = 2) 

x_try, y_try = sm.fit_sample(x_train, y_train.ravel()) 
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_train, test_size=0.2, random_state=42)
np.random.seed(42)

from sklearn.ensemble import RandomForestClassifier



score_train_RF = []

score_test_RF = []



for i in range(5,20,1):

    rf = RandomForestClassifier(n_estimators = 100, max_depth=i,random_state=42)

    rf.fit(X_train, Y_train)

    sc_train = rf.score(X_train,Y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_test,Y_test)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(5,20,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(5,20,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=100, max_depth = 8,random_state=42,min_samples_split=2)

rf.fit(X_train, Y_train)

rf.score(X_test,Y_test)
rf = RandomForestClassifier(n_estimators=2000, max_depth = 8,random_state=42,min_samples_split=2)

rf.fit(X_train, Y_train)

rf.score(X_test,Y_test)
from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score





from sklearn.model_selection import GridSearchCV



rf_temp = RandomForestClassifier(n_estimators = 100,random_state=42)        #Initialize the classifier object



parameters = {'max_depth':[8,14],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters



scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train, Y_train)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
rf = RandomForestClassifier(n_estimators=2000, max_depth = 8,random_state=42,min_samples_split=2)

rf.fit(X_train, Y_train)

pred = rf.predict(test)
res1 = pd.DataFrame(pred)

res1.insert(0,"ID",id_test,True)

res1 = res1.rename(columns={0: "Class"})

print(res1.head())

res1.to_csv('done.csv', index = False)
res1.shape
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"  target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

    create_download_link(res1)