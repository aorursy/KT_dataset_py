

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import pandas_profiling as pp
import seaborn as sns
import matplotlib.pyplot as plt

import os
# Any results you write to the current directory are saved as output.
os.listdir("../input")
df = pd.read_csv("../input/diagnose62.csv")
df.head(2)
df.shape
df.dtypes
df.isna().sum()
df=df.dropna()
#let's check again for null values
df.isna().sum()
df.describe()
pp.ProfileReport(df)
# just like head, now w're looking at tail of datset, just to get an idea about variables and it's values
df.tail(2)
df.Gender.unique()
splot=sns.countplot(x="Gender", data=df, palette="bwr")
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
print("Percentages:\n",(df.Gender.value_counts()/df.shape[0])*100)
df["Location"].unique()
splot=sns.countplot(x="Location", data=df, palette="bwr")
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
print("Percentages:\n",(df.Location.value_counts()/df.shape[0])*100)
df.ID.nunique()/df.shape[0]*100
df=df.drop(['ID'],axis=1)
df.Diagnosis.nunique()
df.Diagnosis.unique()
splot=sns.countplot(x="Diagnosis", data=df, palette="bwr")
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
print("Percentages:\n",(df.Diagnosis.value_counts()/df.shape[0])*100)
pd.crosstab(df.Location, df.Diagnosis).plot(kind='bar')
pd.crosstab(df.Gender, df.Diagnosis).plot(kind='bar')
df.hist()
df_new=df
df=pd.get_dummies(df)
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train,test=train_test_split(df,test_size=0.2,random_state=0,stratify=df['Diagnosis'])
train_X=train.drop('Diagnosis',axis=1)
train_Y=train['Diagnosis']
test_X=test.drop('Diagnosis',axis=1)
test_Y=test['Diagnosis']
X=df.drop('Diagnosis',axis=1)
Y=df['Diagnosis']
model=RandomForestClassifier()
model.fit(train_X,train_Y)
pred_y = model.predict(test_X)
accuracy = np.sqrt(mean_squared_error(pred_y,test_Y))
accuracy
features = train_X.columns
importances = model.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10,15))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
# drop the least significant variables who does not affect much or none on predictive variable according to the feature importance chart.
df=df.drop(["x13","x5","x8","x9","x3","x6","x12","x11",'Gender_Male','Gender_Female','Location_London','Location_Belfast','Location_Dublin'],axis=1)
df.head(2)
train,test=train_test_split(df,test_size=0.2,random_state=0,stratify=df['Diagnosis'])
train_X=train.drop('Diagnosis',axis=1)
train_Y=train['Diagnosis']
test_X=test.drop('Diagnosis',axis=1)
test_Y=test['Diagnosis']
X=df.drop('Diagnosis',axis=1)
Y=df['Diagnosis']
model=RandomForestClassifier()
model.fit(train_X,train_Y)
pred_y = model.predict(test_X)
accuracy = np.sqrt(mean_squared_error(pred_y,test_Y))
accuracy
df.head(2)
train,test=train_test_split(df,test_size=0.2,random_state=0,stratify=df['Diagnosis'])
train_X=train.drop('Diagnosis',axis=1)
train_Y=train['Diagnosis']
test_X=test.drop('Diagnosis',axis=1)
test_Y=test['Diagnosis']
X=df.drop('Diagnosis',axis=1)
Y=df['Diagnosis']
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
k_range = list(range(1,100))
weight_options = ["uniform", "distance"]

param_grid = dict(n_neighbors = k_range, weights = weight_options)
#print (param_grid)
knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
grid.fit(train_X,train_Y)


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_X, train_Y)
print (CV_rfc.best_score_)
print (CV_rfc.best_params_)
print (CV_rfc.best_estimator_)
# Declaring Multiple Classifiers
models = [
        RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=6, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False),
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                     weights='distance')
    
        
]

model_scores = list()
reports=list()

for model in models:
    try:
        model_name = str(model).split("(")[0]
        print(model_name)
        model.fit(train_X,train_Y)
        pred_y = model.predict(test_X)
        rmse = accuracy_score(pred_y,test_Y)*100
        model_scores.append([model_name,rmse])
        reports.append([model_name,classification_report(test_Y, pred_y)])
    except:
        print("except")
        pass

model_names = pd.DataFrame(model_scores)[0].tolist()
rmses = pd.DataFrame(model_scores)[1].tolist()
model_scores = pd.Series(rmses,index=model_names)
model_scores
model_scores.sort_values(ascending = True).plot(kind = 'bar', grid=False)
print(reports[0][1])
print(reports[1][1])