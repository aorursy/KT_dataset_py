# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 
import matplotlib.pyplot as plt # Visualisation
%matplotlib inline

from wordcloud import WordCloud

from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import explained_variance_score
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# import the csv file into a pandas dataframe with read_csv
data = pd.read_csv('../input/train.csv', index_col='PassengerId')
data.head()
data.info()
data.describe(include='all')
data['title'] = data['Name'].apply(lambda name: name.split(', ')[1].split('.')[0])
data['title'].value_counts()
text_title = ' '.join(data['title'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_title)
plt.imshow(wordcloud)
plt.title('Top Title')
plt.axis("off")
data.drop(['Name','Ticket','Cabin'], axis=1, inplace = True)
data.dropna(axis=0, subset=['Age'], inplace = True)
data.describe(include='all')
data.isnull().values.any()
data['Embarked'].fillna('S', inplace = True) 
data.head()
data.isnull().values.any()
se = preprocessing.LabelEncoder()
se.fit(data['Sex'])
print(list(se.classes_))
data['Sex_encode'] = se.transform(data['Sex']) 

ee = preprocessing.LabelEncoder()
ee.fit(data['Embarked'])
print(list(ee.classes_))
data['Embarked_encode'] = ee.transform(data['Embarked']) 

te = preprocessing.LabelEncoder()
te.fit(data['title'])
print(list(te.classes_))
data['title_encode'] = te.transform(data['title']) 
data.head()
data.drop(['Sex','Embarked','title'], axis=1, inplace = True)
data.head()
Y = data.iloc[:,0].values
X = data.iloc[:,1:].values.astype('float64') # convertion of all type to float in order to don't have DataConvertionWarning during data procession
print("Y shape {}".format(Y.shape))
print("X shape {}".format(X.shape))
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)
pca = decomposition.PCA(n_components=2)
pca.fit(X_scaled)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
X_projected = pca.transform(X_scaled)

plt.scatter(X_projected[:, 0], X_projected[:, 1],
    c= Y)
plt.show()
pcs = pca.components_

for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
    plt.plot([0, x], [0, y], color='k')
    plt.text(x, y, data.columns[i+1].replace("_encode",""), fontsize='14')

plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')

plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')


plt.show()
model = XGBClassifier()
model.fit(X, Y)
# feature importance
print(model.feature_importances_)
# plot
feature_names = [f.replace("_encode","") for f in data.columns[1:]]
plt.bar(feature_names, model.feature_importances_)
plt.show()
plot_importance(model)
corr = data.corr()
labels = [f.replace("_encode","") for f in data.columns]
plt.figure(figsize=(10, 10))
plt.imshow(corr, cmap='RdYlGn', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), labels, rotation='vertical')
plt.yticks(range(len(corr)), labels);
plt.suptitle('Titanic Correlations Heat Map', fontsize=15, fontweight='bold')
plt.show()
xgb_clf = XGBClassifier()
#clf.fit(X, Y)
scores = cross_val_score(xgb_clf, X, Y, cv=5)
print("XGB: {:.4f}".format(scores.mean()))
rf_clf = RandomForestClassifier(n_estimators = 100)
#clf.fit(X, Y)
scores = cross_val_score(rf_clf, X, Y, cv=5)
print("RandomForest {:.4f}".format(scores.mean()))
et_clf = ExtraTreesClassifier(n_estimators = 1000)
#clf.fit(X, Y)
scores = cross_val_score(et_clf, X, Y, cv=5)
print(" ExtraTrees {:.4f}".format(scores.mean()))
ab_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5),n_estimators = 1000)
#clf.fit(X, Y)
scores = cross_val_score(ab_clf, X, Y, cv=5)
print("AdaBoost {:.4f}".format(scores.mean()))
gb_clf = GradientBoostingClassifier(n_estimators = 100)
#clf.fit(X, Y)
scores = cross_val_score(gb_clf, X, Y, cv=5)
print("GradientBoosting {:.4f}".format(scores.mean()))
test = pd.read_csv('../input/test.csv', index_col='PassengerId')
test.head()
test.describe(include='all')
X_age = np.delete(X,1,1)
y_age = X[:,1]
X_fare = np.delete(X,4,1)
y_fare = X[:,4]
xgb_age = XGBRegressor()
xgb_age.fit(X_age,y_age)
predictions = xgb_age.predict(X_age)
print(explained_variance_score(predictions,y_age))
xgb_fare = XGBRegressor()
xgb_fare.fit(X_fare,y_fare)
predictions = xgb_fare.predict(X_fare)
print(explained_variance_score(predictions,y_fare))
test['title'] = test['Name'].apply(lambda name: name.split(', ')[1].split('.')[0])
test['title'].unique()
test.replace('Dona','Don', inplace = True)
test['title'].unique()
test.drop(['Name','Ticket','Cabin'], axis=1, inplace = True)
test['Sex_encode'] = se.transform(test['Sex']) 
test['Embarked_encode'] = ee.transform(test['Embarked'])
test['title_encode'] = te.transform(test['title'])
test.drop(['Sex','Embarked','title'], axis=1, inplace = True)
test.describe(include='all')

without_age = test[test['Age'].isnull()]
for idx,row in without_age.iterrows():
    age = xgb_age.predict([np.delete(row.values,1,0)])
    test.at[idx, 'Age'] = age

    
without_fare = test[test['Fare'].isnull()]
for idx,row in without_fare.iterrows():
    fare = xgb_fare.predict([np.delete(row.values,4,0)])
    test.at[idx, 'Fare'] = fare
test.describe(include='all')
model = GradientBoostingClassifier(n_estimators = 100)
model.fit(X, Y)
y_pred = model.predict(test.values)
df = pd.DataFrame(y_pred, columns=['Survived'])
df.index.name='PassengerId'
df.index = test.index
df.to_csv('submission.csv', header=True)
df.head(20)