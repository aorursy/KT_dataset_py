#import neccesary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
#load data
wine_data = pd.read_csv("../input/redwinequality/datasets_4458_8204_winequality-red.csv")
#checking the datas
wine_data.head()
#counting the frequency of each class
wine_data.quality.value_counts()
#check for any null values
wine_data.isnull().sum()
wine_data.info()
#visualizing the data
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine_data)
#grouping the data with new features
bins = (2, 6, 8)
group_names = ['bad', 'good']
wine_data['quality_name'] = pd.cut(wine_data['quality'], bins = bins, labels = group_names)
wine_data.head()
#non numeric to numeric
le = LabelEncoder()
wine_data['quality_mark'] = le.fit_transform(wine_data.quality_name)
#check the value counts
wine_data.quality_name.value_counts()
#value count after encoding
wine_data.quality_mark.value_counts()
#checking for any outliers
for ingredients in wine_data.columns[:11]:
    plt.title(ingredients)
    sns.boxplot(x='quality_mark', y=ingredients, data=wine_data)
    plt.show()
pd.plotting.scatter_matrix(wine_data.drop(['quality','quality_name','quality_mark'], axis=1), c=wine_data.quality_mark)
#correlation between the datas
sns.heatmap(wine_data.drop(['quality','quality_name'], axis=1),annot=True, cmap='YlGnBu')
#importing libraries for model building
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

X = wine_data.drop(['quality','quality_name','quality_mark'], axis=1)
y = wine_data.quality_mark
#splitting the data in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
#considering the apis in one list
classifiers = [LinearRegression(), SVC(), LinearSVC(), SGDClassifier(penalty=None), RandomForestClassifier(n_estimators=100, max_depth=6)]
#normalizing the datas
st_scl = StandardScaler()
X_train = st_scl.fit_transform(X_train)
X_test = st_scl.fit_transform(X_test)
#fitting and testing the different models
train_model = []

for classes in classifiers:
    classes.fit(X_train, y_train)
    train_score = classes.score(X_train, y_train)
    test_score = classes.score(X_test, y_test)
    train_model.append([classes, train_score, test_score])
#removing models with less trainning score
for train_score in train_model:
    if train_score[1]<0.88:
        train_model.remove(train_score)
#qualified models
train_model
#For SVC
train_model[0][0].fit(X_train, y_train)
y_predict_svc = train_model[0][0].predict(X_test)
print(accuracy_score(y_test, y_predict_svc))
print(classification_report(y_test, y_predict_svc))
#For LinearSVC
train_model[1][0].fit(X_train, y_train)
y_predict_lsvc = train_model[1][0].predict(X_test)
print(accuracy_score(y_test, y_predict_lsvc))
print(classification_report(y_test, y_predict_lsvc))
#For RandomForestClassifier
train_model[2][0].fit(X_train, y_train)
y_predict_rfc = train_model[2][0].predict(X_test)
print(accuracy_score(y_test, y_predict_rfc))
print(classification_report(y_test, y_predict_rfc))
#as SVC as close train and test accuracy so loss of data is minimum hence tunning hyperparameter in SVC
params = {
    'C': list(np.linspace(0.1,2,20)),
    'kernel':['linear', 'rbf'],
    'gamma' :list(np.linspace(0.1,2,20))
}
svc=SVC()
grid_svc = GridSearchCV(estimator=svc, param_grid=params, scoring='roc_auc', cv=5, refit=True, return_train_score=True)
grid_svc.fit(X_train, y_train)
grid_svc.best_params_
#using new parameters
svc_new = SVC(C=0.8999999999999999, gamma=1.0999999999999999, kernel='rbf')
svc_new.fit(X_train, y_train)
y_predict = svc_new.predict(X_test)
accuracy_score(y_test, y_predict)
print(classification_report(y_test, y_predict))