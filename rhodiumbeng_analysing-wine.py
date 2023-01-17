import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Modelling Helpers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# Modelling Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
redWine = pd.read_csv('../input/winequality-red.csv')
whiteWine = pd.read_csv('../input/winequality-white.csv')
redWine.describe()
whiteWine.describe()
#Checking for duplicates
print("Number of duplicates in red wine: "+ str(np.sum(np.array(redWine.duplicated()))))
print("Number of duplicates in white wine:  "+ str(np.sum(np.array(whiteWine.duplicated()))))
# Combining the red and white wine data
wine_df = redWine.append(whiteWine)
print(wine_df.shape)
wine_df.describe()
# Features for the wine data
sns.set()
pd.DataFrame.hist(wine_df, figsize = [15,15], color='green')
plt.show()
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(wine_df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
cols_to_scale = wine_df.columns.tolist()
cols_to_scale.remove('quality')
scaled_wine_df = wine_df
scaler = MinMaxScaler()
scaled_wine_df[cols_to_scale] = scaler.fit_transform(scaled_wine_df[cols_to_scale])
scaled_wine_df.groupby('quality').mean().plot(kind='bar', figsize=(15,5))
plt.xlim(-1,9)
y = wine_df['quality']
X = wine_df.drop('quality', axis=1)
y1 = y > 5 # is the rating > 5? 
# plot histograms of original target variable (quality)
# and aggregated target variable

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.hist(y, color='black')
plt.title('Wine Quality Distribution')
plt.xlabel('original target value')
plt.ylabel('count')

plt.subplot(122)
plt.title('Wine Quality Distribution')
plt.hist(y1, color='black')
plt.xlabel('aggregated target value')
plt.show()
seed = 8 # for reproducibility
X_train,X_test,y_train,y_test = train_test_split(X, y1, test_size=0.2, random_state=seed)
X_train.shape, y_train.shape, X_test.shape, y_test.shape 
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('RF', RandomForestClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('SVM_rbf', SVC()))
models.append(('SVM_linear', SVC(kernel='linear')))
# Evaluate each model in turn
train_results = []
test_results = []
names = []
for name, model in models:
    cv_train_results = cross_val_score(model, X_train, y_train, 
                                       cv=10, scoring='accuracy')
    train_results.append(cv_train_results)
    clf = model.fit(X_train, y_train)
    cv_test_results = accuracy_score(y_test, clf.predict(X_test))
    test_results.append(cv_test_results)
    names.append(name)
    result = "%s: %f (%f) %f" % (name, cv_train_results.mean(), cv_train_results.std(), 
                                cv_test_results)
    print(result)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(train_results)
ax.set_xticklabels(names)
plt.show()
RF = RandomForestClassifier(random_state=seed)
RF.fit(X_train, y_train)
names = list(X_train.columns.values)
importances = RF.feature_importances_
# Plot the feature importances of the forest
plt.figure(figsize=(10,5))
plt.title("Feature Importances")
y_pos = np.arange(len(names))
plt.bar(y_pos, importances, align='center')
plt.xticks(y_pos, names, rotation=90)
plt.show()
clf = RandomForestClassifier()
grid_values = {'max_features':['auto','sqrt','log2'],'max_depth':[None, 10, 5, 3, 1],
              'min_samples_leaf':[1, 5, 10, 20, 50]}
clf
grid_clf = GridSearchCV(clf, param_grid=grid_values, cv=10, scoring='accuracy')
grid_clf.fit(X_train, y_train) # fit and tune model
grid_clf.best_params_
clf = RandomForestClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Training Accuracy :: ', accuracy_score(y_train, clf.predict(X_train)))
print('Test Accuracy :: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
