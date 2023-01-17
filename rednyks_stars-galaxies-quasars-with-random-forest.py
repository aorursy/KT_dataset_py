import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
%matplotlib inline
import matplotlib.pyplot as plt
sdss_pd = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
sdss_pd.head()
sdss_pd.info()
sdss_pd["class"].value_counts().sort_index()
sdss_pd.columns.values
sdss_pd.drop(["objid","specobjid","run","rerun","camcol","field"], axis = 1, inplace = True)
sdss_pd.head()
print("Mapping: ", dict(enumerate(["GALAXY","QSO","STAR"])))
sdss_pd["class"] = sdss_pd["class"].astype("category")
sdss_pd["class"] = sdss_pd["class"].cat.codes
print(sdss_pd["class"].value_counts().sort_index())
corr_matrix = sdss_pd.corr()
corr_matrix["class"].sort_values(ascending = False)
sdss_feat = sdss_pd.drop("class", axis = 1)
sdss_labels = sdss_pd["class"].copy()
X_train, X_test, y_train, y_test = train_test_split(sdss_feat, sdss_labels, test_size=0.2, random_state=42, stratify=sdss_labels)
default_forest = RandomForestClassifier(random_state = 42)
default_forest.fit(X_train, y_train)
default_forest.get_params()
print("Test accuracy for default forest:", default_forest.score(X_test, y_test))
y_pred = default_forest.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_pd = pd.DataFrame(data = conf_matrix, 
                              index = ["GALAXY","QSO","STAR"],
                              columns = ["GALAXY","QSO","STAR"])
conf_matrix_pd
feat_imp_pd = pd.DataFrame(data = default_forest.feature_importances_,
                          index = sdss_feat.columns,
                          columns = ["Importance"])
feat_imp_pd = feat_imp_pd.sort_values(by = 'Importance', ascending = False)
feat_imp_pd
feat_imp_pd.plot(kind = "bar", figsize = (10,5), grid = True)
plt.show()
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]
max_features = ['auto', 'log2']
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               scoring = 'accuracy', 
                               n_iter = 25, 
                               cv = 4, 
                               verbose = 2, 
                               random_state = 42,
                               n_jobs = -1)
rf_random.fit(X_train, y_train)
rf_random.best_score_
best_forest = rf_random.best_estimator_
best_forest
print("Test accuracy for best forest:", best_forest.score(X_test, y_test))
y_pred = best_forest.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_pd = pd.DataFrame(data = conf_matrix,
                              index = ["GALAXY","QSO","STAR"],
                              columns = ["GALAXY","QSO","STAR"])
conf_matrix_pd
feat_imp_pd = pd.DataFrame(data = best_forest.feature_importances_,
                           index = sdss_feat.columns,
                           columns = ["Importance"])
feat_imp_pd = feat_imp_pd.sort_values(by = 'Importance', ascending = False)
feat_imp_pd
feat_imp_pd.plot(kind = "bar", figsize = (10,5), grid = True)
plt.show()