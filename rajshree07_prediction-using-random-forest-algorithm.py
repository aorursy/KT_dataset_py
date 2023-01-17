import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplot
import numpy as np
# Import the Titanic Dataset
X = pd.read_csv('../input/train.csv')
y = X.pop("Survived")
X.head()
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"
    
# Clean Cabin
X["Cabin"] = X.Cabin.apply(clean_cabin)

# Define categorical features
categorical_variables = ["Sex", "Cabin", "Embarked"]

# Impute missing age with median
X["Age"].fillna(X["Age"].median(), inplace=True)

# Drop PassengerId, Name, Ticket
X.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)

# Impute missing categorical variables and dummify them
for variable in categorical_variables:
    X[variable].fillna("Missing", inplace=True)
    dummies = pd.get_dummies(X[variable], prefix=variable)
    X = pd.concat([X, dummies], axis=1)
    X.drop([variable], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Get a list of numerica features
numeric_variables = ['Pclass','Age','SibSp','Parch','Fare']
X_numeric = X_train[numeric_variables]
X_numeric.head()
X_numeric["Age"].fillna(X_numeric["Age"].mean(), inplace=True)
# Create the baseline 
model_1 = RandomForestClassifier(oob_score=True, random_state=42)
# Fit and Evaluate OOB
model_1 = model_1.fit(X_numeric, y_train)
# Calculate OOB Score
print("The OOB Score is: " + str(model_1.oob_score_))
rf_result = cross_val_score(model_1, X_numeric, y_train, scoring='accuracy')
rf_result.mean()
pred_train = np.argmax(model_1.oob_decision_function_,axis=1)
rf_numeric_auc = roc_auc_score(y_train, pred_train)
rf_numeric_auc
# Copy the whole train set
X_cat = X_train
X_cat.head(3)
model_2 = RandomForestClassifier(oob_score=True, random_state=40)
# Fit and Evaluate OOB
model_2 = model_2.fit(X_cat, y_train)
# Calculate OOB Score
print("The OOB Score is: " + str(model_2.oob_score_))
X_cat_scaled = StandardScaler().fit(X_cat).transform(X_cat)
X_cat_scaled
# Create the baseline 
model_3= RandomForestClassifier(oob_score=True, random_state=40)
# Fit and Evaluate OOB
model_3 = model_3.fit(X_cat_scaled, y_train)
# Calculate OOB Score
model_3.oob_score_
# AUC Score
pred_train = np.argmax(model_2.oob_decision_function_,axis=1)
rf_cat_auc = roc_auc_score(y_train, pred_train)
rf_cat_auc
# Create ROC Graph
from sklearn.metrics import roc_curve
rf_numeric_fpr, rf_numeric_tpr, rf_numeric_thresholds = roc_curve(y_test, model_1.predict_proba(X_test[X_numeric.columns])[:,1])
rf_cat_fpr, rf_cat_tpr, rf_cat_thresholds = roc_curve(y_test, model_2.predict_proba(X_test)[:,1])

# Plot Random Forest Numeric ROC
plt.plot(rf_numeric_fpr, rf_numeric_tpr, label='RF Numeric (area = %0.2f)' % rf_numeric_auc)

# Plot Random Forest Cat+Numeric ROC
plt.plot(rf_cat_fpr, rf_cat_tpr, label='RF Cat+Num (area = %0.2f)' % rf_cat_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1], ls="--", label='Base Rate')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()
results  =  []
results2 = []
max_depth_size  = [1,2,3,4,5,10,20,50,100]

for depth in max_depth_size:
    model = RandomForestClassifier(depth, oob_score=True, n_jobs=-1, random_state=44)
    #model.fit(X, y)
    model.fit(X_train, y_train)
    print(depth, 'depth')
    pred = model.predict(X_train)
    pred2 = model.predict(X_test)
    roc1 = roc_auc_score(y_train, pred)
    roc2 = roc_auc_score(y_test, pred2)
    print('AUC Train: ', roc1)
    print('AUC Test: ', roc2)
    results.append(roc1)
    results2.append(roc2)
    print (" ")


plt.plot(max_depth_size, results, label='Train Set')
plt.plot(max_depth_size, results2, label='Test Set')
plt.xlabel('Max Depth Size')
plt.ylabel('AUC Score')
plt.title('Train VS Test Scores')
plt.legend(loc="lower right")
plt.show()
results = []
n_estimator_options = [1, 2, 3, 4, 5, 15, 20, 25, 40, 50, 70, 100]

for trees in n_estimator_options:
    model = RandomForestClassifier(trees, oob_score=True, random_state=42)
    #model.fit(X, y)
    model.fit(X_train, y_train)
    print(trees, 'trees')
    AUC = model.oob_score_
    print('AUC: ', AUC)
    results.append(AUC)
    print (" ")    
plt.plot(n_estimator_options, results, label='OOB Score')
plt.xlabel('# of Trees')
plt.ylabel('OOB Score')
plt.title('OOB Score VS Trees')
plt.legend(loc="lower right")
plt.show()
results = []
max_features_options = ["auto", None, "sqrt", "log2", 0.7, 0.2]

for max_features in max_features_options:
    model = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
    model.fit(X_train, y_train)
    print(max_features, "option")
    auc = model.oob_score_
    print('AUC: ', auc)
    results.append(auc)
    print (" ")
    
pd.Series(results, max_features_options).plot(kind='barh')
results = []
min_samples_leaf_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,20]

for min_samples in min_samples_leaf_options:
    model = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features="auto", min_samples_leaf=min_samples)
    model.fit(X_train, y_train)
    print(min_samples, "min samples")
    auc = model.oob_score_
    print('AUC: ', auc)
    results.append(auc)
    print (" ")
    
pd.Series(results, min_samples_leaf_options).plot()
feature_importances = pd.Series(model_2.feature_importances_, index=X.columns)
print(feature_importances)
feature_importances.sort_values(inplace=True)
feature_importances.plot(kind='barh', figsize=(7,6))
# Create function to combine feature importances
def graph_feature_importances(model, feature_names, autoscale=True, headroom=0.1, width=10, summarized_columns=None):  
    feature_dict=dict(zip(feature_names, model.feature_importances_))
    
    if summarized_columns:
        for col_name in summarized_columns:
            sum_value = sum(x for i, x in feature_dict.items() if col_name in i )
            keys_to_remove = [i for i in feature_dict.keys() if col_name in i ]
            for i in keys_to_remove:
                feature_dict.pop(i)
            feature_dict[col_name] = sum_value
    results = pd.Series(feature_dict, index=feature_dict.keys())
    results.sort_values(inplace=True)
    print(results)
    results.plot(kind='barh', figsize=(width, len(results)/4), xlim=(0, .30))
 
# Create combined feature importances
graph_feature_importances(model_2, X.columns, summarized_columns=categorical_variables)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer
clf = GradientBoostingClassifier()
titanic_X_colns = ['Pclass','Age', 'Fare']
titanic_X = X_train[titanic_X_colns]
my_imputer = Imputer()
imputed_titanic_X = my_imputer.fit_transform(titanic_X)

clf.fit(imputed_titanic_X, y_train)
titanic_plots = plot_partial_dependence(clf, features=['Pclass','Age', 'Fare'], X=titanic_X, 
                                        feature_names=titanic_X_colns, grid_resolution=7)
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=model_2, dataset=X_train, model_features=X_train.columns, feature='Fare')

# plot it
pdp.pdp_plot(pdp_goals, 'Fare')
plt.show()