import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
import seaborn as sns
!pip install shap
import shap
import matplotlib.pyplot as plt
df = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")
print(df.shape)
df_countNull = df.isnull().sum()
df_countNullunder5000 = df_countNull.loc[df_countNull<5000]
print(df_countNullunder5000, len(df_countNullunder5000))
Nullunder5000NamesArr = df_countNullunder5000.index.values
df_filtered = df[Nullunder5000NamesArr]
df_filtered = df_filtered.dropna()
print(df_filtered.shape)
df_filtered.head()

df_filtered.groupby('SARS-Cov-2 exam result').count()
data_filtered = df_filtered.drop(['Patient ID','Patient addmited to regular ward (1=yes, 0=no)',
                                  'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                                  'Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1)

# Factorize categorical data and keep mapping codes for later
col_info = [(col_n, str(col_t)) for col_n, col_t in zip(list(data_filtered), data_filtered.dtypes)]

factorized_codes = {}
for (col_n, col_type) in col_info:
    if col_type == 'object':
        factor = pd.factorize(data_filtered.loc[:, col_n])
        data_filtered.loc[:, col_n] = factor[0]
        factorized_codes[col_n] = factor[1]

data_filtered.head()
feature_cols = data_filtered.drop(["SARS-Cov-2 exam result"], axis=1).columns.to_list()
X = data_filtered[feature_cols]
y = data_filtered["SARS-Cov-2 exam result"]
parameters = {}
parameters['model__n_estimators'] = [10,50,70,100,150,200,300]
parameters['model__criterion'] = ['gini', 'entropy']
parameters['model__max_depth'] = [None, 3, 4, 5]
parameters['model__max_features'] = ['auto', 'sqrt', 'log2', None]
parameters['model__min_impurity_decrease'] = [0.0, 1e-3]
parameters['model__min_samples_split'] = [2, 10, 40]
parameters['model__min_samples_leaf'] = [1, 5]


model1 = RandomForestClassifier(
    max_depth=5,
    criterion = 'gini',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=1,
    random_state=0,
    verbose=0,
    warm_start=False,
    class_weight='balanced'
)
my_pipeline1 = Pipeline(steps=[('standard', StandardScaler()),
                              ('model', model1)
                             ])

skf = StratifiedKFold(n_splits=5)


grid = GridSearchCV(estimator=my_pipeline1,
                            param_grid=parameters,
                            scoring='recall',
                            cv=skf,
                            n_jobs=-1)


fitted_model = grid.fit(X,y)

results = grid.cv_results_
print(grid.best_estimator_)

# Use best model to get predictions
my_pipeline1 = Pipeline(steps=[('standard', StandardScaler()),
                              ('model', grid.best_estimator_['model'])
                             ])

y_pred1 = cross_val_predict(my_pipeline1, X, y, cv=skf)
conf_mat1 = confusion_matrix(y, y_pred1)
def plot_confusion_matrix(cm):
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax,fmt='g',cmap=sns.color_palette("GnBu_d"), cbar=False, linewidths=1, linecolor='black');
    ax.set_xlabel('Predict');ax.set_ylabel('True'); 
    ax.set_title('Confusion matrix'); 
    ax.xaxis.set_ticklabels(['Negative', 'Positive']); ax.yaxis.set_ticklabels(['Negative','Positive']);
print("Balanced Accuracy: "+str(balanced_accuracy_score(y,y_pred1)*100)+"%\n")
print(classification_report(y,y_pred1))

X_trans = StandardScaler().fit_transform(X)

plt.figure(1)
plot_confusion_matrix(conf_mat1)
plt.figure(2)
shap_values1 = shap.TreeExplainer(
        grid.best_estimator_['model'].fit(X_trans,y), feature_perturbation='tree_path_dependent'
).shap_values(X)
shap.summary_plot(shap_values1, X_trans, plot_type="bar",feature_names = X.columns.tolist())
test_data = df[Nullunder5000NamesArr]

# Select rows with missing values as the test data
na_rows = test_data.isnull().any(axis=1)
test_data = test_data.loc[na_rows, :]

test_data = test_data.drop(['Patient ID','Patient addmited to regular ward (1=yes, 0=no)',
                            'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                            'Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1)

test_data.head()
for col_n in factorized_codes:
    replacement = {category: code for code, category in enumerate(factorized_codes[col_n].values)}
    
    # Feature has just one category
    if len(replacement) == 1:
        # Check whether there are more categories in the testing set
        mask = test_data.loc[:, col_n].isnull()
        unqs = np.unique(test_data.loc[~mask, col_n]).tolist()
        if len(unqs) > len(replacement):
            
            # Add new categories
            for new_code, category in enumerate(unqs):
                if category not in replacement:
                    replacement[category] = new_code + 1  # new code
        
        
    test_data.loc[:, col_n] = test_data.loc[:, col_n].replace(replacement)
test_data.head()
y_test = test_data['SARS-Cov-2 exam result'].copy()
X_test = test_data.loc[:, test_data.columns != 'SARS-Cov-2 exam result'].copy()

X_test.shape
X_train = X.copy()

normalizer = {}

colnames = set(list(X_train))

# Normalize training and testing data before creating Inputer
for (col_n, col_t) in col_info:
    if col_n not in colnames:  # Target
        continue
    
    if col_t != 'object':  # Min-max normalization just for the numeric values
        cmin = np.min(X_train.loc[:, col_n])
        cmax = np.max(X_train.loc[:, col_n])
        
        mask = X_test.loc[:, col_n].isnull()
        X_train.loc[:, col_n] = (X_train.loc[:, col_n] - cmin) / (cmax - cmin)
        X_test.loc[~mask, col_n] = (X_test.loc[~mask, col_n] - cmin) / (cmax - cmin)
        
        normalizer[col_n] = {'cmin': cmin, 'cmax': cmax}

# Data inputation
inputer = KNNImputer(weights='distance', n_neighbors=5)
inputer.fit(X_train)

X_test = pd.DataFrame(inputer.transform(X_test), columns=list(X_test))
X_test.head()
for (col_n, col_t) in col_info:
    if col_n not in colnames:  # Target
        continue
    
    if col_t != 'object':  # Denormalize
        X_test[col_n] = (normalizer[col_n]['cmax'] - normalizer[col_n]['cmin']) * \
                X_test[col_n] + normalizer[col_n]['cmin']
    else:
        X_test.loc[:, col_n][X_test.loc[:, col_n] <= 0.5] = 0
        X_test.loc[:, col_n][X_test.loc[:, col_n] > 0.5] = 1

X_test.head()
clf = clone(grid.best_estimator_['model'])
scaler = StandardScaler().fit(X)
X_train_transf = scaler.fit_transform(X)
X_test_transf = scaler.transform(X_test)


clf.fit(X_train_transf, y)

y_pred = clf.predict(X_test_transf)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print("Balanced Accuracy: " + str(balanced_accuracy_score(y_test,y_pred)*100)+"%\n")
print(classification_report(y_test, y_pred))