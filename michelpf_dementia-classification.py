import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

df_long = pd.read_csv('../input/oasis_longitudinal.csv')
df_long
df_long.isnull().values.any()
df_long.isnull().head(5)
nans = lambda df: df[df.isnull().any(axis=1)]
nans(df_long)
df_long.isin([0]).sum()
df_long["SES"].fillna(df_long["SES"].mean(), inplace=True)
df_long["MMSE"].fillna(df_long["MMSE"].mean(), inplace=True)
nans(df_long)
import numpy as np

def outliers_z_score(ys):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def generate_chars(field):
    plt.figure()
    g = None
    if field == "MR Delay":
        df_query_mri = df_long[df_long["MR Delay"] > 0]
        g = sns.countplot(df_query_mri["MR Delay"])
        g.figure.set_size_inches(18.5, 10.5)
    else:
        g = sns.countplot(df_long[field])
        g.figure.set_size_inches(18.5, 10.5)
    
sns.despine()
group_map = {"Demented": 1, "Nondemented": 0}
gender_map = {"M": 1, "F": 0}
hand_map = {"R": 1, "L": 0}

df_long['Group'] = df_long['Group'].replace(['Converted'], ['Demented'])

df_long['Group'] = df_long['Group'].map(group_map)
df_long['M/F'] = df_long['M/F'].map(gender_map)
df_long['Hand'] = df_long['Hand'].map(hand_map)
generate_chars("M/F")
generate_chars("Hand")
generate_chars("MR Delay")
generate_chars("Age")
generate_chars("EDUC")
generate_chars("SES")
generate_chars("MMSE")
generate_chars("CDR")
generate_chars("eTIV")
generate_chars("nWBV")
generate_chars("ASF")
list_atributes = ["M/F", "MR Delay", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
for item in list_atributes:
    print(outliers_z_score(df_long[item]))
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5

corr_matrix = df_long.corr()
rcParams['figure.figsize'] = 15, 10
sns.heatmap(corr_matrix)
df_long.corr()
def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.show()
from sklearn.model_selection import train_test_split

feature_col_names = ["M/F", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
predicted_class_names = ['Group']

X = df_long[feature_col_names].values
y = df_long[predicted_class_names].values

split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)
print("{0:0.2f}% para Treinamento".format((len(X_train)/len(df_long.index)) * 100))
print("{0:0.2f}% para Testes".format((len(X_test)/len(df_long.index)) * 100))
print("Original Demented : {0} ({1:0.2f}%)".format(len(df_long.loc[df_long['Group'] == 1]), 100 * (len(df_long.loc[df_long['Group'] == 1]) / len(df_long))))
print("Original Nondemented : {0} ({1:0.2f}%)".format(len(df_long.loc[df_long['Group'] == 0]), 100 * (len(df_long.loc[df_long['Group'] == 0]) / len(df_long))))
print("")
print("Training Demented : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), 100 * (len(y_train[y_train[:] == 1]) / len(y_train))))
print("Training Nondemented : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), 100 * (len(y_train[y_train[:] == 0]) / len(y_train))))
print("")
print("Test Demented : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), 100 * (len(y_test[y_test[:] == 1]) / len(y_test))))
print("Test Nondemented : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), 100 * (len(y_test[y_test[:] == 0]) / len(y_test))))
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=len(feature_col_names),
                           n_informative=3, n_redundant=0, n_repeated=0,
                           n_classes=2, random_state=0, shuffle=False)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_*100
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:\n")

for f in range(X.shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f], feature_col_names[f], importances[indices[f]]))

col_name = lambda x : feature_col_names[x]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), list(map(col_name,range(X.shape[1]))))
plt.xlim([-1, X.shape[1]])
plt.show()
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(max_depth=2, random_state=0)
rf_model.fit(X_train, y_train.ravel())
from sklearn import metrics

def report_performance(model):

    model_test = model.predict(X_test)

    print("Confusion Matrix")
    print("{0}".format(metrics.confusion_matrix(y_test, model_test)))
    print("")
    print("Classification Report")
    print(metrics.classification_report(y_test, model_test))
    
    cm = metrics.confusion_matrix(y_test, model_test)
    show_confusion_matrix(cm, ["Nondemented","Demented"])
report_performance(rf_model)
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [2, 4, 6, 8, 12, 14],
              "max_features": [9],
              "min_samples_split": [2, 4, 6, 8, 10],
              "min_samples_leaf": [2, 4, 6, 8, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
clf_rf = RandomForestClassifier(n_estimators=20)

grid_search = GridSearchCV(clf_rf, param_grid=param_grid, cv=10)
grid_search.fit(X, y)
report(grid_search.cv_results_)
from sklearn.ensemble import RandomForestClassifier

rf_model_opt = RandomForestClassifier(bootstrap=True, criterion="entropy", max_depth=8, 
                                  max_features=9, min_samples_leaf=2, min_samples_split=10, random_state=0)
rf_model_opt.fit(X_train, y_train.ravel())
report_performance(rf_model_opt)
from sklearn import svm

svm_linear_model = svm.SVC(kernel='linear')
svm_linear_model.fit(X_train, y_train.ravel())
report_performance(svm_linear_model)
import scipy

param_grid_svm = [
  {'C': [1, 10, 1000], 'kernel': ['linear'], 'shrinking': [True, False], 'decision_function_shape': ['ovo', 'ovr', None]},
  {'C': [1, 10, 1000], 'kernel': ['rbf'], 'degree' : [2, 4], 'gamma': [0.1, 0.01, 0.001], 'shrinking': [True, False], 'decision_function_shape': ['ovo', 'ovr', None]},  
  {'kernel': ['poly', 'sigmoid'], 'degree' : [2, 4], 'coef0': [0, 1, 2, 3], 'shrinking': [True, False], 'decision_function_shape': ['ovo', 'ovr', None]}
 ]
from sklearn.model_selection import GridSearchCV

clf_svm = svm.SVC()

grid_search = GridSearchCV(clf_svm, param_grid=param_grid_svm, cv=10)
grid_search.fit(X, y)
report(grid_search.cv_results_)
from sklearn import svm

svm_model_opt = svm.SVC(C=1000, gamma=0.01, kernel="rbf", degree=2, shrinking=True, decision_function_shape="ovo")
svm_model_opt.fit(X_train, y_train.ravel())
report_performance(svm_model_opt)
