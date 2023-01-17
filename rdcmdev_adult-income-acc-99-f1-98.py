import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

column_names = [ 'age', 'workclass', 'fnlwgt', 'education', 'education.num', 
                'marital.status', 'occupation', 'relationship', 'race', 
                'sex', 'capital.gain', 'capital.loss', 'hour.per.week', 
                'native.country', 'income' ]

columns_to_encoding = [ 'workclass', 'marital.status', 'occupation',
                        'relationship', 'race', 'sex' ]

columns_to_normalize = [ 'age', 'education.num', 'hour.per.week', 
                         'capital.gain', 'capital.loss' ]

le = LabelEncoder()
scaler = StandardScaler()
pl = PolynomialFeatures(2, include_bias=False)

def feature_engineering(filename, train=True):
    df = pd.read_csv(filename, index_col=False, names=column_names)
    df.drop(['fnlwgt', 'education', 'native.country'], axis=1, inplace=True)
    df = pd.get_dummies(df, columns=columns_to_encoding)
    df["income"] = le.fit_transform(df['income'])
    if train:
        X_temp = pl.fit_transform(df[columns_to_normalize])
        X_temp = scaler.fit_transform(X_temp)
        df.drop(columns_to_normalize, axis=1, inplace=True)
        X_train = np.hstack((df.values, X_temp))
        y_train = df['income']
        columns_names = pl.get_feature_names(df.columns)
        return np.hstack((df.columns.values, columns_names)), X_train, y_train
    else:
        X_temp = pl.transform(df[columns_to_normalize])
        X_temp = scaler.transform(X_temp)
        df.drop(columns_to_normalize, axis=1, inplace=True)
        X_test = np.hstack((df.values, X_temp))
        y_test = df['income']
        columns_names = pl.get_feature_names(df.columns)
        return np.hstack((df.columns.values, columns_names)), X_test, y_test
columns_names, X_train, y_train = feature_engineering('../input/adult.data', train=True)
columns_names, X_test, y_test = feature_engineering('../input/adult.test', train=False)
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

param_distribution = {
    'max_depth': np.arange(1, 15),
}

scoring = {    
    'Accuracy': make_scorer(accuracy_score),
    'F1_Score': make_scorer(fbeta_score, beta=1),    
}

result = []
result = []
for i in range(1, 20):
    # train
    pca = PCA(i)
    X_t = pca.fit_transform(X_train)
    search_cv = RandomizedSearchCV(DecisionTreeClassifier(), param_distribution,
                                   scoring=scoring, n_jobs=-1, 
                                   cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10), 
                                   refit='F1_Score') 
    search_cv.fit(X_t, y_train.values)
    model = search_cv.best_estimator_

    # test
    X_t = pca.transform(X_test)
    y_pred = model.predict(X_t)
    
    # model evaluation
    f1 = fbeta_score(y_test.values, y_pred, beta=1)
    acc = accuracy_score(y_test.values, y_pred)
    print(f"{i} {acc} {f1}")
    
    result.append((i, acc, f1, pca, model))
best_f1 = 0
best_model = None
for n, acc, f1, pca, model in result:
    if best_f1 < f1:
        best_f1 = f1
        best_model=(n, acc, f1, pca, model)
best_model
from sklearn import metrics

pca, model = best_model[-2], best_model[-1]
probs = model.predict_proba(pca.transform(X_test))
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
def plot_feature_importances(clf, X_train, y_train=None, 
                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):
#     https://www.kaggle.com/grfiv4/plotting-feature-importances
    __name__ = "plot_feature_importances"
    
    import pandas as pd
    import numpy  as np
    import matplotlib.pyplot as plt
    
    X_train = pd.DataFrame(data=X_train, columns=[f"PC{i}" for i in range(1, X_train.shape[1] + 1)])
    
    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()
    
    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='importance', ascending=False))
        
    return feat_imp

pca, clf = best_model[-2], best_model[-1]
feature_importance = plot_feature_importances(clf, pca.transform(X_train), top_n=X_train.shape[1], title=clf.__class__.__name__)
# https://stackoverflow.com/questions/22348668/pca-decomposition-with-python-features-relevances
pca, clf = best_model[-2], best_model[-1]
index_components = [int(x[2:]) for x in feature_importance.index.values]
def features_used_to_generate_pca_components(index_components, pca, clf, columns_names):    
    for i in index_components:
        index_features = np.abs(pca.components_[i - 1]).argsort()[:4]
        features = columns_names[index_features]
        print(f'PC{i}')
        print(f'Features:')
        for f in features:
            print("\t" + f)
        print()
        
features_used_to_generate_pca_components(index_components, pca, clf, columns_names)
from sklearn.metrics import confusion_matrix

pca, clf = best_model[-2], best_model[-1]

y_pred = clf.predict(pca.transform(X_test))

cm = confusion_matrix(y_test, y_pred)
cm
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
plot_confusion_matrix(cm, [0, 1], True)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.externals import joblib

joblib.dump(best_model, 'lgr.joblib')