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
        return df, X_train, y_train
    else:
        X_temp = pl.transform(df[columns_to_normalize])
        X_temp = scaler.transform(X_temp)
        df.drop(columns_to_normalize, axis=1, inplace=True)
        X_test = np.hstack((df.values, X_temp))
        y_test = df['income']
        return df, X_test, y_test
df_train, X_train, y_train = feature_engineering('../input/adult.data', train=True)
df_test, X_test, y_test = feature_engineering('../input/adult.test', train=False)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

param_distribution = {
    'C': np.arange(1, 15),
}

scoring = {    
    'Accuracy': make_scorer(accuracy_score),
    'F1_Score': make_scorer(fbeta_score, beta=1),    
}

result = []
result = []
for i in range(1, 13):
    # train
    pca = PCA(i)
    X_t = pca.fit_transform(X_train)
    search_cv = RandomizedSearchCV(LogisticRegression(solver='lbfgs'), param_distribution,
                                   scoring=scoring, n_jobs=-1, 
                                   cv=StratifiedKFold(n_splits=10, shuffle=True), 
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
from sklearn.externals import joblib

joblib.dump(best_model, 'lgr.joblib')