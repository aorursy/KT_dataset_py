import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
import numpy as np

%matplotlib inline
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.drop('customerID', axis='columns', inplace=True)
data.info()
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
print(data.shape)
data.dropna(inplace=True)
print(data.shape)

columns = list(data.columns)
columns.remove('MonthlyCharges')
columns.remove('tenure')
columns.remove('TotalCharges')
print(columns)
sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.75)
for column in sample(columns, 5):
    plt.figure(figsize=(8,8))
    sns.countplot(x=column, hue='Churn', data=data)
    plt.xticks(rotation='45')
    plt.margins(0.2)
    plt.show()
    print('\n\n')
plt.figure(figsize=(8,8))
sns.distplot(data.loc[data['Churn']=='No', 'MonthlyCharges'], label='Churn: No')
sns.distplot(data.loc[data['Churn']=='Yes', 'MonthlyCharges'], label='Churn: Yes')
plt.legend()
plt.show()

plt.figure(figsize=(8,8))
sns.distplot(data.loc[data['Churn']=='No', 'tenure'], label='Churn: No')
sns.distplot(data.loc[data['Churn']=='Yes', 'tenure'], label='Churn: Yes')
plt.legend()
plt.show()

plt.figure(figsize=(8,8))
sns.distplot(data.loc[data['Churn']=='No', 'TotalCharges'], label='Churn: No')
sns.distplot(data.loc[data['Churn']=='Yes', 'TotalCharges'], label='Churn: Yes')
plt.legend()
plt.show()
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
def full_report(classifier, X_test, y_test, class_labels):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
    
    print('Best train score (accuracy) {0:.2f}'.format(clf.best_score_))
    print('Best parameters {}\n'.format(clf.best_params_))

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    test_accuracy = accuracy_score(y_test,y_pred)
    test_f1_score = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_proba)
    
    print('Test Accuracy: {0:.2f}'.format(test_accuracy))
    print('Test f1 score: {0:.2f}'.format(test_f1_score))
    print('Test ROC-AUC: {0:.2f}\n'.format(test_roc_auc))

    print(classification_report(y_test,y_pred, target_names=class_labels))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    sns.set(style="darkgrid")
    sns.set_context("notebook", font_scale=1.75)
    
    cnf_matrix = 100*(cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis])

    plt.figure(figsize=(8,8))
    plt.title('Confusion matrix (%)')
    sns.heatmap(cnf_matrix, 
                annot=True,
                cmap='RdYlBu',
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Observed')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC Curve (area = %0.2f)' % roc_auc_score(y_test, y_proba))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    
    return [test_accuracy, test_f1_score, test_roc_auc]

def FeaturesImportance(importances, labels):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    labels = labels[importances > 0]
    importances = importances[importances > 0]
    
    importances = list(importances)
    labels = list(labels)
    
    ordered_labels =[]
    ordered_importances = []
    
    for _ in range(len(importances)):
        i_max = importances.index(max(importances))
        ordered_labels.append(labels[i_max])
        ordered_importances.append(importances[i_max])
        importances.pop(i_max)
        labels.pop(i_max)

    plt.figure(figsize=(8,8))
    plt.title('Features Importance')
    sns.barplot(x=ordered_importances, y=ordered_labels)
    plt.xlabel('Relative importance')
    plt.ylabel('Feature')
    #plt.tight_layout()
    plt.show()
X = data[columns[:-1]]

le = LabelEncoder()
y = le.fit_transform(data['Churn'])

dummie_columns = []
for column in columns[:-1]:
    if (len(np.unique(data[column]))==2):
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    else:
        dummie_columns.append(column)

print('Dummie columns: {}'.format(dummie_columns))

X = pd.get_dummies(X, columns=dummie_columns)

X['MonthlyCharges'] = data['MonthlyCharges']
X['tenure'] = data['tenure']
X['TotalCharges'] = data['TotalCharges']
X[['MonthlyCharges', 'tenure', 'TotalCharges']] = StandardScaler().fit_transform(data[['MonthlyCharges', 'tenure', 'TotalCharges']])

X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
summary = {}
from sklearn.linear_model import LogisticRegression

params = {
    'penalty':['l1','l2'],
    'tol':[1e-6,1e-5,1e-4,1e-3,1e-2],
    'C':[10,1,0.1,0.01,0.001],  
}

log_reg = LogisticRegression(class_weight='balanced',
                            random_state=42)

clf = GridSearchCV(log_reg, param_grid=params, 
                   cv=5, scoring='accuracy', 
                   verbose=1, n_jobs=-1)

clf.fit(X_train, y_train)
summary['LogisticRegression'] = full_report(clf, X_test, y_test, ['No', 'Yes'])
from sklearn.linear_model import SGDClassifier

params = {
    'loss':['hinge', 'log',  'perceptron'],
    'penalty': [None, 'l2', 'l1', 'elasticnet'],
    'alpha':10.0**-np.arange(1,5),
    'tol':10.0**-np.arange(1,5),
    'eta0':10.0**-np.arange(1,5),
    'learning_rate':['constant', 'optimal', 'invscaling'],
}

sgd_clf = SGDClassifier(class_weight='balanced', 
              n_jobs=-1, random_state=42)

clf = GridSearchCV(sgd_clf, param_grid=params, 
                   cv=5, scoring='accuracy', 
                   verbose=1, n_jobs=-1)

clf.fit(X_train, y_train)
summary['StochasticGradientDescent'] = full_report(clf, X_test, y_test, ['No', 'Yes'])
from sklearn.tree import DecisionTreeClassifier

params = {
    'criterion':['gini','entropy'],
    'splitter':['best','random'],
    'max_depth':list(range(1,11)),
    'min_samples_split':list(range(2,11)),
    'min_samples_leaf':list(range(1,11)),
    'max_features':['auto','sqrt','log2',None],
    
}

dt_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')

clf = GridSearchCV(dt_clf, param_grid=params, 
                   cv=5, scoring='accuracy', 
                   verbose=1, n_jobs=-1)

clf.fit(X_train, y_train)
summary['DecisionTreeClassifier'] = full_report(clf, X_test, y_test, ['No', 'Yes'])
FeaturesImportance(clf.best_estimator_.feature_importances_, X.columns) 
from sklearn.ensemble import RandomForestClassifier

params = {
    'n_estimators':[10]+list(range(50,301,50)),
    'criterion':['gini','entropy'],
    'max_features':['auto','sqrt','log2',None],
    'max_depth':list(range(1,4)),
    'min_samples_split':list(range(2,5)),
    'min_samples_leaf':list(range(1,4)),
}

rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

clf = GridSearchCV(rf_clf, param_grid=params, 
                   cv=5, scoring='accuracy', 
                   verbose=1, n_jobs=-1)

clf.fit(X_train, y_train)
summary['RandomForestClassifier'] = full_report(clf, X_test, y_test, ['No', 'Yes'])
FeaturesImportance(clf.best_estimator_.feature_importances_, X.columns)
summary = pd.DataFrame.from_dict(summary, orient='index')#, columns=['Accuracy', 'f1 score', 'ROC area'])
summary.rename(columns ={0:'Accuracy', 1:'f1 score', 2:'ROC area'}, inplace=True)
print(summary)