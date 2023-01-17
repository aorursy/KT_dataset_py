import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 50)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
df = pd.read_csv('../input/data.csv')
df.sample(10)
print(df.diagnosis.nunique())
df.diagnosis.unique()
df['diagnosis_cat'] = pd.factorize(df['diagnosis'])[0]
df.shape
df.info()
df.columns
df.isnull().any()
df = df.drop(['id', 'Unnamed: 32'], 1)
df.describe()
plt.rcParams['figure.figsize']=(20,19)
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt = ".2f", cmap="BuPu");
plt.rcParams['figure.figsize']=(8,8)
ax = sns.countplot(x = 'diagnosis', data = df, palette = 'hls');
ax.set_title(label='Diagnosis distribution', fontsize=15);
names= 'B', 'M'
size=df['diagnosis'].value_counts()

my_circle=plt.Circle((0,0), 0.7, color='white')

plt.pie(size, labels=names, colors=['skyblue','red'])
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
g = sns.pairplot(df.iloc[:,0:11], hue = 'diagnosis');
g = g.map_diag(plt.hist, histtype="step", linewidth=3)
sns.pairplot(df.iloc[:,11:21]);
sns.pairplot(df.iloc[:,21:31]);
v = sns.PairGrid(df.iloc[:,21:31])
v.map_lower(sns.kdeplot);
v.map_upper(plt.scatter);
v.map_diag(sns.kdeplot);
def feats(df):
    feats_from_df = set(df.select_dtypes([np.int, np.float]).columns.values)
    bad_feats = {'diagnosis_cat'}
    return list(feats_from_df - bad_feats)

df_scaled = df
df_scaled[feats(df)] = preprocessing.scale(df[feats(df)])
plt.subplots(figsize=(20,5))
df_melted = pd.melt(df_scaled, id_vars = "diagnosis", 
                      value_vars = ('radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                                    'smoothness_mean', 'compactness_mean', 'concavity_mean',
                                    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'))
sns.violinplot(x = "variable", y = "value", hue="diagnosis",data= df_melted);
plt.subplots(figsize=(20,5))
df_melted = pd.melt(df_scaled, id_vars = "diagnosis", 
                      value_vars = ('radius_se', 'texture_se', 'perimeter_se', 'area_se',
                                    'smoothness_se', 'compactness_se', 'concavity_se',
                                    'concave points_se', 'symmetry_se', 'fractal_dimension_se'))
sns.violinplot(x = "variable", y = "value", hue="diagnosis",data= df_melted);
plt.subplots(figsize=(20,5))
df_melted = pd.melt(df_scaled, id_vars = "diagnosis", 
                      value_vars = ('radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                                    'smoothness_worst', 'compactness_worst', 'concavity_worst',
                                    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'))
sns.violinplot(x = "variable", y = "value", hue="diagnosis",data= df_melted);
plt.rcParams['figure.figsize']=(20,5)

mean_value = ('radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
              'smoothness_mean', 'compactness_mean', 'concavity_mean',
               'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean')

for i, feat in enumerate(mean_value):
    m = plt.hist(df[df["diagnosis"] == "M"][feat],bins=30,fc = (1,0,0,0.5),label = "Malignant")
    b = plt.hist(df[df["diagnosis"] == "B"][feat],bins=30,fc = (0,1,0,0.5),label = "Bening")
    plt.legend()
    plt.xlabel(mean_value[i] + ' values')
    plt.ylabel("Frequency")
    plt.title("Histogram of " + mean_value[i] +  " for bening and malignant breast cancer")
    plt.show()
def feats(df):
    feats_from_df = set(df.select_dtypes([np.int, np.float]).columns.values)
    bad_feats = {'diagnosis', 'diagnosis_cat'}
    return list(feats_from_df - bad_feats)

def model_train_predict(model, X, y, success_metric=accuracy_score):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return success_metric(y_val, y_pred)

def plot_learning_curve(model, title, X, y, ylim=None, cv = None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure(figsize=(12,8))
    plt.title(title)
    if ylim is not None:plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing score")

    plt.legend(loc="best")
    return plt
X = df_scaled[feats(df_scaled)].values
y = df_scaled['diagnosis_cat']
models = [
    LogisticRegression(penalty = 'l2'),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10)
]

for model in models:
    print(str(model) + ": ")
    %time score = model_train_predict(model, X, y)
    print(str(score) + "\n")
    plt = plot_learning_curve(model, "Learning Curves", X, y, ylim=(0.5, 1.2), cv=15, n_jobs=4)
    plt.show()
models = [
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10)
]

for model in models:
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title('Feature importances: ' + str(model).split('(')[0])
    plt.bar(range(X.shape[1]), model.feature_importances_[indices],
           color = 'b', align = 'center')
    plt.xticks(range(X.shape[1]), [ feats(df_scaled)[x] for x in indices])
    plt.xticks(rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
def compute(params):
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    #print("Score: %.2f" % score)
    #print(params)
    return (1 - score)

space = {
        'max_depth':  hp.choice('max_depth', range(4,6)),
        'min_child_weight': hp.uniform('min_child_weight', 0, 10),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05)
    }

best = fmin(compute, space, algo=tpe.suggest, max_evals=250)
print(best)
model = xgb.XGBClassifier(**best)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy_score(y_val, y_pred)
plt = plot_learning_curve(model, "Learning Curves", X, y, ylim=(0.5, 1.2), cv=15, n_jobs=4)
plt.show()
def confusion_matrix(y_val,y_pred):
    confusion_matrix = metrics.confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(5,5))
    ax= plt.subplot()
    sns.heatmap(confusion_matrix, annot=True,fmt='g', ax = ax);
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    plt.show()
confusion_matrix(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred)
print('AUC: %.3f' % auc)
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.show()
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.fit_transform(X_val)

model_pca = LogisticRegression()
model_pca.fit(X_val_pca, y_val)
model_pca.score(X_val_pca, y_val)
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
df_pca = pd.DataFrame(data = X_train_pca, columns = ['PCA_1', 'PCA_2', 'PCA_3'])
df_pca = pd.concat([df_pca, df_scaled['diagnosis']], axis =1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1: {0}%'.format(per_var[0]), fontsize = 15)
ax.set_ylabel('Principal Component 2: {0}%'.format(per_var[1]), fontsize = 15)

targets = ['B', 'M']
colors = ['g', 'r',]
for target, color in zip(targets,colors):
    indicesToKeep = df_pca['diagnosis'] == target
    ax.scatter(df_pca.loc[indicesToKeep, 'PCA_1']
               , df_pca.loc[indicesToKeep, 'PCA_2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
model = SVC(gamma='auto')
model.fit(X_train, y_train)
model.score(X_val, y_val)
plt = plot_learning_curve(model, "Learning Curves", X, y, ylim=(0.5, 1.2), cv=15, n_jobs=4)
plt.show()
model = LinearSVC(random_state=0, tol=1e-5)
model.fit(X_train, y_train)
model.score(X_val, y_val)
plt = plot_learning_curve(model, "Learning Curves", X, y, ylim=(0.5, 1.2), cv=15, n_jobs=4)
plt.show()
model = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)
model.score(X_val, y_val)
plt = plot_learning_curve(model, "Learning Curves", X, y, ylim=(0.5, 1.2), cv=15, n_jobs=4)
plt.show()
model = GaussianNB()
model.fit(X_train, y_train)
model.score(X_val, y_val)
plt = plot_learning_curve(model, "Learning Curves", X, y, ylim=(0.5, 1.2), cv=15, n_jobs=4)
plt.show()