# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
        
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print (os.listdir("../input"))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
oasisdf = pd.read_csv("../input/oasis_longitudinal.csv")
oasisdf.tail(10)
oasisdf.isnull().sum()
oasisdf.dropna().describe()
oasisdf.dropna(axis=1).describe()
oasisdf['SES'].describe()
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
oasisdf['SES'] = imp.fit_transform(oasisdf[['SES']])

oasisdf['SES'].describe()
oasisdf['MMSE'].describe()
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
oasisdf['MMSE'] = imp.fit_transform(oasisdf[['MMSE']])

oasisdf['MMSE'].describe()
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
oasisdfScaled = mms.fit_transform(oasisdf[['SES', 'EDUC']])
oasisdf[['SES', 'EDUC']] = oasisdfScaled
oasisdf[['SES', 'EDUC']].describe()
oasisdf.head()
oasisdf = pd.get_dummies(oasisdf, columns = ['M/F', 'Group']) 
oasisdf.head()
list(set(oasisdf.dtypes.tolist()))
oasisdf['Group_Demented'] = oasisdf['Group_Demented'].astype(np.float64)
oasisdf['Group_Demented'].dtypes
df_num = oasisdf.select_dtypes(include = ['float64', 'int64'])
df_num.head()
cols = ['Age', 'EDUC', 'SES', 'M/F_F', 'Group_Demented', 'Hand']
nr_rows = 3
nr_cols = 2

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*8,nr_rows*4))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        
        i = r*nr_cols+c       
        ax = axs[r][c]
        sns.countplot(oasisdf[cols[i]], hue=oasisdf["Group_Demented"], ax=ax)
        ax.set_title(cols[i])
        ax.legend() 
        
plt.tight_layout() 
sns.barplot(x='SES', y='Group_Demented', data=oasisdf)
plt.ylabel("Dementia Probability")
plt.title("Dementia as function of Social Class")
plt.show()
sns.barplot(x='EDUC', y='Group_Demented', data=oasisdf)
plt.ylabel("Dementia Probability")
plt.title("Dementia as function of Education")
plt.show()
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); 
oasisdf_num_corr = oasisdf.corr()['Group_Demented'][:-1]
golden_features_list = oasisdf_num_corr[abs(oasisdf_num_corr) > 0.2].sort_values(ascending=False)
print("There are {} strongly correlated values with Group_Demented:\n{}".format(len(golden_features_list), golden_features_list))
for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['Group_Demented'])
corrmat = df_num.corr()
fig,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True)
sns.boxplot('Group_Demented','EDUC', data = df_num)
plt.show()
sns.boxplot('Group_Demented','SES', data = df_num)
plt.show()
from pandas.tools.plotting import scatter_matrix
sm = scatter_matrix(df_num, alpha=0.2, figsize=(14,14), diagonal='kde')
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage # hacer enclaces
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk #algoritmos de machine learning
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.formula.api as smf #regresión lineal
df_num.head()
features = ['Visit', 'MR Delay', 'Age', 'EDUC', 'SES', 'CDR', 'eTIV', 'nWBV', 'ASF','MMSE']
df_num['Group_Demented'].value_counts()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_num[features],
                                                    df_num["Group_Demented"],
                                                    test_size=0.3,
                                                    stratify=df_num['Group_Demented'])

from sklearn import linear_model
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X_train, y_train)


print("Logistic Regression score (Train): {0:.2}".format(lr.score(X_train, y_train)))
print("Logistic Regression score (Test): {0:.2}".format(lr.score(X_test, y_test)))
#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print("KNN score (Train): {0:.2}".format(neigh.score(X_train, y_train)))
print("KNN score (Test): {0:.2}".format(neigh.score(X_test, y_test)))
#http://scikit-learn.org/stable/modules/svm.html
from sklearn import svm
svclass = svm.SVC(gamma='scale')
svclass.fit(X_train, y_train) 
print("SVM score (Train): {0:.2}".format(svclass.score(X_train, y_train)))
print("SVM score (Test): {0:.2}".format(svclass.score(X_test, y_test)))
#http://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
print("Decision Tree score (Train): {0:.2}".format(dt.score(X_train, y_train)))
print("Decision Tree score (Test): {0:.2}".format(dt.score(X_test, y_test)))
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)
X_train.head()
forest.fit(X_train, y_train)
print("Random Forest score (Train): {0:.2}".format(forest.score(X_train, y_train)))
print("Random Forest score (Test): {0:.2}".format(forest.score(X_test, y_test)))
model=lr
#WAAAAARNING: not all models have the "feature_importances_" functions
plt.bar(np.arange(len(features)), model.feature_importances_)
plt.xticks(np.arange(len(features)), features, rotation='vertical', ha='left')
plt.tight_layout()
from sklearn.model_selection import cross_val_score
def validate(model, X_train, y_train, k=10):
    result = 'K-fold cross validation:\n'
    scores = cross_val_score(estimator=model,
                             X=X_train,
                             y=y_train,
                             cv=k,
                             n_jobs=1)
    for i, score in enumerate(scores):
        result += "Iteration %d:\t%.3f\n" % (i, score)
    result += 'CV accuracy:\t%.3f +/- %.3f' % (np.mean(scores), np.std(scores))
    return result
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def learningCurve(model, X_train, y_train, k=10):
    train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=model,
                                   X=X_train,
                                   y=y_train,
                                   train_sizes=np.linspace(0.1, 1.0, 10),
                                   cv=k,
                                   n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.rcParams["figure.figsize"] = [6,6]
    fsize=14
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples', fontsize=fsize)
    plt.ylabel('Accuracy', fontsize=fsize)
    plt.legend(loc='lower right')
    plt.ylim([0.4, 1.03])
    plt.tight_layout()
    plt.show()
from sklearn.model_selection import validation_curve

def validationCurve(model, X_train, y_train,p_name, p_range, k=10, scale=False):
    train_scores, test_scores = validation_curve(
                    estimator=model, 
                    X=X_train, 
                    y=y_train, 
                    param_name=p_name,
                    param_range=p_range,
                    cv=k)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.rcParams["figure.figsize"] = [6,6]
    fsize=14
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.plot(p_range, train_mean, 
             color='blue', marker='o', 
             markersize=5, label='training accuracy')

    plt.fill_between(p_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(p_range, test_mean, 
             color='green', linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')

    plt.fill_between(p_range, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color='green')

    plt.grid()
    if scale:
        plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter %s' % p_name, fontsize=fsize)
    plt.ylabel('Accuracy', fontsize=fsize)
    plt.ylim([0.7, 1.0])
    plt.tight_layout()
    plt.show()
from sklearn.metrics import roc_curve, roc_auc_score

def rocCurve(model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams["figure.figsize"] = [8,8]
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
from sklearn.metrics import confusion_matrix

def confusionMatrix(model, X_train, y_train, X_test, y_test): 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.8)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X_train, y_train)


print("Logistic Regression score (Train): {0:.2}".format(lr.score(X_train, y_train)))
print("Logistic Regression score (Test): {0:.2}".format(lr.score(X_test, y_test)))
print(validate(lr, X_train, y_train))

learningCurve(lr, X_train, y_train)
rocCurve(lr, X_test, y_test)

confusionMatrix(lr, X_train, y_train, X_test, y_test)
#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print("KNN score (Train): {0:.2}".format(neigh.score(X_train, y_train)))
print("KNN score (Test): {0:.2}".format(neigh.score(X_test, y_test)))
print(validate(neigh, X_train, y_train))
learningCurve(neigh, X_train, y_train)
rocCurve(neigh, X_train, y_train)
confusionMatrix(neigh, X_train, y_train, X_test, y_test)

#http://scikit-learn.org/stable/modules/svm.html
from sklearn.svm import SVC
svclass = SVC(probability=True)
svclass.fit(X_train, y_train) 
print("SVM score (Train): {0:.2}".format(svclass.score(X_train, y_train)))
print("SVM score (Test): {0:.2}".format(svclass.score(X_test, y_test)))
print(validate(svclass, X_train, y_train))
learningCurve(svclass, X_train, y_train)
rocCurve(svclass, X_test, y_test)
confusionMatrix(svclass, X_train, y_train, X_test, y_test)
#http://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
print("Decision Tree score (Train): {0:.2}".format(dt.score(X_train, y_train)))
print("Decision Tree score (Test): {0:.2}".format(dt.score(X_test, y_test)))
print(validate(dt, X_train, y_train))
learningCurve(dt, X_train, y_train)

rocCurve(dt, X_train, y_train)
confusionMatrix(dt, X_train, y_train, X_test, y_test)
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=3,
                                criterion='gini',
                                max_depth=3,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)
X_train.head()
forest.fit(X_train, y_train)
print("Random Forest score (Train): {0:.2}".format(forest.score(X_train, y_train)))
print("Random Forest score (Test): {0:.2}".format(forest.score(X_test, y_test)))
print(validate(forest, X_train, y_train))
learningCurve(forest, X_train, y_train)
rocCurve(forest, X_test, y_test)
confusionMatrix(forest, X_train, y_train, X_test, y_test)
validationCurve(lr, X_train, y_train, p_name='C', p_range=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], scale=True)
validationCurve(neigh, X_train, y_train, p_name='n_neighbors', p_range=[1,2,3,4,5,10,15,20])
validationCurve(dt, X_train, y_train, p_name='max_depth', p_range=[1,2,3,4,5,10,20])
validationCurve(forest, X_train, y_train, p_name='n_estimators', p_range=[1,2,3,4,5,10,20])