# imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import pprint
import lightgbm as lgb
#from feature_selector import FeatureSelector
pp = pprint.PrettyPrinter(indent=4)
%matplotlib inline
pd.set_option('display.max_rows', 500)
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
#Loading the data

#friday = pd.read_csv('/kaggle/input/Friday-WorkingHours-Afternoon-DDos.csv',low_memory = False)
wednesday = pd.read_csv('/kaggle/input/Wednesday-workingHours.csv',low_memory = False)
#friday = friday.rename(str.lstrip, axis='columns')
wednesday = wednesday.rename(str.lstrip, axis='columns')
#drop rows with NA (only done because NA is not defining element in the feature set)
df = wednesday
#df = df.sample(n = 200000)
print(df['Label'].unique())
print(df.shape)
df = df.dropna()
print(df.shape)
#defining important features in the set| not included in feature selection.
imp = ['Destination Port', 'Flow Duration', 'Total Fwd Packets','Total Backward Packets']
#for performing feature selection
X = df.loc[:,df.columns != "Label"]
#removing first four features considered very important
X = X.drop(imp,axis = 1)
y = df.loc[:,df.columns == "Label"]
print(X.shape)
#class distribution <- classifying feature
def ClassDistribution(y):
    ax = sns.countplot(y['Label'],label="Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    print(y.groupby('Label').size())

ClassDistribution(y)
#B, M = y.value_counts()
#print('Number of Benign: ',B)
#print('Number of Malignant : ',M)
#remove columns with more than 75% missing values
def RemoveMissing(train):
    train_missing = (train.isnull().sum() / len(train)).sort_values(ascending = False)
    #train_missing.head()
    train_missing = train_missing.index[train_missing > 0.75]
    all_missing = list(set(train_missing))
    train.drop(all_missing,axis = 1)
    print('There are %d columns with more than 75%% missing values' % len(all_missing))
    return train

X = RemoveMissing(X)
#remove columns with only 1 unique value
def remove_single_unique_values(dataframe):
    cols_to_drop = dataframe.nunique()
    cols_to_drop = cols_to_drop.loc[cols_to_drop.values==1].index
    print('There are %d columns with only 1 unique value' % len(cols_to_drop))
    dataframe = dataframe.drop(cols_to_drop,axis=1)
    return dataframe

X = remove_single_unique_values(X)
#heatmap
f,ax = plt.subplots(figsize=(18, 18))
corr = X.corr()
sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
columns = np.full((corr.shape[0],), True, dtype=bool)
to_drop = []
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.7:
            if columns[j]:
                columns[j] = False
                to_drop.append((corr.columns)[j])
print(to_drop)
X = X.drop(to_drop,axis = 1)
print(X.shape)
#add back the important features removed earlier
for i in imp:
    X[i] = df[i]

print(X.shape)
print(X.columns)
#encoding and scaling stuff.
categorical = []
for x in X.columns:
    if X[x].dtype == 'object':
        categorical.append(x)
#encoding and scaling

encoder = LabelEncoder()
for a in categorical:
    X[a] = encoder.fit_transform(X[a])

# feature scaling
scaler = RobustScaler()
X = scaler.fit_transform(X)
y = encoder.fit_transform(y.values)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=123, stratify=y)
#DOES NOT ENHANCE PERFORMACE DO NOT RECOMMEND.
#added in version 14 imputation and undersampling
#for benchmarking against unaltered data distribution run V13
"""

#Impute missing values
print(x_train.shape)
imp = SimpleImputer()
imp.fit(x_train)
x_train = imp.transform(x_train)
x_test = imp.transform(x_test)
print(x_train.shape)

# Implement RandomUnderSampler
random_undersampler = RandomUnderSampler()
x_train, y_train = random_undersampler.fit_sample(x_train, y_train)
print(x_train.shape)
# Shuffle the data
perms = np.random.permutation(x_train.shape[0])
x_train = x_train[perms]
y_train = y_train[perms]
print(x_train.shape)
"""
"""
#added in version 14 imputation and undersampling
#for benchmarking against unaltered data distribution run V13
# SMOTE + Random Undersampling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

over = SMOTE()
under = RandomUnderSampler()
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
x_train, y_train = pipeline.fit_resample(x_train, y_train)
"""
print(x_train.shape)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(n_estimators = 10, random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")
logreg_clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial',n_jobs = 3)

rf_clf = RandomForestClassifier(n_estimators=10, random_state = 43)

#does not support multilabel
#pca_clf = PCA(n_components='mle')

ann_clf = MLPClassifier(hidden_layer_sizes=(13,7),activation = 'logistic')

lda_clf = LinearDiscriminantAnalysis(solver='svd')

#knn_clf = KNeighborsClassifier(n_neighbors=3)

nb_clf = GaussianNB()

dtree_clf = DecisionTreeClassifier()


models = {"logreg": logreg_clf,
            "rf": rf_clf,
            "Neural-Network": ann_clf,
            "LDA" : lda_clf,
#            "KNN" : knn_clf,
            "Naive-Bayes" : nb_clf,
            "Decision-Tree" : dtree_clf}


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Scores = []
def Utility():
    for x in models:
        print(x)
        score = (cross_val_score(models[x], x_train, y_train, cv=5))
        Scores.append(score)
        print(score)
        

Utility()

for x in Scores:
    print(np.mean(x))

#For two classes
#prints classificationreport , ROC, PR, and CM
def plot_confusion_matrix(y_true, y_pred,normalize=True,title=None,cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    classes = np.unique(y_test)
    np.append(classes,np.unique(y_pred))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    labels = classes
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    return ax

#fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#y_pred = []

for name,model in models.items() :
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    #y_prob = model.predict_proba(x_test)[:, 1:]
    print(name)
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test,y_pred, title = name)
    
    #precision, recall,_ = precision_recall_curve(y_test, y_prob)
    #model_auc_score = roc_auc_score(y_test, y_prob)
    #fpr, tpr, _ = roc_curve(y_test, y_prob)
    #axes[0].plot(fpr, tpr, label= f"{name}, auc = {model_auc_score:.3f}")
    #axes[1].plot(recall, precision, label= f"{name}")
"""
axes[0].legend(loc="lower right")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].set_title("ROC curve")
axes[1].legend()
axes[1].set_xlabel("recall")
axes[1].set_ylabel("precision")
axes[1].set_title("PR curve")
plt.tight_layout()
plt.show()
"""
plt.show()
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy.interpolate import interp1d
#Multiclass plots of PR and ROC
def Multiclassplots(classifier,Name):
    print(classifier)
    #print(Name)
    #print(len(set(y.Label)))
    
    n_classes = len(set(y))

    Y = label_binarize(y, classes=[*range(n_classes)])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42,stratify = Y)

    fig,ax = plt.subplots(1,2, figsize = (14,6))
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train,y_train)
    y_score = clf.predict_proba(X_test)
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:,i])
        ax[0].plot(recall[i],precision[i], lw=2, label='class {}'.format(i))

    fpr = dict()
    tpr = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i],
                                      y_score[:,i])
        auc_score = roc_auc_score(y_test[:,i], y_score[:,i])
        ax[1].plot(fpr[i], tpr[i], lw=2, label='class {}, auc = {stacked_auc_score:.3f}'.format(i))

    ax[0].legend(loc="lower right")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title("ROC curve "+ Name)
    ax[1].legend()
    ax[1].set_xlabel("recall")
    ax[1].set_ylabel("precision")
    ax[1].set_title("PR curve " + Name)
    plt.tight_layout()
    plt.show()


#for x in models:
#    Multiclassplots(models[x],x)


# Define meta-learner
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import FunctionTransformer
from imblearn.ensemble import BalancedBaggingClassifier

logreg_clf = LogisticRegression(penalty="l2", C=100, fit_intercept=True)
# Fitting voting clf --> average ensemble
voting_clf = VotingClassifier([("mlp", ann_clf),
                               ("LDA", lda_clf),
                               ("decision_tree", dtree_clf)],
                              voting="soft",
                              flatten_transform=True)
xgb_clf = xgb.XGBClassifier(objective="multi:softmax",
                            learning_rate=0.1,
                            n_estimators=100,
                            max_depth=10,
                            random_state=123)
bbg_clf = BalancedBaggingClassifier(base_estimator= ann_clf,random_state=42)


xgb_clf.fit(x_train,y_train)
y_score = xgb_clf.predict_proba(x_test)
bbg_clf.fit(x_train,y_train)
y_score = bbg_clf.predict_proba(x_test)
voting_clf.fit(x_train, y_train)
mlp_model, lda_model, dtree_model = voting_clf.estimators_
models = {"mlp": mlp_model,
          "lda": lda_model,
          "dtree": dtree_model,
          "avg_ensemble": voting_clf}

# Build first stack of base learners
first_stack = make_pipeline(voting_clf,
                            FunctionTransformer(lambda X: X[:, 1::2]))

# Use CV to generate meta-features
meta_features = cross_val_predict(first_stack, x_train, y_train, cv=3, method="transform")
print('check1')
# Refit the first stack on the full training set
first_stack.fit(x_train, y_train)
print('first stack fitted')
# Fit the meta learner
second_stack = logreg_clf.fit(meta_features, y_train)
third_stack = xgb_clf.fit(meta_features,y_train)
fourth_stack = bbg_clf.fit(meta_features,y_train)
print('second third and fourth')
models["stacked"] = logreg_clf
models["boosting"] = third_stack
models["bagging"] = fourth_stack
for x in models:
    Multiclassplots(models[x],x)