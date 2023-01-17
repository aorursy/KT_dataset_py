import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from imblearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_regression
### get data from given csv files ###

filename = "train_set.csv"
data_train = pd.read_csv(filename)
data_train=data_train.fillna(value=-1)

filename = "test_set.csv"
data_test = pd.read_csv(filename)
data_test=data_test.fillna(value=-1)


def scale(X_data):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
    X_data = min_max_scaler.fit_transform(X_data) 
    return X_data
    
def pca(X_data):
    pca=decomposition.KernelPCA(kernel='sigmoid',n_components=40)
    pca=pca.fit(X_data)
    X_data=pca.transform(X_data)
    return X_data

## drop columns with over 40% missing values ###

to_drop_missing=list()
for column in data_train.columns:
    counts = data_train[column].value_counts(ascending=True)
    for count in range (counts.size):
        if counts.index[count]==-1 and float(counts.iloc[count])/float(data_train[column].size) >0.4:
            to_drop_missing.append(column)
        
data_train=data_train.drop(columns=to_drop_missing)
### replace missing values with median ###

def getColumnsString(substring,data_train):
    substring_col=list()
    for column_name in data_train.columns:
        if column_name.find(substring)!=-1:
            substring_col.append(column_name)
    return substring_col


def replaceMissingValues(data_train):

    #for binary and categorical data we use the most frequent value
    bin_col=data_train[getColumnsString('bin',data_train)]
    cat_col=data_train[getColumnsString('cat',data_train)]

    bin_cat_col=pd.concat([bin_col,cat_col],axis=1)


    imputer = SimpleImputer(missing_values=-1,strategy='most_frequent')
    bin_cat_col = pd.DataFrame(imputer.fit_transform(bin_cat_col),
                               index = bin_cat_col.index,
                               columns = bin_cat_col.columns)


    #for numeric data we use the mean value
    data_train=data_train.drop(columns=bin_cat_col)
    imputer = SimpleImputer(missing_values=-1,strategy='mean')
    data_train = pd.DataFrame(imputer.fit_transform(data_train),
                              index = data_train.index,
                              columns = data_train.columns)

    data_train=pd.concat([data_train,bin_cat_col],axis=1)
    
    return data_train


### transform categorical data to binary data ###

def transformCategoricalData(data_train):
    cat_columnheads=list()
    for head in data_train.columns:
        if 'cat' in head:
            cat_columnheads=np.append(cat_columnheads,head)
        
    data_train=pd.get_dummies(data_train, columns= cat_columnheads)
    
    return data_train


def removeHighCorrelatedFeatures(data_train):
    # Create correlation matrix
    corr_matrix = data_train.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    
    return to_drop
### remove weak correlated features to target ###


def removeWeakCorrelatedFeaturesToTarget(data_train):
    corr_matrix = data_train.corr()
    to_drop=np.squeeze(np.where(corr_matrix["target"]<0.01))
    indexes=list()
    for i in range (to_drop.shape[0]):
        feature=data_train.columns[to_drop[i]]
        if feature.find('cat')==-1 and feature.find('bin')==-1:
            indexes.append(i)
    to_drop=np.delete(to_drop,indexes)
    
    return to_drop

data_train=replaceMissingValues(data_train)
data_train=transformCategoricalData(data_train)

high_correlated=removeHighCorrelatedFeatures(data_train)
data_train=data_train.drop(columns=high_correlated)

#weak_correlated_target=removeWeakCorrelatedFeaturesToTarget(data_train)
#data_train=data_train.drop(columns=data_train.columns[weak_correlated_target],axis=1)
#print(np.asarray(data_train).shape)
### set X_train and Y_train ###

Y_train = np.asarray(data_train['target'])
fea_col = data_train.columns[2:]
X_train = np.asarray(data_train[fea_col])
### scaling | dimensionality reduction

X_train=scale(X_train)
#X_train=pca(X_train)

selector = SelectKBest(f_classif, k=35)
selector = selector.fit(X_train, Y_train)
X_train = selector.transform(X_train)

#lda = LinearDiscriminantAnalysis(n_components=1)
#X_train = lda.fit_transform(X_train, Y_train)
#X_train=clf.transform(X_train)
### build classifier ###



clf = RandomForestClassifier(n_estimators= 80, min_samples_split= 15,min_samples_leaf= 16, 
                             max_features= 'auto', max_depth= 55,class_weight={0:0.52,1:0.48},
                            min_impurity_decrease=0.0001)

#cl = RandomForestClassifier( min_impurity_decrease=0.0001)

#random_grid = {#'class_weight':[{0: w} for w in [0.2, 0.4, 0.6,0.8]]}
#                'randomforestclassifier__max_depth': [6,10, 20, 40, 55,70, None],
#               'randomforestclassifier__max_features': ['auto','log2'],
#               'randomforestclassifier__min_samples_leaf': [1,2, 4, 6, 8,16,24,None],
#               'randomforestclassifier__min_samples_split': [2, 5, 10, 15,20,30,None],
#               'randomforestclassifier__n_estimators': [40,60,80,100,None]}

#cc = RandomUnderSampler()
#x_train, y_train = cc.fit_resample(X_train, Y_train)

#smote = SMOTE()
#x_train, y_train = oversample.fit_resample(x_train, y_train)

#pipeline = make_pipeline(smote, cl)

#clf_random = RandomizedSearchCV(estimator=pipeline, param_distributions = random_grid, n_iter = 80, cv = 3, verbose=2, n_jobs = 6)
#Fit the random search model
#clf_random.fit(X_train, Y_train)

#print(clf_random.best_params_)
#print(clf_random.best_score_)
#print(clf_random.best_estimator_)
### train/test classifier using KFold strategy ###

kf = KFold(n_splits=5,shuffle = True)
for train_index, test_index in kf.split(X_train):
    x_train, x_test = X_train[train_index], X_train[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]

    print(x_test.shape)
    print(y_test.shape)

    #cc = RandomUnderSampler()
    #x_train, y_train = cc.fit_resample(x_train, y_train)
    
    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(x_train, y_train)

    clf = clf.fit(x_train, y_train)
    
  
    y_pred = clf.predict(x_train)
    
    print(classification_report(y_train, y_pred))
    print(confusion_matrix(y_train, y_pred))
    
    y_pred = clf.predict(x_test)
            
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

#for testing on test_set.csv

filename = "test_set.csv"
data_test = pd.read_csv(filename)
data_test=data_test.fillna(value=-1)

#remove features with over 40% missing values
data_test=data_test.drop(columns=to_drop_missing)
                         
data_test=replaceMissingValues(data_test)
data_test=transformCategoricalData(data_test)

fea_col = data_train.columns[2:]
data_test2=data_test[fea_col]

X_test = np.asarray(data_test2)


X_test=scale(X_test)
X_test = selector.transform(X_test)


oversample = SMOTE()
X_train, Y_train = oversample.fit_resample(X_train, Y_train)

#cc = RandomUnderSampler()
#X_train, Y_train = cc.fit_resample(X_train, Y_train)

clf = clf.fit(X_train, Y_train)
    

print("\n")
y_pred = clf.predict(X_train)
print(classification_report(Y_train, y_pred))
    
y_pred = clf.predict(X_test)


data_out = pd.DataFrame(data_test['id'].copy().astype('int32'))
data_out.insert(1, "target", y_pred.astype(int), True) 
data_out.to_csv('submission.csv',index=False)

#got a bad score for testing so we predict more 0 with changing the probability threshold
decisions = clf.predict_proba(X_test)
y_pred=list()
for dec in decisions:
    if dec[0]>=0.318: 
        y_pred.append(0)
    else:
        y_pred.append(1)
y_pred=[int(target) for target in y_pred]
data_out = pd.DataFrame(data_test['id'].copy().astype('int32'))
data_out.insert(1, "target", y_pred, True) 
data_out.to_csv('submission.csv',index=False)