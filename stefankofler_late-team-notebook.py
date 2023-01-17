# Quick load dataset and check
import pandas as pd

pathTrainSet = "./templates/train_set.csv"
pathTestSet = "./templates/test_set.csv"
filename = pathTrainSet
data_train = pd.read_csv(filename)
filename = pathTrainSet
data_test = pd.read_csv(filename)
data_test.describe()
import numpy as np
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_regression

##Each function returns a pandadataframe for train- and testdata

## Function to impute the missing values with most_frequent for bins and cats and mean for else
def impute(data, testdata):
    else_col=data.columns[data.columns.str.contains(pat = '\d+$')]
    bin_col=data.columns[data.columns.str.contains(pat = 'bin')]
    cat_col=data.columns[data.columns.str.contains(pat = 'cat')]
    imputed_data = data.copy()
    imputed_testdata = testdata.copy()
    imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
    imp.fit(data[cat_col])
    imputed_data[cat_col] = imp.transform(data[cat_col])
    imputed_testdata[cat_col] = imp.transform(testdata[cat_col])
    imp.fit(data[bin_col])
    imputed_data[bin_col] = imp.transform(data[bin_col])
    imputed_testdata[bin_col] = imp.transform(testdata[bin_col])
    
    imptwo = SimpleImputer(missing_values=-1., strategy='mean')
    imptwo.fit(data[else_col])
    imputed_data[else_col] = imptwo.transform(data[else_col])
    imputed_testdata[else_col] = imptwo.transform(testdata[else_col])

    return imputed_data, imputed_testdata

## Function to encode the cat_columns
def encode_cat_columns (data_X, testdata_X, data_Y):
    traindata = data_X.copy()
    testdata = testdata_X.copy()
    cat_col=data_X.columns[data_X.columns.str.contains(pat = 'cat')]
    target_enc = ce.TargetEncoder(cols=cat_col)
    target_enc.fit(data_X[cat_col], data_Y)
    traindata[cat_col] = target_enc.transform(data_X[cat_col])
    testdata[cat_col] = target_enc.transform(testdata_X[cat_col])
    return traindata, testdata

### Function to remove highly correlated data
def removecorrelateddata(data, testdata):
    selected_data = data.copy()
    selected_testdata = testdata.copy()
    corr = selected_data.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.6:
                if columns[j]:
                    columns[j] = False
    selected_columns = selected_data.columns[columns]
    selected_data = selected_data[selected_columns]
    selected_testdata = selected_testdata[selected_columns]
    return selected_data, selected_testdata

##Oversample minority class
def oversampling(datax, datay):
    smote= SMOTE(sampling_strategy = 'auto')
    resampled_trainx, resampled_trainy = smote.fit_sample(datax, datay)
    return resampled_trainx, resampled_trainy
    
##Univariate selection of best datafeatures
def featureselection(datax, testdatax, datay):
    feature_cols = datax.columns[:]
    selector = SelectKBest(f_classif, k=32, score_func=chi2)
    data_new = selector.fit_transform(datax, datay)
    selected_features = pd.DataFrame(selector.inverse_transform(data_new), 
                                 index=datax.index, 
                                 columns=feature_cols)
    selected_columns = selected_features.columns[selected_features.var() != 0]
    return datax[selected_columns], testdatax[selected_columns]

def undersampling(datax, datay):
    undersample = RandomUnderSampler(sampling_strategy = 0.9)
    resampled_datax, resampled_trainy = undersample.fit_resample(datax, datay)
    return resampled_datax, resampled_trainy


def prepare_data_MLP(datax, testdatax, datay):
    datax, testdatax = impute(datax, testdatax)
    datax, testdatax = encode_cat_columns(datax, testdatax, datay)
    datax, testdatax = removecorrelateddata(datax, testdatax)
    datax, testdatax = featureselection(datax, testdatax, datay)
    datax, datay = oversampling(datax, datay)
    return datax, testdatax, datay
    
import numpy as np
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_regression
from imblearn.under_sampling import RandomUnderSampler
## Select target and features
fea_col = data_train.columns[2:]
data_Y = data_train['target']
data_X = data_train[fea_col]

data_test_X = data_test.drop(columns=['id'])

ind_col = data_X.columns[data_X.columns.str.contains(pat = 'ind')]
reg_col = data_X.columns[data_X.columns.str.contains(pat = 'reg')]
car_col = data_X.columns[data_X.columns.str.contains(pat = 'car')]
correlation_data = data_X[ind_col]
correlation_data[reg_col] = data_X[reg_col]
correlation_data[car_col] = data_X[car_col]

import seaborn as sns
corr = correlation_data.corr()
sns.heatmap(corr)
imputed_data, imputed_testdata = impute(data_X, data_test_X)

##-----------------------------------------
#Test ob noch Nullwerte
if -1 in imputed_data.to_numpy():
    print("Nullwerte enthalten")
if -1 in imputed_testdata.to_numpy():
    print("Nullwerte enthalten")
#--------------------------#----------------

encoded_traindata, encoded_testdata = encode_cat_columns(imputed_data, imputed_testdata, data_Y)

removed_correlated_traindata, removed_correlated_testdata = removecorrelateddata(encoded_traindata, encoded_testdata)

selected_traindata, selected_testdata = featureselection(removed_correlated_traindata, removed_correlated_testdata, data_Y)

selected_traindata

x_train, x_val, y_train, y_val = train_test_split(selected_traindata, data_Y, test_size = 0.3, shuffle = True)
x_oversampled, y_oversampled = oversampling(x_train, y_train)
x_new, y_new = undersampling(x_oversampled, y_oversampled)

clf = RandomForestClassifier(max_depth=7, random_state=0, class_weight='balanced', n_estimators=300)
clf = clf.fit(x_new, y_new)
y_pred = clf.predict(x_val)

print("Percentage der Treffer", sum(y_pred==y_val)/len(y_val))
##-----------------------------------


def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)
print("Percentage der Treffer des 1-Labels", sum(y_pospred==y_pos)/len(y_pos))

X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
print("Percentage der Treffer des 0-Labels", sum(y_negpred==y_neg)/len(y_neg))
##--------------------------------------------------------------------------------------
    

print("truenegative",sum(y_negpred==0))
print("truepositive",sum(y_pospred==1))
print("falsepostive",sum(y_pospred==0))
print("falsenegative",sum(y_negpred==1))



print(metrics.f1_score(y_val, y_pred, average='macro'))

y_target = clf.predict(selected_testdata)
print("negative outputs: ", sum(y_target==0))
print("positive outputs: ", sum(y_target==1))
data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)
from imblearn.over_sampling import SMOTE

smote= SMOTE(sampling_strategy='auto')

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

x_new, y_new = smote.fit_sample(x_train, y_train)

print (x_new.describe())
print (y_new.describe())

clf = DecisionTreeClassifier()
clf = clf.fit(x_new, y_new)
y_pred = clf.predict(x_val)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

print("staring methode")

fea_col = data_train.columns[2:]
data_Y_train = data_train['target']
data_X_train = data_train[fea_col]
data_X_test = data_test.drop(columns=['id'])

##---------- prepare data
data_X_train, data_X_test, data_Y_train = prepare_data_MLP(data_X_train, data_X_test, data_Y_train)

##---------- split into train and test
x_train, x_val, y_train, y_val = train_test_split(data_X_train, data_Y_train, test_size = 0.3, shuffle = True)

print("Started Training model, takes a while")

clf = MLPClassifier(verbose = True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)

print("Percentage der Treffer", sum(y_pred==y_val)/len(y_val))
##-----------------------------------


def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)
print("Percentage der Treffer des 1-Labels", sum(y_pospred==y_pos)/len(y_pos))

X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
print("Percentage der Treffer des 0-Labels", sum(y_negpred==y_neg)/len(y_neg))
y_target = clf.predict(data_X_test)

data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
# reduce data so there are not 96% positive anymore instead 50% positive
data_pos = data_train.loc[data_train['target'] == 1]
data_neg = data_train.loc[data_train['target'] == 0]
data_neg_reduced = data_neg.sample(len(data_pos))

#combine and shuffel those 2 sets
data = data_pos.append(data_neg_reduced)
data = shuffle(data)

##prepare for training
fea_col = data_train.columns[2:]

data_Y = data['target']
data_X = data[fea_col]

##train model
x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)
clf = DecisionTreeClassifier(min_impurity_decrease = 0.001, min_samples_split = 20)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)

print("overall accuracy", sum(y_pred==y_val)/len(y_val))

def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)
print("accuracy for positive values:", sum(y_pospred==y_pos)/len(y_pos))

X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_pospred = clf.predict(X_neg)
print("accuracy for negative values:", sum(y_pospred==y_neg)/len(y_neg))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

fea_col = data_train.columns[2:]
data_Y_train = data_train['target']
data_X_train = data_train[fea_col]
data_X_test = data_test.drop(columns=['id'])

##---------- prepare data
data_X_train, data_X_test, data_Y_train = prepare_data_MLP(data_X_train, data_X_test, data_Y_train)

x_train, x_val, y_train, y_val = train_test_split(data_X_train, data_Y_train, test_size = 0.3, shuffle = True)

clf = RandomForestClassifier(n_estimators = 19, min_impurity_decrease = 0.001, class_weight="balanced")
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)

print("Percentage der Treffer", sum(y_pred==y_val)/len(y_val))
##-----------------------------------


def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)
print("Percentage der Treffer des 1-Labels", sum(y_pospred==y_pos)/len(y_pos))

X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
print("Percentage der Treffer des 0-Labels", sum(y_negpred==y_neg)/len(y_neg))
##--------------------------------------------------------------------
y_target = clf.predict(data_X_test)

data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

## Select target and features
fea_col = data_train.columns[2:]
data_Y = data_train['target']
data_X = data_train[fea_col]


##-----------------------------------------------------------------------------------

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

##------------------------------------------------------------------------------------
##  Gewichtung der Daten

## imbalanced data werden mit class_weight=balanced gewichtet
clf = DecisionTreeClassifier(min_impurity_decrease = 0.001, class_weight="balanced")
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)

print("Percentage der Treffer", sum(y_pred==y_val)/len(y_val))
##-----------------------------------


def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)
print("Percentage der Treffer des 1-Labels", sum(y_pospred==y_pos)/len(y_pos))

X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
print("Percentage der Treffer des 0-Labels", sum(y_negpred==y_neg)/len(y_neg))
##--------------------------------------------------------------------------------------
    

print("negative",sum(y_negpred==0))
print("positive",sum(y_pospred==1))
print("falsenegative",sum(y_pospred==0))
print("falsepositive",sum(y_negpred==1))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

## Oversampling (can be replaced with smote etc./could not get it to work on my machine)
data_pos = data_train.loc[data_train['target'] == 1]
data_neg = data_train.loc[data_train['target'] == 0]
data_pos_oversampled = data_pos.sample(len(data_neg), replace=True)

data = data_pos_oversampled.append(data_neg)
data = shuffle(data)

## Select target and features
fea_col = data_train.columns[2:]
data_Y = data['target']
data_X = data[fea_col]

##-----------------------------------------------------------------------------------


clf = LDA(n_components=None, priors=None, shrinkage='auto', solver='lsqr',
store_covariance=False, tol=0.0001)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)

print("Percentage der Treffer (oversampled)", sum(y_pred==y_val)/len(y_val))
##-----------------------------------


def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)
print("Percentage der Treffer des 1-Labels", sum(y_pospred==y_pos)/len(y_pos))

X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
print("Percentage der Treffer des 0-Labels", sum(y_negpred==y_neg)/len(y_neg))
##--------------------------------------------------------------------------------------
    

print("truenegative",sum(y_negpred==0))
print("truepositive",sum(y_pospred==1))
print("falsenegative",sum(y_pospred==0))
print("falsepositive",sum(y_negpred==1))
data_test_X = data_test.drop(columns=['id'])
y_target = clf.predict(data_test_X)
print("negative outputs: ", sum(y_target==0))
print("positive outputs: ", sum(y_target==1))
data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)
data_out