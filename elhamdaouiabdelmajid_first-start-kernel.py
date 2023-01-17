!pip install sklearn_evaluation

!pip install xgboost
!pip install lightgbm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel



import matplotlib.pyplot as plt # plotting

from sklearn_evaluation import plot

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split,GridSearchCV



from sklearn import preprocessing

from sklearn.externals import joblib





%matplotlib inline

print(os.listdir('../input/kdd-cup-1999-data'))
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 70]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (8 * nGraphPerRow, 10 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth, dataframeName):

    filename = dataframeName#df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

def list_files(startpath):

    for root, dirs, files in os.walk(startpath):

        level = root.replace(startpath, '').count(os.sep)

        indent = ' ' * 4 * (level)

        print('{}{}/'.format(indent, os.path.basename(root)))

        subindent = ' ' * 4 * (level + 1)

        for f in files:

            print('{}{}'.format(subindent, f))

list_files('../input/')

    
# with open("../input/kddcup.names", 'r') as f:

#     print(f.read())

cols = """

    duration,

protocol_type,

service,

flag,

src_bytes,

dst_bytes,

land,

wrong_fragment,

urgent,

hot,

num_failed_logins,

logged_in,

num_compromised,

root_shell,

su_attempted,

num_root,

num_file_creations,

num_shells,

num_access_files,

num_outbound_cmds,

is_host_login,

is_guest_login,

count,

srv_count,

serror_rate,

srv_serror_rate,

rerror_rate,

srv_rerror_rate,

same_srv_rate,

diff_srv_rate,

srv_diff_host_rate,

dst_host_count,

dst_host_srv_count,

dst_host_same_srv_rate,

dst_host_diff_srv_rate,

dst_host_same_src_port_rate,

dst_host_srv_diff_host_rate,

dst_host_serror_rate,

dst_host_srv_serror_rate,

dst_host_rerror_rate,

dst_host_srv_rerror_rate"""

cols = [c.strip() for c in cols.split(",") if c.strip()]

cols.append('target')

print(len(cols))
attacks_type = {

'normal': 'normal',

'back': 'dos',

'buffer_overflow': 'u2r',

'ftp_write': 'r2l',

'guess_passwd': 'r2l',

'imap': 'r2l',

'ipsweep': 'probe',

'land': 'dos',

'loadmodule': 'u2r',

'multihop': 'r2l',

'neptune': 'dos',

'nmap': 'probe',

'perl': 'u2r',

'phf': 'r2l',

'pod': 'dos',

'portsweep': 'probe',

'rootkit': 'u2r',

'satan': 'probe',

'smurf': 'dos',

'spy': 'r2l',

'teardrop': 'dos',

'warezclient': 'r2l',

'warezmaster': 'r2l',

    }
df = pd.read_csv("../input/kdd-cup-1999-data/kddcup.data_10_percent/kddcup.data_10_percent", names=cols)

df['Attack'] = df.target.apply(lambda r: attacks_type[r[:-1]])

print("The data shape is (lines, columns):",df.shape)

df.head(5)
hajar_to_cup = {

    'is_hot_login' : 'is_host_login',

'urg' : 'urgent',

'protocol' : 'protocol_type',

'count_sec' : 'count',

'srv_count_sec' : 'srv_count',

'serror_rate_sec' : 'serror_rate',

'srv_serror_rate_sec' : 'srv_serror_rate',

'rerror_rate_sec' : 'rerror_rate',

'srv_error_rate_sec' : 'srv_rerror_rate',

'same_srv_rate_sec' : 'same_srv_rate',

'diff_srv_rate_sec' : 'diff_srv_rate',

'srv_diff_host_rate_sec' : 'srv_diff_host_rate',

'count_100' : 'dst_host_count',

'srv_count_100' : 'dst_host_srv_count',

'same_srv_rate_100' : 'dst_host_same_srv_rate',

'diff_srv_rate_100' : 'dst_host_diff_srv_rate',

'same_src_port_rate_100' : 'dst_host_same_src_port_rate',

'srv_diff_host_rate_100' : 'dst_host_srv_diff_host_rate',

'serror_rate_100' : 'dst_host_serror_rate',

'srv_serror_rate_100' : 'dst_host_srv_serror_rate',

'rerror_rate_100' : 'dst_host_rerror_rate',

'srv_rerror_rate_100' : 'dst_host_srv_rerror_rate',

}

for k,v in hajar_to_cup.items():

    print(k,v)
df.Attack.value_counts()
df.target.unique(), df.Attack.unique()
print(df.shape)

df.Attack.value_counts()


plotPerColumnDistribution(df[[

    'protocol_type',

    'service',

    'flag',

    'logged_in',

    'srv_serror_rate',

    'srv_diff_host_rate',

]], nGraphShown=30, nGraphPerRow=2)
plotPerColumnDistribution(df[['target']], nGraphShown=20, nGraphPerRow=4)
plotPerColumnDistribution(df[['Attack']], nGraphShown=20, nGraphPerRow=4)
plotCorrelationMatrix(df, graphWidth=20, dataframeName="Packets")
#plotScatterMatrix(df, plotSize=10, textSize=2)
for c in df.columns:

    print("%30s : %d"%(c, sum(pd.isnull(df[c]))))


# for c in X_train.columns:

#     if str(X_train[c].dtype) == 'object':

#         print(c, "::", X_train[c].dtype, X_train[c].value_counts())

#         print(c, "::", X_test[c].dtype, X_test[c].value_counts())

#         print("=======")



# le_X = preprocessing.LabelEncoder()

# le_y = preprocessing.LabelEncoder()



# for c in X_train.columns:

#     if str(X_train[c].dtype) == 'object': 

#         X_train[c] = le_X.fit_transform(X_train[c])

#         X_test[c] = le_X.transform(X_test[c])

    



# y_train = le_y.fit_transform(y_train.values)

# y_test = le_y.fit_transform(y_test.values)
df_std = df.std() # STD of all features

df_std = df_std.sort_values(ascending=True) # Std sorted for all features

df_std
plt.figure(figsize=(15,10))

plt.plot(list(df_std.index) ,list(df_std.values), 'go')



plt.show()
# To do well we can plot the STD without the greath values (without the feature have std > 3.0)

plt.figure(figsize=(15,10))

plt.plot(list(df_std.index)[:-7] ,list(df_std.values)[:-7] , 'go')



plt.show()
#============== MAPPING THE COLUMNS 

def standardize_columns(df, cols_map=hajar_to_cup):

    """

    1- Delete the 'service' column.

    2- Verify if TCPDUMP columns exists, then they will renamed

    """

    df = df.drop(['service'], axis = 1)

    df.rename(columns = cols_map)

    return df



df = standardize_columns(df, cols_map=hajar_to_cup)
df = df.drop(['target',], axis=1)

print(df.shape)

# Target variable and train set

y = df.Attack

X = df.drop(['Attack',], axis=1)

# Split test and train data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape, X_test.shape)

print(y_train.shape, y_test.shape)
# for c in X_train.columns:

#     if str(X_train[c].dtype) == 'object':

#         print(c, "::", X_train[c].dtype, X_train[c].value_counts())

#         print(c, "::", X_test[c].dtype, X_test[c].value_counts())

#         print("=======")



        
le_X_cols = {}

le_y = preprocessing.LabelEncoder()



for c in X_train.columns:

    if str(X_train[c].dtype) == 'object': 

        le_X = preprocessing.LabelEncoder()

        X_train[c] = le_X.fit_transform(X_train[c])

        X_test[c] = le_X.transform(X_test[c])

        le_X_cols[c] = le_X

#------

y_train = le_y.fit_transform(y_train.values)

y_test = le_y.transform(y_test.values)



# save the labelers for depploy

joblib.dump(le_X_cols, 'le_X_cols.pkl') 

joblib.dump(le_y, 'le_y.pkl') 
class_names, class_index = le_y.classes_, np.unique(y_train)

class_names, class_index
# Feature Scaling

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

X_train[['dst_bytes','src_bytes']] = scaler.fit_transform(X_train[['dst_bytes','src_bytes']])

X_test[['dst_bytes','src_bytes']] = scaler.transform(X_test[['dst_bytes','src_bytes']])

#== save the scaler for deploy it

joblib.dump(scaler, 'scaler_1.pkl') 

X_train[['dst_bytes','src_bytes']].head(5)
from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42)

classifier.fit(X_train, y_train)

print("Train score is:", classifier.score(X_train, y_train))

print("Test score id:",classifier.score(X_test,y_test))# New data, not included in Training data

diff_base = abs(classifier.score(X_train, y_train) - classifier.score(X_test,y_test))

print("We see that same over/under fitting for this model, ", diff_base)
# Predicting the Test set results

y_pred = classifier.predict(X_test)

#

reversefactor = dict(zip(class_index,class_names))

y_test_rev = np.vectorize(reversefactor.get)(y_test)

y_pred_rev = np.vectorize(reversefactor.get)(y_pred)

# Making the Confusion Matrix

print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))







fig, ax = plt.subplots(figsize=(15, 10))

plot.confusion_matrix(y_test_rev, y_pred_rev, ax=ax)

plt.show()
clf = RandomForestClassifier(n_estimators=30)

clf = clf.fit(X_train, y_train)

fti = clf.feature_importances_

model = SelectFromModel(clf, prefit=True, threshold= 0.005)

X_train_new = model.transform(X_train)

X_test_new = model.transform(X_test)

selcted_features = X_train.columns[model.get_support()]

print(X_train_new.shape)
selcted_features
parameters = {

    'n_estimators'      : [20,40,128,130],

    'max_depth'         : [None,14, 15, 17],

    'criterion' :['gini','entropy'],

    'random_state'      : [42],

    #'max_features': ['auto'],

    

}

clf = GridSearchCV(RandomForestClassifier(), parameters, cv=2, n_jobs=-1, verbose=5)

clf.fit(X_train_new, y_train)
print("clf.best_estimator_:",clf.best_estimator_)

print("clf.best_params_",clf.best_params_)

print("results:")

#print(clf.cv_results_)
print("CV Train score,",clf.best_score_)

print("CV Test score,",clf.score(X_test_new,y_test))

diff_fst = abs(clf.best_score_ - clf.score(X_test_new,y_test))

print("Diff", diff_fst)

print("Diff feature selection is best than base model ? ", diff_base > diff_fst)

# Predicting the Test set results

y_pred = clf.predict(X_test_new)

#

reversefactor = dict(zip(class_index,class_names))

y_test_rev = np.vectorize(reversefactor.get)(y_test)

y_pred_rev = np.vectorize(reversefactor.get)(y_pred)

# Making the Confusion Matrix

print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))







fig, ax = plt.subplots(figsize=(15, 10))

plot.confusion_matrix(y_test_rev, y_pred_rev, ax=ax)

plt.show()
fig, ax = plt.subplots(figsize=(15, 10))

plot.feature_importances(clf.best_estimator_, top_n=5, ax=ax, feature_names=list(selcted_features))

plt.show()
joblib.dump(clf, 'random_forest_classifier.pkl') 

#To load it: clf_load = joblib.load('saved_model.pkl') 
from sklearn.multiclass import OneVsRestClassifier

from xgboost import XGBClassifier

from sklearn.preprocessing import MultiLabelBinarizer
clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4, n_estimators=70, random_state=42,verbosity=1))



# You may need to use MultiLabelBinarizer to encode your variables from arrays [[x, y, z]] to a multilabel 

# format before training.

lb = preprocessing.LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)

lb.fit(y_train)

y_train_xgb = lb.transform(y_train)

y_test_xgb = lb.transform(y_test)



clf.fit(X_train[selcted_features], y_train_xgb)

y_pred_xgb = clf.predict(X_test[selcted_features])



print("Train score is:", clf.score(X_train[selcted_features], y_train_xgb))

print("Test score id:",clf.score(X_test[selcted_features],y_test_xgb))# New data, not included in Training data

diff_xgb = abs(clf.score(X_train[selcted_features], y_train_xgb) - clf.score(X_test[selcted_features],y_test_xgb))

print("The diff, ", diff_xgb)

y_pred_xgb = np.argmax(y_pred_xgb, axis=1)


#

reversefactor = dict(zip(class_index,class_names))

y_test_rev = np.vectorize(reversefactor.get)(y_test)

y_pred_rev = np.vectorize(reversefactor.get)(y_pred_xgb)

# Making the Confusion Matrix

print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))







fig, ax = plt.subplots(figsize=(15, 10))

plot.confusion_matrix(y_test_rev, y_pred_rev, ax=ax)

plt.show()
import xgboost as xgb

print(X_train.shape)



xgb_model = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4, n_estimators=70, random_state=42,verbosity=1))



parameters = {'estimator__nthread':[4,], #when use hyperthread, xgboost may become slower

              'estimator__objective':['binary:logistic',],

              'estimator__learning_rate': [0.1,0.08], #so called `eta` value

              'estimator__max_depth': [4,6],

              'estimator__min_child_weight': [1,],

              'estimator__silent': [1,],

              'estimator__subsample': [1,],

              'estimator__colsample_bytree': [1,],

              'estimator__n_estimators': [70,100], #number of trees, change it to 1000 for better results

              'estimator__random_state':[42],

              }





clf = GridSearchCV(xgb_model, parameters, 

                   cv=2, n_jobs=-1, verbose=5, refit=True)



clf.fit(X_train[selcted_features], y_train_xgb)
print("CV Train score,",clf.best_score_)

print("Params", clf.best_params_)

print("CV Test score,",clf.score(X_test[selcted_features],y_test_xgb))

diff_fst = abs(clf.best_score_ - clf.score(X_test[selcted_features],y_test_xgb))

print("Diff", diff_fst)



print("Diff feature selection is best than xgb base model ? ", diff_xgb > diff_fst)

print("Diff feature selection is best than RDF best model ? ", diff_base > diff_fst)

#

y_pred_xgb = clf.predict(X_test[selcted_features])

y_pred_xgb = np.argmax(y_pred_xgb, axis=1)

reversefactor = dict(zip(class_index,class_names))

y_test_rev = np.vectorize(reversefactor.get)(y_test)

y_pred_rev = np.vectorize(reversefactor.get)(y_pred_xgb)

# Making the Confusion Matrix

print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))







fig, ax = plt.subplots(figsize=(15, 10))

plot.confusion_matrix(y_test_rev, y_pred_rev, ax=ax)

plt.show()
joblib.dump(clf, 'xgboost_classifier.pkl') 

#To load it: clf_load = joblib.load('saved_model.pkl') 
scaler_1 = joblib.load('scaler_1.pkl') 

le_X_cols = joblib.load('le_X_cols.pkl') 

le_y = joblib.load('le_y.pkl') 

xgb_clf = joblib.load('xgboost_classifier.pkl') 

rdf_clf = joblib.load('random_forest_classifier.pkl') 
def do_what_we_want(X, 

                    scaler_1, 

                    le_X_cols, 

                    selcted_features, 

                    map_cols,

                    rdf_clf,

                    xgb_clf):

    X[['dst_bytes','src_bytes']] = scalscaler_1er.fit_transform(X[['dst_bytes','src_bytes']])

    for c in X.columns:

        if str(X[c].dtype) == 'object': 

            le_X = le_X_cols[c]

            X[c] = le_X.transform(X[c])

            

    X = standardize_columns(X, cols_map=map_cols) # Rename the columns, and delete the 

    

    X = X[selcted_features]

    #====

    rd_prediction = rdf_clf.predict(X)

    xgb_prediction = xgb_clf.predict(X)

    

    return rd_prediction, xgb_prediction

    



    

