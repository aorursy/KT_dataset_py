import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import seaborn as sns
sns.set(rc={'figure.figsize':(22,22)})

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, mean_squared_error
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

import feather

from xgboost import XGBClassifier
train = pd.read_csv("../input/wec-ml-mentorship-contest/WEC_Contest_sample_submission.csv", low_memory=False)
train.head()
train = pd.read_csv("../input/wec-ml-mentorship-contest/WEC_Contest_train.csv", low_memory=False)
label = train["Label"]
trainRows = train.shape[0]

test = pd.read_csv("../input/wec-ml-mentorship-contest/WEC_Contest_test.csv", low_memory=False)

sample = pd.read_csv("../input/wec-ml-mentorship-contest/WEC_Contest_sample_submission.csv", low_memory=False)

test.to_feather("test")
train.to_feather("train")
test.shape
sample.shape
train.head()
train.info()
colInfo = {"I"       :"continuous",
"II"      :"categorical",
"III"     :"categorical",
"IV"      :"categorical",
"V"       :"continuous",
"VI"      :"continuous",
"VII"     :"continuous",
"VIII"    :"continuous",
"IX"      :"continuous",
"X"       :"continuous",
"XI"      :"boolean",
"XII"     :"categorical",
"XIII"    :"continuous",
"XIV"     :"boolean",
"XV"      :"boolean",
"XVI"     :"categorical",
"XVII"    :"continuous",
"XVIII"   :"continuous",
"XIX"     :"continuous",
"XX"      :"continuous"} 
def feature_types():
    '''
    Creates lists for boolean, categorical, continuous features.
    Prints and returns these lists.
    '''
    boolean = []
    categorical = []
    continuous = []

    for col in colInfo:
        if (colInfo[col] == "boolean"):
            boolean.append(col)
        elif (colInfo[col] == "categorical"):
            categorical.append(col)
        else:
            continuous.append(col)
    
    print("boolean: {}".format(boolean))
    print("categorical: {}".format(categorical))
    print("continuous: {}".format(continuous))

    return boolean, categorical, continuous
boolean, categorical, continuous = feature_types()        
train.shape
test.shape
def null_columns(data):
    '''
    returns a list of columns with NaN values
    
    Arguements
    data: the dataframe that you want to check
    
    Returns:
    list of columns containing null values
    '''
    l = [col for col in data.columns if data[col].isnull().sum() > 0]
    
    if not l:
        print("No NaN columns!")

    return l
def plot_dist(col, train, test):
    '''
    Plots the distribution of train and test set.
    With the col values sorted in ascending order
    
    Arguements:
    col: takes the column name of the DataFrame
    train: train DataFrame
    test: test DataFrame
    
    Returns:
    nothing
    '''
    _, axes = plt.subplots(2, 1)

    plot1 = sns.countplot(x=col, data=train, ax=axes[0])
    plot1.set_title("Traning Set")
    plot1.set_xticklabels(plot1.get_xticklabels(), rotation=60, ha="right")

    plot2 = sns.countplot(x=col, data=test, ax=axes[1])
    plot2.set_title("Test Set")
    plot2.set_xticklabels(plot1.get_xticklabels(), rotation=60, ha="right")

    plt.tight_layout()
    plt.show()
    print("\n\n\n")
    
def plot_line(col, train, test):
    '''
    Plots the distribution of train and test set.
    With the col values sorted in ascending order
    
    Arguements:
    col: takes the column name of the DataFrame
    train: train DataFrame
    test: test DataFrame
    
    Returns:
    nothing
    '''
    _, axes = plt.subplots(2, 1)

#     plot1 = sns.lineplot(x=col, data=train, ax=axes[0])
#     plot1.set_title("Traning Set")
#     plot1.set_xticklabels(plot1.get_xticklabels(), rotation=60, ha="right")

#     plot2 = sns.lineplot(x=col, data=test, ax=axes[1])
#     plot2.set_title("Test Set")
#     plot2.set_xticklabels(plot1.get_xticklabels(), rotation=60, ha="right")


    plot1 = train[col].plot.hist(ax=axes[0])
    plot1.set_title("Traning Set")
    plot1.set_xticklabels(plot1.get_xticklabels(), rotation=60, ha="right")

    plot2 = test[col].plot.hist(ax=axes[1])
    plot2.set_title("Test Set")
    plot2.set_xticklabels(plot1.get_xticklabels(), rotation=60, ha="right")
    
    plt.tight_layout()
    plt.show()
    print("\n\n\n")
def split_null(data, col):
    '''
    splits data into null and non-null dataframes and col based on the col
    
    Arguements:
        data: a pd.DataFrame
        col: a pd.Series that has NaN values
        
    Return:
        returns a nonNullData, nullData, nonNullCol, nullCol
    
    '''
    nonNullData = data.loc[~col.isnull()]
    nonNullCol = col[~col.isnull()]
    
    nullData = data.loc[col.isnull()]
    nullCol = col[col.isnull()]
    
    return nonNullData, nullData, nonNullCol, nullCol
def impute_nans(col, colNull):
    '''
    Returns a column with NaN values removed
    
    Arguments:
        col: pd.Series whose NaN values you want to impute
        colNull: a pd.Series whose indices corresspond to indices of NaN values
                    and values are the impute values
                    
    Returns:
        col: a pd.Series with all the NaN values imputed
    '''
    for i in range(colNull.shape[0]):
        col[colNull.index[i]] = colNull.values[i]
        print("col[colNull.index[i]]: {}".format(col[colNull.index[i]]))
    return col
def corr_matrix(data, threshold):
    '''
    Displays the correlation for DatFrame data. Prints and returns the correlations of features above threshold
    
    Arguements:
        data: the DataFrame
        threshold: the correlation thershold value
        
    Returns:
        corrList: contains a list of indices and correlation value for features above the correlation threshold
    '''
    corr = train.corr()
    
    numberOfCols = len(data.columns)
    corrList = [ [i, j, corr.iloc[i,j]] for i in range(0, numberOfCols -1) for j in range(i+1, numberOfCols) if (abs(corr.iloc[i,j]) >= threshold)]
    
    corrList.sort(key=lambda x: x[2], reverse=True)
    for c in corrList:
        print("{}\tand\t{}:\t{:0.3f}".format(c[0], c[1], c[2]))
        
    corr.index = train.columns
    sns.heatmap(corr, annot = True, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap", fontsize=6)
    plt.show()
    
    
    return corrList
    
def plot_pair(data, corrList):
    '''
    Plots seaborn pairplots for a given corrList
    
    Arguements:
        data: a DataFrame
        corrList: a corrList from corr_matirx()
        
    Returns:
        nothing
    '''
    cols = data.columns
    for i, j, _ in corrList:
        sns.pairplot(data, size=6, x_vars=cols[i],y_vars=cols[j] )
        plt.show()
def score_stats(scores):
    print("scores: {}".format(scores*100))
    print("scores.mean(): {}".format(scores.mean()*100))
    print("scores.std(): {}".format(scores.std()))
    print("scores.median(): {}".format(np.median(scores)*100)) 
    print()
def label_list(npArray):
    labelList = []
    for arr in npArray:
        labelList.append(str(str(arr[0])+" "+str(arr[1])+" "+str(arr[2])))

    return labelList
def write_to_file(args, fileName):
    submission = pd.DataFrame(data={"Id":sample["Id"].values, "Expected": label_list(args)} )
    submission.to_csv(fileName, index=False)
null_columns(train)
null_columns(test)
print("VIII: {}".format(colInfo["VIII"]))
print("XVI: {}".format(colInfo["XVI"]))
plot_dist("VIII", train, test)
plot_dist("XVI", train, test)
train["Label"].value_counts().sort_values()
plot1 = sns.countplot(x="Label", data=train)
plot1.set_title("Traning Set")
plot1.set_xticklabels(plot1.get_xticklabels(), rotation=60, ha="right")


plt.tight_layout()
plt.show()
for col in continuous:
    plot_dist(col, train, test)
colInfo["XX"] = colInfo["XIII"] = colInfo["XVIII"] = colInfo["X"] = "categorical"

# to change the feature types for one hot encoding
boolean, categorical, continuous = feature_types()
train[continuous].describe()
train[continuous] = np.log1p(train[continuous])
test[continuous] = np.log1p(test[continuous])
train.to_feather("train")
test.to_feather("test")
train[continuous].describe()
train.head()
for col in continuous:
    plot_line(col, train, test)
for col in categorical:
    plot_dist(col, train, test)
for col in boolean:
    plot_dist(col, train=train, test=test)
train = feather.read_dataframe("train")
test = feather.read_dataframe("test")
combined = pd.concat([train, test], ignore_index=True)

col8 = combined["VIII"]
col16 = combined["XVI"]
pd.Series(col16).rename(columns={"0":"XVI"}, inplace=True)
combined.drop(columns=["Label", "VIII", "XVI"], inplace=True)

combined.to_feather("combined")

try:
    categorical.remove("XVI")
except:
    pass

combined = pd.get_dummies(combined, columns=categorical)

scaler = StandardScaler()
combined[combined.columns]= scaler.fit_transform(combined)
# df[df.columns] = scaler.fit_transform(df[df.columns])

display(combined.head())
print("combined.shape: {}".format(combined.shape))
train["VIII"].describe()
test["VIII"].describe()
nonNullCombined16, nullCombined16, nonNullCol16, nullCol16 = split_null(combined, col16)
skf = StratifiedKFold(n_splits=10, random_state=42)
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
scores = cross_val_score(knn, nonNullCombined16, nonNullCol16, scoring="f1_micro", cv=skf.split(nonNullCombined16, nonNullCol16))

score_stats(scores)

knn.fit(nonNullCombined16, nonNullCol16)

col16Pred = knn.predict(nullCombined16)
nullCol16 = pd.concat([nullCol16, pd.Series(col16Pred, index=nullCol16.index, name="temp")], axis=1, ignore_index=False)
nullCol16.drop(columns=["XVI"], inplace=True)
nullCol16.rename(columns={"temp":"XVI"}, inplace=True)

col16 = impute_nans(col16, nullCol16)

combined = feather.read_dataframe("combined")
combined16 = pd.concat([combined, col16], axis=1)
display(combined16.head())

combined16.to_feather("combined")
categorical.append("XVI")
categorical = list(set(categorical))
combined16 = pd.get_dummies(combined16, columns=categorical)

scaler = StandardScaler()
combined16[combined16.columns]= scaler.fit_transform(combined16)

combined16.head()
nonNullCombined8, nullCombined8, nonNullCol8, nullCol8 = split_null(combined, col8)
skf = KFold(n_splits=10, random_state=42)
knnReg = KNeighborsRegressor(n_neighbors=1, n_jobs=-1)
scores = cross_val_score(knnReg, nonNullCombined8, nonNullCol8, scoring="neg_mean_squared_error", cv=skf.split(nonNullCombined8, nonNullCol8))

score_stats(scores)

knnReg.fit(nonNullCombined8, nonNullCol8)

col8Pred = knnReg.predict(nullCombined8)
nullCol8 = pd.concat([nullCol8, pd.Series(col8Pred, index=nullCol8.index, name="temp")], axis=1, ignore_index=False)
nullCol8.drop(columns=["VIII"], inplace=True)
nullCol8.rename(columns={"temp":"VIII"}, inplace=True)

col8 = impute_nans(col8, nullCol8)

combined = feather.read_dataframe("combined")
combined8 = pd.concat([combined, col8], axis=1)
combined8.head()
colNames = combined8.columns.tolist()
del colNames[len(colNames) - 1]

colNames.append("VIII")
colNames.append("XXI")
colNames.append("XXII")
print(colNames)
len(colNames)
train = feather.read_dataframe("train")
test = feather.read_dataframe("test")
combined = pd.concat([train, test],ignore_index=True)

col8 = combined["VIII"].isnull() * 1
col16 = combined["XVI"].isnull() * 1
combined8 = pd.concat([combined8, col8.rename("VIII_nan"), col16.rename("XVI_nan")], axis=1,ignore_index=True)
combined8.head()
combined8.shape
combined8.columns = colNames
combined8.to_feather("combinedFinal")
combined8 = pd.get_dummies(data=combined8, columns=categorical)
trainNew = combined8.iloc[:trainRows, :]
testNew = combined8.iloc[trainRows:, :]

trainNew.reset_index(inplace=True, drop=True)
testNew.reset_index(inplace=True, drop=True)

trainNew.to_feather("trainNew")
testNew.to_feather("testNew")
print(trainNew.shape, testNew.shape)
trainNew[trainNew.columns] = scaler.fit_transform(trainNew[trainNew.columns])
testNew[testNew.columns] = scaler.transform(testNew[testNew.columns])
trainNew.head()
%%time
skf = StratifiedKFold(n_splits=10, random_state=42)
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, weights="uniform")
scores = cross_val_score(knn, trainNew, label, scoring="f1_macro", cv=skf.split(trainNew, label), n_jobs=-1)

score_stats(scores)

knn.fit(trainNew, label)
knnPred = knn.predict_proba(testNew)
# print(np.argsort(-knnPred, axis=1)[0][:3])
knnArgs = np.argsort(-knnPred, axis=1)[:, :3]

# 0.19
%%time
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, oob_score=True
                            , class_weight="balanced", min_samples_split=2)
scores = cross_val_score(rf, trainNew, label, scoring="f1_macro", cv=skf.split(trainNew, label), n_jobs=-1)

score_stats(scores)

rf.fit(trainNew, label)

print(rf.oob_score_)

rfPred = rf.predict_proba(testNew)
# print(np.argsort(-rfPred, axis=1)[0][:3])
rfArgs = np.argsort(-rfPred, axis=1)[:, :3]
%%time
xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=1, n_jobs=2, random_state=42)
scores = cross_val_score(xgb, trainNew, label, scoring="f1_macro", cv=skf.split(trainNew, label), n_jobs=-1)

score_stats(scores)

xgb.fit(trainNew, label)

xgbPred = xgb.predict_proba(testNew)
# print(np.argsort(-xgbPred, axis=1)[0][:3])
xgbArgs = np.argsort(-xgbPred, axis=1)[:, :3]
%%time
logit = LogisticRegression(max_iter=1000, C=1)
scores = cross_val_score(logit, trainNew, label, scoring="f1_macro", cv=skf.split(trainNew, label), n_jobs=-1)

score_stats(scores)

logit.fit(trainNew, label)

logitPred = logit.predict_proba(testNew)
# print(np.argsort(-logitPred, axis=1)[0][:3])
logitArgs = np.argsort(-logitPred, axis=1)[:, :3]
knnArgs
logitArgs
write_to_file(knnArgs, "knn.csv")