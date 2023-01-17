# Quick load dataset and check
import pandas as pd
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas.testing as tm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sklearn.naive_bayes as nb
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import math
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
filename = "train_set.csv"
data_train = pd.read_csv(filename)
filename = "test_set.csv"
data_test = pd.read_csv(filename)

data_train.describe()
## Column
def get_all_elements(data, noprint=False):
    cols = data.columns[1:]
    thisdict = dict(zip(cols,list(map(lambda x: list(set(x)),\
                                      data.values[:,1:].T))))
    
    if noprint:
        return thisdict
    np.set_printoptions(threshold=20)
    np.set_printoptions(threshold=None)
    return thisdict
def find_missing(data, dict_):
    missing_features = [ele[0] for ele in list(dict_.items()) if -1 in ele[1]]
    [missing_features.append(ele[0]) for ele in list(dict_.items()) if np.nan in ele[1]]
    return missing_features
def replace_missing(bad_features, data):
    feature_dict = {}
    for feature in bad_features:
        data, most_related = replace_missing_correlation(feature, data)
        feature_dict[feature] = most_related
        
    dict_train = get_all_elements(data)
    bad_features = find_missing(data, dict_train)
    imputer = SimpleImputer(missing_values=-1,strategy='median')
    data = pd.DataFrame(imputer.fit_transform(data),
                        index = data.index,
                        columns = data.columns)
    return data, feature_dict

def replace_missing_correlation(feature, data, related_feature=None):
    if related_feature == None:
        df_all_corr = data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
        df_all_corr.rename(columns={"level_0": "Feature 1", 
                                    "level_1": "Feature 2", 
                                    0: 'Correlation Coefficient'}, 
                           inplace=True)
        most_related = df_all_corr[df_all_corr['Feature 1'] == feature]['Feature 2'].iloc[1]
        pd.set_option('display.max_rows', 10)
        df_all_corr[df_all_corr['Feature 1'] == feature]
    else:
        most_related = related_feature
    data.loc[:, feature] = data[feature].replace(-1, np.nan)
    data.loc[:, feature] = data.groupby([most_related])[feature].apply(lambda x: x.fillna(x.mean()))
    data.loc[:, feature] = data[feature].replace(np.nan, -1)
    return data, most_related
def hot_encoding(data, feature):
    print(data.shape)
    data = pd.get_dummies(data, columns=[feature], prefix=[feature])
    print(data.shape)
    return data
def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

def test_results(clf, x_val, y_val):
    X_pos, y_pos = extrac_one_label(x_val, y_val, 0)
    print("Score 0:", clf.score(X_pos, y_pos))
    X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
    print("Score 1:", clf.score(X_pos, y_pos))
    return 0
def pca_transform(data, n=40):
    pca = PCA(n_components=n)
    pca.fit(data)
    data_new = pca.transform(data)
    print(data_new.shape)
    return data_new, pca

def feature_target_corr(x_data, y_data):
    df_correllations = x_data.corrwith(y_data).abs()
    print(df_correllations.sort_values()[0:20])
    return df_correllations.sort_values()

def cat_feature_encoding(data, dict_):
    cat_features = [x for x in list(dict_.keys()) if x[-3:]=="cat"]
    for feature in cat_features:
        print(feature)
        data = hot_encoding(data, feature)
    return data

def over_sampling(data):
    shuffled_df = data.sample(frac=1, random_state=42)
    pos_df = shuffled_df.loc[shuffled_df['target'] == 1]
    neg_df = shuffled_df.loc[shuffled_df['target'] == 0]
    multi = math.ceil(len(neg_df)/len(pos_df))
    pos_df_over = pd.concat([pos_df]*multi).sample(n=len(neg_df))
    over_df = pd.concat([pos_df_over, neg_df])
    return over_df
data_train = pd.read_csv("train_set.csv")

# replace missing features
dict_train = get_all_elements(data_train)
bad_features = find_missing(data_train, dict_train)
data_train, corr_features = replace_missing(bad_features, data_train)

# oversampling
data_train = over_sampling(data_train)

# split into x- and y-data
data_X = data_train[data_train.columns[2:]]
data_Y = data_train["target"]

# encode categorical features
#data_X = cat_feature_encoding(data_X, dict_train)

# split data into training- and test-data
x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y,
                                                  test_size = 0.2, shuffle = True)

# encode features to have mean 0 and standard deviation of 1
scaler = StandardScaler()
scaler.fit(x_train)
x_train = pd.DataFrame(scaler.transform(x_train),
                       index = x_train.index,
                       columns = x_train.columns)
x_val = pd.DataFrame(scaler.transform(x_val),
                     index = x_val.index,
                     columns = x_val.columns)

# select n features using KBest
#selector = SelectKBest(f_classif, k=40)
#selector.fit(data_X, data_Y)
#data_X_new = selector.transform(data_X)

#for corr_len in [10, 12, 14, 16, 18, 20]:
#for alpha in [10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7]:
corr_cols = feature_target_corr(x_train, y_train)
data_X_new = x_train.drop(corr_cols.index.values[0:10], axis=1)
x_val_new = x_val.drop(corr_cols.index.values[0:10], axis=1)

#data_X_new, pca = pca_transform(x_train, 40)
#x_val_new = pca.transform(x_val)


clf = MLPClassifier(hidden_layer_sizes=(120), solver="adam", alpha=10**-6, random_state=1, activation="logistic",
                    verbose=False, max_iter=500)

#clf = SGDClassifier(tol=1e-6, loss="log")
#clf = DecisionTreeClassifier( )
#clf = CategoricalNB()
#clf = GaussianNB()


clf.fit(data_X_new, y_train)

print("Accuracy: ",clf.score(x_val_new, y_val))
test_results(clf, x_val_new, y_val)
print("F-score: ",f1_score(y_val, clf.predict(x_val_new), average="macro"))

# read test_data
data_test = pd.read_csv("test_set.csv")
ids = data_test["id"]
data_test = data_test.drop(columns=['id'])

# calculate and replace missing features
dict_test = get_all_elements(data_test)
bad_features = find_missing(data_test, dict_test)
for feature in bad_features:
    data_test, missing_feature = replace_missing_correlation(feature, data_test, corr_features[feature])

# apply standard scaling
data_test_X = pd.DataFrame(scaler.transform(data_test),
                           index = data_test.index,
                           columns = data_test.columns)

#data_test_X_new = pca.transform(data_test_X)

# remove features with low correlation
data_test_X_new = data_test_X.drop(corr_cols.index.values[0:10], axis=1)

print(data_test_X_new.shape)

y_target = clf.predict(data_test_X_new)
sum(y_target==0)
data_test.columns
print(data_test_X_new.shape)

y_target = clf.predict(data_test_X_new)
sum(y_target==0)
data_test = pd.read_csv("test_set.csv")

data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True)
data_out = data_out.astype({"target": int})
data_out.to_csv('submission.csv',index=False)
data_out
