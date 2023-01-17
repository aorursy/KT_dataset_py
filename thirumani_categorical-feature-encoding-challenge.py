# Importing required packages



import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 50)



import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,confusion_matrix,roc_curve,auc

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures,LabelEncoder,OneHotEncoder

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

test_ids = test["id"]
# Check number of features and data points in train and test

print("Number of data points in train: %d" % train.shape[0])

print("Number of features in train: %d" % train.shape[1])



print("Number of data points in test: %d" % test.shape[0])

print("Number of features in test: %d" % test.shape[1])
train.head()
test.tail(10).T
# Unique values in each column

for col in train.columns:

    print("Unique entries in",col," -", train[col].nunique())
train['bin_3'] = train["bin_3"].apply(lambda x: 0 if x == "F" else 1)

train['bin_4'] = train["bin_4"].apply(lambda x: 0 if x == "N" else 1)



test['bin_3'] = test["bin_3"].apply(lambda x: 0 if x == "F" else 1)

test['bin_4'] = test["bin_4"].apply(lambda x: 0 if x == "N" else 1)



train['ord_5a'] = train["ord_5"].str[0]

train['ord_5b'] = train["ord_5"].str[1]



test['ord_5a'] = test["ord_5"].str[0]

test['ord_5b'] = test["ord_5"].str[1]
train.info()
test.info()
# Checking for NULL/missing values

train.isnull().sum()
# Checking for NULL/missing values

test.isnull().sum()
# Univariate analysis

print(train['target'].value_counts(),'\n')

print(train['target'].value_counts(normalize=True)*100,'\n')

sns.countplot(train["target"])
train.head()
train['ord_1'].str.lower().value_counts().sort_values()
train['ord_2'].str.lower().value_counts().sort_values()
train.head()
test.head()
high_card_feats = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

for col in high_card_feats:

    train[f'hash_{col}'] = train[col].apply( lambda x: hash(str(x)) % 5000 )

    test[f'hash_{col}'] = test[col].apply( lambda x: hash(str(x)) % 5000 )



train.drop(["id","nom_5","nom_6","nom_7","nom_8","nom_9","ord_5"], axis=1, inplace=True)

test.drop(["id","nom_5","nom_6","nom_7","nom_8","nom_9","ord_5"], axis=1, inplace=True)
train.head()
test.head()
# train_new = pd.DataFrame()

# le = LabelEncoder()

# for c in train.columns:

#     if(train[c].dtype == 'object'): train_new[c] = le.fit_transform(train[c])

#     else:      train_new[c] = train[c]



# train_new.head()
# test_new = pd.DataFrame()

# for c in test.columns:

#     if(test[c].dtype == 'object'): test_new[c] = le.transform(test[c])

#     else:      test_new[c] = test[c]



# test_new.head()
# One Hot Encoding



# str_cols= train.loc[:, train.dtypes=='object'].columns.tolist()

# str_cols



# train = pd.get_dummies(train, columns=str_cols, drop_first=True)

# test = pd.get_dummies(test, columns=str_cols, drop_first=True)



# onehot_enc = OneHotEncoder()

# # train = onehot_enc.fit_transform(train)

# # test = onehot_enc.transform(test)



# import category_encoders as ce

# ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)

# train = ohe.fit_transform(train)

# test = ohe.transform(test)
target = train.pop('target')

data = pd.concat([train, test])

dummies = pd.get_dummies(data, columns=data.columns, drop_first=True, sparse=True)

train = dummies.iloc[:train.shape[0], :]

test = dummies.iloc[train.shape[0]:, :]

train = train.sparse.to_coo().tocsr()

test = test.sparse.to_coo().tocsr()
# train2 = train.drop(['target'], axis=1)

# target = train["target"]

# train2 = train2.loc[:, test.columns]
# # Using imblearn for Balancing Data

# from imblearn.over_sampling import SMOTE

# sm = SMOTE(random_state=2019)



# from imblearn.over_sampling import ADASYN

# sm = ADASYN()



# from imblearn.over_sampling import SVMSMOTE

# sm = SVMSMOTE(random_state=2019)



# from imblearn.combine import SMOTETomek

# sm = SMOTETomek(ratio='auto')



# from imblearn.combine import SMOTEENN

# sm = SMOTEENN(random_state=2019)



# train2, target = sm.fit_sample(train2, target.ravel())



# from collections import Counter

# print('Resampled dataset shape %s' % Counter(target))



# from imblearn.under_sampling import NearMiss

# nr = NearMiss()

# train2, target = nr.fit_sample(train2, target.ravel())

# np.bincount(target)
# # Scaling Data

# #scaler = MinMaxScaler()

# scaler = StandardScaler()

# train2 = scaler.fit_transform(train2)

# test = scaler.transform(test)
# poly = PolynomialFeatures(degree=1)

# train2 = poly.fit_transform(train2)

# test = poly.transform(test)

# poly

# print("train2 shape:", train2.shape)



# # # train2 shape: (14272, 44)
# pca = PCA(random_state=2019)

# #pca = PCA(random_state=2019, n_components=200)

# train2 = pca.fit_transform(train2)



# test = pca.transform(test)

# pca
x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.25, random_state=123)

print(x_train.shape, x_val.shape)
sns.set('talk', 'whitegrid', 'dark', font_scale=1, font='Ricty',rc={"lines.linewidth": 2, 'grid.linestyle': '--'})



# Receiver Operating Characteristic

def plotAUC(truth, pred, lab):

    fpr, tpr, _ = roc_curve(truth,pred)

    roc_auc = auc(fpr, tpr)

    lw = 2

    c = (np.random.rand(), np.random.rand(), np.random.rand())

    plt.plot(fpr, tpr, color= c,lw=lw, label= lab +'(AUC = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curve')

    plt.legend(loc="lower right")
# #clf_NN = MLPClassifier(random_state=2019, hidden_layer_sizes=(100, 100, 100))

# clf_NN = MLPClassifier(activation='tanh', alpha=0.0001, max_iter=200, hidden_layer_sizes=(50, 50, 50), random_state=2019, solver='sgd')

# clf_NN.fit(x_train, y_train)

# y_val_pred = clf_NN.predict(x_val)

# predictprob = clf_NN.predict_proba(x_val)[:,1]

# y_val_pred

# y_pred = clf_NN.predict(test)
#clf_NN
# Random Forest

# rf = RandomForestClassifier(n_estimators=800, random_state = 2019).fit(x_train, y_train)

# rf
rf = LogisticRegression(C=0.1338, solver="lbfgs", tol=0.003, max_iter=4000)

rf.fit(x_train, y_train)
y_val_pred = rf.predict(x_val)

predictprob = rf.predict_proba(x_val)[:,1]

y_pred = rf.predict(test)

y_pred
Accuracy = accuracy_score(y_val, y_val_pred)

print(Accuracy)

plotAUC(y_val, predictprob, 'MLP')

plt.show()



# ROC 0.745

# AUC 0.76
# Confusion Matrix

cm = confusion_matrix(y_val, y_val_pred).T

cm = cm.astype('float')/cm.sum(axis=0)

ax = sns.heatmap(cm, annot=True, cmap='Blues');

ax.set_xlabel('True Label',size=12)

ax.set_ylabel('Predicted Label',size=12)



# # TP 0.815 TN 0.44

# RF # TP 0.96 TN 0.17

# SMOTE TP 0.96 TN 0.63
submission = pd.DataFrame(data = {"id":test_ids, "target":y_pred})

print(submission['target'].value_counts())

submission.to_csv("/kaggle/working/submission.csv", index=False)