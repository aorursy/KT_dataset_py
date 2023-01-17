# Quick load dataset and check
import pandas as pd
filename = "train_set.csv"
data_train = pd.read_csv(filename)
filename = "test_set.csv"
data_test = pd.read_csv(filename)
data_train['target'].value_counts()
data_train.describe()

from sklearn.tree import DecisionTreeClassifier

## Select target and features
fea_col = data_train.columns[2:]

data_Y = data_train['target']
data_X = data_train[fea_col]

clf = DecisionTreeClassifier()
clf = clf.fit(data_X,data_Y)
y_pred = clf.predict(data_X)
sum(y_pred==data_Y)/len(data_Y)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)

sum(y_pred==y_val)/len(y_val)
def extrac_one_label(x_test, y_test, label):
    X_pos = x_test[y_test == label]
    y_pos = y_test[y_test == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_test, y_test, 1)
y_pospred = clf.predict(X_pos)
sum(y_pospred==y_pos)/len(y_pos)
X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
sum(y_negpred==y_neg)/len(y_neg)
print(sum(data_Y==0)/len(data_Y), sum(data_Y==1))
data = data_X
fea_col = data_train.columns[1:]
data = data_train[fea_col]
data = data.loc[(data==-1).mean(axis=1) < 0.1]

data_Y = data['target']
data = data.drop(['target'],axis=1)

data.describe()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score


data = data_X
fea_col = data_train.columns[1:]
data = data_train[fea_col]
data = data.loc[(data==-1).mean(axis=1) < 0.1]

data_Y = data['target']
data = data.drop(['target'],axis=1)

imputer = SimpleImputer(missing_values=-1, strategy='median', copy=False)
median_data_X = imputer.fit_transform(data)

scaler = StandardScaler()
centered_data_X = scaler.fit_transform(median_data_X)

scaler = MinMaxScaler()
minmax_data_X = scaler.fit_transform(centered_data_X)

data = minmax_data_X

x_train, x_test, y_train, y_test = train_test_split(data, data_Y, test_size = 0.1, shuffle = True)

rfc = RandomForestClassifier(n_estimators=100)
rfc = rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

print(sum(y_pred==y_test)/len(y_test))
print(sum(y_pred==0)/len(y_pred))
f1_score(y_test, y_pred, average='macro')
x_test = data_test.drop(columns=['id'])
y_target = rfc.predict(x_test)

print(sum(y_target==0)/len(y_target))
sum(y_target==1)
data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission_final_version.csv',index=False)
data_out

