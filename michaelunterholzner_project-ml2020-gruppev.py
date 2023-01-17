# Quick load dataset and check
import pandas as pd

filename = "train_set.csv"
data_train = pd.read_csv(filename)
filename = "test_set.csv"
data_test = pd.read_csv(filename)
data_train.describe()
data_test.describe()

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
clf = DecisionTreeClassifier(min_impurity_decrease = 0.001)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)

sum(y_pred==y_val)/len(y_val)
def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)
sum(y_pospred==y_pos)/len(y_pos)
X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
sum(y_negpred==y_neg)/len(y_neg)
print(sum(data_Y==0)/len(data_Y), sum(data_Y==1))
## Your work
#Please run the full provided project for necessary imports and variables
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

rus = RandomUnderSampler(random_state=0)
x_train, y_train = rus.fit_sample(x_train, y_train)

clf = LogisticRegression(solver='liblinear', random_state=0)
clf = clf.fit(x_train, y_train)
#test evaluation

#threshold
#need to do this because some 1 are wrong predicted
#probabilty 0.4 got the best results
probability = clf.predict_proba(x_val)
y_pred=list()
for proba in probability:
    if proba[0]<0.4: 
        y_pred.append(1)
    else:
        y_pred.append(0)

print(classification_report(y_val, y_pred))
data_test_X = data_test.drop(columns=['id'])
probability = clf.predict_proba(data_test_X)
y_target=list()
for proba in probability:
    if proba[0]<0.4: 
        y_target.append(1)
    else:
        y_target.append(0)
data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)
data_out
