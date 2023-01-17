import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
sample_submission = pd.read_csv('../input/gender_submission.csv')
test = pd.read_csv('../input/test.csv')
print(data.shape, test.shape)
data.head()
data.info()
data['Survived'].value_counts()
from sklearn.model_selection import train_test_split
data_dummies = pd.get_dummies(data.drop(['Ticket','Name','Cabin'], axis=1))
train, test = train_test_split(data_dummies, test_size=0.2, random_state=100)
train_x = train.drop('Survived', axis=1)
validate_x = test.drop('Survived', axis=1)
train_y = train['Survived']
validate_y = test['Survived']

## Standarized Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)

train_x_scaled = pd.DataFrame(scaler.transform(train_x), columns=train_x.columns)
validate_x_scaled = pd.DataFrame(scaler.transform(validate_x), columns = validate_x.columns)
train_y.shape
data['Name'].nunique()
data['Embarked'].nunique()
### How to extract and grouping
data['Cabin'].apply(lambda v: v[0] if not pd.isnull(v) else float('nan'))
pd.isnull(data['Cabin']).sum()/data.shape[0]+100
### Missing Value Imputation
### Method 1: Impute with column mean 
data_dummies['Age_m1'] = data_dummies['Age'].fillna(data_dummies['Age'].mean())
pd.isnull(data_dummies['Age_m1']).sum()

### Method 2: Impute with column Median
data_dummies['Age_m2'] = data_dummies['Age'].fillna(data_dummies['Age'].median())
pd.isnull(data_dummies['Age_m2']).sum()

### Method 3: Impute using other columns
#data_dummies[pd.isnull(data_dummies['Age'])].head()

avg_age_female = round(data[data['Sex']=='female']['Age'].mean())
avg_age_male = round(data[data['Sex']=='male']['Age'].mean())
print(avg_age_female, avg_age_male)

def method3_impute(row):
    if row.Sex_female == 1 and pd.isnull(row.Age):
        row.Age = avg_age_female
    elif row.Sex_female == 0 and pd.isnull(row.Age):
        row.Age = avg_age_male
    else:
        pass
    return row
data_dummies_m3 = data_dummies.apply(method3_impute, axis=1).head()
data_dummies_m3['Age'].plot.density()

### Method 4: Using KNN
cols_drop = ['Age','Age_m1', 'Age_m2', 'Survived']
train_x_impute = data_dummies[~pd.isnull(data_dummies['Age'])]
train_x_impute = train_x_impute.drop(cols_drop,axis=1)

test_x_impute = data_dummies[pd.isnull(data_dummies['Age'])]
test_x_impute = test_x_impute.drop(cols_drop, axis=1)

train_y_impute = data_dummies.loc[train_x_impute.index, 'Age']
test_y_impute = data_dummies.loc[train_x_impute.index, 'Age']
print(train_x_impute.shape)

### To impute data with KNN need to do standardization 
from sklearn.preprocessing import StandardScaler
scaler_impute = StandardScaler()
scaler_impute.fit(train_x_impute)
train_x_impute_scaled = scaler_impute.transform(train_x_impute)
test_x_impute_scaled = scaler_impute.transform(test_x_impute)

### Fit KNN Model
from sklearn.neighbors import KNeighborsRegressor
model_impute = KNeighborsRegressor(n_neighbors=5)
model_impute.fit(train_x_impute_scaled, train_y_impute)
test_y_impute_pred = model_impute.predict(test_x_impute_scaled)

### Impute Predictions in original data
data_dummies.loc[test_x_impute.index, 'Age'] = test_y_impute_pred
#pd.isnull(data_dummies['Age']).sum()
### After Data Imputation we need to split data again
data_dummies = data_dummies.drop(['Age_m1', 'Age_m2'], axis=1)
train, validate = train_test_split(data_dummies, test_size=0.2, random_state=100)
train_x = train.drop('Survived', axis=1)
validate_x = validate.drop('Survived', axis=1)

train_y = train['Survived']
validate_y = validate['Survived']
pd.isnull(data_dummies).sum()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

m1 = DecisionTreeClassifier(max_depth=4, random_state=100)
m1.fit(train_x, train_y)
validate_pred = pd.DataFrame(m1.predict_proba(validate_x), columns=['Neg_0', 'Pos_1'])

from sklearn.metrics import confusion_matrix, roc_curve, auc
fpr_m1, tpr_m1, cutoffs_1 = roc_curve(validate_y, validate_pred['Pos_1'])
auc_m1 = auc(fpr_m1, tpr_m1)*100
auc_m1
m2 = DecisionTreeClassifier(max_depth=10, random_state=100)
m2.fit(train_x, train_y)
validate_pred_2 = pd.DataFrame(m2.predict_proba(validate_x), columns=['Neg_0', 'Pos_1'])

from sklearn.metrics import confusion_matrix, roc_curve, auc
fpr_m2, tpr_m2, cutoffs_2 = roc_curve(validate_y, validate_pred_2['Pos_1'])
auc_m2 = auc(fpr_m2, tpr_m2)*100
auc_m2
m3 = RandomForestClassifier(random_state = 100, n_estimators = 300)
m3.fit(train_x, train_y)
validate_pred_3 = pd.DataFrame(m3.predict_proba(validate_x), columns=['Neg_0', 'Pos_1'])

from sklearn.metrics import confusion_matrix, roc_curve, auc
fpr_m3, tpr_m3, cutoffs_3 = roc_curve(validate_y, validate_pred_3['Pos_1'])
auc_m3 = auc(fpr_m3, tpr_m3)*100
auc_m3
from sklearn.ensemble import AdaBoostClassifier
m4 = AdaBoostClassifier(random_state = 100, n_estimators = 300)
m4.fit(train_x, train_y)
validate_pred_4 = pd.DataFrame(m4.predict_proba(validate_x), columns=['Neg_0', 'Pos_1'])

from sklearn.metrics import confusion_matrix, roc_curve, auc
fpr_m4, tpr_m4, cutoffs_4 = roc_curve(validate_y, validate_pred_4['Pos_1'])
auc_m4 = auc(fpr_m4, tpr_m4)*100
auc_m4
import matplotlib.pyplot as plt
p1 = plt.plot(fpr_m1, tpr_m1, color = 'red')
plt.xlabel('False Positive rate (FPR)')
plt.ylabel('True Positive rate (TPR)')
p2 = plt.plot(fpr_m2, tpr_m2, color='green')
p3 = plt.plot(fpr_m3, tpr_m3, color='Steelblue')
p4 = plt.plot(fpr_m4, tpr_m4, color='orange')
plt.legend(['Decision Tree Max Depth(4); Auc = %.2f' % auc_m1,
           'Decision Tree Max Depth(10); Auc = %.2f' % auc_m2,
            'RandomForest; Auc = %.2f' % auc_m3,
           'AdaBoost; Auc = %.2f' % auc_m4])


test = pd.read_csv('../input/test.csv')
test.head()
pd.isnull(test).sum()
cols_drop = ['Cabin','Name','Ticket']
test_dummies = pd.get_dummies(test.drop(cols_drop, axis=1))
print(test.columns)
test_x_impute = scaler_impute.transform(test_dummies.drop('Age', axis=1))
test_x_impute = pd.DataFrame(test_x_impute, columns=test_dummies.columns.drop('Age'))

test_age_miss_rows = test[pd.isnull(test['Age'])].index
fare_avg = test_x_impute['Fare'].mean()
test_age_miss = test_x_impute.loc[test_age_miss_rows,:]
test_age_miss['Fare'] = test_age_miss['Fare'].fillna(fare_avg)
test_age_miss_pred = model_impute.predict(test_age_miss)

test_x_impute['Age'] = test['Age']
test_x_impute.loc[test_age_miss_rows, 'Age'] = test_age_miss_pred
test_x_impute['Fare'] = test_x_impute['Fare'].fillna(fare_avg)
test_pred = m3.predict(test_x_impute)
submission = pd.DataFrame({'PassengerId': test['PassengerId'].values,
                          'Survived': test_pred})
submission.to_csv('submission_1.csv', index=False)
