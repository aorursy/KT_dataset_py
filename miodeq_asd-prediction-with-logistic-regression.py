import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
df = pd.read_csv(
    '../input/Autism_Data.arff', 
    na_values='?',
)
data = df.copy()
data.shape
data.head()
data.describe()
data.info()
data.dtypes.value_counts()
total_missing_data = data.isnull().sum().sort_values(ascending=False)

percent_of_missing_data = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)

missing_data = pd.concat(
    [
        total_missing_data, 
        percent_of_missing_data
    ], 
    axis=1, 
    keys=['Total', 'Percent']
)

missing_data.head()
data.iloc[:, 0:10].sum(axis=1).head(10)
data.columns = map(lambda x: x.strip().lower(), data.columns)
data.rename(columns={'class/asd': 'decision_class'}, inplace=True)
data.jundice = data.jundice.apply(lambda x: 0 if x == 'no' else 1)
data.austim = data.austim.apply(lambda x: 0 if x == 'no' else 1)
data.used_app_before = data.used_app_before.apply(lambda x: 0 if x == 'no' else 1)
data.decision_class = data.decision_class.apply(lambda x: 0 if x == 'NO' else 1)
le = LabelEncoder()

data.gender = le.fit_transform(data.gender) 
data.contry_of_res = data.contry_of_res.astype('str')
data.contry_of_res = data.contry_of_res.str.lower()
data.contry_of_res = data.contry_of_res.str.replace("'", "")
data.contry_of_res = data.contry_of_res.str.strip()
data.relation = data.relation.replace(np.nan, 'unknown', regex=True)
data.relation = data.relation.astype('str')
data.relation = data.relation.str.lower()
data.relation = data.relation.str.replace("'", "")
data.relation = data.relation.str.strip()
data.ethnicity = data.ethnicity.replace(np.nan, 'unknown', regex=True)
data.ethnicity = data.ethnicity.astype('str')
data.ethnicity = data.ethnicity.str.lower()
data.ethnicity = data.ethnicity.str.replace("'", "")
data.ethnicity = data.ethnicity.str.strip()
data.gender[data.decision_class == 0].value_counts() # 0-female, 1-male
data.gender[data.decision_class == 1].value_counts() # 0-female, 1-male
data.ethnicity[data.decision_class == 0].value_counts()
data.ethnicity[data.decision_class == 0].value_counts().plot(kind='bar')
data.ethnicity[data.decision_class == 1].value_counts()
data.ethnicity[data.decision_class == 1].value_counts().plot(kind='bar')
data.relation[data.decision_class == 0].value_counts()
data.relation[data.decision_class == 0].value_counts().plot(kind='bar')
data.relation[data.decision_class == 1].value_counts()
data.relation[data.decision_class == 1].value_counts().plot(kind='bar')
data.contry_of_res[data.decision_class == 0].value_counts().head(10)
data.contry_of_res[data.decision_class == 0].value_counts().head(15).plot(kind='bar')
data.contry_of_res[data.decision_class == 1].value_counts().head(10)
data.contry_of_res[data.decision_class == 1].value_counts().head(15).plot(kind='bar')
lb = LabelBinarizer()

lb.fit(data.contry_of_res.values)

binarized_data = lb.transform(data.contry_of_res.values)

binarized_contry_of_res_matrix_data = np.vstack(binarized_data)
binarized_contry_of_res_matrix_data.shape
lb = LabelBinarizer()

lb.fit(data.relation.values)

binarized_data = lb.transform(data.relation.values)

binarized_result_matrix_data = np.vstack(binarized_data)
binarized_result_matrix_data.shape
lb = LabelBinarizer()

lb.fit(data.ethnicity.values)

binarized_data = lb.transform(data.ethnicity.values)

binarized_ethnicity_matrix_data = np.vstack(binarized_data)
binarized_ethnicity_matrix_data.shape
data.drop(['age_desc', 'result'], axis=1, inplace=True)
data.head()
data.age.max(), data.age.min()
print('The oldest patient: {} years.'.format(data.age.max()))
print('The youngest patient: {} years.'.format(data.age.min()))
print('Average age: {} years.'.format(data.age.mean()))
print('Median age: {} years.'.format(data.age.median(skipna=True)))
data.loc[(data.age == 383)]
data.age.median()
data.age.replace(data.age.max(), data.age.median(), inplace=True)
plt.figure(figsize=(15,8))

sns.kdeplot(
    data.age[data.decision_class == 1], 
    color="darkturquoise", 
    shade=True
)

sns.kdeplot(
    data.age[data.decision_class == 0], 
    color="lightcoral", 
    shade=True
)

plt.legend(['ASD', 'NOT ASD'])
plt.title('age vs decision_class')
plt.xlim(data.age.min() - 10, data.age.max() + 10)
plt.show()
data.decision_class.value_counts()
data.gender[data.decision_class == 1].value_counts()
data.gender[data.decision_class == 0].value_counts()
data.isnull().sum()
to_update_nans_dict = {}

columns = [
    'age'
]

for _decision_class in [0, 1]:
    for column in columns:
        vals = data[data.decision_class == _decision_class][column].value_counts()
        
        to_update_nans_dict['{decision_class}_{column}'.format(
            decision_class=_decision_class,
            column=column
        )] = vals.idxmax()
to_update_nans_dict
data.iloc[62, data.columns.get_loc('age')] = to_update_nans_dict.get('0_age')
data.iloc[91, data.columns.get_loc('age')] = to_update_nans_dict.get('1_age')
corr = data.corr()

sns.heatmap(
    data=corr,
    annot=True,
    fmt='.2f',
    linewidths=.5,
    cmap='RdYlGn',
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values
)

fig = plt.gcf()
fig.set_size_inches(40, 20)

plt.show()
feature_names = list(
    set(data.columns[0:-1].tolist()).difference(['contry_of_res', 'relation', 'ethnicity'])
)

X = data[feature_names].as_matrix()

X_new = np.hstack((
    X, 
    binarized_contry_of_res_matrix_data,
    binarized_ethnicity_matrix_data,
    binarized_result_matrix_data,
))
                   
y = data.decision_class
X_new.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(
    X_new,
    y,
    random_state=42,
    test_size=0.3
)
log_reg = LogisticRegression(
    C=1,
    penalty='l1',
    solver='liblinear',
    random_state=42,
    multi_class='ovr'

)
log_reg.fit(X_train, y_train)
log_reg_predict = log_reg.predict(X_test)
log_reg.score(X_test, y_test)
preds = log_reg.predict(X_test)
log_reg_predict_proba = log_reg.predict_proba(X_test)[:, 1]
print('\nLogistic Regression Accuracy: {:.2f}%'.format(accuracy_score(y_test, log_reg_predict) * 100))
print('Logistic Regression AUC: {:.2f}%'.format(roc_auc_score(y_test, log_reg_predict) * 100))
print('Logistic Regression Classification report:\n\n', classification_report(y_test, log_reg_predict))
print(confusion_matrix(y_test, preds))
fpr, tpr, thresholds = roc_curve(
    y_test, 
    log_reg_predict_proba
)

plt.plot(
    [0, 1], 
    [0, 1], 
    'k--'
)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 14

plt.title('ROC curve for Logistic Regression')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
kfold = StratifiedKFold(
    n_splits=10, 
    shuffle=True, 
    random_state=42
)

predicted = cross_val_predict(
    log_reg, 
    X_new, 
    y, 
    cv=kfold
)

scores = cross_val_score(
    log_reg, 
    X_new, 
    y, 
    cv=kfold,
    scoring='f1'
)

print('Cross-validated scores: {}\n'.format(scores))

print(classification_report(y, predicted))

print("\nLogisticRegression: F1 after 10-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))
