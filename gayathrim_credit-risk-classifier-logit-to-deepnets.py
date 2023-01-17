import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization
# import os
# print(os.listdir("../input"))
data = pd.read_csv('../input/cleaned_statlog.csv', header=None)
data.head()
data.info()
data.columns = ['account_check_status','duration','cred_history','credit_k','savings','employment','personal_status','residence',
               'property','age','other_installment','exisiting_credits','maintenance_people','telephone','foreign_worker','loan_new_car','loan_used_car','other_debtor_none',
                'other_debtor_coapplicant','housing_rent','housing_own','unskilled_nonresident','unskilled_resident','skilled_employee',
              'default']
data.head()
data['default'] -= 1
data['default'].value_counts() # 0 - good candidate
y = data['default']
X = data.drop(['default'],axis=1)
from sklearn.model_selection import train_test_split as split
X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=0, stratify=y)
print('Negative Class Distribution- \n{0} in train data \n{1} in test data'.format(sum(y_train)/len(y_train), sum(y_test)/len(y_test)))
# Pair plot
col_group = data.columns[:-1].tolist()
n = 4
for each in [col_group[i::n] for i in range(n)]:
    sns.pairplot(data[['default']+each], hue='default')
# Heat map
sns.heatmap(data.corr())
# naive baseline
naive_baseline = sum(data['default']==0)/len(data)
naive_baseline
# Model evaluation

from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True, gamma='auto'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

log_cols = ["Classifier", "Accuracy"]
log  = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = data.drop(['default'],axis=1).values
y = data['default'].values

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
plt.show()
log.sort_values('Accuracy', ascending=False)
y = data['default']
X = data.drop(['default'],axis=1)
from sklearn.model_selection import train_test_split as split
X_train, X_test, y_train, y_test = split(X, y, test_size=0.1, random_state=0, stratify=y)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
n_features = X_train.shape[1]

scalar = MinMaxScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)

model = Sequential()
model.add(Dense(n_features, input_dim=(n_features), activation='relu')) # input
model.add(Dense(round(n_features/2), activation='relu')) # hidden
model.add(Dense(1, activation='sigmoid')) # output
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Learning
model.fit(X_train, y_train, epochs=100, batch_size=10)
y_pred = np.round(model.predict(scalar.transform(X_test)))
# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)
confusion_m
tn, fp, fn, tp = confusion_m.ravel()
accuracy = (tn+tp)/len(y_test)
fn, accuracy, fn/(fn+tp)
