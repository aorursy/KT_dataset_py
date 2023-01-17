import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve,roc_auc_score

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv', index_col=0)

test_df = pd.read_csv('../input/test.csv', index_col=0)

gender_sub = pd.read_csv('../input/gender_submission.csv')

combined_df = pd.concat([train_df, test_df])
train_df.info(), test_df.info()
train_df.Cabin.isnull().sum()/len(train_df), test_df.Cabin.isnull().sum()/len(test_df)
combined_df['Age'].fillna(combined_df['Age'].mode().values[0], inplace=True)

combined_df['Embarked'].fillna(combined_df.Embarked.mode().values[0], inplace=True)

combined_df['Fare'].fillna(combined_df.Fare.median(), inplace=True)

combined_df.drop(columns = 'Cabin', inplace=True)
combined_df.info()
feats = test_df.columns

train_df = combined_df[~combined_df.Survived.isnull()]

test_df = combined_df[combined_df.Survived.isnull()].drop(columns='Survived')

data_sets = [train_df, test_df]
train_df.info()

test_df.info()
sns.countplot(train_df.Survived);

plt.title("Histogram of Survival in training data");
train_df.Survived.value_counts()/len(train_df)*100
train_df.dtypes
feats = train_df.columns.tolist()

len(train_df.Ticket.unique())
titles = ['Mr', 'Mrs', 'Miss', 'Master']
def assign_titles(data):

    temp_titles_df = pd.DataFrame(index = data.index)

    temp_titles_df['Title1'] = data['Name'].apply(lambda x: titles[0] if titles[0] in x else None)

    temp_titles_df['Title2'] = data['Name'].apply(lambda x: titles[1] if titles[1] in x else None)

    temp_titles_df['Title3'] = data['Name'].apply(lambda x: titles[2] if titles[2] in x else None)

    temp_titles_df['Title4'] = data['Name'].apply(lambda x: titles[3] if titles[3] in x else None)

    

    def _return_corect_col(row):

        value = "Other"

        if row['Title1']:

            value = row['Title1']

        if row['Title2']:

            value = row['Title2']

        if row['Title3']:

            value = row['Title3']

        if row['Title4']:

            value = row['Title4']

        return value

    

    temp_titles_df['Title'] = temp_titles_df.apply(lambda x : _return_corect_col(x), axis=1)

    

    return pd.merge(data, temp_titles_df[['Title']], left_index=True, right_index=True)['Title']

for data in data_sets:

    data['Title'] = assign_titles(data) # engineering Title from feature Name

    data['FamilySize'] = data['SibSp'] + data['Parch'] # new feature as a combination of two others
data_sets[0].head()
feats_to_drop = ['Ticket', 'Name']

for data in data_sets:

    data.drop(columns=feats_to_drop, inplace=True)
cat_feats = train_df.dtypes[train_df.dtypes == 'object'].index.values.tolist()

num_feats = train_df.dtypes[train_df.dtypes != 'object'].index.values.tolist()

cat_feats, num_feats
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

plt.title('Fare')

plt.boxplot(train_df.Fare);



plt.subplot(1,2,2)

plt.title('Age')

plt.boxplot(train_df.Age);
print ("Fare median: {}, Fare mean: {} \nAge median: {}, Age mean: {}".format(

    np.median(train_df.Fare), np.mean(train_df.Fare), np.median(train_df.Age), np.mean(train_df.Age)))
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)



sns.distplot(train_df[train_df.Survived == 0]['Fare'], label='Not Survived');

sns.distplot(train_df[train_df.Survived == 1]['Fare'], label='Survived');

plt.legend();



plt.subplot(1,2,2)

sns.distplot(train_df[train_df.Survived == 0]['Age'], label='Not Survived');

sns.distplot(train_df[train_df.Survived == 1]['Age'], label='Survived');

plt.legend();
a = sns.FacetGrid(train_df, hue = 'Survived', aspect=3)

a.map(sns.kdeplot, 'Age', shade= True );

plt.title("Ladny wykres");
train_df = data_sets[0]



plt.figure(figsize=(16,10))



plt.subplot(2,3,1)

plt.title("Survival by class");

sns.countplot(x='Pclass', hue='Survived', data=train_df);

plt.legend();



plt.subplot(2,3,2)

plt.title("Survival by sex");

sns.countplot(x='Sex', hue='Survived', data=train_df);

plt.legend();



plt.subplot(2,3,3)

plt.title("Survival by Family Size");

sns.countplot(x='FamilySize', hue='Survived', data=train_df);

plt.legend();



plt.subplot(2,3,4)

plt.title("Survival by Embarked");

sns.countplot(x='Embarked', hue='Survived', data=train_df);

plt.legend();



plt.subplot(2,3,5)

plt.title("Survival by Title");

sns.countplot(x='Title', hue='Survived', data=train_df);

plt.legend();
# combined_df['Sex_cat'] = pd.factorize(combined_df.Sex)[0]

# combined_df['Embarked_cat'] = pd.factorize(combined_df.Embarked)[0]
train_df.head()
corr_df = train_df.corr()

sns.heatmap(corr_df, vmin=-1, vmax=1, center=0,

    cmap= sns.diverging_palette(130, 275, n=200),

    square=True);
corr_df = train_df.corr().abs().unstack().sort_values(ascending=False).reset_index().rename(

    columns={'level_0':'Feature 1', 'level_1':'Feature 2', 0:'Corr coef'})

corr_df.drop(corr_df[corr_df['Corr coef']==1].index, inplace=True)

corr_df = corr_df.iloc[1::2]

corr_df.reset_index(drop=True).iloc[range(0,10),:]
for i in range(0, len(data_sets)):

    data_sets[i] = pd.get_dummies(data_sets[i])
train_df, test_df = data_sets

X_train, X_val, Y_train, Y_val = train_test_split(train_df.drop(columns='Survived'), train_df.Survived)

X_test = data_sets[1]
rf_clf = RandomForestClassifier(random_state=42)

rf_clf.fit(X_train, Y_train)

y_train_pred = rf_clf.predict(X_train)

y_val_pred = rf_clf.predict(X_val)

y_val_pred_prob = rf_clf.predict_proba(X_val)[:,1]

metrics.accuracy_score(Y_train, y_train_pred), metrics.accuracy_score(Y_val, y_val_pred)
conf_mat = confusion_matrix(y_val_pred, Y_val)

sns.heatmap(pd.DataFrame(conf_mat, index=['Survived', 'Not Survived'], columns = ['Survived', 'Not Survived']), annot=True, vmin=0);

plt.xlabel('Actual');

plt.ylabel('Predicted');
tp, fn, fp, tn = confusion_matrix(y_val_pred, Y_val).ravel()

tp, fn, fp, tn
metrics.precision_recall_fscore_support(Y_val, y_val_pred)
fpr, tpr, thresholds = roc_curve(Y_val, y_val_pred_prob)

plt.plot(fpr, tpr);

plt.plot([[0,0], [1,1]], linestyle='dashed');

plt.ylabel('True positive rate');

plt.xlabel('False positive rate');

plt.title('AUC is {}'.format(roc_auc_score(Y_val, y_val_pred_prob)));
params = {'n_estimators' : [5, 10, 20, 50, 100, 200], 'criterion' : ['gini', 'entropy'],

          'max_depth': [2, 4, 6, 8, 10, None], 'random_state':[0]}
rf = RandomForestClassifier()

clf = GridSearchCV(rf, params, cv=3)

clf.fit(X_train, Y_train)
clf.best_params_
rf_clf2 = RandomForestClassifier(criterion='entropy', max_depth=8, n_estimators= 100, random_state= 0)

rf_clf2.fit(X_train, Y_train)

y_train_pred = rf_clf2.predict(X_train)

y_val_pred = rf_clf2.predict(X_val)

y_val_pred_prob = rf_clf2.predict_proba(X_val)[:,1]

metrics.accuracy_score(Y_train, y_train_pred), metrics.accuracy_score(Y_val, y_val_pred)
fpr, tpr, thresholds = roc_curve(Y_val, y_val_pred_prob)

plt.plot(fpr, tpr);

plt.plot([[0,0], [1,1]], linestyle='dashed');

plt.ylabel('True positive rate');

plt.xlabel('False positive rate');

plt.title('AUC is {}'.format(roc_auc_score(Y_val, y_val_pred_prob)));
rf_clf3 = RandomForestClassifier(criterion='entropy', max_depth=8, n_estimators= 100, random_state= 0)
rf_clf3.fit(train_df.drop(columns='Survived'), train_df.Survived)

y_pred_test = rf_clf3.predict(X_test)
submission = pd.DataFrame(y_pred_test, index=X_test.index, columns=['Survived'])

submission.head()