# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from scipy import stats

from sklearn.feature_extraction import DictVectorizer as DV

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# load data

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
# Setting PassengerID to index and separating target column

train_df.set_index('PassengerId',inplace = True)

y_train = train_df.Survived

del train_df['Survived']

test_df.set_index('PassengerId',inplace = True)
train_df.head()
train_df.info()
sns.pairplot(train_df)
# Distribution of classes

y_train.hist(bins = 3)

plt.title('Target distribution')
numeric_cols = ['Age','Fare']

cat_cols = ['Pclass','Sex','SibSp','Parch','Ticket','Cabin','Embarked']
numeric_train = train_df[numeric_cols]

cat_train = train_df[cat_cols].astype(str)



numeric_test = test_df[numeric_cols]

cat_test = test_df[cat_cols].astype(str)
# filling nans with mean for numeric and 'None' for categorial

# train

numeric_train =  numeric_train.fillna(numeric_train.mean())

cat_train = cat_train.fillna('None')

# test

numeric_test = numeric_test.fillna(numeric_test.mean())

cat_test = cat_test.fillna('None')
train_df.Ticket.str.replace('\D','').replace('',np.nan).dropna().astype(int).hist()

plt.title('Distribution of numeric data from Ticket feature')
stats.spearmanr(train_df.Ticket.str.replace('\D','').replace('',0).astype(int),y_train)


train_df.Cabin.str.replace('\D','').replace('',np.nan).dropna().astype(int).hist()

plt.title('Distribution of numeric data from Cabin feature')
stats.spearmanr(train_df.Cabin.str.replace('\D','').replace('',0).fillna(0).astype(int),y_train)
def get_num_info(data):

    """

    Get numbers from string as int

    """

    _num = data.str.replace('\D','')

    _num = _num.replace('',np.nan)

    _num = _num.fillna(_num.dropna().astype(int).median())

    return _num.astype(int)

def get_text_info(data):

    """

    Remove all digits and special symbols, bring remaining to upper case

    """

    _text = data.str.replace('\d+', '').replace('','None') # removing all digits

    _text = _text.str.strip() # removing spaces

    _text = _text.str.replace('.','')

    _text = _text.str.replace('/','')

    _text = _text.str.upper() # everything to upper case

    return _text
# tickets for train

numeric_train['Tickets_number'] = get_num_info(cat_train.Ticket)

cat_train['Tickets_text'] = get_text_info(cat_train.Ticket)

# cabin for train

numeric_train['Cabin_number'] = get_num_info(cat_train.Cabin)

cat_train['Cabin_text'] = get_text_info(cat_train.Cabin)

#tickets for test

numeric_test['Tickets_number'] = get_num_info(cat_test.Ticket)

cat_test['Tickets_text'] = get_text_info(cat_test.Ticket)

#cabin for test

numeric_test['Cabin_number'] = get_num_info(cat_test.Cabin)

cat_test['Cabin_text'] = get_text_info(cat_test.Cabin)



del cat_train['Ticket'],cat_train['Cabin'],cat_test['Ticket'],cat_test['Cabin']
numeric_train.corrwith(y_train,method = 'spearman')
sns.pairplot(numeric_train.join(y_train))
prob_check = cat_train.join(y_train)

for feature in cat_train.columns:

    print('----------')

    print(feature)

    print('----------')

    for item in cat_train[feature].unique():

        print(item, np.round(len(prob_check[(prob_check[feature] == item)&(prob_check['Survived'] == 1)])/len(prob_check[prob_check['Survived'] == 1]),2))     
len(train_df) - train_df['Cabin'].count() 
cat_train['Ticket_has_text'] = np.where(cat_train['Tickets_text'] == 'NONE','1','0')

cat_test['Ticket_has_text'] = np.where(cat_test['Tickets_text'] == 'NONE','1','0')
encoder = DV(sparse = False)

X_cat_train = encoder.fit_transform(cat_train.drop(['Cabin_text','Tickets_text'],axis = 1).T.to_dict().values())

X_cat_submission = encoder.transform(cat_test.drop(['Cabin_text','Tickets_text'],axis = 1).T.to_dict().values())
(X_num_known_train, 

 X_num_known_test, 

 y_train_known, y_test_known) = train_test_split(numeric_train[['Fare','Tickets_number']], y_train,

                                     stratify = y_train,

                                     test_size = 0.2, 

                                     random_state=0)

(X_cat_known_train,

 X_cat_known_test) = train_test_split(X_cat_train,

                                   stratify = y_train,

                                   test_size=0.2, 

                                   random_state=0)
scaler = StandardScaler()

X_num_known_train = scaler.fit_transform(X_num_known_train)

X_num_known_test = scaler.transform(X_num_known_test)
X_num_sumbission = scaler.transform(numeric_test[['Fare','Tickets_number']])
# Putting numerical and categorial data together

train_data_known = np.hstack((X_num_known_train,X_cat_known_train))

test_data_known = np.hstack((X_num_known_test,X_cat_known_test))

submission_data = np.hstack((X_num_sumbission,X_cat_submission))
np.shape(train_data_known)
np.shape(test_data_known)
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn import metrics
models = {'RandomForestClassifier': RandomForestClassifier(class_weight = 'balanced'),

          'LogisticRegression' : LogisticRegression(class_weight = 'balanced'),

          'KNeighborsClassifier' : KNeighborsClassifier(),

          'GradientBoostingClassifier': GradientBoostingClassifier(),

          'MLPClassifier': MLPClassifier(max_iter = 1000)}
results = []

for item in models.values():

    item.fit(train_data_known,y_train_known)

    results.append(metrics.accuracy_score(y_test_known,item.predict(test_data_known)))
list(zip(models.keys(),results))
from sklearn.model_selection import GridSearchCV
parameters = {

    "learning_rate": [0.05, 0.1],

    "max_depth":[10,50],

    "subsample":[0.5, 1.0],

    "n_estimators":[100,1000],

    "max_features":[0.2,0.5,None]

    }

clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv = 3, n_jobs=-1)

clf.fit(train_data_known,y_train_known)
clf.best_score_
clf.best_params_
gbc = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 10, max_features = 0.2, n_estimators = 100, subsample = 0.5)
gbc.fit(train_data_known,y_train_known)
y_predicted = gbc.predict(test_data_known)
def plot_roc_curve(fpr,tpr):

    plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], '--', color = 'grey', label = 'random')

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curve')

    plt.legend(loc = "lower right")

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

print(metrics.classification_report(y_test_known, y_predicted))
print("Accuracy:",metrics.accuracy_score(y_test_known, y_predicted))

fpr, tpr, _ = metrics.roc_curve(y_test_known, gbc.predict_proba(test_data_known)[:,1])

print('ROC AUC: ',metrics.auc(fpr, tpr))

plot_roc_curve(fpr,tpr)
print_confusion_matrix(metrics.confusion_matrix(y_test_known, y_predicted),[0,1])
submission = pd.DataFrame(data = test_df.index,columns = ['PassengerId'])

submission['Survived'] = gbc.predict(submission_data)

submission.to_csv('first_submission.csv',index = False)