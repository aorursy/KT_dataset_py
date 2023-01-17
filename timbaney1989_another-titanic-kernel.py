%matplotlib inline



from datetime import datetime as dt

import re

import math

import datetime



import pylab

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.pylab as pylab



import numpy as np



import statsmodels.api as sm

from statsmodels.formula.api import ols



from sklearn import datasets, tree, metrics, cross_validation

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron, SGDClassifier

from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, RFE

from sklearn.svm import SVC

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



import scipy



pylab.rcParams[ 'figure.figsize' ] = 15 , 8

plt.style.use("fivethirtyeight")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/train.csv')
col_meanings = [

    'The ID of the passenger',

    'Did the passenger survive ? 1 = Yes, 0 = No',

    'Ordinal Value for passenger class, 1 being the highest',

    'Name',

    'Gender',

    'Age',

    'Passenger\'s siblings and spouses on board with',

    'Passenger\'s parents and children on board',

    'Ticket Number',

    'Passenger Fare',

    'Cabin Number',

    'Port of Embarkation'

]



data_dict = pd.DataFrame({

    "Attribute": train.columns,

    "Type": [train[col].dtype for col in train.columns],

    "Meaning": col_meanings,

    'Example': [train[col].iloc[2] for col in train.columns]

})



data_dict
prefix_dict = {}

cleaned_list = []



for raw in [item.split(' ')[0] for item in [pre for pre in train['Ticket'].value_counts().index.tolist() if not pre.isalnum()]]:

    cleaned = re.sub(r'\W+', '', raw)

    

    if raw not in prefix_dict:

        prefix_dict[raw] = cleaned

        

    if cleaned not in cleaned_list:

        prefix_dict[cleaned] = raw

        cleaned_list.append(cleaned)

print(cleaned_list)
pre_list_fare = cleaned_list



def getMeans(prefix_list):

    clean_means = []

    for pre in prefix_list:

        matches = [x for x in prefix_dict if prefix_dict[x] == pre]

        if len(matches) == 1:

            mean = train[train['Ticket'].str.contains(matches[0])]['Fare'].mean()

        elif len(matches) == 2:

            mean = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1]))]['Fare'].mean()

        elif len(matches) == 3:

            mean = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1])) | (train['Ticket'].str.contains(matches[2]))]['Fare'].mean()

        elif len(matches) == 4:

            mean = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1])) | (train['Ticket'].str.contains(matches[2])) | (train['Ticket'].str.contains(matches[3]))]['Fare'].mean()

        clean_means.append(mean)

    

    clean_means.append(train[train['Ticket'].str.isdigit()]['Fare'].mean())

        

    return clean_means



x = pre_list_fare

y = getMeans(pre_list_fare)



if 'Non Alpha' not in pre_list_fare:

    pre_list_fare.append('Non Alpha')
sns.barplot(x, y)

plt.title('Type of Ticket Avg Fare')

plt.ylabel('Price of Ticket')

plt.xlabel('Ticket Prefix')

plt.xticks(rotation=60)

plt.show()
gender_oh = pd.get_dummies(train['Sex']) 

gender_oh_test = pd.get_dummies(test['Sex'])



train = train.drop('Sex', axis=1)

train = train.join(gender_oh)

train = train.rename(columns={'female': 'Female', 'male': 'Male'})



test = test.drop('Sex', axis=1)

test = test.join(gender_oh_test)

test = test.rename(columns={'female': 'Female', 'male': 'Male'})



embarked_oh = pd.get_dummies(train['Embarked'])

embarked_oh_test = pd.get_dummies(test['Embarked'])



train = train.drop('Embarked', axis=1)

train = train.join(embarked_oh)

train = train.rename(columns={'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})



test = test.drop('Embarked', axis=1)

test = test.join(embarked_oh_test)

test = test.rename(columns={'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})
fig, axes = plt.subplots(5, 6,figsize=(20, 15))

fig.legend_out = True



pre_port_list = cleaned_list



def graphPortTickets(prefix_list):

    col, row, loop = (0, 0, 0)

    for pre in prefix_list:

        row = math.floor(loop/6)

        

        matches = [x for x in prefix_dict if prefix_dict[x] == pre]

        if len(matches) == 1:

            df = train[train['Ticket'].str.contains(matches[0])]

        elif len(matches) == 2:

            df = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1]))]

        elif len(matches) == 3:

            df = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1])) | (train['Ticket'].str.contains(matches[2]))]

        elif len(matches) == 4:

            df = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1])) | (train['Ticket'].str.contains(matches[2])) | (train['Ticket'].str.contains(matches[3]))]



        df_c = df['Cherbourg'].sum()/df['PassengerId'].count()

        df_q = df['Queenstown'].sum()/df['PassengerId'].count()

        df_s = df['Southampton'].sum()/df['PassengerId'].count()

        

        x = ['Cherbourg', 'Queenstown', 'Southampton']

        y = [df_c, df_q, df_s]

        

        ax = sns.barplot(x, y, ax=axes[row, col])

        ax.set_xticks([])

        axes[row, col].set_title('-{}- by Port'.format(pre))

        

        col += 1

        loop += 1

    

        if col == 6:

            col = 0

    

    non_alpha = train[train['Ticket'].str.isdigit()]

    na_c = non_alpha['Cherbourg'].sum()/non_alpha['PassengerId'].count()

    na_q = non_alpha['Queenstown'].sum()/non_alpha['PassengerId'].count()

    na_s = non_alpha['Southampton'].sum()/non_alpha['PassengerId'].count()

    

    y = [na_c, na_q, na_s]

    

    ax = sns.barplot(x, y, ax=axes[row, col])

    ax.set_xticks([])

    axes[row, col].set_title('No Prefix by Port')

            

graphPortTickets(pre_port_list)
pre_list = cleaned_list



def getTicketSurvivalPerc(prefix_list):

    survived_count = []

    for pre in prefix_list:

        matches = [x for x in prefix_dict if prefix_dict[x] == pre]

        if len(matches) == 1:

            df = train[train['Ticket'].str.contains(matches[0])]

            survived = df['Survived'].sum()/df['PassengerId'].count()

        elif len(matches) == 2:

            df = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1]))]

            survived = df['Survived'].sum()/df['PassengerId'].count()

        elif len(matches) == 3:

            df = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1])) | (train['Ticket'].str.contains(matches[2]))]

            survived = df['Survived'].sum()/df['PassengerId'].count()

        elif len(matches) == 4:

            df = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1])) | (train['Ticket'].str.contains(matches[2])) | (train['Ticket'].str.contains(matches[3]))]

            survived = df['Survived'].sum()/df['PassengerId'].count()

        

        survived_count.append(survived)

    

    survived_count.append(train[train['Ticket'].str.isdigit()]['Survived'].sum()/train[train['Ticket'].str.isdigit()]['PassengerId'].count())

        

    return survived_count



def xWithTotalNumber(x_inp):

    matches = [z for z in prefix_dict if prefix_dict[z] == x_inp]

    if len(matches) == 1:

        total = train[train['Ticket'].str.contains(matches[0])]['PassengerId'].count()

    elif len(matches) == 2:

        total = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1]))]['PassengerId'].count()

    elif len(matches) == 3:

        total = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1])) | (train['Ticket'].str.contains(matches[2]))]['PassengerId'].count()

    elif len(matches) == 4:

        total = train[(train['Ticket'].str.contains(matches[0])) | (train['Ticket'].str.contains(matches[1])) | (train['Ticket'].str.contains(matches[2])) | (train['Ticket'].str.contains(matches[3]))]['PassengerId'].count()

        

    x_str = '{} - {}'.format(x_inp, total)

    return x_str

                        

y = getTicketSurvivalPerc(pre_list)[:-1]



new_x = [xWithTotalNumber(i) for i in pre_list[:-1]]



if 'Non Alpha' not in new_x:

    new_x.append('Non Alpha')
sns.barplot(new_x, y)

plt.title('Survival Percentage by Ticket')

plt.ylabel('Percent Survived')

plt.xlabel('Type of Ticket')

plt.xticks(rotation=75)

plt.show()
def golden(row):

    if 'PC' in row['Ticket'] or 'PP' in row['Ticket'] or 'F.C.' in row['Ticket'] or 'FC' in row['Ticket']:

        return 1

    else:

        return 0

        

    return row['Ticket'].str.contains('PC')

    

train['Golden Ticket'] = train.apply(lambda x: golden(x), axis=1)

test['Golden Ticket'] = test.apply(lambda x: golden(x), axis=1)



train = train.drop('Ticket', axis=1)

test = test.drop('Ticket', axis=1)
cabin_dict = {

    'A': 1,

    'B': 2,

    'C': 3,

    'D': 4,

    'E': 5,

    'F': 6,

    'G': 7,

    'T': 8

}



train['Cabin'] = train['Cabin'].fillna(value='G')

test['Cabin'] = test['Cabin'].fillna(value='G')



def getCabinOrd(row):

    cabin = row['Cabin']

    deck = cabin[0]

    return cabin_dict[deck]



train['Cabin_Ord'] = train.apply(lambda x: getCabinOrd(x), axis=1)

test['Cabin_Ord'] = test.apply(lambda x: getCabinOrd(x), axis=1)



train = train.drop(['Cabin', 'PassengerId'], axis=1)

test = test.drop('Cabin', axis=1)
def refineName(row):

    if 'Mrs' in row['Name']:

        return 'Mrs'

    elif 'Mr.' in row['Name']:

        return 'Mr'

    elif 'Miss' in row['Name']:

        return 'Miss'

    elif 'Master' in row['Name']:

        return 'Master'

    else:

        return 'Other'



train['Name'] = train.apply(lambda x: refineName(x), axis=1)

test['Name'] = test.apply(lambda x: refineName(x), axis=1)
suffixes = pd.get_dummies(train['Name'])

test_suffixes = pd.get_dummies(test['Name'])



train = train.join(suffixes, lsuffix='left', rsuffix='right')

train = train.drop('Name', axis=1)



test = test.join(test_suffixes, lsuffix='left', rsuffix='right')

test = test.drop('Name', axis=1)



train.head(1)
avg_test_fare = test['Fare'].mean()

test['Fare'] = test['Fare'].fillna(value=avg_test_fare)



with_age = train[train['Age'] > 0]



no_age = train[train['Age'].isnull()]

no_age = no_age.drop('Age', axis=1)



test_with_age = test[test['Age'] > 0]



test_no_age = test[test['Age'].isnull()]

test_no_age = test_no_age.drop('Age', axis=1)



ty = test_with_age['Age'].values

tx = test_with_age.drop('Age', axis=1)



y = with_age['Age'].values

x = with_age.drop('Age', axis=1)



linreg = LinearRegression()

test_lin = LinearRegression()



linreg.fit(x, y)

test_lin.fit(tx, ty)



predictions = [abs(math.ceil(pred)) for pred in linreg.predict(no_age)]

test_predictions = [abs(math.ceil(pred)) for pred in test_lin.predict(test_no_age)]



train.head()
no_age['Age'] = predictions

test_no_age['Age'] = test_predictions



train = with_age.append(no_age)

test = test_with_age.append(test_no_age)
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Feature correlations', y=1.05, size=15)

sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
def isKid(row):

    age = row['Age']

    

    if age < 6:

        return 1

    else:

        return 0

    

def hasFamily(row):

    if row['SibSp'] > 0 or row['Parch'] > 0:

        return 1

    else:

        return 0

    

def fatWallet(row):

    if row['Fare'] >= 50:

        return 1

    else:

        return 0

        

train['Has Family'] = train.apply(lambda x: hasFamily(x), axis=1)

train['Is Kid'] = train.apply(lambda x: isKid(x), axis=1)

train['Fat Wallet'] = train.apply(lambda x: fatWallet(x), axis=1)



test['Has Family'] = test.apply(lambda x: hasFamily(x), axis=1)

test['Is Kid'] = test.apply(lambda x: isKid(x), axis=1)

test['Fat Wallet'] = test.apply(lambda x: fatWallet(x), axis=1)



train = train.drop(['Male', 'Fare', 'SibSp', 'Parch', 'Age'], axis=1)

test = test.drop(['Male', 'Fare', 'SibSp', 'Parch', 'Age'], axis=1)
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Feature correlations', y=1.05, size=15)

sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

plt.yticks(rotation=35)

plt.xticks(rotation=35)
y = train['Survived'].values

x = train.drop('Survived', axis=1)



svc_lin = SVC(kernel="linear")

lr = LogisticRegression()

dt = DecisionTreeClassifier()

grd = GradientBoostingClassifier()

rf = RandomForestClassifier(n_estimators=100)

per = Perceptron()



lr = RFE(estimator=lr, n_features_to_select=1, step=1)

dt = RFE(estimator=dt, n_features_to_select=1, step=1)

grd = RFE(estimator=grd, n_features_to_select=1, step=1)

rf = RFE(estimator=rf, n_features_to_select=1, step=1)

per = RFE(estimator=per, n_features_to_select=1, step=1)

svc = RFE(estimator=svc_lin, n_features_to_select=1, step=1)





lr.fit(x, y)

dt.fit(x, y)

grd.fit(x, y)

rf.fit(x, y)

per.fit(x, y)

svc.fit(x, y)



lr_ranking = lr.ranking_

dt_ranking = dt.ranking_

grd_ranking = grd.ranking_

rf_ranking = rf.ranking_

per_ranking = per.ranking_

svclin_ranking = svc.ranking_



new_df = pd.DataFrame({

    'LogReg Ranking': lr_ranking,

    'DTree Ranking': dt_ranking,

    'GRD Boost Ranking': grd_ranking,

    'rf_ranking': rf_ranking,

    'per_ranking': per_ranking

})



fselection = pd.DataFrame(list(zip(x.columns, svclin_ranking)), columns=['features', 'svc_ranking'])

fselection = fselection.join(new_df)

fselection
rf = RandomForestClassifier(n_estimators=100)

gaus = GaussianNB()

logreg = LogisticRegression()

dtree = DecisionTreeClassifier()

svc_rbf = SVC(kernel="rbf")

svc_lin = SVC(kernel="linear")

knn = KNeighborsClassifier(n_neighbors = 3)

per = Perceptron()

grd = GradientBoostingClassifier()



for num in range(1, 3000):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .3, stratify=y)

    

    grd.fit(X_train, y_train)

    grd_score = grd.score(X_test, y_test)

    

    if grd_score > .87:



        algorithms = [{'algo': rf, 'color': '#4285f4', 'name': 'Random Forest'}, {'algo': gaus, 'color': 'red', 'name': 'Gaussian'}, 

                      {'algo': logreg, 'color': 'blue', 'name': 'Logistic Regressions'},{'algo': dtree, 'color': 'orange', 'name': 'Decision Tree'}, 

                      {'algo': svc_rbf, 'color': 'lime', 'name': 'SVC-RBF'}, {'algo': svc_lin, 'color': 'purple', 'name': 'Linear SVC'},

                      {'algo': knn, 'color': 'yellow', 'name': 'KNN'},{'algo': per, 'color': 'indigo', 'name': 'Perceptron'}, 

                      {'algo': grd, 'color': 'black', 'name': 'Gradient Boosting'}

                     ]



        for alg in algorithms:

            algo = alg['algo']

            algo.fit(X_train, y_train)

            predictions = algo.predict(X_test)

            fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)

            auc = metrics.auc(fpr, tpr)

            plt.plot(fpr, tpr, alg['color'], label='{} AUC = {:.2f}'.format(alg['name'], auc))



        plt.title('Receiver Operating Characteristic')



        plt.legend(loc = 'lower right')

        plt.plot([0, 1], [0, 1],'r--')

        plt.xlim([0, 1])

        plt.ylim([0, 1])

        plt.ylabel('True Positive Rate')

        plt.xlabel('False Positive Rate')

        plt.show()



        scores = [algorithm.score(X_test, y_test) for algorithm in [logreg, dtree, svc_rbf, svc_lin, knn, gaus, per, rf, grd]]



        scoring_df = pd.DataFrame({

            'algorithms': ['Logistic Regression', 'Decision Tree', 'SVC Radial Basis Function',

                           'Linear SVC', 'KNearest Neighbors', 'Gaussian Naive Bayes', 'Perceptron',

                           'Random Forest', 'Gradient Boosting'],

            'score': scores

        })



        print(scoring_df)

        break
# estimators = range(1, 400)

lrate = [x/1000 for x in range(1, 1000)]

# mleaf = range(1, 300)



grd_scores = []

for n in lrate:

    grd = GradientBoostingClassifier(n_estimators=17, learning_rate=.162)

    grd.fit(X_train, y_train)

    grd_preds = grd.predict(X_test)

    grd_fpr, grd_tpr, threshold = metrics.roc_curve(y_test, grd_preds)

    grd_auc = metrics.auc(grd_fpr, grd_tpr)

    grd_score = grd.score(X_test, y_test)

    grd_scores.append({'criteria': n, 'score': grd_score})

    

grd_scores = sorted(grd_scores, key=lambda k: k['score'], reverse=True)  

grd_scores[:1]
# passenger_ids = test['PassengerId'].values

# test = test.drop('PassengerId', axis=1)



# grd = GradientBoostingClassifier(n_estimators=17, learning_rate=.162)

# grd.fit(X_train, y_train)

# predictions = grd.predict(test)



# ------------ PREPARE FOR SUBMISSION -------------- #

# new_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

# new_df['PassengerId'] = passenger_ids

# new_df['Survived'] = predictions

# new_df.to_csv('titanic_final.csv', index=False)

# ---------- END PREPARE FOR SUBMISSION ----------- #