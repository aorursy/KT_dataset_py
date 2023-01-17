import numpy as np

import pandas as pd

import pylab as P

from sklearn.ensemble import RandomForestClassifier 

from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score

from sklearn import preprocessing

from sklearn.cross_validation import StratifiedKFold

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics

from IPython.display import display, HTML



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def load_training():

    train_init = pd.read_csv('../input/train.csv', header=0)

    df = train_init.copy()

    return df



def load_testing():

    test_init = pd.read_csv('../input/test.csv', header=0)

    test = test_init.copy()

    return test





df_train = load_training()

display(df_train.head()) 

display(df_train.info())

def generate_new_features(df):

    df['Family_size'] = df['Parch'] + df['SibSp'] + 1 

    df['Title'] = df["Name"].map(lambda name:name.split(',')[1].split('.')[0].strip())

    titles = {

        'Capt': 6,

        'Col': 6,

        'Don': 3,

        'Dona': 4,

        'Dr': 6,

        'Jonkheer': 3,

        'Lady': 2,

        'Major': 6,

        'Master': 8,

        'Miss': 7,

        'Mlle': 7,

        'Mme': 4,

        'Mr': 5,

        'Mrs': 4,

        'Ms': 7,

        'Rev': 6,

        'Sir': 3,

        'the Countess': 2

    }

    df['Title_val'] = df['Title'].map(lambda title:titles[title])

    df['Title_val'] = df['Title_val'].fillna('Unknown')

    title_cols = pd.get_dummies(df['Title_val'],prefix='Title_')

    df = pd.concat([df,title_cols],axis=1)

    df['Lastname'] = df['Name'].map(lambda name:name.split(',')[0])

    df['Alone'] = df['Family_size'].map(lambda size: 1 if size > 1 else 0)



    #df["Lastname_count"] = 1

    #grouped_lastname = df.groupby('Lastname')['Lastname_count'].transform('count')

    #df['Lastname_count'] = grouped_lastname



    return df

    

def clean_fill_data(df):

    df['Sex_val'] = df['Sex'].apply(lambda sex:1 if sex=='male' else 0)

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    #Fill the missing ages by looking at the age median of people with the same title value

    age_grouped = df.groupby('Title_val')['Age'].transform('median')

    df['Age'] = df['Age'].fillna(age_grouped)

    df['Age'] = df['Age'].fillna(df.Age.mean())

    df['Embarked'] = df['Embarked'].fillna('S')

    df['Embarked_num'] = df['Embarked'].map(lambda emb: 1 if emb == 'S' else 2 if emb == 'C' else 3)

    return df





df_train = generate_new_features(df_train)

df_train = clean_fill_data(df_train)

fig = P.figure(figsize=(18,12), dpi=1600) 





ax0 = P.subplot2grid((2,2),(0,0))

survived_sex = df_train[df_train['Survived']==1]['Sex'].value_counts()

dead_sex = df_train[df_train['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex,dead_sex])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, color=['r','g'], alpha = 0.5, ax=ax0); ax0.set_title('Survival per gender')



ax1= P.subplot2grid((2,2),(0,1))

ax1.hist([df_train[df_train['Survived']==1]['Age'], df_train[df_train['Survived']==0]['Age']], stacked=True, color = ['g','r'],

         bins = 16,label = ['Survived','Dead'], alpha = 0.5); ax1.set_title('Survival per age')

ax1.set_xlabel('Age')

ax1.set_ylabel('Number of passengers')

ax1.legend()



ax2 = P.subplot2grid((2,2),(1,0))

ax2.scatter(df_train[df_train['Survived']==1]['Age'],df_train[df_train['Survived']==1]['Fare'],c='g',s=40, lw=0.5, alpha = 0.5)

ax2.scatter(df_train[df_train['Survived']==0]['Age'],df_train[df_train['Survived']==0]['Fare'],c='r',s=40, lw=0.5 , alpha = 0.5)

ax2.set_xlabel('Age')

ax2.set_ylabel('Fare')

ax2.set_title('Fare per age')

ax2.legend(('survived','dead'),scatterpoints=1,loc='upper right')



ax3 = P.subplot2grid((2,2),(1,1))

ax3.hist([df_train[df_train['Survived']==1]['Pclass'], df_train[df_train['Survived']==0]['Pclass']], stacked=True, color = ['g','r'],

         bins = 3,label = ['Survived','Dead'], alpha = 0.5); ax1.set_title('Survival per age')

ax3.set_xlabel('Class')

ax3.set_ylabel('Number of passengers')

ax3.set_title('Survival per Class')



ax3.legend(('survived','dead'),scatterpoints=1,loc='upper right')



P.show()
def basic_random_forest(df, features = []):

    importance_plot = False

    if len(features) == 0: 

        importance_plot = True

        survived = df['Survived']

        df = df.drop(['Embarked', 'Cabin', 'Title', 'Name', 'Lastname', 'Ticket', 'PassengerId', 'Survived', 'Sex'], axis=1)

        features = df.columns.values

        df['Survived'] = survived

    cls = RandomForestClassifier(n_estimators=100)

    tree_train = train_basic_tree(df, features, cls, importance_plot)

    return {'score' : tree_train[0], 'cls': cls, 'features': tree_train[1].index.values[0:8]}

    

def train_basic_tree(df, features, cls, plot=False):

    kf = StratifiedKFold(df["Survived"], n_folds=5)

    scores = cross_validation.cross_val_score(cls, df[features], df["Survived"], cv=kf)

    print('Estimated score:')

    print(scores.mean())

    predictions = []

    arr_score = []

    

    for train, test in kf:

        train_target = df["Survived"].iloc[train]

        full_test_predictions = []

        cls = cls.fit(df[features].iloc[train,:], train_target)

        test_predictions = cls.predict(df[features].iloc[test,:].astype(float))

        predictions.append(test_predictions)

    

    predictions = np.concatenate(predictions, axis=0)

    

    display(pd.crosstab(predictions,df["Survived"]))

    print(metrics.classification_report(y_true=df["Survived"], y_pred=predictions))

    

    importances = pd.DataFrame({'feature':np.array(features),'importance':cls.feature_importances_})

    importances = importances.sort_values('importance',ascending=False).set_index('feature')

    if plot:

        importances.plot.bar()

        P.show()

    

        

    return (scores.mean()), importances



forest = basic_random_forest(df_train)
features = forest['features']

forest = basic_random_forest(df_train, forest['features'])
def select_features(df, test=False):

    survived = []

    if test == False: 

        survived = df['Survived']

    passengerId = df['PassengerId']

    df = df.drop(['Embarked', 'Cabin', 'Title', 'Name', 'Lastname', 'Ticket', 'PassengerId', 'Survived', 'Sex'], axis=1)

    features = df.columns.values

    selector = SelectKBest(f_classif, k=8)

    selector.fit(df[features].values, survived)



    # Get the raw p-values for each feature, and transform from p-values into scores

    scores = -np.log10(selector.pvalues_)

    

    P.bar(range(len(features)), scores)

    P.xticks(range(len(features)), features, rotation='vertical')

    P.show()

   

    return features[selector.get_support(indices=True)], passengerId, survived
def scale_features(df):

    min_max_scaler = preprocessing.MinMaxScaler()

    df['Age'] = min_max_scaler.fit_transform(df['Age'])

    df['Fare'] = min_max_scaler.fit_transform(df['Fare'])

    df['Pclass'] = min_max_scaler.fit_transform(df['Pclass'])

    df['SibSp'] = min_max_scaler.fit_transform(df['SibSp'])

    df['Parch'] = min_max_scaler.fit_transform(df['Parch'])

    df['Family_size'] = min_max_scaler.fit_transform(df['Family_size'])



    return df



def train_forest(df, features):

    cls = RandomForestClassifier(n_estimators=100, min_samples_split=4, min_samples_leaf=2)

    return [train_general(df, features, cls,), cls]



def train_log_reg(df, features):

    cls = LogisticRegression()

    return [train_general(df, features, cls), cls]



def train_grad_boost(df, features):

    cls = GradientBoostingClassifier(n_estimators=100, max_depth=4)

    return [train_general(df, features, cls), cls]



def train_general(df, features, cls):

    df = scale_features(df)

    print(df.head())

    kf = StratifiedKFold(df["Survived"], n_folds=5)

    scores = cross_validation.cross_val_score(cls, df[features], df["Survived"], cv=kf)

    print('Estimated score:')

    print(scores.mean())



    predictions = []



    for train, test in kf:

        train_target = df["Survived"].iloc[train]

        full_test_predictions = []

        cls = cls.fit(df[features].iloc[train,:], train_target)

        test_predictions = cls.predict(df[features].iloc[test,:].astype(float))

        predictions.append(test_predictions)

    

    predictions = np.concatenate(predictions, axis=0)

    display(pd.crosstab(predictions,df["Survived"]))

    print(metrics.classification_report(y_true=df["Survived"],

                              y_pred=predictions) )

    return (scores.mean())





def test_general(df, features, cls, fn='a'):

    pred = cls.predict(df[features])

    df["Survived"] = df["Survived"].astype(float)



    tp=0

    fp=0

    tp =  len(df[df["Survived"] == pred & (df["Survived"] == 1)])

    fp =  len(df[df["Survived"] != pred & (df["Survived"] == 0)])

    print(tp/float(tp+fp))

    submission = pd.DataFrame({'PassengerId': df['PassengerId'], 'Survived': pred})

    submission.to_csv("kaggle"+fn+".csv", index=False)

    return (pred)

    

df_train = load_training()

df_train = generate_new_features(df_train)

df_train = clean_fill_data(df_train)

selected_data = select_features(df_train)

#selected_data = [features, 0]





df_test = load_testing()

df_test = generate_new_features(df_test)

df_test = clean_fill_data(df_test)



print('Random Forest:')

forest = train_forest(df_train, selected_data[0])

# print len(test_general(df_test, selected_data[0], forest[1],'tree'))



print('Logistic Regression:')

log_reg = train_log_reg(df_train, selected_data[0])

# print len(test_general(df_test, selected_data[0], log_reg[1],'log'))



print('Gradient Boosting:')

grad_boost = train_grad_boost(df_train, selected_data[0])

# print len(test_general(df_test, selected_data[0], grad_boost[1],'boost'))
