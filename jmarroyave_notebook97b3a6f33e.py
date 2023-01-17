# Imports

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.externals import joblib

from sklearn import tree

from sklearn import cross_validation

from sklearn import preprocessing

from sklearn.cross_validation import KFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif

from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.cross_validation import StratifiedKFold



# Configuraciones

matplotlib.style.use('ggplot')

%matplotlib inline

pd.options.display.max_columns = 100

pd.options.display.max_rows = 100



survivedColors = ['r','g']

figsize = (14,7)



def showGeneralGraph(data,col,vrs) :  

    fig = matplotlib.pyplot.gcf()

    fig.set_size_inches((14,5), forward=True)



    plt.subplot(1,3,1)

    d = [data[col].notnull().sum(), data[col].isnull().sum()]

    plt.pie(d, labels=['','NULL'], autopct='%1.1f%%', shadow=True, startangle=0)

    

    plt.subplot(1,3,2)

    d = data[col].value_counts().sort_index();

    plt.pie(d, labels=d.index.tolist(), autopct='%1.1f%%', shadow=True, startangle=0)



    ax = plt.subplot(1,3,3)

    ct = pd.crosstab(data[col], data[vrs])

    r = ct.plot.bar(color=survivedColors, stacked=True, ax=ax)

    autolabel(r)

    plt.legend(('Dead', 'Survived'),loc='best') ;



    plt.show()

    

def autolabel(r):

    """

    Attach a text label above each bar displaying its percentage

    """

    lines, labels = r.get_legend_handles_labels()

    k = len(labels)

    n = int(len(r.patches) / k)

    for x in range(0, n) :

        tot = 0

        for y in range(0, k) :

            tot += r.patches[x + (n * y)].get_height()

        if tot == 0 :

            continue

        for y in range(0, k) :

            rect = r.patches[x + (n * y)]

            height = rect.get_height()

            per = round((height / tot) * 100, 2)

            r.text(rect.get_x() + rect.get_width() / 2., rect.get_y() + height / 2., per, ha='center', va='bottom')    
from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

pre{

    padding-bottom: 10px;

}

</style>

""")
train_url = "../input/train.csv"

train = pd.read_csv(train_url)



test_url = "../input/test.csv"

test = pd.read_csv(test_url)
train.shape
eda = train.copy()

eda = eda.drop('PassengerId', 1)
eda.head(7).T
fig = matplotlib.pyplot.gcf()

fig.set_size_inches(figsize, forward=True)



print(eda[['Sex','Embarked','Name','Ticket','Cabin', 'Fare']].isnull().sum())



# Categorical Columns

# Small

plt.subplot(1,3,1)

cat_columns = ['Sex', 'Embarked']

cols = eda[cat_columns].apply(pd.Series.nunique)

cols.plot(kind='bar')

# Id

plt.subplot(1,3,2)

cat_columns = ['Name']

cols = eda[cat_columns].apply(pd.Series.nunique)

cols.plot(kind='bar')



# Unknown

plt.subplot(1,3,3)

cat_columns = ['Ticket','Cabin', 'Fare']

cols = eda[cat_columns].apply(pd.Series.nunique)

cols.plot(kind='bar')
eda.describe()
eda.hist(bins=10,figsize=figsize,grid=False)
showGeneralGraph(eda, 'Sex', 'Survived')
def featAgeRange(data) :

    bins = np.array([-1,0,1,4,12,19,30,50,60,100])

    data['AgeRange'] = pd.cut(data.Age, bins)



featAgeRange(eda)

showGeneralGraph(eda, 'AgeRange', 'Survived')
fig = matplotlib.pyplot.gcf()

fig.set_size_inches(figsize, forward=True)



plt.subplot(2,2,1)

x = np.sort(eda.Age)

y = np.arange(1, len(x) + 1) / len(x)

plt.plot(x, y, marker='.', linestyle='none')

plt.xlabel("% of people with age")

plt.ylabel("ECDF")

plt.margins(0.02)



plt.subplot(2,2,2)

plt.scatter(eda[eda['Survived']==1]['Age'],eda[eda['Survived']==1]['Fare'],c='green',s=40, alpha=0.4)

plt.scatter(eda[eda['Survived']==0]['Age'],eda[eda['Survived']==0]['Fare'],c='red',s=40,  alpha=0.4)

plt.xlabel('Age')

plt.ylabel('Fare')

plt.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=20,)





plt.subplot(2,2,3)

_ = sns.swarmplot(x='Survived', y='Age', data=eda)

plt.xlabel('Survived')

plt.ylabel('Age')





plt.show()
def featTitle(data) :

    data['Title'] = data.SibSp + data.Parch



    data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    Title_Dictionary = {

                "Capt":       "Officer",

                "Col":        "Officer",

                "Major":      "Officer",

                "Jonkheer":   "Royalty",

                "Don":        "Royalty",

                "Sir" :       "Royalty",

                "Dr":         "Officer",

                "Rev":        "Officer",

                "the Countess":"Royalty",

                "Dona":       "Royalty",

                "Mme":        "Mrs",

                "Mlle":       "Miss",

                "Ms":         "Mrs",

                "Mr" :        "Mr",

                "Mrs" :       "Mrs",

                "Miss" :      "Miss",

                "Master" :    "Master",

                "Lady" :      "Royalty"

    }



    data.Title = data.Title.map(Title_Dictionary)



featTitle(eda)

showGeneralGraph(eda, 'Title', 'Survived')
def featNameLength(data) :

    data["NameLength"] = data["Name"].apply(lambda x: len(x))

    bins = [0, 20, 40, 57, 85]

    group_names = ['short', 'okay', 'good', 'long']

    data['NlengthD'] = pd.cut(data['NameLength'], bins, labels=group_names)



featNameLength(eda)

showGeneralGraph(eda, 'NlengthD', 'Survived')
eda[(eda.Name.str.contains('Mr.')) & (eda.Parch > 0) & (eda.Sex == 'male')].head(10)
def featSibSpRange(data) :

    bins = np.array([-1,0,1,3,10])

    data['SibSpRange'] = pd.cut(data.SibSp, bins)



featSibSpRange(eda)

showGeneralGraph(eda, 'SibSpRange', 'Survived')
def featParchRange(data) :

    bins = np.array([-1,0,1,3,10])

    data['ParchRange'] = pd.cut(data.SibSp, bins)



featParchRange(eda)

showGeneralGraph(eda, 'ParchRange', 'Survived')
# Same Ticket?

def featTicketCount(data) :

    j = data.groupby(["Ticket"]).size().reset_index(name='TicketCount')

    return data.join(j.set_index('Ticket'), on='Ticket')



eda = featTicketCount(eda)
eda[eda.Ticket == '113803']
def featIsWithFamily(data) :

    data['IsWithFamily'] = 1

    data.loc[(data.Parch == 0) & (data.SibSp == 0), ['IsWithFamily']] = 0



featIsWithFamily(eda)
showGeneralGraph(eda, 'IsWithFamily', 'Survived')
def featIsAlone(data) :

    data['IsAlone'] = 0

    data.loc[(data.TicketCount == 1) & (data.IsWithFamily == 0), ['IsAlone']] = 1



featIsAlone(eda)
showGeneralGraph(eda, 'IsAlone', 'Survived')
def featGroup(data) :   

    # Group

    data['FamilySize'] = data.SibSp + data.Parch + 1

    data.loc[data["FamilySize"] == 1, "FsizeD"] = 'singleton'

    data.loc[(data["FamilySize"] > 1)  &  (data["FamilySize"] < 5) , "FsizeD"] = 'small'

    data.loc[data["FamilySize"] >4, "FsizeD"] = 'large'

    

featGroup(eda)
showGeneralGraph(eda, 'FamilySize', 'Survived')
showGeneralGraph(eda, 'FsizeD', 'Survived')
def featIsFather(data) : 

    data['IsFather'] = 0

    data.loc[(data.Name.str.contains('Mr.')) & (data.Parch > 0) & (data.Sex == 'male'), ['IsFather']] = 1



featIsFather(eda)
showGeneralGraph(eda, 'IsFather', 'Survived')
### Is Mother
def featIsMother(data) :

    data['IsMother'] = 0

    data.loc[(data.Name.str.contains('Mrs.')) & (data.Parch > 0) & (data.Sex == 'female'), ['IsMother']] = 1

    

featIsMother(eda)
showGeneralGraph(eda, 'IsMother', 'Survived')
showGeneralGraph(eda, 'Pclass', 'Survived')
def featFareRange(data):

    bins = np.array([-1,0,8,12.5,25,50,100,600])

    data['FareRange'] = pd.cut(data.Fare, bins)



featFareRange(eda)

showGeneralGraph(eda, 'FareRange', 'Survived')
fig = matplotlib.pyplot.gcf()

fig.set_size_inches(figsize, forward=True)



plt.subplot(1,2,1)

x = np.sort(eda.Fare)

y = np.arange(1, len(x) + 1) / len(x)

plt.plot(x, y, marker='.', linestyle='none')

plt.xlabel("% of people with age")

plt.ylabel("ECDF")

plt.margins(0.02)



plt.subplot(1,2,2)

_ = sns.swarmplot(x='Survived', y='Fare', data=eda)

# Label the axes

plt.xlabel('Survived')

plt.ylabel('Fare')



plt.show()
# Relation with SibSp and Parch

g = sns.factorplot(x="Fare", y="SibSp",

                    row="Pclass",

                    data=eda[eda.Embarked.notnull()],

                    orient="h",  aspect=3.5,                  

                    split=True, cut=0, bw=.2);
def featFarePerTicket(data):

    data['FarePerTicket'] = data.Fare / data.TicketCount



featFarePerTicket(eda)
def featFarePerTicketRange(data):

    bins = np.array([-1,5,7.5,10,15,25,35,600])

    data['FarePerTicketRange'] = pd.cut(data.FarePerTicket, bins)



featFarePerTicketRange(eda)

showGeneralGraph(eda, 'FarePerTicketRange', 'Survived')
showGeneralGraph(eda, 'Embarked', 'Survived')
def featDeck(data) :

    def cleanCabin(ticket):

        if(type(ticket) == str):          

            if (len(ticket)) > 0:

                return ticket[0]

        else: 

            return 'U'



    data['Deck'] = data.Cabin.map(cleanCabin)



    Deck_Dictionary = {

        'U':'U',

        'A':'T',

        'B':'B',

        'C':'C',

        'D':'D',

        'E':'E',

        'F':'T',

        'G':'T',

        'T':'T'

    }

    data.Deck = data.Deck.map(Deck_Dictionary)



featDeck(eda)

showGeneralGraph(eda, 'Deck', 'Survived')
prep = eda.copy()



pd.DataFrame(prep.columns).T
def getColumnsToReview(data) :

    cols = pd.DataFrame(data.isnull().sum())

    return cols.loc[cols[0] > 0, :]



getColumnsToReview(prep)
def imputeMissingData(data) : 

    data["Fare"].fillna(data["Fare"].median(), inplace=True)

    featFarePerTicket(data)

    data["Cabin"].fillna('U', inplace=True)

    data["Embarked"].fillna('C', inplace=True)

    

imputeMissingData(prep)
def convertData(data) :

    labelEnc = LabelEncoder()

    cat_vars = ['Embarked','Sex',"Title","FsizeD","NlengthD",'Deck','Ticket','Cabin']

    for col in cat_vars:

        data[col] = labelEnc.fit_transform(data[col])



convertData(prep)
def inferMissing(data):

    def fill_missing_age(df):

        #Feature set

        age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',

                     'Title','Pclass','FamilySize',

                     'FsizeD','NameLength',"NlengthD",'Deck']]

        # Split sets into train and test

        tK  = age_df.loc[ (df.Age.notnull()) ]# known Age values

        tU = age_df.loc[ (df.Age.isnull()) ]# null Ages



        if len(tU) == 0 :

            return 0



        # All age values are stored in a target array

        y = tK.values[:, 0]



        # All the other values are stored in the feature array

        X = tK.values[:, 1::]



        # Create and fit a model

        rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

        rtr.fit(X, y)



        # Use the fitted model to predict the missing values

        predictedAges = rtr.predict(tU.values[:, 1::])



        # Assign those predictions to the full data set

        df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 



        return len(tU)



    fill_missing_age(data)

    featAgeRange(data)



inferMissing(prep)
getColumnsToReview(prep)
def featScaling(data) :

    data[['Age', 'Fare']] = preprocessing.normalize(data[['Age', 'Fare']])



featScaling(prep)
def getCorrelation(data) :

    corr=data.corr()

    plt.figure(figsize=figsize)



    sns.heatmap(corr, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='YlGnBu',linecolor="white")

    plt.title('Correlation between features');

    

getCorrelation(prep)
prep.corr()["Survived"]
pd.DataFrame(prep.columns).T
def settingDataset(data) : 

    data = featTicketCount(data)

    featIsMother(data)

    featIsFather(data)

    featFarePerTicket(data)

    featGroup(data)

    featIsWithFamily(data)

    featIsAlone(data)

    featTitle(data)

    featNameLength(data)

    featDeck(data)

    

    getColumnsToReview(data)

    imputeMissingData(data)

    convertData(data)

    inferMissing(data)

    data['AgeRange'] = LabelEncoder().fit_transform(data['AgeRange'])

    featScaling(data)    

    return data
train2 = settingDataset(train.copy())
test2 = settingDataset(test.copy())
test2.T
cols = ['Pclass', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'TicketCount',

       'IsMother', 'IsFather', 'FarePerTicket', 'FamilySize', 'FsizeD',

       'IsWithFamily', 'IsAlone', 'Title', 'NameLength', 'NlengthD', 'Deck','AgeRange']



F = train2[cols]

Fx = train2.Survived



# feature extraction

model = ExtraTreesClassifier()

model.fit(F, Fx)

print(model.feature_importances_)

pd.DataFrame({'rank':model.feature_importances_, 'cols':cols}).sort_values(['rank'], ascending=[False])
predictors = ['PassengerId','Sex','Age','Ticket','Fare','NameLength','Title','FarePerTicket','Pclass']

target = 'Survived'
X = train2[predictors]

y = train2[target]

TK = test2[predictors]

TK = TK.drop('PassengerId', 1)

ids = test2['PassengerId']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_test, random_state=23)



ids_train = X_train['PassengerId']

X_train = X_train.drop('PassengerId', 1)



ids_test = X_test['PassengerId']

X_test = X_test.drop('PassengerId', 1)



X = X.drop('PassengerId', 1)
forest = RandomForestClassifier(max_features='sqrt')



parameters = {'n_estimators':      [15, 20, 30],

              'max_features':      ['log2', 'sqrt','auto'], 

              'criterion':         ['entropy', 'gini'],

              'max_depth':         [11, 12, 14], 

              'min_samples_split': [7, 8, 9],

              'min_samples_leaf':  [2, 3, 4]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)

grid_search = GridSearchCV(forest,param_grid=parameters,scoring=acc_scorer)

grid_search = grid_search.fit(X_train, y_train)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))



# Fit the best algorithm to the data. 

forest = grid_search.best_estimator_

forest.fit(X_train, y_train)



predictions = forest.predict(X_test)

print(accuracy_score(y_test, predictions))
lr = LogisticRegression(penalty='l2')



parameters = {

                 'tol' :     [0.00002, 0.00005, 0.00008],

                 'max_iter': [90, 100, 120],

            }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)

grid_search = GridSearchCV(lr,param_grid=parameters,scoring=acc_scorer)

grid_search = grid_search.fit(X_train, y_train)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))



# Fit the best algorithm to the data. 

lr = grid_search.best_estimator_

lr.fit(X_train, y_train)



predictions = lr.predict(X_test)

print(accuracy_score(y_test, predictions))
adb = AdaBoostClassifier()



parameters = {

             'n_estimators': [100,140,155]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)

grid_search = GridSearchCV(adb,param_grid=parameters,scoring=acc_scorer)

grid_search = grid_search.fit(X_train, y_train)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))



# Fit the best algorithm to the data. 

adb = grid_search.best_estimator_

adb.fit(X_train, y_train)



predictions = lr.predict(X_test)

print(accuracy_score(y_test, predictions))
eclf1 = VotingClassifier(estimators=[

        ('lr', lr), ('rf', forest), ('adb', adb)], voting='soft')



eclf1 = eclf1.fit(X_train, y_train)



pred = eclf1.predict(np.array(X_test))

print(accuracy_score(y_test, pred))



pred = eclf1.predict(np.array(TK))

pred = pd.DataFrame(np.array(pred).astype(int), ids, columns = ["Survived"])

pred.to_csv('pred.csv')
def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X.values[train_index], X.values[test_index]

        y_train, y_test = y.values[train_index], y.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 

    

run_kfold(eclf1)