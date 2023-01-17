#import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import fancyimpute
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn import tree
from sklearn import neighbors 
from sklearn import naive_bayes
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import clock
from statsmodels.graphics.mosaicplot import mosaic
import warnings
warnings.filterwarnings("ignore")
#import our data
train = pd.read_csv("../input/train.csv", )
test = pd.read_csv("../input/test.csv")
# Columns Age, Cabin and Embarked look like they have some missing data
train.info()
#Let's express the categorical variables we have, namely Sex. This will be usefull for the following section
tempo = pd.get_dummies(train[['Sex']])
train['male'] = tempo['Sex_male']
train['female'] = tempo['Sex_female']
def impute(data):
    """Impute missing values in the Age, Deck, Embarked, and Fare features.
    """
    impute_missing = data.drop(['Survived'], axis = 1)
    impute_missing_cols = list(impute_missing)
    filled_soft = fancyimpute.MICE().complete(np.array(impute_missing))
    results = pd.DataFrame(filled_soft, columns = impute_missing_cols)
    assert results.isnull().sum().sum() == 0, 'Not all NAs removed'
    results['Survived'] = list(data['Survived'])
    return results

def train_test_model(model, hyperparameters, X_train, y_train,folds = 5):
    optimized_model = GridSearchCV(model, hyperparameters, cv = folds)
    optimized_model.fit(X_train, y_train)
    print ('Optimized parameters:', optimized_model.best_params_)
    kfold_score = np.mean(cross_validation.cross_val_score(
            optimized_model.best_estimator_, X_train, y_train, cv = folds, n_jobs = -1))
    print ('Model accuracy ({0}-fold):'.format(str(folds)), kfold_score, '\n')
    return optimized_model

def make_submission_file(filename, predictions):
    results = pd.DataFrame()
    results['Survived'] = [int(i) for i in predictions]
    results['PassengerId'] = np.arange(892, 892+418)
    results.to_csv(filename,index=False)
    
def create_dummy_nans(data, col_name):
    """Create dummies for a column in a DataFrame, and preserve np.nans in their 
    original places instead of in a separate _nan column.
    """
    deck_cols = [col for col in list(data) if col_name in col]
    for deck_col in deck_cols:
        data[deck_col] = np.where(
            data[col_name + 'nan'] == 1.0, np.nan, data[deck_col])
    return data.drop([col_name + 'nan'], axis = 1)
data = train.append(test)
data.drop(["female", "male"], axis=1, inplace=True)
# Here is a pivot table with the Titles and the gender. Note that there should be a 
# high correlation between the two variables but that is ok for now.
titles = []
for name in data.Name:
    titles.append(re.search(r'[A-Z][a-z]+\.', name).group())
data['Title'] = titles
pd.crosstab(data['Sex'], data['Title'],rownames=['Gender'], colnames=['Title'])
titles =  {
    "Capt.": "Esteemed",
    "Col.": "Esteemed",
    "Countess.": "Royalty",
    "Don.":  "Royalty",
    "Dona.": "Esteemed", 
    "Dr.": "Esteemed",
    "Jonkheer.": "Esteemed",
    "Lady.":  "Royalty", 
    "Major.": "Esteemed",
    "Master.": "Master",
    "Miss.": "Miss",
    "Mlle.": "Miss",
    "Mme.": "Mme.",
    "Mr.": "Mr",
    "Mrs.": "Mrs",
    "Ms.": "Miss",
    "Rev.": "Esteemed",
    "Sir.":  "Royalty",
}
data['Title'] = data['Title'].apply(lambda x:titles[x]) 
#Survival in this case does not seem to depend to much on the title. perhaps we should revisit that feature to make 
# it more informative.
pd.crosstab(data['Survived'], data['Title'],rownames=['Survived'], colnames=['Title'])
data['Fsize'] = np.array(data['Parch'])+np.array(data['SibSp']) + 1
# Now let's explore the relation between family size and survival rate
survived = data[data['Survived']==1]
died = data[data['Survived']==0]
died['Died'] = 1
died.drop("Survived", axis=1, inplace=True)
survived = survived.groupby('Fsize')[['Survived']].sum()
died = died.groupby('Fsize')[['Died']].sum()
result = pd.concat([died, survived], axis=1)
result = result.fillna(0)
# convert to proportions
total = np.array(result['Died'])+np.array(result['Survived'])
result['Died'] = np.array(result["Died"])/total
result['Survived'] = np.array(result["Survived"])/total
result.plot(kind='bar')
# Clearly we see that the only types of families that survived are mainly those of 2, 3 and 4 people while others seem
# to have perished
#Hence, we can encode the family size into 3 categorical variables, small, medium and large based on their size
def encode_fsize(x):
    if x==1:
        return 'small'
    elif x<5:
        return 'medium'
    else:
        return 'large'
data['Fsize']=data['Fsize'].apply(encode_fsize)
f, a = mosaic(data, ['Fsize', 'Survived'])
f.set_figheight(10)
f.set_figwidth(10)
data.drop(['SibSp','Parch'], axis=1, inplace=True)
data['Deck'] = data['Cabin'].apply(lambda x: x[0] if type(x)==type('str') else 'UNK')
data.drop('Cabin', axis=1, inplace=True)
data['Ticket'].head()
data['TicketPrefix'] = data["Ticket"].apply(lambda i: i.split(" ")[0] if len(i.split(" "))>=2 else '0')
data['TicketPrefix'] = data['TicketPrefix'].apply(lambda x: re.sub(r'[/\.]', "", x))
data.info()
imputed_data = data.pipe(pd.get_dummies, columns = ['Pclass', 'Fsize', 'Title', 'Ticket']).pipe(pd.get_dummies, 
columns = ['Embarked'], dummy_na = True).pipe(pd.get_dummies, 
columns = ['Deck'], dummy_na = True).assign(Sex = lambda x: np.where(x.Sex == 'male', 1, 0)).pipe(create_dummy_nans,
'Embarked_').drop(['Name', 'PassengerId', 'TicketPrefix'], axis=1).pipe(impute)
# Clearly these people are not babies, and the almost 0 fare is probably due to some error in our data
# Let's replace those people's fare(including those with nan values) with the mean fare of their class 
# and port of embarkation.
free_loaders = data[data['Fare']<5]
free_loaders[['Age', 'Fare', 'Pclass']][:10]
class_port_means = pd.crosstab(data['Pclass'], data['Embarked'], values = data['Fare'], aggfunc='mean',
                               rownames=['Pclass'], colnames=['Title'])
class_port_means
new_fares = []
for i in range(len(data)):
    if np.array(data['Fare'])[i]<5 or np.isnan(np.array(data['Fare'])[i]):
        new_fares.append(class_port_means[np.array(data.Embarked)[i]][np.array(data.Pclass)[i]])
    else:
        new_fares.append(np.array(data['Fare'])[i])
data['Fare'] = new_fares
mean_age = np.mean(data['Age'])
data = data.fillna(mean_age)
data['Embarked'] = data.replace(to_replace=29.881137667304014, value='S')['Embarked']
cols_to_normalize = ["Fare", "Age"]
data['Fare'] = preprocessing.scale(data['Fare'])
data['Age'] = preprocessing.scale(data['Age'])
print("Mean Age is ", np.mean(data['Age']))
print("Mean Fare is ", np.mean(data['Fare']))
print("SD Age is ", np.std(data['Age']))
print("SD Fare is ", np.std(data['Fare']))
train = data[np.logical_or(data['Survived']==1, data['Survived']==0)]
vars = ['Sex', 'Pclass', 'Embarked']
# Survival based on gender
table = train.groupby(['Sex', 'Survived']).size()
table = table.unstack() #splits the data into 2 columns, 0, 1, each indexed by the
                                    #other variable
normedtable = table.div(table.sum(1), axis=0) #divides the counts by the totals
normedtable.plot(kind='barh', stacked=True, colormap='brg')
#Survival based on passenger class
# Survival based on gender
table = train.groupby(['Pclass', 'Survived']).size()
table = table.unstack() #splits the data into 2 columns, 0, 1, each indexed by the
                                    #other variable
normedtable = table.div(table.sum(1), axis=0) #divides the counts by the totals
normedtable.plot(kind='barh', stacked=True, colormap='brg')
#Survival based on the port of embarkation
# Survival based on gender
table = train.groupby(['Embarked', 'Survived']).size()
table = table.unstack() #splits the data into 2 columns, 0, 1, each indexed by the
                                    #other variable
normedtable = table.div(table.sum(1), axis=0) #divides the counts by the totals
normedtable.plot(kind='barh', stacked=True, colormap='brg')
# specifies the parameters of our graphs
fig = plt.figure(figsize=(18,6), dpi=1600) 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55
plt.subplot2grid((2,3),(1,0), colspan=2)
# plots a kernel density estimate of the subset of the 1st class passangers's age
data.Age[data.Pclass == 1].plot(kind='kde')    
data.Age[data.Pclass == 2].plot(kind='kde')
data.Age[data.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')
to_be_replaced = ['Sex', "Embarked", 'Deck', 'TicketPrefix', 'Fsize', 'Title']
added_cols = pd.get_dummies(data[to_be_replaced])
added_cols['PassengerId'] = data['PassengerId']
data = pd.merge(data, added_cols, on='PassengerId')
tmp = pd.get_dummies(data['Pclass'])
tmp.columns=["Class 1", "Class 2", "Class 3"]
tmp['PassengerId'] = data['PassengerId']
data.drop(to_be_replaced, axis=1, inplace=True)
data.drop("Pclass", axis=1, inplace=True)
data = pd.merge(data, tmp, on='PassengerId')
train = data[:len(train)]
test = data[len(train):]
X=train.drop(['Ticket', 'Name', 'PassengerId', 'Survived'], axis=1).as_matrix()
Y= np.array(train['Survived'])
#At this point we will define a utility function which will abstract away some of the code for us
def report(clf, X, Y):
    start=clock()
    predicted = cross_validation.cross_val_predict(clf, X, Y, cv=10)
    end = clock()
    print("Accuracy: ", metrics.accuracy_score(Y, predicted))
    print("Recall: ", metrics.recall_score(Y, predicted))
    print("Precision: ", metrics.precision_score(Y, predicted))
    print("F1: ", metrics.f1_score(Y, predicted))
    print("Time elapsed: ", end-start)
# Now that everything is set, use cross validation to test different learning algorithms and see how they perform
print('---------------------------------')
print('Logistic Regression')
clf = linear_model.LogisticRegression()
report(clf, X, Y)
print('---------------------------------')
print('SVM')
clf = svm.SVC()
report(clf, X, Y)
print('---------------------------------')
print('Random Forest')
clf = ensemble.RandomForestClassifier()
report(clf, X, Y)
print('---------------------------------')
print('Naive Bayes')
clf = naive_bayes.BernoulliNB()
report(clf, X, Y)
print('---------------------------------')
print('Decision Tree Classifier')
clf = tree.DecisionTreeClassifier()
report(clf, X, Y)