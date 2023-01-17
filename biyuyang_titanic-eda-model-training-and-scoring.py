dfTrain = pd.read_csv('/kaggle/input/titanic/train.csv')
dfTest = pd.read_csv('/kaggle/input/titanic/test.csv')
print(dfTrain.shape)
print(dfTest.shape)
print('There are {t} passengers in train data, of which {s} made it and {d} did not'.format(t = dfTrain.PassengerId.nunique(), s = dfTrain[dfTrain['Survived'] == 1].PassengerId.nunique(), d = dfTrain[dfTrain['Survived'] == 0].PassengerId.nunique()))
dfTrain.head()
dfTest.head()
print(dfTrain.columns)
dfTrain.describe()
# Turn Pclass into a cateogrical variable
dfTrain['Pclass'] = dfTrain.Pclass.astype('category')
# Show missing value counts by columns
dfTrain.isnull().sum()
dfTrain.groupby(['Sex', 'Pclass']).agg({'Age': ['mean', 'median', 'min', 'max']}).reset_index()
# assign passengers with missing age the average age of gender and class
avgAge = dfTrain.groupby(['Sex', 'Pclass']).Age.mean().reset_index()
dfTrain = dfTrain.merge(avgAge, 'left', on = ['Sex', 'Pclass'], suffixes = ('_orig', '_avg'))
dfTrain.head(5)
# Fill in average age if original age value is NaN
def fill_in_avg_age(row):
    if np.isnan(row['Age_orig']):
        return row['Age_avg']
    else:
        return row['Age_orig']

dfTrain['Age'] = dfTrain.apply(lambda row: fill_in_avg_age(row), axis = 1) 
# Double check if things are filled in correctly
dfTrain[dfTrain.Age_orig.isnull()].head()
print(dfTrain.Cabin.unique())
print('There are {n} different unique Cabin values'.format(n = dfTrain.Cabin.nunique()))
dfTrain[dfTrain.Cabin.isnull()].groupby('Pclass').PassengerId.nunique()
# Fill unknown Cabin with 'U'
dfTrain.loc[dfTrain.Cabin.isnull(), 'Cabin'] = 'U'

# extract first letter of Cabin to indicate location on the ship
dfTrain['CabinLoc'] = dfTrain['Cabin'].str[0]

# assign an indicator to show if the passenger has Cabin assignment or not
def assign_cabin_ind(row):
    if row['Cabin'] == 'U':
        return 0
    else:
        return 1

dfTrain['CabinInd'] = dfTrain.apply(lambda x: assign_cabin_ind(x), axis = 1).astype('category')
print(dfTrain.groupby('CabinLoc').PassengerId.nunique().reset_index())
print(dfTrain.groupby(['Pclass', 'CabinInd']).PassengerId.nunique().reset_index())
print(dfTrain.groupby(['Pclass', 'CabinInd']).agg({'PassengerId': 'count', 'Fare': 'mean'}).reset_index())
dfTrain['CabinInd'] = dfTrain['CabinInd'].astype('category')
# assign embark based on people's fare
dfTrain.groupby(['Embarked']).agg({'Fare': ['min', 'mean', 'max']}).reset_index()
dfTrain['CLow'] = dfTrain[dfTrain['Embarked'] == 'C'].Fare.quantile(0.15)
dfTrain['CHigh'] = dfTrain[dfTrain['Embarked'] == 'C'].Fare.quantile(0.85)
dfTrain['QLow'] = dfTrain[dfTrain['Embarked'] == 'Q'].Fare.quantile(0.15)
dfTrain['QHigh'] = dfTrain[dfTrain['Embarked'] == 'Q'].Fare.quantile(0.85)
dfTrain['SLow'] = dfTrain[dfTrain['Embarked'] == 'S'].Fare.quantile(0.15)
dfTrain['SHigh'] = dfTrain[dfTrain['Embarked'] == 'S'].Fare.quantile(0.85)
dfTrain.loc[dfTrain.Embarked.isnull(), 'Embarked'] = 'U'
def assign_missing_embarked(row):
    if row['Embarked'] != 'U':
        return row['Embarked']
    else:
        if row['Fare'] <= row['CHigh'] and row['Fare'] >= row['CLow']:
            return 'C'
        elif row['Fare'] <= row['SHigh'] and row['Fare'] >= row['SLow']:
            return 'S'
        elif row['Fare'] <= row['QHigh'] and row['Fare'] >= row['QLow']:
            return 'Q'
        else:
            return 'U'

dfTrain['Embarked_clean'] = dfTrain.apply(lambda x: assign_missing_embarked(x), axis = 1)
dfTrain.loc[dfTrain['Embarked'] == 'U', ['Embarked', 'Embarked_clean']]
dfTrain = dfTrain.drop(columns = ['Age_orig', 'Embarked', 'Age_avg', 'Cabin', 'CLow', 'CHigh', 'QLow', 'QHigh', 'SLow', 'SHigh'])
dfTrain = dfTrain.rename(columns = {'Embarked_clean': 'Embarked'})
print(dfTrain.columns)
print(dfTrain.isnull().sum())
print(dfTrain.shape)
# code to extract titles from names
def extract_title(row):
    return row['Name'].split(',')[1].split('.')[0].strip()

dfTrain['title'] = dfTrain.apply(lambda x: extract_title(x), axis = 1)
dfTrain.groupby('title').PassengerId.nunique()
# categorize titles into Military, Religion, Noble and Civilian
def categorize_titles(row):
    if row['title'] in ['Capt', 'Col', 'Major']:
        return 'Military'
    elif row['title'] in ['Rev', 'Dr']:
        return 'Religion'
    elif row['title'] in ['Don', 'Dona', 'Jonkheer', 'Lady', 'Master', 'Sir', 'the Countess']:
        return 'Noble'
    else:
        return 'Civilian'

dfTrain['TitleCate'] = dfTrain.apply(lambda x: categorize_titles(x), axis = 1)
dfTrain.groupby('TitleCate').PassengerId.count()
import re
def extract_names(row):
    if row['title'] in ['Mrs', 'the Countess']:
        s = row['Name'].split(',')[1]
        return re.sub('^.*\((.*?)\)[^\(]*$', '\g<1>', s)
    else:
        return row['Name'].split(',')[1].split('.')[1].strip() + ' ' + row['Name'].split(',')[0]
        

dfTrain['RealName'] = dfTrain.apply(lambda x: extract_names(x), axis = 1)
dfTrain.head()
# for Mrs's, extract their husbands name and create a list of husband names
def extract_husband_name(row):
    if row['title'] == 'Mrs':
        return row['Name'].split(',')[1].split('.')[1].split('(')[0].strip() + ' ' + row['Name'].split(',')[0].strip()
    else:
        return 'Unknown'
    
dfTrain['HusbandName'] = dfTrain.apply(lambda x: extract_husband_name(x), axis = 1)
husband_list = dfTrain[dfTrain['HusbandName'] != 'Unknown'].HusbandName.tolist()
name_list = dfTrain['RealName'].tolist()
dfTrain.head()

def assign_couple_onboard_ind(row):
    if row['title'] == 'Mrs':
        if row['HusbandName'] in name_list:
            return 1
        else:
            return 0
    else:
        if row['RealName'] in husband_list:
            return 1
        else:
            return 0

dfTrain['CoupleOnboardInd'] = dfTrain.apply(lambda x: assign_couple_onboard_ind(x), axis = 1)
dfTrain['CoupleOnboardInd'] = dfTrain['CoupleOnboardInd'].astype('category')
dfTrain.groupby(['CoupleOnboardInd']).PassengerId.count()
dfTrain.head()
dfTrain = dfTrain.drop(columns = ['Name', 'RealName', 'HusbandName', 'Ticket'])
dfTrain['TravelCompanionSize'] = dfTrain['SibSp'] + dfTrain['Parch']
def travel_companions(row):
    if row['TravelCompanionSize'] == 0:
        return 'Single Traveler'
    elif row['TravelCompanionSize'] <= 4:
        return 'Small Travel Group'
    else:
        return 'Big Travel Group'

def family_size(row):
    if row['Parch'] == 0:
        return 'Not with family'
    elif row['Parch'] <= 4:
        return 'Small family'
    else:
        return 'Big family'

dfTrain['TravelType'] = dfTrain.apply(lambda x: travel_companions(x), axis = 1)
dfTrain['FamilySize'] = dfTrain.apply(lambda x: family_size(x), axis = 1)
dfTrain.head()
import matplotlib.pyplot as plt
import seaborn as sns
f, axes = plt.subplots(3, 2, figsize = (15, 20))
sns.countplot(x = 'Pclass', hue = 'Sex', data = dfTrain, ax = axes[0, 0])
sns.boxplot(x = "Pclass", y = "Age", hue = "Sex", data = dfTrain, ax = axes[0, 1])
sns.stripplot(x = 'Pclass', y = 'SibSp', jitter = False, data = dfTrain, ax = axes[1, 0])
sns.stripplot(x = 'Pclass', y = 'Parch', jitter = False, data = dfTrain, ax = axes[1, 1])
sns.boxplot(x = 'Pclass', y = 'Fare', data = dfTrain, ax = axes[2, 0])
sns.boxplot(x = 'Embarked', y = 'Fare', data = dfTrain, ax = axes[2, 1])
f.show()
f, axes = plt.subplots(2, 4, figsize = (30, 15))
sns.countplot(x = 'Sex', hue = 'Survived', data = dfTrain, ax = axes[0, 0])
sns.countplot(x = 'Pclass', hue = 'Survived', data = dfTrain, ax = axes[0, 1])
sns.boxplot(x = "Survived", y = "Age", hue = "Sex", data = dfTrain, ax = axes[0, 2])
sns.countplot(y = 'CabinLoc', hue = 'Survived', order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'], data = dfTrain, ax = axes[0, 3])
sns.countplot(x = 'TravelType', hue = 'Survived', data = dfTrain, ax = axes[1, 0])
sns.countplot(x = 'FamilySize', hue = 'Survived', data = dfTrain, ax = axes[1, 1])
sns.countplot(x = 'CoupleOnboardInd', hue = 'Survived', data = dfTrain, ax = axes[1, 2])
sns.countplot(x = 'TitleCate', hue = 'Survived', data = dfTrain, ax = axes[1, 3])
f.show()
dfTest.describe()
dfTest.isnull().sum()
 # data processing function to feature engineer test data
def feature_engineering(dat):
    df = dat.copy()
    
    # fill missing Age
    avgAge = df.groupby(['Sex', 'Pclass']).Age.mean().reset_index()
    df = df.merge(avgAge, 'left', on = ['Sex', 'Pclass'], suffixes = ('_orig', '_avg'))
    df['Age'] = df.apply(lambda row: fill_in_avg_age(row), axis = 1) 
    
    # fill missing Fare
    avgFare = df.groupby('Embarked').Fare.mean().reset_index()
    df = df.merge(avgFare, 'left', on = 'Embarked', suffixes = ('_orig', '_avg'))
    def fill_fare(row):
        if np.isnan(row['Fare_orig']):
            return row['Fare_avg']
        else:
            return row['Fare_orig']
    df['Fare'] = df.apply(lambda x: fill_fare(x), axis = 1)
    
    # fill missing Cabin
    df.loc[df.Cabin.isnull(), 'Cabin'] = 'U'
    df['CabinLoc'] = df['Cabin'].str[0]
    df['CabinInd'] = df.apply(lambda x: assign_cabin_ind(x), axis = 1).astype('category')
    
    # add feature engineered columns
    df['title'] = df.apply(lambda x : extract_title(x), axis = 1)
    df['TitleCate'] = df.apply(lambda x: categorize_titles(x), axis = 1)
    df['RealName'] = df.apply(lambda x: extract_names(x), axis = 1)
    df['HusbandName'] = df.apply(lambda x: extract_husband_name(x), axis = 1)
    husband_list = df[df['HusbandName'] != 'Unknown'].HusbandName.tolist()
    name_list = df['RealName'].tolist()
    df['CoupleOnboardInd'] = df.apply(lambda x: assign_couple_onboard_ind(x), axis = 1)
    
    # final clean-up
    df['Pclass'] = df['Pclass'].astype('category')
    df['CabinInd'] = df['CabinInd'].astype('category')
    df['CoupleOnboardInd'] = df['CoupleOnboardInd'].astype('category')
    df = df.drop(columns = ['Age_orig', 'Age_avg', 'Fare_orig', 'Fare_avg', 'Name', 'RealName', 'HusbandName', 'Ticket'])
    
    return df
    
dfTestClean = feature_engineering(dfTest)
dfTestClean.head()
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import time as t
colID = ['PassengerId']
colLabel = ['Survived']
colNum = ['Age', 'SibSp', 'Parch', 'Fare']
colCat = ['Sex', 'Pclass', 'CabinLoc', 'CabinInd', 'Embarked', 'title', 'TitleCate', 'CoupleOnboardInd']
y = dfTrain['Survived'].astype('category')
X = dfTrain[colNum + colCat]
XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size = 0.15, random_state = 777, stratify = y)
# Center and scale numeric variables and one hot coding for categorical variables
# train encoders on training data and apply it on validation and test data
scaler = StandardScaler().fit(XTrain[colNum])
encoder = OneHotEncoder(handle_unknown = 'ignore').fit(XTrain[colCat])
def apply_scaler_encoder(dat):
    
    df = dat.copy()
    print('Shape of original data')
    print(df.shape)
    dfScaled = scaler.transform(df[colNum])
    dfEncoded = encoder.transform(df[colCat]).toarray()
    dfFinal = np.concatenate([dfScaled, dfEncoded], axis = 1)
    print('Shape of processed data')
    print(dfFinal.shape)
    
    return dfFinal
XTrainFinal = apply_scaler_encoder(XTrain)
XValidFinal = apply_scaler_encoder(XValid)
modelsToFit = {
    'Logistic Regression': LogisticRegression(random_state = 777),
    'SVM': SVC(random_state = 777, probability = True),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state = 777),
    'Random Forest': RandomForestClassifier(random_state = 777),
    'AdaBoost': AdaBoostClassifier(random_state = 777),
    'GBT': GradientBoostingClassifier(random_state = 777),
    'XGB': XGBClassifier(random_state = 777)
}



def batch_fit_models(xT, yT, xV, yV, models):

    # initiate a dictionary to record model results
    resultCols = [
        'Model', 'Train Time', 
        'Train Accuracy', 'Validation Accuracy',
        'Train Precision', 'Validation Precision',
        'Train Recall', 'Validation Recall',
        'Train f1', 'Validation f1',
        'Train AUC', 'Validation AUC'
    ]

    result = dict([(key, []) for key in resultCols])
    
    # batch train models
    for model_name, model in models.items():
        
        result['Model'].append(model_name)
        
        # train model and record time laps
        trainStart = t.process_time()
        fit = model.fit(xT, yT)
        trainEnd = t.process_time()
        
        # back fit the model on train data
        predLabelTrain = fit.predict(xT)
        predScoreTrain = fit.predict_proba(xT)[:,1]
        
        # fit the model on validation data
        predLabel = fit.predict(xV)
        predScore = fit.predict_proba(xV)[:,1]
        
        # create data for result dict
        result['Train Time'].append(trainEnd - trainStart)
        result['Train Accuracy'].append(accuracy_score(yT, predLabelTrain))
        result['Validation Accuracy'].append(accuracy_score(yV, predLabel))
        result['Train Precision'].append(precision_score(yT, predLabelTrain))
        result['Validation Precision'].append(precision_score(yV, predLabel))
        result['Train Recall'].append(recall_score(yT, predLabelTrain))
        result['Validation Recall'].append(recall_score(yV, predLabel))
        result['Train f1'].append(f1_score(yT, predLabelTrain))
        result['Validation f1'].append(f1_score(yV, predLabel))
        result['Train AUC'].append(roc_auc_score(yT, predScoreTrain))
        result['Validation AUC'].append(roc_auc_score(yV, predScore))
        
    # turn result dict into a df
    dfResult = pd.DataFrame.from_dict(result)
    
    return dfResult
batch_fit_models(XTrainFinal, yTrain, XValidFinal, yValid, modelsToFit).sort_values(by = 'Validation AUC', ascending = False)
svmFit = modelsToFit['SVM'].fit(XTrainFinal, yTrain)
svmFit.get_params()
svmPredLabel = svmFit.predict(XValidFinal)
svmPredScore = svmFit.predict(XValidFinal)
print(classification_report(yValid, svmPredLabel))
confusion_matrix(yValid, svmPredLabel)
from sklearn.model_selection import GridSearchCV
paramGrid = {
    'kernel': ('linear', 'rbf', 'poly'),
    'gamma': ('auto', 'scale'),
    'C': [0.1, 0.5, 1],
    'degree': [3, 5]
}


svcTune = GridSearchCV(
    estimator = modelsToFit['SVM'],
    param_grid = paramGrid,
    scoring = 'accuracy'
)

svcCVResults = svcTune.fit(XTrainFinal, yTrain)
print('Best model parameters')
print(svcCVResults.best_params_)
print('Best model score')
print(svcCVResults.best_score_)
bestSVC = SVC(
    random_state = 777,
    probability = True,
    C = 0.5,
    degree = 3,
    gamma = 'scale',
    kernel = 'rbf'
)
print(classification_report(yValid, bestSVC.fit(XTrainFinal, yTrain).predict(XValidFinal)))
testID = dfTestClean[colID]
XTest = dfTestClean[colNum + colCat]
XTestFinal = apply_scaler_encoder(XTest)
testPred = bestSVC.predict(XTestFinal)
submission = pd.concat([testID, pd.DataFrame(testPred)], axis = 1)
submission = submission.rename(columns = {0: 'Survived'})
submission.to_csv('titanic_submission_20200601.csv', index = False)
submission