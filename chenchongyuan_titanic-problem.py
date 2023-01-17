import numpy as np
import pandas as pd

raw_train_csv = pd.read_csv("../input/train.csv")  #Use pandas to read file
data_train_csv = pd.read_csv("../input/train.csv")
data_train_csv  #This will return a pandas DataFrame
data_train_csv.info()  #Concise summary of a DataFrame
data_train_csv.describe()  #Generates some descriptive statistics
raw_test_csv = pd.read_csv("../input/test.csv")
data_test_csv = pd.read_csv("../input/test.csv")
data_test_csv
data_test_csv.info()
data_test_csv.loc[(data_test_csv.Fare.isnull()), 'Fare' ] = 0  #Set 0 to where Fare is null
data_test_csv.describe()
import matplotlib.pyplot as plt

data_train_csv.Survived.value_counts().plot(kind='bar')  # histogram
plt.title("Rescued situation(1 represent Rescued)")
plt.ylabel("Number of people")
plt.xlabel('Survived')
plt.show()
data_train_csv.Pclass.value_counts().plot(kind="bar")
plt.ylabel("Number of people")
plt.xlabel('Pclass')
plt.title("The number of people by Pclass")
plt.show()
plt.scatter(data_train_csv.Survived, data_train_csv.Age)
plt.grid(b=True, which='major', axis='y')  #draw horizontal line
plt.ylabel('Age')
plt.xlabel('Survived')
plt.title("Rescued situation with age")
plt.show()
for i in range(len(data_train_csv.Pclass.value_counts())):
    data_train_csv.Age[data_train_csv.Pclass == i+1].plot(kind='kde')
plt.xlabel('Age')
plt.ylabel('Density') 
plt.title("People's Pclass by age")
plt.legend(('Pclass1', 'Pclass2','Pclass3'), loc='best')
plt.show()
data_train_csv.Embarked.value_counts().plot(kind='bar')
plt.title("The number of people by embarkation port")
plt.xlabel("Embarkation port")
plt.ylabel("Number of people")
plt.show()
Survived_0 = data_train_csv.Pclass[data_train_csv.Survived == 0].value_counts()
Survived_1 = data_train_csv.Pclass[data_train_csv.Survived == 1].value_counts()
df=pd.DataFrame({'Survived':Survived_1, 'Not survive':Survived_0})
df.plot(kind='bar', stacked=True)  #stacked bar
plt.title("Rescued situation by Pclass")
plt.xlabel('Pclass')
plt.ylabel("Number of people")
plt.show()
Pclass1 = data_train_csv.Survived[data_train_csv.Pclass==1][data_train_csv.Sex=='male'].value_counts()
Pclass2 = data_train_csv.Survived[data_train_csv.Pclass==2][data_train_csv.Sex=='male'].value_counts()
Pclass3 = data_train_csv.Survived[data_train_csv.Pclass==3][data_train_csv.Sex=='male'].value_counts()
df = pd.DataFrame({'Pclass1': Pclass1, 'Pclass2': Pclass2, 'Pclass3': Pclass3})
df.plot(kind='bar')
plt.title("Rescued situation of male")
plt.xlabel('Survived')
plt.ylabel('Number of people')
plt.show()
Pclass1 = data_train_csv.Survived[data_train_csv.Pclass==1][data_train_csv.Sex=='female'].value_counts()
Pclass2 = data_train_csv.Survived[data_train_csv.Pclass==2][data_train_csv.Sex=='female'].value_counts()
Pclass3 = data_train_csv.Survived[data_train_csv.Pclass==3][data_train_csv.Sex=='female'].value_counts()
df = pd.DataFrame({'Pclass1': Pclass1, 'Pclass2': Pclass2, 'Pclass3': Pclass3})
df.plot(kind='bar')
plt.title("Rescued situation of female")
plt.xlabel('Survived')
plt.ylabel('Number of people')
plt.show()
Survived_0 = data_train_csv.Embarked[data_train_csv.Survived == 0].value_counts()
Survived_1 = data_train_csv.Embarked[data_train_csv.Survived == 1].value_counts()
df=pd.DataFrame({'Survived':Survived_1, 'Not survive':Survived_0})
df.plot(kind='bar')
plt.title("Rescued situation by embarkation port")
plt.xlabel("Embarkation port") 
plt.ylabel("Number of people") 
plt.show()
Survived_cabin = data_train_csv.Survived[pd.notnull(data_train_csv.Cabin)].value_counts()
Survived_nocabin = data_train_csv.Survived[pd.isnull(data_train_csv.Cabin)].value_counts()
df=pd.DataFrame({'Cabin':Survived_cabin, 'Nocabin':Survived_nocabin})
print("Have Cabin number or not:")
df
def fill_missing_Age(df, estimator, fitted=False):
    df_age = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    
    known_age = df_age[df_age.Age.notnull()].as_matrix()
    unknown_age = df_age[df_age.Age.isnull()].as_matrix()
    
    X_train = known_age[:, 1:]
    y_train = known_age[:, 0]
    X_test = unknown_age[:, 1:]
    
    if fitted == False:
        estimator.fit(X_train, y_train)
        
    predict_age = estimator.predict(X_test)
    
    df.loc[(df.Age.isnull()), 'Age'] = predict_age
    
    if fitted:
        return df
    else:
        return df, estimator
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df
from sklearn.ensemble import RandomForestRegressor  #Use RandomForest

rf_reg = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=0)

data_train_filled, rf_reged = fill_missing_Age(data_train_csv, rf_reg)
data_train_filled = set_Cabin_type(data_train_filled)

data_test_filled = fill_missing_Age(data_test_csv, rf_reged, fitted=True)
data_test_filled = set_Cabin_type(data_test_filled)
def factorization(df):
    dummy_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummy_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummy_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummy_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    
    df_dummy = pd.concat([df, dummy_Cabin, dummy_Embarked, dummy_Sex, dummy_Pclass], axis=1)
    df_dummy.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    return df_dummy
data_train_dummy = factorization(data_train_filled)
data_test_dummy = factorization(data_test_filled)
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

reshape_Age = data_train_dummy['Age'].values.reshape(-1, 1)
reshape_Fare = data_train_dummy['Fare'].values.reshape(-1, 1)

std_scaler_Age = std_scaler.fit(reshape_Age)
data_train_dummy['Age_scaled'] = std_scaler.fit_transform(reshape_Age, std_scaler_Age)

std_scaler_Fare = std_scaler.fit(reshape_Fare)
data_train_dummy['Fare_scaled'] = std_scaler.fit_transform(reshape_Fare, std_scaler_Fare)
reshape_Age = data_test_dummy['Age'].values.reshape(-1, 1)
reshape_Fare = data_test_dummy['Fare'].values.reshape(-1, 1)

std_scaler_Age = std_scaler.fit(reshape_Age)
data_test_dummy['Age_scaled'] = std_scaler.fit_transform(reshape_Age, std_scaler_Age)

std_scaler_Fare = std_scaler.fit(reshape_Fare)
data_test_dummy['Fare_scaled'] = std_scaler.fit_transform(reshape_Fare, std_scaler_Fare)
from sklearn.linear_model import LogisticRegression

def filtering(train, test, regex_train, regex_test):
    data_train_filter_csv = train.filter(regex=regex_train)
    data_test_filter_csv = test.filter(regex=regex_test)

    data_train = data_train_filter_csv.as_matrix()
    data_test = data_test_filter_csv.as_matrix()
    X_train = data_train[:, 2:]
    y_train = data_train[:, 1]
    X_test = data_test
    
    return X_train, X_test, y_train, data_train_filter_csv, data_test_filter_csv
regex_train = 'PassengerId|Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*'
regex_test = 'Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*'
X_train, X_test, y_train, data_train_filter_csv, data_test_filter_csv = filtering(data_train_dummy, data_test_dummy, regex_train, regex_test)

log_reg = LogisticRegression(C=1.0, penalty='l2', tol=1e-8)
log_reg.fit(X_train, y_train)
prediction_baseline = log_reg.predict(X_test)
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.05, 1, 20), verbose=0, plot=True):
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    #plot
    if plot:      
        if ylim is not None:
            plt.ylim(ylim)
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.3)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.3)
        
        plt.plot(train_sizes, train_scores_mean, 'o-', label="Train")
        plt.plot(train_sizes, test_scores_mean, 'o-', label="Test")
        
        plt.grid()
        
        plt.title(title)
        plt.xlabel("Samples in training set")
        plt.ylabel("Score")
        plt.legend(loc='best')
        plt.show()
    
    #calculate
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    
    return midpoint, diff
plot_learning_curve(log_reg, "Learning curve", X_train, y_train, cv=5)
log_reg.coef_
pd.DataFrame({'coefficient': list(log_reg.coef_.T), 'Feature': list(data_train_filter_csv.columns)[2:]})
from sklearn.model_selection import train_test_split

data_train_split_csv, data_test_split_csv = train_test_split(data_train_filter_csv, test_size=0.25)
X_train_split = data_train_split_csv.as_matrix()[:, 2:]
y_train_split = data_train_split_csv.as_matrix()[:, 1]
X_test_split = data_test_split_csv.as_matrix()[:, 2:]
y_test_split = data_test_split_csv.as_matrix()[:, 1]

log_reg_split = LogisticRegression(C=1.0, penalty='l2', tol=1e-8)
log_reg_split.fit(X_train_split, y_train_split)
from sklearn.metrics import accuracy_score

prediction_split = log_reg_split.predict(X_test_split)
accuracy_score(prediction_split, y_test_split)
from sklearn.model_selection import cross_val_score

cross_val_score(log_reg_split, X_train_split, y_train_split, cv=5)
good_case = data_train_csv.loc[data_train_csv['PassengerId'].isin(data_test_split_csv[prediction_split==y_test_split]['PassengerId'].values)]
good_case
bad_case = data_train_csv.loc[data_train_csv['PassengerId'].isin(data_test_split_csv[prediction_split!=y_test_split]['PassengerId'].values)]
bad_case
train_csv = pd.read_csv("../input/train.csv")
test_csv = pd.read_csv("../input/test.csv")
Mr_average_age = np.average(train_csv[train_csv.Name.str.contains('Mr') & train_csv.Age.notnull()].Age)
Ms_average_age = np.average(train_csv[train_csv.Name.str.contains('Ms') & train_csv.Age.notnull()].Age)
Mrs_average_age = np.average(train_csv[train_csv.Name.str.contains('Mrs') & train_csv.Age.notnull()].Age)
Miss_average_age = np.average(train_csv[train_csv.Name.str.contains('Miss') & train_csv.Age.notnull()].Age)
Master_average_age = np.average(train_csv[train_csv.Name.str.contains('Master') & train_csv.Age.notnull()].Age)
Dr_average_age = np.average(train_csv[train_csv.Name.str.contains('Dr') & train_csv.Age.notnull()].Age)
Pclass3_average_fare = np.average(train_csv[train_csv.Pclass==3].Fare)

train_csv.loc[(train_csv.Name.str.contains('Mr') & train_csv.Age.isnull()), 'Age'] = Mr_average_age
train_csv.loc[(train_csv.Name.str.contains('Ms') & train_csv.Age.isnull()), 'Age'] = Ms_average_age
train_csv.loc[(train_csv.Name.str.contains('Mrs') & train_csv.Age.isnull()), 'Age'] = Mrs_average_age
train_csv.loc[(train_csv.Name.str.contains('Miss') & train_csv.Age.isnull()), 'Age'] = Miss_average_age
train_csv.loc[(train_csv.Name.str.contains('Master') & train_csv.Age.isnull()), 'Age'] = Master_average_age
train_csv.loc[(train_csv.Name.str.contains('Dr') & train_csv.Age.isnull()), 'Age'] = Dr_average_age

test_csv.loc[(test_csv.Name.str.contains('Mr') & test_csv.Age.isnull()), 'Age'] = Mr_average_age
test_csv.loc[(test_csv.Name.str.contains('Ms') & test_csv.Age.isnull()), 'Age'] = Ms_average_age
test_csv.loc[(test_csv.Name.str.contains('Mrs') & test_csv.Age.isnull()), 'Age'] = Mrs_average_age
test_csv.loc[(test_csv.Name.str.contains('Miss') & test_csv.Age.isnull()), 'Age'] = Miss_average_age
test_csv.loc[(test_csv.Name.str.contains('Master') & test_csv.Age.isnull()), 'Age'] = Master_average_age
test_csv.loc[(test_csv.Name.str.contains('Dr') & test_csv.Age.isnull()), 'Age'] = Dr_average_age
test_csv.loc[test_csv.Fare.isnull(), 'Fare'] = Pclass3_average_fare
train_csv['Sex_Pclass'] = train_csv.Sex + '_' + train_csv.Pclass.map(str)
test_csv['Sex_Pclass'] = test_csv.Sex + '_' + test_csv.Pclass.map(str)
train_csv.loc[:, 'Age_dummy'] = pd.cut(train_csv.Age, 40, labels=range(1, 41))
test_csv.loc[:, 'Age_dummy'] = pd.cut(test_csv.Age, 40, labels=range(1, 41))
train_csv['Child'] = 0.0
test_csv['Child'] = 0.0
train_csv.loc[train_csv.Age<=13, 'Child'] = 1.0
test_csv.loc[test_csv.Age<=13, 'Child'] = 1.0
train_csv['Mother'] = 0.0
test_csv['Mother'] = 0.0
train_csv.loc[train_csv.Name.str.contains('Mrs') & (train_csv.Parch>0), 'Mother'] = 1.0
test_csv.loc[test_csv.Name.str.contains('Mrs') & (test_csv.Parch>0), 'Mother'] = 1.0
train_csv['Family'] = 0.0
test_csv['Family'] = 0.0
train_csv['Family'] = train_csv.SibSp + train_csv.Parch
test_csv['Family'] = test_csv.SibSp + test_csv.Parch
def factorization2(df):
    dummy_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummy_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummy_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummy_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    dummy_Sex_Pclass = pd.get_dummies(df['Sex_Pclass'], prefix='Sex_Pclass')
    dummy_Age = pd.get_dummies(df['Age_dummy'], prefix='Age')
    
    df_dummy = pd.concat([df, dummy_Cabin, dummy_Embarked, dummy_Sex, 
                          dummy_Age, dummy_Pclass, dummy_Sex_Pclass], axis=1)
    df_dummy.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 
                   'Age_dummy', 'Sex_Pclass'], axis=1, inplace=True)

    return df_dummy
train_csv = set_Cabin_type(train_csv)
test_csv = set_Cabin_type(test_csv)

data_train_dummy2 = factorization2(train_csv)
data_test_dummy2 = factorization2(test_csv)
reshape_Age2 = data_train_dummy2['Age'].values.reshape(-1, 1)
reshape_Fare2 = data_train_dummy2['Fare'].values.reshape(-1, 1)
reshape_Family = data_train_dummy2['Family'].values.reshape(-1, 1)

std_scaler_Age2 = std_scaler.fit(reshape_Age2)
data_train_dummy2['Age_scaled'] = std_scaler.fit_transform(reshape_Age2, std_scaler_Age2)

std_scaler_Fare2 = std_scaler.fit(reshape_Fare2)
data_train_dummy2['Fare_scaled'] = std_scaler.fit_transform(reshape_Fare2, std_scaler_Fare2)

std_scaler_Family = std_scaler.fit(reshape_Family)
data_train_dummy2['Family_scaled'] = std_scaler.fit_transform(reshape_Family, std_scaler_Family)
reshape_Age2 = data_test_dummy2['Age'].values.reshape(-1, 1)
reshape_Fare2 = data_test_dummy2['Fare'].values.reshape(-1, 1)
reshape_Family = data_test_dummy2['Family'].values.reshape(-1, 1)

std_scaler_Age2 = std_scaler.fit(reshape_Age2)
data_test_dummy2['Age_scaled'] = std_scaler.fit_transform(reshape_Age2, std_scaler_Age2)

std_scaler_Fare2 = std_scaler.fit(reshape_Fare2)
data_test_dummy2['Fare_scaled'] = std_scaler.fit_transform(reshape_Fare2, std_scaler_Fare2)

std_scaler_Family = std_scaler.fit(reshape_Family)
data_test_dummy2['Family_scaled'] = std_scaler.fit_transform(reshape_Family, std_scaler_Family)
regex_train2 = "PassengerId|Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Embarked_.*|Child|Mother|Family"
regex_test2 = "Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Embarked_.*|Child|Mother|Family"
X_train2, X_test2, y_train2, data_train_filter_csv2, data_test_filter_csv2 = filtering(data_train_dummy2, data_test_dummy2, regex_train2, regex_test2)
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        'penalty':['l1'],
        'C':[i for i in np.arange(0.05, 3, 0.05)],
        'tol':[1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    },
    {
        'penalty':['l2'],
        'C':[i for i in np.arange(0.05, 3, 0.05)],
        'tol':[1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    }
]

log_reg_search = LogisticRegression()
grid_search = GridSearchCV(estimator=log_reg_search, param_grid=param_grid, n_jobs=-1, verbose=5)
grid_search.fit(X_train2, y_train2)
grid_search.best_estimator_
log_reg2 = LogisticRegression(C=0.85, penalty='l1', tol=1e-4)
log_reg2.fit(X_train2, y_train2)
cross_val_score(log_reg2, X_train2, y_train2, cv=5)
prediction = log_reg2.predict(X_test2)
result = pd.DataFrame({'PassengerId':data_test_csv['PassengerId'].as_matrix(), 'Survived':prediction.astype(np.int32)})
result.to_csv("prediction.csv", index=False)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

dt_reg = DecisionTreeClassifier()
ada_clf = AdaBoostClassifier(dt_reg, n_estimators=100)
ada_clf.fit(X_train2, y_train2)
prediction_ada = ada_clf.predict(X_test2)
# result = pd.DataFrame({'PassengerId':data_test_csv['PassengerId'].as_matrix(), 'Survived':prediction_ada.astype(np.int32)})
# result.to_csv("prediction.csv", index=False)