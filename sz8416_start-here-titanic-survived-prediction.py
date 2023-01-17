import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, MICEImputer
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
submission = pd.read_csv('../input/submission/final submission.csv')
submission.to_csv('test_submission.csv', index=False)
print('train:', train.shape[0], "rows and", train.shape[1],'columns')
print('train:', test.shape[0], "rows and", test.shape[1],'columns')
def find_missing(data):
    # number of missing values
    count_missing = data.isnull().sum().values
    # total records
    total = data.shape[0]
    # percentage of missing
    ratio_missing = count_missing/total
    # return a dataframe to show: feature name, # of missing and % of missing
    return pd.DataFrame(data={'missing_count':count_missing, 'missing_ratio':ratio_missing}, index=data.columns.values)
find_missing(train)
def plot_categorical(data, col, size=[8 ,6], xlabel_angle=0, title=''):
    '''use this for ploting the count of categorical features'''
    plotdata = data[col].value_counts() / len(data)
    plt.figure(figsize = size)
    sns.barplot(x = plotdata.index, y=plotdata.values, )
    plt.title(title)
    if xlabel_angle!=0: 
        plt.xticks(rotation=xlabel_angle)
    plt.show()
plot_categorical(data=train, col='Survived', size=[8 ,4], xlabel_angle=0, title='Train Set: Survived')
def plot_categorical_bylabel(data, col, type='count', size=[12 ,6], xlabel_angle=0, title=''):
    '''use it to compare the distribution between label 1 and label 0'''
    plt.figure(figsize = size)
    l1 = data.loc[data.Survived==1, col].value_counts()
    l0 = data.loc[data.Survived==0, col].value_counts()
    if type == 'ratio':
        l1 = l1 / l1.sum()
        l0 = l0 / l0.sum()
    plt.subplot(1,2,1)
    sns.barplot(x = l1.index, y=l1.values)
    plt.title('Default: '+title)
    plt.xticks(rotation=xlabel_angle)
    plt.subplot(1,2,2)
    sns.barplot(x = l0.index, y=l0.values)
    plt.title('Non-default: '+title)
    plt.xticks(rotation=xlabel_angle)
    plt.show()
plot_categorical_bylabel(train, 'Pclass', type='ratio', title='Class')
plot_categorical_bylabel(train, 'Sex', type='ratio', title='Gender')
# the first letter of cabin
def find_cabin(cabin_list):
    find_cabin = []
    nan_find = cabin_list.isnull()
    for i in range(len(cabin_list)):
        if nan_find[i]:
            temp = cabin_list[i]
        else:
            temp = cabin_list[i][0]
        find_cabin.append(temp)
    return find_cabin
train['Cabin_first_letter'] = find_cabin(train.Cabin)
plot_categorical_bylabel(train, 'Cabin_first_letter', type='ratio', title='Cabin')
def plot_numerical_bylabel(data, col, size=[12, 6]):
    # print out the correlation
    corr = data['Survived'].corr(data[col])
    print('The correlation between %s and the TARGET is %0.4f' % (col, corr))
    plt.figure(figsize = size)
    sns.kdeplot(data.ix[data['Survived'] == 0, col], label = 'Survived == 0')
    sns.kdeplot(data.ix[data['Survived'] == 1, col], label = 'Survived == 1')
    
    plt.xlabel(col); plt.ylabel('Density'); plt.title('%s Distribution' % col)
    plt.legend()
    plt.show()
plot_numerical_bylabel(train, 'Age')
plot_numerical_bylabel(train, 'Fare')
# from name column, find title 
def find_title(name_list):
    title_list=[]
    for i in name_list:
        i_list = i.split()
        if i_list[1] in ['Mrs.', 'Miss.', 'Master.', 'Mr.']:
            title_list.append(i_list[1])
        elif i_list[2] in ['Mrs.', 'Miss.', 'Master.', 'Mrs']:
            title_list.append(i_list[2])
        else:
            title_list.append('No title')
    return title_list
# whether cabin is missing value
def cabin_exist(cabin_list):
    cabin_exist = []
    nan_find = cabin_list.isnull()
    for i in range(len(cabin_list)):
        if nan_find[i]:
            temp=1
        else:
            temp=0
        cabin_exist.append(temp)
    return cabin_exist
# find the first letter of cabin
def find_cabin(cabin_list):
    find_cabin = []
    nan_find = cabin_list.isnull()
    for i in range(len(cabin_list)):
        if nan_find[i]:
            temp = cabin_list[i]
        else:
            temp = cabin_list[i][0]
        find_cabin.append(temp)
    return find_cabin
train.Cabin = find_cabin(train.Cabin)
def feature_engineering(df):
    df2 = df.copy()
    # passenger in class 3
    df2['is_class3'] = [1 if i == 3 else 0 for i in df2.Pclass]
    # passenger is less than 10
    df2['is_child'] = [1 if i <= 10 else 0 for i in df2.Age]
    # passenger has no page
    df2['no_parch'] = [1 if i == 0 else 0 for i in df2.Parch]
    df2['low_fare'] = [1 if i < 5 else 0 for i in df2.Fare]
    # passenger's cabin information is not null
    df2['Cabin_exist'] = cabin_exist(df2.Cabin)
    df2.Cabin = find_cabin(df2.Cabin)
    # passenger is in cabin B/C/D/E
    df2['safe_cabin']=[1 if i in ['B','C','D','E'] else 0 for i in df2.Cabin]
    df2['title']=find_title(df2.Name)
    # MR.passenger
    df2['is_mr']=[1 if i == 'Mr.' else 0 for i in df2.title]
    # miss passenger
    df2['is_miss']=[1 if i == 'Miss.' else 0 for i in df2.title]
    df2['Ticket'] = [i.split()[-1][0] for i in df2.Ticket]
    df2['Ticket'] = [0 if i=='L' else i for i in df2.Ticket]
    # passager is at position 3 - 8
    df2['Position_3to8'] = [1 if ((int(i)>=3)&(int(i)<=8)) else 0 for i in df2.Ticket]
    df2 = df2.drop(['Pclass', 'SibSp', 'Parch', 'Cabin', 'Ticket', 'Name', 'title'], 1)
    df2 = pd.get_dummies(df2).drop(['Embarked_Q', 'Embarked_C','Sex_male'],1)
    return df2
# create dataset for modeling
train_use = feature_engineering(train)
test_use = feature_engineering(test)
print(train_use.shape)
print(test_use.shape)
train_use[train_use.columns.tolist()] = MICEImputer(initial_strategy='median', n_imputations=50, n_nearest_features=20, verbose=False).fit_transform(train_use)
test_use[test_use.columns.tolist()] = MICEImputer(initial_strategy='median', n_imputations=50, n_nearest_features=20, verbose=False).fit_transform(test_use)
X = train_use.iloc[:, 2:]
y = train_use.Survived
X_pred = test_use.iloc[:, 1:]
if len(X.columns.tolist()) > 20:
    X = X.iloc[:, :13]
params1 = {'learning_rate':np.arange(0.01,0.3, 0.01)}
gdbt = GradientBoostingClassifier(n_estimators=100, min_samples_split=300,
                                  min_samples_leaf=20,max_depth=10, subsample=0.8,random_state=10)
gridscgdbt = GridSearchCV(gdbt, params1, cv=5)
gridscgdbt.fit(X, y)
print(gridscgdbt.best_params_)
print(gridscgdbt.best_score_)
params2 = {'max_depth':range(2,8,1), 'min_samples_split':range(20,200,20)}
gdbt2 = GradientBoostingClassifier(n_estimators=100, learning_rate=gridscgdbt.best_params_['learning_rate'], 
                                   subsample=0.8,random_state=10)
gridscgdbt2 = GridSearchCV(gdbt2, params2, cv=5)
gridscgdbt2.fit(X, y)
print(gridscgdbt2.best_params_)
print(gridscgdbt2.best_score_)
params3 = {'max_features':np.arange(0.5,1,0.1), 'subsample':np.arange(0.5, 1, 0.1)}
gdbt3= GradientBoostingClassifier(n_estimators=100, learning_rate=gridscgdbt.best_params_['learning_rate'], 
                                   max_depth=gridscgdbt2.best_params_['max_depth'],
                                   min_samples_split=gridscgdbt2.best_params_['min_samples_split'],
                                   subsample=0.8,random_state=10)
gridscgdbt3 = GridSearchCV(gdbt3, params3, cv=5)
gridscgdbt3.fit(X, y)
print(gridscgdbt3.best_params_)
print(gridscgdbt3.best_score_)
# final model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gbdt_best = GradientBoostingClassifier(n_estimators=100, learning_rate=gridscgdbt.best_params_['learning_rate'], 
                                   max_depth=gridscgdbt2.best_params_['max_depth'],
                                   min_samples_split=50,
                                   subsample=gridscgdbt3.best_params_['subsample'],
                                   max_features=gridscgdbt3.best_params_['max_features'],
                                   random_state=3)
gbdt_best.fit(X_train, y_train)
print('The validation accuracy is:', str(round((gbdt_best.predict(X_test) == y_test).mean(),3)))
pd.DataFrame({'importance':gbdt_best.feature_importances_}, index=X.columns)
test_ID = test.PassengerId
y_pred_test = gbdt_best.predict(X_pred)
final = pd.DataFrame({'PassengerId': test_ID, 'Survived': y_pred_test})
final.Survived = final.Survived.astype(int)
final.to_csv('final submission.csv', index=False)