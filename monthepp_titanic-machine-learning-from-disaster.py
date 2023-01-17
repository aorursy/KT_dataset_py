# import python standard library

import re



# import data manipulation library

import numpy as np

import pandas as pd



# import data visualization library

import matplotlib.pyplot as plt

import seaborn as sns



# import sklearn model class

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# import sklearn model selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



# import sklearn model evaluation classification metrics

from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
# acquiring training and testing data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# visualize head of the training data

df_train.head(n=5)
# visualize tail of the testing data

df_test.tail(n=5)
# combine training and testing dataframe

df_train['DataType'], df_test['DataType'] = 'training', 'testing'

df_test.insert(1, 'Survived', np.nan)

df_data = pd.concat([df_train, df_test], ignore_index=True)
def countplot(categorical_x: list or str, categorical_y: list or str, data: pd.DataFrame, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return a count plot applied for categorical variable in x-axis vs categorical variable in y-axis.

    

    Args:

        categorical_x (list or str): The categorical variable in x-axis.

        categorical_y (list or str): The categorical variable in y-axis.

        data (pd.DataFrame): The data to plot.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    categorical_x, categorical_y = [categorical_x] if type(categorical_x) == str else categorical_x, [categorical_y] if type(categorical_y) == str else categorical_y

    if nrows is None: nrows = (len(categorical_x)*len(categorical_y) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    _ = [sns.countplot(x=vj, hue=vi, data=data, ax=axes[i*len(categorical_x) + j], rasterized=True) for i, vi in enumerate(categorical_y) for j, vj in enumerate(categorical_x)]

    return fig
def swarmplot(categorical_x: list or str, numerical_y: list or str, data: pd.DataFrame, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return a swarm plot applied for categorical variable in x-axis vs numerical variable in y-axis.

    

    Args:

        categorical_x (list or str): The categorical variable in x-axis.

        numerical_y (list or str): The numerical variable in y-axis.

        data (pd.DataFrame): The data to plot.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    categorical_x, numerical_y = [categorical_x] if type(categorical_x) == str else categorical_x, [numerical_y] if type(numerical_y) == str else numerical_y

    if nrows is None: nrows = (len(categorical_x)*len(numerical_y) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    _ = [sns.swarmplot(x=vj, y=vi, data=data, ax=axes[i*len(categorical_x) + j], rasterized=True) for i, vi in enumerate(numerical_y) for j, vj in enumerate(categorical_x)]

    return fig
def violinplot(categorical_x: list or str, numerical_y: list or str, data: pd.DataFrame, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return a violin plot applied for categorical variable in x-axis vs numerical variable in y-axis.

    

    Args:

        categorical_x (list or str): The categorical variable in x-axis.

        numerical_y (list or str): The numerical variable in y-axis.

        data (pd.DataFrame): The data to plot.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    categorical_x, numerical_y = [categorical_x] if type(categorical_x) == str else categorical_x, [numerical_y] if type(numerical_y) == str else numerical_y

    if nrows is None: nrows = (len(categorical_x)*len(numerical_y) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    _ = [sns.violinplot(x=vj, y=vi, data=data, ax=axes[i*len(categorical_x) + j], rasterized=True) for i, vi in enumerate(numerical_y) for j, vj in enumerate(categorical_x)]

    return fig
# describe training and testing data

df_data.describe(include='all')
# convert dtypes numeric to object

col_convert = ['Survived', 'Pclass', 'SibSp', 'Parch']

df_data[col_convert] = df_data[col_convert].astype('object')
# list all features type number

col_number = df_data.select_dtypes(include=['number']).columns.tolist()

print('features type number:\n items %s\n length %d' %(col_number, len(col_number)))



# list all features type object

col_object = df_data.select_dtypes(include=['object']).columns.tolist()

print('features type object:\n items %s\n length %d' %(col_object, len(col_object)))
# feature extraction: surname

df_data['Surname'] = df_data['Name'].str.extract(r'([A-Za-z]+),', expand=False)
# feature extraction: title

df_data['Title'] = df_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

df_data['Title'] = df_data['Title'].replace(['Capt', 'Rev'], 'Crew')

df_data['Title'] = df_data['Title'].replace('Ms', 'Miss')

df_data['Title'] = df_data['Title'].replace(['Col', 'Countess', 'Don', 'Dona', 'Jonkheer', 'Lady', 'Major', 'Mlle', 'Mme', 'Sir'], 'Royal')

df_data['Title'].value_counts()
# feature exploration: sex

col_object = df_data.select_dtypes(include=['object']).columns.drop(['Name', 'Ticket', 'Cabin', 'Surname']).tolist()

_ = countplot(col_object, 'Sex', df_data)
# feature exploration: age

col_object = df_data.select_dtypes(include=['object']).columns.drop(['Name', 'Ticket', 'Cabin', 'Surname']).tolist()

_ = swarmplot(col_object, 'Age', df_data)
# feature extraction: age

df_data['Age'] = df_data['Age'].fillna(df_data.groupby(['Title'], as_index=True)['Age'].transform('mean'))
# feature extraction: family size

df_data['FamilySize'] = df_data['SibSp'] + df_data['Parch'] + 1
# feature extraction: ticket string

df_data['TicketString'] = df_data['Ticket'].apply(lambda x: ''.join(re.findall(r'[a-zA-Z]+', x)))

df_data['TicketString'] = df_data['TicketString'].replace(['CASOTON', 'SOTONO', 'STONO', 'STONOQ'], 'SOTONOQ')

df_data['TicketString'] = df_data['TicketString'].replace(['SC', 'SCParis'], 'SCPARIS')

df_data['TicketString'] = df_data['TicketString'].replace('FCC', 'FC')

df_data['TicketString'] = df_data['TicketString'].replace(df_data['TicketString'].value_counts()[df_data['TicketString'].value_counts() < 10].index.tolist(), 'OTHER')

df_data['TicketString'].value_counts()
# feature extraction: has ticket string

df_data['HasTicketString'] = df_data['TicketString'].apply(lambda x: 1 if x else 0).astype('object')
# feature exploration: fare

col_object = df_data.select_dtypes(include=['object']).columns.drop(['Name', 'Ticket', 'Cabin', 'Surname']).tolist()

_ = swarmplot(col_object, 'Fare', df_data)
# feature extraction: fare

df_data['Fare'] = df_data['Fare'].fillna(df_data.groupby(['Pclass'], as_index=True)['Fare'].transform('mean'))
# feature extraction: cabin

df_data['Cabin'] = df_data['Cabin'].fillna(0)
# feature extraction: cabin string

df_data['CabinString'] = df_data['Cabin'].str.extract(r'([A-Za-z]+)', expand=False)
# feature extraction: has cabin

df_data['HasCabin'] = df_data['CabinString'].apply(lambda x: 0 if pd.isnull(x) else 1).astype('object')
# feature exploration: embarked

col_object = df_data.select_dtypes(include=['object']).columns.drop(['Name', 'Ticket', 'Cabin', 'Surname']).tolist()

_ = countplot(col_object, 'Embarked', df_data)
# feature extraction: embarked

df_data['Embarked'] = df_data['Embarked'].fillna(df_data['Embarked'].value_counts().idxmax())
# list all features type number

col_number = df_data.select_dtypes(include=['number']).columns.tolist()

print('features type number:\n items %s\n length %d' %(col_number, len(col_number)))



# list all features type object

col_object = df_data.select_dtypes(include=['object']).columns.tolist()

print('features type object:\n items %s\n length %d' %(col_object, len(col_object)))
# feature exploration: survived

col_number = df_data.select_dtypes(include=['number']).columns.drop(['PassengerId']).tolist()

col_object = df_data.select_dtypes(include=['object']).columns.drop(['Name', 'Ticket', 'Cabin', 'Surname']).tolist()

_ = swarmplot('Survived', col_number, df_data[df_data['DataType'] == 'training'])

_ = countplot(col_object, 'Survived', df_data[df_data['DataType'] == 'training'])
# feature exploration: survived where family size equal to 1

col_number = df_data.select_dtypes(include=['number']).columns.drop(['PassengerId']).tolist()

col_object = df_data.select_dtypes(include=['object']).columns.drop(['Name', 'Ticket', 'Cabin', 'Surname']).tolist()

_ = swarmplot('Survived', col_number, df_data[(df_data['DataType'] == 'training') & (df_data['FamilySize'] == 1)])

_ = countplot(col_object, 'Survived', df_data[(df_data['DataType'] == 'training') & (df_data['FamilySize'] == 1)])
# feature exploration: survived where family size more than 1

col_number = df_data.select_dtypes(include=['number']).columns.drop(['PassengerId']).tolist()

col_object = df_data.select_dtypes(include=['object']).columns.drop(['Name', 'Ticket', 'Cabin', 'Surname']).tolist()

_ = swarmplot('Survived', col_number, df_data[(df_data['DataType'] == 'training') & (df_data['FamilySize'] > 1)])

_ = countplot(col_object, 'Survived', df_data[(df_data['DataType'] == 'training') & (df_data['FamilySize'] > 1)])
# feature extraction: ticket dataframe

df_ticket = pd.get_dummies(df_data[df_data['FamilySize'] > 1], columns=['Pclass', 'Sex', 'Embarked', 'DataType', 'Title', 'CabinString', 'HasCabin'], drop_first=False)

df_ticket['Survived'] = df_ticket['Survived'].astype(float)

df_ticket = df_ticket.groupby(['Ticket'], as_index=False).agg({

    'Survived': 'mean',

    'Pclass_1': sum, 'Pclass_2': sum,  'Pclass_3': sum,

    'Sex_male': sum, 'Sex_female': sum,

    'Embarked_C': sum, 'Embarked_Q': sum, 'Embarked_S': sum,

    'DataType_training': sum, 'DataType_testing': sum,

    'Title_Crew': sum, 'Title_Dr': sum, 'Title_Master': sum, 'Title_Miss': sum, 'Title_Mr': sum, 'Title_Mrs': sum, 'Title_Royal': sum,

    'CabinString_A': sum, 'CabinString_B': sum, 'CabinString_C': sum, 'CabinString_D': sum, 'CabinString_E': sum, 'CabinString_F': sum, 'CabinString_G': sum,

    'HasCabin_0': sum, 'HasCabin_1': sum

})
# describe ticket dataframe

df_ticket.describe(include='all')
# convert dtypes numeric to object

col_convert = df_ticket.columns.drop('Ticket').tolist()

df_ticket[col_convert] = df_ticket[col_convert].astype('object')
# convert dtypes object to numeric

col_convert = ['Survived']

df_ticket[col_convert] = df_ticket[col_convert].astype(float)
# feature extraction: together

df_ticket['Together'] = df_ticket['Survived'].apply(lambda x: 1 if x == 0 or x == 1 else 0).astype('object')
# feature exploration: survived

col_object = df_ticket.select_dtypes(include=['object']).columns.drop('Ticket').tolist()

_ = swarmplot(col_object, 'Survived', df_ticket)

_ = violinplot(col_object, 'Survived', df_ticket)
# feature extraction: with sex and title

df_data = pd.merge(df_data, df_ticket[['Ticket', 'Sex_male', 'Sex_female', 'Title_Crew', 'Title_Dr', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Royal']], how='left', left_on='Ticket', right_on='Ticket').rename(columns={

    'Sex_male': 'WithSexMale', 'Sex_female': 'WithSexFemale',

    'Title_Crew': 'WithTitleCrew', 'Title_Dr': 'WithTitleDr', 'Title_Master': 'WithTitleMaster', 'Title_Miss': 'WithTitleMiss', 'Title_Mr': 'WithTitleMr', 'Title_Mrs': 'WithTitleMrs', 'Title_Royal': 'WithTitleRoyal'

})

col_fillnas = ['WithSexMale', 'WithSexFemale', 'WithTitleCrew', 'WithTitleDr', 'WithTitleMaster', 'WithTitleMiss', 'WithTitleMr', 'WithTitleMrs', 'WithTitleRoyal']

for col_fillna in col_fillnas: df_data[col_fillna] = df_data[col_fillna].fillna(0)
# feature extraction: ticket_self dataframe

df_temp = df_data.copy(deep=True)

df_temp['Survived'] = df_temp['Survived'].astype(float)

df_ticket_self = df_temp.groupby(['Ticket'], as_index=True)



# feature extraction: survived peer

count = df_ticket_self['Survived'].transform('count')

mean = df_ticket_self['Survived'].transform('mean')

df_data['SurvivedPeer'] = (mean * count - df_data['Survived'].astype(float)) / (count - 1)

df_data['SurvivedPeer'] = df_data['SurvivedPeer'].astype(float).fillna(-1)
# feature extraction: ticket_title dataframe

df_temp = df_data.copy(deep=True)

df_temp['Survived'] = df_temp['Survived'].astype(float)

col_revises = ['Crew', 'Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Royal']

for col_revise in col_revises:

    df_temp['Survived' + col_revise] = df_temp['Survived']

    df_temp.loc[df_temp['Title'] != col_revise, 'Survived' + col_revise] = np.nan

df_ticket_title = df_temp.groupby(['Ticket'], as_index=True)



# feature extraction: survived peer title

for col_revise in col_revises:

    count = df_ticket_title['Survived' + col_revise].transform('count')

    mean = df_ticket_title['Survived' + col_revise].transform('mean')

    df_data['SurvivedPeer' + col_revise] = (mean * count - df_temp['Survived' + col_revise].astype(float)) / (count - 1)

    df_data['SurvivedPeer' + col_revise] = df_data['SurvivedPeer' + col_revise].astype(float).fillna(-1)
# feature exploration: survived peer and with sex and title

col_number = df_data.select_dtypes(include=['number']).columns.drop(['PassengerId']).tolist()

_ = swarmplot('Survived', col_number, df_data[(df_data['DataType'] == 'training') & (df_data['WithTitleMaster'] >= 1)])

_ = violinplot('Survived', col_number, df_data[(df_data['DataType'] == 'training') & (df_data['WithTitleMaster'] >= 1)])
# feature extraction: survived

df_data['Survived'] = df_data['Survived'].fillna(-1)
# convert category codes for data dataframe

df_data = pd.get_dummies(df_data, columns=['Pclass', 'Sex', 'Embarked', 'DataType', 'Title', 'TicketString', 'HasTicketString', 'CabinString', 'HasCabin'], drop_first=True)
# convert dtypes object to numeric for data dataframe

col_convert = ['Survived', 'SibSp', 'Parch', 'FamilySize']

df_data[col_convert] = df_data[col_convert].astype(int)
# describe data dataframe

df_data.describe(include='all')
# verify dtypes object

df_data.info()
# compute pairwise correlation of columns, excluding NA/null values and present through heat map

corr = df_data[df_data['DataType_training'] == 1].corr()

fig, axes = plt.subplots(figsize=(20, 15))

heatmap = sns.heatmap(corr, annot=True, cmap=plt.cm.RdBu, fmt='.1f', square=True, vmin=-0.8, vmax=0.8)
# select all features to evaluate the feature importances

x = df_data[df_data['DataType_training'] == 1].drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Surname', 'FamilySize', 'DataType_training', 'SurvivedPeerCrew', 'SurvivedPeerDr', 'SurvivedPeerMaster', 'SurvivedPeerMiss', 'SurvivedPeerMr', 'SurvivedPeerMrs', 'SurvivedPeerRoyal'], axis=1)

y = df_data.loc[df_data['DataType_training'] == 1, 'Survived']
# set up random forest classifier to find the feature importances

forestclf = RandomForestClassifier(n_estimators=100, random_state=58).fit(x, y)

feat = pd.DataFrame(data=forestclf.feature_importances_, index=x.columns, columns=['FeatureImportances']).sort_values(['FeatureImportances'], ascending=False)
# plot the feature importances

feat.plot(y='FeatureImportances', figsize=(20, 5), kind='bar', logy=True)

plt.axhline(0.005, color="grey")
# list feature importances

model_feat = feat[feat['FeatureImportances'] > 0.005].index
# select the important features

x = df_data.loc[df_data['DataType_training'] == 1, model_feat]

y = df_data.loc[df_data['DataType_training'] == 1, 'Survived']
# perform train-test (validate) split

x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.25, random_state=58)
# logistic regression model setup

model_logreg = LogisticRegression(solver='lbfgs', max_iter=1024)



# logistic regression model fit

model_logreg.fit(x_train, y_train)



# logistic regression model prediction

model_logreg_ypredict = model_logreg.predict(x_validate)



# logistic regression model metrics

model_logreg_f1score = f1_score(y_validate, model_logreg_ypredict)

model_logreg_accuracyscore = accuracy_score(y_validate, model_logreg_ypredict)

model_logreg_cvscores = cross_val_score(model_logreg, x, y, cv=20, scoring='accuracy')

print('logistic regression\n  f1 score: %0.4f, accuracy score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_logreg_f1score, model_logreg_accuracyscore, model_logreg_cvscores.mean(), 2 * model_logreg_cvscores.std()))
# decision tree classifier model setup

model_treeclf = DecisionTreeClassifier(splitter='best', min_samples_split=5)



# decision tree classifier model fit

model_treeclf.fit(x_train, y_train)



# decision tree classifier model prediction

model_treeclf_ypredict = model_treeclf.predict(x_validate)



# decision tree classifier model metrics

model_treeclf_f1score = f1_score(y_validate, model_treeclf_ypredict)

model_treeclf_accuracyscore = accuracy_score(y_validate, model_treeclf_ypredict)

model_treeclf_cvscores = cross_val_score(model_treeclf, x, y, cv=20, scoring='accuracy')

print('decision tree classifier\n  f1 score: %0.4f, accuracy score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_treeclf_f1score, model_treeclf_accuracyscore, model_treeclf_cvscores.mean(), 2 * model_treeclf_cvscores.std()))
# random forest classifier model setup

model_forestclf = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=58)



# random forest classifier model fit

model_forestclf.fit(x_train, y_train)



# random forest classifier model prediction

model_forestclf_ypredict = model_forestclf.predict(x_validate)



# random forest classifier model metrics

model_forestclf_f1score = f1_score(y_validate, model_forestclf_ypredict)

model_forestclf_accuracyscore = accuracy_score(y_validate, model_forestclf_ypredict)

model_forestclf_cvscores = cross_val_score(model_forestclf, x, y, cv=20, scoring='accuracy')

print('random forest classifier\n  f1 score: %0.4f, accuracy score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_forestclf_f1score, model_forestclf_accuracyscore, model_forestclf_cvscores.mean(), 2 * model_forestclf_cvscores.std()))
# specify the hyperparameter space

params = {'n_estimators': [100],

          'max_depth': [10, 20, None],

          'min_samples_split': [3, 5, 7, 9],

          'random_state': [58],

}



# random forest classifier grid search model setup

model_forestclf_cv = GridSearchCV(model_forestclf, params, cv=5)



# random forest classifier grid search model fit

model_forestclf_cv.fit(x_train, y_train)



# random forest classifier grid search model prediction

model_forestclf_cv_ypredict = model_forestclf_cv.predict(x_validate)



# random forest classifier grid search model metrics

model_forestclf_cv_f1score = f1_score(y_validate, model_forestclf_cv_ypredict)

model_forestclf_cv_accuracyscore = accuracy_score(y_validate, model_forestclf_cv_ypredict)

model_forestclf_cv_cvscores = cross_val_score(model_forestclf_cv, x, y, cv=20, scoring='accuracy')

print('random forest classifier grid search\n  f1 score: %0.4f, accuracy score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_forestclf_cv_f1score, model_forestclf_cv_accuracyscore, model_forestclf_cv_cvscores.mean(), 2 * model_forestclf_cv_cvscores.std()))

print('  best parameters: %s' %model_forestclf_cv.best_params_)
# model selection

final_model = model_forestclf



# prepare testing data and compute the observed value

x_test = df_data.loc[df_data['DataType_training'] == 0, model_feat]

y_test = pd.DataFrame(final_model.predict(x_test), columns=['Survived'], index=df_data.loc[df_data['DataType_training'] == 0, 'PassengerId'])
# submit the results

out = pd.DataFrame({'PassengerId': y_test.index, 'Survived': y_test['Survived']})

out.to_csv('submission.csv', index=False)