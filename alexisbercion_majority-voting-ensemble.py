#Data munging

import pandas as pd

import numpy as np



#Data Viz 

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



#Configuration of viz

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')



#Importing models



# Simple classifier that could fit to our data with  High interpreability

from sklearn.linear_model import LogisticRegressionCV  



#More accurate classifiers, less interpratable

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier 



#Set of metrics to evaluate our models

from sklearn import metrics
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



df = pd.concat([train, test], ignore_index = True)



del train, test



train = df[0:891]

test = df[891:] # Use a view of the full dataset allow for transformationson the whole data
def plot_distribution( df , var , target , **kwargs ):

    """Plot a density plot of var, depending on target. Creates other row/cols

    for each unique value in the row/col Series"""

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    # Interesting method : Create a conditional subplot grid. Max 3 variables

    # then apply a plotting method to the grid

    facet.map( sns.kdeplot , var , shade= True )

    # Fit and plot a univariate or bivariate kernel density estimate.

    facet.set( xlim=( 0 , int(df[ var ].max()) ) ) # Limit of the x-axis

    facet.add_legend() # Legend of the density color

print('Imported plot_distribution( df , var , target , row, col )')





def plot_categories( df , cat , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , row = row , col = col )

    facet.map( sns.barplot , cat , target )

    facet.add_legend()

print('Imported plot_categories( df , cat , target , row, col )')





def plot_correlation_map( df ):

    corr = df.corr() #DF method that returns cov matrix

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    #pallete diverging : from bad to good =/= sequential : from low to high

    _ = sns.heatmap(

        corr,

        cmap = cmap,

        square=True,

        cbar_kws={ 'shrink' : .9 }, # Color bar arg : kws dict of arg to set params

        ax=ax, #Really can't get the principle of 'ax'

        annot = True, # Write data in each cell

        annot_kws = { 'fontsize' : 12 }

    )

print('Imported plot_correlation_map( df )')
print(df.info())

df.head(8)
train.describe()
# Testing the fcts

#plot_distribution( train, var = 'Age', target = 'Survived', row='Sex')
plot_correlation_map(train)
plot_categories(train, cat='Embarked', target='Survived')
sns.countplot(data= train, x='SibSp', hue="Survived")
train.groupby(by='Embarked')[['Fare', 'Pclass']].describe()
final = df.copy().drop('PassengerId', axis=1) # Dataset on which transformations will be performed
# Extracting ticket label and c

import re

def cleanT(ticket):

    ticket = re.sub(r'\.|/', '', ticket).split() #re.sub to replace each occurence of a regex

    ticket = [t.strip() for t in ticket] 

    if len(ticket) == 1:

        return np.array(['XXX', ticket[0]])

    else:

        return np.array([ticket[0], ticket[1]])  #Version keeping labels

        #return np.array(['LAB', ticket[1]])

    

final.Ticket = final.Ticket.map(cleanT)

ticket = pd.DataFrame.from_records(final.Ticket, columns=['T_label', 'T_num'])

final = (pd.concat([final,ticket], axis=1)

                .drop('Ticket', axis=1))



df = pd.concat([df, ticket],axis=1)
final = pd.get_dummies(final, columns=['T_label'], prefix='T_label').drop('T_num', axis=1)

print(final.shape)
def get_title(name):

    return name.split(',')[1].split('.')[0].strip()



titles = {

    'Mr' : 'Mr',

    'Miss': 'Miss',

    'Mrs' : 'Mrs',

    'Master': 'Master',

    'Rev' : 'Mr',

    'Dr': 'Mr',

    'Col' : 'Rare',

    'Major':'Rare',

    'Ms': 'Miss',

    'Mlle' : 'Miss',

    'Dona': 'Rare',

    'Mme' : 'Mrs',

    'Capt': 'Rare',

    'Don': 'Rare',

    'Sir': 'Rare',

    'the Countess': 'Rare',

    'Sir': 'Rare',

    'Jonkheer':'Rare',

    'Lady': 'Rare',

}



df['Title'] = df['Name'].map(get_title).map(titles)

final['Title'] = df['Title'].copy()

print(final.shape)
final = pd.get_dummies(final, columns=['Title'], prefix='Title').drop('Name', axis=1)
age_group = pd.DataFrame(df.groupby(by=['Sex', 'Pclass', 'Title']).Age.median())



replace = {} # Use of a dict to map each missing value with corresponding average

age_null = df[pd.isnull(df.Age)]



for i in age_null.index:

    replace[i] =age_group.loc[age_null['Sex'][i], age_null['Pclass'][i], age_null['Title'][i]].values

    

df['Age']= df.Age.fillna(axis=0, value=replace)

final.Age = df.Age.copy()
plot_distribution(train, target='Survived', var='Age', row='Sex');
final['Sex'] = final['Sex'].map(lambda x: 1 if x== 'male' else 0)
final['male_age15'] = final.apply(lambda x: (x['Age'] < 15) * x['Sex'], axis = 1)

final['male_age25'] = final.apply(lambda x: (x['Age'] > 25) * x['Sex'], axis = 1)
df['Cabin'] = df.Cabin.map(lambda t: str(t)[0] if not str(t) == 'nan' else 'NC')

final['Cabin'] = df['Cabin'].copy()
final = pd.get_dummies(final, columns=['Cabin'], prefix='Deck')

final.drop(['Deck_NC'], axis=1, inplace=True)
df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna('S')



final['Embarked'] = df['Embarked'].copy()



final = pd.get_dummies(final, columns=['Embarked'], prefix='Embarked')
for i in df[df['Fare'].isnull()].index:

    df.loc[i, 'Fare'] = df['Fare'][df['Pclass'] == df['Pclass'][i]].median()
final.Fare = df.Fare.copy()
final = pd.get_dummies(final, columns=['Pclass'], prefix='Pclass')
def family():

    size = df.Parch + df.SibSp

    df['Family'] = True

    for i in size.index:

        if size[i] == 0:

            df.loc[i, 'Family'] = 'Single'

        elif size[i] < 4:

            df.loc[i, 'Family'] = 'Small'

        else:

            df.loc[i, 'Family'] = 'Large'

    print(df.Family.head(5))



family()

final['Family']= df['Family'].copy()

final = pd.get_dummies(final, columns=['Family'], prefix='Family')
final.Survived[:891].isnull().sum()
# Comparing ticket label values on train and test sets

train = df.loc[:890]

test = df.loc[891:]
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(final[['Age', 'Fare']])

final[['Age','Fare']] = std_scale.transform(final[['Age','Fare']])
# Cycle of feature selection and 
# Initialize the vectors

drop_features = [feature for feature in final.columns if 'T_label' in feature]

final = final.drop(drop_features, axis=1)

Y_train = final.loc[:890]['Survived'].copy()

X_train = final.loc[:890].copy().drop('Survived', axis=1)#.drop(drop_features, axis=1)



X_test = final.loc[891:].copy().drop('Survived', axis=1)#.drop(drop_features, axis=1)



Y_pred = pd.DataFrame({'PassengerId' : df.loc[891:, 'PassengerId']})
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.model_selection import ShuffleSplit, cross_val_score

seed = 5



cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state= seed)
# Use of LogistricRegression CV which selects automatically the regularization term

logreg = LogisticRegressionCV(cv = cv, Cs = 10)

"""penalty='l1', solver='liblinear', n_jobs=-1, cv=10,Cs=10"""

#features = pd.DataFrame(X_train.columns, columns=['variables'])

#features['coeflr'] = pd.Series(logreg.coef_[0])



scores = {}

scores['logreg']= cross_val_score(logreg, X_train, Y_train, cv = cv)



logreg.fit(X_train, Y_train)

Y_pred['logreg'] = logreg.predict(X_test).astype(int)
#Get the confusion matrix to see where our model lacks.

print(metrics.confusion_matrix(y_true= Y_train,

                        y_pred = logreg.predict(X_train).astype(int)))



print(metrics.classification_report(y_true = Y_train,

                                   y_pred = logreg.predict(X_train).astype(int)))
# Creating a coef importance plotting tool for linear models





def imp_coef(model, features, top_features = 10):

    coef = model.coef_.ravel()

    max_feat = np.argsort(coef)[-top_features:]

    min_feat = np.argsort(coef)[:top_features]

    top_coef = np.hstack([min_feat, max_feat])

    

    # Plotting

    plt.figure(figsize= (15,7))

    plt.bar(np.arange(2 * top_features), coef[top_coef])

    feature_names = np.array(features)

    plt.xticks(np.arange(1, 1+2*top_features), feature_names[top_coef],

               rotation=50, ha='right')

    plt.show()



names = X_train.columns

imp_coef(logreg, names)
# Use of a random forest classifier, using CVgrid to compute the hyperparameters



from sklearn.model_selection import GridSearchCV



rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state= 5 , n_jobs=-1)



grid = {

    'n_estimators' : [50, 100, 200,500,700],

    'criterion' : ['gini', 'entropy'],

    'min_samples_split' : [2, 4 ,10, 16],

    'min_samples_leaf' : [1, 5, 10] 

}



gs = GridSearchCV(estimator=rf, param_grid = grid, scoring='accuracy', cv=cv, n_jobs=-1 )

%time

gs = gs.fit(X_train, Y_train)



print(gs.best_score_)

print(gs.best_params_)
rf = RandomForestClassifier(max_features='auto',

                            oob_score=True,

                            random_state=5,

                            n_jobs=-1,

                            n_estimators = 700,

                           criterion='gini',

                           min_samples_leaf=1,

                           min_samples_split = 16)



scores['rf'] = cross_val_score(rf, X_train, Y_train, cv = cv)



rf = rf.fit(X_train, Y_train)

Y_pred['rf'] = rf.predict(X_test).astype(int)
#Get the confusion matrix to see where our model lacks.

print(metrics.confusion_matrix(y_true= Y_train,

                        y_pred = rf.predict(X_train).astype(int)))



print(metrics.classification_report(y_true = Y_train,

                                   y_pred = rf.predict(X_train).astype(int)))
# Creating a coef importance plotting tool for models using feature_importance_ attribute





def imp_feat(model, features, top_features = 20):

    feat = model.feature_importances_.ravel()

    max_feat = np.argsort(feat)[-top_features:]

    

    

    std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)

    feature_names = np.array(features)

    

    # Plotting

    plt.figure(figsize= (15,7))

    b = plt.bar(np.arange(top_features), feat[max_feat], label = feature_names[max_feat], 

                yerr= std[max_feat])

    

    xticks_pos = [0.65*patch.get_width() + patch.get_xy()[0] for patch in b] # usefull line to align Xtick

    plt.xticks(xticks_pos, feature_names[max_feat],

               rotation=45, ha='right')

    plt.xlim([-1, top_features])

    plt.show()



names = X_train.columns

imp_feat(rf, names)
# Fitting 3 other models to create majority vote model

#Support Vector Classifier

sv = SVC(kernel = 'linear', C = 0.25)

scores['sv'] = cross_val_score(sv, X_train, Y_train, cv= cv)



sv = sv.fit(X_train, Y_train)



Y_pred['sv'] = sv.predict(X_test).astype(int)



#AdaBoost Classifier

ada = AdaBoostClassifier(learning_rate = 0.75, n_estimators = 500)

scores['ada'] = cross_val_score(ada, X_train, Y_train, cv=cv)



ada = ada.fit(X_train, Y_train)

Y_pred['ada'] = ada.predict(X_test).astype(int)

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier()

scores['GBC']  = cross_val_score(GBC, X_train, Y_train, cv=cv)



GBC = GBC.fit(X_train, Y_train)

Y_pred['GBC'] = GBC.predict(X_test).astype(int)



imp_feat(GBC, names)
scores = pd.DataFrame(scores)

scores = scores.T

scores['mean'] = scores.iloc[:, :4].mean(axis=1)

scores['std'] = scores.iloc[:, :4].std(axis=1)

scores
Y_pred['Survived'] = 0

Y_pred['Survived'] = Y_pred.iloc[:, 1:5].mode(axis=1)

Y_pred['Survived'] = Y_pred['Survived'].astype(int)
Y_pred.head()
submission = pd.DataFrame({

        "PassengerId" : Y_pred.PassengerId,

        "Survived" : Y_pred['rf']

    })

submission.to_csv('data/submission.csv', index=False)

submission.head()