# Importing generic packages
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import xgboost as xgb


%matplotlib inline
sns.set(style="darkgrid")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings("ignore")
# Creating dataframe from train and test to make sure that the same changes are applied to both sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_len = len(train)
df = pd.concat([train, test], axis=0).reset_index(drop=True)
df.describe().T
df.head().T
df = df.fillna(np.NaN)
df.isnull().sum()
# Favourite functions

def hot_encoder(df, column):
    hot_encoder = pd.get_dummies(df[column])
    hot_encoder.reset_index(drop=True)
    df = df.join(hot_encoder).drop(columns=[column])
    return df

def correlation_graph(dataset):
    fig = plt.figure(figsize=(18, 14))
    corr = dataset.corr()
    c = plt.pcolor(corr)
    plt.yticks(np.arange(0.5, len(corr.index), 1), corr.index)
    plt.xticks(np.arange(0.5, len(corr.columns), 1), corr.columns)
    fig.colorbar(c)

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

def plot_feature_importances(clf, X_train, y_train=None, 
                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):
    '''
    plot feature importances of a tree-based sklearn estimator
    
    Note: X_train and y_train are pandas DataFrames
    
    Note: Scikit-plot is a lovely package but I sometimes have issues
              1. flexibility/extendibility
              2. complicated models/datasets
          But for many situations Scikit-plot is the way to go
          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html
    
    Parameters
    ----------
        clf         (sklearn estimator) if not fitted, this routine will fit it
        
        X_train     (pandas DataFrame)
        
        y_train     (pandas DataFrame)  optional
                                        required only if clf has not already been fitted 
        
        top_n       (int)               Plot the top_n most-important features
                                        Default: 10
                                        
        figsize     ((int,int))         The physical size of the plot
                                        Default: (8,8)
        
        print_table (boolean)           If True, print out the table of feature importances
                                        Default: False
        
    Returns
    -------
        the pandas dataframe with the features and their importanc   
    Author
    ------
        George Fisher
    '''
    __name__ = "plot_feature_importances"
    
    import pandas as pd
    import numpy  as np
    import matplotlib.pyplot as plt
    
#    from xgboost.core     import XGBoostError
#    from lightgbm.sklearn import LightGBMError
    
    try: 
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_train.values, y_train.values.ravel())

            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError("{} does not have feature_importances_ attribute".
                                    format(clf.__class__.__name__))
                
    except (XGBoostError, LightGBMError, ValueError):
        clf.fit(X_train.values, y_train.values.ravel())
            
    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()
    
    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='importance', ascending=False))
        
    return feat_imp    
    
#correlation_heatmap(df)
correlation_graph(df)
g = sns.FacetGrid(df, col='Pclass', hue='Survived')
g = g.map(plt.scatter, 'Fare', 'Age' ).add_legend()
df.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False)
df.Fare.isnull().sum()
df.Fare.fillna(df.Fare.median(), inplace=True)
df.Cabin = df.Cabin.fillna(0).astype(str).str[0]
df.drop(columns = ['PassengerId', 'Name', 'Ticket'], inplace=True)
df.sample(5).T
df = hot_encoder(df, 'Cabin')
correlation_graph(df)
df['B-E'] = df['B']+df['C']+df['D']+df['E']
df['F-T'] = df['F']+df['G']+df['T']
df.drop(columns=['B','C','D','E','F','G','T'], inplace=True)
df = hot_encoder(df, 'Pclass')
to_rename = {1:'First Class', 2:'Second Class', 3:'Third Class', '0':'Classless'}
df.rename(to_rename,axis='columns', inplace=True)
df.columns

df = hot_encoder(df, 'SibSp')
df['2 and more SibSp'] = df[2]+df[3]+df[4]+df[5]+df[8]
to_rename_SibSp = {0:'No SibSp',1:'1 SibSp'}
df.rename(to_rename_SibSp, axis='columns', inplace=True)
df.drop(columns = [2,3,4,5,8], inplace=True)
correlation_graph(df)
df.Age.fillna(df.Age.mean(), inplace=True)
df.Age.plot(kind = 'hist')
a = sns.FacetGrid( df, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , df['Age'].max()))
a.add_legend()
bins = [0, 15, 25, 32, np.inf]
labels=['0-15','15-25','25-32','32-100']
cuts = pd.cut(df['Age'], bins=bins, labels=labels)

df.groupby(cuts)['Survived'].value_counts(normalize=True)
bins = [0, 5, 25, 32, np.inf]
labels=['0-5','15-25','25-32','32-100']
cuts = pd.cut(df['Age'], bins=bins, labels=labels)

df.groupby(cuts)['Survived'].value_counts(normalize=True)
df['Is a child']= df['Age']<5
df['Is a child'] = df['Is a child']*1
df.Sex.replace({'male':0, 'female':1}, inplace=True)
g = sns.factorplot(x='Parch', y='Survived', data=df, kind='bar', size=6, palette = 'muted')
g.despine(left=True)
g = g.set_ylabels('Survival probability', size=16).set_xlabels(size=16)
df = hot_encoder(df, 'Parch')
df['Parch 1-3'] = df[1] + df[2] + df[3]
df['Parch 4-6'] = df[4] + df[5] + df[6] + df[9] 
df.rename(columns={0 :'Parch 0'}, inplace=True)
df.drop(columns=[1,2,3,4,5,6,9], inplace=True)
df.head().T
df.Fare.skew()
df['Fare'] = df['Fare'].map(lambda x: np.log(x) if x>0 else 0)
g = sns.distplot(df.Fare)
df.Fare.skew()
df.Fare.describe()
a = sns.FacetGrid( df, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Fare', shade= True )
a.set(xlim=(0 , df['Fare'].max()))
a.add_legend()
df.loc[df.Fare <= 1.5, 'Fare'] =0
df.loc[(df.Fare > 1.5) & (df.Fare <= 2), 'Fare'] = 1
df.loc[(df.Fare > 2) & (df.Fare <= 2.4), 'Fare'] = 2
df.loc[(df.Fare > 2.4) & (df.Fare <= 4), 'Fare'] = 3
df.loc[df.Fare > 4, 'Fare'] = 4
df.Fare = df.Fare.astype('int32')
g = sns.factorplot(x='Embarked', y='Survived', data=df, kind='bar', size=6, palette = 'muted')
g.despine(left=True)
g = g.set_ylabels('Survival probability', size=16).set_xlabels(size=16)
g = sns.factorplot(x='Embarked', y='Fare', data=df, kind='bar', size=6, palette = 'muted')
g.despine(left=True)
g = g.set_ylabels('Fare, log', size=16).set_xlabels(size=16)
df = hot_encoder(df, 'Embarked')
df.rename(columns ={'S': 'Embarked_S','C':'Embarked_C', 'Q':'Embarked_Q'}, inplace=True)
df.head().T

# TRAINING
# I'm dropping 'Age' as the most important information gathered form the Series is if the passenger was a child or not. 
# Elsewhere the data  Age/Survived is too collineard to be a factor in classification learning techniques
train = df[0:train_len]
test = df[train_len::]
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
X = train.drop(columns=['Survived', 'Age'])
y = train.Survived
test.head()
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
]
log_cols = ['Classifier', 'F1 score']
log = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=10, random_state=0)
# change accuracy for f1 score
acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = f1_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('F1 score')
plt.title('Classifier Accuracy')

sns.set_color_codes('muted')
sns.barplot(x='F1 score', y='Classifier', data=log, palette="Blues_d")

from sklearn.tree import DecisionTreeClassifier

# Build a classification task using 3 informative features
clf = DecisionTreeClassifier()
X_train = X
y_train = y
X_test = test.drop(columns=['Survived','Age'])
y_test = test.Survived
clf.fit(X_train, y_train)


plot_feature_importances(clf, X_train, y_train)
test_predictions = clf.predict(X_test)
#f1_score(y_test, train_predictions)

submission = pd.DataFrame({
        "PassengerId": pd.read_csv('../input/test.csv')["PassengerId"],
        "Survived": test_predictions
    })

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv('titanic.csv', index=False)
submission.head()
