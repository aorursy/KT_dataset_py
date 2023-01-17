#Load Packages
import numpy as np # linear algebra
import matplotlib as mpl
import pandas as pd
import pandas_ml as pdml
from pandas_ml import ConfusionMatrix
import seaborn as sns
import re 
from IPython.display import display_html
import itertools
import math
import random
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import *
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from statistics import variance, stdev, mode
from scipy import interp
#Load sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score, hamming_loss
from sklearn import linear_model
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# Import Classifiers
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import  AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, matthews_corrcoef, log_loss, hinge_loss, cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
#Learning curve
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.model_selection import cross_val_predict,train_test_split, cross_val_score, validation_curve
from sklearn.preprocessing import StandardScaler,  LabelEncoder
from IPython.display import display_html
import warnings


# for inline plots
%matplotlib inline
warnings.filterwarnings('ignore')

mpl.rcParams['figure.figsize'] = (8, 6)
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["axes.labelsize"] = 15
mpl.rc('xtick', labelsize = 15) 
mpl.rc('ytick', labelsize = 15)
sns.set(style = 'whitegrid', palette = 'muted', font_scale = 2)
    
print('Libraries Imported')
# provide some statistics for numerics
def Stats(feature):
    mean  = np.nanmean(X_train[feature])
    median = np.nanmedian(X_train[feature])
    mode_ = stats.mode(X_train[feature])
    
    variation = np.nanvar(X_train[feature])
    stdv = np.nanstd(X_train[feature])
    range_ = np.max(X_train[feature]) - np.min(X_train[feature])
    Quantile = (X_train[feature])
    
    print('Stats of %s:'%(feature.upper()))
    print('Mean: %.2f'%(mean))
    print('Median: %.2f'%(median))
    print('Mode: %.2f'%(mode_[0]))
    print('Range: %.2f'%(range_))    
    print('Variance: %.2f'%(variation))
    print('Standard Deviation: %.2f'%(stdv))
    print('Quantile:')
    for val in [10, 25, 50, 75, 90, 100]:
        perc = np.nanpercentile(X_train[feature],val)
        print('\t%s %%: %.2f'%(val, perc))  
        
        
#sets up the parametes for plotting.. size and font
def PlotParams(Font, sizex, sizey):
    mpl.rcParams['figure.figsize'] = (sizex,sizey)
    plt.rcParams["legend.fontsize"] = Font
    plt.rcParams["axes.labelsize"] = Font
    mpl.rc('xtick', labelsize = Font) 
    mpl.rc('ytick', labelsize = Font)

#sets up Seaborn parametes for plotting
def snsParams(font, colour_scheme):
    #eaborn.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    sns.set(style = 'whitegrid', palette = colour_scheme, font_scale = font)

#determined ht emissing data
def Missing (X):
    total = X.isnull().sum().sort_values(ascending = False)
    percent = round(X.isnull().sum().sort_values(ascending = False)/len(X)*100, 2)
    missing = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
    return(missing) 

#plots number of dataframes side by side
def SideSide(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw = True)

#makes heat map of correllations
def PlotCorr(X):
    corr = X.corr()
    #fig , ax = plt.figure( figsize = (6,6 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    sns.heatmap(
        corr, cmap = cmap, square = True, cbar = False, cbar_kws = { 'shrink' : 1 }, 
     annot = True, annot_kws = { 'fontsize' : 14 }
    )
    plt.yticks(rotation = 0)
    plt.xticks(rotation = 90) 
    
#plot top correlatins in a heat map
def TopCorr(X, lim):
    corr = X.corr()
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    #fig , ax = plt.subplots( figsize = (6,6 ) )
    sns.heatmap(corr[(corr >= lim) | (corr <= -lim)], 
         vmax = 1.0,  cmap = cmap, vmin = -1.0, square = True, cbar = False, linewidths = 0.2, annot = True, 
                annot_kws = {"size": 14})
    plt.yticks(rotation = 0)
    plt.xticks(rotation = 90)
# get data from csv files
test  = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')

#determine sizes of datasets
n_train, m_train = train.shape
n_test, m_test = test.shape

# divide into X and y data
X_train = pd.DataFrame(train.iloc[:,1: m_train])
y_train = pd.DataFrame(train.iloc[0:, 1])
X_test_original = test
X_test = test

print('Data Imported\n\n')
print('FULL DATA')
print('Number of features (m): %.0f'%(m_train))
print('Number of traing samples (n): %.0f'%(n_train))

print('\n\nTest DATA')
print('Number of features (m): %.0f'%(m_test))
print('Number of traing samples (n): %.0f'%(n_test))

cnt = 0
# print out the features
print('\n\nFeatures: ')
for feature in X_train.columns:
    cnt += 1
    print('%d. '%(cnt), feature,'\t\t')

# take a sample of what the data looks like
X_train.head(20)
# provide information about the types of data we are dealing with
print('ORIGINAL TRAINING DATA:\n')
X_train.info()

print('\n\n\nORIGINGAL TEST DATA:\n')
X_test.info()

#summarise the types of data
print('\ndata types of features:')

cnt = 0
d_type = ['float64', 'int64','object','dtype']
print('\n\tTRAIN \t\t TEST')
for c1, c2 in zip(X_train.get_dtype_counts(), X_test.get_dtype_counts()):
    cnt += 1
    print("%s:\t%-9s \t%s"%(d_type[cnt],c1, c2))
    
#finds missing values
missing_train = Missing(X_train)
missing_test = Missing(X_test)
    
print('TRAIN DATA','\t\t','TEST DATA')
SideSide(missing_train, missing_test)
X_train.describe(include = "all")
X_train.hist(figsize = (16,10),bins = 20)
sns.pairplot(X_train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked']], 
             hue = 'Survived', palette = 'muted',size = 2.2,
             diag_kind = 'kde', dropna = True, diag_kws = dict(shade = True), plot_kws = dict(s=20) )

X_train.skew()

skew_train = pd.DataFrame(X_train.skew())
skew_test = pd.DataFrame(X_test.skew())
    
print('TRAIN DATA','\t\t','TEST DATA')
SideSide(skew_train, skew_test)
#show the correlations between all the featured in a heatmap
plt.figure(figsize = (20,6))
PlotCorr(X_train);
# highest correlated with correlation of features with 'Survived'
print('Featured hights correlation with survival')
print('Feature\tCorrelation')
Survive_Corr = X_train.corr()["Survived"]
Survive_Corr = Survive_Corr[1:9] # remove the 'Survived'
Survive_Corr= Survive_Corr[np.argsort(Survive_Corr, axis = 0)[::-1]] #sort in descending order
print(Survive_Corr)

# plot survival count for male and female
plt.figure(figsize = (10,6))
ax = sns.barplot(x = np.arange(len(Survive_Corr)), y = np.array(Survive_Corr.values), palette = 'muted', orient= 'v');
ax.set_xlabel("Feature",fontsize = 15)
ax.set_ylabel("Correlation Coefficient (with Survival)",fontsize = 15)
ax.set_xticklabels(Survive_Corr.index)

f,ax = plt.subplots(1,2,figsize =(18,8))
X_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
plt.tight_layout()
sns.countplot('Survived',data=X_train,ax=ax[1],palette="muted")
ax[1].set_title('Survived')
snsParams(2, 'muted')
# plot survival count for male and female
plt.figure(figsize = (20,5))
plt.subplot(1, 3, 1)
ax = sns.countplot(x = 'Survived',hue = 'Sex', data = X_train);
ax.set_xlabel("Survived",fontsize = 15)
ax.set_ylabel("Count",fontsize = 15)
ax.legend(fontsize = 14)


#survival probability of males and females
plt.subplot(1, 3, 2)
ax = sns.barplot(x = "Sex", y = "Survived",data = X_train)
ax = ax.set_ylabel("Survival Probability")

plt.subplot(1, 3, 3)
sns.violinplot(y = 'Survived', x = 'Sex', data = X_train, inner = 'quartile')

Stats('Age')

# plot survival number for age dependandcy
fig, axes = plt.subplots(figsize = (20,6), nrows = 1, ncols = 3)

ax = sns.distplot(X_train[X_train['Survived'] == 1].Age.dropna(), bins = 20, label = 'Survived')
ax = sns.distplot(X_train[X_train['Survived'] == 0].Age.dropna(), bins = 20, label = 'Not Survived')

ax = sns.kdeplot(X_train["Age"][(X_train["Survived"] == 0) & (X_train["Age"].notnull())], color = "Green", shade = False)
ax = sns.kdeplot(X_train["Age"][(X_train["Survived"] == 1) & (X_train["Age"].notnull())], ax = ax, color = "Blue", shade= False)

ax.set_xlabel("Age",fontsize = 15)
ax.set_ylabel("Frequency",fontsize = 15)
ax = ax.legend(["Not Survived","Survived"],fontsize = 15)
plt.xlim(0,80)
plt.ylim(0,0.04)
plt.grid(True)

women = X_train[X_train['Sex'] == 'female']
men = X_train[X_train['Sex'] == 'male']

#For womwn
ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins = 20, label = 'survived', ax = axes[0], kde = False)
ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins = 20, label = 'not survived', ax = axes[0], kde = False)
ax.set_xlabel("Age",fontsize = 15)
ax.set_ylabel("Count",fontsize = 15)
ax.legend(fontsize = 15)
ax.set_title('Female', fontsize = 15)
ax.set(xlim = (0, X_train['Age'].max()));
ax.set(ylim = (0, 50));
    
    
#For men
ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins = 20, label = 'survived', ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins = 20, label = 'not survived', ax = axes[1], kde = False)
ax.set_xlabel("Age",fontsize = 15)
ax.set_ylabel("Count",fontsize = 15)
ax.legend(fontsize = 15)
ax.set_title('Male', fontsize = 15)
ax.set(xlim = (0, X_train['Age'].max()))
ax.set(ylim = (0, 50));




plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
sns.boxplot( y = "Age", x = "Survived",data = X_train, palette = "muted")

plt.subplot(2,3,2)
sns.violinplot("Pclass","Age", hue = "Survived", data = X_train, split = True, palette = 'muted')

plt.subplot(2,3,3)
sns.violinplot("Sex","Age", hue = "Survived", data = X_train, split = True, palette = 'muted')

plt.subplot(2,3,4)
sns.boxplot(y = "Age", x = "Sex", data = X_train, palette = "muted")

plt.subplot(2,3,5)
sns.boxplot(y = "Age", x = "Sex", hue = "Pclass", data = X_train, palette = "muted")

plt.subplot(2,3,6)
sns.boxplot(y = "Age", x = "Parch", data = X_train, palette = "muted")

plt.subplot(2,3,4)
sns.boxplot(y = "Age", x = "SibSp", data = X_train, palette = "muted")
plt.figure(figsize = (16,10))
plt.subplot(2, 3, 1)
sns.barplot(x = 'Pclass', y = 'Survived', data = X_train)

plt.subplot(2, 3, 2)
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=X_train);

plt.subplot(2, 3, 3)
sns.barplot(x = "Pclass", y = "Survived", hue = "Sex", data = X_train)

plt.subplot(2, 3, 4)
sns.countplot(x = 'Survived',hue = 'Pclass',data = X_train);

plt.subplot(2, 3, 5)
sns.violinplot(y = 'Survived', x = 'Pclass', data = X_train, inner = 'quartile')
plt.subplot(2, 3, 6)
sns.violinplot(x='Pclass', y = 'Age', hue = 'Survived', data = X_train, split = True)


plt.subplots(figsize=(16,5))
plt.subplot(131)
sns.boxplot(x = "Pclass", y = "Age", hue = "Sex", data = X_train);
plt.ylim(0,90)

plt.subplot(132)
sns.boxplot(y = "Age", x = "Sex", hue = "Pclass", data = X_train)
plt.subplot(133)
X_train.Age[X_train['Pclass'] == 1].plot(kind = 'kde')    
X_train.Age[X_train['Pclass'] == 2].plot(kind = 'kde')
X_train.Age[X_train['Pclass'] == 3].plot(kind = 'kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes", fontsize = 15)
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'), loc = 'best') ;
plt.xlim(0,80)
plt.ylim(0,0.04)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(X_train, col = 'Survived', row = 'Pclass', size = 3, aspect = 3.2)
grid.map(plt.hist, 'Age', alpha = 0.8, bins=20)
grid.add_legend();
# Explore Embarked vs Survived 
plt.figure(figsize = (16,6))
plt.subplot(1, 3, 1)
sns.barplot(x = "Embarked", y = "Survived",  data = X_train)

# Explore Pclass vs Survived by Sex
plt.subplot(1, 3, 2)
sns.barplot(x = "Embarked", y = "Survived", hue = "Sex", data = X_train)
#g = g.set_ylabels("survival probability")
plt.subplot(1, 3, 3)
sns.countplot(x = 'Survived',hue = 'Embarked',data = X_train);
plt.figure(figsize = (15,5))
sns.boxplot(y = "Age", x = "Embarked", hue = "Pclass", data = X_train)
# Explore Pclass vs Embarked 
PlotParams(15, 8, 6)
snsParams(2,'muted')

g = sns.factorplot("Pclass", col = "Embarked",  data = X_train, size = 8, 
                   kind = "count", palette = "muted")
g = g.set_ylabels("Count")
g = sns.factorplot("Pclass", col = "Embarked",  data = X_train,
                   hue = "Sex", size = 8, kind = "count", palette = "muted")

g = g.set_ylabels("Count")

PlotParams(15, 10, 6)
plt.figure(figsize = (16,5))
plt.subplot(1, 2, 1)
sns.barplot(x = "Parch", y = "Survived",  data = X_train, palette = "muted")
plt.subplot(1, 2, 2)
sns.barplot(x = "SibSp", y = "Survived",  data = X_train, palette = "muted")

plt.figure(figsize=(20,5))
plt.subplot(1, 2, 1)
sns.violinplot(y = 'Survived', x = 'Parch', data = X_train, palette = "muted", inner = 'quartile')
plt.subplot(1, 2, 2)
sns.violinplot(y = 'Survived', x = 'SibSp', data = X_train, palette = "muted", inner = 'quartile')
PlotParams(15, 8, 6)

plt.figure(figsize = (20,6))
plt.subplot(1,3,1)
sns.kdeplot(X_train["Fare"])
plt.xlim(0,160)
plt.ylim(0,.040)
plt.xlabel('Fare')
plt.ylabel('Survival Probability')

plt.subplot(1,3,2)
ax = sns.distplot(X_train[X_train['Survived'] == 1].Fare.dropna(), bins = 80, label = 'Survived')
ax = sns.distplot(X_train[X_train['Survived'] == 0].Fare.dropna(), bins = 80, label = 'Not Survived')
ax = sns.kdeplot(X_train["Fare"][(X_train["Survived"] == 0) & (X_train["Fare"].notnull())], color = "Green", shade = False)
ax = sns.kdeplot(X_train["Fare"][(X_train["Survived"] == 1) & (X_train["Fare"].notnull())], ax = ax, color = "Blue", shade= False)
ax.set_xlabel("Fare",fontsize = 15)
ax.set_ylabel("Frequency",fontsize = 15)
ax = ax.legend(["Not Survived","Survived"],fontsize = 15)
plt.ylim(0,0.1)
plt.xlim(0,160)
plt.grid(True)

plt.subplot(1,3,3)
ax1 = sns.boxplot(x = "Embarked", y = "Fare", hue = "Pclass", data = X_train);
plt.ylim(0,200)
Stats('Fare')
# Fill empty values with NaN
X_train = X_train.fillna(np.nan)
X_test = X_test.fillna(np.nan)

#finds missing values
missing_train = Missing(X_train)
missing_test = Missing(X_test)
    
print('TRAIN DATA','\t\t','TEST DATA')
SideSide(missing_train, missing_test)

#plot missing data in heatmap for visualisation
print('\n\n  MISSING TRAINING DATA \t\t\t MISSING TEST DATA')
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
plt.figure(figsize = (10,5));
plt.subplot(1, 2, 1)
sns.heatmap(X_train.isnull(), yticklabels = False, cbar = False, cmap = cmap)
plt.subplot(1, 2, 2)
sns.heatmap(X_test.isnull(), yticklabels = False, cbar = False,cmap = cmap);

#combine the tets and training data so that operations can be performed together
full_data = [X_train, X_test] 
#fill in Embarked datta with S as it is the most common
for X in full_data:
    X['Embarked'] = X['Embarked'].fillna("S")
# fill missing Fare with median fare for each Pclass
for X in full_data:
    X["Fare"].fillna(X.groupby("Pclass")["Fare"].transform("median"), inplace = True)
    X["Fare Group"] = X["Fare"]
PlotCorr(X_train[["Age","Sex","SibSp","Parch","Pclass"]])

#correlation of features with target variable
Age_Corr = X_train.corr()["Age"]
#Age_Corr= Age_Corr[np.argsort(Age_Corr, axis = 0)[::-1]] #sort in descending order
Age_Corr = Age_Corr[1:10] # remove the 'Survived'
print(Age_Corr)
from sklearn.ensemble import RandomForestRegressor

#use random forest to predict age
def MissingAges(X, AGE_features):
    
    age_data = X[age_features]

    known_ages = age_data[age_data['Age'].notnull()].as_matrix()
    unknown_ages = age_data[age_data['Age'].isnull()].as_matrix()

    # Create target and eigenvalues for known ages
    target = known_ages[:, 0]
    eigen_val = known_ages[:, 1:]

    # apply random forest regressor
    RFR_age = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
    RFR_age.fit(eigen_val, target)


    return (RFR_age, unknown_ages)

# age distribution BEFORE filling in missing values
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values')
axis2.set_title('New Age values')
# plot new Age Values
X_train['Age'].hist(bins = 70, ax = axis1)
plt.xlabel('Age')
plt.ylabel('Counts')
plt.xlim(0,80)

#the features used to determine missing ages
age_features = ["Age", "SibSp", "Parch", "Pclass"]

# filling ing the training data
RFR_age, unknown_ages_train = MissingAges(X_train, age_features)
Age_predictions_train = RFR_age.predict(unknown_ages_train[:, 1::])
X_train.loc[(X_train['Age'].isnull()), "Age"] = Age_predictions_train
X_train["Age"] = X_train["Age"].astype(int)

# filling in the test data
_, unknown_ages_test = MissingAges(X_test, age_features)
Age_predictions_test = RFR_age.predict(unknown_ages_test[:, 1::])
X_test.loc[(X_test['Age'].isnull()), "Age"] = Age_predictions_test
X_test["Age"] = X_test["Age"].astype(int)


# age distribution AFTER filling in missing values
X_train['Age'].hist(bins = 70, ax = axis2)
plt.xlabel('Age')
plt.ylabel('Counts')
plt.xlim(0,80)
print('Ages filled in')
# cabin Vrs no cabine survival rates
for X in full_data:
    X["CabinBool"] = (X["Cabin"].notnull().astype('int'))
    
#draw a bar plot of CabinBool vs. survival
sns.barplot(x = "CabinBool", y = "Survived", data = X_train)
plt.show()
# Extract deck 
def extract_cabin(x):
    return x != x and 'Other' or x[0]

for X in full_data:
    X['Cabin'] = X['Cabin'].apply(extract_cabin)
    X['Deck'] = X['Cabin']

train_deck = pd.DataFrame(X_train.groupby('Deck').size())
test_deck = pd.DataFrame(X_test.groupby('Deck').size())

print('TRAIN \t\t TEST')
SideSide(train_deck,test_deck )
snsParams(1.2, 'muted')
plt.figure(figsize = (16,5))

plt.subplot(1, 3, 1)
g = sns.countplot(X_train["Cabin"], palette = "muted")
plt.subplot(1, 3, 2)
g = sns.barplot(x = "Deck", y = "Survived",  data = X_train, palette = "muted")

plt.subplot(1, 3, 3)
sns.countplot(x = 'Survived',hue = 'Deck',data = X_train, palette = "muted");

snsParams(2, 'muted')
plt.figure(figsize = (16,5))
g = sns.factorplot("Deck", col = "Pclass",  data = X_train, size = 8, 
                   kind = "count", palette = "muted")
g = g.set_ylabels("Count")
g = sns.factorplot("Deck", col = "Embarked",  data = X_train,
                   hue = "Sex", size = 8, kind = "count", palette = "muted")
g = g.set_ylabels("Count")
PlotParams(15, 8, 6)
# determine size of family on board
for X in full_data:
    X['Family Size'] = X['SibSp'] + X['Parch'] + 1 
    X['Alone'] = [1 if i<2 else 0 for i in X['Family Size']]
    X['Surname'] = X['Name'].str.extract('(\w+),', expand = False)
    X['Large Family'] = [1 if i > 5 else 0 for i in X['Family Size']]
    
    #X['First Name'] = X['Name'].str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)',expand = False)[1]
    
axes = sns.factorplot('Family Size','Survived', hue = 'Sex', data = X_train, aspect = 2)
plt.grid(True)
axes = sns.factorplot('Family Size','Survived',  data = X_train, aspect = 2)
plt.grid(True)
pd.crosstab(X_train['Family Size'], X_train['Survived']).plot(kind = 'bar', stacked = True)
    

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(18,6))
sns.barplot(x = "Family Size", y = "Survived", hue = "Sex", data = X_train, ax = axis1);
sns.barplot(x = "Alone", y = "Survived", hue = "Sex", data = X_train, ax = axis2);
sns.barplot(x = "Alone", y = "Survived", data = X_train)
plt.show()
   

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Create a new feature Title, containing the titles of passenger names
for X in full_data:
    X['Title'] = X['Name'].apply(get_title)
    
# Group all non-common titles into one single grouping "Rare"
for X in full_data:
    X['Title'] = X['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona'], 'Noble')
    X['Title'] = X['Title'].replace(['Capt', 'Col', 'Dr', 'Major', 'Rev'], 'Officer')
    X['Title'] = X['Title'].replace('Mlle', 'Miss')
    X['Title'] = X['Title'].replace('Ms', 'Miss')
    X['Title'] = X['Title'].replace('Mme', 'Mrs')

    
print('TRAIN TITLE \t TEST TITLES')
train_titles = pd.DataFrame(X_train.Title.value_counts())
test_titles = pd.DataFrame(X_test.Title.value_counts())

SideSide(train_titles,test_titles)

plt.figure(figsize = (16,6))
plt.subplot(1, 3, 1)
g = sns.barplot(x = "Title", y = "Survived",  data = X_train)
plt.xticks(rotation = 90)

plt.subplot(1, 3, 2)
sns.countplot(x = 'Survived', hue = 'Title',data = X_train);
plt.xticks(rotation = 90)

plt.subplot(1, 3, 3)
sns.boxplot(data = X_train, x = "Title", y = "Age");
plt.xticks(rotation = 90)

tab = pd.crosstab(X_train['Title'], X_train['Pclass'])
tab_prop = tab.div(tab.sum(1).astype(float), axis=0)

tab_prop.plot(kind = "bar", stacked = True)
plt.xticks(rotation = 90)
#sort the ages into logical categories
## create bins for age
def AgeCategory(age):
    a = ''
    if age <= 3:
        a = 'Baby'
    elif age <= 12: 
        a = 'Child'
    elif age <= 18:
        a = 'Teenager'
    elif age <= 35:
        a = 'Young Adult'
    elif age <= 65:
        a = 'Adult'
    elif age == 'NaN':
        a = 'NaN'
    else:
        a = 'Senior'
    return a
        
for X in full_data:
    X['Age Group'] = X['Age'].map(AgeCategory)
    X['Age*Class'] = X['Age'] * X['Pclass']

plt.figure(figsize = (16,6))
plt.subplot(1, 3, 1)
g = sns.barplot(x = "Age Group", y = "Survived",  data = X_train)
plt.xticks(rotation = 90)

plt.subplot(1, 3, 2)
sns.countplot(x = 'Survived', hue = 'Age Group',data = X_train)

plt.subplot(1, 3, 3)
sns.boxplot(data = X_train, x = "Age Group", y = "Age");
plt.xticks(rotation = 90)
def GetPerson(X):
    age, sex = X
    return 'child' if age < 16 else sex

for X in full_data:
    X['Person'] = X[['Age','Sex']].apply(GetPerson, axis = 1)

    
for X in full_data:
    X.loc[ X['Fare Group'] <= 7.91, 'Fare Group'] = 0
    X.loc[(X['Fare Group'] > 7.91) & (X['Fare Group'] <= 14.454), 'Fare Group'] = 1
    X.loc[(X['Fare Group'] > 14.454) & (X['Fare Group'] <= 31), 'Fare Group']   = 2
    X.loc[(X['Fare Group'] > 31) & (X['Fare Group'] <= 99), 'Fare Group']   = 3
    X.loc[(X['Fare Group'] > 99) & (X['Fare Group'] <= 250), 'Fare Group']   = 4
    X.loc[X['Fare Group'] > 250, 'Fare Group'] = 5
    X['Fare Group'] = X['Fare Group'].astype(int)   
#map each Sex value to a numerical value
sex_map = {"male": 0, "female": 1}
person_map = {'child': 0, "male": 1, "female": 2}
Embark_map = {"C": 1,"S": 2, "Q": 3}
deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "Other": 9}
age_map = {"Baby": 1, "Child": 2, "Teenager": 3, "Young Adult": 4, "Adult": 5, "Senior": 6}
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Officer": 5, "Noble": 5}

for X in full_data:
    X["Sex"] = X["Sex"].map(sex_map)
    X["Embarked"] = X["Embarked"].map(Embark_map)
    X["Person"] = X["Person"].map(person_map)
    X["Deck"] = X["Deck"].map(deck_map)
    X["Age Group"] = X["Age Group"].map(age_map)
    X["Title"] = X["Title"].map(title_mapping)
X_train = X_train.drop("Name", axis = 1) 
X_test = X_test.drop("Name", axis = 1) 
X_train = X_train.drop("Ticket", axis = 1) 
X_test = X_test.drop("Ticket", axis = 1) 
X_train = X_train.drop("Cabin", axis = 1) 
X_test = X_test.drop("Cabin", axis = 1) 
X_train = X_train.drop("Surname", axis = 1) 
X_test = X_test.drop("Surname", axis = 1) 
#X_train = X_train.drop("Age", axis = 1) 
#X_test = X_test.drop("Age", axis = 1) 
X_test = X_test.drop("PassengerId", axis = 1) 
X_train.head()
# Feature Scaling

def Norm(X):
    X1 = (X - np.mean(X)) / (np.max(X) - np.min(X))
    return(X1)

X_train['Fare'] = Norm(X_train['Fare'])
X_train['Age'] = Norm(X_train['Age'])
X_test['Fare'] = Norm(X_test['Fare'])
X_test['Age'] = Norm(X_test['Age'])

#X_train['Fare'] = X_train['Fare'].astype(int)
#X_train['Age'] = X_train['Age'].astype(int)
#X_test['Fare'] = X_test['Fare'].astype(int)
#X_test['Age'] = X_test['Age'].astype(int)
#X_train['Age*Class'] = X_train['Age*Class'].astype(int)
#X_test['Age*Class'] = X_test['Age*Class'].astype(int)
X_train.head(10)
plt.figure(figsize = (20,12))
PlotCorr(X_train);
plt.figure(figsize = (20,12))
TopCorr(X_train, 0.25)
# highest correlated with correlation of features with 'Survived'
print('Featured hights correlation with survival')
print('Feature\tCorrelation')
Survive_Corr = X_train.corr()["Survived"]
Survive_Corr = Survive_Corr[1:20] # remove the 'Survived'
Survive_Corr= Survive_Corr[np.argsort(Survive_Corr, axis = 0)[::-1]] #sort in descending order
print(Survive_Corr)


correlations = X_train.corr() # determines parameters that are correlated to Survival
# most correlated featues = features with correlation to Survival >0.1
top_correlations = correlations.index[abs(correlations["Survived"]) > 0.1]
plt.figure(figsize=(12,10))
sns.set(font_scale = 1.5)
g = sns.heatmap(X_train[top_correlations].corr(), annot = True, cmap = cmap, annot_kws={"size": 10})
plt.title('Features most correlated with Survival (>0.1)')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)

snsParams(2, 'muted')
plt.figure(figsize = (10,6))
ax = sns.barplot(x = np.arange(len(Survive_Corr)), y = np.array(Survive_Corr.values), 
                 palette = 'muted', orient= 'v');
ax.set_xlabel("Feature",fontsize = 15)
ax.set_ylabel("Correlation Coefficient",fontsize = 15)
ax.set_xticklabels(Survive_Corr.index)
plt.xticks(rotation = 90)
# can drop a few more features
X_train = X_train.drop("Age", axis = 1) 
X_test = X_test.drop("Age", axis = 1) 
X_train = X_train.drop("SibSp", axis = 1) 
X_test = X_test.drop("SibSp", axis = 1) 
X_train = X_train.drop("Parch", axis = 1)
X_test = X_test.drop("Parch", axis = 1)
X_train = X_train.drop("Family Size", axis = 1)
X_test = X_test.drop("Family Size", axis = 1)
X_train = X_train.drop("Age Group", axis = 1)
X_test = X_test.drop("Age Group", axis = 1)

# final features
Survive_Corr = X_train.corr()["Survived"]
Survive_Corr = Survive_Corr[1:9] # remove the 'Survived'
Survive_Corr= Survive_Corr[np.argsort(Survive_Corr, axis = 0)[::-1]] #sort in descending order

snsParams(2, 'muted')
plt.figure(figsize = (10,6))
ax = sns.barplot(x = np.arange(len(Survive_Corr)), y = np.array(Survive_Corr.values), 
                 palette = 'muted', orient= 'v');
ax.set_xlabel("Feature",fontsize = 15)
ax.set_ylabel("Correlation Coefficient",fontsize = 15)
ax.set_xticklabels(Survive_Corr.index)
plt.xticks(rotation = 90)

X_train = X_train.drop("Survived", axis = 1)
sns.pairplot(X_train)
print('TRAINING')
print(X_train.info())
print('\n\nTEST')
print(X_train.info())

X_train.head(0)
X_test.head(0)

cnt = 0
d_type = ['float64', 'int64','object','dtype']
print('\n\tTRAIN \t\t TEST')
for c1, c2 in zip(X_train.get_dtype_counts(), X_test.get_dtype_counts()):
    cnt += 1
    print("%s:\t%-9s \t%s"%(d_type[cnt],c1, c2))
    
    classes = ['Dead','Survived']
    cv = ShuffleSplit(n_splits = 100, test_size = 0.25, random_state = 0)
    train_sizes = np.linspace(.1, 1.0, 10)
def GridSearcher(model, GridParams, X, y):
    print('Performing Grid Search...')
    kfold = StratifiedKFold(n_splits = 10)
    
    model_GS = GridSearchCV(estimator = model, param_grid = GridParams, cv = kfold, scoring = "accuracy", n_jobs = 2, verbose = 1)

    model_GS.fit(X, y['Survived'])
    
    model_best = model_GS.best_estimator_
    model_best_score = model_GS.best_score_
    model_best_params = model_GS.best_params_
    
    print("\nBest Estimatot:", model_GS.best_estimator_,
          "\nBest Score:", model_GS.best_score_, # Mean cross-validated score of the best_estimator
          "\nBest parameters:", model_GS.best_params_)
          
    return (model_best, model_best_score, model_best_params)
    
def Confuse(y, y_pred, classes):
    cnf_matrix1 = confusion_matrix(y, y_pred)
    
    cnf_matrix = cnf_matrix1.astype('float') / cnf_matrix1.sum(axis = 1)[:, np.newaxis] *100
    c_train = pd.DataFrame(cnf_matrix, index = classes, columns = classes)  
    plt.subplot(2, 3, 3)
    ax = sns.heatmap(c_train, annot = True, cmap = cmap, square = True, cbar = False, 
                          fmt = '.2f', annot_kws = {"size": 20})
    plt.title('Confusion Matrix (%)')
    
    return(ax, cnf_matrix1)

def FitModel(model, X, y):
    print('Fitting Model...')
    model.fit(X, y)
    y_pred  = model.predict(X)
    CV_score = round(np.median(cross_val_score(model, X, y, cv = cv)), 4) * 100
    
    return (model, y_pred, CV_score)

def LearningCurve(X, y, model, cv, train_sizes, title):
    print('Evaluating Learning Curve...')
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv = cv, n_jobs = 4, 
                                                            train_sizes = train_sizes)

    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std   = np.std(train_scores, axis = 1)
    val_scores_mean  = np.mean(val_scores, axis = 1)
    val_scores_std   = np.std(val_scores, axis = 1)
    
    train_Error_mean = np.mean(1- train_scores, axis = 1)
    train_Error_std  = np.std(1 - train_scores, axis = 1)
    val_Error_mean  = np.mean(1 - val_scores, axis = 1)
    val_Error_std   = np.std(1 - val_scores, axis = 1)

    train_sc = train_scores_mean[-1] 
    val_sc = val_scores_mean[-1]
    
    train_sc_std = train_scores_std [-1]
    val_sc_std = val_scores_std[-1]
    
    Learn_Results = [train_sc * 100, train_sc_std * 100, val_sc * 100, val_sc_std * 100]
    
    plt.figure(figsize = (20,15))
    plt.subplot(2, 3, (1,2))
    plt.fill_between(train_sizes, train_Error_mean - train_Error_std,
                     train_Error_mean + train_Error_std, alpha = 0.1, color = "r")
    plt.fill_between(train_sizes, val_Error_mean - val_Error_std, 
                     val_Error_mean + val_Error_std, alpha = 0.1, color = "g")
    plt.plot(train_sizes, train_Error_mean, 'o-', color = "r",label = "Training Error")
    plt.plot(train_sizes, val_Error_mean, 'o-', color = "g",label = "Cross-validation Error")
    plt.xlabel('Training Examples (m)')
    plt.title('Learning Curve %s'%(title))
    plt.ylabel('Error')
    plt.legend(loc = "best")
    plt.grid(True)
     
    return (Learn_Results)
    
def PlotPrecisionRecall(model, X, y):

    # getting the probabilities of our predictions
    y_scores = model.predict_proba(X) # probability estimates

    
    y_scores = y_scores[:,1]
    
    precision, recall, threshold = precision_recall_curve(y, y_scores)

    plt.subplot(2,3,4)
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth = 2)
    plt.plot(threshold, recall[:-1], "b", label = "Recall", linewidth = 2)
    plt.xlabel("Threshold", fontsize = 19)
    plt.ylabel("Precision or Recall", fontsize = 19)
    plt.title("Precision & Recall", fontsize = 19)
    plt.legend(loc = "best", fontsize = 19)
    plt.ylim([0, 1])

    plt.subplot(2,3,5)
    plt.plot(recall[:-1], precision[:-1], color = "r", linewidth = 2)
    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, step = 'post', alpha = 0.2,
                 color = 'b')
    #plt.plot(threshold,  color = "g", label = "recall", linewidth = 2)
    plt.title("Precision - Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1])
    plt.xlim([0.0, 1])
    plt.legend(loc = "best")

def PlotROC(model, X, y):

    print('Evaluating ROC Curve...')
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    random_state = np.random.RandomState(0)

    i = 0
    y = y['Survived']
    
    for train, test in cv.split(X,y):
        prob = model.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])[:,1]
        fpr, tpr, t = roc_curve(y[test], prob)
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        i= i + 1
        
    plt.subplot(2, 3, 6)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw = 2, alpha=1)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='red', alpha = .3,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.0, 1.0])
    plt.ylim([-0.0, 1.0])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = "best")
    plt.show()
    
    return()

def PlotKS(model, X, y):
    y_probas = model.predict_proba(X)
    
    skplt.metrics.plot_ks_statistic(y, y_probas, title = 'Kolmogorov–Smirnov test', 
                                    text_fontsize = 13, title_fontsize = 13, figsize = [6,5])
        
def PlotLift(model, X, y):
    y_probas = model.predict_proba(X)
    skplt.metrics.plot_lift_curve(y, y_probas,title = 'Lift Curve', 
                                  text_fontsize = 13, title_fontsize = 13, figsize = [6,5])

def PlotCumGain(model, X, y):
    y_probas = model.predict_proba(X)
    skplt.metrics.plot_cumulative_gain(y, y_probas, title = 'Cumulative Gain',
                                       text_fontsize = 13, title_fontsize = 13, figsize = [6,5])

def PlotPR(model, X, y):

    y_probas = model.predict_proba(X)
    skplt.metrics.plot_precision_recall(y, y_probas,
                                        text_fontsize = 13, title_fontsize = 13, figsize = [6,5])

def Classification_Analysis(model_best, title, title_abrv,X, X_test, y):
    
    #Fitting Model
    (model_best, y_pred, CV_score) = FitModel(model_best, X, y)

    y_train_pred = pd.Series(model_best.predict(X), name = title_abrv)
    y_test_pred = pd.Series(model_best.predict(X_test), name = title_abrv)
    
    # Learning Curve Analysis
    LearnResults = LearningCurve(X, y, model_best, cv, train_sizes, title)
    #Confuson Matrix
    Confuse_fig, cnf_matrix = Confuse(y, y_train_pred, classes)
    #Precision - Recall Curve
    PlotPrecisionRecall(model_best, X, y)
    #plt scikit-plot
    PlotROC(model_best, X, y)
    PlotKS(model_best, X, y)
    PlotLift(model_best, X, y)
    PlotPR(model_best, X, y)
    PlotCumGain(model_best, X, y)
    
    Summary = PrintResults(title, model_best,X_train,
                          y, y_train_pred, CV_score, LearnResults, cnf_matrix)

    return (Summary, y_train_pred, y_test_pred)

def TreeImportance():
    nrows = ncols = 2
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

    names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

    nclassifier = 0
    for row in range(nrows):
        for col in range(ncols):
            name = names_classifiers[nclassifier][0]
            classifier = names_classifiers[nclassifier][1]
            indices = np.argsort(classifier.feature_importances_)[::-1][:40]
            g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , 
                            orient='h',ax=axes[row][col])
            g.set_xlabel("Relative importance",fontsize=12)
            g.set_ylabel("Features",fontsize=12)
            g.tick_params(labelsize=9)
            g.set_title(name + " feature importance")
            nclassifier += 1
            
#def PrintResults(y, y_pred, Confuse_fig, learn_fig, CV_score, Score_mean, Scores_std):
def PrintResults(title, model,X, y, y_pred, CV_score, LearnResults, cnf_matrix ):
    y_scores = model.predict_proba(X)[:, 1]
    
    precision = precision_score(y, y_pred, average = 'macro') * 100
    recall = recall_score(y, y_pred,average = 'macro') * 100
    f1score = f1_score(y, y_pred,average = 'macro') * 100
    Accuracy = accuracy_score(y, y_pred)
    MCC = matthews_corrcoef(y, y_pred) 
    Lg_loss = log_loss(y, y_pred)
    Zero_one_loss= zero_one_loss(y, y_pred, normalize = False)
    Hinge = hinge_loss(y, y_pred) 
    Cohen_kappa = cohen_kappa_score(y, y_pred) 
    Hamming = hamming_loss(y, y_pred)
    AUC = roc_auc_score(y, y_scores)
    Brier = brier_score_loss(y, y_scores )
    
    Population = np.sum(cnf_matrix)  
    PP = np.sum(y == 1)
    NP = np.sum(y == 0)
    PP_t = np.sum(cnf_matrix[:,1])
    NP_t = np.sum(cnf_matrix[:,0])
    TP = cnf_matrix[1,1]
    
    TN = cnf_matrix[0,0]
    FP = cnf_matrix[0,1]
    FN = cnf_matrix[1,0]
    
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)
    FPR = FP/(FP + TN)
    FNR = FN/(FN + TP)
    
    P_sum = np.sum(cnf_matrix[:,1])
    N_sum = np.sum(cnf_matrix[:,0])
    PPV = TP/P_sum
    NPV = TN/N_sum
       
    FDR = FP/P_sum
    FOR = FN/N_sum
    
    Acc = (TP + TN)/(P_sum + N_sum)
    F1_scr = (2 * TP) / (2*TP + FP + FN)
    MCC = ((TP * TN) - (FP * FN))/np.sqrt(P_sum * N_sum * (TP + FN) * (TN + FP))
    
    BM = TPR + TNR - 1
    MK = PPV + NPV - 1 
    
    LR_minus = FNR/TNR
    LR_plus = TPR/FPR
    DOR = LR_plus/LR_minus
    
    
    print('\nRESULTS: %s'%(title.upper()),'CLASSIFIER')
    print('--------------------------------')
    print('\n Model Settings: %s'%(title.upper()),'CLASSIFIER')
    print('\t %s'%(model))
    print('--------------------------------')
    print('\nLearning Curve Results %s'%(title.upper()),'CLASSIFIER')
    print('\tTraining')
    print('\t\tScore: %.2f %%'%(LearnResults[0]))
    print('\t\tStdv: %.2f %%'%(LearnResults[1]))
    print('\tValidation')
    print('\t\tScore: %.2f%%'%(LearnResults[2]))
    print('\t\tStdv: %.2f %%'%(LearnResults[3]))
    print('-------------------------------------------------------')
    print('\nFull Fitting Results %s'%(title.upper()),'CLASSIFIER')
    print('\tAccuracy Score: %.2f %%'%(Accuracy*100))
    print('\tCross-Validation Score: %.2f %%'%(CV_score))
    print("\tPrecision: %.2f %%"%(precision))
    print("\tRecall: %.2f %%"%(recall))
    print('\tf1-score: %.2f %%'%(f1score))
    print('\tC-Statistic or (AUC-ROC): %.2f %%'%(AUC * 100))
    print('\tCohens Kappa: %.2f %%'%(Cohen_kappa * 100))
    print('\tKolmogorov–Smirnov (KS) Statistic: %.2f'%(00))
    
    print('\nLosses: %s'%(title.upper()),'CLASSIFIER')
    print('\tLog Loss: %.2f'%(Lg_loss))
    print('\tZero-One-Loss: %.2f'%(Zero_one_loss))
    print('\tHamming Loss: %.2f'%(Hamming))
    print('\tBrier Loss: %.2f'%(Brier))
    print('\tHinge Loss: %.2f'%(Hinge))    
    print('-------------------------------------------------------')
    print('\nConfusion Matrix: %s'%(title.upper()),'CLASSIFIER')
    print('\nClassification Report (weigthed results):')
    print(classification_report(y, y_pred, digits = 4)) 
    print(' \n\t\t\t\t\tCounts \t Percentage')
    print('\tPopulation: \t\t\t%.0f'%(Population))
    print('\tPositive Population (P): \t%.0f \t%.2f %% (Prevalence)'%(PP,PP/Population * 100))     
    print('\tNegative Population (N): \t%.0f \t%.2f %%'%(NP, NP/Population * 100))
    print('\n')
    print('\tPositive Population (P_test): \t%.0f \t%.2f %%'%(PP_t,PP_t/Population * 100))     
    print('\tNegative Population (N_test): \t%.0f \t%.2f %%'%(NP_t, NP_t/Population * 100))
    print('\n')
    print('\tTrue Positive (TP):  \t\t%.0f \t %.2f %% (Sensitivity / Recall / Hit Rate/ True Positive Rate (TPR))'%(TP,TPR * 100))
    print('\tTrue Negative (TN):  \t\t%.0f \t %.2f %% (Specificity / Selectivity / True Negative Rate (TNR))'%(TN, TNR * 100))
    print('\tFalse Positive (FP): \t\t%.0f \t %.2f %% (Fall-Out / False Positive Rate (FPR))'%(FP,FPR * 100))
    print('\tFalse Negative (FN): \t\t%.0f \t %.2f %% (Miss Rate / False Negative Rate (FNR))'%(FN,FNR * 100))
    print('\n')
    print('\tPositive Predictive Value (PPV): \t%.2f %% (Precision)'%(PPV * 100))
    print('\tNegative Predictive Value (NPV): \t%.2f %%'%(NPV * 100))
    print('\n')
    print('\tFalse Discovery Rate(FDR): \t\t%.2f %%'%(FDR * 100))
    print('\tFalse Omission Rate (FOR): \t\t%.2f %%'%(FOR * 100))
    print('\n')
    print('\tAccuracy (Acc): \t\t\t%.2f %%'%(Acc * 100))
    print('\tF1-Score (F1): \t\t\t\t%.2f %%'%(F1_scr * 100))
    print('\tMathews Correlation Coefficient (MCC): \t%.2f %%'%(MCC * 100))
    print('\n')
    print('\tBookmaker Informedness (BM): \t\t%.2f %%'%(BM * 100))
    print('\tMarkedness (Acc): \t\t\t%.2f %%'%(MK * 100))
    print('\n')
    print('\tNegative Likelihood Ratio(LR_minus): \t%.2f '%(LR_minus))
    print('\tPositive Likelihood Ratio (LR_plus): \t%.2f '%(LR_plus))
    print('\tDiagnostic Odds Ratio (DOR): \t\t%.2f'%(DOR))
    
    Summary = pd.DataFrame({
                    'Model': title,
                    'Accuracy': Accuracy,
                    'CV Score': CV_score,
                    'Precision': precision, 
                    'Recall': recall, 
                    'F1-Score': f1score,
                    'Train Score': LearnResults[0],
                    'Train Stdv': LearnResults[1],   
                    'Val Score': LearnResults[2],
                    'Val std': LearnResults[3],
                    'ROC AUC':  AUC * 100, 
                    'MCC': MCC,
                    'Cohens Kappa':Cohen_kappa},index = [0])

    return (Summary)
#Logistic Regresion
title = 'Logistic Regression'
title_abrv = 'LR'

model = LogisticRegression()

LR_GS_Params = {'penalty': ['l1', 'l2'],
                 'C': np.logspace(0, 10, 10)}

# the grid search was run and the resutls are shown below... for now it is commented so not repet the s 
#(model_best_LR, model_best_score, model_best_params) = GridSearcher(model, LR_GS_Params, X_train, y_train)

#----------------------------------------------------------------------------------------------------
#                                   These are the results of the GridSearch
#----------------------------------------------------------------------------------------------------
#Best Estimatot: LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
#          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#          verbose=0, warm_start=False) 
#----------------------------------------------------------------------------------------------------

model_best_LR = LogisticRegression(C = 1.0, class_weight = None, dual = False, fit_intercept = True,
          intercept_scaling = 1, max_iter = 500, multi_class='ovr', n_jobs = -1,
          penalty = 'l2', random_state = None, solver ='liblinear', tol = 0.0001,
          verbose=0, warm_start = False) 

Summary_LR, y_train_LR, y_test_LR = Classification_Analysis(model_best_LR, title,title_abrv, 
                                                       X_train, X_test, y_train);
#Support Vector Maachine
title = 'Support Vector Machine'
title_abrv = 'SVM'
model = SVC(probability = True)

SVM_GS_Params = {'kernel': ['rbf'], 
                  'gamma': [0.0008, 0.005],
                  'C': [1, 50, 100, 110,125],
                  'decision_function_shape':('ovo','ovr'),
                 'shrinking':(True, False)}


# the grid search was run and the resutls are shown below... for now it is commented so not repet the s 
#(model_best_SVM, model_best_score, model_best_params) = GridSearcher(model, SVM_GS_Params, X_train, y_train)
#----------------------------------------------------------------------------------------------------
#                                   These are the results of the GridSearch
#----------------------------------------------------------------------------------------------------

#Best Estimatot: SVC(C=125, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovo', degree=3, gamma=0.0008, kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False) 
#Best Score: 0.8215488215488216 
#Best parameters: {'C': 125, 'decision_function_shape': 'ovo', 'gamma': 0.0008, 'kernel': 'rbf', 'shrinking': True}

#----------------------------------------------------------------------------------------------------
model_best_SVM = SVC(C = 125, cache_size = 200, class_weight = None, coef0 = 0.0,
  decision_function_shape='ovo', degree = 3, gamma = 0.0008, kernel = 'rbf',
  max_iter = -1, probability = True, random_state = None, shrinking = True,
  tol = 0.001, verbose = False) 

Summary_SVM, y_train_SVM, y_test_SVM = Classification_Analysis(model_best_SVM, title,title_abrv, 
                                                               X_train, X_test, y_train);
# Random Forest
title = 'Random Forest'
title_abrv = 'RF'
model = RandomForestClassifier()

RF_GS_Params = {"max_depth": [None],
              "max_features": [4,5,6,7],
              "min_samples_split": [3,4,5],
              "min_samples_leaf": [3, 4,5],
              "n_estimators" :[250, 300, 300]}

# the grid search was run and the resutls are shown below... for now it is commented so not repet the s 
#(model_best_RF, model_best_score, model_best_params) = GridSearcher(model, RF_GS_Params, X_train, y_train)

#----------------------------------------------------------------------------------------------------
#                                   These are the results of the GridSearch
#----------------------------------------------------------------------------------------------------
#Best Estimatot: RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features=5, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=4, min_samples_split=4,
#            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False) 
#Best Score: 0.8462401795735129 
#Best parameters: {'max_depth': None, 'max_features': 5, 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 300} 
#----------------------------------------------------------------------------------------------------
model_best_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=5, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=4,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

Summary_RF, y_train_RF, y_test_RF = Classification_Analysis(model_best_RF, title,title_abrv, 
                                                            X_train, X_test, y_train);
#Extra Tree

title = 'Extra Tree'
title_abrv = 'ET'
model = ExtraTreesClassifier()

## Search grid for optimal parameters
ET_param_grid = {"max_depth": [None],
              "max_features": [7,8,9,10],
              "min_samples_split": [13,14],
              "min_samples_leaf": [1],
              "bootstrap": [False],
              "n_estimators" :[ 600, 700, 800],
              "criterion": ["gini"]}

#(model_best_ET, model_best_score, model_best_params) = GridSearcher(model, ET_param_grid, X_train, y_train)

#----------------------------------------------------------------------------------------------------
#                                   These are the results of the GridSearch
#----------------------------------------------------------------------------------------------------
#Best Estimatot: ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
#           max_depth=None, max_features=9, max_leaf_nodes=None,
#          min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=1, min_samples_split=13,
#           min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=1,
#           oob_score=False, random_state=None, verbose=0, warm_start=False) 
#Best Score: 0.8462401795735129 
#Best parameters: {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 9, 'min_samples_leaf': 1, 'min_samples_split': 13, 'n_estimators': 600}
#---------------------------------------------------------------------

model_best_ET = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features=9, max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=13,
           min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False) 

Summary_ET, y_train_ET, y_test_ET = Classification_Analysis(model_best_ET, title, title_abrv, X_train, X_test, y_train);

# Gradient Boosting

model = GradientBoostingClassifier()

title = 'Gradient Boosting'
title_abrv = 'GB'

#GB_param_grid = {'loss' : ["deviance"],
#              'n_estimators' : [100,200,300,500],
#              'learning_rate': [0.1, 0.05, 0.01,],
#              'max_depth': [4,6, 8],
#              'min_samples_leaf': [50,100,150],
#              'max_features': [0.1, 0.3, 0.5] 
#              }

GB_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [400,450],
              'learning_rate': [0.1],
              'max_depth': [8],
              'min_samples_leaf': [50],
              'max_features': [0.01, 0.02, 0.05] 
              }

#(model_best_GB, model_best_score, model_best_params) = GridSearcher(model, GB_param_grid, X_train, y_train)

#----------------------------------------------------------------------------------------------------
#                                   These are the results of the GridSearch
#----------------------------------------------------------------------------------------------------
#Best Estimatot: GradientBoostingClassifier(criterion='friedman_mse', init=None,
#              learning_rate=0.1, loss='deviance', max_depth=8,
#              max_features=0.02, max_leaf_nodes=None,
#              min_impurity_decrease=0.0, min_impurity_split=None,
#              min_samples_leaf=50, min_samples_split=2,
#              min_weight_fraction_leaf=0.0, n_estimators=400,
#              presort='auto', random_state=None, subsample=1.0, verbose=0,
#              warm_start=False) 
#Best Score: 0.8406285072951739 
#Best parameters: {'learning_rate': 0.1, 'loss': 'deviance', 'max_depth': 8, 'max_features': 0.02, 'min_samples_leaf': 50, 'n_estimators': 400}

#----------------------------------------------------------------------------------------------------

model_best_GB = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=8,
              max_features=0.02, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=50, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=400,
              presort='auto', random_state=None, subsample=1.0, verbose=0,
              warm_start=False)

Summary_GB, y_train_GB, y_test_GB = Classification_Analysis(model_best_GB, title,title_abrv, X_train, X_test, y_train);
# KNN
title = 'K-Nearest Neighbour'
title_abrv = 'KNN'
model = KNeighborsClassifier()

KNN_param_grid = {'algorithm': ['auto'], 'n_neighbors': [1, 2, 3],
                 'leaf_size':[1,2,3,4,5,7],
                 'weights': ['uniform', 'distance']}

#(model_best_KNN, model_best_score, model_best_params) = GridSearcher(model, KNN_param_grid, X_train, y_train)
#----------------------------------------------------------------------------------------------------
#                                   These are the results of the GridSearch
#----------------------------------------------------------------------------------------------------
#Best Estimatot: KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
#           metric_params=None, n_jobs=1, n_neighbors=2, p=2,
#           weights='uniform') 
#Best Score: 0.7822671156004489 
#Best parameters: {'algorithm': 'auto', 'leaf_size': 1, 'n_neighbors': 2, 'weights': 'uniform'}
#----------------------------------------------------------------------------------------------------

model_best_KNN = KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=2, p=2,
           weights='uniform') 

Summary_KNN, y_train_KNN, y_test_KNN = Classification_Analysis(model_best_KNN, title,title_abrv, X_train, X_test, y_train);
# Gaussian Naive Bayes
title = 'Gaussian Naive Bayes'
title_abrv = 'GNB'
model_best_GNB = GaussianNB()
Summary_GNB, y_train_GNB, y_test_GNB = Classification_Analysis(model_best_GNB, title, title_abrv, 
                                                               X_train, X_test, y_train);
# Decision Tree

title = 'Decision Tree'
title_abrv = 'DT'
model = DecisionTreeClassifier()

DT_param_grid = {'max_depth': [1, 2, 3, 4, 5],
                  'max_features': [ 4,6, 10,11,12],
                 'min_samples_split': [2,4,5]
                }
        
#(model_best_DT, model_best_score, model_best_params) = GridSearcher(model, DT_param_grid, X_train, y_train)

#----------------------------------------------------------------------------------------------------
#                                   These are the results of the GridSearch
#----------------------------------------------------------------------------------------------------    
#Best Estimatot: DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4, max_features=12, 
#    max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, 
#     min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best') 

#Best Score: 0.8327721661054994 
#----------------------------------------------------------------------------------------------------   
model_best_DT =    DecisionTreeClassifier(class_weight = None, criterion = 'gini', max_depth = 4,
            max_features = 12, max_leaf_nodes = None,
            min_impurity_decrease = 0.0, min_impurity_split=None,
            min_samples_leaf = 1, min_samples_split = 2,
            min_weight_fraction_leaf = 0.0, presort = False, random_state = None,
            splitter = 'best')
    
Summary_DT, y_train_DT, y_test_DT = Classification_Analysis(model_best_DT, title, title_abrv, X_train, X_test, y_train);   
#AdaBoost with Decision Tree Classifier

title = 'AdaBoost - Decision Tree'
title_abrv = 'ABDT'
model = AdaBoostClassifier(model_best_DT, random_state=7)

ABDT_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}


#(model_best_ABDT, model_best_score, model_best_params) = GridSearcher(model, ABDT_param_grid, X_train, y_train)
 
#----------------------------------------------------------------------------------------------------
#                                   These are the results of the GridSearch
#----------------------------------------------------------------------------------------------------    
#Best Estimatot: AdaBoostClassifier(algorithm='SAMME.R',
#   base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4, max_features=12, max_leaf_nodes=None,
#    min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, 
#    presort=False, random_state=None, splitter='random'), learning_rate=0.001, n_estimators=2, random_state=7) 

#Best Score: 0.8226711560044894 
#---------------------------------------------------------------------------------------------------- 
    
model_best_ABDT =    AdaBoostClassifier(algorithm = 'SAMME.R',
    base_estimator = DecisionTreeClassifier(class_weight = None, criterion = 'gini', max_depth = 4,
    max_features = 12, max_leaf_nodes = None, min_impurity_decrease = 0.0, min_impurity_split = None,
    min_samples_leaf = 1, min_samples_split = 2, min_weight_fraction_leaf = 0.0, presort = False, 
    random_state = None, splitter = 'random'), learning_rate = 0.001, n_estimators = 2, random_state = 7)
  
Summary_ABDT, y_train_ABDT, y_test_ABDT = Classification_Analysis(model_best_ABDT, title, title_abrv, X_train, X_test, y_train);    
#Which is the best Model ?

Class_Results = pd.concat([Summary_LR, Summary_SVM, Summary_RF, Summary_GB, Summary_KNN, Summary_ET,
                          Summary_DT, Summary_ABDT, Summary_GNB], ignore_index = True)
    
    
Class_Results = Class_Results.sort_values(by = 'CV Score', ascending=False)
#Class_Results = Class_Results.set_index('CV Score')
Class_Results.head(12)

g = sns.barplot(Class_Results["CV Score"],Class_Results["Model"],data = Class_Results, 
                palette = "muted",orient = "h",**{'xerr': Class_Results['Val std']})
g.set_xlabel("Cross Validation Score")
g = g.set_title("Cross validation scores")
# Concatenate all classifier results
y_test_Results = pd.concat([y_test_LR, y_test_SVM, y_test_RF, y_test_ET, y_test_GB, y_test_KNN, y_test_GNB,
                              y_test_DT, y_test_ABDT], axis = 1)

y_train_Results = pd.concat([y_train_LR, y_test_SVM, y_train_RF, y_train_ET, y_test_GB, y_train_KNN, y_train_GNB,
                               y_train_DT, y_train_ABDT], axis = 1)


plt.figure(figsize = (14, 7))
plt.subplot(1,2,1)
PlotCorr(y_train_Results)
plt.title('Training data')
plt.subplot(1,2,2)
PlotCorr(y_test_Results)
plt.title('Test data')
nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("AdaBoosting", model_best_ABDT),
                     ("Extra Trees", model_best_ET),
                     ("RandomForest",model_best_RF),
                     ("GradientBoosting",model_best_GB)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , palette = 'muted', orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance")
        g.set_ylabel("Features")
        g.set_title(name)
        
        nclassifier += 1


Voting = VotingClassifier(estimators = [('RF', model_best_RF),
                                      ('SVM', model_best_SVM),
                                      ('ET', model_best_ET),
                                      ('GB',model_best_GB),
                                      #('LR',model_best_LR),
                                      ('KNN',model_best_KNN),
                                      ('GNB',model_best_GNB),
                                      ('DT',model_best_DT),
                                      ('ABDT',model_best_ABDT)], voting='soft', n_jobs = 2)

Voting = Voting.fit(X_train, y_train)

y_test_V = pd.Series(Voting.predict(X_test), name = "V")

#Voting = 78.468% = highest  in Kaggle
submission = pd.DataFrame({
        "PassengerId": X_test_original["PassengerId"],
        "Survived": y_test_V
    })
submission.to_csv('Titanic Submission Voting2.csv', index = False)

print('Done')


