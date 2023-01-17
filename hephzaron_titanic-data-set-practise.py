import numpy as np
import pandas as pd
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import statsmodels.graphics.mosaicplot as mosaicplt
import seaborn as sns
from string import Template
from matplotlib import rcParams
from fancyimpute import MICE, KNN
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import norm
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
import re
import string
import itertools
color = sns.color_palette()
%matplotlib inline
dataset_url = Template('../input/$filename')
train_df = pd.read_csv(dataset_url.substitute(filename='train.csv'))
test_df = pd.read_csv(dataset_url.substitute(filename='test.csv'))
train_df.shape[0]
test_df.shape[0]
def distComparison(df1, df2):
    a = len(df1.columns)
    if a%2 != 0:
        a += 1
    
    n = np.floor(np.sqrt(a)).astype(np.int64)
    
    while a%n != 0:
        n -= 1
    
    m = (a/n).astype(np.int64)
    coords = list(itertools.product(list(range(m)), list(range(n))))
    
    numerics = df1.select_dtypes(include=[np.number]).columns
    cats = df1.select_dtypes(include=['category']).columns
    
    fig = plt.figure(figsize=(15, 15))
    axes = gs.GridSpec(m, n)
    axes.update(wspace=0.25, hspace=0.25)
    
    for i in range(len(numerics)):
        x, y = coords[i]
        ax = plt.subplot(axes[x, y])
        col = numerics[i]
        sns.kdeplot(df1[col].dropna(), ax=ax, label='df1').set(xlabel=col)
        sns.kdeplot(df2[col].dropna(), ax=ax, label='df2')
        
    for i in range(0, len(cats)):
        x, y = coords[len(numerics)+i]
        ax = plt.subplot(axes[x, y])
        col = cats[i]

        df1_temp = df1[col].value_counts()
        df2_temp = df2[col].value_counts()
        df1_temp = pd.DataFrame({col: df1_temp.index, 'value': df1_temp/len(df1), 'Set': np.repeat('df1', len(df1_temp))})
        df2_temp = pd.DataFrame({col: df2_temp.index, 'value': df2_temp/len(df2), 'Set': np.repeat('df2', len(df2_temp))})

        sns.barplot(x=col, y='value', hue='Set', data=pd.concat([df1_temp, df2_temp]), ax=ax).set(ylabel='Percentage')
        
distComparison(train_df.drop(['Survived'], 1), test_df)
train_df.head()
test_df.head()
dtype_df = train_df.dtypes.reset_index()
print(dtype_df)
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
# Get title from passenger names
def get_title (name):
    try:
        found = re.search('\S\w+\.',name).group(0).strip('.')
    except AttributeError:
        found = 'Unknown'
    return found

train_df['Title']=train_df['Name'].apply(lambda name: get_title(name))
grouped_df = train_df.groupby(['Sex','Title'])['Title'].aggregate('count').unstack(fill_value=0)
pd.options.display.float_format = '{:,.2g}'.format
grouped_df
# Get the list of column headers
title_list = list(grouped_df.columns.values)
# Search for substrings
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print (big_string)
    return np.nan
#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
train_df['Title']=train_df.apply(replace_titles, axis=1)
# Get the count of newly assigned titles.
train_df.groupby('Title')['Survived'].aggregate('count').reset_index()
# Get title from passenger names
def get_deck (cabin):
    try:
        found = re.search('^[a-zA-Z]',cabin).group(0)
    except TypeError:
        found = 'Unknown'
    return found

train_df['Deck']=train_df['Cabin'].apply(lambda cabin: get_deck(cabin))
decked_df = train_df.groupby(['Deck'])['Survived'].aggregate('count').reset_index()
pd.options.display.float_format = '{:,.2g}'.format
decked_df
cabin_list = list(decked_df['Deck'].values)
cabin_list
#Creating new family_size column with the passenger inclusive
train_df['Family_Size']=train_df['SibSp']+train_df['Parch'] + 1
# Preview overall attributes so far
train_df.head()
family_df = train_df.groupby(['Family_Size','Survived'])['Survived'].aggregate('count').unstack(fill_value=0)
pd.options.display.float_format = '{:,.2g}'.format
family_df
ax = sns.catplot(x="Family_Size", hue="Survived", kind="count",data=train_df, aspect = 1.5)
# Function to discretize family size
def conv_discrete (size):
    if (size == 1):
        return 'singleton'
    elif (1 < size <= 4) :
        return 'small'
    elif (size > 4):
        return 'large'
    else:
        return 'unspecified'
    
# Discretize family size
train_df['Family_Size_D'] = train_df['Family_Size'].apply(lambda size: conv_discrete(size))

# train_df.head() - uncomment to view 
##train_df.loc[:,['Survived','Family_Size_D']]
# Visualize multivariate categorical data in a rigorous and informative way.
mosaicplt.mosaic(train_df,index=['Survived','Family_Size_D'], gap=0.02,title='Family size by survival', statistic = True) 
#Create a new function:
def count_missing(x):
  return sum(x.isnull())

#Applying per column:
print ("Missing values per column:")
print (train_df.apply(count_missing, axis=0)) #axis=0 defines that function is to be applied on each column

#Applying per row:
print ("\nMissing values per row:")
print (train_df.apply(count_missing, axis=1).head()) #axis=1 defines that function is to be applied on each row
missing_embarked = train_df['Embarked'].isnull()
train_df[missing_embarked]
notnull_embarked = train_df['Embarked'].notnull() ## Exclude rows with null Embarked feature
fig, ax = plt.subplots()
sns.pointplot(x="x",y="y",kind="point" , color="r",
              linestyles="--",
              markers="",
              data=pd.DataFrame({'x':[0,'S','C','Q'],'y':[80, 80,80,80]}),
              ax=ax)
sns.catplot(x="Embarked", y="Fare", hue="Pclass", kind="box", data=train_df[notnull_embarked] , ax=ax)
def repl_null_embarked(x):
    if (x == 62 or 830):
        return 'C'
    else:
        return x
    
train_df['Embarked'] = train_df['PassengerId'].apply(lambda x: repl_null_embarked(x))
## Uncomment to confirm Missing values have been added
##missing_embarked_added = train_df['PassengerId'] == 830
##train_df[missing_embarked_added]
#Applying per column: Missing Embarked value should be fixed
print ("Missing values per column:")
train_df.apply(count_missing, axis=0) #axis=0 defines that function is to be applied on each column
dtype_df_n = train_df.dtypes.reset_index()
print(dtype_df_n)
def prepForModel(df):
    new_df = df.copy()
    new_df['Sex'] = new_df['Sex'].astype('category')
    new_df['Embarked'] = new_df['Embarked'].astype('category')
    new_df.Pclass = new_df.Pclass.astype("int")
    new_df.SibSp = new_df.SibSp.astype("int")
    new_df.Parch = new_df.Parch.astype("int")
    new_df.Fare = new_df.Fare.astype("float")
    cat_columns = new_df.select_dtypes(['category']).columns
    new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
    return new_df

train_cl = prepForModel(train_df)

train_cl.dtypes.reset_index()
rf = RandomForestClassifier(n_estimators=1000,max_depth=None,min_samples_split=10)
train_cl = prepForModel(train_df)

Xcol = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare','Embarked', 'Family_Size','Age']
Ycol = 'Survived'

X = train_cl.loc[:, Xcol]
Y = train_cl.loc[:, Ycol]

Xmice = MICE(n_imputations=200, impute_type='col', verbose=False).complete(X)
Ymice = Y

##train_df['Age'] = Xmice['Age']
mice_err = cross_val_score(rf, Xmice, Y, cv=10, n_jobs=-1).mean()
print("[MICE] Estimated RF Test Error (n = {}, 10-fold CV): {}".format(len(Xmice), mice_err))


X_df = pd.DataFrame(Xmice)
X_df.columns = Xcol
#Applying per column:
print ("Missing values per column after age insertion:")
X_df.apply(count_missing, axis=0)
plt.figure(figsize=(10,4))
sns.set_color_codes()
sns.distplot(train_df['Age'].dropna(), color='R', label="before imputations", hist=False, rug=True)
sns.distplot(X_df['Age'], color='Y',label="after imputations", hist=False) # Kernel Density Estimation
plt.legend()
plt.xlabel('Age distribution ', fontsize=12)
plt.show()
plt.figure(figsize=(10,4))
sns.set_color_codes()
sns.distplot(train_df['Age'].dropna(), color='G',kde=False, label="before imputations", hist=False, fit=norm)
sns.distplot(X_df['Age'], color='Y',kde=False,label="after imputations", hist=False, fit=norm, rug=True) # Kernel Density Estimation
plt.legend()
plt.xlabel('Fitted curve for age distribution', fontsize=12)
plt.show()
train_df['Age'] = X_df['Age']

print ("Missing values per column after age insertion:")
train_df.apply(count_missing, axis=0)
Xcol = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare','Embarked','Title','Deck','Family_Size_D','Age']
Ycol = 'Survived'

Xtrain = train_df.loc[:, Xcol]
Ytrain = train_df.loc[:, Ycol]
##RANDOM FOREST

le = preprocessing.LabelEncoder()
def encode_str(df):
    enc_df = df.copy()
    enc_df['Sex'] = enc_df['Sex'].astype('object')
    enc_df['Embarked'] = enc_df['Embarked'].astype('object')
    for column_name in enc_df.columns:
        if enc_df[column_name].dtype == object:
            enc_df[column_name] = le.fit_transform(enc_df[column_name])
        else:
            pass
    return enc_df

## Split into train and test data set
X_tr = encode_str(Xtrain)
X_train, X_test, y_train, y_test = train_test_split(X_tr, Ytrain, test_size=0.5)
## Create Random forest classification model
rf.fit(X_train, y_train)

ypred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, ypred_rf)
print(rf.feature_importances_)
### DECISION TREES
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

ypred_clf = clf.predict_proba(X_test)[:, 1]
fpr_clf, tpr_clf, _ = roc_curve(y_test, ypred_clf)
print(clf.feature_importances_)
### LOGISTIC REGRESSION
lr = LogisticRegression(penalty="l2", n_jobs=-1)
lr.fit(X_train, y_train)
ypred_lr = clf.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, ypred_lr)
print(lr.coef_)
### NAIVE BAYES - GUASSIAN NB
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
ypred_nb = nb.predict_proba(X_test)[:, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test, ypred_nb)
### KNN - K Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(X_train, y_train)
ypred_knc = knc.predict_proba(X_test)[:, 1]
fpr_knc, tpr_knc, _ = roc_curve(y_test, ypred_knc)
### SVM - Support Vector Machines
from sklearn.svm import SVC
svc = SVC(probability=True)
svc.fit(X_train, y_train)
ypred_svc = svc.predict_proba(X_test)[:, 1]
fpr_svc, tpr_svc, _ = roc_curve(y_test, ypred_svc)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot(fpr_clf, tpr_clf, label='Decision Tree')
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.plot(fpr_nb, tpr_nb, label='GaussianNB')
plt.plot(fpr_knc, tpr_knc, label='K-Nearest Neighbour')
plt.plot(fpr_svc, tpr_svc, label='Support Vector Classifier')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
test_df.apply(count_missing, axis=0)
## Get titles from dataset
test_df['Title']=test_df['Name'].apply(lambda name: get_title(name))
test_grouped_df = test_df.groupby(['Sex','Title'])['Title'].aggregate('count').unstack(fill_value=0)
pd.options.display.float_format = '{:,.2g}'.format
test_grouped_df
# Replace Titles with custom title
test_df['Title']=test_df.apply(replace_titles, axis=1)
## Create feature deck form cabin column
test_df['Deck']=test_df['Cabin'].apply(lambda cabin: get_deck(cabin))
#Creating new family_size column with the passenger inclusive
test_df['Family_Size']=test_df['SibSp']+train_df['Parch'] + 1
# Discretize family size
test_df['Family_Size_D'] = test_df['Family_Size'].apply(lambda size: conv_discrete(size))
test_cl = prepForModel(test_df)
X_pred = encode_str(test_cl.loc[:, Xcol])

Xmice_pred = MICE(n_imputations=200, impute_type='col', verbose=False).complete(X_pred)

Xtest = pd.DataFrame(Xmice_pred)
Xtest.columns = Xcol
#Applying per column:
print ("Missing values per column after age insertion:")
Xtest.apply(count_missing, axis=0)
Y_pred = rf.predict(Xtest)
Y_out = pd.DataFrame(Y_pred)
Y_out.columns =['Survived']
Y_out['PassengerId'] = test_df['PassengerId']

Y_out.head()
