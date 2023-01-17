import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import math



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

# reading test data



test = pd.read_csv('../input/test.csv')



# extracting and then removing the targets from the training data 

targets = train.Survived

train.drop('Survived', 1, inplace=True)

    

# merging train data and test data for future feature engineering

combined = train.append(test)

combined.reset_index(inplace=True)

combined.drop('index', inplace=True, axis=1)
# lets check it out

combined.head()
mapping_sex = {'female':0,'male':1}

mapping_embarked = {'Q':0,'S':1,'C':2}



combined.replace({'Sex':mapping_sex},inplace=True)

combined.replace({'Embarked':mapping_embarked},inplace=True)



combined["Family_Size"] = combined["Parch"]+ combined["SibSp"] + 1

#lets check out how it looks now

combined.head()
#checking number of NaN values 

missing_df = combined.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

fig, ax = plt.subplots(figsize=(8,10))

rects = ax.barh(ind, missing_df.missing_count.values, color='#4A148C')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_xlabel("Column Name")
age = np.array(combined['Age'])

age = [value for value in age if not math.isnan(value)]



sns.set(rc={"figure.figsize": (10, 6)})

sns.distplot(age,kde = False)
zeroage = []

oneage = []

agee = combined['Age'].values

survive = targets.values



for i in range(0,len(survive)):

    if survive[i] == 1:

        oneage.append(agee[i])

    else:

        zeroage.append(agee[i])

        

zeroage = [value for value in zeroage if not math.isnan(value)]

oneage = [value for value in oneage if not math.isnan(value)]



sns.set(rc={'axes.facecolor':'#C5E1A5', 'figure.facecolor':'#C5E1A5'})

ax = sns.distplot(zeroage, color = '#33691E', kde = False)

ax.set(xlabel='Distribution of Age of people who din\'t survive')

plt.show(ax)

# many between 20-40 age died


sns.set(rc={'axes.facecolor':'#B2EBF2', 'figure.facecolor':'#B2EBF2'})

ax = sns.distplot(oneage, color = '#01579B', kde = False)

ax.set(xlabel='Distribution of Age of people who survived')

plt.show(ax)

# many young kids and babies survived :) and again many in the 20-40 range survived. The age gives good insight  
sns.set(rc={'axes.facecolor':'#F48FB1', 'figure.facecolor':'#F48FB1'})

ax = sns.distplot(combined.Sex.values, color = '#AD1457', kde = False)

ax.set(xlabel='Distribution of Sex of people who survived')

plt.show(ax)

# so females survived far more than males
sns.set(rc={'axes.facecolor':'#FFF9C4', 'figure.facecolor':'#FFF9C4'})

embark =  [value for value in combined.Embarked.values if not math.isnan(value)]

ax = sns.distplot(embark, color = '#F57F17', kde = False)

ax.set(xlabel='Distribution of Embarked port of people who survived')

plt.show(ax)

# many embarked on one particular port
combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

# a map of more aggregated titles

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

    

# we map each title

combined['Title'] = combined.Title.map(Title_Dictionary)

mapping_title = {

    'Officer':1,

    'Royalty':2,

    "Mr":3,

    "Mrs":4,

    "Miss":5,

    "Master":6

}

combined.replace({'Title':mapping_title},inplace=True)
combined.Cabin.fillna('U', inplace=True)

    

# mapping each Cabin value with the cabin letter

combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])

    

mapping_cabin = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7,'U':8}



combined.replace({'Cabin':mapping_cabin},inplace=True)

#I'm doubtful that it maybe of any use as there are many NAN values in it, lets see how it will fare in our models
#I think we have gotten all that we can from the Name column

#lets delete it

del combined["Name"]
#there are a great number of tickets that just dont have any information other than a number

# if we try feature extraction here, it might create more noise than help

del combined['Ticket']
# since there are so many missing values in Age, it seems that we will bias the data if we simply plug in the mean for each NaN

#lets predict the age using KNN 

#there are only 2 NAN in Embarked and one in Fare, we can fill it using mean, it wont matter that much
combined["Fare"]=combined['Fare'].fillna(4)

combined["Embarked"]=combined['Embarked'].fillna(1)

# 1 is the most popular port, wont hurt much
combined.info()
# lets use a placeholder for NaN in age

combined["Age"]=combined['Age'].fillna(-999)
combined.sort_values('Age')
#creating a dataframe with only the -999 values for age

newthing = combined.loc[combined['Age'] == -999]
combined = combined[combined.Age != -999]

pre = combined['Age'] 

#we cant have the Age column in both the dataframes if we want to predict it

del combined['Age']

del newthing['Age']
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error as mse

X_train, X_test, y_train, y_test = train_test_split(combined, pre, test_size=0.2, random_state=42)
#KNN

knc = KNeighborsRegressor(n_neighbors=4)

knc.fit(X_train,y_train)

np.sqrt(mse(knc.predict(X_test),y_test))

#SVM

svr = SVR(kernel = 'linear')

svr.fit(X_train,y_train)

np.sqrt(mse(svr.predict(X_test),y_test))

newage = svr.predict(newthing)
newthing.loc[:,'Age'] = pd.Series(newage, index=newthing.index)
combined.loc[:,'Age'] = pd.Series(pre, index=combined.index)
combined = combined.append(newthing)
sns.set(rc={'axes.facecolor':'#FFE0B2', 'figure.facecolor':'#FFE0B2'})

age = np.array(combined['Age'])

sns.set(rc={"figure.figsize": (10, 6)})

sns.distplot(age,kde = False, color = '#FF3D00')
#Now that we have our missing values with a bit more meaning lets see the corr 

combined = combined.sort_values('PassengerId')
train = combined.loc[0:890]

test = combined.loc[891:]

# we need the survived column to correlate it with other columns

train['Survived'] = targets
data = [

    go.Heatmap(

        z= train.corr().values,

        x= train.columns.values,

        y= train.columns.values,

        colorscale='Viridis',

        text = True ,

        opacity = 1.0

        

    )

]





layout = go.Layout(

    title='Pearson Correlation of all Columns',

    xaxis = dict(ticks='', nticks=12),

    yaxis = dict(ticks='' ),

    width = 800, height = 600,

    

)





fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='labelled-heatmap')
# we cant have survived anymore

del train['Survived']
# I'm using vecstack for stacking, pretty handy

import vecstack
#lets see how stacking fares

X_train, X_test, y_train, y_test = train_test_split(train, targets, test_size=0.2, random_state=42)

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score
#im trying this without stacking, so first we can choose which models to use

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



models = [

    RandomForestClassifier(n_estimators=100),

    MLPClassifier(),

    ExtraTreesClassifier(),

    GradientBoostingClassifier(),

    KNeighborsClassifier(),

    SVC(),

    GaussianProcessClassifier(),

    DecisionTreeClassifier(),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()

    

]



for model in models:

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    print("Model : ", model)

    print("**"*40)

    print("Score : ",score)



#we need to convert the pandas df to a numpy array

X_tra = X_train.as_matrix()

X_tes = X_test.as_matrix()

y_tra = y_train.as_matrix()

y_tes = y_test.as_matrix()



#First level

models = [

    ExtraTreesClassifier(n_estimators=10, n_jobs=1, oob_score=False, random_state= 41),

    

    GaussianNB(priors=None),

    

    GradientBoostingClassifier(criterion='friedman_mse', init=None,

              learning_rate=0.1, loss='deviance', max_depth=3,

              max_features=None, max_leaf_nodes=None,

              min_impurity_split=1e-07, min_samples_leaf=1,

              min_samples_split=2, min_weight_fraction_leaf=0.0,

              n_estimators=200, presort='auto', random_state=None,

              subsample=1.0, verbose=0, warm_start=False),

    

    RandomForestClassifier( n_jobs = -1, 

        n_estimators = 100, max_depth = 3),

        

    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,

          learning_rate=0.1, n_estimators=100, random_state=32)]

    

# Compute stacking features

S_train, S_test = vecstack.stacking(models, X_tra, y_tra, X_tes, 

    regression = False, metric = accuracy_score, n_folds = 4, 

     shuffle = True, random_state = 0, verbose = 2)



# Initialize 2-nd level model

model = AdaBoostClassifier( n_estimators = 100)

    

# Fit 2-nd level model

model = model.fit(S_train, y_train)



# Predict

y_pred = model.predict(S_test)



# Final prediction score

print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))