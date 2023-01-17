import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#The last few columns of the CSV from Kaggle are blank, so we'll ignore them

kick = pd.read_csv("ksprojects.csv", usecols=range(13))
kick.dtypes
kick.head()
kick.rename(columns=lambda x: x.strip(), inplace=True)
#Convert the deadline and launched dates to datetime objects

kick['deadline'] = pd.to_datetime(kick['deadline'], errors='coerce')

kick['launched'] = pd.to_datetime(kick['launched'], errors='coerce')
#Convert the goal, pledged, and backers columns to numeric

kick['goal'] = pd.to_numeric(kick['goal'], errors='coerce')

kick['pledged'] = pd.to_numeric(kick['pledged'], errors='coerce')

kick['usd pledged'] = pd.to_numeric(kick['usd pledged'], errors='coerce')

kick['backers'] = pd.to_numeric(kick['backers'], errors='coerce')
#Check that everything worked smoothly

kick.dtypes
#Now, drop all of the rows that have NaN in them

print("Pre-drop: " + str(len(kick)))

kick.dropna(inplace=True)

print("Post-drop: " + str(len(kick)))
#Distribution of project status

kick['state'].value_counts().plot(kind='bar', color='#2ADC75')

plt.title('Kickstarter Project Status')

plt.ylabel('Projects')
#Distribution of main categories

kick['main_category'].value_counts().plot(kind='bar', color='#2ADC75')

plt.title('Kickstarter Categories')

plt.ylabel('Projects')
#Distribution of countries

kick['country'].value_counts().plot(kind='bar', color='#2ADC75')

plt.title('Kickstarter Projects by Country')

plt.ylabel('Projects')
#Distribution of goals

pd.set_option('display.float_format', lambda x: '%.2f' % x)

kick['goal'].describe()
kick['pledged'].describe()
stateRatio = kick.groupby('state').agg({'pledged': np.mean, 'goal': np.mean})

stateRatio['ratio'] = stateRatio['pledged'] / stateRatio['goal']

stateRatio
catRatio = kick.groupby('main_category').agg({'pledged': np.mean, 'goal': np.mean})

catRatio['ratio'] = catRatio['pledged'] / catRatio['goal']

catRatio['ratio'].sort_values(ascending=False).plot(kind='bar', color='#2ADC75')

plt.title('Pledged to Goal Ratio on Kickstarter')

plt.xlabel('')

plt.ylabel('Pledged / Goal Ratio')
catPivot = kick.pivot_table(index='main_category', columns='state', values='ID', aggfunc='count')

catPivot['WLratio'] = catPivot['successful'] / catPivot['failed']

catPivot['WLratio'].sort_values(ascending=False).plot(kind='bar', color='#2ADC75')

plt.title('Success to Failure Ratio on Kickstarter')

plt.xlabel('')

plt.ylabel('Ratio')
#First, turn our pivot table columns into percentages instead of absolute numbers

catPivot = kick.pivot_table(index='main_category', columns='state', values='ID', aggfunc='count')

catPivot['total'] = catPivot.sum(axis=1)
#Change all columns to percentages of total

for column in catPivot.columns[:5]:

    catPivot[column] = catPivot[column] / catPivot['total']
#Plot

catPivot.iloc[:,:5].plot(kind='bar', stacked=True, figsize=(9,6), 

                         color=['#034752', '#88C543', 'black','#2ADC75', 'white'])

plt.title('Project Outcome by Category on Kickstarter')

plt.legend(loc=2, prop={'size': 9})

plt.xlabel('')

plt.ylabel('Percentage of Projects')
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()



kickML = kick.copy()

for column in ['category', 'main_category','country']:

    kickML[column] = enc.fit_transform(kickML[column])
kickML = kickML[kickML['state'].apply(lambda x: x in ['successful', 'failed'])]
from sklearn.utils import shuffle

kickML = shuffle(kickML)
def dataSplit(features, target):

    trainx = kickML.iloc[:223526][features]

    trainy = kickML.iloc[:223526][target]

    testx = kickML.iloc[223527:271425][features]

    testy = kickML.iloc[223527:271425][target]

    cvx = kickML.iloc[271426:len(kickML)][features]

    cvy = kickML.iloc[271426:len(kickML)][target]

    

    return trainx, trainy, testx, testy, cvx, cvy
trainx, trainy, testx, testy, cvx, cvy = dataSplit(['category', 'main_category', 'goal', 'country'], 'state')
from sklearn import linear_model

model = linear_model.LogisticRegression()
model.fit(trainx, trainy)
from sklearn.metrics import confusion_matrix
def printCM(y,y_pred):

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    print("True positives: " + str(tp))

    print("False positives: " + str(fp))

    print("True negatives: " + str(tn))

    print("False negatives: " + str(fn))

    print('\n')

    print("Overall accuracy: " + str((tp+tn)/float((tp+tn+fp+fn))))

    print("Precision (tp/tp+fp): " + str(tp/float((tp+fp))))

    print("Recall (tp/tp+fn): " + str(tp/float((tp+fn))))
printCM(trainy, model.predict(trainx))
kickML['state'].value_counts()['failed'] / float(len(kickML))
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(trainx, trainy)
printCM(trainy, clf.predict(trainx))
printCM(cvy, clf.predict(cvx))
from sklearn.ensemble import RandomForestClassifier

clfF = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)
clfF.fit(trainx, trainy)
printCM(cvy, clfF.predict(cvx))
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
printCM(trainy, gnb.fit(trainx, trainy).predict(trainx))
printCM(testy, gnb.fit(testx, testy).predict(testx))
kickML['completion'] = kickML['usd pledged'] / kickML['goal']
trainx, trainy, testx, testy, cvx, cvy = dataSplit(['category', 'main_category', 'goal', 'country'], 'completion')
from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import f_regression

import sklearn.metrics

lr = LinearRegression()
model = lr.fit(trainx, trainy)
def regScore(trainy, predy):

    print("R^2: " + str(m.r2_score(trainy, predy)))

    print("MSE: " + str(m.mean_squared_error(trainy, predy)))

    print("MAE: " + str(m.mean_absolute_error(trainy, predy)))
regScore(trainy, model.predict(trainx))
pd.DataFrame({'coefficient': model.coef_, 

              'F-score': f_regression(trainx, trainy)[0], 

              'p-value': f_regression(trainx, trainy)[1]},

            index = trainx.columns)
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_depth=5)
dtr.fit(trainx, trainy)
regScore(trainy, dtr.predict(trainx))
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(trainx, trainy)
regScore(trainy, rfr.predict(trainx))
def graphRFRR2(trainx, trainy):

    

    #For adjusting max_depth

    rsmax_depth = []

    for depth in np.arange(2,112, 10):

        rfr = RandomForestRegressor(max_depth=depth)

        rsmax_depth.append(m.r2_score(trainy, rfr.fit(trainx, trainy).predict(trainx)))

        

    #For adjusting max_lead_nodes

    rsmax_leaf_nodes = []

    for nodes in np.arange(2,112, 10):

        rfr = RandomForestRegressor(max_leaf_nodes=nodes)

        rsmax_leaf_nodes.append(m.r2_score(trainy, rfr.fit(trainx, trainy).predict(trainx)))

   

    return pd.DataFrame({'max_depth': rsmax_depth, 'max_leaf_nodes': rsmax_leaf_nodes}, index=np.arange(2,112,10))
plotR = graphRFRR2(trainx, trainy)
plotR.plot(color=['#2ADC75', 'black'])

plt.title('R^2 For Random Forest Regressor')

plt.xlabel('parameter')

plt.ylabel('R^2')

plt.ylim(0,1)