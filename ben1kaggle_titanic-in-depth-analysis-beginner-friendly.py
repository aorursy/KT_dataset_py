# Importing general packages and set basic plot styling

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

from IPython.display import YouTubeVideo

import warnings

warnings.filterwarnings('ignore')



sns.set_style("white")

sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

plt.rcParams['axes.color_cycle'] = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', 

                                    u'#9467bd', u'#8c564b', u'#e377c2']
YouTubeVideo('NdZ6TY1pxL8')

# https://www.youtube.com/watch?v=NdZ6TY1pxL8 
# Load and merge the train and test data

df1 = pd.read_csv('../input/train.csv')

df2 = pd.read_csv('../input/test.csv')

df1['Set'] = 'train'

df2['Set'] = 'test'

df=df1.append(df2)

df=df.reset_index()

df.info()

df.head()
# Missing values Embarked 

display(df[df['Embarked'].isnull()])



df['Embarked'] = df['Embarked'].fillna('S') 
# Missing value Fare

display(df[df['Fare'].isnull()])



a = df['Fare'].loc[(df['Pclass']==3) & (df['Embarked']=='S')]



plt.figure(figsize=[7,3])

sns.distplot(a.dropna(), color='C0')

plt.plot([a.median(), a.median()], [0, 0.16], '--', color='C1')



df['Fare'] = df['Fare'].fillna(a.median())



sns.despine(bottom=0, left=0)

plt.title('Fare for 3rd class embarked in S')

plt.xlabel('Fare')

plt.legend(['median'])

plt.show()
# Clean up and feature engineering



# Label Survived for plot

df['Survived'] = df['Survived'].replace([0, 1], ['no', 'yes']) 



# Label Sex for plot

df['Sex'] = df['Sex'].replace([0, 1], ['male', 'female']) 



# Transform Fare to today's US dollar, for fun

df['Fare'] = df['Fare']*108*1.3 #historic gbp to current gbp to current usd



# Get personal title from Name, merge rare titles

df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split(' ')[1])

toreplace = ['Jonkheer.', 'Ms.', 'Mlle.', 'Mme.', 'Capt.', 'Don.', 'Major.', 

             'Col.', 'Sir.', 'Dona.', 'Lady.', 'the']

replacewith = ['Master.', 'Miss.', 'Miss.', 'Mrs.', 'Sir.', 'Sir.', 'Sir.',

              'Sir.', 'Sir.', 'Lady.', 'Lady.', 'Lady.']

df['Title'] = df['Title'].replace(toreplace, replacewith)



# Get family names

df['FamName'] = df['Name'].apply(lambda x: x.split(',')[0])



# Get family sizes based on Parch and SibSp, classify as single/small/large

df['FamSize'] = df['Parch'] + df['SibSp'] + 1

df['FamSize2'] = pd.cut(df['FamSize'], [0, 1, 4, 11], labels=['single', 'small', 'large'])



# Get group sizes based on Ticket, classify as single/small/large

df['GrpSize'] = df['Ticket'].replace(df['Ticket'].value_counts())

df['GrpSize2'] = pd.cut(df['GrpSize'], [0, 1, 4, 11], labels=['single', 'small', 'large'])



# Get Deck from Cabin letter

def getdeck(cabin):

    if not pd.isnull(cabin) and cabin[0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:

        return cabin[0]

    else:

        return 'X'    

    

df['Deck'] = df['Cabin'].apply(getdeck)



# Get a rough front/mid/back location on the ship based on Cabin number

'''

A front

B until B49 is front, rest mid

C until C46 is front, rest mid

D until D50 is front, rest back

E until E27 is front, until E76 mid, rest back

F back

G back

Source: encyclopedia-titanica.org/titanic-deckplans/

'''

def getfmb(cabin):

    

    if not pd.isnull(cabin) and len(cabin)>1:

        if (cabin[0]=='A'

            or cabin[0]=='B' and int(cabin[1:4])<=49

            or cabin[0]=='C' and int(cabin[1:4])<=46

            or cabin[0]=='D' and int(cabin[1:4])<=50

            or cabin[0]=='E' and int(cabin[1:4])<=27):

            return 'front'

        

        elif (cabin[0]=='B' and int(cabin[1:4])>49

            or cabin[0]=='C' and int(cabin[1:4])>46

            or cabin[0]=='E' and int(cabin[1:4])>27 and int(cabin[1:4])<=76):

            return 'mid'



        elif (cabin[0]=='F'

           or cabin[0]=='G'

           or cabin[0]=='D' and int(cabin[1:4])>50):

            return 'back'

        

        else:

            return 'unknown'

    else:

        return 'unknown'        

    

df['CabinLoc'] = df['Cabin'].apply(getfmb)



dfstrings = df.copy() # save df containing string features to use for plotting later
# Factorize the string features



df['CabinLoc'] = df['CabinLoc'].replace(['unknown', 'front', 'mid', 'back'], range(4))



df['Deck'] = df['Deck'].replace(['X', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], range(8))



df['GrpSize2'] = df['GrpSize2'].astype(str) #convert from category dtype

df['GrpSize2'] = df['GrpSize2'].replace(['single', 'small', 'large'], range(3))



df['FamSize2'] = df['FamSize2'].astype(str) #convert from category dtype

df['FamSize2'] = df['FamSize2'].replace(['single', 'small', 'large'], range(3))



df['Title'] = df['Title'].replace(df['Title'].unique(), range(8))



df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], range(3))



df['Sex'] = df['Sex'].replace(['male', 'female'], range(2)) 



df['Survived'] = df['Survived'].replace(['no', 'yes'], range(2)) 



dfnum = df.copy() # save df containing factorized features to use for subsequent analysis
# Multiple linear regression modeling of Age

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics



feats = ['Sex', 'Embarked', 'Pclass', 'Fare', 'Title', 'Parch', 'SibSp', 'FamSize', 'FamSize2', 

         'GrpSize', 'GrpSize2', 'Deck', 'CabinLoc' ]



dffeats = df[feats][df['Age'].notnull()]

dfresp = df['Age'][df['Age'].notnull()]

dfmiss = df[feats][df['Age'].isnull()]



X_train, X_test, y_train, y_test = train_test_split(dffeats, dfresp, test_size=0.25, random_state=100)



lm = LinearRegression()

lm.fit(X_train, y_train )



y_predtrain = lm.predict(X_train)

print('Mean train: ' + str(np.mean(y_predtrain)))

print('Std predtrain: ' + str(np.std(y_predtrain)))

print('RMSE predtrain: ' + str(np.sqrt(metrics.mean_squared_error(y_train, y_predtrain))))



y_predtest = lm.predict(X_test)

print('Mean test: ' + str(np.mean(y_predtest)))

print('Std predtest: ' + str(np.std(y_predtest)))

print('RMSE predtest: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_predtest))))



pred1 = lm.predict(dffeats)

pred2 = lm.predict(dfmiss)

print(pred2.min(), pred2.max())

# Plots for Age regression

fig, [ax1, ax2, ax3] = plt.subplots(3,1, figsize=[7,6])



sns.distplot(y_predtrain, hist=False, label='prediction of train set (n=784)', ax=ax1)

sns.distplot(y_predtest, hist=False, label='prediction of test set (n=262)', ax=ax1)

ax1.set_xlim([0, 80])



sns.distplot(pred1, hist=False, label='prediction of known (n=1046)', ax=ax2)

sns.distplot(pred2, hist=False, label='prediction of missing (n=263)', ax=ax2)

ax2.set_xlim([0, 80])

ax2.set_ylim([0, 0.15])



sns.distplot(dfresp, hist=False, label='known (n=1046)')

sns.distplot(dfresp.values.tolist() + pred2.tolist(), hist=False, label='known + predicted missing (n=1309)')

ax3.set_xlim([0, 80])

ax3.set_ylim([0, 0.05])



fig.tight_layout()

sns.despine(bottom=0, left=0)
# Updating the dataframe (strings version) with Age related features

df = dfstrings

df['Age'].loc[df['Age'].isnull()] = pred2



# Classify age groups

df['AgeGrp'] = pd.cut(df['Age'], [0, 12, 20, 200], labels = ['child', 'teen', 'adult'])



# Classify age decade

df['AgeDec'] = pd.cut(df['Age'], range(0,90,10), labels=range(8))

df['AgeDec'] = df['AgeDec'].astype(int)



# Classify male/female/child

df['PersonType'] = df['Sex']

df.loc[df['Age']<12,'PersonType'] = 'child'

# Updating the dataframe (numbers version) with Age related features

dfnum['Age'] = df['Age']



dfnum['AgeGrp'] = df['AgeGrp'].astype(str) #convert from category dtype

dfnum['AgeGrp'] = dfnum['AgeGrp'].replace(['child', 'teen', 'adult'], range(3))



dfnum['AgeDec'] = df['AgeDec']



dfnum['PersonType'] = df['PersonType'].replace(['male', 'female', 'child'], range(3)) 

# Plot histograms of all the features split on 'Survived'



fig, axes = plt.subplots(5,4, figsize=[12,13])

axes = axes.ravel()

axnr = 0

for i in ['Survived', 'Sex', 'Age', 'Age', 'AgeGrp', 'AgeDec', 'PersonType', 'Embarked', 'Pclass', 

          'Fare', 'Fare', 'Title', 'Parch', 'SibSp', 'FamSize', 'FamSize2', 'GrpSize', 'GrpSize2', 

          'Deck', 'CabinLoc' ]: 

    sns.countplot(i, hue='Survived', data=df, ax=axes[axnr])

    axes[axnr].set(xlabel=i, ylabel="")

    axes[axnr].legend().set_visible(False)

    axnr += 1



axes[0].cla() #clear and replace plot

sns.countplot('Survived', data=df, ax=axes[0])

axes[0].set(xlabel='Survived', ylabel="")

    

axes[2].cla()

sns.distplot(df['Age'][df['Survived']=='no'], ax=axes[2], bins=range(0, 100, 2))

sns.distplot(df['Age'][df['Survived']=='yes'], ax=axes[2], bins=range(0, 100, 2))



axes[3].cla()

sns.distplot(df['Age'][df['Survived']=='no'], ax=axes[3], bins=range(0, 100, 2))

sns.distplot(df['Age'][df['Survived']=='yes'], ax=axes[3], bins=range(0, 100, 2))

axes[3].set(xlim=[0,40], ylim=[0,0.05])



axes[9].cla()

sns.distplot(df['Fare'][df['Survived']=='no'], ax=axes[9], bins=range(0, 72000, 300))

sns.distplot(df['Fare'][df['Survived']=='yes'], ax=axes[9], bins=range(0, 72000, 300))

axes[9].set(ylim=[0,0.0005])



axes[10].cla()

sns.distplot(df['Fare'][df['Survived']=='no'], ax=axes[10], bins=range(0, 72000, 300))

sns.distplot(df['Fare'][df['Survived']=='yes'], ax=axes[10], bins=range(0, 72000, 300))

axes[10].set(xlim=[0,10000], ylim=[0,0.0005])



axes[11].set_xticklabels(axes[11].get_xticklabels(), rotation = 45, size='x-small', ha="center")



axes[18].cla()

sns.countplot(df['Deck'][df['Deck'] != 'X'], hue='Survived', data=df, 

              ax=axes[18], order=['A', 'B', 'C', 'D', 'E', 'F', 'G'])

axes[18].set(xlabel='Deck', ylabel="")

axes[18].legend().set_visible(False)



axes[19].cla()

sns.countplot(df['CabinLoc'][df['CabinLoc'] != 'unknown'], hue='Survived', data=df, 

              ax=axes[19], order=['front', 'mid', 'back'])

axes[19].set(xlabel='CabinLoc', ylabel="")

axes[19].legend().set_visible(False)    

    

plt.title('Histograms')    

fig.tight_layout()

sns.despine(bottom=1, left=1)
df = dfnum

fig = plt.subplots(figsize=[15, 15])



sns.heatmap(df[['Survived', 'Sex', 'Age', 'AgeGrp', 'AgeDec', 'PersonType', 'Embarked', 'Pclass', 

                'Fare', 'Title', 'Parch', 'SibSp', 'FamSize', 'FamSize2', 'GrpSize', 'GrpSize2', 

                'Deck', 'CabinLoc']].corr(), 

            annot=True, fmt=".2f", square=1, cmap="RdBu_r", vmin=-1, vmax=1)

plt.title('Correlation heatmap')

plt.show()
fig, [ax1, ax2] = plt.subplots(1,2, figsize=[10,4])

x='AgeDec'

y='Survived'

df['Grpcount'] = df.groupby([x, y]).transform('count')['index'].values

df['Grpmean'] = df.groupby([x]).transform('mean')[y].values

sns.regplot(x, y, data=df, scatter_kws={'s': df['Grpcount']*7}, line_kws={'color':'C2'}, ax=ax1)

ax1.plot(df[x], df['Grpmean'], 'o', color='C3')

ax1.set_ylim([-0.15, 1.1])



x='Embarked'

y='Pclass'

df['Grpcount'] = df.groupby([x, y]).transform('count')['index'].values

df['Grpmean'] = df.groupby([x]).transform('mean')[y].values

sns.regplot(x, y, data=df, scatter_kws={'s': df['Grpcount']*7}, line_kws={'color':'C2'}, ax=ax2, color='C0')

ax2.plot(df[x], df['Grpmean'], 'o', color='C3')

ax2.set_ylim([0.7, 3.4])

ax2.set_xlim([-0.4, 2.3])

ax2.set_xticks(range(3))



fig.tight_layout()

sns.despine(bottom=0, left=0)

plt.show()
# Split into train/test

feats = ['Sex', 'Age', 'AgeGrp', 'AgeDec', 'PersonType', 'Embarked', 'Pclass', 'Fare', 'Title',

         'Parch', 'SibSp', 'FamSize', 'FamSize2', 'GrpSize', 'GrpSize2', 'Deck', 'CabinLoc'] 



dff = df[['PassengerId', 'Set', 'Survived'] + feats]



train = dff[dff['Set'] == 'train'].drop('Set', 1)

test = dff[dff['Set'] == 'test'].drop('Set', 1)



trainx = train[feats].values

trainy = train['Survived'].values.astype(int)



testx = test[feats].values
# Fit random forest to the data

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=500, verbose=0, random_state=1)

clf.fit(trainx, trainy)

# Visualize a tree

import graphviz 

from sklearn import tree



dotdata = tree.export_graphviz(clf.estimators_[0], out_file=None, feature_names=feats, filled=True, 

                               rounded=True, class_names=True, special_characters=False, 

                               leaves_parallel=False)  

# graphviz.Source(dotdata)

# I uploaded a .png image instead, for viewing convenience

# Plot confusion matrix

from sklearn.metrics import confusion_matrix



xtrain, xtest, ytrain, ytest = train_test_split(trainx, trainy, test_size=.2, random_state=0)

clf.fit(xtrain, ytrain)



confm = confusion_matrix(ytest, clf.predict(xtest))

confm = confm.astype(float)

#confm = confm/len(ytest)

sns.heatmap(confm, annot=True, fmt=".2f", square=1, cmap='Blues', vmin=0, vmax=100)



plt.title('Confusion matrix')

plt.xlabel('predicted survival')

plt.ylabel('actual survival')

plt.xticks([0.5,1.5],['no','yes'])

plt.yticks([0.5,1.5],['no','yes'])

plt.show()



print('Accuracy: ' + str((confm[0,0] + confm[1,1]) / np.sum(confm))) # (true neg + true pos) / total

print('Sensitivity: ' + str(confm[1,1] / np.sum(confm, 1)[1])) # true pos / actual pos

print('Specificity: ' + str(confm[0,0] / np.sum(confm, 1)[0])) # true neg / actual neg
# Plot ROC Curve

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score



xtrain, xtest, ytrain, ytest = train_test_split(trainx, trainy, test_size=.2, random_state=0)

clf.fit(xtrain, ytrain)



fpr, tpr, th = roc_curve(ytest, clf.predict_proba(xtest)[:,1])



plt.figure(figsize=[4,4])

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], 'k--')



plt.title('ROC curve')

plt.xlabel('false pos rate')

plt.ylabel('true pos rate')

sns.despine(left=False, bottom=False)

plt.show()



print('AUC: ' + str(roc_auc_score(ytest, clf.predict_proba(xtest)[:,1])))
# Get learning curve data (can take minutes)

from sklearn.model_selection import learning_curve



train_sizes, train_scores, test_scores = learning_curve(

    clf, trainx, trainy, train_sizes = np.linspace(0.1, 1.0, 10), cv=5, verbose=0) #10 steps, cv=5
# Plot the learning curves

plt.figure(figsize=[4,3])

sns.tsplot(test_scores.transpose(), time=train_sizes, ci=95, color='C1', marker='o')

sns.tsplot(train_scores.transpose(), time=train_sizes, ci=95, color='C0', marker='o')

plt.ylim((0.6, 1.01))

plt.gca().invert_yaxis()

plt.legend(['test','train'])

leg = plt.gca().get_legend()

leg.legendHandles[0].set_alpha(1)

leg.legendHandles[1].set_alpha(1)

plt.xlabel('N of samples')

plt.ylabel('Score')

plt.title('Learning curves')

sns.despine(left=False, bottom=False)

plt.show()
# Plot feature importances

featstats = pd.DataFrame(feats, columns=['featnames'])

featstats['featimp'] = clf.feature_importances_

featstats['featstd'] = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

featstats = featstats.sort_values('featimp', ascending=True)



plt.figure(figsize=[8,6])

xerr=featstats['featstd']

plt.barh(range(len(featstats)), featstats['featimp'], color=sns.color_palette()[0], xerr=xerr)

plt.yticks(range(len(featstats)), featstats['featnames'])



plt.xlabel('Decrease in impurity')

plt.title('Feature importances')

sns.despine(left=False, bottom=False)

plt.show()
# Set proper format and export for kaggle submission

result = pd.DataFrame(index=test['PassengerId'])

result['Survived'] = clf.predict(testx)



#result.to_csv('prediction.csv')