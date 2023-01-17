%matplotlib inline



import pandas as pd

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

import numpy as np
dfCatan = pd.read_csv("../input/catanstats.csv")
dfCatan['win'] = (dfCatan['points'] >= 10).astype(int)

dfCatan['me'].fillna(0, inplace = True)

dfCatan.head()
#Is the position of the player important for winning?

fig = plt.figure(figsize=(13,4))



ax = fig.add_subplot(1,2,1)

player_win = dfCatan[dfCatan['win'] == 1]['player'].value_counts()

player_loss = dfCatan[dfCatan['win'] == 0]['player'].value_counts()/3



dfTempPlot = pd.DataFrame([player_win,player_loss])

dfTempPlot.index = ['Win','Loss']

dfTempPlot.plot(kind = 'bar',stacked = True, title = 'Winning depends on player position...', ax=ax)

ax.set_ylabel('Games (scaled "loss" bar for comparison)')



#Does this also hold for "me"?

ax2 = fig.add_subplot(1,2,2)

me_win = dfCatan[(dfCatan['win'] == 1) & (dfCatan['me'] == 1.0)]['player'].value_counts()

me_loss = dfCatan[(dfCatan['win'] == 0) & (dfCatan['me'] == 1.0)]['player'].value_counts()



dfTempPlot = pd.DataFrame([me_win,me_loss])

dfTempPlot.index = ['Win','Loss']

dfTempPlot.plot(kind = 'bar',stacked = True, title = '... also for "me" - being player 2 is a good thing', ax=ax2)

ax2.set_ylabel('Games')



print('"me" wins ' + str(100*sum(dfCatan[dfCatan['me'] == 1.0]['win'])/float(max(dfCatan['gameNum']))) + '% of all games!')
#Let's look at some correlations - small data set, so take p-value with grain of salt

from scipy.stats.stats import pearsonr



fig = plt.figure(figsize=(15,6))

ax = fig.add_subplot(2,3,1)

ax.scatter(dfCatan['production'], dfCatan['points'], c='black')

ax.set_title('Points vs Production')

ax.text(20, 12, 'r = '+ str(round(pearsonr(dfCatan['production'], dfCatan['points'])[0],2)))



ax2 = fig.add_subplot(2,3,2)

ax2.scatter(dfCatan['tradeGain'], dfCatan['points'], c='black')

ax2.set_title('Points vs Trade gain')

ax2.text(0, 12, 'r = '+ str(round(pearsonr(dfCatan['tradeGain'], dfCatan['points'])[0],2)))



ax3 = fig.add_subplot(2,3,3)

ax3.scatter(dfCatan['robberCardsGain'], dfCatan['points'], c='black')

ax3.set_title('Points vs Robber cards gain')

ax3.text(0, 12, 'r = '+ str(round(pearsonr(dfCatan['robberCardsGain'], dfCatan['points'])[0],2)))



ax4 = fig.add_subplot(2,3,4)

ax4.scatter(dfCatan['tribute'], dfCatan['points'], c='red')

ax4.set_title('Points vs Tribute')

ax4.text(0, 1, 'r = '+ str(round(pearsonr(dfCatan['tribute'], dfCatan['points'])[0],2)))



ax5 = fig.add_subplot(2,3,5)

ax5.scatter(dfCatan['tradeLoss'], dfCatan['points'], c='red')

ax5.set_title('Points vs Trade loss')

ax5.text(0, 1, 'r = '+ str(round(pearsonr(dfCatan['tradeLoss'], dfCatan['points'])[0],2)))



ax6 = fig.add_subplot(2,3,6)

ax6.scatter(dfCatan['robberCardsLoss'], dfCatan['points'], c='red')

ax6.set_title('Points vs Robber cards loss')

ax6.text(0, 1, 'r = '+ str(round(pearsonr(dfCatan['robberCardsLoss'], dfCatan['points'])[0],2)))



fig.tight_layout()
#Create train and test data sets

from sklearn.cross_validation import train_test_split



#Let's use production, tradeGain, robberCardsGain and tradeLoss only

inputData = dfCatan[['production','tradeGain','robberCardsGain','tradeLoss']]

targetVal = dfCatan['win']



#And scale the data to have mean=0 and std=1

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(inputData)

inputData_scaled = scaler.transform(inputData)



X_train, X_test, y_train, y_test = train_test_split(inputData_scaled, targetVal, test_size=0.20)#, random_state=42)
#Let's make an RF model

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV



forest = RandomForestClassifier()



parameter_grid = {'max_depth': [4,5,6,7,8], 

                  'n_estimators': [120,200,250,500]

                   }

cross_validation = StratifiedKFold(y_train, n_folds=5)

grid_search = GridSearchCV(estimator=forest, param_grid=parameter_grid, cv=cross_validation)



#Fit the model

grid_search.fit(X_train, y_train)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
#Piece of code for plotting a neat confusion matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(2)

    plt.xticks(tick_marks, ['Loss', 'Win'], rotation=45)

    plt.yticks(tick_marks, ['Loss', 'Win'])

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.metrics import confusion_matrix



#Now it's time for predictions...

output = grid_search.predict(X_test)

print('Accuracy: ' + str(sum(output == y_test)/float(len(y_test))))



cm = confusion_matrix(y_test, output)



print('Confusion matrix')

print(cm)

fig = plt.figure(figsize=(10,4))

plot_confusion_matrix(cm)