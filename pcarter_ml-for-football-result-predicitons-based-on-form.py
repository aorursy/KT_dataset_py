import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/FootballEurope.csv')
print(df.head())
years = []
month = []
day = []
for i in range(len(df['date'])):
    years.append(df['date'][i].split('-')[0])
    month.append(df['date'][i].split('-')[1])
    day.append(df['date'][i].split('-')[2])        
df["Year"] = years
df["Month"] = month
df["Day"] = day
df['Year'] = df['Year'].apply(pd.to_numeric)
df['Month'] = df['Month'].apply(pd.to_numeric)
df['Day'] = df['Day'].apply(pd.to_numeric)

#1-Home, 0-Draw, -1-Away
result1 = []
result2 = []
result3 = []
for i in range(len(df['homeGoalFT'])):
    if(df['homeGoalFT'][i] > df['awayGoalFT'][i]):
        result1.append(1)
    elif(df['homeGoalFT'][i] == df['awayGoalFT'][i]):
        result1.append(0)
    else:
        result1.append(-1)
df["Result"] = result1
df['Result'] = df['Result'].apply(pd.to_numeric)
print(df['date']+' '+df['homeTeam'])
print(df['Year'])
#Sort the dataset into chronological order
df = df.sort_values(by='date')
print(df['date']+' '+df['homeTeam'])
print(df['Year'])
df = df.reindex(index=df.index[::-1])
#Correlation Matrix of features available

corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()
#Prints the top correlated features

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop
def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
print(get_top_abs_correlations(corrmat, 30))
#Print all unique teams
Teams=df['homeTeam'].unique()
Teams
#Both features from both teams previous preformance will indicate the outcome, (wins, goals, corners)
alist = []
namelist = []
teamagainstlist = []
#Total_avg[:] = Teams[:]
for k in range(0, len(Teams)):
    Xtmp = df[np.logical_or(df['homeTeam']==Teams[k], df['awayTeam']==Teams[k])]
    #print(Xtmp[['homeTeam']])
    Ytmp = Xtmp[['Result']]
    #print(list(Xtmp))
    #print(Xtmp['homeTeam'].iloc[k])
    #Xtmp = np.asarray(Xtmp)
    Ztmp = np.zeros((len(Xtmp), 6))
    for i in range(0, len(Xtmp)):
        if(Xtmp['homeTeam'].iloc[i]==Teams[k]):
            #Ztmp = Ztmp.append({'GoalFT': Xtmp['homeGoalFT'].iloc[i], 'CornersFT': Xtmp['homeCornersTotalFT'].iloc[i], 'Result': Xtmp['Result'].iloc[i], 'oppTeam': Xtmp['awayTeam'].iloc[i], 'date': Xtmp['date'].iloc[i]}, ignore_index=True)
            Ztmp[i, 0] = Xtmp['homeGoalFT'].iloc[i]
            Ztmp[i, 1] = Xtmp['homePossessionFT'].iloc[i]
            Ztmp[i, 2] = Xtmp['Result'].iloc[i]
            #Ztmp[i, 3] = Xtmp['awayTeam'].iloc[i]
            Ztmp[i, 3] = Xtmp['Day'].iloc[i]
            Ztmp[i, 4] = Xtmp['Month'].iloc[i]
            Ztmp[i, 5] = Xtmp['Year'].iloc[i]
            #Ztmp[i, 6] = float(Rank[k])
            #for kl in range(0, len(Teams)):
            #    if(Xtmp['awayTeam'].iloc[i]==Teams[kl]):
            #        Ztmp[i, 7] = float(Rank[kl])
    
        elif(Xtmp['awayTeam'].iloc[i]==Teams[k]):
            #Ztmp = Ztmp.append({'GoalFT': Xtmp['awayGoalFT'].iloc[i], 'CornersFT': Xtmp['awayCornersTotalFT'].iloc[i], 'Result': Xtmp['Result'].iloc[i], 'oppTeam': Xtmp['homeTeam'].iloc[i], 'date': Xtmp['date'].iloc[i]}, ignore_index=True)
            Ztmp[i, 0] = Xtmp['awayGoalFT'].iloc[i]
            Ztmp[i, 1] = Xtmp['awayPossessionFT'].iloc[i]
            Ztmp[i, 2] = -Xtmp['Result'].iloc[i]
            #Ztmp[i, 3] = Xtmp['homeTeam'].iloc[i]
            Ztmp[i, 3] = Xtmp['Day'].iloc[i]
            Ztmp[i, 4] = Xtmp['Month'].iloc[i]
            Ztmp[i, 5] = Xtmp['Year'].iloc[i]
            #Ztmp[i, 6] = float(Rank[k])
            #for kl in range(0, len(Teams)):
            #    if(Xtmp['homeTeam'].iloc[i]==Teams[kl]):
            #        Ztmp[i, 7] = float(Rank[kl])
    
    #Ztmp = Ztmp[np.logical_and(Ztmp[:, 6]!=0,Ztmp[:, 7]!=0)] ###Fix for now there is a bug here needs to be fixed
    
    Ztmp_2 = np.zeros((len(Ztmp)-10, 7))
    for i in range(10, len(Ztmp)):
        #Ztmp_2[i-6, 0] = Ztmp[i, 0]
        #Ztmp_2[i-6, 1] = Ztmp[i, 1]
        #Ztmp_2[i-6, 2] = Ztmp[i, 2]
        #print(Ztmp[i-6:i, 7]/Ztmp[i-6:i, 6])
        Ztmp_2[i-10, 0] = np.mean(Ztmp[i-10:i, 0])#, weights=(Ztmp[i-6:i, 7]/Ztmp[i-6:i, 6])) #Last 6 games average goals
        #Ztmp_2[i-6, 4] = np.mean(Ztmp[i-6:i, 1])
        Ztmp_2[i-10, 1] = np.mean(Ztmp[i-10:i, 1])#, weights=(Ztmp[i-6:i, 7]/Ztmp[i-6:i, 6]))
        Ztmp_2[i-10, 2] = np.mean(Ztmp[i-10:i, 2])#, weights=(Ztmp[i-6:i, 7]/Ztmp[i-6:i, 6]))#Last 6 games average wins
        Ztmp_2[i-10, 3] = Ztmp[i, 3]
        Ztmp_2[i-10, 4] = Ztmp[i, 4]
        Ztmp_2[i-10, 5] = Ztmp[i, 5]
        Ztmp_2[i-10, 6] = Ztmp[i, 2]
        #Ztmp_2[i-6, 7] = Ztmp[i, 6]
        #Ztmp_2[i-6, 3] = Ztmp[i, 4]
    #print(Ztmp_2)
    alist.append(Ztmp_2)
    namelist.append(Teams[k])
    #NEED TO OUTPUT TEAM THEY WERE PLAYING AGAINST and DATE to cross-check then build train set and test set
    #what need to do is loop through all games and then use both teams current stats to determine outcome. (HavgWins, HavgGoals, AavgWins, AavgGoals, GameResult)
    
print(alist[11])
print(namelist[11])
#Loop over all games and take the 4 features 2home/2away and result to go into ML
Full = np.zeros((len(df), 7))

#print(alist[0])

for i in range(0, len(df)):
    #print(df['homeTeam'].iloc[i])
    #print(df['awayTeam'].iloc[i])
    #print(df['Day'].iloc[i])
    #print(df['Month'].iloc[i])
    #print(df['Year'].iloc[i])
    
    for j in range(0, len(Teams)):
        if(namelist[j]==df['homeTeam'].iloc[i]):
            for k in range(0, len(alist[j])):
                if(np.logical_and(alist[j][k, 3]==df['Day'].iloc[i], alist[j][k, 4]==df['Month'].iloc[i])):
                    if(alist[j][k, 5]==df['Year'].iloc[i]):
                        Full[i, 0] = alist[j][k][0]
                        Full[i, 1] = alist[j][k][1]
                        Full[i, 2] = alist[j][k][2]
                        #Full[i, 3] = alist[j][k][7]
                        Full[i, 6] = alist[j][k][6]
        elif(namelist[j]==df['awayTeam'].iloc[i]):
            for k in range(0, len(alist[j])):
                if(np.logical_and(alist[j][k, 3]==df['Day'].iloc[i], alist[j][k, 4]==df['Month'].iloc[i])):
                    if(alist[j][k, 5]==df['Year'].iloc[i]):
                        Full[i, 3] = alist[j][k][0]
                        Full[i, 4] = alist[j][k][1]
                        Full[i, 5] = alist[j][k][2]
                        #Full[i, 7] = alist[j][k][7]
Fulln=Full[1000:len(Full)]

X_train = Fulln[0:int(0.9*len(Fulln)), 0:6]
Y_train = Fulln[0:int(0.9*len(Fulln)), 6]

X_test = Fulln[int(0.9*len(Fulln)):len(Fulln), 0:6]
Y_test = Fulln[int(0.9*len(Fulln)):len(Fulln), 6]

print(X_train[0])
# machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
#acc_log

correct = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        correct=correct+1
correct = correct/len(X_test)
acc_log = correct*100
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
#acc_svc

print(Y_pred)
print(Y_test)

correct = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        correct=correct+1
correct = correct/len(X_test)
acc_svc = correct*100
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
#acc_knn

correct = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        correct=correct+1
correct = correct/len(X_test)
acc_knn = correct*100
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
#acc_gaussian

correct = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        correct=correct+1
correct = correct/len(X_test)
acc_gaussian = correct*100
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
#acc_perceptron

correct = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        correct=correct+1
correct = correct/len(X_test)
acc_preceptron = correct*100
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
#acc_linear_svc

correct = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        correct=correct+1
correct = correct/len(X_test)
acc_linear_svc = correct*100
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
#acc_sgd

correct = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        correct=correct+1
correct = correct/len(X_test)
acc_sgd = correct*100
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
#acc_decision_tree

correct = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        correct=correct+1
correct = correct/len(X_test)
acc_decision_tree = correct*100
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#acc_random_forest

correct = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        correct=correct+1
correct = correct/len(X_test)
acc_random_forest = correct*100
Y_pred = np.random.uniform(-1, 1, len(X_test))
print(Y_pred[0])
for i in range(0, len(Y_pred)):
    Y_pred[i] = int(round(Y_pred[i]))
correct = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        correct=correct+1
correct = correct/len(X_test)
acc_uniform_random = correct*100
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Uniform Random'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree, acc_uniform_random]})
models.sort_values(by='Score', ascending=False)
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
#acc_svc

#print(Y_pred)
#print(Y_test)

correct1 = 0
correct2 = 0
correct3 = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        if(Y_test[i]==1):
            correct1=correct1+1
        elif(Y_test[i]==0):
            correct2 = correct2+1
        else:
            correct3 = correct3+1
correct1 = correct1/len(Y_test[(Y_test[:]==1)])
acc_svm1 = correct1*100
correct2 = correct2/len(Y_test[(Y_test[:]==0)])
acc_svm2 = correct2*100
correct3 = correct3/len(Y_test[(Y_test[:]==-1)])
acc_svm3 = correct3*100

print('HW accuracy: ', acc_svm1, '% of', len(Y_test[(Y_test[:]==1)]))
print('Draw: ', acc_svm2, '% of', len(Y_test[(Y_test[:]==0)]))
print('AW accuracy: ', acc_svm3, '% of', len(Y_test[(Y_test[:]==-1)]))
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
#acc_log

correct1 = 0
correct2 = 0
correct3 = 0
for i in range(0, len(X_test)):
    if(Y_test[i] == Y_pred[i]):
        if(Y_test[i]==1):
            correct1=correct1+1
        elif(Y_test[i]==0):
            correct2 = correct2+1
        else:
            correct3 = correct3+1
correct1 = correct1/len(Y_test[(Y_test[:]==1)])
acc_lr1 = correct1*100
correct2 = correct2/len(Y_test[(Y_test[:]==0)])
acc_lr2 = correct2*100
correct3 = correct3/len(Y_test[(Y_test[:]==-1)])
acc_lr3 = correct3*100

print('HW accuracy: ', acc_lr1, '% of', len(Y_test[(Y_test[:]==1)]))
print('Draw: ', acc_lr2, '% of', len(Y_test[(Y_test[:]==0)]))
print('AW accuracy: ', acc_lr3, '% of', len(Y_test[(Y_test[:]==-1)]))
