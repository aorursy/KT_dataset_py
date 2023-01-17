import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1")
columns = ['iid',
'age',
'gender',
'idg', 
'pid',
'match',
'samerace',
'age_o',
'race_o',
'dec_o',
'field_cd',
'race',
'imprace',
'imprelig',
'from',
'goal',
'date',
'go_out',
'career_c',
'sports',
'tvsports',
'exercise',
'dining',
'museums',
'art',
'hiking',
'gaming',
'clubbing', 
'reading',
'tv',
'theater',
'movies',
'concerts',
'music',
'shopping',
'yoga',
'dec',
'attr',
'sinc',
'intel',
'fun',
'amb',
'like',
'prob',
'met'
]

data = data[columns]
data.dropna(inplace=True)
racedict = { 1 : 'black',
             2 : 'white',
             3 : 'latino',
             4 : 'asian',
             5 : 'native american',
             6 : 'other'}

def raceindex(n):
    if n < 1 or n > 6:
        return racedict[0]
    return racedict[n]

data['race'] = data['race'].apply(raceindex)
data['race_o'] = data['race_o'].apply(raceindex)

m = {0 : 'female', 1: 'male'}
data['gender'] = data['gender'].map(m)
cut_data = data
cut_data.drop_duplicates(subset='iid', inplace=True)

melted_data = pd.melt(cut_data, ['iid', 'gender','idg', 'pid', 'match', 'samerace', 'age_o', 'race_o', 'age',
                             'dec_o','race', 'imprace', 'imprelig', 'from', 'goal', 'date',
                             'go_out', 'field_cd', 'career_c', 'dec', 'attr', 'sinc', 'intel', 'fun', 'amb', 'like', 
                             'prob', 'met'], var_name='interest')

melted_data = melted_data.rename(columns={'value' : 'vote'})
melted_data.reset_index(drop=True)
melted_data.tail()
plt.figure(figsize=(20,20))
plt.title('Distribution of Interests Between Genders')

sns.boxplot(x='interest', y='vote', data=melted_data, hue='gender')
melted_data_race = pd.melt(cut_data, ['iid', 'gender','idg', 'pid', 'match', 'samerace', 'age_o', 'race_o',
                             'dec_o', 'field_cd', 'age', 'race', 'from', 'goal', 'date',
                             'go_out', 'career_c', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 
                             'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies',
                             'concerts', 'music', 'shopping', 'yoga', 'dec', 'attr', 'sinc', 'intel', 'fun', 
                             'amb', 'like', 'prob', 'met'], var_name='importance')

melted_data_race = melted_data_race.rename(columns={'value' : 'rating'})
melted_data_race.reset_index(drop=True)
plt.figure(figsize=(10,15))
plt.title('Distribution of Importance of Race/Religion Between Races')

sns.boxplot(x='importance', y='rating', data=melted_data_race, hue='race')
fig, ax = plt.subplots(figsize=(20,15), ncols=3, nrows=2)

ax[0][0].set_title("Attractiveness Distribution")
ax[0][1].set_title("Sincerity Distribution"     )
ax[0][2].set_title("Intelligence Distribution"  )
ax[1][0].set_title("Fun Distribution"           )
ax[1][1].set_title("Ambition Distribution"      )
ax[1][2].set_title("Like Distribution"          )

sns.distplot(data.attr , kde = False, ax=ax[0][0])
sns.distplot(data.sinc , kde = False, ax=ax[0][1])
sns.distplot(data.intel, kde = False, ax=ax[0][2])
sns.distplot(data.fun  , kde = False, ax=ax[1][0])
sns.distplot(data.amb  , kde = False, ax=ax[1][1])
sns.distplot(data.like , kde = False, ax=ax[1][2])
cut_data_yes = cut_data[cut_data['dec'] == 1]
cut_data_no  = cut_data[cut_data['dec'] == 0]
plt.figure(figsize=(10,10))
plt.title('Dater vs. Datee Race for "Yes" Decisions')

sns.countplot(x='race_o', data=cut_data_yes, hue='race')
plt.figure(figsize=(10,10))
plt.title('Dater vs. Datee Race for "No" Decisions')

sns.countplot(x='race_o', data=cut_data_no, hue='race')
corr = data.corr()['dec']
corr.sort_values(ascending=False)
corr_columns = [
'gender',
'match',
'samerace',
'age_o',
'race_o',
'imprace',
'imprelig',
'dec',
'attr',
'sinc',
'intel',
'fun',
'amb',
'like',
]

data_corr = data[corr_columns]

mask = np.zeros_like(data_corr.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (15,15))
sns.heatmap (data_corr.corr(), 
             annot=True,
             mask = mask,
             cmap = 'RdBu_r',
             linewidths=0.1, 
             linecolor='white',
             vmax = .9,
             square=True)

plt.title("Correlations Among Features", y = 1.03,fontsize = 20);
avg_attr_dec     = cut_data[cut_data['dec'] == 1]['attr'].mean()
avg_attr_not_dec = cut_data[cut_data['dec'] == 0]['attr'].mean()

print('The average attractiveness rating of the people who were chosen is: '     + str(avg_attr_dec    ))
print('The average attractiveness rating of the people who were not chosen is: ' + str(avg_attr_not_dec))
import scipy.stats as stats

stats.ttest_1samp(a = cut_data[cut_data['dec'] == 1]['attr'],
                 popmean = avg_attr_not_dec)
degree_freedom = len(cut_data[cut_data['dec'] == 1])

lq = stats.t.ppf(0.025, degree_freedom)
rq = stats.t.ppf(0.975, degree_freedom)

print ('The left quartile range of t-distribution is: '  + str(lq))
print ('The right quartile range of t-distribution is: ' + str(rq))
data = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1")
columns = ['iid',
'age',
'gender',
'pid',
'samerace',
'field_cd',
'age_o',
'race_o',
'dec_o',
'round',
'order',
'race',
'imprace',
'imprelig',
'date',
'go_out',
'career_c',
'sports',
'tvsports',
'exercise',
'dining',
'museums',
'art',
'hiking',
'gaming',
'clubbing', 
'reading',
'tv',
'theater',
'movies',
'concerts',
'music',
'shopping',
'yoga',
'exphappy',
'dec'
]

data = data[columns]
data.dropna(inplace=True)
data.shape
def getPartner(pid, interest):
    if (data['iid'] == pid).any():
        row = data[data['iid'] == pid].iloc[0]
        return row[interest]
    else:
        return np.nan

data['partner_sports']   = data['pid'].apply(lambda x : getPartner(x, 'sports'  ))
data['partner_tvsports'] = data['pid'].apply(lambda x : getPartner(x, 'tvsports'))
data['partner_exercise'] = data['pid'].apply(lambda x : getPartner(x, 'exercise'))
data['partner_dining']   = data['pid'].apply(lambda x : getPartner(x, 'dining'  ))
data['partner_museums']  = data['pid'].apply(lambda x : getPartner(x, 'museums' ))
data['partner_art']      = data['pid'].apply(lambda x : getPartner(x, 'art'     ))
data['partner_hiking']   = data['pid'].apply(lambda x : getPartner(x, 'hiking'  ))
data['partner_gaming']   = data['pid'].apply(lambda x : getPartner(x, 'gaming'  ))
data['partner_clubbing'] = data['pid'].apply(lambda x : getPartner(x, 'clubbing'))
data['partner_reading']  = data['pid'].apply(lambda x : getPartner(x, 'reading' ))
data['partner_tv']       = data['pid'].apply(lambda x : getPartner(x, 'tv'      ))
data['partner_theater']  = data['pid'].apply(lambda x : getPartner(x, 'theater' ))
data['partner_movies']   = data['pid'].apply(lambda x : getPartner(x, 'movies'  ))
data['partner_concerts'] = data['pid'].apply(lambda x : getPartner(x, 'concerts'))
data['partner_music']    = data['pid'].apply(lambda x : getPartner(x, 'music'   ))
data['partner_shopping'] = data['pid'].apply(lambda x : getPartner(x, 'shopping'))
data['partner_yoga']     = data['pid'].apply(lambda x : getPartner(x, 'yoga'    ))
data['partner_career']   = data['pid'].apply(lambda x : getPartner(x, 'career_c'))
data['partner_exphappy'] = data['pid'].apply(lambda x : getPartner(x, 'exphappy'))
data.dropna(inplace=True)
data.shape
x = data.drop(['dec'], axis=1)
y = data['dec']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
logreg = LogisticRegression(solver='liblinear', penalty='l1')
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
logreg_accy = round(accuracy_score(y_pred, y_test),3)
print(logreg_accy)
print (classification_report(y_test, y_pred, labels=logreg.classes_))
print (confusion_matrix(y_pred, y_test))
C_vals = [0.0001, 0.001, 0.01, 0.1,0.13,0.2, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 4.0,4.5,5.0,5.1,5.5,6.0, 10.0, 100.0, 1000.0]
penalties = ['l1','l2']

param = {'penalty': penalties, 'C': C_vals, }

grid = GridSearchCV(logreg, param,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=15,shuffle=True), n_jobs=1)
grid.fit(x_train,y_train)
logreg_grid = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])
logreg_grid.fit(x_train,y_train)
y_pred = logreg_grid.predict(x_test)
logreg_accy = round(accuracy_score(y_test, y_pred), 3)
print (logreg_accy)
print(classification_report(y_test, y_pred, labels=logreg_grid.classes_))
y_score = logreg_grid.decision_function(x_test)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.figure(figsize =[11,9])
plt.plot(fpr, tpr, label= 'ROC curve(area = %0.2f)'%roc_auc, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Speed Dates', fontsize= 18)
plt.show()
precision, recall, _ = precision_recall_curve(y_test, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Speed Daters', fontsize=18)
plt.legend(loc="lower right")
plt.show()
dectree = DecisionTreeClassifier(max_depth = 5, 
                                 class_weight = 'balanced', 
                                 min_weight_fraction_leaf = 0.01)

dectree.fit(x_train, y_train)
y_pred = dectree.predict(x_test)
dectree_accy = round(accuracy_score(y_pred, y_test), 3)
print(dectree_accy)
scores = []
best_pred = [-1, -1]
for i in range(10, 100, 10):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='minkowski', p=2)
    knn.fit(x_train, y_train)
    score = accuracy_score(y_test, knn.predict(x_test))
    
    if score > best_pred[1]:
        best_pred = [i, score]
    scores.append(score)
plt.figure(figsize=[11,9])
plt.plot(range(10,100, 10), scores)
plt.xlabel('N Neighbors', fontsize=18)
plt.ylabel('Accuracy Score', fontsize=18)
plt.title('Accuracy vs. Number of Neighbors for Prediction of Dec', fontsize=18)

plt.show()
data = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1")
columns = ['iid',
'age',
'gender',
'pid',
'samerace',
'field_cd',
'race_o',
'dec_o',
'round',
'order',
'race',       
'imprace',
'imprelig',
'date',
'go_out',
'career_c',
'sports',
'tvsports',
'exercise',
'dining',
'museums',
'art',
'hiking',
'gaming',
'clubbing', 
'reading',
'tv',
'theater',
'movies',
'concerts',
'music',
'shopping',
'yoga',
'exphappy',
'match',
'attr',
'sinc',
'intel',
'fun',
'amb',
'like',
'prob',
'met'
]

data = data[columns]
data.dropna(inplace=True)
data.shape
def getPartner(pid, iid, interest):
        row = data[(data['pid'] == iid) & (data.loc[:, 'iid'] == pid)]
        if row.empty == False:
            return row[interest].iloc[0]
        else:
            return np.nan
data['partner_sports']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'sports'  ), axis=1)
data['partner_tvsports'] = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'tvsports'), axis=1)
data['partner_exercise'] = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'exercise'), axis=1)
data['partner_dining']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'dining'  ), axis=1)
data['partner_museums']  = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'museums' ), axis=1)
data['partner_art']      = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'art'     ), axis=1)
data['partner_hiking']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'hiking'  ), axis=1)
data['partner_gaming']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'gaming'  ), axis=1)
data['partner_clubbing'] = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'clubbing'), axis=1)
data['partner_reading']  = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'reading' ), axis=1)
data['partner_tv']       = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'tv'      ), axis=1)
data['partner_theater']  = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'theater' ), axis=1)
data['partner_movies']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'movies'  ), axis=1)
data['partner_concerts'] = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'concerts'), axis=1)
data['partner_music']    = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'music'   ), axis=1)
data['partner_shopping'] = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'shopping'), axis=1)
data['partner_yoga']     = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'yoga'    ), axis=1)
data['partner_career']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'career_c'), axis=1)
data['partner_attr']     = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'attr'    ), axis=1)
data['partner_sinc']     = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'exphappy'), axis=1)
data['partner_intel']    = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'sinc'    ), axis=1)
data['partner_fun']      = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'fun'     ), axis=1)
data['partner_amb']      = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'amb'     ), axis=1)
data['partner_like']     = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'like'    ), axis=1)
data['partner_prob']     = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'prob'    ), axis=1)
data['partner_met']      = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'met'     ), axis=1)

data.dropna(inplace=True)
data.shape
x = data.drop(['match'], axis=1)
y = data['match']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
logreg = LogisticRegression(solver='liblinear', penalty='l1')
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
logreg_accy = round(accuracy_score(y_pred, y_test),3)
print(logreg_accy)
C_vals = [0.0001, 0.001, 0.01, 0.1,0.13,0.2, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 4.0,4.5,5.0,5.1,5.5,6.0, 10.0, 100.0, 1000.0]
penalties = ['l1','l2']

param = {'penalty': penalties, 'C': C_vals, }

grid = GridSearchCV(logreg, param,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=15,shuffle=True), n_jobs=1)
grid.fit(x_train,y_train)
logreg_grid = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])
logreg_grid.fit(x_train,y_train)
y_pred = logreg_grid.predict(x_test)
logreg_accy = round(accuracy_score(y_test, y_pred), 3)
print (logreg_accy)
print(classification_report(y_test, y_pred, labels=logreg_grid.classes_))
y_score = logreg_grid.decision_function(x_test)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.figure(figsize =[11,9])
plt.plot(fpr, tpr, label= 'ROC curve(area = %0.2f)'%roc_auc, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Speed Dates', fontsize= 18)
plt.show()
y_score = logreg_grid.decision_function(x_test)

precision, recall, _ = precision_recall_curve(y_test, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Speed Daters', fontsize=18)
plt.legend(loc="lower right")
plt.show()
dectree = DecisionTreeClassifier(max_depth = 5, 
                                 class_weight = 'balanced', 
                                 min_weight_fraction_leaf = 0.01)

dectree.fit(x_train, y_train)
y_pred = dectree.predict(x_test)
dectree_accy = round(accuracy_score(y_pred, y_test), 3)
print(dectree_accy)
scores = []
best_pred = [-1, -1]
for i in range(10, 100, 10):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='minkowski', p=2)
    knn.fit(x_train, y_train)
    score = accuracy_score(y_test, knn.predict(x_test))
    
    if score > best_pred[1]:
        best_pred = [i, score]
    scores.append(score)
print (best_pred)
plt.figure(figsize=[11,9])
plt.plot(range(10,100, 10), scores)
plt.xlabel('N Neighbors', fontsize=18)
plt.ylabel('Accuracy Score', fontsize=18)
plt.title('Accuracy vs. Number of Neighbors for Prediction of Match', fontsize=18)

plt.show()
