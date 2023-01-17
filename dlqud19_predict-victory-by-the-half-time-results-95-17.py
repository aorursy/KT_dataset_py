import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

epl = pd.read_csv('../input/epl-results-19932018/EPL_Set.csv')

epl
epl_data = epl.dropna()

epl_data
epl_data.describe(include="all")
# Make varaible name shorter

epl = epl_data
epl_home = epl['HomeTeam']
epl_away = epl['AwayTeam']
'''
key = team_name 
value = total_games
'''
team_total_dic = {}
team_home_dic = {}
team_away_dic = {}
team_list = list(epl['HomeTeam'].unique())
'''

Total home games
Total away games

Total gmaes = Total home games + Total away games

'''

for team in team_list:
    home_games_cnt = 0

    for home in epl_home:
        if home == team:
            home_games_cnt += 1
    
    team_home_dic[team] = home_games_cnt
    
sorted(team_home_dic.items(), key=lambda team : team[1], reverse=True)
for team in team_list:
    away_games_cnt = 0

    for away in epl_away:
        if away == team:
            away_games_cnt += 1
    
    team_away_dic[team] = away_games_cnt
    
sorted(team_away_dic.items(), key=lambda team : team[1], reverse=True)
# team is key of dic
for team in team_home_dic:
    team_total_dic[team] = team_home_dic[team] + team_away_dic[team]


# team_total_dic = team_home_dic + team_away_dic

sorted(team_total_dic.items(), key=lambda team : team[1], reverse=True)
# Full time result

epl_ftr = epl['FTR']
# Get total wins each teams

team_win_dic = {}

for team in team_list:
    win_cnt = 0
    
    # The index had been cleared, so add 924 to access pre index
    for idx, ftr in enumerate(epl_ftr):
        if ftr == 'H' and epl['HomeTeam'][idx + 924] == team:
            win_cnt += 1
        elif ftr == 'A' and epl['AwayTeam'][idx + 924] == team:
            win_cnt += 1
               
    team_win_dic[team] = win_cnt

sorted(team_win_dic.items(), key=lambda team : team[1], reverse=True)
# Get each teams winning rate


total_win_rate = {}

for team in team_list:

    total_win_rate[team] = round((team_win_dic[team] / team_total_dic[team]) * 100, 2)

sorted(total_win_rate.items(), key=lambda team : team[1], reverse=True)
# Get home wins each teams

home_win_cnt = {}
home_win_rate = {}


for team in team_list:
    win_cnt = 0
    
    # The index had been cleared, so add 924 to access pre index
    for idx, ftr in enumerate(epl_ftr):
        if ftr == 'H' and epl['HomeTeam'][idx + 924] == team:
            win_cnt += 1
               
    home_win_cnt[team] = win_cnt



for team in team_list:

    home_win_rate[team] = round((home_win_cnt[team] / team_home_dic[team]) * 100, 2)

sorted(home_win_rate.items(), key=lambda team : team[1], reverse=True)
# Get away wins each teams

away_win_cnt = {}
away_win_rate = {}


for team in team_list:
    win_cnt = 0
    
    # The index had been cleared, so add 7582 to access pre index
    for idx, ftr in enumerate(epl_ftr):
        if ftr == 'A' and epl['AwayTeam'][idx + 924] == team:
            win_cnt += 1
               
    away_win_cnt[team] = win_cnt



for team in team_list:

    away_win_rate[team] = round((away_win_cnt[team] / team_away_dic[team]) * 100, 2)

sorted(away_win_rate.items(), key=lambda team : team[1], reverse=True)
# Is there any team have been more wins in away?

strong_away_team = False
print('Teams that strong in when away:')

for team in team_list:
    if away_win_rate[team] > home_win_rate[team]:
        print(team)
        strong_away_team = True
       

if strong_away_team == False:
    print("No such team")
# There isn`t...
# Have to preprocessing

# Conver float to int

epl.FTAG = epl.FTAG.astype(int)
epl.FTHG = epl.FTHG.astype(int)
epl.HTAG = epl.HTAG.astype(int)
epl.HTHG = epl.HTHG.astype(int)
# Visualization

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set()

epl.hist(figsize=(21.7, 13.27))
print('FTAG : ',epl['FTAG'].sum())
print('FTHG : ',epl['FTHG'].sum())
print('HTAG : ',epl['HTAG'].sum())
print('HTHG : ',epl['HTHG'].sum())
epl.boxplot(figsize=(8,8))
sns.boxplot(data=epl)
total_win_rate = pd.DataFrame.from_dict(total_win_rate, orient='index')

total_win_rate.columns = ['Win rate']
# Modify total_win_rate by descending in Win rate

total_win_rate_desc = total_win_rate.sort_values("Win rate", ascending=False)
total_win_rate_desc
# Divide teams by win rate 
# top, mid, low

top = total_win_rate_desc[:16]

mid = total_win_rate_desc[16:32]

low =total_win_rate_desc[32:48]
top.plot.bar()
mid.plot.bar()
low.plot.bar()
# Transpose for get easily team list

top_t = top.T
mid_t = mid.T
low_t = low.T
top_team_list = []

for team in top_t:
    top_team_list.append(team)
    
top_team_list
mid_team_list = []

for team in mid_t:
    mid_team_list.append(team)
    
mid_team_list
low_team_list = []

for team in low_t:
    low_team_list.append(team)
    
low_team_list
epl['HTR'].value_counts(normalize=True)
epl['FTR'].value_counts(normalize=True)
epl['FTR'].value_counts(normalize=True)
sns.countplot(y=epl['HTR'])
sns.countplot(y=epl['FTR'])
# Mean of HTHG and HTAG 

print(epl['HTHG'].groupby(epl['HTR']).mean(),'\n\n', epl['HTAG'].groupby(epl['HTR']).mean())
epl['HTHG'].groupby(epl['HTR']).mean().plot.bar()
epl['HTAG'].groupby(epl['HTR']).mean().plot.bar()
sns.countplot(epl['HTHG'], hue=epl['HTR'])
sns.countplot(epl['HTAG'], hue=epl['HTR'])
sns.countplot(epl['FTHG'], hue=epl['FTR'])
sns.countplot(epl['FTAG'], hue=epl['FTR'])
# Drop the non valued columns

epl_prep = epl.drop(['Div'], axis=1)
epl_scatt = epl.drop(['Div'], axis=1)
from sklearn import preprocessing

le_hda = preprocessing.LabelEncoder()
le_hda.fit(epl_prep['FTR'])
le_hda_pred = le_hda.transform(epl_prep['FTR'])
epl_prep.insert(0, 'Predicted', le_hda_pred)
epl_prep
le_hda_htr = le_hda.transform(epl_prep['HTR'])
le_hda_ftr = le_hda.transform(epl_prep['FTR'])

epl_prep['HTR'] = le_hda_htr
epl_prep['FTR'] = le_hda_ftr
le_date = preprocessing.LabelEncoder()
le_date.fit(epl_prep['Date'])

le_date = le_date.transform(epl_prep['Date'])

epl_prep['Date'] = le_date
le_ssn = preprocessing.LabelEncoder()
le_ssn.fit(epl_prep['Season'])

le_ssn = le_ssn.transform(epl_prep['Season'])

epl_prep['Season'] = le_ssn
# Encoidng the team top, mid, low

'''

top_team_list 0
mid_team_list 1
low_team_list 2

'''

for team in epl_prep['HomeTeam']:
    for top in top_team_list:
        if team == top:
            epl_prep['HomeTeam'] = np.where(epl_prep.HomeTeam == top, 0, epl_prep.HomeTeam)

    for mid in mid_team_list:
        if team == mid:
            epl_prep['HomeTeam'] = np.where(epl_prep.HomeTeam == mid, 1, epl_prep.HomeTeam)
            
    for low in low_team_list:
        if team == low:
            epl_prep['HomeTeam'] = np.where(epl_prep.HomeTeam == low, 2, epl_prep.HomeTeam)




for team in epl_prep['AwayTeam']:
    for top in top_team_list:
        if team == top:
            epl_prep['AwayTeam'] = np.where(epl_prep.AwayTeam == top, 0, epl_prep.AwayTeam)

    for mid in mid_team_list:
        if team == mid:
            epl_prep['AwayTeam'] = np.where(epl_prep.AwayTeam == mid, 1, epl_prep.AwayTeam)
            
    for low in low_team_list:
        if team == low:
            epl_prep['AwayTeam'] = np.where(epl_prep.AwayTeam == low, 2, epl_prep.AwayTeam)

# Delet the FTR, its the same as 'Predicted' attribute

epl_prep = epl_prep.drop(['FTR'], axis=1)
# Date is too specific attribute so drop it

epl_prep = epl_prep.drop(['Date'], axis=1)
# FTHG and FTAG is not in frist half so drop it

epl_prep = epl_prep.drop(['FTHG'], axis=1)
epl_prep = epl_prep.drop(['FTAG'], axis=1)
# Categorized data (numerical)

epl_prep
from sklearn.model_selection import train_test_split

# Make train_test_split

epl_train = epl_prep.iloc[:, epl_prep.columns != 'Predicted']
epl_test = epl_prep.iloc[:, epl_prep.columns == 'Predicted']


X_train, X_test, y_train, y_test = train_test_split(epl_train,
                                                     epl_test,
                                                     test_size = 0.20, 
                                                     random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                   y_train,
                                                   test_size = 0.25, 
                                                   random_state = 42)
from sklearn.ensemble import RandomForestClassifier

# Validation check

rf_v = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=2,
                             min_samples_leaf=10,
                             max_features='auto',
                             oob_score=True,
                             random_state=42,
                             n_jobs=-1)

rf_v.fit(X_train, y_train.values.ravel())
print("OOB Score : %.4f" % rf_v.oob_score_)
score = rf_v.score(X_val, y_val)
print("Score : ", score)
# Test data check

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=2,
                             min_samples_leaf=10,
                             max_features='auto',
                             oob_score=True,
                             random_state=42,
                             n_jobs=-1)

rf.fit(X_train, y_train.values.ravel())
print("OOB Score : %.4f" % rf.oob_score_)
score = rf.score(X_test, y_test)
print("Score : ", score)
# Get the best parameters

# -- This take some minutes, so run it if you want --

# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier


# param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}

# gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

# gs = gs.fit(X_train, y_train.values.ravel())

# print(gs.best_score_)

# print(gs.best_params_)

# print(gs.cv_results_)


# -- This is the results --

# -> 0.648048048048048

# -> {'criterion': 'gini', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 1000}

# -> {'mean_fit_time': array([0.32810688, 0.55880944, 2.31270464, 4.24339898, 5.48761559, ...
plt.scatter(epl_prep['HTHG'],
            
            epl_prep['HTAG'],
            
            alpha=0.42)

plt.xlabel('Half Time Home Goal', fontsize=14)
plt.ylabel('Half Time Away Goal', fontsize=14)
plt.legend()
sns.scatterplot(x='HTHG', 

                y='HTAG', 

                hue='HTR',
                
                s=90,

                style='FTR',

                data=epl_scatt)

plt.show()
sns.scatterplot(x='HTHG', 

                y='FTHG', 

                hue='HTR',
                
                s=90,

                style='FTR',

                data=epl_scatt)

plt.show()
sns.scatterplot(x='HomeTeam', 

                y='HTHG', 

                hue='HTR',
                
                s=90,

                style='FTR',

                data=epl_scatt)

sns.set(font_scale=0.4)
sns.set(rc={'figure.figsize':(1001.7,800.27)})


plt.show()
# Importance of attributes

epl_imp = pd.concat((pd.DataFrame(epl_prep.iloc[:, 1:].columns, columns = ['Attribute']), 
          pd.DataFrame(rf.feature_importances_, columns = ['Importance'])),    
          axis = 1).sort_values(by='Importance', ascending=False)

epl_imp
# Ascending importance

epl_imp.sort_values('Importance', ascending=True, inplace=True)
epl_imp.plot(kind='barh', x='Attribute', y='Importance', legend=False, figsize=(6, 10))

plt.title('Random forest feature importance', fontsize = 24)
plt.xlabel('')
plt.ylabel('')
plt.xticks([], [])
plt.yticks(fontsize=20)
plt.show()