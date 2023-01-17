import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('fivethirtyeight')



import warnings

warnings.filterwarnings('ignore')
match_ovr = pd.read_csv('../input/LeagueofLegends.csv')

match_obj = pd.read_csv('../input/objValues.csv')
#function to help format data for analysis



def obj_count(x, b_obj_type, r_obj_type):

    #data set-up

    x = match_obj.loc[match_obj['ObjType'].isin([b_obj_type, r_obj_type])]

    

    #blue obj

    b_obj = x.loc[x['ObjType'] == b_obj_type]

    x[b_obj_type] = b_obj.count(axis=1) -2

    

    #red obj

    r_obj = x.loc[x['ObjType'] == r_obj_type]

    x[r_obj_type] = r_obj.count(axis=1) -2

    

    x = x.groupby('MatchHistory').sum()[[b_obj_type, r_obj_type]]

    

    return(x)
#calling function obj_count & joining data into single df



match_stats = obj_count(match_obj, 'bInhibs', 'rInhibs')



for i,j in [('bTowers','rTowers'), ('bDragons','rDragons'), ('bBarons','rBarons'), ('bHeralds','rHeralds')]:

    match_stats = match_stats.join(obj_count(match_obj, i, j))



for i,j,k in [('bInhibs','rInhibs','Inhibs_Dif'), ('bTowers','rTowers', 'Tower_Dif'), ('bDragons','rDragons', 'Dragon_Dif'), ('bBarons','rBarons','Baron_Dif'), ('bHeralds','rHeralds', 'Heralds_Dif')]:

    match_stats[k] = match_stats[i] - match_stats[j]



df1 = match_stats.join(match_ovr[['MatchHistory', 'bResult']].set_index('MatchHistory')).astype(int)
#getting frequency of blue, red by objective type -- used to size markers in scatter plot later

r_b = pd.Series(df1['bTowers'].map(str) + ',' + df1['rTowers'].map(str), name='bTowers').to_frame()



for i,j in [('bDragons','rDragons'),('bInhibs','rInhibs'), ('bBarons','rBarons'), ('bHeralds','rHeralds')]:

    r_b = r_b.join(pd.Series(df1[i].map(str)+','+df1[j].map(str), name=i).to_frame())
df1.head(2)
print('Blue won {:.0f}% of games'.format(df1['bResult'].mean()*100))
def some_plots(obj_type, b_obj, r_obj, obj_name, des_kil):

    fig = plt.figure(figsize=(10,10))

    

    ax1 = plt.subplot2grid((2,4), (0,0), colspan=3)

    sns.barplot(x=obj_type, y='bResult', data=df1, ax=ax1, errwidth=2)

    #ax1.set_xlim(right=22.5)

    ax1.set_title('Win Percent by {}'.format(obj_name), fontsize=18)

    ax1.set_ylabel('Win Percentage')

    ax1.set_xlabel('')

    

    ax2 = plt.subplot2grid((2,4), (0,3))

    sns.boxplot(df1[obj_type], color='#4daf8b', orient='v')

    ax2.set_title('Boxplot: {}'.format(obj_name), fontsize=18)

    ax2.set_ylabel('')

    ax2.set_xlabel('')



    ax3 = plt.subplot2grid((2,4), (1,0), colspan=2)

    ax3.scatter(x=df1[b_obj], y=df1[r_obj], s=r_b[b_obj].value_counts()*.5, edgecolors='face')

    ax3.set_title('Blue vs. Red')

    ax3.set_xlabel('{} by Blue'.format(des_kil))

    ax3.set_ylabel('{} by Red'.format(des_kil))

    ax3.grid(axis='x')



    ax4 = plt.subplot2grid((2,4), (1,2), colspan=2)

    sns.distplot(df1[obj_type], ax=ax4, norm_hist=True, axlabel=None)

    ax4.set_title('{} Distribution (%)'.format(obj_name))

    ax4.set_xlabel('')

    ax4.grid(axis='x')



    plt.tight_layout;

    

    return(ax1,ax2,ax3,ax4)
some_plots('Tower_Dif', 'bTowers', 'rTowers', 'Tower Differential', 'Destroyed');
some_plots('Inhibs_Dif', 'bInhibs', 'rInhibs', 'Inhib Differential', 'Destroyed');
some_plots('Dragon_Dif', 'bDragons', 'rDragons', 'Dragon Differential', 'Killed');
#quick look at StDev

for i in ['Tower_Dif', 'Inhibs_Dif','Dragon_Dif']:

    print('{0} Stanard Deviations: {1:.2f}'.format(i, df1[i].std()))
some_plots('Baron_Dif', 'bBarons', 'rBarons', 'Baron Differential', 'Killed');
df1['Baron_Dif'].std()
some_plots('Heralds_Dif', 'bHeralds', 'rHeralds', 'Herald Differential', 'Killed');
dif_list = ['Tower_Dif','Inhibs_Dif','Baron_Dif','Dragon_Dif','Heralds_Dif']
sns.heatmap(df1[dif_list].corr(), annot=True)
from sklearn.preprocessing import normalize

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold
dif_list.remove('Inhibs_Dif')



x = normalize(df1[dif_list].values)

y = df1['bResult'].values
kf = StratifiedKFold(y, n_folds=5, random_state=1)

lr = LogisticRegression(C=3, random_state=1)



scores = []



for tr, cv in kf:

    model = lr.fit(x[tr], y[tr])

    probs = model.predict_proba(x[cv])

    scores.append(lr.score(x[cv], y[cv]))



np.mean(scores)
#plot feature coef_ values

feat_imp = pd.DataFrame({'feat':dif_list,

                         'feat_imp':model.coef_.ravel()})#.sort_values(by='feat_imp', ascending=False)



plt.figure(figsize=(10,7))

sns.barplot(x='feat', y='feat_imp', data=feat_imp)
#check a few scenarios

inp = np.array([0,0,0,0]) #can adjust [tower, baron, dragon, heralds] - don't know if trustworthy



out = model.predict_proba(inp.reshape(1,-1)).ravel()



for i,j in enumerate(dif_list):

    print('{0} input: {1}'.format(j, inp[i]))



print('\nBlue Win Perc: {:.0f}%'.format(out[1]*100))