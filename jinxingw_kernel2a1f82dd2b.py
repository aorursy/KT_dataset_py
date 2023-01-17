import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
alldata = pd.read_csv('../input/data.csv')

data = alldata.copy()
data['team_id'].astype('category').dtypes
data['team_name'].astype('category').dtypes
data.drop('team_id', axis=1, inplace=True)#一样的

data.drop('team_name', axis=1, inplace=True)#一样的
import seaborn as sns

sns.pairplot(data, vars=['loc_x', 'lon', 'loc_y', 'lat'], hue='shot_made_flag', size=3)

plt.show()
data.drop('lon', axis=1, inplace=True)#loc_x

data.drop('lat', axis=1, inplace=True)#loc_y
t = data.loc[:,['game_id', 'shot_made_flag']]

t = t[~t['shot_made_flag'].isnull()]



ft = t.groupby(['game_id']).sum()/t.groupby(['game_id']).count()

data_frame = pd.DataFrame(ft, columns=['shot_made_flag'])

index = list(data_frame.index)



plt.figure(figsize=(16, 8))

plt.scatter(index, data_frame['shot_made_flag'], c='r')

plt.xlabel('game_id')

plt.ylabel('accuracy')

plt.show()
t = data.loc[:,['game_event_id', 'shot_made_flag']]

t = t[~t['shot_made_flag'].isnull()]



ft = t.groupby(['game_event_id']).sum()/t.groupby(['game_event_id']).count()

data_frame = pd.DataFrame(ft, columns=['shot_made_flag'])

index = list(data_frame.index)



plt.figure(figsize=(16, 8))

plt.scatter(index, data_frame['shot_made_flag'], c='r')

plt.xlabel('game_event_id')

plt.ylabel('accuracy')

plt.show()
data[data['game_event_id']>550][['game_event_id', 'shot_made_flag']].sort_values(by='game_event_id')
data.drop('game_id', axis=1, inplace=True)#独立

data.drop('game_event_id', axis=1, inplace=True)#独立
data['shot_id']
data.drop('shot_id', axis=1, inplace=True)
data[['season', 'game_date']]
plt.figure(figsize=(16, 8))

sns.countplot(x="season", hue="shot_made_flag", data=data)

plt.show()
data['game_date'] = pd.to_datetime(data['game_date'])

data['game_year'] = data['game_date'].dt.year

data['game_month'] = data['game_date'].dt.month

data.drop('game_date', axis=1, inplace=True)

data.drop('season', axis=1, inplace=True)
data['home_play'] = data['matchup'].str.contains('vs').astype('int')

data.drop('matchup', axis=1, inplace=True)
plt.figure(figsize=(8, 8))

sns.countplot(x="home_play", hue="shot_made_flag", data=data)

plt.show()#横轴0表示客场，1表示主场
data[['loc_x', 'loc_y', 'shot_distance']]
data['dist'] = np.sqrt(data['loc_x']**2 + data['loc_y']**2)



loc_x_zero = data['loc_x'] == 0

data['angle'] = np.array([0]*len(data))

data['angle'][~loc_x_zero] = np.arctan(data['loc_y'][~loc_x_zero] / data['loc_x'][~loc_x_zero])

data['angle'][loc_x_zero] = np.pi / 2
data[['shot_zone_area', 'shot_zone_basic', 'shot_zone_range']]
from matplotlib.patches import Circle, Rectangle, Arc

def draw_court(ax=None, color='black', lw=2, outer_lines=False):

    # If an axes object isn't provided to plot onto, just get current one

    if ax is None:

        ax = plt.gca()



    # Create the various parts of an NBA basketball court



    # Create the basketball hoop

    # Diameter of a hoop is 18" so it has a radius of 9", which is a value

    # 7.5 in our coordinate system

    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)



    # Create backboard

    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)



    # The paint

    # Create the outer box 0f the paint, width=16ft, height=19ft

    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,

                          fill=False)

    # Create the inner box of the paint, widt=12ft, height=19ft

    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,

                          fill=False)



    # Create free throw top arc

    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,

                         linewidth=lw, color=color, fill=False)

    # Create free throw bottom arc

    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,

                            linewidth=lw, color=color, linestyle='dashed')

    # Restricted Zone, it is an arc with 4ft radius from center of the hoop

    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,

                     color=color)



    # Three point line

    # Create the side 3pt lines, they are 14ft long before they begin to arc

    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,

                               color=color)

    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)

    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop

    # I just played around with the theta values until they lined up with the 

    # threes

    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,

                    color=color)



    # Center Court

    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,

                           linewidth=lw, color=color)

    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,

                           linewidth=lw, color=color)



    # List of the court elements to be plotted onto the axes

    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,

                      bottom_free_throw, restricted, corner_three_a,

                      corner_three_b, three_arc, center_outer_arc,

                      center_inner_arc]



    if outer_lines:

        # Draw the half court line, baseline and side out bound lines

        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,

                                color=color, fill=False)

        court_elements.append(outer_lines)



    # Add the court elements onto the axes

    for element in court_elements:

        ax.add_patch(element)



    return ax



import matplotlib as mpl

def Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages):

    

    fig, h = plt.subplots();

    for i, (mean, covarianceMatrix) in enumerate(zip(gaussianMixtureModel.means_, gaussianMixtureModel.covariances_)):

        # get the eigen vectors and eigen values of the covariance matrix

        v, w = np.linalg.eigh(covarianceMatrix)

        v = 2.5*np.sqrt(v) # go to units of standard deviation instead of variance

        

        # calculate the ellipse angle and two axis length and draw it

        u = w[0] / np.linalg.norm(w[0])    

        angle = np.arctan(u[1] / u[0])

        angle = 180 * angle / np.pi  # convert to degrees

        currEllipse = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=ellipseColors[i])

        currEllipse.set_alpha(0.5)

        h.add_artist(currEllipse)

        h.text(mean[0]+7, mean[1]-1, ellipseTextMessages[i], fontsize=13, color='blue')
from sklearn.mixture import GaussianMixture

numGaussians = 13

gaussianMixtureModel = GaussianMixture(n_components=numGaussians, covariance_type='full', 

                                               init_params='kmeans', n_init=50, 

                                               verbose=0, random_state=5)

gaussianMixtureModel.fit(data.loc[:,['loc_x','loc_y']])



# add the GMM cluster as a field in the dataset

data['shotLocationCluster'] = gaussianMixtureModel.predict(data.loc[:,['loc_x','loc_y']])
data
plt.rcParams['figure.figsize'] = (13, 10)

plt.rcParams['font.size'] = 15



variableCategories = data['shotLocationCluster'].value_counts().index.tolist()



ellipseColors = ['red','green','purple','cyan','magenta','yellow','blue','orange','silver','maroon','lime','olive','brown','darkblue']



clusterAccuracy = {}

for category in variableCategories:

    shotsAttempted = np.array(data['shotLocationCluster'] == category).sum()

    shotsMade = np.array(data.loc[data['shotLocationCluster'] == category,'shot_made_flag'] == 1).sum()

    clusterAccuracy[category] = float(shotsMade)/shotsAttempted



ellipseTextMessages = [str(100*clusterAccuracy[x])[:4]+'%' for x in range(numGaussians)]

Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages)

draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270); plt.title('shot accuracy')
plt.rcParams['figure.figsize'] = (13, 10)

plt.rcParams['font.size'] = 15



plt.figure(); draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270); plt.title('cluser assignment')

plt.scatter(x=data['loc_x'],y=data['loc_y'],c=data['shotLocationCluster'],s=40,cmap='hsv',alpha=0.1)
data.drop('loc_x', axis=1, inplace=True)

data.drop('loc_y', axis=1, inplace=True)

data.drop('dist', axis=1, inplace=True)
data.drop('shot_zone_area', axis=1, inplace=True)

data.drop('shot_zone_basic', axis=1, inplace=True)

data.drop('shot_zone_range', axis=1, inplace=True)
data['seconds_from_period_end'] = 60 * data['minutes_remaining'] + data['seconds_remaining']#距离每节结束的秒数

data['secondsFromPeriodStart'] =((data['period'] <= 4).astype(int))*60*(11-data['minutes_remaining'])+((data['period']>4).astype(int))*60*(4-data['minutes_remaining'])+(60-data['seconds_remaining'])#距离每节开始的秒数

data['secondsFromGameStart']=(data['period'] <= 4).astype(int)*(data['period']-1)*12*60+(data['period'] > 4).astype(int)*((data['period']-5)*5*60 + 4*12*60) + data['secondsFromPeriodStart']#距离每场比赛开始的秒数

#data[['period','minutes_remaining','seconds_remaining','secondsFromGameStart']] 加了三列（与比赛阶段的时间相关）
plt.rcParams['figure.figsize'] = (16, 12)

plt.rcParams['font.size'] = 16



binSizeInSeconds = 20#条形图的宽度

timeBins = np.arange(0,60*(4*12+3*5),binSizeInSeconds)+0.01#时间序列（x轴）--一场比赛+三个加时

shotAttemp, b = np.histogram(data['secondsFromGameStart'], bins=timeBins)#统计出手次数

shotFlag, b = np.histogram(data.loc[data['shot_made_flag']==1,'secondsFromGameStart'], bins=timeBins)#统计进球次数

shotAttemp[shotAttemp<1] = 1

shotAccuracy = shotFlag.astype(float)/shotAttemp#计算准确率

shotAccuracy[shotAttemp<=50] = 0#过滤掉样本不足的情况

#print(shotAccuracy)



maxHeight = max(shotAttemp) + 30

barWidth = 0.999*(timeBins[1]-timeBins[0])

#第一个图形

plt.figure()

plt.subplot(2,1,1)

plt.bar(timeBins[:-1], shotAttemp, align='edge', width=barWidth)

plt.xlim((-20, 3400))

plt.ylim((0, maxHeight))

plt.ylabel('attempt')

plt.title(str(binSizeInSeconds)+' second time bins')

plt.vlines(x=[0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60,4*12*60+2*5*60,4*12*60+3*5*60], ymin=0,ymax=maxHeight, colors='r')



#第二个图形

plt.subplot(2,1,2)

plt.bar(timeBins[:-1], shotAccuracy, align='edge', width=barWidth)

plt.xlim((-20, 3400))

plt.ylabel('accuracy')

plt.xlabel('time [seconds from start of game]')

plt.vlines(x=[0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60,4*12*60+2*5*60,4*12*60+3*5*60], ymin=0,ymax=0.8, colors='r')

plt.show()

#---------------每节最后几秒出手次数多，但是准确率较低---------------#
data.drop('minutes_remaining', axis=1, inplace=True)

data.drop('seconds_remaining', axis=1, inplace=True)

data.drop('secondsFromPeriodStart', axis=1, inplace=True)

data.drop('secondsFromGameStart', axis=1, inplace=True)

#data.drop('shotLocationCluster', axis=1, inplace=True)

data['last_24_sec_in_period'] = data['seconds_from_period_end'] < 24

data.drop('seconds_from_period_end', axis=1, inplace=True)
data["action_type"] = data["action_type"].astype('category')

data["action_type"].unique()
data["combined_shot_type"] = data["combined_shot_type"].astype('category')

data["combined_shot_type"].unique()
#data.drop('combined_shot_type', axis=1, inplace=True)

data.drop('action_type', axis=1, inplace=True)
target = alldata['shot_made_flag'].copy()#Y值

data.drop('shot_made_flag', axis=1, inplace=True)

#data.drop('game_month', axis=1, inplace=True)

#data.drop('game_year', axis=1, inplace=True)

#data.drop('opponent', axis=1, inplace=True)
data.dtypes
#data["period"] = data["period"].astype('category')

data["playoffs"] = data["playoffs"].astype('category')

data["shot_type"] = data["shot_type"].astype('category')

#data["game_year"] = data["game_year"].astype('object')

#data["game_month"] = data["game_month"].astype('object')

data["home_play"] = data["home_play"].astype('category')

data["shotLocationCluster"] = data["shotLocationCluster"].astype('category')

data["last_24_sec_in_period"] = data["last_24_sec_in_period"].astype('category')
data.dtypes
categorial_cols = [

    'combined_shot_type', 'shot_type', 'opponent'

    ]



for cc in categorial_cols:

    dummies = pd.get_dummies(data[cc])

    dummies = dummies.add_prefix("{}#".format(cc))

    data.drop(cc, axis=1, inplace=True)

    data = data.join(dummies)
data.columns
unknown_mask = alldata['shot_made_flag'].isnull()

unknown_mask
data_submit = data[unknown_mask]



# Separate dataset for training

X = data[~unknown_mask]

Y = target[~unknown_mask]
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X, Y)



feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])

feat_imp_3 = feature_imp.sort_values("importance", ascending=False).head(3).index

feat_imp_3
features = feat_imp_3

data = data.ix[:, features]

data_submit = data_submit.ix[:, features]

X = X.ix[:, features]



print('Clean dataset shape: {}'.format(data.shape))

print('Subbmitable dataset shape: {}'.format(data_submit.shape))

print('Train features shape: {}'.format(X.shape))

print('Target label shape: {}'. format(Y.shape))
from sklearn.model_selection import KFold

seed = 7

processors=1

num_folds=5

num_instances=len(X)

scoring='neg_log_loss'



kfold = KFold(n_splits=num_folds, random_state=seed)

kfold.get_n_splits(X)
from sklearn.model_selection import GridSearchCV



rf_grid = GridSearchCV(

    estimator = RandomForestClassifier(warm_start=True, random_state=seed),

    param_grid = {

        'n_estimators': [70],

        'criterion': ['gini', 'entropy'],

        'max_depth': [7],

        'bootstrap': [True]

    }, 

    #'max_features': [20, 30]

    cv = kfold, 

    scoring = scoring, 

    n_jobs = processors)



rf_grid.fit(X, Y)



print(rf_grid.best_score_)

print(rf_grid.best_params_)
from sklearn.linear_model import LogisticRegression

lr_grid = GridSearchCV(

    estimator = LogisticRegression(random_state=seed),

    param_grid = {

        'penalty': ['l1', 'l2'],

        'C': [0.001, 0.01, 1, 10, 100, 1000]

    }, 

    cv = kfold, 

    scoring = scoring, 

    n_jobs = processors)



lr_grid.fit(X, Y)



print(lr_grid.best_score_)

print(lr_grid.best_params_)
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score

estimators = []

estimators.append(('lr', LogisticRegression(penalty='l2', C=0.01)))

estimators.append(('rf', RandomForestClassifier(bootstrap=True, max_depth=7, n_estimators=70, max_features=3, criterion='entropy', random_state=seed)))

ensemble = VotingClassifier(estimators, voting='soft', weights=[2,3])

results = cross_val_score(ensemble, X, Y, cv=kfold, scoring=scoring,n_jobs=processors)

print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
model = ensemble



model.fit(X, Y)

preds = model.predict_proba(data_submit)



submission = pd.DataFrame()

submission["shot_id"] = data_submit.index+1

submission["shot_made_flag"]= preds[:,0]



submission.to_csv("sub.csv",index=False)