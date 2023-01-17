

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb # classifiers using boosted trees



from bokeh.plotting import figure, show

from bokeh.layouts import gridplot

from bokeh.io import output_notebook

from bokeh.models import ColumnDataSource, ColorBar

from bokeh.transform import linear_cmap

from bokeh.palettes import viridis
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train['SexNumerical'] = train.apply(lambda x: 0.0 if x['Sex']=='male' else 1.0, axis = 1)

test['SexNumerical'] = test.apply(lambda x: 0.0 if x['Sex']=='male' else 1.0, axis = 1)

mask = [False if np.isnan(xi) == True else True for xi in train['Age'].values.tolist()]

train = train[mask]

train_male = train[train['Sex']=='male']

train_female = train[train['Sex']=='female']

train_male_age_1 = train_male[train_male['Survived']==1]['Age'].values

train_male_age_0 = train_male[train_male['Survived']==0]['Age'].values

train_female_age_1 = train_female[train_female['Survived']==1]['Age'].values

train_female_age_0 = train_female[train_female['Survived']==0]['Age'].values



hist_male_1, edges_male_1 = np.histogram(train_male_age_1, density = False, bins = range(0,80,5))

hist_male_0, edges_male_0 = np.histogram(train_male_age_0, density = False, bins = range(0,80,5))

hist_female_1, edges_female_1 = np.histogram(train_female_age_1, density = False, bins = range(0,80,5))

hist_female_0, edges_female_0 = np.histogram(train_female_age_0, density = False, bins = range(0,80,5))

xm = np.linspace(min(min(train_male_age_1), min(train_male_age_0)), max(max(train_male_age_0), max(train_male_age_1)), 100)

xm = np.linspace(min(min(train_female_age_1), min(train_female_age_0)), max(max(train_female_age_0), max(train_female_age_1)), 100)
male_survivability = len(train_male_age_1)/len(train_male)

female_survivability = len(train_female_age_1)/len(train_female)

print('Survivability Rates')

print('Males: ' + str(np.round(male_survivability*100., decimals=2)) + '%, Females: '+ str(np.round(female_survivability*100., decimals=2)) + '%')
output_notebook()

pm = figure(title = 'Male Survivability with Age', x_range = (0, 85), y_range = (0, 80))

pm.quad(top = hist_male_1, bottom = 0, left = edges_male_1[:-1], right = edges_male_1[1:], fill_color = 'navy', alpha = 0.5, legend = 'Survived')

pm.quad(top = hist_male_0, bottom = 0, left = edges_male_0[:-1], right = edges_male_0[1:], fill_color = 'red', alpha = 0.5, legend = 'Perished')

pm.legend.location = 'center_right'

pm.xaxis.axis_label = 'Age'

pm.yaxis.axis_label = 'Individuals'

pf = figure(title = 'Female Survivability with Age', x_range = (0, 85), y_range = (0, 80))

pf.quad(top = hist_female_1, bottom = 0, left = edges_female_1[:-1], right = edges_female_1[1:], fill_color = 'navy', alpha = 0.5, legend = 'Survived')

pf.quad(top = hist_female_0, bottom = 0, left = edges_female_0[:-1], right = edges_female_0[1:], fill_color = 'red', alpha = 0.5, legend = 'Perished')

pf.legend.location = 'center_right'

pf.xaxis.axis_label = 'Age'

pf.yaxis.axis_label = 'Individuals'

show(gridplot([pm,pf], ncols = 2, plot_width = 400, plot_height = 400, toolbar_location = None))
train_fare_1 = train[train['Survived']==1]['Fare'].dropna()

train_fare_0 = train[train['Survived']==0]['Fare'].dropna()

hist_fare_1, edges_fare_1 = np.histogram(train_fare_1, density = False, bins = range(0,300,10))

hist_fare_0, edges_fare_0 = np.histogram(train_fare_0, density = False, bins = range(0,300,10))



classes = sorted(train['Pclass'].unique())

class_df = pd.DataFrame(index = classes, columns = ['Survivability'])

for c in classes:

    c_tot = train[train['Pclass']==c]

    c_1 = c_tot[c_tot['Survived']==1]

    class_df.at[c, 'Survivability'] = float(len(c_1))/float(len(c_tot))*100
output_notebook()

p1 = figure(title = 'Survivability with Fare Amount', x_range = (0, 300), y_range = (0, 200))

p1.quad(top = hist_fare_1, bottom = 0, left = edges_fare_1[:-1], right = edges_fare_1[1:], fill_color = 'navy', alpha = 0.5, legend = 'Survived')

p1.quad(top = hist_fare_0, bottom = 0, left = edges_fare_0[:-1], right = edges_fare_0[1:], fill_color = 'red', alpha = 0.5, legend = 'Perished')

p1.xaxis.axis_label = 'Fare Amount'

p1.yaxis.axis_label = 'Individuals'

p1.legend.location = 'center_right'

cats = ['Class ' + str(x) for x in class_df.index.values.tolist()]



mapper = linear_cmap(field_name='counts', palette=viridis(256) ,low=0. ,high=100.)

color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0))



source = ColumnDataSource(data = dict(cats=cats, counts = class_df['Survivability']))

p2 = figure(title = 'Survivability Rate and Fare Class', x_range = cats, y_range = (0, 100))

p2.vbar(x = 'cats', top = 'counts', color = mapper, width = 0.9, source = source)

p2.yaxis.axis_label = 'Chance of Survival (%)'

p2.add_layout(color_bar, 'right')

show(gridplot([p1,p2], ncols = 2, plot_width = 400, plot_height = 400, toolbar_location = None))



train['Family Size']= train['SibSp']+train['Parch']+1

train['Family Size']= train.apply(lambda x: str(x['Family Size']), axis = 1)

test['Family Size']= test['SibSp']+train['Parch']+1

test['Family Size']= test.apply(lambda x: str(x['Family Size']), axis = 1)

train_fs_1 = train[train['Survived'] == 1][['PassengerId','Family Size']].groupby('Family Size').count().reset_index()

train_fs_totals = train[['PassengerId', 'Family Size']].groupby('Family Size').count().reset_index()

train_fs_1 = pd.merge(train_fs_1, train_fs_totals, on = 'Family Size')

train_fs_1['Chance'] = train_fs_1['PassengerId_x']/train_fs_1['PassengerId_y']*100.

source = ColumnDataSource(train_fs_1)



mapper = linear_cmap(field_name='Chance', palette=viridis(256) ,low=0 ,high=100.)

color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0))



output_notebook()

p = figure(title = 'Survivability with Family Size', x_range = train_fs_1['Family Size'], y_range = (0, 100))

p.vbar(x = 'Family Size', top = 'Chance', color = mapper, width = 0.9, source = source)

p.xaxis.axis_label = 'Family Size'

p.yaxis.axis_label = 'Chance of Survival (%)'

p.add_layout(color_bar, 'right')

show(gridplot([p], ncols = 1, plot_width = 400, plot_height = 400, toolbar_location = None))

def train_test_split(X,y, perc):

    '''Performs a simple random sample of the training data returning a training and testing set'''

    trainx = X.sample(frac=perc, replace = False, random_state = 0)

    sel = trainx.index.values.tolist()

    trainy = y.loc[sel]

    notsel = []

    for i in X.index.values.tolist():

        if i not in sel:

            notsel.append(i)

    testx = X.loc[notsel]

    testy = y.loc[notsel]

    return trainx, trainy, testx, testy


trainx = train[['Pclass','Fare','SexNumerical','Age','Family Size']].copy()

trainx['Family Size'] = trainx.apply(lambda x: float(x['Family Size']), axis=1)

trainy = train['Survived']

trainx, trainy, testx, testy = train_test_split(trainx, trainy, 0.8)

param = {'max_depth':5, 'eta': 0.5, 'silent': 1, 'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric':'error'}

num_round = 10

Dtrain = xgb.DMatrix(trainx, label = trainy)

Dtest = xgb.DMatrix(testx, label = testy)

watchlist = [(Dtest, 'eval'), (Dtrain, 'train')]
bst = xgb.train(param, Dtrain, num_round, watchlist)
ypred = bst.predict(Dtest)

labels = Dtest.get_label()

print('error=%f' % (sum(1 for i in range(len(ypred)) if int(ypred[i] > 0.5) != labels[i]) / float(len(ypred))))
trainx = train[['Pclass','Fare','SexNumerical','Age','Family Size']].copy()

trainx['Family Size'] = trainx.apply(lambda x: float(x['Family Size']), axis=1)

trainy = train['Survived']

Dtrain = xgb.DMatrix(trainx, label = trainy )

bst = xgb.train(param, Dtrain, num_round, watchlist)
xeval = test[['Pclass','Fare','SexNumerical','Age','Family Size']].copy()

xeval['Age']=xeval['Age'].fillna(np.nanmean(xeval['Age']))

xeval['Family Size']=xeval.apply(lambda x: float(x['Family Size']), axis = 1)

xeval['Family Size']=xeval['Family Size'].fillna(np.nanmean(xeval['Family Size']))

xeval['Fare']=xeval['Fare'].fillna(np.nanmean(xeval['Fare']))

Deval = xgb.DMatrix(xeval)

ypred_eval = bst.predict(Deval)

ypred_eval_out = np.hstack([test['PassengerId'].values.reshape(-1,1), ypred_eval.reshape(-1,1)])

ypred_eval_out = pd.DataFrame(ypred_eval_out, columns=['PassengerId', 'Score'])

ypred_eval_out['Survived']=ypred_eval_out.apply(lambda x: 1 if x['Score']>=0.5 else 0, axis =1)

ypred_eval_out['PassengerId']=ypred_eval_out.apply(lambda x: int(x['PassengerId']), axis =1 )

ypred_eval_out = ypred_eval_out[['PassengerId', 'Survived']].set_index('PassengerId')

ypred_eval_out.to_csv('output.csv')
ypred_eval.shape
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



pipe = Pipeline(steps = [('minmax',MinMaxScaler()), ('svc',SVC(C=1., kernel = 'rbf', gamma = 'auto'))])



params = {'svc__C':[0.1,1.0,10.,100., 1000.]}



clf = GridSearchCV(pipe, params, cv=4).fit(trainx, trainy)

print(clf.best_score_, clf.best_params_)

svm_predictions = pd.DataFrame(columns = ['PassengerId', 'Survived'], data = np.hstack([test['PassengerId'].values.reshape(-1,1), clf.predict(xeval).reshape(-1,1)])).set_index('PassengerId')

svm_predictions.to_csv('outputsvm.csv')