import numpy as np 

import pandas as pd 

from tqdm import tqdm

import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import seaborn as sns  

import statsmodels.api as sm



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

sample_submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
Y_train = train[['target']]

X_train = train.drop(['target'], axis=1)



X_train_par = pd.DataFrame()

X_test_par = pd.DataFrame()



full_data = [X_train, test]

dat_frame = [X_train_par, X_test_par]
X_train.head()
for data in range(len(full_data)):

    

    #for i in range(full_data[data].shape[0]):

    for i in tqdm(range(full_data[data].shape[0])):

        

        x = full_data[data].iloc[i, 1:201]



        dat_frame[data].loc[i, 'mean'] = x.mean()

        dat_frame[data].loc[i, 'des'] = np.var(x)

        dat_frame[data].loc[i, 'std'] = x.std()

        dat_frame[data].loc[i, 'max'] = x.max()

        dat_frame[data].loc[i, 'min'] = x.min()



        dat_frame[data].loc[i, 'quan0.25'] = np.quantile(x, 0.25)

        dat_frame[data].loc[i, 'quan0.5'] = np.quantile(x, 0.5)

        dat_frame[data].loc[i, 'quan0.75'] = np.quantile(x, 0.75)
X_train_par.head(3)
X_test_par.head(3)
X_train_par['target'] = Y_train.target
X_train_par_1 = X_train_par[X_train_par['target']==1].iloc[0:10000,:]

X_train_par_0 = X_train_par[X_train_par['target']==0].iloc[0:10000,:]
X_train_box = X_train_par.iloc[0:10000,:]

x = list(X_train_box.target)
X_train_par['target'].hist()
fig = go.Figure()



fig.add_trace(go.Box(

    y=list(X_train_box['quan0.25']),

    x=x,

    name='quan0.25',

    marker_color='#3D9970'

))



fig.update_layout(

    yaxis_title='quantile 1/4',

    boxmode='group' # group together boxes of the different traces for each value of x

)



fig.show()
fig = go.Figure()



fig.add_trace(go.Box(

    y=list(X_train_box['quan0.5']),

    x=x,

    name='quan0.5',

    marker_color='#FF4136'

))



fig.update_layout(

    yaxis_title='quantile 1/2',

    boxmode='group' # group together boxes of the different traces for each value of x

)

fig.show()
fig = go.Figure()



fig.add_trace(go.Box(

    y=list(X_train_box['quan0.75']),

    x=x,

    name='radishes',

    marker_color='#000080'

))



fig.update_layout(

    yaxis_title='quantile 3/4',

    boxmode='group' # group together boxes of the different traces for each value of x

)

fig.show()
# X_train_par_1

plt.figure(figsize=[17,4])



plt.subplot(131)

sns.distplot(X_train_par_1['mean'])

plt.subplot(132)

sns.distplot(X_train_par_1['des'])

plt.subplot(133)

sns.distplot(X_train_par_1['std'])
# X_train_par_0 

plt.figure(figsize=[17,4])



plt.subplot(131)

sns.distplot(X_train_par_0['mean'])

plt.subplot(132)

sns.distplot(X_train_par_0['des'])

plt.subplot(133)

sns.distplot(X_train_par_0['std'])
fi_m = sm.qqplot(X_train_box['mean'], fit=True, line='45')

fi_d = sm.qqplot(X_train_box['des'], fit=True, line='45')

fi_s = sm.qqplot(X_train_box['std'], fit=True, line='45')
X_train_par = X_train_par.drop(['target'], axis=1)
rf = RandomForestClassifier()
random_forest = GridSearchCV(estimator=rf, param_grid={'n_estimators': [100, 300]}, cv=5)

random_forest.fit(X_train_par,Y_train)

best_random_forest = random_forest.best_estimator_

best_random_forest.fit(X_train_par, Y_train)
y_predicted = best_random_forest.predict(X_test_par)  

y_predicted = list(y_predicted)
finall_F = pd.DataFrame.from_dict({'ID_code': list(test.ID_code), 'target': y_predicted})
finall_F.to_csv("Submission.csv", index=False)