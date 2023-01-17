import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

import plotly.offline as py

#py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from mpl_toolkits.mplot3d import Axes3D

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
potential = pd.read_csv('../input/potential_energy.csv')

potential.head()
charge = pd.read_csv('../input/mulliken_charges.csv')

charge.head()
scalar = pd.read_csv('../input/scalar_coupling_contributions.csv')

scalar.head()
structures = pd.read_csv('../input/structures.csv')

structures.head()
magnetic = pd.read_csv('../input/magnetic_shielding_tensors.csv')

magnetic.head()
dipole = pd.read_csv('../input/dipole_moments.csv')

dipole.head()
%config InlineBackend.print_figure_kwargs = {'bbox_inches':None}

fig = plt.figure(figsize=(15,10))

N = 10000

ax = fig.add_subplot(111, projection='3d')

ax.scatter(dipole['X'].values[0:N],dipole['Y'].values[0:N],dipole['Z'].values[0:N],

           c=dipole['Z'].values[0:N],s=20,alpha=0.2)

ax.set_xlabel("X", fontsize=18)

ax.set_ylabel("Y", fontsize=18)

ax.set_zlabel("Z", fontsize=18)

ax.set_title("Dipole Moment of Molecules",fontsize=18)

plt.show()
f,ax = plt.subplots(1,3,figsize=(20,10))

n1 = sns.scatterplot(dipole['X'].values[0:N],dipole['Y'].values[0:N],hue=dipole['Z'].values[0:N],s=50,ax=ax[0])

n2 = sns.scatterplot(dipole['Y'].values[0:N],dipole['Z'].values[0:N],hue=dipole['Z'].values[0:N],s=50,ax=ax[1])

n3 = sns.scatterplot(dipole['Z'].values[0:N],dipole['X'].values[0:N],hue=dipole['Z'].values[0:N],s=50,ax=ax[2])

ax[0].set_xlabel("X", fontsize=18)

ax[1].set_xlabel("Y", fontsize=18)

ax[2].set_xlabel("Z", fontsize=18)

ax[0].set_ylabel("Y", fontsize=18)

ax[1].set_ylabel("Z", fontsize=18)

ax[2].set_ylabel("X", fontsize=18)

plt.show()
# Implementation with plotly

#N = 10000

#trace1 = go.Scatter3d(

#    x=dipole['X'].values[0:N],

#    y=dipole['Y'].values[0:N],

#    z=dipole['Z'].values[0:N],

#    mode='markers',

#    marker=dict(

#        size=5,

#        color=dipole['Z'],                # set color to an array/list of desired values

#        colorscale='Jet',   # choose a colorscale

#        opacity=0.2,

#        showscale=True,

#        line = dict(

#            width = 2,

#        )

#    )

#)



#data = [trace1]

#layout = go.Layout(

#    title = 'Dipole Moments',

#    scene = dict(

#        xaxis = dict(title="X"),

#        yaxis = dict(title="Y"),

#        zaxis = dict(title="Z"),

#    ),

#    margin=dict(

#        l=0,

#        r=0,

#        b=0,

#        t=0

#    ),

    

#)

#fig = go.Figure(data=data, layout=layout)

#py.iplot(fig, filename='Dipole_Moments')
%config InlineBackend.print_figure_kwargs = {'bbox_inches':None}

fig = plt.figure(figsize=(15,15))

N = 10000

ax = fig.add_subplot(111, projection='3d')

ax.scatter(structures['y'].values[0:N],structures['x'].values[0:N],structures['z'].values[0:N],

           c=structures['atom_index'].values[0:N],s=30,alpha=0.2,)

ax.set_xlabel("Y", fontsize=18)

ax.set_ylabel("X", fontsize=18)

ax.set_zlabel("Z", fontsize=18)

ax.set_title("Molecule Structures",fontsize=18)

plt.show()
f,ax = plt.subplots(1,3,figsize=(20,10))

n1 = sns.scatterplot(structures['x'].values[0:N],structures['y'].values[0:N],hue=structures['atom_index'].values[0:N],palette="Blues_r",linewidth=0,s=20,ax=ax[0])

n2 = sns.scatterplot(structures['z'].values[0:N],structures['y'].values[0:N],hue=structures['atom_index'].values[0:N],palette="Blues_r",linewidth=0,s=20,ax=ax[1])

n3 = sns.scatterplot(structures['x'].values[0:N],structures['z'].values[0:N],hue=structures['atom_index'].values[0:N],palette="Blues_r",linewidth=0,s=20,ax=ax[2])

ax[0].set_xlabel("X", fontsize=18)

ax[1].set_xlabel("Z", fontsize=18)

ax[2].set_xlabel("X", fontsize=18)

ax[0].set_ylabel("Y", fontsize=18)

ax[1].set_ylabel("Y", fontsize=18)

ax[2].set_ylabel("Z", fontsize=18)

plt.show()
# Implementation with Plotly

#N = 10000

#trace2 = go.Scatter3d(

#    x=structures['x'].values[0:N],

#    y=structures['y'].values[0:N],

#    z=structures['z'].values[0:N],

#    mode='markers',

#    marker=dict(

#        size=5,

#        color=structures['atom_index'],                # set color to an array/list of desired values

#        colorscale='Jet',   # choose a colorscale

#        opacity=0.2,

#        showscale=True,

#        line = dict(

#            width = 2,

#        )      

#    )

#)



#data = [trace2]

#layout = go.Layout(

#    title = 'Molecule Structures',

#    scene = dict(

#        xaxis = dict(title="X"),

#        yaxis = dict(title="Y"),

#        zaxis = dict(title="Z"),

#    ),

#    margin=dict(

#        l=0,

#        r=0,

#        b=0,

#        t=0

#    ),

    

#)

#fig = go.Figure(data=data, layout=layout)

#py.iplot(fig, filename='Molecule_Structures')
f,ax = plt.subplots(figsize=(20,15))

sns.heatmap(scalar[['fc','sd','pso','dso']].corr(), ax=ax,cmap="YlGnBu")

plt.title("Correlation Matrix",fontsize=20)

plt.show()
f = plt.figure()

sns.pairplot(scalar[['fc','sd','pso','dso']])

#plt.title("Pair plot",fontsize=20)

plt.show()
f = plt.figure(figsize=(20,15))

sns.set_color_codes("bright")

sns.barplot(scalar['dso'],scalar['type'],label="Diamagnetic spin-orbit", color="b")

sns.set_color_codes("pastel")

sns.barplot(scalar['pso'],scalar['type'],label="Paramagnetic spin-orbit", color="b")

sns.set_color_codes("muted")

sns.barplot(scalar['sd'],scalar['type'],label="Spin-dipolar", color="b")

plt.title("Bar plot",fontsize=20)

plt.legend(loc = 'lower right',fontsize=20)

plt.show()
f = plt.figure(figsize=(20,15))



sns.jointplot(charge['atom_index'],charge['mulliken_charge'],kind="kde", color="#4CB391")

plt.show()
# Implementation with Plotly

#N = 10000

#data = [

#    go.Surface(

#        z=magnetic.values[:N]

#    )

#]

#layout = go.Layout(

#    title='magnetic shielding tensors',

#    autosize=False,

#    scene=dict(camera=dict(eye=dict(x=1.87, y=0.88, z=-0.64))),

#    margin=dict(

#        l=0,

#        r=0,

#        b=0,

#        t=0

#    )

#)

#fig = go.Figure(data=data, layout=layout)

#py.iplot(fig, filename='elevations-3d-surface')
train = pd.read_csv('../input/train.csv')

train.head()
k = train.type.astype('category').cat.codes
N = 10000

plt.figure(figsize=(15,15))

plt.scatter(scalar['sd'][:N],train['scalar_coupling_constant'][:N], c='b',edgecolor='black', marker='o',alpha=0.5, label='Spin-dipolar')

plt.scatter(scalar['pso'][:N],train['scalar_coupling_constant'][:N], c='red',edgecolor='w', marker='o',alpha=0.5, label='Paramagnetic spin-orbit')

plt.scatter(scalar['dso'][:N],train['scalar_coupling_constant'][:N], c='y',edgecolor='b', marker='o',alpha=0.5, label='Diamagnetic spin-orbit')

plt.legend(loc='upper left')

plt.show()
# Implementation with Plotly

#N = 10000

#trace0 = go.Scatter(

#    y = train['scalar_coupling_constant'][:N],

#    x = scalar['sd'][:N],

#    mode = 'markers',

#    name = 'Spin-dipolar',

#    marker = dict(

#        opacity=0.2,

        

#        line = dict(

#            width = 0.5,

#        )

#)

#)

#trace1 = go.Scatter(

#    y = train['scalar_coupling_constant'][:N],

#    x = scalar['pso'][:N],

#    mode = 'markers',

#    name = 'Paramagnetic spin-orbit ',

#    marker = dict(

#        opacity=0.2,

        

#        line = dict(

#            width = 0.5,

#        )

#)

#)

#trace2 = go.Scatter(

#    y = train['scalar_coupling_constant'][:N],

#    x = scalar['dso'][:N],

#    mode = 'markers',

#    name = 'Diamagnetic spin-orbit',

#    marker = dict(

#        opacity=0.2,

        

#        line = dict(

#            width = 0.5,

#        )

#)

#)

#data = [trace1, trace2, trace0]



#py.iplot(data, filename='line-mode')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
test.head()
#train = pd.merge(train, scalar, how = 'left',

#                  left_on  = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],

#                  right_on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

#train.head()
#test = pd.merge(test, scalar, how = 'left')

                  #left_on  = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],

                  #right_on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

#test.head()
def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df
def make_features(df):

    df['dx']=df['x_1']-df['x_0']

    df['dy']=df['y_1']-df['y_0']

    df['dz']=df['z_1']-df['z_0']

    df['distance']=(df['dx']**2+df['dy']**2+df['dz']**2)**(1/2)

    return df
train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)
train_=make_features(train)

test_=make_features(test) 
train_.drop("molecule_name", axis=1, inplace=True)

test_.drop("molecule_name", axis=1, inplace=True)
test_id = test_['id']

train_.drop("id", axis=1, inplace=True)

test_.drop("id", axis=1, inplace=True)
train_type = pd.get_dummies(train_['type'])

test_type = pd.get_dummies(test_['type'])
train_new = pd.concat([train_, train_type], axis=1)

train_new.drop("type", axis=1, inplace=True)

test_new = pd.concat([test_, test_type], axis=1)

test_new.drop("type", axis=1, inplace=True)
train_new.head()
train_new['atom_0'] = train_new['atom_0'].astype("category").cat.codes

train_new['atom_1'] = train_new['atom_1'].astype("category").cat.codes

test_new['atom_0'] = test_new['atom_0'].astype("category").cat.codes

test_new['atom_1'] = test_new['atom_1'].astype("category").cat.codes
from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score

from sklearn.neighbors import KNeighborsRegressor

from catboost import CatBoostRegressor
y = train_new["scalar_coupling_constant"]

train_new.drop("scalar_coupling_constant", axis=1, inplace=True)

X = train_new
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
model_KNN = KNeighborsRegressor(n_neighbors=75, algorithm='auto', n_jobs=-1,p=2)

model_KNN.fit(x_train, y_train)
y_pred = model_KNN.predict(x_test)
print("Variance_Score(KNN_Regressor)\t:"+str(explained_variance_score(y_test,y_pred)))

print("Mean_Absolute_Error(KNN_Regressor)\t:"+str(mean_absolute_error(y_test,y_pred)))

print("Mean_Squared_Error(KNN_Regressor)\t:"+str(mean_squared_error(y_test,y_pred)))

print("R2-Score(KNN_Regressor)\t:"+str(r2_score(y_test,y_pred)))
model_cat = CatBoostRegressor(iterations=3000,depth= 13,random_seed = 23,

                           task_type = "GPU")

model_cat.fit(x_train, y_train)
y_pred_cat = model_cat.predict(x_test)
print("Variance_Score(cat_Regressor)\t:"+str(explained_variance_score(y_test,y_pred_cat)))

print("Mean_Absolute_Error(cat_Regressor)\t:"+str(mean_absolute_error(y_test,y_pred_cat)))

print("Mean_Squared_Error(cat_Regressor)\t:"+str(mean_squared_error(y_test,y_pred_cat)))

print("R2-Score(cat_Regressor)\t:"+str(r2_score(y_test,y_pred_cat)))
y_pred_test = model_cat.predict(test_new)
my_submission = pd.DataFrame({'id': test_id, 'scalar_coupling_constant': y_pred_test})

my_submission.to_csv('SubmissionVictor2.csv', index=False)