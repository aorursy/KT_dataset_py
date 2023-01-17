# The classics
import numpy as np
import pandas as pd

# Visualisation tools
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style

import plotly
from plotly.offline import iplot, init_notebook_mode
from plotly.graph_objs import Scatter3d, Layout, Figure

import graphviz 

# Machine learning unavoidables
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import normalize,MinMaxScaler,PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2

# Defining the name of each column as it was given in the dataset
col_list = ['Pelvic_incidence',
               'Pelvic_tilt',
               'Lumbar_lordosis_angle',
               'Sacral_slope',
               'Pelvic_radius',
               'Degree_spondylolisthesis',
               'Pelvic_slope',
               'Direct_tilt',
               'Thoracic_slope',
               'Cervical_tilt',
               'Sacrum_angle',
               'Scoliosis_slope',
               'Attribute',
               'To_drop']

# Loading the data
data = pd.read_csv("../input/Dataset_spine.csv", names=col_list, header=1)

# The last column contained meta-data about the other columns and is irrelevant in our study
data.drop('To_drop', axis=1, inplace=True)


data.head()
# Checking for the integrity of the data is good practice
data.info()
sns.set_style("white")
g=sns.factorplot(x='Attribute', hue='Attribute', data= data, kind='count',size=5,aspect=.8)
# Replacing our attribute with binary values : 
data['Attribute'] = data['Attribute'].map({'Abnormal': 1, 'Normal': 0})
sns.set(style="white")
d = data
corr = d.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
g=sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.pairplot(data, hue="Attribute")
plt.show()
# Creating the arrays
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# Creating the shallow decision tree
clf = DecisionTreeClassifier(max_depth=3)

# Fitting the decision tree to the data set
clf = clf.fit(X, y)

# Plotting the results
dot = tree.export_graphviz(clf,out_file=None,
                         feature_names=col_list[:-2],  
                         class_names=['Normal','Abnormal'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot) 
graph.render("back problems")
graph
# A simple reusable function to plot the distribution of one feature with different colours for each class
def hist_graph(column):
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.distplot(data[data['Attribute']==1][column], color='r')
    sns.distplot(data[data['Attribute']==0][column],ax=ax, color='b') 

hist_graph('Degree_spondylolisthesis')
# A simple and reusable function to show the numbers in a dataframe
def compare_df(column):
    norm = data[data['Attribute']==0][[column]].describe()
    abnorm = data[data['Attribute']==1][[column]].describe()

    df = pd.DataFrame(data = norm)
    df['Normal'] = df[column]
    df.drop(column, axis=1, inplace=True)
    df['Abnormal']= abnorm
    return df

compare_df('Degree_spondylolisthesis')
hist_graph('Sacral_slope')
compare_df('Sacral_slope')
hist_graph('Pelvic_incidence')
compare_df('Pelvic_incidence')
hist_graph('Pelvic_tilt')
compare_df('Pelvic_tilt')
hist_graph('Pelvic_radius')
compare_df('Pelvic_radius')
hist_graph('Lumbar_lordosis_angle')
compare_df('Lumbar_lordosis_angle')
# Feature scaling
sc = MinMaxScaler()
X_std = sc.fit_transform(X)

# Creating the PCA
pca = PCA(n_components=3)

# Fitting the PCA to the data set
pca.fit(X_std)
pca.explained_variance_ratio_
PF = PolynomialFeatures(degree=2, include_bias=False)
X_std_pf= PF.fit_transform(X_std)
new_feats = PF.get_feature_names()
X_std_pf.shape
Kbest =  SelectKBest(chi2, k=10)
X_std_pf_chi10 = Kbest.fit_transform(X_std_pf, y)
selected = Kbest.get_support()
features=[]
for feat, sel in zip(new_feats, selected) : 
    if sel == True :
        features.append(feat)

feat_col=[]
for i in features :
    split = i.split()
    if len(split)==1 :
        pow = split[0].split('^')
        if len(pow) == 1:
            nb =int(''.join([j for j in pow[0] if j.isdigit()]))
            col=data.columns[nb]
            feat_col.append(col)
        else :
            nb =int(''.join([j for j in pow[0] if j.isdigit()]))
            col=data.columns[nb]+'^'+pow[1]
            feat_col.append(col)
    else:
        clean =''.join([j for j in i if j.isdigit()])
        col=data.columns[int(clean[0])]+'*'+data.columns[int(clean[1])]
        feat_col.append(col)
feat_col
box_deg = (data.Degree_spondylolisthesis > data[data['Attribute']==0].Degree_spondylolisthesis.max()).map({False: 0, True: 1})
box_ss  = ((data.Sacral_slope > data[data['Attribute']==0].Sacral_slope.max()) & (data.Sacral_slope > data[data['Attribute']==0].Sacral_slope.min())).map({False: 0, True: 1})
box_pi  = (data.Pelvic_incidence > data[data['Attribute']==0].Pelvic_incidence.max()).map({False: 0, True: 1})
box_pt  = ((data.Pelvic_tilt > data[data['Attribute']==0].Pelvic_tilt.max()) & (data.Pelvic_tilt > data[data['Attribute']==0].Pelvic_tilt.min())).map({False: 0, True: 1})
box_pr  = ((data.Pelvic_radius > data[data['Attribute']==0].Pelvic_radius.max()) & (data.Pelvic_radius > data[data['Attribute']==0].Pelvic_radius.min())).map({False: 0, True: 1})
box_lla = ((data.Lumbar_lordosis_angle > data[data['Attribute']==0].Lumbar_lordosis_angle.max()) & (data.Lumbar_lordosis_angle > data[data['Attribute']==0].Lumbar_lordosis_angle.min())).map({False: 0, True: 1})
X_box = np.array([box_deg,box_ss,box_pi,box_pt,box_pr,box_lla]).reshape(309,6)
# Adding the boxing features to the other ones
X_std_box = np.hstack([X_std,X_box])
X_std_pf_chi10_box = np.hstack([X_std_pf_chi10,X_box])
# Creating a list through which we'll iterate the Decision Tree
X_list=[X_std, X_std_pf, X_std_pf_chi10,X_std_box, X_std_pf_chi10_box]
results = []

clf = DecisionTreeClassifier(random_state=42)

# Getting cross-validated scores for each of the new data sets
for X_set in X_list :
    clf.fit(X_set,y)
    y_pred = clf.predict(X_set)
    rez = cross_val_score(clf, X_set, y, scoring='roc_auc', cv=100 )
    results.append(rez.mean())
results
df_new = pd.DataFrame(X_std_pf_chi10, columns=feat_col)
df_new['Attribute'] = data['Attribute']
df_new.head()
param_grid = {'max_depth': np.arange(1, 10),
             'min_samples_leaf' : np.arange(1, 10),
             'max_features' : ['auto','sqrt','log2',None],
             'random_state' : [37,]}

trees = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring='roc_auc')
trees.fit(X_std_pf_chi10, y)

print("The best parameters are : ", trees.best_params_,' giving an AUC : ', trees.best_score_)
clf_tree = trees.best_estimator_

dot = tree.export_graphviz(clf_tree,out_file=None,
                         feature_names=df_new.columns[:-1],  
                         class_names=['Normal','Abnormal'], 
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot) 
graph.render("back problems2")
graph
clf_tree.feature_importances_

param_grid = {'penalty': ['l1','l2'],
             'tol'     : [1e-5, 1e-4, 1e-3, 1e-2],
             'C'        : [1.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0] ,
             'solver'    : ['liblinear',  'saga'],
             'random_state' : [37,],
             'max_iter' : [700,]}

logit = GridSearchCV(LogisticRegression(), param_grid, scoring='roc_auc',verbose=0)
logit.fit(X_std_pf_chi10, y)
print("The best parameters are : ", logit.best_params_,' giving an AUC : ', logit.best_score_)
clf = logit.best_estimator_
clf.coef_
init_notebook_mode(connected=True)

X = df_new.iloc[:,:-1]
y = df_new.iloc[:,-1]

pca = PCA(n_components = 3)
X_PCA = pca.fit_transform(X)
    
xs = X_PCA[:,0]
ys = X_PCA[:,1]
zs = X_PCA[:,2]

# Recreating the df with the new coordinates
df = pd.DataFrame(dict(x=xs, y=ys, z=zs, Attribute=y)) 
l = []
names = ['Normal','Abnormal']

for i in [0,1]:    
    trace= Scatter3d(
        x= df[df['Attribute']==i]['x'],
        y= df[df['Attribute']==i]['y'],
        z= df[df['Attribute']==i]['z'],
        mode= 'markers',
        marker= dict(size= 5,
                    line= dict(width=1),
                    color= i,
                    colorscale='Jet',
                    opacity= 0.8
                   ),#name= y[i],
        name = names[i],
        text= df[df['Attribute']==i].index,# The hover text goes here...
        hoverinfo = 'text+name'
    )

    l.append(trace)

layout= Layout(
    title= '3D Representation of the patients characteristics using PCA',
    hovermode= 'closest',
    showlegend= True)

fig= Figure(data=l, layout=layout)
plotly.offline.iplot(fig)
print('The PCA explains the variance of the data by {:3f}%'.format(100*pca.explained_variance_ratio_.sum()))
from sklearn.manifold import TSNE
tsne = TSNE(n_components=3,learning_rate=115.0)
X_tsne = tsne.fit_transform(X, y)


xs = X_tsne[:,0]
ys = X_tsne[:,1]
zs = X_tsne[:,2]

# Recreating the df with the new coordinates
df = pd.DataFrame(dict(x=xs, y=ys, z=zs, Attribute=y)) 
l = []
names = ['Normal','Abnormal']

for i in [0,1]:    
    trace= Scatter3d(
        x= df[df['Attribute']==i]['x'],
        y= df[df['Attribute']==i]['y'],
        z= df[df['Attribute']==i]['z'],
        mode= 'markers',
        marker= dict(size= 5,
                    line= dict(width=1),
                    color= i,
                    colorscale='Jet',
                    opacity= 0.8
                   ),#name= y[i],
        name = names[i],
        text= df[df['Attribute']==i].index,# The hover text goes here...
        hoverinfo = 'text+name'
    )

    l.append(trace)

layout= Layout(
    title= '3D Representation of the patients characteristics using TSNE',
    hovermode= 'closest',
    showlegend= True)

fig= Figure(data=l, layout=layout)
plotly.offline.iplot(fig)

from sklearn.manifold import SpectralEmbedding

SE = SpectralEmbedding(n_components=3)
X_SE = SE.fit_transform(X, y)


xs = X_SE[:,0]
ys = X_SE[:,1]
zs = X_SE[:,2]

# Recreating the df with the new coordinates
df = pd.DataFrame(dict(x=xs, y=ys, z=zs, Attribute=y)) 
l = []
names = ['Normal','Abnormal']

for i in [0,1]:    
    trace= Scatter3d(
        x= df[df['Attribute']==i]['x'],
        y= df[df['Attribute']==i]['y'],
        z= df[df['Attribute']==i]['z'],
        mode= 'markers',
        marker= dict(size= 5,
                    line= dict(width=1),
                    color= i,
                    colorscale='Jet',
                    opacity= 0.8
                   ),#name= y[i],
        name = names[i],
        text= df[df['Attribute']==i].index,# The hover text goes here...
        hoverinfo = 'text+name'
    )

    l.append(trace)

layout= Layout(
    title= '3D Representation of the patients characteristics using Spectral Embedding',
    hovermode= 'closest',
    showlegend= True)

fig= Figure(data=l, layout=layout)
plotly.offline.iplot(fig)
