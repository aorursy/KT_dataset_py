!pip install pynndescent
# importing necessary modules

import numpy as np

import pandas as pd



# loading clean data (easy to get from original notebook)

df = pd.read_csv('../input/clean-2018-kaggle-survey-data/clean_dataset_v2.csv').drop('Unnamed: 0', axis=1)



# getting non-US residents, males and working in finance

people_in_my_context = (df

                        [(df['Q3-United States of America'] == 0) & # non-US residents

                         (df['Q1-Male'] == 1) & # males

                         (df['Q7-Accounting/Finance'] == 1)]) # working in finance



# getting average likelihood of being in top 20% for this group

likelihood_top20 = people_in_my_context.groupby('Q6-Software Engineer')['top20'].mean()



# importing plotly to show in a bar graph

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



# trace storing data for bar graph

trace = go.Bar(x=['Software Engineer', 'NOT Software Engineer'],

               y=likelihood_top20.values[::-1] * 100,

               text = ((likelihood_top20 * 100).round(2).astype(str) + '%').values[::-1],

               textposition='auto',

               textfont=dict(size=14),

               width=[0.5,0.5])



# layout of plot

layout = go.Layout(title='Likelihood of being on Top 20%: non-US, Male, and working in Finance',

                   xaxis=dict(tickfont=dict(size=16)), width=800, height=400)



# building figure and plotting

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
# checking effect of US

US_effect = df.groupby(['Q3-United States of America','Q6-Software Engineer'])['top20'].mean().reset_index()



# trace storing data for bar graph

trace1 = go.Bar(x=['NOT US Resident', 'US Resident'],

                y=US_effect[US_effect.iloc[:,1] == 0]['top20'] * 100,

                text = ((US_effect[US_effect.iloc[:,1] == 0]['top20'] * 100).round(2).astype(str) + '%'),

                textposition='auto',

                textfont=dict(size=14),

                name = 'NOT Software Engineer')



# trace storing data for bar graph

trace2 = go.Bar(x=['NOT US Resident', 'US Resident'],

                y=US_effect[US_effect.iloc[:,1] == 1]['top20'] * 100,

                text = ((US_effect[US_effect.iloc[:,1] == 1]['top20'] * 100).round(2).astype(str) + '%'),

                textposition='auto',

                textfont=dict(size=14),

                name = 'Software Engineer')



# layout of plot

layout = go.Layout(title='Likelihood of being on Top 20%: Effect of being in the US',

                   xaxis=dict(tickfont=dict(size=16)), width=800, height=400,

                   legend=dict(orientation='h', x=0.25))



# building figure and plotting

fig = go.Figure(data=[trace1, trace2], layout=layout)

iplot(fig)
# observing result

pd.set_option('display.max_columns', 200)

df.head()
# creating new target variables

df['top50'] = (df['numerical_compensation'] > df['numerical_compensation'].quantile(0.5)).astype(int)

df['percentile'] = np.round(df['numerical_compensation'].rank(pct=True) * 100)

df['decile'] = np.round(df['numerical_compensation'].rank(pct=True) * 10) * 10

df['quintile'] = np.round(df['numerical_compensation'].rank(pct=True) * 5) * 20
# list of targets

targets = ['top20','top50','percentile','decile', 'quintile', 'numerical_compensation','normalized_numerical_compensation']



# explanatory variables dataframe

explanatory_vars_df = df.drop(targets, axis=1)



# target variables dataframe

targets_df = df[targets]
# choice of treatment variable - we do not include it in the model

treat_var = 'Q6-Software Engineer'

W = df[treat_var]



# design matrix, dropping treatment variable

X = explanatory_vars_df.drop(treat_var,axis=1)



# target variable

y = targets_df['top20']
# importing packages that we'll use

import graphviz

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier



# let us fit a decision tree

dt = DecisionTreeClassifier(max_leaf_nodes=3, min_samples_leaf=100)

dt.fit(X, y)



# let us plot it

dot_data = tree.export_graphviz(dt, out_file=None, 

                                feature_names=X.columns.str.replace('<','less than'),  

                                filled=True, rounded=True,  

                                special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
# creating a df to measure the effects

dt_effect_df = pd.DataFrame({'cluster': dt.apply(X), 'Q6-Software Engineer': W, 'avg. outcome': y})



# let us check the effects

dt_effect_df.groupby(['cluster','Q6-Software Engineer']).mean().round(2)
# libraries

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import KFold, cross_val_score



# CV method

kf = KFold(n_splits=5, shuffle=True)



# model 

et = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, bootstrap=True, n_jobs=-1, random_state=42)



# generating validation predictions

result = cross_val_score(et, X, y, cv=kf, scoring='roc_auc')



# calculating result

print('Cross-validation results (AUC):', result, 'Average AUC:', np.mean(result))
# let us train our model with the full data

et.fit(X, y)



# let us check the most important variables

importances = pd.DataFrame({'variable': X.columns, 'importance': et.feature_importances_})

importances.sort_values('importance', ascending=False, inplace=True)

importances.head(10)
# and get the leaves that each sample was assigned to

leaves = et.apply(X)

leaves
# calculating similarities 

sims_from_first = (leaves[0,:] == leaves[1:,:]).sum(axis=1)



# most similar row w.r.t the first

max_sim, which_max = sims_from_first.max(), sims_from_first.argmax()

print('The most similar row from the first is:', which_max, ', having co-ocurred', max_sim, 'times with it in the forest')
# number of variables that these guys are equal

n_cols_equal = (X.iloc[0] == X.iloc[6592]).loc[importances['variable']].head(20).sum()

print('Rows 0 and 6592 are equal in {} of the 20 most important variables'.format(n_cols_equal))

X.iloc[[0, 6592]].loc[:,importances['variable'].head(20)]
# let us build a supervised embedding using UMAP

# 'hamming' metric is equal to the proportion of leaves

# that two samples have NOT co-ocurred (dissimilarity metric)

from umap import UMAP

embed = UMAP(metric='hamming', random_state=42).fit_transform(leaves)
# opening figure

import matplotlib.pyplot as plt

plt.style.use('bmh')

fig = plt.figure(figsize=(12,7), dpi=120)



# scatterplot

plt.scatter(embed[y==0,0], embed[y==0,1], s=2, color='black', alpha=0.8, label='Bottom 80% earners')

plt.scatter(embed[y==1,0], embed[y==1,1], s=2, color='gold', alpha=0.8, label='Top 20% earners')



# titles

plt.title('Supervised Embedding built from the leaves of our forest: natural clusters')

plt.ylabel('$x_1$'); plt.xlabel('$x_0$')



# legend

plt.legend(markerscale=5)

plt.show()
# variable to set the color

var = X['Q3-United States of America']



# opening figure

fig = plt.figure(figsize=(12,7), dpi=120)



# scatterplot

plt.scatter(embed[var==0,0], embed[var==0,1], s=2, color='red', alpha=0.8, label='NOT US Residents')

plt.scatter(embed[var==1,0], embed[var==1,1], s=2, color='blue', alpha=0.8, label='US Residents')



# titles

plt.title('Supervised Embedding: US vs. non-US residents')

plt.ylabel('$x_1$'); plt.xlabel('$x_0$')



# legend

plt.legend(markerscale=5)

plt.show()
# variable to set the color

var = X['Q24-10-20 years']



# opening figure

fig = plt.figure(figsize=(12,7), dpi=120)



# scatterplot

plt.scatter(embed[var==0,0], embed[var==0,1], s=2, color='red', alpha=0.8, label='NOT 10-20 years of experience')

plt.scatter(embed[var==1,0], embed[var==1,1], s=2, color='blue', alpha=0.8, label='10-20 years of experience')



# titles

plt.title('Supervised Embedding: people with 10-20 years of experience')

plt.ylabel('$x_1$'); plt.xlabel('$x_0$')



# legend

plt.legend(markerscale=5)

plt.show()
# variable to set the color

var = X['Q11-Build prototypes to explore applying machine learning to new areas']



# opening figure

fig = plt.figure(figsize=(12,7), dpi=120)



# scatterplot

plt.scatter(embed[var==0,0], embed[var==0,1], s=2, color='red', alpha=0.8, label='NOT build ML prototypes')

plt.scatter(embed[var==1,0], embed[var==1,1], s=2, color='blue', alpha=0.8, label='build ML prototypes')



# titles

plt.title('Supervised Embedding: people that build ML prototypes')

plt.ylabel('$x_1$'); plt.xlabel('$x_0$')



# legend

plt.legend(markerscale=5)

plt.show()
# opening figure

fig = plt.figure(figsize=(12,7), dpi=120)



# scatterplot

var = W

#plt.scatter(embed[var==0,0], embed[var==0,1], s=10, color='red', alpha=0.8, label='NOT Software Engineers')

#plt.scatter(embed[var==1,0], embed[var==1,1], s=10, color='blue', alpha=0.8, label='US Software Engineers')

plt.scatter(embed[np.logical_and(y==0, W==0),0], embed[np.logical_and(y==0, W==0),1], s=45, color='red', alpha=0.5, label='Bottom 80% earners | NOT Software Engineers', marker='o')

plt.scatter(embed[np.logical_and(y==1, W==0),0], embed[np.logical_and(y==1, W==0),1], s=45, color='red', alpha=0.5, label='Top 20% earners | NOT Software Engineers', marker='x')

plt.scatter(embed[np.logical_and(y==0, W==1),0], embed[np.logical_and(y==0, W==1),1], s=75, color='blue', alpha=0.8, label='Bottom 80% earners | Software Engineers', marker='o')

plt.scatter(embed[np.logical_and(y==1, W==1),0], embed[np.logical_and(y==1, W==1),1], s=75, color='blue', alpha=0.8, label='Top 20% earners | Software Engineers', marker='x')



# titles

plt.title('Checking distribution of Software Engineers and top earners for one cluster')

plt.ylabel('$x_1$'); plt.xlabel('$x_0$')

#plt.ylim(-5.6,-1.00); plt.xlim(-7.5,-3.4)

plt.ylim(-0.45,1.0); plt.xlim(-6.7,-5.99)



# legend

plt.legend(markerscale=1.5)
# so we can be agle to draw rectangles

import matplotlib.patches as patches



# opening figure

fig = plt.figure(figsize=(12,7), dpi=120)



# scatterplot

var = W

#plt.scatter(embed[var==0,0], embed[var==0,1], s=10, color='red', alpha=0.8, label='NOT Software Engineers')

#plt.scatter(embed[var==1,0], embed[var==1,1], s=10, color='blue', alpha=0.8, label='US Software Engineers')

plt.scatter(embed[np.logical_and(y==0, W==0),0], embed[np.logical_and(y==0, W==0),1], s=45, color='red', alpha=0.5, label='Bottom 80% earners | NOT Software Engineers', marker='o')

plt.scatter(embed[np.logical_and(y==1, W==0),0], embed[np.logical_and(y==1, W==0),1], s=45, color='red', alpha=0.5, label='Top 20% earners | NOT Software Engineers', marker='x')

plt.scatter(embed[np.logical_and(y==0, W==1),0], embed[np.logical_and(y==0, W==1),1], s=75, color='blue', alpha=0.8, label='Bottom 80% earners | Software Engineers', marker='o')

plt.scatter(embed[np.logical_and(y==1, W==1),0], embed[np.logical_and(y==1, W==1),1], s=75, color='blue', alpha=0.8, label='Top 20% earners | Software Engineers', marker='x')



# rectangle showing neighborhood

rect = patches.Rectangle((-6.6,0.2),0.2,0.4,linewidth=3,linestyle='dashed',edgecolor='black',facecolor='none')

ax = plt.gca(); ax.add_patch(rect)





# computing effects

embed_df = pd.DataFrame({'x0':embed[:,0], 'x1':embed[:,1], 'W':W, 'y':y})

embed_df['x0'] = embed_df['x0'].astype(float) 

embed_df['x1'] = embed_df['x1'].astype(float) 

local_effect = embed_df.query('0.2 < x1 < 0.6').query('-6.6 < x0 < -6.4').groupby('W')['y'].mean()



# arrow

plt.annotate('', xy=(-6.4, 0.4), xytext=(-6.25, 0.3),

            arrowprops=dict(facecolor='black', shrink=0.05))



# rectangle

rect = patches.Rectangle((-6.24,0.13),0.11,0.32,linewidth=1,edgecolor='black',facecolor='white', zorder=100)

ax = plt.gca(); ax.add_patch(rect)





# red ratio

plt.scatter([-6.08 - 0.13], [-0.3 + 0.63 + 0.08], c=['red'], marker='x', s=60, alpha=0.7, zorder=101)

plt.annotate('+', xy=(-6.085 - 0.13, -0.39 + 0.63  + 0.08), zorder=101)

plt.plot([-6.1- 0.13, -6.06- 0.13], [-0.34 + 0.63 + 0.08,-0.34 + 0.63 + 0.08], 'k', zorder=101)

plt.scatter([-6.1 - 0.13], [-0.38 + 0.63 + 0.08], c=['red'], marker='x', s=60, alpha=0.7, zorder=101)

plt.scatter([-6.06 - 0.13], [-0.38 + 0.63 + 0.08], c=['red'], marker='o', s=60, alpha=0.7, zorder=101)

plt.annotate('= {0:.0f}%'.format(local_effect[0]*100), xy=(-6.05- 0.13, -0.34 + 0.63+ 0.08), zorder=101)



# blue ratio

plt.scatter([-6.08 - 0.13], [-0.46 + 0.63 + 0.08], c=['blue'], marker='x', s=60, alpha=0.7, zorder=101)

plt.annotate('+', xy=(-6.085 - 0.13, -0.55 + 0.63 + 0.08), zorder=101)

plt.plot([-6.1 - 0.13, -6.06 - 0.13], [-0.50 + 0.63 + 0.08,-0.50 + 0.63 + 0.08], 'k', zorder=101)

plt.scatter([-6.1 - 0.13], [-0.54 + 0.63 + 0.08], c=['blue'], marker='x', s=60, alpha=0.7, zorder=101)

plt.scatter([-6.06 - 0.13], [-0.54 + 0.63 + 0.08], c=['blue'], marker='o', s=60, alpha=0.7, zorder=101)

plt.annotate('= {0:.0f}%'.format(local_effect[1]*100), xy=(-6.05 - 0.13, -0.50 + 0.63+ 0.08), zorder=101)



# titles

plt.title('Looking at a local neighborhood')

plt.ylabel('$x_1$'); plt.xlabel('$x_0$')

#plt.ylim(-5.6,-1.00); plt.xlim(-7.5,-3.4)

plt.ylim(-0.45,1.0); plt.xlim(-6.7,-5.99)



# legend

plt.legend(markerscale=1.5)
# importing NNDescent

from pynndescent import NNDescent



# let us use neighborhoods to estimate treatment effects

index = NNDescent(leaves, metric='hamming')



# querying 100 nearest neighbors

nearest_neighs = index.query(leaves, k=201)
# creating a df with treatment assignments and outcomes

y_df = pd.DataFrame({'neighbor': range(X.shape[0]), 'y':y, 'W':W})



# creating df with nearest neighbors

nearest_neighs_df = pd.DataFrame(nearest_neighs[0]).drop(0, axis=1)



# creating df with nearest neighbor weights

nearest_neighs_w_df = pd.DataFrame(1 - nearest_neighs[1]).drop(0, axis=1)



# processing the neighbors df

nearest_neighs_df = (nearest_neighs_df

                     .reset_index()

                     .melt(id_vars='index')

                     .rename(columns={'index':'reference','value':'neighbor'})

                     .reset_index(drop=True))



# processing the neighbor weights df

nearest_neighs_w_df = (nearest_neighs_w_df

                       .reset_index()

                       .melt(id_vars='index')

                       .rename(columns={'index':'reference','value':'weight'})

                       .reset_index(drop=True))



# joining the datasets and adding weighted y variable

nearest_neighs_df = (nearest_neighs_df

                     .merge(nearest_neighs_w_df)

                     .drop('variable', axis=1)

                     .merge(y_df, on='neighbor', how='left')

                     .assign(y_weighted = lambda x: x.y*(x.weight))

                     .sort_values('reference'))



# let us check the neighbors dataframe

nearest_neighs_df.head(10)
# processing to get the effects

treat_effect_df = nearest_neighs_df.assign(count=1).groupby(['reference','W']).sum()

treat_effect_df['y_weighted'] = treat_effect_df['y_weighted']/treat_effect_df['weight']

treat_effect_df['y'] = treat_effect_df['y']/treat_effect_df['count']

treat_effect_df = treat_effect_df.pivot_table(values=['y', 'y_weighted','weight','count'], columns='W', index='reference')



# calculating treatment effects

treat_effect_df.loc[:,'effect'] = treat_effect_df['y'][1] - treat_effect_df['y'][0]

treat_effect_df.loc[:,'effect_weighted'] = treat_effect_df['y_weighted'][1] - treat_effect_df['y_weighted'][0]



# not computing effect for clusters with few examples

min_sample_effect = 10

treat_effect_df.loc[(treat_effect_df['count'][0] < min_sample_effect) | (treat_effect_df['count'][1] < min_sample_effect), 'effect_weighted'] = np.nan

treat_effect_df.loc[(treat_effect_df['count'][0] < min_sample_effect) | (treat_effect_df['count'][1] < min_sample_effect), 'effect'] = np.nan



# observing the result

treat_effect_df.head(10)
# opening figure

fig = plt.figure(figsize=(12,7), dpi=120)



# scatterplot

plt.scatter(embed[:,0], embed[:,1], s=2, 

            c=treat_effect_df['effect_weighted'], 

            alpha=0.8, vmin=-0.08, vmax=0.08, cmap='viridis')



# titles

plt.title('Effect of being a Software Engineer: Top 20% earner probability lift')

plt.ylabel('$x_1$'); plt.xlabel('$x_0$')



# legend

plt.colorbar()

plt.show()
# let us fit a decision tree

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_leaf_nodes=5, min_samples_leaf=100)

dt.fit(X.iloc[treat_effect_df['effect_weighted'].dropna().index], treat_effect_df['effect_weighted'].dropna())
# let us plot a decision tree

import graphviz 

dot_data = tree.export_graphviz(dt, out_file=None, 

                                feature_names=X.columns.str.replace('<','less than'),  

                                filled=True, rounded=True,  

                                special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
# class for computing effects decision tree

class TreatmentEffectEstimate:

    

    # initializing

    def __init__(self, explanatory_vars, target_vars, model=ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, bootstrap=True, n_jobs=-1)):

        

        # storing variables

        self.explanatory_vars = explanatory_vars

        self.target_vars = target_vars

        self.model = model

                

    # running CV for model parameters

    def compute_cv_results(self, metric='accuracy'):

        

        # CV method

        kf = KFold(n_splits=5, shuffle=True)



        # generating validation predictions

        result = cross_val_score(self.model, self.X, self.y, cv=kf, scoring=metric)



        # calculating result

        return result, np.mean(result)

    

    # generating manifold with UMAP

    def compute_manifold(self):

        

        # let us check the embedding with response and treatments

        self.embed = UMAP(metric='hamming').fit_transform(self.leaves)

        

        # returning 

        return self.embed

    

    # running model and neighbors

    def compute_effects(self, treat_var, target_var, n_neighbors=201, dt_max_leaves=5, dt_min_samples_leaf=100):

        

        # separating explanatory vars, treatment var and target var

        self.X = self.explanatory_vars.drop(treat_var, axis=1)

        self.W = self.explanatory_vars[treat_var].copy()

        self.y = self.target_vars[target_var].copy()

        

        # creating a df with treatment assignments and outcomes

        self.y_df = pd.DataFrame({'neighbor': range(self.X.shape[0]), 'y':self.y, 'W':self.W})

        

        # let us train our model with the full data

        self.model.fit(self.X, self.y)



        # and get the leaves that each sample was assigned to

        self.leaves = self.model.apply(self.X)

        

        # let us use neighborhoods to estimate treatment effects in the neighborhood

        self.index = NNDescent(self.leaves, metric='hamming')

        

        # querying 100 nearest neighbors

        self.nearest_neighs = self.index.query(self.leaves, k=n_neighbors)

        

        # creating df with nearest neighbors

        self.nearest_neighs_df = pd.DataFrame(self.nearest_neighs[0]).drop(0, axis=1)



        # creating df with nearest neighbor weights

        self.nearest_neighs_w_df = pd.DataFrame(1 - self.nearest_neighs[1]).drop(0, axis=1)



        # processing the neighbors df

        self.nearest_neighs_df = (self.nearest_neighs_df

                                  .reset_index()

                                  .melt(id_vars='index')

                                  .rename(columns={'index':'reference','value':'neighbor'})

                                  .reset_index(drop=True))



        # processing the neighbor weights df

        self.nearest_neighs_w_df = (self.nearest_neighs_w_df

                                    .reset_index()

                                    .melt(id_vars='index')

                                    .rename(columns={'index':'reference','value':'weight'})

                                    .reset_index(drop=True))



        # joining the datasets and adding weighted y variable

        self.nearest_neighs_df = (self.nearest_neighs_df

                                  .merge(self.nearest_neighs_w_df)

                                  .drop('variable', axis=1)

                                  .merge(self.y_df, on='neighbor', how='left')

                                  .assign(y_weighted = lambda x: x.y*(x.weight))

                                  .sort_values('reference'))

        

        # processing to get the effects

        self.treat_effect_df = self.nearest_neighs_df.assign(count=1).groupby(['reference','W']).sum()

        self.treat_effect_df['y_weighted'] = self.treat_effect_df['y_weighted']/self.treat_effect_df['weight']

        self.treat_effect_df['y'] = self.treat_effect_df['y']/self.treat_effect_df['count']

        self.treat_effect_df = self.treat_effect_df.pivot_table(values=['y', 'y_weighted','weight','count'], columns='W', index='reference')



        # calculating treatment effects

        self.treat_effect_df.loc[:,'effect'] = self.treat_effect_df['y'][1] - self.treat_effect_df['y'][0]

        self.treat_effect_df.loc[:,'effect_weighted'] = self.treat_effect_df['y_weighted'][1] - self.treat_effect_df['y_weighted'][0]



        # not computing effect for clusters with few examples

        self.min_sample_effect = 10

        self.treat_effect_df.loc[(self.treat_effect_df['count'][0] < self.min_sample_effect) | (self.treat_effect_df['count'][1] < self.min_sample_effect), 'effect_weighted'] = np.nan

        self.treat_effect_df.loc[(self.treat_effect_df['count'][0] < self.min_sample_effect) | (self.treat_effect_df['count'][1] < self.min_sample_effect), 'effect'] = np.nan

        

        # let us fit a decision tree

        dt = DecisionTreeRegressor(max_leaf_nodes=dt_max_leaves, min_samples_leaf=dt_min_samples_leaf, random_state=42)

        dt.fit(self.X.iloc[self.treat_effect_df['effect_weighted'].dropna().index], self.treat_effect_df['effect_weighted'].dropna())

        

        # let us plot a decision tree

        dot_data = tree.export_graphviz(dt, out_file=None, 

                                        feature_names=self.X.columns.str.replace('<','less than'),  

                                        filled=True, rounded=True,  

                                        special_characters=True)  

        graph = graphviz.Source(dot_data)  

        return graph 
# instance of our causal inference model

tee = TreatmentEffectEstimate(explanatory_vars_df, targets_df)
# treatment variable

treat_var = 'Q6-Software Engineer'

target_var = 'quintile'



# calculating

tee.compute_effects(treat_var, target_var)
# treatment variable

treat_var = 'Q6-Data Scientist'

target_var = 'quintile'



# calculating

tee.compute_effects(treat_var, target_var)
# treatment variable

treat_var = 'Q1-Female'

target_var = 'quintile'



# calculating

tee.compute_effects(treat_var, target_var)
# treatment variable

treat_var = 'Q16-R'

target_var = 'quintile'



# calculating

tee.compute_effects(treat_var, target_var)
# treatment variable

treat_var = 'Q16-Python'

target_var = 'quintile'



# calculating

tee.compute_effects(treat_var, target_var)
# treatment variable

treat_var = 'Q31-Genetic Data'

target_var = 'quintile'



# calculating

tee.compute_effects(treat_var, target_var)