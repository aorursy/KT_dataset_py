import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import random
random.seed(2018)

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from IPython.display import display

# Supervised Learning
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

# Unsupervised Models
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

# Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import scipy.stats as st

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Specify index/ target name
id_col = "ID"
target_var = "target"

# House Keeping Parameters
rstate = 23
Debug = False
Home = False

debug_num = 200
if Home is True:
    import os
    path = r"D:\My Computer\DATA\Santander"
    os.chdir(path)
    
    print("Data Load Stage")
    training = pd.read_csv('train.csv', index_col = id_col)
    if Debug is True : training = training.sample(debug_num)
    traindex = training.index
    testing = pd.read_csv('test.csv', index_col = id_col)
    if Debug is True : testing = testing.sample(debug_num)
    testdex = testing.index
else:
    print("Data Load Stage")
    training = pd.read_csv('../input/train.csv', index_col = id_col)
    if Debug is True : training = training.sample(debug_num)
    traindex = training.index
    testing = pd.read_csv('../input/test.csv', index_col = id_col)
    if Debug is True : testing = testing.sample(debug_num)
    testdex = testing.index

y = np.log1p(training[target_var])
training.drop(target_var,axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))
# Feature Names
feat_names = df.columns

# Modeling Datasets
test_df = df.loc[testdex,:]
X = df.loc[traindex,:]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15,random_state=rstate)
print("X Train", X_train.shape, "\ny Train",y_train.shape, "\nX Test",X_test.shape, "\ny Test",y_test.shape)

print("Starting Model. Train shape: {}, Test shape: {}".format(X.shape,test_df.shape))
print("Feature Num: ",len(feat_names))

# Utility functions:
def rmse(y_true, y_pred):
    return abs(np.sqrt(np.mean((y_true-y_pred)**2)))
scoring = make_scorer(rmse, greater_is_better=False)

# Report best scores
def report(results, n_top=3):
    for i in list(range(2, n_top + 2)):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate]*-1,
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# Storage for Model and Results
results = pd.DataFrame(columns=['Model','Para','Validation_Score','CV Mean','CV STDEV'])
submission_df = pd.DataFrame(index=testdex)

def save(model, modelname):
    global results
    model.best_estimator_.fit(X, y)
    submission = np.expm1(model.predict(test_df))
    
    df = pd.DataFrame({'ID':testdex, 
                        'target':submission})
    df.to_csv("{}.csv".format(modelname),header=True,index=False)
    submission_df[modelname] = submission
    
    model.best_estimator_.fit(X_train, y_train)
    top = np.flatnonzero(model.cv_results_['rank_test_score'] == 1)
    CV_scores = (model.cv_results_['mean_test_score'][top][0])*-1
    STDev = model.cv_results_['std_test_score'][top][0]
    Test_scores = rmse(y_test, model.predict(X_test))
    
    # CV and Save Scores
    results = results.append({'Model': modelname,'Para': model.best_params_,'Validation_Score': Test_scores,
                             'CV Mean':CV_scores, 'CV STDEV': STDev}, ignore_index=True)
    
    # Print Evaluation
    print("\nEvaluation Method: RMSE")
    print("Optimal Model Parameters: {}".format(grid.best_params_))
    print("Training Set RMSE: ", rmse(y_train, model.predict(X_train)))
    print("CV Accuracy: {0:.2f} (+/- {1:.2f}) [%{2}]".format(CV_scores, STDev, modelname))
    print('Unseen Data Validation Score:', Test_scores)

print("Functions Defined..")
# Save All Search Output
search_output = {}
alldimmodels= time.time()

# Dimensionality Reduction Central
pca = PCA(random_state=rstate)
tsvd = TruncatedSVD(random_state=rstate)
ica = FastICA(random_state=rstate)
grp = GaussianRandomProjection(random_state=rstate,eps=0.1)
srp = SparseRandomProjection(random_state=rstate, dense_output=True,eps=1)

dimensionality_reduction_models = [(srp,"srp"),
                                   (pca, "pca"),
                                   (tsvd,"tsvd"),
                                   # (ica,"ica"), # Hits a block..
                                   (grp,"grp")
                                  ]

component_range = [5,300]
iterations = 50
print("Extra Tree Regressor Parameters:{}\n".format(ExtraTreesRegressor().get_params().keys()))
for DR,DR_name in dimensionality_reduction_models:
    modelstart= time.time()
    print("Starting ExtraTree Regression with",DR_name.upper())
    print("Parameters for {}:\n{}".format(DR_name.upper(),SVR().get_params().keys()))
    model = Pipeline(steps=[(DR_name,DR), ('xtree', ExtraTreesRegressor())])

    # Use Scipy to create Parameter Distributions to sample from
    param_grid = {DR_name + '__n_components': st.randint(*component_range),
                  'xtree__n_estimators': st.randint(200,800),
                  'xtree__max_depth': [4,8,12]
                 }
    grid = RandomizedSearchCV(model, param_grid, cv=2, verbose=1, n_iter=iterations, random_state=rstate, scoring=scoring, return_train_score=True, iid= False)
    
    # Train and Save
    grid.fit(X_train, y_train)
    save(grid,DR_name)
    temp = pd.DataFrame.from_dict(grid.cv_results_)
    search_output[DR_name] = abs(pd.concat([temp.drop(['params'], axis=1), temp['params'].apply(pd.Series)], axis=1)).astype("float")
    
    #print("\nReport")
    #report(grid.cv_results_)
    print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
    del temp, grid, model
    print("\n######################################################################\n######################################################################\n")
print("All Model Runtime: %0.2f Minutes"%((time.time() - alldimmodels)/60))
# Score vs. Component Number
component_scores = pd.DataFrame(columns = ["N Components","Score", "Max Depth","Tree Number", "Dimensionality Method"])
for dimred in [algo[1]for algo in dimensionality_reduction_models]:
    temp = search_output[dimred].loc[:,["param_" + str(dimred) + "__n_components", "mean_test_score", 'param_xtree__max_depth', "param_xtree__n_estimators"]]
    temp["Dimensionality Method"] = str(dimred.upper())
    temp.columns = ["N Components","Score", "Max Depth","Tree Number", "Dimensionality Method"]
    component_scores = pd.concat([component_scores, temp], axis=0)
    
# Plot
component_scores["N Components"] = component_scores["N Components"].astype(int)
g = sns.lmplot(x="N Components", y="Score", data=component_scores,
               hue="Dimensionality Method", size=6, aspect = 1.5, order=2)
plt.title("Comparing Dimensionality Reduction Methods\nScored on Extra Trees")
plt.show()
sns.pairplot(component_scores, hue="Dimensionality Method",diag_kind="kde")
plt.show()
g = sns.FacetGrid(component_scores, col="Max Depth", hue = "Dimensionality Method")
g = (g.map(sns.regplot, "N Components", "Score",lowess=True).add_legend())
g.fig.suptitle('Number of Dimensionality Components')
plt.subplots_adjust(top=0.85)
plt.show()
print("What About Extra Tree Ensemble Size?")
g = sns.FacetGrid(component_scores, col="Max Depth", hue = "Dimensionality Method")
g = (g.map(sns.regplot, "Tree Number", "Score",lowess=True).add_legend())
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Number of Trees')
plt.show()
n_bins = 15
# Bin Continuous Variables
for bin_col in ["Tree Number","N Components"]:
    lower, higher = component_scores[bin_col].min().astype(int), component_scores[bin_col].max().astype(int)
    edges = range(lower, higher, (higher - lower)//n_bins) # the number of edges is 8
    lbs = [round((edges[i] + edges[i+1])/2) for i in range(len(edges)-1)]
    component_scores[bin_col + " Bins"] = pd.cut(component_scores[bin_col], bins=n_bins, labels=lbs, include_lowest=True)
    
# Plot
f, ax = plt.subplots(figsize=(8,8))
ax.set_title('Dimensionality Reduction Components and Tree Count Heatmap for Score')
sns.heatmap(component_scores.loc[component_scores["Dimensionality Method"] == "GRP",:].pivot_table(values="Score", index="Tree Number Bins", columns="N Components Bins", aggfunc='mean'),
                annot=False, linewidths=.5, ax=ax,cbar_kws={'label': 'RMSE Score'}, cmap="viridis")
plt.show()
print("Algorithms Correlation Matrix")
display(submission_df.corr())

print("Algorigthms and N Components - Sorted Scores")
display(component_scores.sort_values(by= "Score",ascending=True)[:15])
# Simple Ensemble
submission_df["target"] = submission_df.mean(axis=1)
submission_df["target"].to_csv("mean_score.csv",index=True)

display(submission_df["target"].head())
print("All Model Runtime: %0.2f Minutes"%((time.time() - alldimmodels)/60))
