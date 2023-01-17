import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
import random
import math
from scipy.stats.stats import pearsonr
from scipy.stats.stats import ttest_ind
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from sklearn import tree
from sklearn.tree import _tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pprint import pprint

%matplotlib inline
plt.style.use('ggplot')
chocolate = pd.read_csv("../input/chocolate-bar-ratings/flavors_of_cacao.csv")
chocolate.head()
chocolateOriginal = chocolate.copy()
chocolateOriginal.head()
chocolate.columns = ["Company","SpecificOrigin","Ref","ReviewDate","CocoaPercent","Location","Rating","BeanType","BroadOrigin"]
chocolate.dtypes
chocolate.describe()
chocolate.describe(exclude=[np.number])
fig, ax = plt.subplots(1,1,figsize=(11, 5))
g1 = sns.distplot(chocolate.Rating, kde=False, bins=np.arange(1, 5.5, 0.25),
                  hist_kws={"rwidth": 0.9, "align": "left", "alpha": 1.0})
plt.suptitle("Chocolate Ratings", fontsize=16)
fig.show()
chocolate[chocolate["Rating"] == 5]
chocolate[chocolate["Rating"] <= 1]
chocolate.CocoaPercent = chocolate.CocoaPercent.apply(lambda x : float(x.rstrip("%"))/100.0)
chocolate.describe()
g = chocolate.CocoaPercent.hist(bins=58, figsize=(18,9))
plt.title("Chocolate - Cocoa Percent")
plt.show(g)
chocolate.ReviewDate.value_counts()
g = chocolate.ReviewDate.hist(bins=12, figsize=(18,9))
plt.title("Chocolate - Review Date")
plt.show(g)
ref = chocolate.Ref.copy()
print("Number of unique ref values:" + str(ref.nunique()))
ref.sort_values(inplace=True)
print("\nCount of ref values by number of occurrences")
print(pd.DataFrame(ref.value_counts()).groupby('Ref').Ref.count())
ref = chocolate.groupby('ReviewDate').Ref
ref = pd.concat([ref.count(), ref.min(), ref.max(), ref.nunique()], axis=1).reset_index()
ref.columns = ['ReviewDate','count','minRef','maxRef','uniqueRef']
g = chocolate.boxplot(column="Ref", by="ReviewDate", figsize=(18,12))
ref
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
g = plt.subplots(figsize=(18, 9))
g = sns.barplot(x=chocolate.Company.value_counts().index[0:50], y=chocolate.Company.value_counts()[0:50], palette="Blues_d")
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Chocolate Companies")
plt.show(g)
chocolate["Company"] = chocolate.Company.apply(lambda x: re.split("\(|aka ",x.rstrip(")"))[-1])
company = chocolate[["Company","Ref"]].groupby("Company").count().reset_index() #sort_values(by="Ref", ascending=False).reset_index()
company["newCompany"] = company.apply(lambda x: x.Company if x.Ref > 20 else "other", axis=1)
company
chocolate = chocolate.merge(company[["Company","newCompany"]], how="left", on="Company")
fig,(ax1,ax2) = plt.subplots(2, 1, sharex=True)

g1 = sns.countplot(chocolate.newCompany, order=chocolate.newCompany.value_counts().index, palette="Blues_d", ax=ax1)
g1.set_xticklabels('')
g1.set_xlabel('')
g1.set_ylabel('')
g1.set_ylim(1400,1460)

g2 = sns.countplot(chocolate.newCompany, order=chocolate.newCompany.value_counts().index, palette="Blues_d", ax=ax2)
g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
g2.set_ylim(0,60)

plt.suptitle("Chocolate Companies", fontsize=16)
plt.subplots_adjust(hspace=0.2)
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
g = plt.subplots(figsize=(18, 9))
g = sns.countplot(chocolate.Location, order=chocolate.Location.value_counts().index, palette="Blues_d")
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Company Locations")
plt.show(g)
locs = chocolate.Location.value_counts()
chocolate["LocName"] = chocolate.Location.apply(lambda x: "Other" if x in locs[locs < 30].index else x)
#locs[locs > 30].index
chocolate.LocName.value_counts()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
g = plt.subplots(figsize=(18, 9))
g = sns.countplot(chocolate.LocName, order=chocolate.LocName.value_counts().index, palette="Blues_d")
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Company Locations")
plt.show(g)
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

g = plt.subplots(figsize=(18, 9))
g = sns.countplot(chocolate.BeanType, order=chocolate.BeanType.value_counts().index, palette="YlOrBr_r")
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Chocolate Bean Types")
plt.show(g)
chocolate['BeanT'] = chocolate.BeanType.replace(np.nan, 'not-specified', regex=True).replace('\xa0', 'not-specified').apply(
    lambda x : ("Blend-Criollo" if "Criollo" in re.split(" |,|\)",str(x)) else "Blend") if any(
        word in x for word in ['Blend',',']) else x).apply(lambda x : (x.split()[0]))
chocolate.describe(exclude=[np.number])
chocolate.groupby('BeanT').BeanT.count()
chocolate['BeanT'] = chocolate['BeanT'].apply(
    lambda x: "Other" if x in ["Amazon","Beniano","CCN51","EET","Matina","Nacional"] else x)

fig, ax = plt.subplots(1,1,figsize=(11, 5))
g1 = sns.countplot(chocolate.BeanT, palette="YlOrBr_r") #, ax=ax[0])
g1.set_xticklabels(g1.get_xticklabels(), rotation=90)
plt.suptitle("Chocolate Bean Types and Blend", fontsize=16)
fig.show()
chocolate.groupby('BeanT').BeanT.count()
#chocolate.groupby('SpecificOrigin').SpecificOrigin.count().sort_values(ascending=False).head(10)
print(chocolate.groupby('BroadOrigin').BroadOrigin.count().sort_index())
chocolate["Origin"] = chocolate.BroadOrigin.replace(np.nan, 'not specified', regex=True).replace(
    '\xa0', 'not specified').str.replace('Dom.*','Dominican Republic').str.replace('Ven.*','Venezuela').apply(
    lambda x: re.split(',|\(|\/|\&|\-',str(x))[0].rstrip().replace('Cost ','Costa ').replace('DR','Dominican Republic').replace(
        'Tobago','Trinidad').replace('Trinidad','Trinidad and Tobago').replace("Carribean","Caribbean"))
print(chocolate.groupby('Origin').Origin.count().sort_index())
chocolate["Origin"] = chocolate.Origin.apply(
    lambda x: x.replace('Gre.','Grenada').replace('Guat.','Guatemala').replace("Hawaii","United States of America").replace(
        'Mad.','Madagascar').replace('PNG','Papua New Guinea').replace('Principe','Sao Tome').replace(
        'Sao Tome','Sao Tome and Principe'))
print(chocolate.groupby('Origin').Origin.count().sort_index())
countriesRaw = pd.read_csv("../input/country-to-continent/countryContinent.csv", encoding='iso-8859-1') 
countriesRaw
countries = countriesRaw[["country","sub_region","continent"]]
countries.country = countries.country.apply(lambda x: re.split("\(|\,",x)[0].rstrip())
countries = countries.drop_duplicates()
countries
chocolate["Origin"] = chocolate["Origin"] = chocolate.Origin.apply(
    lambda x: x.replace("St.","Saint").replace("Vietnam","Viet Nam").replace("Burma","Myanmar").replace(
        "Ivory Coast","Côte d'Ivoire").replace("West","Western").replace(" and S. "," "))
print(chocolate.groupby('Origin').Origin.count().sort_index())
chocolate = chocolate.merge(countries[["country","sub_region"]], how="left", left_on="Origin", right_on="country")
chocolate[chocolate.country.isnull()].groupby("Origin").Origin.count().sort_index()
chocolate.loc[chocolate.Origin=="Hawaii","country"] = "United States of America"
chocolate.loc[chocolate.Origin=="Hawaii","sub_region"] = "Northern America"

chocolate.loc[chocolate.country.isnull(),"sub_region"] = chocolate.loc[chocolate.country.isnull(),"Origin"]
chocolate.loc[chocolate.country.isnull(),"country"] = "--"
regions = countries[["sub_region","continent"]].drop_duplicates()
chocolate = chocolate.merge(regions, how="left", on="sub_region")
chocolate.loc[chocolate.Origin=='Africa',"continent"] = 'Africa'
chocolate.continent = chocolate.continent.replace(np.nan,"other")
print(chocolate[["continent","sub_region","country","Origin"]].groupby(["continent","sub_region","country"]).count())
chocCounts = chocolate[["Origin","Ref"]].groupby(["Origin"]).count()
chocCounts.columns = ["countryCount"]
chocRollup = chocolate.merge(chocCounts, how="left", left_on="Origin", right_index=True)[["Origin","sub_region","countryCount"]]
chocolate.Origin = chocRollup.apply(lambda x: x.sub_region if x.countryCount < 28 else x.Origin, axis=1)
print(chocolate[["continent","sub_region","country","Origin"]].groupby(["continent","sub_region","Origin"]).count())
chocCounts = chocolate[["Origin","Ref"]].groupby(["Origin"]).count()
chocCounts.columns = ["countryCount"]
chocRollup = chocolate.merge(chocCounts, how="left", left_on="Origin", right_index=True)[["Origin","continent","countryCount"]]
chocolate.Origin = chocRollup.apply(lambda x: x.continent if x.countryCount < 28 else x.Origin, axis=1)
print(chocolate[["continent","country","Origin"]].groupby(["continent","Origin"]).count())
#print(chocolate[["continent","sub_region","country","Origin"]].groupby(["continent","sub_region","Origin"]).count())
print(chocolate.loc[chocolate.Origin.str.contains("America"),["Origin","BroadOrigin","country"]].groupby(["Origin","BroadOrigin"]).count())
chocolate.loc[chocolate.Origin.isin(["Americas","Central America"]),"Origin"] = "Central and South America"
print(chocolate[["continent","country","Origin"]].groupby(["continent","Origin"]).count())
chocolate.SpecificOrigin.describe()
origin = chocolate[['SpecificOrigin', 'Ref']].groupby(['SpecificOrigin']).count().reset_index()
origin[origin.Ref >= 20]
chocolate.head()
chocolate=chocolate.loc[:,["Rating", "CocoaPercent", "newCompany", "LocName", "BeanT", "Origin"]]
chocolate.columns = ["Rating","CocoaPercent","Company","Location","BeanType","Origin"]
chocolate.dtypes
chocolate.head()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
g1 = sns.lmplot(x="CocoaPercent", y="Rating", data=chocolate, y_jitter=0.2, x_jitter=0.01)
plt.title("Chocolate: Cocoa Percentage vs Rating")
fig.show()
pearsonr(chocolate.CocoaPercent, chocolate.Rating)
comp_lm = ols('Rating ~ Company', data=chocolate).fit()
print(comp_lm.params)
print(sm.stats.anova_lm(comp_lm, typ=2))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
g = plt.subplots(figsize=(18, 9))
g = sns.boxplot(x=chocolate.Company, y=chocolate.Rating, palette="YlOrBr_r",
                order=chocolate[["Company","Rating"]].groupby("Company").mean().sort_values("Rating", ascending=False).index)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Chocolate: Company vs Rating")
plt.show(g)
loc_lm = ols('Rating ~ Location', data=chocolate).fit()
print(loc_lm.params)
print(sm.stats.anova_lm(loc_lm, typ=2))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
g = plt.subplots(figsize=(18, 9))
g = sns.boxplot(x=chocolate.Location, y=chocolate.Rating, palette="YlOrBr_r",
                order=chocolate[["Location","Rating"]].groupby("Location").mean().sort_values("Rating", ascending=False).index)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Chocolate: Location vs Rating")
plt.show(g)
bean_lm = ols('Rating ~ BeanType', data=chocolate).fit()
print(bean_lm.params)
print(sm.stats.anova_lm(bean_lm, typ=2))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
g = plt.subplots(figsize=(18, 9))
g = sns.boxplot(x=chocolate.BeanType, y=chocolate.Rating, palette="YlOrBr_r",
                order=chocolate[["BeanType","Rating"]].groupby("BeanType").mean().sort_values("Rating", ascending=False).index)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Chocolate: Bean Types vs Rating")
plt.show(g)
orig_lm = ols('Rating ~ Origin', data=chocolate).fit()
print(orig_lm.params)
print(sm.stats.anova_lm(orig_lm, typ=2))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
g = plt.subplots(figsize=(18, 9))
g = sns.boxplot(x=chocolate.Origin, y=chocolate.Rating, palette="YlOrBr_r",
                order=chocolate[["Origin","Rating"]].groupby("Origin").mean().sort_values("Rating", ascending=False).index)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Chocolate: Bean Origin vs Rating")
plt.show(g)
print("Contingency Tests for Categorical attributes")
print("Company and Location: {}".format(chi2_contingency(pd.crosstab(chocolate.Company,chocolate.Location))[1]))
print("Company and BeanType: {}".format(chi2_contingency(pd.crosstab(chocolate.Company,chocolate.BeanType))[1]))
print("Company and Origin: {}".format(chi2_contingency(pd.crosstab(chocolate.Company,chocolate.Origin))[1]))
print("Location and BeanType: {}".format(chi2_contingency(pd.crosstab(chocolate.Location,chocolate.BeanType))[1]))
print("Location and Origin: {}".format(chi2_contingency(pd.crosstab(chocolate.Location,chocolate.Origin))[1]))
print("BeanType and Origin: {}".format(chi2_contingency(pd.crosstab(chocolate.BeanType,chocolate.Origin))[1]))
print("ANOVA tests between CocoaPercent and categorical attributes")
print(sm.stats.anova_lm(ols('CocoaPercent ~ Company', data=chocolate).fit(), typ=2))
print(sm.stats.anova_lm(ols('CocoaPercent ~ Location', data=chocolate).fit(), typ=2))
print(sm.stats.anova_lm(ols('CocoaPercent ~ BeanType', data=chocolate).fit(), typ=2))
print(sm.stats.anova_lm(ols('CocoaPercent ~ Origin', data=chocolate).fit(), typ=2))
random.seed(12345)
testSize = len(chocolate) // 5
testIndices = random.sample(range(len(chocolate)),testSize)
testIndices.sort()
chocTest = chocolate.iloc[testIndices,]
print("Test data set has {} observations and {} attributes".format(chocTest.shape[0],chocTest.shape[1]))
chocTrain = chocolate.drop(testIndices)
print("Training data set has {} observations and {} attributes".format(chocTrain.shape[0],chocTrain.shape[1]))
trainX = pd.get_dummies(chocTrain.iloc[:,1:])
trainY = chocTrain.Rating
print("Training data set has {} observations and {} attributes".format(trainX.shape[0],trainX.shape[1]))
testX = pd.get_dummies(chocTest.iloc[:,1:])
testY = chocTest.Rating
print("Test data set has {} observations and {} attributes".format(testX.shape[0],testX.shape[1]))
olsModel = ols('Rating ~ CocoaPercent + BeanType + Origin + Location + Company', data=chocTrain).fit()
print(olsModel.params)
reg = linear_model.BayesianRidge()
reg.fit(trainX,trainY)
reg.coef_
lrResults = pd.DataFrame(trainY[0:10])
lrResults["Ols"] = round(olsModel.predict(chocTrain.iloc[0:10])*4)/4
lrResults["Reg"] = np.round(reg.predict(trainX.iloc[0:10])*4)/4
lrResults
dtrModel = tree.DecisionTreeRegressor(max_depth=5)
dtrModel.fit(trainX,trainY)
def tree_to_code(tree, feature_names):

    '''
    Outputs a decision tree model as a Python function
    
    Parameters:
    -----------
    tree: decision tree model
        The decision tree to represent as a function
    feature_names: list
        The feature names of the dataset used for building the decision tree
    '''

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
tree_to_code(dtrModel,trainX.columns)
random.seed(8765)

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [8, 10, 12],
    'max_features': [8, 9, 10],
    'min_samples_leaf': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8, 10, 12, 14, 16],
    'splitter': ['best', 'random']
}
# Create a based model
dtr = tree.DecisionTreeRegressor(max_depth=5)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = dtr, param_grid = param_grid, 
                           cv = 10, n_jobs = -1, verbose = 2)
grid_search.fit(trainX, trainY)
bestDtr = grid_search.best_estimator_
grid_search.best_params_
tree_to_code(bestDtr,trainX.columns)
dtResults = pd.DataFrame(trainY[0:20])
dtResults["First"] = np.round(dtrModel.predict(trainX.iloc[0:20])*4)/4
dtResults["Tuned"] = np.round(bestDtr.predict(trainX.iloc[0:20])*4)/4
dtResults
random.seed(2468)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 10 fold cross validation, 
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100,
                               cv = 10, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(trainX, trainY)
bestrf_random = rf_random.best_estimator_
rf_random.best_params_
random.seed(2468)
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [15, 20, 25],
    'max_features': [6, 8, 10],
    'min_samples_leaf': [2],
    'min_samples_split': [10],
    'n_estimators': [800, 1000, 1200]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
rf_grid = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 2)
rf_grid.fit(trainX, trainY)
bestrf_grid = rf_grid.best_estimator_
rf_grid.best_params_
dtResults["RF"] = np.round(bestrf_grid.predict(trainX.iloc[0:20])*4)/4
dtResults
random.seed(97531)
param_grid = {
    'C': [0.01, 0.1, 1.0],
    'epsilon': [0.01, 0.1, 1.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': [0.01, 0.1, 1],
    'coef0': [-1, 0, 1]
}
svr = SVR()
svmGrid = GridSearchCV(estimator = svr, param_grid = param_grid, 
                           cv = 10, n_jobs = -1, verbose = 2)
svmGrid.fit(trainX, trainY)
best_svr = svmGrid.best_estimator_
svmGrid.best_params_
meanTrain = np.round(trainY.mean()*4)/4
print("Baseline prediction using training mean\nRMSE: {:5.3}".format(math.sqrt(((testY - meanTrain) ** 2).mean())))
testResults = pd.DataFrame(["ols","reg","dtr","dtr_tuned","rf","svm"])
testResults.columns = ["model"]
testResults["test"] = [0,0,0,0,0,0]
testResults["train"] = [0,0,0,0,0,0]
testResults
def rmse(predict, labels):
    return math.sqrt(((predict-labels)**2).mean())
testResults.loc[0,"test"] = rmse(np.round(olsModel.predict(chocTest)*4)/4, testY)
testResults.loc[0,"train"] = rmse(np.round(olsModel.predict(chocTrain)*4)/4, trainY)
testResults.loc[1,"test"] = rmse(np.round(reg.predict(testX)*4)/4, testY)
testResults.loc[1,"train"] = rmse(np.round(reg.predict(trainX)*4)/4, trainY)
testResults.loc[2,"test"] = rmse(np.round(dtrModel.predict(testX)*4)/4, testY)
testResults.loc[2,"train"] = rmse(np.round(dtrModel.predict(trainX)*4)/4, trainY)
testResults.loc[3,"test"] = rmse(np.round(bestDtr.predict(testX)*4)/4, testY)
testResults.loc[3,"train"] = rmse(np.round(bestDtr.predict(trainX)*4)/4, trainY)
testResults.loc[4,"test"] = rmse(np.round(bestrf_grid.predict(testX)*4)/4, testY)
testResults.loc[4,"train"] = rmse(np.round(bestrf_grid.predict(trainX)*4)/4, trainY)
testResults.loc[5,"test"] = rmse(np.round(best_svr.predict(testX)*4)/4, testY)
testResults.loc[5,"train"] = rmse(np.round(best_svr.predict(trainX)*4)/4, trainY)
testResults
results = pd.DataFrame(testY)
results["Predict"] = np.round(bestrf_grid.predict(testX)*4)/4
results["Error"] = np.abs(results.Rating - results.Predict)
results
results[['Error', 'Predict']].groupby(['Error']).count().reset_index()
