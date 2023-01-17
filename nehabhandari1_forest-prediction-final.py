!pip install seaborn
# for data manipulation

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for visualization

from matplotlib import pyplot as plt

# to include graphs inline within the frontends next to code

import seaborn as sns



# preprocessing functions and evaluation models

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV,RandomizedSearchCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.dummy import DummyClassifier



# machine learning models

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,ExtraTreesClassifier

from xgboost import XGBClassifier
submission_sample = pd.read_csv('../input/forest-cover-type-prediction/sampleSubmission.csv')

train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')

test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')
print("Number of rows and columns in the train dataset are:", train.shape)
train.head()
train.tail()
train.dtypes
print(list(enumerate(train.columns)))
train.iloc[:,1:10].describe()
train.nunique()
train.isna().sum()
def outlier_function(df, col_name):

    ''' this function detects first and third quartile and interquartile range for a given column of a dataframe

    then calculates upper and lower limits to determine outliers conservatively

    returns the number of lower and uper limit and number of outliers respectively

    '''

    first_quartile = np.percentile(np.array(df[col_name].tolist()), 25)

    third_quartile = np.percentile(np.array(df[col_name].tolist()), 75)

    IQR = third_quartile - first_quartile

                      

    upper_limit = third_quartile+(3*IQR)

    lower_limit = first_quartile-(3*IQR)

    outlier_count = 0

                      

    for value in df[col_name].tolist():

        if (value < lower_limit) | (value > upper_limit):

            outlier_count +=1

    return lower_limit, upper_limit, outlier_count

# loop through all columns to see if there are any outliers

for i in train.columns:

    if outlier_function(train, i)[2] > 0:

        print("There are {} outliers in {}".format(outlier_function(train, i)[2], i))
trees = train[(train['Horizontal_Distance_To_Fire_Points'] > outlier_function(train, 'Horizontal_Distance_To_Fire_Points')[0]) &

              (train['Horizontal_Distance_To_Fire_Points'] < outlier_function(train, 'Horizontal_Distance_To_Fire_Points')[1])]

trees.shape
size=10

Uni = []

for i in range(size+1,len(train.columns)-1):

    Uni.append(pd.unique(train[train.columns[i]].values))

Uni
train.iloc[:,11:15].sum(axis=1).sum()
train.iloc[:,15:55].sum(axis=1).sum()
plt.figure(figsize=(15,10))

sns.countplot(train['Cover_Type'])

plt.xlabel("Type of Cpver", fontsize=12)

plt.ylabel("Rows Count", fontsize=12)

plt.show()
fig, axes = plt.subplots(nrows = 2,ncols = 5,figsize = (25,15))

g= sns.FacetGrid(train, hue='Cover_Type',height=5)

(g.map(sns.distplot,train.columns[1],ax=axes[0][0]))

(g.map(sns.distplot, train.columns[2],ax=axes[0][1]))

(g.map(sns.distplot, train.columns[3],ax=axes[0][2]))

(g.map(sns.distplot, train.columns[4],ax=axes[0][3]))

(g.map(sns.distplot, train.columns[5],ax=axes[0][4]))

(g.map(sns.distplot, train.columns[6],ax=axes[1][0]))

(g.map(sns.distplot, train.columns[7],ax=axes[1][1]))

(g.map(sns.distplot, train.columns[8],ax=axes[1][2]))

(g.map(sns.distplot, train.columns[9],ax=axes[1][3]))

(g.map(sns.distplot, train.columns[10],ax=axes[1][4]))

plt.close(2)

plt.legend()
size=10

fig, axes = plt.subplots(nrows = 2,ncols = 5,figsize = (25,10))

for i in range(0,size):

    row = i // 5

    col = i % 5

    ax_curr = axes[row, col]

    sns.boxplot(x="Cover_Type", y=train.columns[i], data=train,ax=ax_curr);
sns.pairplot(train, hue='Cover_Type', vars=train.columns[1:11])
# Bivariate EDA

pd.crosstab(train.Soil_Type31, train.Cover_Type)
#Convert dummy features back to categorical

x = train.iloc[:,15:55]

y = train.iloc[:,11:15]

y = pd.DataFrame(y)

x = pd.DataFrame(x)

s2 = pd.Series(x.columns[np.where(x!=0)[1]])

s3 = pd.Series(y.columns[np.where(y!=0)[1]])

train['soil_type'] = s2

train['Wilderness_Area'] = s3

train.head()
# Create a new dataset exluding dummies variable for Mutivariate EDA

df_viz = train.iloc[:, 0:15]

df_viz = df_viz.drop(['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 

                      'Wilderness_Area4'], axis = 1)

df_viz.head()
plt.figure(figsize=(15,10))

pd.crosstab(train.Wilderness_Area, train.Cover_Type).plot.bar(figsize=(10,10),stacked = True)
#plt.figure(figsize=(30,30))

pd.crosstab(train.soil_type, train.Cover_Type).plot.bar(figsize=(20,10),stacked = True)
corr = df_viz.corr()



# plot the heatmap

plt.figure(figsize=(14,12))

colormap = plt.cm.RdBu

sns.heatmap(corr,linewidths=0.1, 

            square=False, cmap=colormap, linecolor='white', annot=True)

plt.title('Pearson Correlation of Numeric Features', size=14)
train=trees
def add_feature(data):   

    data['Ele_minus_VDtHyd'] = data.Elevation-data.Vertical_Distance_To_Hydrology

    data['Ele_plus_VDtHyd'] = data.Elevation+data.Vertical_Distance_To_Hydrology

    data['Distanse_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology']**2+data['Vertical_Distance_To_Hydrology']**2)**0.5

    data['Hydro_plus_Fire'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Fire_Points']

    data['Hydro_minus_Fire'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Fire_Points']

    data['Hydro_plus_Road'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Roadways']

    data['Hydro_minus_Road'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Roadways']

    data['Fire_plus_Road'] = data['Horizontal_Distance_To_Fire_Points']+data['Horizontal_Distance_To_Roadways']

    data['Fire_minus_Road'] = data['Horizontal_Distance_To_Fire_Points']-data['Horizontal_Distance_To_Roadways']

    return data
train = add_feature(train)

test = add_feature(test)
#X_train = train.drop(['Id','Cover_Type','soil_type','Wilderness_Area'], axis = 1)

X_train = train.drop(['Id','Cover_Type'], axis = 1)

y_train = train.Cover_Type

X_test = test.drop(['Id'], axis = 1)
%%time 



lr_pipe = Pipeline(

    steps = [

        ('scaler', MinMaxScaler()),

        ('classifier', LogisticRegression(solver='lbfgs', n_jobs=-1))

    ]

)



lr_param_grid = {

    'classifier__C': [1, 10, 100,1000],

}





np.random.seed(1)

grid_search = GridSearchCV(lr_pipe, lr_param_grid, cv=5, refit='True')

grid_search.fit(X_train, y_train)



print(grid_search.best_score_)

print(grid_search.best_params_)

%%time 



rf_pipe = Pipeline(

    steps = [

        ('classifier', RandomForestClassifier(n_estimators=500))

    ]

)



param_grid = {

       'classifier__min_samples_leaf': [1,4,7],

    'classifier__max_depth': [34,38,32],

}



np.random.seed(1)

rf_grid_search = GridSearchCV(rf_pipe, param_grid, cv=5, refit='True', n_jobs=-1)

rf_grid_search.fit(X_train, y_train)



print(rf_grid_search.best_score_)

print(rf_grid_search.best_params_)
rf_model = rf_grid_search.best_estimator_



cv_score = cross_val_score(rf_model, X_train, y_train, cv = 5)

print(cv_score)

print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))
rf = rf_grid_search.best_estimator_.steps[0][1]
feat_imp = rf.feature_importances_

feat_imp_df = pd.DataFrame({

    'feature':X_train.columns,

    'feat_imp':feat_imp

})



feat_imp_df.sort_values(by='feat_imp', ascending=False).head(10)
sorted_feat_imp_df = feat_imp_df.sort_values(by='feat_imp', ascending=True)

plt.figure(figsize=[6,6])

plt.barh(sorted_feat_imp_df.feature[-20:], sorted_feat_imp_df.feat_imp[-20:])

plt.show()
%%time 



xgd_pipe = Pipeline(

    steps = [

        ('classifier', XGBClassifier(n_estimators=50, subsample=0.5))

    ]

)



param_grid = {

    'classifier__learning_rate' : [0.45],

    'classifier__min_samples_split' : [8, 16, 32],

    'classifier__min_samples_leaf' : [2],

    'classifier__max_depth': [15]

    

}



np.random.seed(1)

xgd_grid_search = GridSearchCV(xgd_pipe, param_grid, cv=5,

                              refit='True', verbose = 10, n_jobs=-1)

xgd_grid_search.fit(X_train, y_train)



print(xgd_grid_search.best_score_)

print(xgd_grid_search.best_params_)
xgd_model = xgd_grid_search.best_estimator_



cv_score = cross_val_score(xgd_model, X_train, y_train, cv = 5)

print(cv_score)

print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))
xrf_pipe = Pipeline(

    steps = [

        ('classifier', ExtraTreesClassifier(n_estimators=500,random_state=0, criterion = 'entropy'))

    ]

)





xrf_param_grid = {

    'classifier__min_samples_leaf': [1,4,7],

    'classifier__max_depth': [34,38,32],

}



np.random.seed(1)

xrf_grid_search = GridSearchCV(xrf_pipe, xrf_param_grid, cv=5, refit='True', n_jobs=-1)

xrf_grid_search.fit(X_train, y_train)



print(xrf_grid_search.best_score_)

print(xrf_grid_search.best_params_)
xrf_model = xrf_grid_search.best_estimator_



cv_score = cross_val_score(xrf_model, X_train, y_train, cv = 5)

print(cv_score)

print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))
final_model = xrf_grid_search.best_estimator_.steps[0][1]

final_model.fit(X_train, y_train)













y_pred = final_model.predict(X_test)
print(len(test.Id))
print(len(y_pred))
from collections import Counter

Counter(y_pred)
submission_sample.head()
submission = pd.DataFrame({'Id': test.Id,

                           'Cover_Type': y_pred})

submission.head()
submission.to_csv('submission.csv', index=False)