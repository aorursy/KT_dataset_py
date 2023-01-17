import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pickle



pd.set_option('display.max_columns', 100)

pd.options.mode.chained_assignment = None
train_path = '../input/forest-cover-type-prediction/train.csv'

test_path = '../input/forest-cover-type-prediction/test.csv'

submit_path = '../input/forest-cover-type-prediction/sampleSubmission.csv'
dtrain = pd.read_csv(train_path, index_col=0)

dtest = pd.read_csv(test_path, index_col=0)
dtrain['Cover_Type'].value_counts()
dtrain.info()
# Now this includes values for all classes, better to groupyby the target variable and then get description.

dtrain.describe()
print(dtrain.skew())
grouped_dataframe = dtrain.groupby(['Cover_Type'])

# Dictionary of the Cover_type and the label 



label_dict = {1 : 'Spruce/Fir', 2 : 'Lodgepole Pine', 3 : 'Ponderosa Pine', 4 : 'Cottonwood/Willow', 5 :

              'Aspen', 6 : 'Douglas-fir', 7 : 'Krummholz'}
from IPython.display import display



for cover in dtrain.Cover_Type.unique():

    print(f'Forest Cover Type - {cover}')

    display(grouped_dataframe.get_group(cover).describe())
# Only continuous columns

d_train_cont=dtrain.iloc[:,:10] 

# To plot multiple distributions filtered by the target for each continuous variable.

import math 

targets = dtrain.Cover_Type.unique()

fig = plt.figure()

height = 34

width = 18

fig.set_figheight(height)

fig.set_figwidth(width)

for i, col in enumerate(d_train_cont.columns):

    ax = fig.add_subplot(math.ceil(len(d_train_cont.columns.to_list())/2), 2, i+1) 

    for cover_type in targets:

        temp = d_train_cont.loc[dtrain.Cover_Type == cover_type]

        sns.distplot(temp[col], label = label_dict[cover_type])

    ax.legend()

    ax.set_title(col)

#plt.savefig('Graph/Univariate_cont_dist.jpg')

plt.show()
d_train_cont['Cover_type'] = dtrain.Cover_Type

fig = plt.figure()

fig.set_figheight(34)

fig.set_figwidth(18)



for i, item in enumerate(d_train_cont.columns.to_list()):

    fig.add_subplot(math.ceil(len(d_train_cont.columns.to_list())/2), 2, i+1)

    sns.violinplot(y= item, x = 'Cover_type', data = d_train_cont)



#plt.savefig('Graph/Bivariate_feat_cover.jpg')

plt.show()
# Correlation heatmap would be too large, find largest correlations.

plt.figure(figsize=(9, 7))

sns.heatmap(d_train_cont.corr(),annot=True, cbar = True)

plt.show()
# 2nd Method, get all corrrelations and corresponding rows and columns in numpyarray from the top triangle

# of matrix. Then sort this array.



corr_list = []

for row_num, row in enumerate(d_train_cont.corr().index):

    for col_num, col in enumerate(d_train_cont.corr().index):

        # Ignoring comparison between the same columns

        if col_num > row_num:

            corr_list.append([row, col, np.abs(d_train_cont.corr().iloc[row_num, col_num])])

            

corr_array = np.array(corr_list)

corr_array = corr_array[corr_array[:,2].argsort()][::-1]

corr_array[:10]
# Iterating over the corr_array array and then using the column names from the 1st, 2nd element of list.

# create new figure and add subplots inside loop

fig = plt.figure()



fig.set_figheight(30)

fig.set_figwidth(22)

fig.set_dpi(120)

for i, item in enumerate(corr_array[:10]):

    fig.add_subplot(math.ceil(len(corr_array[:10])/2), 2, i+1 )

    sns.scatterplot(x = item[0], y = item[1], data = dtrain, hue = 'Cover_Type', legend = 'full', palette=sns.husl_palette(7))

#plt.savefig('Graph/data_interaction.jpg')

plt.show()
# Filter cover type and then barplot of wilderness area to see if any trees grow exclusively in a region.

#data.describe()

data = dtrain.groupby(['Cover_Type'])[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].sum()

# Transpose to get numbers by wilderness type.

data.T.plot(kind = 'bar', figsize = (12,8))

plt.show()
# Drop Soil type 15,7 - They have no variation. 

dtrain.drop(['Soil_Type7', 'Soil_Type15'], axis = 1, inplace = True)

# filtering all columns that contain the str Soil

soil_columns = dtrain.columns[dtrain.columns.str.contains('Soil')].to_list()
data_soil = dtrain.groupby(['Cover_Type'])[soil_columns[:10]].sum()

data_soil.T.plot(kind = 'bar', figsize = (18,8))

plt.show()
data_soil = dtrain.groupby(['Cover_Type'])[soil_columns[10:20]].sum()

data_soil.T.plot(kind = 'bar', figsize = (18,8))

plt.show()
data_soil = dtrain.groupby(['Cover_Type'])[soil_columns[20:30]].sum()

data_soil.T.plot(kind = 'bar', figsize = (18,8))

plt.show()
data_soil = dtrain.groupby(['Cover_Type'])[soil_columns[30:]].sum()

data_soil.T.plot(kind = 'bar', figsize = (18,8))

plt.show()
label = dtrain['Cover_Type']

dtrain.drop(['Cover_Type'], axis = 1, inplace=True)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, classification_report,confusion_matrix

from sklearn.tree import DecisionTreeClassifier 

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
x_train, x_test, y_train, y_test = train_test_split(dtrain, label, test_size = .3)

dirty_clf = RandomForestClassifier()

dirty_clf.fit(x_train, y_train)

print(dirty_clf.score(x_test, y_test))

imp_feat = pd.DataFrame(index= dtrain.columns.to_list() , data= dirty_clf.feature_importances_)

imp_feat.rename(columns={0 : 'Importance'}, inplace=True)

imp_feat.sort_values(by='Importance', axis =0, ascending=False)[:15]
baseline_features = ['Elevation', 'Horizontal_Distance_To_Roadways']

features = ['Elevation', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Hydrology',

            'Horizontal_Distance_To_Fire_Points', 'Aspect','Wilderness_Area1', 'Wilderness_Area4', 'Soil_Type3',

            'Soil_Type4','Soil_Type10', 'Soil_Type29',

            'Soil_Type38']

x_train, x_test, y_train, y_test = train_test_split(dtrain[features], label, test_size = .3)
clf = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_split=2, class_weight= None, max_features=None,

                                              max_leaf_nodes=None,

                                              min_impurity_decrease=0.0,

                                              min_impurity_split=None,

                                              min_samples_leaf=1,

                                              min_weight_fraction_leaf=0.0,

                                              presort='deprecated',

                                              random_state=None,)
grid_params = {'criterion' : ["gini", "entropy"]}

grid = GridSearchCV(estimator=clf, param_grid=grid_params, n_jobs=-1, cv = 5)

grid.fit(x_train, y_train)

grid.score(x_test, y_test)
grid.best_estimator_

y_pred = grid.predict(x_test)
clf.fit(x_train, y_train)

print(f'No of Leaves : {clf.get_n_leaves()}')

clf.feature_importances_
# With the Selected Features.

print(classification_report(y_test, y_pred, labels= list(label_dict.keys()), target_names=list(label_dict.values())))
rnd_clf = RandomForestClassifier()

grid_params_1 = {'max_depth' : [18], 'n_estimators' : [127], 'criterion':['entropy']}

grid = GridSearchCV(estimator=rnd_clf, param_grid=grid_params_1, n_jobs=-1, cv = 5)

grid.fit(x_train, y_train)
print(grid.best_score_)

print(grid.score(x_test, y_test))

#grid.cv_results_

#grid.best_estimator_
final_clf = RandomForestClassifier(max_depth=18, n_estimators=127, criterion='entropy')

final_clf.fit(x_train, y_train)

print(final_clf.score(x_train, y_train))

print(final_clf.score(x_test, y_test))

y_hat = final_clf.predict(x_test)
print(classification_report(y_test, y_hat, target_names=label_dict.values()))
plt.figure(figsize=(8,8))

sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred),

                         index = label_dict.values(), columns= label_dict.values()), annot=True, cbar = False)

plt.show()
imp_feat = pd.DataFrame(index= features , data= final_clf.feature_importances_)

imp_feat.rename(columns={0 : 'Importance'}, inplace=True)

imp_feat.sort_values(by='Importance', axis =0, ascending=False)
xgb_clf = XGBClassifier(n_estimators=100, max_depth = 12)
#grid_params = {'max_depth' : [12,14,16]}

#grid_xgb = GridSearchCV(xgb_clf, grid_params, cv= 5)

#grid_xgb.fit(x_train, y_train)

#print(grid_xgb.best_score_)

#grid_xgb.cv_results_

#grid_xgb.score(x_test, y_test)
xgb_clf.fit(x_train, y_train)

xgb_clf.score(x_test, y_test)
y_pred = xgb_clf.predict(x_test)
print(classification_report(y_test, y_pred, target_names=label_dict.values()))
# Final Fit

xgb_clf.fit(dtrain[features], label)
y_test_hat = xgb_clf.predict(dtest[features])
dtest['Predicted_cover_type'] = y_test_hat

sns.countplot(x = 'Predicted_cover_type', data = dtest)
sns.distplot(dtest.Elevation)
test_targets = dtest.Predicted_cover_type.unique()

plt.figure(figsize=(10,6))

for target in test_targets:

    temp = dtest.loc[dtest.Predicted_cover_type == target]

    sns.distplot(temp.Elevation, label = label_dict[target])



plt.legend()

plt.title('Distribution of Elevation of Predicted Cover Type')

#plt.savefig('Graph/Predicted_classes.jpg')

plt.show()
df_submit = pd.read_csv(submit_path, index_col=0)

df_submit['Cover_Type'] =y_test_hat

df_submit.to_csv('submit_kaggle.csv')