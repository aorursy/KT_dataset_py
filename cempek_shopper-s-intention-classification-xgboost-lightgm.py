import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset = pd.read_csv('../input/online-shoppers-intention/online_shoppers_intention.csv')

dataset.head()
def missing_data(dataset):

    missing_count = dataset.isnull().sum()

    missing_percentage = dataset.isnull().sum() / dataset.count() * 100

    missing_table = pd.concat([missing_count, missing_percentage], axis = 1, keys = ['No_missing_data', 'Percentege_missing_data'])

    return(np.transpose(missing_table))

missing_data(dataset)
dataset = dataset.dropna()
dataset.describe()
# Percentage based on the type of webpage

df_administrative = dataset[dataset.Administrative_Duration > 0]

df_informational = dataset[dataset.Informational_Duration > 0]

df_ProductRelated_Duration = dataset[dataset.ProductRelated_Duration > 0]

df_admin_info = dataset[(dataset.Administrative_Duration > 0) & (dataset.Informational_Duration > 0)]

df_admin_product = dataset[(dataset.Administrative_Duration > 0) & (dataset.ProductRelated_Duration > 0)]

df_info_product = dataset[(dataset.Informational_Duration > 0) & (dataset.ProductRelated_Duration > 0)]

df_admin_info_product = dataset[(dataset.Administrative_Duration > 0) & (dataset.Informational_Duration > 0) & (dataset.ProductRelated_Duration > 0)]





df_website_percentage = {}

df_website_percentage = pd.DataFrame(df_website_percentage)



df_website_percentage.loc['Website Usage','Admininstrative'] = df_administrative['Administrative_Duration'].count() / dataset['Administrative_Duration'].count() * 100

df_website_percentage.loc['Revenue Rate','Admininstrative'] = df_administrative['Revenue'].sum()  /  df_administrative['Administrative_Duration'].count() * 100



df_website_percentage.loc['Website Usage','Informational'] = df_informational['Informational_Duration'].count() / dataset['Informational_Duration'].count() * 100

df_website_percentage.loc['Revenue Rate','Informational'] = df_informational['Revenue'].sum()  /  df_informational['Informational_Duration'].count() * 100



df_website_percentage.loc['Website Usage','Product Related'] = df_ProductRelated_Duration['ProductRelated_Duration'].count() / dataset['ProductRelated_Duration'].count() * 100

df_website_percentage.loc['Revenue Rate','Product Related'] = df_ProductRelated_Duration['Revenue'].sum()  /  df_ProductRelated_Duration['ProductRelated_Duration'].count() * 100



df_website_percentage.loc['Website Usage','Admin & Info'] = df_admin_info['Administrative_Duration'].count() / dataset['Administrative_Duration'].count() * 100

df_website_percentage.loc['Revenue Rate','Admin & Info'] = df_admin_info['Revenue'].sum()  /  df_admin_info['ProductRelated_Duration'].count() * 100



df_website_percentage.loc['Website Usage','Admin & Product'] = df_admin_product['Administrative_Duration'].count() / dataset['Administrative_Duration'].count() * 100

df_website_percentage.loc['Revenue Rate','Admin & Product'] = df_admin_product['Revenue'].sum()  /  df_admin_product['Administrative_Duration'].count() * 100



df_website_percentage.loc['Website Usage','Info & Product'] = df_info_product['ProductRelated_Duration'].count() / dataset['Administrative_Duration'].count() * 100

df_website_percentage.loc['Revenue Rate','Info & Product'] = df_info_product['Revenue'].sum()  /  df_info_product['ProductRelated_Duration'].count() * 100



df_website_percentage.loc['Website Usage','Admin & Info & Product'] = df_admin_info_product['ProductRelated_Duration'].count() / dataset['ProductRelated_Duration'].count() * 100

df_website_percentage.loc['Revenue Rate','Admin & Info & Product'] = df_admin_info_product['Revenue'].sum()  /  df_admin_info_product['ProductRelated_Duration'].count() * 100





# Plot for website percentages baded on usage

fig, ax1 = plt.subplots(figsize = (12,5))

sns.barplot(x = df_website_percentage.columns, y = df_website_percentage.loc['Website Usage',:], ax = ax1)

for p in ax1.patches:

    ax1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

ax1.set_title('The Percentage of Website Usage Based on Type', fontsize = 16)

ax1.set_ylim(0,100)

ax1.tick_params(axis='x', rotation=30)

ax1.set(xlabel = 'The Website Types', ylabel = 'Percentage')

plt.show()
no_column = 0

fig, ax = plt.subplots(2,4, figsize= (12,5))

plt.suptitle('The Percentage of Revenue Based On Website Type', fontsize = 16)

for i in range(2):

    for k in range(4):

        ax[i,k].pie([df_website_percentage.iloc[1,no_column], 100-df_website_percentage.iloc[1,no_column]], autopct='%1.1f%%', labels = ['True', 'False'] ,shadow=True, startangle=90 )

        ax[i,k].axis('equal')

        ax[i,k].set_title(df_website_percentage.columns[no_column])

        no_column += 1

        if no_column == 7:

           fig.delaxes(ax.flatten()[7])

           break



plt.show()
df_website_percentage
fig, ax = plt.subplots(1,7, figsize = (20,4))

plt.suptitle('The percentage of revenue based on website')

for i, col_title in enumerate(df_website_percentage):

    ax[i].pie([df_website_percentage.loc['Revenue Rate',col_title], 100- df_website_percentage.loc['Revenue Rate',col_title]],autopct='%1.1f%%', labels = ['True', 'False'] ,shadow=True, startangle=90 )

    ax[i].set_title(col_title)

    centre_circle = plt.Circle((0,0),0.70,fc='white')

    fig = plt.gcf()

    fig.gca().add_artist(centre_circle)

    ax[i].axis('equal')

    

plt.show()
    # The Revenue by Month

sums_revenue = pd.DataFrame(dataset.groupby(['Month', 'Revenue'])['Revenue'].count().rename('Total')).sort_values(by = ['Total'], ascending = False)

sums_revenue.reset_index(inplace = True)

fig, ax = plt.subplots(figsize = (12,10))

sns.barplot(x = sums_revenue.Month, y = sums_revenue.Total, hue = sums_revenue.Revenue)

plt.title('The Revenue by Month', fontsize = 16 )

plt.show()
fig, ax = plt.subplots(2,5, figsize = (18,5))

no_month = 0

plt.suptitle('The Percentage of Revenue by Month', fontsize = 16)

for i in range(2):

    for k in range(5):

        temp_percentage = sums_revenue[(sums_revenue['Month'] == sums_revenue['Month'].unique()[no_month] )&(sums_revenue['Revenue'] == True)]['Total'] / sums_revenue[sums_revenue['Month'] == sums_revenue['Month'].unique()[no_month]]['Total'].sum() * 100

        temp_percentage = pd.Series([temp_percentage, 100- temp_percentage])

        ax[i,k].pie(temp_percentage, autopct = '%.2f%%', labels = ['True', 'False'])

        ax[i,k].set_title(sums_revenue['Month'].unique()[no_month])

        ax[i,k].axis('equal')

        no_month += 1
pivot2 = dataset.pivot_table(index = ['Month','VisitorType'], values = ['BounceRates','ExitRates'] , aggfunc = np.mean)

pivot2.reset_index(inplace = True)

        # Bar plot

fig, ax = plt.subplots(figsize = (12,10))

sns.barplot(x = pivot2['Month'], y = pivot2['ExitRates'], hue = pivot2['VisitorType'])

plt.title('The Exit Rate based on Visitor Type by Month', fontsize = 16)

plt.show()
for k in range(len(pivot2)):

    if pivot2['Month'][k] == 'June':

        pivot2['Month'][k] = 6

    else:

        pivot2['Month'][k]= datetime.strptime(pivot2['Month'][k], '%b').month

fig, ax = plt.subplots(figsize = (12,10))

sns.barplot(x = pivot2.Month, y = pivot2.ExitRates, hue = pivot2.VisitorType)

plt.title('The Exit Rate based on Visitor Type by Month', fontsize = 16)

plt.show()
VisitorType_groupby = pd.DataFrame(dataset.groupby(['VisitorType', 'Revenue'])['Revenue'].count().rename('Total'))

VisitorType_groupby.reset_index(inplace = True)

fig, ax = plt.subplots(figsize = (12,10))

sns.barplot(x = VisitorType_groupby.VisitorType, y = VisitorType_groupby.Total, hue = VisitorType_groupby.Revenue)

plt.ylabel('Number of Visitors')

plt.title('The Number of Revenue by Visitor Type', fontsize = 16)

plt.show()
VisitorType_Revenue = {'New Visitor Revenue' : [],'Other Revenue' : [], 'Returning Visitor Revenue':[]}

VisitorType_Revenue = pd.DataFrame(VisitorType_Revenue)

VisitorType_Revenue['New Visitor Revenue'] = VisitorType_groupby[(VisitorType_groupby['VisitorType'] == 'New_Visitor') & ( VisitorType_groupby['Revenue'] == True)]['Total'] / VisitorType_groupby[VisitorType_groupby['VisitorType'] == 'New_Visitor']['Total'].sum()

VisitorType_Revenue['Other Revenue'][1] =VisitorType_groupby[(VisitorType_groupby['VisitorType'] == 'Other') & (VisitorType_groupby['Revenue'] == True)]['Total'] / VisitorType_groupby[VisitorType_groupby['VisitorType'] == 'Other']['Total'].sum()

VisitorType_Revenue['Returning Visitor Revenue'][1] = VisitorType_groupby[(VisitorType_groupby['VisitorType'] == 'Returning_Visitor') & (VisitorType_groupby['Revenue'] == True)]['Total'] / VisitorType_groupby[VisitorType_groupby['VisitorType'] == 'Returning_Visitor']['Total'].sum()

VisitorType_Revenue.head()



fig, ax = plt.subplots(1,3, figsize = (12,5))

plt.suptitle('The Percentage of Revenue by Type of Visitor', fontsize = 18)

for i in range(3):

    ax[i].pie([VisitorType_Revenue.iloc[0,i], 1-VisitorType_Revenue.iloc[0,i]], labels = ['True', 'False'], autopct = '%.2f%%' ,shadow=True, startangle=90)

    ax[i].set_title(VisitorType_Revenue.columns[i])

    ax[i].axis('equal')

plt.show()
# Correlation Matrix

import seaborn as sns

dataset_interval = dataset[['Administrative', 'Administrative_Duration', 'Informational',

       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',

       'BounceRates', 'ExitRates', 'PageValues', 'Revenue']]

correlation_matrix = dataset_interval.corr()

fig, ax = plt.subplots(figsize = (15,15))

sns.heatmap(correlation_matrix, annot =True, annot_kws = {'size': 10})

plt.xticks(rotation = 30)
sns.jointplot(data = dataset, x= 'ExitRates', y = 'BounceRates', kind = 'reg')

plt.suptitle('Exit and Bounce Rate Correlation', fontsize = 16)

plt.show()
sns.jointplot(data = dataset, x = 'ProductRelated_Duration', y = 'ProductRelated',kind = 'reg' )

sns.jointplot(data = dataset, x = 'Administrative_Duration', y = 'Administrative',kind = 'reg' )

plt.show()
dataset.drop(['Administrative','Informational','ProductRelated', 'ExitRates'], axis = 1, inplace = True)
dataset = pd.get_dummies(dataset, columns = ['SpecialDay','Month','OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend','VisitorType'])
dataset.columns

len(dataset.columns)
X = dataset.drop(['Revenue'], axis = 1).values

y = dataset.loc[:,'Revenue'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

classifier_logistic = LogisticRegression(random_state = 0)

classifier_logistic.fit(X_train, y_train)

y_pred_logistic = classifier_logistic.predict(X_test)

y_pred_logistic_proba = classifier_logistic.predict_proba(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

cm_logistic = confusion_matrix(y_test, y_pred_logistic)

sns.heatmap(cm_logistic, annot = True, fmt='d')

print('Logistic Regression accuracy is {:.4f}'.format(accuracy_logistic))
from sklearn.tree import DecisionTreeClassifier

classifier_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_leaf = 5)

classifier_tree.fit(X_train, y_train)

y_pred_tree = classifier_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

cm_tree = confusion_matrix(y_test, y_pred_tree)

sns.heatmap(cm_tree, annot = True, fmt = 'd')

print('Desicion Tree accuracy is {:.4f}'.format(accuracy_tree))
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, y_pred_logistic_proba[:,1])

plt.plot(fpr,tpr)

plt.grid()

plt.show()

auc_logistic_proba = roc_auc_score(y_test,y_pred_logistic_proba[:,1])

print('Logistic regression prob auc score: {:.2f}'.format(auc_logistic_proba))

auc_logistic = roc_auc_score(y_test,y_pred_logistic)

print('Logistic regression auc score: {:.2f}'.format(auc_logistic))
from xgboost import XGBClassifier

classifier_xg = XGBClassifier()

classifier_xg.fit(X_train, y_train)

y_pred_xg = classifier_xg.predict(X_test)
accuracy_xg = accuracy_score(y_test, y_pred_xg)

print("XGBoost Classifier accuracy is : {:.4f}".format(accuracy_xg))

cm_xg = confusion_matrix(y_test, y_pred_xg)

sns.heatmap(cm_xg, annot = True, annot_kws = {'size' : 10}, fmt = 'd')
from sklearn.metrics import classification_report, f1_score, roc_auc_score

print(classification_report(y_test, y_pred_logistic))

print(classification_report(y_test, y_pred_tree))

print(classification_report(y_test, y_pred_xg))
import keras

from keras.models import Sequential

from keras.layers import Dense

classifier_keras = Sequential()

classifier_keras.add(Dense(input_dim = 76,output_dim = 25, init = 'uniform', activation = 'relu' ))

classifier_keras.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu'))

classifier_keras.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier_keras.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier_keras.fit(X_train, y_train , epochs = 100)

y_pred_keras = classifier_keras.predict(X_test)

y_pred_keras = (y_pred_keras > 0.6)
y_pred_keras = classifier_keras.predict(X_test)

y_pred_keras = (y_pred_keras > 0.5)

accuracy_keras = accuracy_score(y_test, y_pred_keras)

print('Keras accuracy score is : {}'.format(accuracy_keras) )

cm_keras = confusion_matrix(y_test, y_pred_keras)

sns.heatmap(cm_keras, annot = True, annot_kws = {'size' : 10}, fmt = 'd')
# Lightgbm Classifier 

import lightgbm as lgb



d_train = lgb.Dataset(X_train, label = y_train)



params = {}



params['learning_rate'] = 0.0002

params['boosting_type'] = 'gbdt'

params['abjective'] = 'binary'

params['metric'] = 'binary_logloss'

params['sub_feature'] = 0.5

params['num_leaves'] = 20

params['min_data'] = 50

params['max_depth'] = 10 



clf = lgb.train(params, d_train, 1000)       #  100 is number of iterations



y_pred_lgm = clf.predict(X_test)



for i in range(len(y_pred_lgm)):

    if y_pred_lgm[i] >= 0.188:

        y_pred_lgm[i] = 1

    else:

        y_pred_lgm[i] = 0

        

from sklearn.metrics import accuracy_score



accuracy_lgm = accuracy_score(y_pred_lgm, y_test)



print(accuracy_lgm)



cm_lgm = confusion_matrix(y_pred_lgm, y_test)

sns.heatmap(cm_lgm, annot = True, fmt = 'd')
# Grid Search for lightgbm



mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',

          objective = 'binary',

          n_jobs = 3)  



from sklearn.model_selection import GridSearchCV



grid_params = {'learning_rate' : [0.5, 0.2 , 0.1 ,0.01 ,0.002 ,0.001 ,0.0001],

               'boosting_type' : ['gbdt'],

               'objective' : ['binary'],

               'metric' : ['binary_logloss'],

               'sub_feature' : [0.5],

               'num_leaves' : [10,20,30,40],

               'min_data' : [20, 40,50],

               'max_depth' : [10,20,30,40],

               'reg_alpha' : [1,1.2],

               'reg_lambda' : [1,1.2,1.4]

                }



grid_search = GridSearchCV(estimator = mdl, param_grid = grid_params, scoring = 'accuracy', cv = 4)

grid_search = grid_search.fit(X_train, y_train)

best_score = grid_search.best_score_

best_parameters = grid_search.best_params_



y_grid_pred_lgm = grid_search.predict(X_test)





accuracy_grid_lgm = accuracy_score(y_grid_pred_lgm, y_test)



print(accuracy_grid_lgm)



cm_lgm = confusion_matrix(y_grid_pred_lgm, y_test)

sns.heatmap(cm_lgm, annot = True, fmt = 'd')
df_comparison = {'Model' : [], 'Accuracy' : [], 'F1_score' : [], 'ROC_AUC_Score': []}

df_comparison = pd.DataFrame(df_comparison)

df_comparison.loc[0,:] = ['Logistic', accuracy_score(y_test, y_pred_logistic), f1_score(y_test, y_pred_logistic), roc_auc_score(y_test, y_pred_logistic)]

df_comparison.loc[1,:] = ['Decision Tree', accuracy_score(y_test, y_pred_tree), f1_score(y_test, y_pred_tree), roc_auc_score(y_test, y_pred_tree)]

df_comparison.loc[2,:] = ['XGBoost', accuracy_score(y_test, y_pred_xg), f1_score(y_test, y_pred_xg), roc_auc_score(y_test, y_pred_xg)]

df_comparison.loc[3,:] = ['Neural Network', accuracy_score(y_test, y_pred_keras), f1_score(y_test, y_pred_keras), roc_auc_score(y_test, y_pred_keras)]

df_comparison.loc[4,:] = ['LightGM', accuracy_score(y_test, y_pred_lgm), f1_score(y_test, y_pred_lgm), roc_auc_score(y_test, y_pred_lgm)]

df_comparison.loc[5,:] = ['LightGM with Grid Search', accuracy_score(y_test, y_grid_pred_lgm), f1_score(y_test, y_grid_pred_lgm), roc_auc_score(y_test, y_grid_pred_lgm)]
df_comparison