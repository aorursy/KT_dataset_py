import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

import networkx as nx

import random

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve,auc,average_precision_score,accuracy_score

%matplotlib inline



print(os.listdir("../input"))



pd.set_option("max_colwidth",1000000)

pd.set_option('max_columns', 15000)
# Read dataset

survey_2019 = pd.read_csv(r'../input/developer_survey_2019/survey_results_public.csv')

schema = pd.read_csv(r'../input/developer_survey_2019/survey_results_schema.csv')
survey_2019.shape
survey_2019.info()
survey_2019.describe()
survey_2019.columns[survey_2019.isnull().mean()==0]
survey_2019.isnull().mean().sort_values(ascending=False)
# List all possible role

survey_2019['DevType'].value_counts(dropna=False)
# Check any missing DevType

np.sum(survey_2019['DevType'].isnull())
# The transpose data set should have 88883-7548=81335 observations.

df = pd.get_dummies(survey_2019['DevType'].str.split(';', expand=True)

   .stack()

   ).sum(level=0)

df.shape
df.head()
fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

plt.title("Correlation of Developer")

ax = sns.heatmap(df.corr(),vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
# Counter the number of developer who is a data scientist and a front-end or full-stack developer

num1 = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)

           & (survey_2019['DevType'].str.contains("front-end",na = False)

           | survey_2019['DevType'].str.contains("full-stack",na = False))].shape[0]

# Counter the number of developer who is a data scientist

num2 = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)].shape[0]

print(num1)

print(num2)

print(num1/num2)
# Count the number of developer who is a front-end or full-stack developer

num3 = survey_2019[survey_2019['DevType'].str.contains("front-end",na = False)

           | survey_2019['DevType'].str.contains("full-stack",na = False)].shape[0]

print(num3)

print(num2/num3)
# Divide the data set into 3 subset and check the number of record for each group

non_data_scientist = survey_2019[~survey_2019['DevType'].str.contains("Data scientist",na = False)]

mixed_data_scientist = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)

                                  & (survey_2019['DevType']!='Data scientist or machine learning specialist')]

pure_data_scientist = survey_2019[survey_2019['DevType']=='Data scientist or machine learning specialist']

print(non_data_scientist.shape[0])

print(mixed_data_scientist.shape[0])

print(pure_data_scientist.shape[0])
# Define a function to show the top feature for each group.

def total_count(df, col1):

    '''

    INPUT:

    df - the pandas dataframe you want to search

    col1 - the column name you want to look through

    

    OUTPUT:

    new_df - a dataframe shows the percentage account for the total observation. 

    '''

    new_df = df[col1].str.split(';', expand=True).stack().value_counts(dropna=False).reset_index()

    new_df.rename(columns={0: 'count'}, inplace=True)

    new_df['percentage'] = new_df['count']/np.sum(df[col1].notnull())

    return new_df

    
total_count(non_data_scientist,'LanguageWorkedWith')
def barplot_group(col1,width=20,height=8):

    '''

    INPUT

    col1 - column name you want to analyze

    width - width of the graph

    height -height of the graph

    

    OUTPUT

    output the a barplot graph showing percentage of column accounting for the total by each group

    '''

    

    df1 = total_count(non_data_scientist,col1)

    df2 = total_count(mixed_data_scientist,col1)

    df3 = total_count(pure_data_scientist,col1)

    df1['role'] = 'non ds'

    df2['role'] = 'mixed ds'

    df3['role'] = 'pure ds'

    df = pd.concat([df1,df2,df3])

 

    fig, ax = plt.subplots()

    fig.set_size_inches(width, height)

    ax = sns.barplot(x="index", y="percentage", hue="role", data=df)

    plt.legend(loc=1, prop={'size': 20})
barplot_group('LanguageWorkedWith',30,8)

plt.title("Language worked with")
barplot_group('DatabaseWorkedWith')
barplot_group('PlatformWorkedWith')
barplot_group('WebFrameWorkedWith')
barplot_group('MiscTechWorkedWith')
barplot_group('DevEnviron',30,8)
barplot_group('OpSys')
# Explore the pure data scientist group

# data_scientist = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]



# Gather the technology variables

temp = pure_data_scientist[['LanguageWorkedWith'

                                ,'DatabaseWorkedWith'

                                ,'PlatformWorkedWith'

                                ,'WebFrameWorkedWith'

                                ,'MiscTechWorkedWith'

                                ,'DevEnviron'

                                ]]



# Create a tech combining all technologies into one variable.

temp['tech'] = temp['LanguageWorkedWith'].map(str)+";"+temp['DatabaseWorkedWith'].map(str)+";"+temp['PlatformWorkedWith'].map(str)+";"+temp['WebFrameWorkedWith'].map(str)+";"+temp['MiscTechWorkedWith'].map(str)+";"+temp['DevEnviron'].map(str)



# Transpose tech to build a one hot matrix 

df = pd.get_dummies(temp['tech'].str.split(';', expand=True)

   .stack()

   ).sum(level=0)



# drop the nan column

df = df.drop(columns=['nan'])



# Convert the value to integer.

df_asint = df.astype(int)



# Create co-occurrence matrix

coocc = df_asint.T.dot(df_asint)

coocc
# networkx time              

# create edges with weight, and a note list

edge_list = []

node_list = []

for index, row in coocc.iterrows():

    i = 0

    for col in row:

        weight = float(col)/df.shape[0]

        

        if weight >=0.2:    # ignore weak weight.

            

            if index != coocc.columns[i]:

                edge_list.append((index, coocc.columns[i], weight))

            

            #create a note list

            if index == coocc.columns[i]:

                node_list.append((index, weight))

        i += 1

# networkx graph

G = nx.Graph()

for i in sorted(node_list):

    G.add_node(i[0], size = i[1])

G.add_weighted_edges_from(edge_list)



# create a list for edges width.

test = nx.get_edge_attributes(G, 'weight')

edge_width = []

for i in nx.edges(G):

    for x in iter(test.keys()):

        if i[0] == x[0] and i[1] == x[1]:

            edge_width.append(test[x])
plt.subplots(figsize=(14,14))

node_scalar = 5000

width_scalar = 10

sizes = [x[1]*node_scalar for x in node_list]

widths = [x*width_scalar for x in edge_width]



#draw the graph

pos = nx.spring_layout(G, k=0.4, iterations=15,seed=1234)



nx.draw(G, pos, with_labels=True, font_size = 8, font_weight = 'bold', 

        node_size = sizes, width = widths,alpha=0.6,edge_color="green")

plt.title("Data Science Tool Relationship Map")
barplot_group('EdLevel',50,8)
barplot_group('UndergradMajor',50,10)
survey_2019['UndergradMajor'].value_counts(dropna=False)
# Check the number of data scientist with social science background.

df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)

                                  & (survey_2019['UndergradMajor']=='A social science (ex. anthropology, psychology, political science)')]

df.shape[0]
# What country do they reside in?

df['Country'].value_counts(dropna=False)
# The percentage of data scientists who reside in US. 

data_scientist = pd.concat([pure_data_scientist,mixed_data_scientist])

print(data_scientist[data_scientist['Country']=='United States'].shape[0]/data_scientist.shape[0])
# Distribution of undergraduate major of data scientist

total_count(data_scientist,'UndergradMajor')
total_count(data_scientist,'EduOther')
total_count(data_scientist,'EdLevel')
# The percentage of data scientists who reside in US grouped by undergraduate major.

df = data_scientist['UndergradMajor'].value_counts(dropna=False).sort_index().reset_index()

df1 = data_scientist[data_scientist['Country']=='United States']['UndergradMajor'].value_counts(dropna=False).sort_index().reset_index()

df2 = pd.merge(df,df1, on='index')

df2['PCT of US'] = df2['UndergradMajor_y']/df2['UndergradMajor_x']

df2
survey_2019['EduOther'].value_counts(dropna=False)
total_count(non_data_scientist,'EduOther')
barplot_group('EduOther',50,10)
df1 = non_data_scientist[non_data_scientist['Age'].notnull()]

df2 = mixed_data_scientist[mixed_data_scientist['Age'].notnull()]

df3 = pure_data_scientist[pure_data_scientist['Age'].notnull()]



# density plot of age among 3 groups

fig, ax = plt.subplots()

fig.set_size_inches(8, 8)

sns.distplot(df1[['Age']], hist=False,color='blue',norm_hist=True)

sns.distplot(df2[['Age']], hist=False,color='red',norm_hist=True)

sns.distplot(df3[['Age']], hist=False,color='green',norm_hist=True)



fig, ax = plt.subplots()

fig.set_size_inches(8, 8)



# density plot of age of pure data scientist between residing in US and non-US.

sns.distplot(df3[df3['Country']=='United States'][['Age']], hist=False,color='green',norm_hist=True)

sns.distplot(df3[df3['Country']!='United States'][['Age']], hist=False,color='red',norm_hist=True)
survey_2019['CurrencySymbol'].value_counts(dropna=False)
# Salaries among 3 groups 





df1 = non_data_scientist[non_data_scientist['ConvertedComp'].notnull()]

df2 = mixed_data_scientist[mixed_data_scientist['ConvertedComp'].notnull()]

df3 = pure_data_scientist[pure_data_scientist['ConvertedComp'].notnull()]

fig, ax = plt.subplots()

fig.set_size_inches(8, 8)

sns.distplot(df1[['ConvertedComp']], hist=False,color='blue',norm_hist=True)

sns.distplot(df2[['ConvertedComp']], hist=False,color='red',norm_hist=True)

sns.distplot(df3[['ConvertedComp']], hist=False,color='green',norm_hist=True)
# Salary vs Age

df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]

df['US'] = df['Country'].apply(lambda x: 1 if x=='United States' else 0)





fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

ax = sns.scatterplot(x="ConvertedComp", y="Age", data=df, hue='US',alpha=0.6)
# Zoom in the compensations less than 500000 USD.

# Just ignore the big salaries, focus on common and reasonable salary.

df1 = df[df["ConvertedComp"]<=500000]



fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

ax = sns.scatterplot(x="ConvertedComp", y="Age", data=df1, hue='US',alpha=0.8)
# Filter the data scientist in the US.



df2 = pd.concat([pure_data_scientist,mixed_data_scientist])

df2 = df2[(df2["ConvertedComp"]<=500000) & (df2["Country"]=='United States')]



fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

ax = sns.scatterplot(x="ConvertedComp", y="Age", data=df2, color="#f28e2b")

ax = sns.regplot(x="ConvertedComp", y="Age", data=df2,color="#f28e2b")
# Just curious whether US data scientist has any dependent.

df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]

df = df[df["Country"]=='United States']



fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

ax = sns.boxplot(x="Dependents",y="Age" , data=df)
# Employment vs salary



fig, ax = plt.subplots()

fig.set_size_inches(20, 8)

ax = sns.boxplot(x="Employment",y="CompTotal" , data=df2)
# OrgSize vs salary



fig, ax = plt.subplots()

fig.set_size_inches(20, 8)

ax = sns.boxplot(x="OrgSize",y="CompTotal" , data=df2)
# df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]

# df = df[(df["Country"]=='United States')]

# df['CompTotal1'] = df.apply(lambda row : min(row['CompTotal'],row['ConvertedComp']),axis=1)



# fig, ax = plt.subplots()

# fig.set_size_inches(20, 20)

# ax = sns.scatterplot(x='CompTotal1', y="Age", data=df)
# df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]

# df = df[(df["Country"]=='United States')]

# df['CompTotal1'] = df.apply(lambda row : min(row['CompTotal'],row['ConvertedComp']),axis=1)

# df = df[df['CompTotal1'] <600000] 



# fig, ax = plt.subplots()

# fig.set_size_inches(20, 20)

# ax = sns.scatterplot(x='CompTotal1', y="Age", data=df)
# get the data first

df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]

df = df[df["Country"]=='United States']

df.shape
# Check the statistics of numeric variables, might remove some outliers.

df.describe()
# Check CompTotal vs ConvertedComp, see which salary makes sense.

fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

ax = sns.scatterplot(x='CompTotal', y="ConvertedComp", data=df)
# if we set CompTotal<700000, will filter out most abnormal salaries.

fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

ax = sns.scatterplot(x='CompTotal', y="ConvertedComp", data=df[df["CompTotal"]<700000])
df['Employment'].value_counts(dropna=False)
# Remove records with missing CompTotal and outliers;

df = df[(df['CompTotal'].notnull()) & (df['CompTotal']<700000) & (df['CompTotal']>0)]



# Exclude the unemployed.

df = df[df['Employment']!='Not employed, but looking for work']



# Drop ConvertedComp, Respondent, Country, DevType, CurrencySymbol,CurrencyDesc

df = df.drop(['ConvertedComp', 'Respondent', 'Country', 'DevType', 'CurrencySymbol','CurrencyDesc'],axis=1)

df.shape
# Retrieve the categorical variables

cat_vars_int = df.select_dtypes(include=['object']).copy().columns

len(cat_vars_int)
# Split and transpose the categorical variable into one-hot columns.

for var in  cat_vars_int:

    # for each cat add dummy var, drop original column

    df = pd.concat([df.drop(var, axis=1), df[var].str.get_dummies(sep=';').rename(lambda x: var+'_' + x, axis='columns')], axis=1)



df.describe()
df.isnull().mean().sort_values(ascending=False)
# Impute missing values with column median

# I choose median because the distributions are not normal

# No needed to impute the missing value for the one-hot columns

df['CodeRevHrs'] = df['CodeRevHrs'].fillna(df['CodeRevHrs'].median())

df['Age'] = df['Age'].fillna(df['Age'].median())

df['WorkWeekHrs'] = df['WorkWeekHrs'].fillna(df['WorkWeekHrs'].median())
# An option reducing features to prevent overfitting?

# df = df.iloc[:, np.where((X.sum() > 10) == True)[0]]

# df.shape
# Plot the distribution of numeric variables, see if any skewed distribution that needs to be normailized. 



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

plt.tight_layout(w_pad=2.0, h_pad=5.0)



plt.subplot(2,2,1)

plt.xlabel('CompTotal')

p1 = sns.distplot(df[['CompTotal']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,2)

plt.xlabel('WorkWeekHrs')

p2 = sns.distplot(df[['WorkWeekHrs']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,3)

plt.xlabel('CodeRevHrs')

p3 = sns.distplot(df[['CodeRevHrs']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,4)

plt.xlabel('Age')

p4 = sns.distplot(df[['Age']], hist=False,color='red',norm_hist=True)
# Use logarithm function to transform the numeric columns

# df['CompTotal_log'] = np.log(df['CompTotal']+100000)

df['WorkWeekHrs_log'] =np.log(df['WorkWeekHrs']+100)

df['CodeRevHrs_log'] = np.log(df['CodeRevHrs']-30)



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

plt.tight_layout(w_pad=2.0, h_pad=5.0)



# plt.subplot(2,2,1)

# plt.xlabel('CompTotal_log')

# p1 = sns.distplot(df[['CompTotal_log']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,2)

plt.xlabel('WorkWeekHrs_log')

p2 = sns.distplot(df[['WorkWeekHrs_log']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,3)

plt.xlabel('CodeRevHrs_log')

p3 = sns.distplot(df[['CodeRevHrs_log']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,4)

plt.xlabel('Age')

p4 = sns.distplot(df[['Age']], hist=False,color='red',norm_hist=True)
# Drop the original numeric columns

df = df.drop(['WorkWeekHrs','CodeRevHrs'],axis=1)
# Split the data into train and test

y = df['CompTotal'].values

X = df.drop(['CompTotal'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
# Xgboost modelling

dtrain = xgb.DMatrix(X_train, label=y_train)

dvalid = xgb.DMatrix(X_test, label=y_test)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
# Set the parameters

# Set the regularization lambda to 100000

# Set the evalutaion metric as rmse (root mean square error)

# Set the early stopping rounds to 5



evals_result = {}





xgb_pars = {'min_child_weight': 5, 'eta':0.5, 'colsample_bytree': 0.8, 

            'max_depth': 10,

'subsample': 0.8, 'lambda': 100000, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

'eval_metric': 'rmse', 'objective': 'reg:linear','seed':1234}



# xgb_pars = {'lambda': 100000, 'booster' : 'gbtree', 

# 'eval_metric': 'rmse', 'objective': 'reg:linear','seed':1234}



model = xgb.train(xgb_pars, dtrain, 10000000, watchlist, early_stopping_rounds=5,

      maximize=False, verbose_eval=1000, evals_result=evals_result)

print('Modeling RMSE %.5f' % model.best_score)
# Model evaluation graph

plt.plot(evals_result['train']['rmse'], linewidth=2, label='Train')

plt.plot(evals_result['valid']['rmse'], linewidth=2, label='Test')

plt.legend(loc='upper right')

plt.title('Model RMSE')

plt.ylabel('rmse')

plt.xlabel('Epoch')

plt.show()
# confusion matrix

dtest = xgb.DMatrix(X_test)

y_pred = model.predict(dtest)



fig, ax = plt.subplots()

fig.set_size_inches(8, 8)



ax = sns.scatterplot(x=y_pred,y=y_test)

plt.title('Residual plot');

plt.xlabel('predicted');

plt.ylabel('actual'); 
# histogram of residual

sns.distplot(y_pred-y_test)
# Display the feature importance.

fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

xgb.plot_importance(model, max_num_features=28, ax=ax)
# Plot WorkWeekHrs vs CompTotal

# Check the tendency

fig, ax = plt.subplots()

fig.set_size_inches(10, 10)



df['WorkWeekHrs'] = np.exp(df['WorkWeekHrs_log'])-100

# ax = sns.boxplot(y='CompTotal', x="MgrMoney_Not sure", data=df[df["CompTotal"]<10000000])

# ax = sns.boxplot(y='CompTotal', x="OrgSize_10,000 or more employees", data=df[df["CompTotal"]<10000000])

ax = sns.scatterplot( x="WorkWeekHrs",y='CompTotal', data=df[df["CompTotal"]<10000000])

ax = sns.regplot(x="WorkWeekHrs", y="CompTotal", data=df[df["CompTotal"]<10000000])
# model = xgb.XGBRegressor(colsample_bytree=0.4,

#                  gamma=0,                 

#                  learning_rate=0.07,

#                  max_depth=3,

#                  min_child_weight=1.5,

#                  n_estimators=10000,                                                                    

#                  reg_alpha=0.75,

#                  reg_lambda=0.45,

#                  subsample=0.6,

#                  seed=42,

#                  verbose=10) 
# model.fit(X_train,y_train)
# predictions = model.predict(X_test)

# # print(explained_variance_score(predictions,y_test))

# from sklearn.metrics import mean_absolute_error

# print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))
# Retrieve the character variables.

# df = survey_2019[survey_2019['Country']=='United States']

df = survey_2019

cat_vars_int = survey_2019.select_dtypes(include=['object']).copy().columns

len(cat_vars_int)

df.shape
# Again split and transpose the categorical variable into one-hot columns.

for var in  cat_vars_int:

    # for each cat add dummy var, drop original column

    df = pd.concat([df.drop(var, axis=1), df[var].str.get_dummies(sep=';').rename(lambda x: var+'_' + x, axis='columns')], axis=1)



df.describe()
df.shape
# Drop Respondent and CompTotal. Comptotal is the salary before curreny conversion 

# It does not provide valuable information unless we focus on one country.

df = df.drop(['Respondent','CompTotal'],axis=1)
# Check the ratio of data scientist and non data scientist

sns.countplot(df['DevType_Data scientist or machine learning specialist'])
# Impute missing values with column median

# The distribution of these variables are not normal

# so I use column median rather than column mean.

# No needed to impute the one-hot columns.

df['CodeRevHrs'] = df['CodeRevHrs'].fillna(df['CodeRevHrs'].median())

df['Age'] = df['Age'].fillna(df['Age'].median())

df['WorkWeekHrs'] = df['WorkWeekHrs'].fillna(df['WorkWeekHrs'].median())

df['ConvertedComp'] = df['ConvertedComp'].fillna(df['ConvertedComp'].median())
# See if the numeric variables need to be normalized.

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

plt.tight_layout(w_pad=2.0, h_pad=5.0)



plt.subplot(2,2,1)

plt.xlabel('ConvertedComp')

p1 = sns.distplot(df[['ConvertedComp']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,2)

plt.xlabel('WorkWeekHrs')

p2 = sns.distplot(df[['WorkWeekHrs']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,3)

plt.xlabel('CodeRevHrs')

p3 = sns.distplot(df[['CodeRevHrs']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,4)

plt.xlabel('Age')

p4 = sns.distplot(df[['Age']], hist=False,color='red',norm_hist=True)
# Use logarithm function to transform the numeric columns

df['ConvertedComp_log'] = np.log(df['ConvertedComp']-90000)

df['WorkWeekHrs_log'] = np.log(df['WorkWeekHrs']-70)

df['CodeRevHrs_log'] = np.log(df['CodeRevHrs']-30)

df['Age_log'] = np.log(df['Age']+30)





fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

plt.tight_layout(w_pad=2.0, h_pad=5.0)



plt.subplot(2,2,1)

plt.xlabel('ConvertedComp_log')

p1 = sns.distplot(df[['ConvertedComp_log']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,2)

plt.xlabel('WorkWeekHrs_log')

p2 = sns.distplot(df[['WorkWeekHrs_log']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,3)

plt.xlabel('CodeRevHrs_log')

p3 = sns.distplot(df[['CodeRevHrs_log']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,4)

plt.xlabel('Age_log')

p4 = sns.distplot(df[['Age_log']], hist=False,color='red',norm_hist=True)
# Split data set into response vector and feature matrix.

y = df['DevType_Data scientist or machine learning specialist'].values

X = df.drop(['DevType_Data scientist or machine learning specialist'], axis=1)
# Drop original numeric columns. 

# We already know that researcher and data engineer might share the role of data scientist

# Drop DevType as it does not provide the information why data scientist is distinct. 



X = X.drop([col for col in X.columns if 'DevType_' in col],axis=1)

X = X.drop(['CodeRevHrs','ConvertedComp','WorkWeekHrs','Age'],axis=1)

X.shape
# Split X y into training set and test set.



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0,stratify=y)
# Define an Xgboost classifer

# Using AUPRC as the evaluation metric which is more sensitive to the minor class 

# As we know the population of data scientist is just 1/8 of other developers



model = xgb.XGBClassifier(

    learning_rate =0.1, n_estimators=1000,

    gamma=0,

    subsample=0.8,

    colsample_bytree=0.8,

    objective= 'binary:logistic', 

    nthread=4,

    scale_pos_weight=7,

    seed=27,

    max_depth = 5,

    min_child_weight = 5

)



def evalauc(preds, dtrain):

    labels = dtrain.get_label()

    precision, recall, thresholds = precision_recall_curve(labels, preds)

    area = auc(recall, precision)

    return 'AUPRC', -area





model.fit(X_train, y_train,

          eval_metric=evalauc,

          eval_set=[(X_train, y_train), (X_test, y_test)],

          early_stopping_rounds=5,

         verbose=True)
# See the prediction result

predict = model.predict(X_test)

print(classification_report(y_test, predict))

print(confusion_matrix(y_test,predict))

print("Accuracy: ")

print(accuracy_score(y_test,predict))
fpr, tpr, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])

auprc = average_precision_score(y_test, predict)



plt.plot(fpr, tpr, lw=1, label='AUPRC = %0.2f'%(auprc))

plt.plot([0, 1], [0, 1], '--k', lw=1)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('XGBOOST AUPRC')

plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

roc_auc = roc_auc_score(y_test, predict)



plt.plot(fpr, tpr, lw=1, label='AUC = %0.2f'%(roc_auc))

plt.plot([0, 1], [0, 1], '--k', lw=1)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('XGBOOST ROC')

plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
feature_name=X.columns.tolist()

#feature_name.remove('DevType_Data scientist or machine learning specialist')

dtrain = xgb.DMatrix(X, label=y,feature_names=feature_name)
# model.get_booster().get_score().items()
# mapper = {'f{0}'.format(i): v for i, v in enumerate(dtrain.feature_names)}

# mapped = { mapper[k]: v for k, v in model.get_booster().get_score().items()}



fig,ax  =  plt.subplots (figsize=(10, 5))

xgb.plot_importance(model, max_num_features=20,ax=ax)

plt.show()


df.groupby(['Employment_Not employed, and not looking for work', 'DevType_Data scientist or machine learning specialist']).size()

# Retrieve the character variables.

# df = survey_2019[survey_2019['Country']=='United States']

df = pd.concat([pure_data_scientist,non_data_scientist])

cat_vars_int = df.select_dtypes(include=['object']).copy().columns

len(cat_vars_int)

df.shape
# Again split and transpose the categorical variable into one-hot columns.

for var in  cat_vars_int:

    # for each cat add dummy var, drop original column

    df = pd.concat([df.drop(var, axis=1), df[var].str.get_dummies(sep=';').rename(lambda x: var+'_' + x, axis='columns')], axis=1)



df.describe()
df.shape
sns.countplot(df['DevType_Data scientist or machine learning specialist'])
np.sum(df['DevType_Data scientist or machine learning specialist'])
# Impute missing values with column median

df['CodeRevHrs'] = df['CodeRevHrs'].fillna(df['CodeRevHrs'].median())

df['Age'] = df['Age'].fillna(df['Age'].median())

df['WorkWeekHrs'] = df['WorkWeekHrs'].fillna(df['WorkWeekHrs'].median())

df['ConvertedComp'] = df['ConvertedComp'].fillna(df['ConvertedComp'].median())
# See if the numeric variables need to be normalized.

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

plt.tight_layout(w_pad=2.0, h_pad=5.0)



plt.subplot(2,2,1)

plt.xlabel('ConvertedComp')

p1 = sns.distplot(df[['ConvertedComp']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,2)

plt.xlabel('WorkWeekHrs')

p2 = sns.distplot(df[['WorkWeekHrs']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,3)

plt.xlabel('CodeRevHrs')

p3 = sns.distplot(df[['CodeRevHrs']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,4)

plt.xlabel('Age')

p4 = sns.distplot(df[['Age']], hist=False,color='red',norm_hist=True)
# Use logarithm function to transform the numeric columns

df['ConvertedComp_log'] = np.log(df['ConvertedComp']+1000)

df['WorkWeekHrs_log'] = np.log(df['WorkWeekHrs']-70)

df['CodeRevHrs_log'] = np.log(df['CodeRevHrs']-30)

df['Age_log'] = np.log(df['Age']+30)





fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

plt.tight_layout(w_pad=2.0, h_pad=5.0)



plt.subplot(2,2,1)

plt.xlabel('ConvertedComp_log')

p1 = sns.distplot(df[['ConvertedComp_log']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,2)

plt.xlabel('WorkWeekHrs_log')

p2 = sns.distplot(df[['WorkWeekHrs_log']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,3)

plt.xlabel('CodeRevHrs_log')

p3 = sns.distplot(df[['CodeRevHrs_log']], hist=False,color='red',norm_hist=True)



plt.subplot(2,2,4)

plt.xlabel('Age_log')

p4 = sns.distplot(df[['Age_log']], hist=False,color='red',norm_hist=True)
# Split data set into response vector and feature matrix.

y = df['DevType_Data scientist or machine learning specialist'].values

X = df.drop(['DevType_Data scientist or machine learning specialist'], axis=1)
X = X.drop(['Respondent','CompTotal'],axis=1)

X = X.drop([col for col in X.columns if 'DevType_' in col],axis=1)

X = X.drop(['CodeRevHrs','ConvertedComp','WorkWeekHrs','Age'],axis=1)

X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0,stratify=y)
model = xgb.XGBClassifier(

    learning_rate =0.01, n_estimators=1000,

    gamma=0,

    subsample=0.8,

    colsample_bytree=0.8,

    objective= 'binary:logistic', 

    nthread=4,

    scale_pos_weight=100,

    seed=27,

    max_depth = 5,

    min_child_weight = 3

)



def evalauc(preds, dtrain):

    labels = dtrain.get_label()

    precision, recall, thresholds = precision_recall_curve(labels, preds)

    area = auc(recall, precision)

    return 'AUPRC', -area





model.fit(X_train, y_train,

          eval_metric=evalauc,

          eval_set=[(X_train, y_train), (X_test, y_test)],

          early_stopping_rounds=5,

         verbose=True)
# See the prediction result

predict = model.predict(X_test)

print(classification_report(y_test, predict))

print(confusion_matrix(y_test,predict))

print("Accuracy: ")

print(accuracy_score(y_test,predict))
fpr, tpr, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])

auprc = average_precision_score(y_test, predict)



plt.plot(fpr, tpr, lw=1, label='AUPRC = %0.2f'%(auprc))

plt.plot([0, 1], [0, 1], '--k', lw=1)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('XGBOOST AUPRC')

plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
fig,ax  =  plt.subplots (figsize=(10, 5))

xgb.plot_importance(model, max_num_features=20,ax=ax)

plt.show()
df.groupby(['DevType_Data scientist or machine learning specialist','MiscTechWorkedWith_TensorFlow']).size()