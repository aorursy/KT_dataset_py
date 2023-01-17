# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import time

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data.csv')
df.head()  # head method show only first 5 rows
# feature names as a list

col = df.columns       # .columns gives columns names in data 

print(col)
# y includes our labels and x includes our features

y = df.diagnosis                          # M or B 
y.describe()
ax = sns.countplot(y,label="Count")       # M = 212, B = 357

B, M = y.value_counts()

print('Number of Benign: ',B)

print('Number of Malignant : ',M)
list = ['Unnamed: 32','id','diagnosis']

x = df.drop(list,axis = 1 )

x.head()
# find the percentage of missing value

missing_values = x.isnull().sum()/len(x)*100

#sorted the missing columns based on descending order

missing_values[missing_values>0].sort_values(ascending = False)
# Saving missing values in a variable

a = x.isnull().sum()/len(x)*100



# saving column names in a variable

variables = x.columns



variable = []

for i in range(0,len(x.columns)):

    if a[i]>=60:   #setting the threshold as 60%

        variable.append(variables[i])



        

variable
#Let’s first impute the missing values in the train set using the mode value of the known observations.

# Known that we dont have any missing values in the columns

columns = x.columns



for col in columns:

    x[col].fillna(x[col].mode()[0],inplace = True)
x.isnull().sum()
x.var().sort_values(ascending=True)
#remove the very less variant variables, since they dont carry much information



numeric = x.select_dtypes(include=[np.number])

var = numeric.var()

numeric = numeric.columns

variable = []

for i in range(0,len(var)):

    if var[i]>=5:   #setting the threshold as 30%

        variable.append(numeric[i])



var
variable
x_less_var = x[variable]

x_less_var.head()
x.corr()
# less variance removed and checked the correlation on the rest of the items 

x[variable].corr(method ='kendall')
f , ax = plt.subplots(figsize = (14,12))

plt.title('Correlation of Features- HeatMap',y=1,size=16)

sns.heatmap(x[variable].corr(),square = True,  vmax=0.8)
# Create correlation matrix

corr_matrix = x[variable].corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper
# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(to_drop)
# Drop features 

x_low_corr = x[variable].drop(x[variable][to_drop], axis=1)

#final correlation table

x_low_corr.corr()
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42,max_depth=8)

x_rfc = pd.get_dummies(x)

rfc.fit(x_rfc,y)
features = x.columns

importances = rfc.feature_importances_

indices = np.argsort(importances[0:20])  # top 5 features

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
from sklearn.feature_selection import SelectFromModel



feature = SelectFromModel(rfc)



Fit = feature.fit_transform(x, y)



Fit.shape
#load the data again

df = pd.read_csv('../input/data.csv')

y = df.diagnosis

y.describe()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)
# Define dictionary to store our rankings

from sklearn.preprocessing import MinMaxScaler

ranks = {}



# Create our function which stores the feature rankings to the ranks dictionary

def ranking(ranks, names, order=1):

    minmax = MinMaxScaler()

    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]

    ranks = map(lambda x: round(x,2), ranks)

    return dict(zip(names, ranks))



# Construct our Linear Regression model

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE



lr = LogisticRegression()

lr.fit(x, y_encoded)



#stop the search when only the last feature is left

rfe = RFE(lr, n_features_to_select=1,verbose=3)

rfe.fit(x, y_encoded)





rfe.ranking_
ranking_list=[]

for rank in map(float, rfe.ranking_):

    ranking_list.append(rank)
ranks["RFE"] = ranking(ranking_list, x.columns, order=-1)

#ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), x.columns, order=-1)

# Create empty dictionary to store the mean value calculated from all the scores

r = {}

for name in x.columns:

    r[name] = round(np.mean([ranks[method][name] 

                             for method in ranks.keys()]), 2)
r
zip(r.keys(), r.values())
# Put the mean scores into a Pandas dataframe

meanplot = pd.DataFrame(data=[(k, v) for (k, v) in r.items()], columns= ['Feature','Mean Ranking'])

# Sort the dataframe

meanplot = meanplot.sort_values('Mean Ranking', ascending=False)

# Let's plot the ranking of the features

sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar",size=8, aspect=0.75, palette='coolwarm')
# first ten features

data_dia = y

data = x

data_n_2 = (data - data.mean()) / (data.std())              # standardization

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=90)
# Second ten features

data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=90)
# Second ten features

data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=90)
# As an alternative of violin plot, box plot can be used

# box plots are also useful in terms of seeing outliers

# I do not visualize all features with box plot

# In order to show you lets have an example of box plot

# If you want, you can visualize other features as well.

plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)
sns.jointplot(x.loc[:,'concavity_worst'], x.loc[:,'concave points_worst'], kind="regg", color="#ce1414")
sns.set(style="white")

df = x.loc[:,['radius_worst','perimeter_worst','area_worst']]

g = sns.PairGrid(df, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot, lw=3)
sns.set(style="whitegrid", palette="muted")

data_dia = y

data = x

data_n_2 = (data - data.mean()) / (data.std())              # standardization

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

tic = time.time()

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)



plt.xticks(rotation=90)
data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)
data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

toc = time.time()

plt.xticks(rotation=90)

print("swarm plot time: ", toc-tic ," s")
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

x_1 = x.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 

x_1.head()



    
#correlation map

f,ax = plt.subplots(figsize=(14, 14))

sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score



# split data train 70 % and test 30 %

x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=42)



#random forest classifier with n_estimators=10 (default)

clf_rf = RandomForestClassifier(random_state=43)      

clr_rf = clf_rf.fit(x_train,y_train)



ac = accuracy_score(y_test,clf_rf.predict(x_test))

print('Accuracy is: ',ac)

cm = confusion_matrix(y_test,clf_rf.predict(x_test))

sns.heatmap(cm,annot=True,fmt="d")
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

# find best scored 5 features

select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)
print('Score list:', select_feature.scores_)

print('Feature list:', x_train.columns)
x_train_2 = select_feature.transform(x_train)

x_test_2 = select_feature.transform(x_test)

#random forest classifier with n_estimators=10 (default)

clf_rf_2 = RandomForestClassifier()      

clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)

ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))

print('Accuracy is: ',ac_2)

cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test_2))

sns.heatmap(cm_2,annot=True,fmt="d")
from sklearn.feature_selection import RFE

# Create the RFE object and rank each pixel

clf_rf_3 = RandomForestClassifier()      

rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)

rfe = rfe.fit(x_train, y_train)

print('Chosen best 5 feature by rfe:',x_train.columns[rfe.support_])
from sklearn.feature_selection import RFECV



# The "accuracy" scoring is proportional to the number of correct classifications

clf_rf_4 = RandomForestClassifier() 

rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation

rfecv = rfecv.fit(x_train, y_train)



print('Optimal number of features :', rfecv.n_features_)

print('Best features :', x_train.columns[rfecv.support_])
# Plot number of features VS. cross-validation scores

import matplotlib.pyplot as plt

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score of number of selected features")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
clf_rf_5 = RandomForestClassifier()      

clr_rf_5 = clf_rf_5.fit(x_train,y_train)

importances = clr_rf_5.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(x_train.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest



plt.figure(1, figsize=(14, 13))

plt.title("Feature importances")

plt.bar(range(x_train.shape[1]), importances[indices],

       color="g", yerr=std[indices], align="center")

plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)

plt.xlim([-1, x_train.shape[1]])

plt.show()
# split data train 70 % and test 30 %

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#normalization

x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())

x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())



from sklearn.decomposition import PCA

pca = PCA()

pca.fit(x_train_N)



plt.figure(1, figsize=(14, 13))

plt.clf()

plt.axes([.2, .2, .7, .7])

plt.plot(pca.explained_variance_ratio_, linewidth=2)

plt.axis('tight')

plt.xlabel('n_components')

plt.ylabel('explained_variance_ratio_')