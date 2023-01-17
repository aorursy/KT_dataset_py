# Kütühaneleri yükleme

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

#

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot 

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import KFold 

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score

#

from sklearn.metrics import classification_report 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

#

from sklearn.linear_model import LogisticRegression 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB 

from sklearn.svm import SVC

#

from sklearn.linear_model import Lasso 

from sklearn.linear_model import ElasticNet 

from sklearn.model_selection import GridSearchCV 

from sklearn.linear_model import LinearRegression 

from sklearn.tree import DecisionTreeRegressor 

from sklearn.neighbors import KNeighborsRegressor 

from sklearn.svm import SVR 

#

from sklearn.ensemble import RandomForestRegressor 

from sklearn.ensemble import GradientBoostingRegressor 

from sklearn.ensemble import ExtraTreesRegressor 

from sklearn.ensemble import AdaBoostRegressor 

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline 

from sklearn.linear_model import LogisticRegression 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

from sklearn.naive_bayes import GaussianNB 

from sklearn.svm import SVC 

from sklearn.ensemble import AdaBoostClassifier 

from sklearn.ensemble import GradientBoostingClassifier 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.ensemble import ExtraTreesClassifier

import warnings

warnings.filterwarnings("ignore")





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")

data.head()
# regions

data.region.unique()
# regions of counts

print(data["region"].value_counts(dropna=False))
# data shape:

row, columns = data.shape

print("Data Row:", row)

print("Data Columns:", columns)
# column names:

data.columns
# descriptions 

display(data.describe().T)
# class distribution 

print("Data is not balanced:",data.groupby('type').size())
# Dataset Correlation

data.corr()
data.isnull().sum()
f,ax = plt.subplots(figsize = (10,7))

ax = sns.countplot(x=data.type,label="Count",palette="viridis")

plt.xlabel('Type of Avocado',fontsize = 15,color='blue')

plt.ylabel('Count',fontsize = 15,color='blue')

plt.title('Avocado',fontsize = 20,color='blue')

#total = len(data['year'])

# how to show counts:

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()+75))
f,ax = plt.subplots(figsize = (10,7))

sns.boxplot(x="type", y="AveragePrice",data=data,palette="viridis");

plt.title("Compare Average Prices & Observe Outliers",fontsize = 20,color='blue')

plt.xlabel('Type of Avocado',fontsize = 15,color='blue')

plt.ylabel('Average Price',fontsize = 15,color='blue')
f,ax = plt.subplots(figsize = (10,7))

sns.boxplot(x="year", y="AveragePrice",hue="type",data=data,palette="viridis");

plt.title("Compare Average Prices of Years & Observe Outliers",fontsize = 20,color='blue')

plt.xlabel('Years',fontsize = 15,color='blue')

plt.ylabel('Average Price',fontsize = 15,color='blue')
f,ax = plt.subplots(figsize = (10,7))

sns.violinplot(x="year", y="AveragePrice", hue="type", data=data,split=True, inner="quart",palette="viridis")

plt.xticks(rotation=90)

plt.title("Compare Average Prices of Years",fontsize = 20,color='blue')

plt.xlabel('Years',fontsize = 15,color='blue')

plt.ylabel('Average Price',fontsize = 15,color='blue')
f,ax = plt.subplots(figsize = (10,7))

sns.barplot(x="year", y="AveragePrice",hue="type",data=data,palette="viridis")

plt.tight_layout() # grafikler daha düzgün gözükecek

plt.title("Compare Average Prices of Years",fontsize = 20,color='blue')

plt.xlabel('Years',fontsize = 15,color='blue')

plt.ylabel('Average Price',fontsize = 15,color='blue')
f,ax = plt.subplots(figsize = (10,7))

#data.drop("Unnamed: 0",axis=1,inplace=True)

sns.heatmap(data.corr(), annot=True,cmap = 'Greens', linewidths=0.5,linecolor="black", fmt= '.2f',ax=ax)
# Split Dataset, "conventional & organic"

data_con = data[data["type"] == "conventional"]

data_org = data[data["type"] == "organic"]
f,ax = plt.subplots(figsize = (12,7))

plt.subplot(2,1,1) # ikiye birlik düzlemde ilk grafik

sns.distplot(data_con.AveragePrice,color="green",label="Average Price");

plt.title("Average Price of Conventional",fontsize = 20,color='blue')

plt.xlabel('Average Price',fontsize = 15,color='blue')

plt.legend()

plt.grid()

#

plt.subplot(2,1,2)

sns.distplot(data_org.AveragePrice,color="darkblue",label="Average Price");

plt.title("Average Price of Organic",fontsize = 20,color='blue')

plt.xlabel('Average Price',fontsize = 15,color='blue')

plt.tight_layout() # grafikler daha düzgün gözükecek

plt.legend()

plt.grid()
# Avocado Average Price

f,ax = plt.subplots(figsize = (17,9))

sns.barplot(x="region", y="AveragePrice",hue="type",data=data,palette="viridis")

plt.xticks(rotation=90)

plt.tight_layout()
#conda install -c conda-forge wordcloud

#from wordcloud import WordCloud 

# how many times using regions in dataset

from wordcloud import WordCloud 

data_region = data.region

plt.subplots(figsize=(10,10))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate("*".join(data_region))

plt.imshow(wordcloud)

plt.title("Wordcloud for Regions")

plt.axis('off')
# Sum(ounce) of Avocados for per year

data_2015 = data[data.year==2015]

sum_2015 = data_2015["Total Volume"].sum()

data_2016 = data[data.year==2016]

sum_2016 = data_2016["Total Volume"].sum()

data_2017 = data[data.year==2017]

sum_2017 = data_2017["Total Volume"].sum()

data_2018 = data[data.year==2018]

sum_2018 = data_2018["Total Volume"].sum()

#**********************************************************************************

# Sum(ounce) of Avocados for per year in conventional

data_con_2015 = data_con[data.year==2015]

sum_con_2015 = data_con_2015["Total Volume"].sum()

data_con_2016 = data_con[data.year==2016]

sum_con_2016 = data_con_2016["Total Volume"].sum()

data_con_2017 = data_con[data.year==2017]

sum_con_2017 = data_con_2017["Total Volume"].sum()

data_con_2018 = data_con[data.year==2018]

sum_con_2018 = data_con_2018["Total Volume"].sum()

#**********************************************************************************

# Sum(ounce) of Avocados for per year in organic

data_org_2015 = data_org[data.year==2015]

sum_org_2015 = data_org_2015["Total Volume"].sum()

data_org_2016 = data_org[data.year==2016]

sum_org_2016 = data_org_2016["Total Volume"].sum()

data_org_2017 = data_org[data.year==2017]

sum_org_2017 = data_org_2017["Total Volume"].sum()

data_org_2018 = data_org[data.year==2018]

sum_org_2018 = data_org_2018["Total Volume"].sum()



labels = data.year.value_counts().index

colors = ['grey','blue','red','yellow']

fracs = [15, 30, 45, 10]

sizes_1 = [sum_con_2015,sum_con_2016,sum_con_2017,sum_con_2018]#for con

fig = plt.figure(figsize = (9,9))

#

sizes_2 = [sum_org_2015,sum_org_2016,sum_org_2017,sum_org_2018]#for org

ax1 = fig.add_axes([0, -0.1, .5, .5], aspect=1)

ax1.pie(sizes_1, labels=labels, radius = 1.2,colors=colors,autopct='%1.2f%%')

#

ax2 = fig.add_axes([0.7, -0.1, .5, .5], aspect=1)

ax2.pie(sizes_2, labels=labels, radius = 1.2,colors=colors,autopct='%1.2f%%')

#

sizes_0 = [sum_2015,sum_2016,sum_2017,sum_2018]

ax3 = fig.add_axes([.35, 0, .5, 1.5], aspect=1)

ax3.pie(sizes_0, labels=labels, radius = 1.2,colors=colors,autopct='%1.2f%%')

#

ax1.set_title('Avocado Consumption in Conventional',color = 'blue',fontsize = 15)

ax2.set_title('Avocado Consumption in Organic',color = 'blue',fontsize = 15)

ax3.set_title('Avocado Consumption ',color = 'blue',fontsize = 15)

plt.show()

plt.tight_layout() # grafikler daha düzgün gözükecek
data.head()
# split date: day,month,year 

liste = []

for date in data.Date:

    liste.append(date.split("-"))

    

# month and day adding to lists

month = []

day = []

for i in range(len(liste)):

    month.append(liste[i][1])

    day.append(liste[i][2])

    

# adding to dataset

data["month"] = month

data["day"] = day



# delete old date column

data.drop(["Date"],axis=1,inplace=True)



#convert objects to int

data.month = data.month.values.astype(int)

data.day = data.day.values.astype(int)
# drop unnecessary features

data.drop(["Unnamed: 0","region"],axis=1,inplace=True)
# find dummy variables

data["type"] = pd.get_dummies(data.type,drop_first=True)
data.head()
# Y

y = data[["type"]][:]
# X

x = data.drop(["type"],axis=1,inplace=True)

x = data.iloc[:,:]
# Scale the data to be between -1 and 1

sc = StandardScaler()

x = sc.fit_transform(x)
# Creating Train and Test Datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# Model List

models = [] 

models.append(('LR', LogisticRegression())) 

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVM', SVC())) 

models.append(('NB', GaussianNB()))

models.append(('DTC', DecisionTreeClassifier()))
# evaluate models using cross validation score:

results = [] 

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=42) 

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy') 

    results.append(cv_results) 

    names.append(name) 

    print("Model Name:{} Model Acc:{:.3f} Model Std:{:.3f}".format(name, cv_results.mean(), cv_results.std()))
# Compare Model's Acc

f,ax = plt.subplots(figsize = (10,7))

sns.boxplot(x=names, y=results,palette="viridis");

plt.title("Compare Model's Accuracies",fontsize = 20,color='blue')

plt.xlabel('Models',fontsize = 15,color='blue')

plt.ylabel('Accuracies',fontsize = 15,color='blue')
# Tuning Decision Tree Model

criterions = ["gini","entropy"]

param_grid = dict(criterion=criterions) 
dtc = DecisionTreeClassifier()

gs = GridSearchCV(estimator=dtc,param_grid=param_grid,scoring="accuracy", cv=10)

grid_search = gs.fit(x_train,y_train)

best_score = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Score:",best_score)

print("Best Parameters:",best_parameters)
# Time to use DFC for dataset:

dtc = DecisionTreeClassifier(criterion="entropy")

dtc.fit(x_train,y_train)

y_pred = dtc.predict(x_test)
#confussion matrix: 

from sklearn import metrics

dtc_cm = confusion_matrix(y_test,y_pred)

dtc_cross = pd.crosstab(y_test["type"], y_pred,rownames=['Actual Values'], colnames=['Predicted Values'])

dtc_acc = metrics.accuracy_score(y_test, y_pred)

print(dtc_cross)

print(dtc_acc)
# Feature Importance

#print(rfc.feature_importances_)

# You can see that we are given an importance score 

# for each attribute where the larger the score, 

# the more important the attribute.

feature = pd.DataFrame(data=[dtc.feature_importances_], columns=data.columns)

feature.head()
# Classiﬁcation Report

from sklearn.metrics import classification_report 

report = classification_report(y_test, y_pred) 

print(report)
# ROC Eğrisi:

import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification

probs = dtc.predict_proba(x_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# ensembles 

ensembles = [] 

ensembles.append(('ABC', AdaBoostClassifier()))

ensembles.append(('GBC', GradientBoostingClassifier()))

ensembles.append(('RFC', RandomForestClassifier()))

ensembles.append(('ETC', ExtraTreesClassifier()))
# evaluate models using cross validatiob score:

results_ensemble = [] 

names_ensemble = []

for name, model in ensembles:

    kfold = KFold(n_splits=10, random_state=42) 

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy') 

    results_ensemble.append(cv_results) 

    names_ensemble.append(name) 

    print("Model Name:{} Model Acc:{:.3f} Model Std:{:.3f}".format(name, cv_results.mean(), cv_results.std()))
# Compare Model's Acc

f,ax = plt.subplots(figsize = (10,7))

sns.boxplot(x=names_ensemble, y=results_ensemble,palette="viridis");

plt.title("Compare Ensemble Model's Accuracies",fontsize = 20,color='blue')

plt.xlabel('Models',fontsize = 15,color='blue')

plt.ylabel('Accuracies',fontsize = 15,color='blue')
# Tuning Extra Trees Class. Model

estimators = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29] 

criterions = ["gini","entropy"]

param_grid = dict(n_estimators=estimators,criterion=criterions) 
etc = ExtraTreesClassifier()

gs = GridSearchCV(estimator=etc,param_grid=param_grid,scoring="accuracy", cv=10)

grid_search = gs.fit(x_train,y_train)

best_score = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Score:",best_score)

print("Best Parameters:",best_parameters)
# Time to use ETC for dataset:

etc = ExtraTreesClassifier(n_estimators=27,criterion="gini")

etc.fit(x_train,y_train)

y_pred = etc.predict(x_test)
#confussion matrix: 

from sklearn import metrics

etc_cm = confusion_matrix(y_test,y_pred)

etc_cross = pd.crosstab(y_test["type"], y_pred,rownames=['Actual Values'], colnames=['Predicted Values'])

etc_acc = metrics.accuracy_score(y_test, y_pred)

print(etc_cross)

print(etc_acc)
# Classiﬁcation Report

from sklearn.metrics import classification_report 

report = classification_report(y_test, y_pred,digits=3) 

print(report)
#model

from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(x_train,y_train)

y_pred = xgb.predict(x_test)
#confussion matrix: 

from sklearn import metrics

xgb_cm = confusion_matrix(y_test,y_pred)

xgb_cross = pd.crosstab(y_test["type"], y_pred,rownames=['Actual Values'], colnames=['Predicted Values'])

xgb_acc = metrics.accuracy_score(y_test, y_pred)

print(xgb_cross)

print(xgb_acc)
# Tuning XGBOOST Model

learning_rates = [0.1,0.01,0.001] 

liste = list(range(250))

estimators = liste

gammas = [1,0.5,0.1,0.01,0.001,0]

boosters = ["gbtree","gblinear","dart"]

param_grid = dict(n_estimators=estimators,learning_rate=learning_rates,gamma=gammas,booster=boosters) 
# Tuning is taking time...

#xgb = XGBClassifier()

#gs = GridSearchCV(estimator=xgb,param_grid=param_grid,scoring="accuracy", cv=10)

#grid_search = gs.fit(x_train,y_train)

#best_score = grid_search.best_score_

#best_parameters = grid_search.best_params_

#print("Best Score:",best_score)

#print("Best Parameters:",best_parameters)
from sklearn.neural_network import MLPClassifier

mlpc = MLPClassifier(verbose=False)

mlpc.fit(x_train, y_train)    

#mlpc.max_iter 

#mlpc.hidden_layer_sizes#node sayısı 

y_pred = mlpc.predict(x_test)
mlpc.get_params()
#confussion matrix: 

from sklearn import metrics

mlpc_cm = confusion_matrix(y_test,y_pred)

mlpc_cross = pd.crosstab(y_test["type"], y_pred,rownames=['Actual Values'], colnames=['Predicted Values'])

mlpc_acc = metrics.accuracy_score(y_test, y_pred)

print(mlpc_cross)

print(mlpc_acc)
models = ["dtc","etc","xgb","nn"]

values = [0.986,0.997,0.991,0.982]
# Compare Model's Acc

f,ax = plt.subplots(figsize = (10,7))

sns.barplot(x=models, y=values,palette="viridis");

plt.title("Compare Ensemble Model's Accuracies",fontsize = 20,color='blue')

plt.xlabel('Models',fontsize = 15,color='blue')

plt.ylabel('Accuracies',fontsize = 15,color='blue')
data = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")

data.head()
# Split Dataset, "conventional & organic"

data_con = data[data["type"] == "conventional"]

data_org = data[data["type"] == "organic"]
f,ax = plt.subplots(figsize = (20,7))

data_con = data_con.sort_values("Date")

plt.plot(data_con['Date'], data_con['AveragePrice'])
f,ax = plt.subplots(figsize = (20,7))

data_org = data_org.sort_values("Date")

plt.plot(data_org['Date'], data_org['AveragePrice'])
# split date: day,month,year 

liste = []

for date in data.Date:

    liste.append(date.split("-"))

    

# month and day adding to lists

month = []

day = []

for i in range(len(liste)):

    month.append(liste[i][1])

    day.append(liste[i][2])

    

# adding to dataset

data["month"] = month

data["day"] = day



# delete old date column

data.drop(["Date"],axis=1,inplace=True)



#convert objects to int

data.month = data.month.values.astype(int)

data.day = data.day.values.astype(int)
# drop unnecessary features

data.drop(["Unnamed: 0"],axis=1,inplace=True)
data.head()
# Split Dataset, "conventional & organic"

data_con = data[data["type"] == "conventional"]

data_org = data[data["type"] == "organic"]
# find dummy variables

data_con = pd.get_dummies(data_con,drop_first=True)

data_org = pd.get_dummies(data_org,drop_first=True)
data_con.head()
data_org.head()
# For Conventional

import statsmodels.api as sm

exog_con = data_con.iloc[:,1:].values

endog_con = data_con.iloc[:,[0]].values

r_ols_con = sm.OLS(endog_con,exog_con) #bağımlı değişken, X_l:bağımsız değişkenlerimiz.

r_con = r_ols_con.fit()

print(r_con.summary())
# For Organic

import statsmodels.api as sm

exog_org = data_org.iloc[:,1:].values

endog_org = data_org.iloc[:,[0]].values

r_ols_org = sm.OLS(endog_org,exog_org) #bağımlı değişken, X_l:bağımsız değişkenlerimiz.

r_org = r_ols_org.fit()

print(r_org.summary())
# Y

y = data_con[["AveragePrice"]][:]
# X

x = data_con.drop(["AveragePrice"],axis=1,inplace=True)

x = data_con.iloc[:,1:]
# Scale the data to be between -1 and 1

sc_x = StandardScaler()

sc_y = StandardScaler()

x = sc_x.fit_transform(x)

y = sc_y.fit_transform(y)
# Creating Train and Test Datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
classic_models = [] 

classic_models.append(('LR', LinearRegression())) 

classic_models.append(('LASSO', Lasso())) 

classic_models.append(('EN', ElasticNet())) 

classic_models.append(('KNN', KNeighborsRegressor())) 

classic_models.append(('DTR', DecisionTreeRegressor())) 

classic_models.append(('SVR', SVR()))
# evaluate models using cross validation score:

classic_results = [] 

classic_names = []

for name, model in classic_models:

    kfold = KFold(n_splits=10, random_state=42) 

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2') 

    classic_results.append(cv_results) 

    classic_names.append(name) 

    print("Model Name:{} Model Score:{:.3f} Model Std:{:.3f}".format(name, cv_results.mean(), cv_results.std()))
# Compare Model's Scores

f,ax = plt.subplots(figsize = (10,7))

sns.boxplot(x=classic_names, y=classic_results,palette="viridis");

plt.title("Compare Model's Scores",fontsize = 20,color='blue')

plt.xlabel('Models',fontsize = 15,color='blue')

plt.ylabel('Scores',fontsize = 15,color='blue')
# Tuning Decision Tree Model

criterions = ["mse","mae"]

param_grid = dict(criterion=criterions) 
dtr = DecisionTreeRegressor()

gs = GridSearchCV(estimator=dtr,param_grid=param_grid,scoring="r2", cv=kfold)

grid_search = gs.fit(x_train,y_train)

best_score = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Score:",best_score)

print("Best Parameters:",best_parameters)
means = grid_search.cv_results_['mean_test_score'] 

stds = grid_search.cv_results_['std_test_score'] 

params = grid_search.cv_results_['params'] 

for mean, stdev, param in zip(means, stds, params): 

    print("%f (%f) with: %r" % (mean, stdev, param))
# Time to use DTR for dataset:

dtr = DecisionTreeRegressor(criterion="mse")

dtr.fit(x_train,y_train)

y_pred = dtr.predict(x_test)
result_DFR = r2_score(y_test, y_pred)

print("{:.2f}".format(result_DFR))
results_models = []

results_models.append(result_DFR)
# ensembles 

ensembles = [] 

ensembles.append(('ABR', AdaBoostRegressor()))

ensembles.append(('GBR', GradientBoostingRegressor()))

ensembles.append(('RFR', RandomForestRegressor()))

ensembles.append(('ETR', ExtraTreesRegressor()))
# evaluate models using cross validatiob score:

results_ensemble = [] 

names_ensemble = []

for name, model in ensembles:

    kfold = KFold(n_splits=10, random_state=42) 

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2') 

    results_ensemble.append(cv_results) 

    names_ensemble.append(name) 

    print("Model Name:{} Model Score:{:.3f} Model Std:{:.3f}".format(name, cv_results.mean(), cv_results.std()))
# Compare Model's Acc

f,ax = plt.subplots(figsize = (10,7))

sns.boxplot(x=names_ensemble, y=results_ensemble,palette="viridis");

plt.title("Compare Ensemble Model's Accuracies",fontsize = 20,color='blue')

plt.xlabel('Models',fontsize = 15,color='blue')

plt.ylabel('Scores',fontsize = 15,color='blue')
# Tuning Extra Trees Regressior Model

estimators = list(range(25,301,25))

criterions = ["mse","mae"]

param_grid = dict(n_estimators=estimators,criterion=criterions) 
# Note: Taking time

# Applying Extra Trees Regressior

#etr = ExtraTreesRegressor()

#gs = GridSearchCV(estimator=etr,param_grid=param_grid,scoring="r2", cv=kfold)

#grid_search = gs.fit(x_train,y_train)

#best_score = grid_search.best_score_

#best_parameters = grid_search.best_params_

#print("Best Score:",best_score)

#print("Best Parameters:",best_parameters)
# Time to use ETR for dataset:

etr = ExtraTreesRegressor(n_estimators=100,criterion="mse")

etr.fit(x_train,y_train)

y_pred = etr.predict(x_test)
result_ETR = r2_score(y_test, y_pred)

print("{:.2f}".format(result_ETR))
results_models.append(result_ETR)
from xgboost import XGBRegressor

xgbr = XGBRegressor(silent=True)  # silent: close to warnings  

xgbr.fit(x_train,y_train)

y_pred = xgbr.predict(x_test)   
result_XGB = r2_score(y_test, y_pred)

print("{:.2f}".format(result_XGB))
results_models.append(result_XGB)
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor()

mlp.fit(x_train,y_train)    

#mlp.max_iter 

#mlp.hidden_layer_sizes#node sayısı 

y_pred = mlp.predict(x_test)   
result_MLP = r2_score(y_test, y_pred)

print("{:.2f}".format(result_MLP))
results_models.append(result_MLP)
models = ["DTR","ETR","XGB","NN"]
# Compare Model's Acc

f,ax = plt.subplots(figsize = (10,7))

sns.barplot(x=models, y=results_models,palette="viridis");

plt.title("Compare Ensemble Model's Scores",fontsize = 20,color='blue')

plt.xlabel('Models',fontsize = 15,color='blue')

plt.ylabel('Scores',fontsize = 15,color='blue')
data_con.head()
# Note that input data must be normalized

x_test_sample = np.array([[78992.15, 1132.00,  71976.41, 72.58, 5811.16, 5000, 133.76, 0, 2015, 8, 17, 1,0,0,0,0,0,0,0,0,0,0,

                           0,0,0,0,0,0,0,0,0,0,

                           0,0,0,0,0,0,0,0,0,0,

                           0,0,0,0,0,0,0,0,0,0,

                           0,0,0,0,0,0,0,0,0,0,0]])

                          

y_predict_sample = etr.predict(x_test_sample)

print('Expected Purchase Amount=', y_predict_sample)

y_predict_sample_orig = sc_y.inverse_transform(y_predict_sample)

print('Expected Purchase Amount=', y_predict_sample_orig)
from fbprophet import Prophet
data = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")

data.head()
prophet_df = data.iloc[:,[1,2]]

prophet_df.head()
prophet_df = prophet_df.sort_values("Date")
prophet_df = prophet_df.rename(columns={'Date':'ds', 'AveragePrice':'y'})
prophet_df.tail()
m = Prophet()

m.fit(prophet_df)
# Forcasting into the future

future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# You can plot the forecast

figure1 = m.plot(forecast, xlabel='Date', ylabel='Price')
# If you want to see the forecast components

figure2 = m.plot_components(forecast)
# pip install h2o

# or 

# pip install http://h2o-release.s3.amazonaws.com/h2o/rel-weierstrass/2/Python/h2o-3.14.0.2-py2.py3-none-any.whl

import h2o

from h2o.automl import H2OAutoML

h2o.init()
data = pd.read_csv("/kaggle/input/avocado/avocado_2.csv")

data.head()
# Load data into H2O

df = h2o.import_file("/kaggle/input/avocado/avocado_2.csv")
df.describe()
y = "C3"
# Parse Df

splits = df.split_frame([0.6, 0.2], seed = 1)
splits
# Parse Df

train = splits[0]

valid = splits[1]

test  = splits[2]
# Run AutoML

aml = H2OAutoML(max_runtime_secs = 300, seed = 1, project_name = "avocado_price")

aml.train(y = y, training_frame = train, leaderboard_frame = valid)
aml.leaderboard.head()
pred = aml.predict(test)

pred
test["C3"]
perf = aml.leader.model_performance(test)

perf