# This Python 3 environment comes with many helpful analytics libraries installed
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import scipy.stats as ss
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
import plotly.graph_objs as go
warnings.filterwarnings("ignore")

# Road the data
NewData = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
NewData
# Exploring the data by.info() .describe()
# - 1-1) Check the Data shape
# - 1-2) Check the Data type
# - 1-3) Check the Missing Values
# Exploring interesting themes by Visualization 
# - Tables / Plots / Correlation between the metrics 
# - Explore interesting themes 
    # Wealthy survive? 
    # By location 
    # Age scatterplot with ticket price 
    # Young and wealthy Variable? 
    # Total spent? 
# Feature engineering 
# preprocess data together or use a transformer? 
    # use label for train and test   
# Scaling?
# Model Baseline 
# Model comparison with CV 
# 1-1) Road the Data / Check the Data Shape
print ("Rows     : " ,NewData.shape[0]) # Check the Rows
print ("Columns  : " ,NewData.shape[1]) # Check the Columns
print ("Data     : " ,NewData.head()) # Check the data columns
# 1-2) Check the Data types
# Check the data types and found "TotalCharges" type is wrong. It changes "obejct" to "float64"
NewData.TotalCharges = pd.to_numeric(NewData.TotalCharges, errors='coerce') 
NewData.TotalCharges.astype(float) # Change TotalCharges "obejct" to "float64"
print(NewData.dtypes) 
# 1-3) Check the missing values
# Check the columns with missing values 
NewData[NewData.isnull().any(axis=1)].head()
print(NewData.isnull().sum()) 
# "TotalCharges" column has 11 missing values. I dropped all the NAs
# This is because the missing values' count is less than 5% of total counts
NewData = NewData.dropna()
print(NewData.isnull().sum()) 
# 1-4) To better understand the numeric data, I want to use the .describe() method. 
# This gives me an understanding of the central tendencies of the data
print(NewData.shape)
print(NewData.describe())
# 2-1-1) Table1: Comparing <contract vs churn>
df_cat_plot = NewData.drop(['customerID','tenure','MonthlyCharges','TotalCharges'],axis=1)
churn_table1 = pd.crosstab(index=df_cat_plot['Contract'],
                          columns=df_cat_plot['Churn'])
churn_table1
# 2-1-2) Table2 : Colored Table 
# cat dataframe -> recode using factorize 
df_cat_plot = NewData.drop(['customerID','tenure','MonthlyCharges','TotalCharges','Churn'],axis=1)
df_cat =df_cat_plot.apply(lambda x : pd.factorize(x)[0])+1
df_cat

df_cont = NewData[['tenure','MonthlyCharges','TotalCharges','Churn']]
df_cont
df_cat = df_cat.merge(df_cont,left_index=True,right_index=True)
df_cat

table1 = df_cat.groupby(['Churn']).mean()
th_props = [
  ('font-size', '12px'),
  ('text-align', 'left'),
  ('background-color', '#f7f7f9')
  ]

# CSS properties for td elements in dataframe
td_props = [
  ('font-size', '11px')
  ]
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]

table1 = table1.style.background_gradient(cmap='PuBu').set_table_styles([{'selector': 'th', 'props': [('font-size', '10pt')]}]).set_table_styles(styles)
table1
# 2-2-1) Plot1 : Check the Churn column count
df = sns.catplot(y="Churn", kind = "count", data=NewData, height = 3.0,
                 palette="Set1",
                 aspect = 2.5, orient = 'h')

# A Frequency table based on number of 
tb1 = pd.crosstab([NewData.Churn], 
                  columns='Number',
                  colnames =[' '],
                  margins = False) 
print(tb1)
# 2-2-2) Plot2 : Check the Churn %
values = NewData.Churn.value_counts()
plt.figure(figsize=(6,6))
plt.pie(values, explode = (0,0.1),autopct='%1.1f%%',labels=['Stay','Leave'], shadow=False,startangle=90,colors=['teal','gold'])
plt.show()
# 2-2-3) Plot 3 : Check the relationship between Churn and continous coloumns
fig, axs = plt.subplots(ncols=2,figsize=(8,5))
sns.set(style="whitegrid", color_codes=True)
np.random.seed(2017)

ax1= sns.pointplot(x="Contract", y="TotalCharges", hue="Churn", data=NewData,
palette={"No": "g", "Yes": "m"},
markers=["^", "o"], linestyles=["-", "--"],
ax=axs[0])
ax1.set_title("Total charges of 3 Contracts")

ax2=sns.pointplot(x="InternetService", y="TotalCharges", hue="Churn", data=NewData,
              palette={"No": "r", "Yes": "y"},
              markers=["^", "o"], linestyles=["-", "--"],ax=axs[1])
ax2.set_title("Total charges of 3 Internet Service")
plt.tight_layout()
plt.show()
# 3) Make a heatmap
df_cat['Churn'] = np.where(df_cat['Churn']=='Yes', 1, 0)
print(df_cat.head())
print(df_cat.dtypes)

corr = df_cat.corr()
sns.set(rc={'figure.figsize':(16,10)})
correlation_matrix = df_cat.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
# 4-1) Demographic analysis
# Data preparation
df_cat_plot = df_cat_plot.merge(df_cont, right_index=True,left_index=True) #original data without factorize
df_cat_plot['Churn'] = np.where(df_cat_plot['Churn']=='Yes', 1, 0)
df_cat_plot

sns.catplot(x="SeniorCitizen", y="MonthlyCharges", kind="box", data=df_cat_plot);
# Plot tells that the senior citizen tend to have higher montly charges as compared to younger population

sns.catplot(x="gender", y="MonthlyCharges", kind="box", data=df_cat_plot);

# Male and females largely have same avg monthly bills

sns.catplot(x="SeniorCitizen", y="tenure", kind="box", data=df_cat_plot);

# Senior citizen have higher mean tenure as compared to young population young people churn often

sns.catplot(x="Dependents", y="MonthlyCharges", kind="box", data=df_cat_plot);

# People with no dependents have higher avg. monthly bills

sns.catplot(x="Partner", y="MonthlyCharges", kind="box", data=df_cat_plot);

plt.figure(figsize=(15, 15))

plt.subplot(3, 2, 1)
sns.countplot('gender', data=NewData, hue='Churn')

plt.subplot(3, 2, 2)
sns.countplot('SeniorCitizen', data=NewData, hue='Churn')

plt.subplot(3, 2, 3)
sns.countplot('Partner', data=NewData, hue='Churn')

plt.subplot(3, 2, 4)
sns.countplot('Dependents', data=NewData, hue='Churn')

plt.subplot(3, 2, 5)
sns.countplot('PhoneService', data=NewData, hue='Churn')

plt.subplot(3, 2, 6)
sns.countplot('PaperlessBilling', data=NewData, hue='Churn')
# People with partners have higher avg. monthly bills

sns.catplot(x="MultipleLines", y="MonthlyCharges", kind="box", data=df_cat_plot);

sns.catplot(x="InternetService", y="MonthlyCharges", kind="box", data=df_cat_plot);

# Fiber optics service has very high monthly charges
sns.catplot(x="PaymentMethod", y="MonthlyCharges", kind="box", data=df_cat_plot);

plt.figure(figsize=(14, 14))

plt.subplot(3, 2, 1)
NewData[NewData.Churn == 'No'].tenure.hist(bins=35, alpha=0.6, label='Churn=No')
NewData[NewData.Churn == 'Yes'].tenure.hist(bins=35, alpha=0.6, label='Churn=Yes')
plt.legend()
plt.xlabel('Number of months with company')

plt.subplot(3, 2, 2)
NewData[NewData.Churn == 'No'].tenure.value_counts().hist(bins=50, alpha=0.6, label='Churn=No')
NewData[NewData.Churn == 'Yes'].tenure.value_counts().hist(bins=50, alpha=0.6, label='Churn=Yes')
plt.legend()
# plt.xlabel() Ziru? please see what goes in Label over here!

plt.subplot(3, 2, 3)
NewData[NewData.Churn == 'No'].MonthlyCharges.hist(bins=35, alpha=0.6, label='Churn=No')
NewData[NewData.Churn == 'Yes'].MonthlyCharges.hist(bins=35, alpha=0.6, label='Churn=Yes')
plt.xlabel('Monthly Payment')
plt.legend()

plt.subplot(3, 2, 4)
NewData[NewData.Churn == 'No'].TotalCharges.hist(bins=35, alpha=0.6, label='Churn=No')
NewData[NewData.Churn == 'Yes'].TotalCharges.hist(bins=35, alpha=0.6, label='Churn=Yes')
plt.xlabel('Total Payment')
plt.legend()
# Check the distribution of some categorical variables that indicates high correlation in the heat map
# Oneline BackUp, Online Security, DeviceProtection, Streaming TV, Streaming Movies and Techsupport 
# These columns have the same range of answers 
plt.figure(figsize=(17, 17))

plt.subplot(3, 3, 6)
sns.countplot('OnlineBackup', data=NewData, hue='Churn')

plt.subplot(3, 3, 1)
sns.countplot('OnlineSecurity', data=NewData, hue='Churn')

plt.subplot(3, 3, 2)
sns.countplot('StreamingTV', data=NewData, hue='Churn')

plt.subplot(3, 3, 3)
sns.countplot('StreamingMovies', data=NewData, hue='Churn')

plt.subplot(3, 3, 4)
sns.countplot('DeviceProtection', data=NewData, hue='Churn')

plt.subplot(3, 3, 5)
plt.tight_layout
sns.countplot('TechSupport', data=NewData, hue='Churn')
 
# So for these conditions, we'd like to keep only one variable.
plt.figure(figsize=(15, 18))
plt.subplot(3, 2, 3)
g = sns.countplot('PaymentMethod', data=NewData, hue='Churn')
g.set_xticklabels(g.get_xticklabels(), rotation=45);

plt.subplot(3, 2, 4)
g = sns.countplot('Contract', data=NewData, hue='Churn')
g.set_xticklabels(g.get_xticklabels(), rotation=45);
# Make tenure to categorical column
def tenure_lab(NewData) :
    
    if NewData["tenure"] <= 12 :
        return "1year"
    elif (NewData["tenure"] > 12) & (NewData["tenure"] <= 24 ):
        return "2years"
    elif (NewData["tenure"] > 24) & (NewData["tenure"] <= 48) :
        return "3years"
    elif (NewData["tenure"] > 48) & (NewData["tenure"] <= 60) :
        return "4years"
    elif NewData["tenure"] > 60 :
        return "over5years"
NewData["Tenure_Category"] = NewData.apply(lambda NewData:tenure_lab(NewData),
                                     axis = 1)
churn     = NewData[NewData["Churn"] == "Yes"]
not_churn = NewData[NewData["Churn"] == "No"]
NewData1 = NewData.groupby(["Tenure_Category","Churn"])[["MonthlyCharges",
                                                    "TotalCharges"]].mean().reset_index()

#function for tracing 
def mean_charges(column,aggregate) :
    tracer = go.Bar(x = NewData1[NewData1["Churn"] == aggregate]["Tenure_Category"],
                    y = NewData1[NewData1["Churn"] == aggregate][column],
                    name = aggregate,marker = dict(line = dict(width = 1)),
                    text = "Churn"
                   )
    return tracer

#function for layout
def layout_plot(title,xaxis_lab,yaxis_lab) :
    layout = go.Layout(dict(title = title,
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = xaxis_lab,
                                         zerolinewidth=1,ticklen=5,gridwidth=2),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = yaxis_lab,
                                         zerolinewidth=1,ticklen=5,gridwidth=2),
                           )
                      )
    return layout
    

#plot1 - mean monthly charges by tenure groups
trace1  = mean_charges("MonthlyCharges","Yes")
trace2  = mean_charges("MonthlyCharges","No")
layout1 = layout_plot("Average Monthly Charges by Tenure groups",
                      "Tenure group","Monthly Charges")
data1   = [trace1,trace2]
fig1    = go.Figure(data=data1,layout=layout1)

#plot2 - mean total charges by tenure groups
trace3  = mean_charges("TotalCharges","Yes")
trace4  = mean_charges("TotalCharges","No")
layout2 = layout_plot("Average Total Charges by Tenure groups",
                      "Tenure group","Total Charges")
data2   = [trace3,trace4]
fig2    = go.Figure(data=data2,layout=layout2)

fig1.show()
fig2.show()
bins = [0,12,36,72]
name = ['Less than 1 year', '1-3 years', 'More than 3 years']
NewData['Duration'] = pd.cut(NewData.tenure, bins, labels=name)

table1 = pd.crosstab(index=NewData.Contract, columns=NewData.Churn)
table2 = pd.crosstab(index=NewData.PaymentMethod, columns=NewData.Churn)
table2.index = pd.Series(['Bank transfer','Credit card','Electronic check','Mailed check'])
table3 = pd.crosstab(index=NewData.InternetService, columns=NewData.Churn)
table4 = pd.crosstab(index=NewData.Duration, columns=NewData.Churn)

fig = plt.figure(figsize=(20,12))
plt.style.use('seaborn-darkgrid')
ax1= fig.add_subplot(2,2,1)
table1.plot(ax=ax1, kind="bar", stacked=True)
plt.xticks(rotation=0)
ax2 = plt.subplot(2,2,2)
table2.plot(ax=ax2,kind="bar", stacked=True)
plt.xticks(rotation=0)
ax3 = plt.subplot(2,2,3)
table3.plot(ax=ax3,kind="bar", stacked=True)
plt.xticks(rotation=0)
ax4 = plt.subplot(2,2,4)
table4.plot(ax=ax4,kind="bar", stacked=True)
plt.xticks(rotation=0)
plt.show()
#1. Do one hot encoding
#2. Do interaction terms
#3. Do polynomial features and log features for numeric
# pca is not suitable for categorical data!

# Create poly faetures
y = NewData['Churn']
print(y.shape)
y = pd.DataFrame(y)

#resetting index
y.reset_index(inplace=True,drop=True)
X = NewData.drop('Churn',axis=1)
print(X.shape)

# Drop Customer ID
X = X.drop('customerID',axis=1)

# Seperate numeric columns
X_num = X[['tenure','MonthlyCharges','TotalCharges']]
# Create polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X_num)
tmp1 = pd.DataFrame(X_poly)
# Create log features
X_log = np.log(X_num)
tmp = pd.DataFrame(X_log)
print(tmp.shape) # same number of columns, but they've changed.
tmp

# reset the index
tmp.reset_index(inplace=True,drop=True)

# Rename columns and adding _log
tmp.columns = [col+'_'+'log' for col in tmp.columns]
# Make interaction term 
# I found there are interesting relationship between Tenure and MonthlyCharges, So I made another column for these two variables
new_col = X_num['tenure'] * X_num['MonthlyCharges']
X_num.insert(loc=3, column='tenure*monthlycharges', value=new_col)
# Merge X_num, tmp1(poly features) and tmp(log features)
# Reset the index of X_num
X_num.reset_index(inplace=True,drop=True)

# Smoosh the two dataframes together
X_smoosh = pd.concat([X_num,tmp1, tmp], axis=1)
X_smoosh.shape # Look at how many more columns there are
X_smoosh  #this is the dataframe which has the original numeric columns from the actual dataset and the log features and polynomial features
# Feature Engineering for Categorical Variables
X_cat = X.drop(['tenure','MonthlyCharges','TotalCharges','Tenure_Category','Duration'],axis=1)
X_cat.reset_index(drop=True,inplace=True)
X_cat.reset_index(drop=False,inplace=True)
X_cat.head() #This has only categorical variables!
X_cat.columns
cat_df = X_cat.iloc[:,1:]

columns = cat_df.columns
df_final = pd.DataFrame(X_cat['index'])
for col in columns:
  df = pd.DataFrame(cat_df[col])
  one_hot = pd.get_dummies(cat_df[col])
  df = df.join(one_hot)
  df_final = df_final.merge(df,left_index=True,right_index=True)

    
df_final = df_final.iloc[:,1:]
df_final.head()
B = ['gender','Female','Male','SeniorCitizen','Not_A_SeniorCitizen','SeniorCitizen','Partner','DoesNotHaveaPartner','HasPartner','Dependent','NoDependents','Has_Dependents',
                      'PhoneService','No_PhoneService','Has_PhoneService','MultipleLines','No_ML','NoPhoneService','HasML','InternetService','DSL_Internet','fiberOptics','NoInternet','OnlineSecurity','NotOptedforOnlineSec',
                      'NotApplicable(noInternet)','OptedforOnlineSec','OnlineBackup','NotOptedforOnlineBackup','NotApplicable(NoInternet)','OptedforOnlineBackup','DeviceProtection','NotOptedforDeviceProtection','NotApplicable(NoInternet)','OptedforDevicePro','TechSupport','notOPtedforTechSupport','NotApplicable','optedforTechssupport','StreamingTV','NotOptedTV','NotApplcable','OPtedTV','StreamingMovies',
                      'NotOptedMovies','NotApplicable','OptedMovies','Contract','Month-to-month','One year','Two year','PaperlessBilling','PaperBilling','PaperlessBilling','PaymentMethod','Bank transfer(auto)','Credit Card(auto)',
                      'Electronic check','Mailed check']
B = pd.DataFrame(B)
B.shape
df_final.columns=['gender','Female','Male','SeniorCitizen','Not_A_SeniorCitizen','SeniorCitizen','Partner','DoesNotHaveaPartner','HasPartner','Dependent','NoDependents','Has_Dependents',
                      'PhoneService','No_PhoneService','Has_PhoneService','MultipleLines','No_ML','NoPhoneService','HasML','InternetService','DSL_Internet','fiberOptics','NoInternet','OnlineSecurity','NotOptedforOnlineSec',
                      'NotApplicable(noInternet)','OptedforOnlineSec','OnlineBackup','NotOptedforOnlineBackup','NotApplicable(NoInternet)','OptedforOnlineBackup','DeviceProtection','NotOptedforDeviceProtection','NotApplicable(NoInternet)','OptedforDevicePro','TechSupport','notOPtedforTechSupport','NotApplicable','optedforTechssupport','StreamingTV','NotOptedTV','NotApplcable','OPtedTV','StreamingMovies',
                      'NotOptedMovies','NotApplicable','OptedMovies','Contract','Month-to-month','One year','Two year','PaperlessBilling','PaperBilling','Paperless Billing','PaymentMethod','Bank transfer(auto)','Credit Card(auto)',
                      'Electronic check','Mailed check']
df_final.head()
# Join all features engineered for categorical and continuous
df_modeling = df_final.merge(X_smoosh,left_index=True,right_index=True)

df_modeling.head() # This is the merger of categorical and numeric feature engineering
#Though,still need to standardize numeric data and get rid of original categorical variables!
y['Churn'].value_counts()
df_modeling.iloc[:,39:].head()
# Data Standardization
from sklearn import preprocessing

std = X_smoosh.values 

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(std)
x_scaled
# Let's convert that x_scaled, numpy array to a pandas dataframe
# Note that x_scaled has no column labels
df_mmstd = pd.DataFrame(x_scaled, columns=X_smoosh.columns)
df_mmstd.head()
# df_mmstd has the all the numeric columns standardized
df_final.head()
df_final_without = df_final.drop(['gender','SeniorCitizen','Partner','Dependent','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
'Contract','PaymentMethod','PaperlessBilling'], axis=1,inplace=False)
df_final_without.shape
# Merge Standardization with Categorical
df_std_modelling = df_final_without.merge(df_mmstd, left_index=True, right_index=True)
df_std_modelling.head() #megered dataframe for modeling (standardized numeric cols and dummy vars for categorical)
col = np.array(df_std_modelling.columns)
col
df_std_modelling.dtypes
df_std_modelling
y
import imblearn.under_sampling as u

df_std_modelling,y = make_classification(n_features = 84, n_samples=2000) 
# Make classification default = 20, so I need to set the number
ros = u.RandomUnderSampler(sampling_strategy='majority')
X_resampled, Y_resampled = ros.fit_resample(df_std_modelling, y)
X_resampled = pd.DataFrame(X_resampled)
print(X_resampled.shape)

Y_resampled = pd.DataFrame(Y_resampled)
print(Y_resampled.shape)
Y_resampled.head()
Y_resampled = Y_resampled.rename(columns={0:'Churn'})
seed = 7
X_train, X_test, y_train, y_test =\
  train_test_split(X_resampled, Y_resampled, 
                   stratify=Y_resampled, 
                   test_size=0.2, random_state=seed)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
X_train.head()
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
#import warnings
#warnings.filterwarnings("ignore")
# Spot-checking 
models = []
models.append(('LR', LogisticRegression(max_iter=1000000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Bagging', BaggingClassifier()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('ExtraTree', ExtraTreesClassifier()))
models.append(('GradientBoosting', GradientBoostingClassifier()))

##################################################
# evaluate each model in turn

results = []
names = []

# store preds
for name, model in models:
  cv_results = cross_val_score(model,X_train,y_train, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names, rotation=45)
pyplot.show()
# Make predictions on validation dataset (Logistic Regression)
lr = LogisticRegression(max_iter=100000)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
## Hyper parameter tuning
#Hypertuning with Grid Search

grid_params_LDA = [{'solver':['svd','lsqr','eigen'], 'tol':[0.0001,0.0002,0.0003]}]

grid_params_LR = [{'penalty':['l1', 'l2', 'elasticnet', 'none'],'solver':['newton-cg','lbfgs', 'liblinear', 'sag', 'saga']}]

gs_LDA = GridSearchCV(estimator=LinearDiscriminantAnalysis(), param_grid=grid_params_LDA, scoring='accuracy', cv=10)

gs_LR = GridSearchCV(estimator=LogisticRegression(), param_grid=grid_params_LR, scoring='accuracy', cv=10)

grids = [gs_LR,gs_LDA]
grid_dict = {0:'LogisticRegression',1:'LinearDiscriminantAnalysis'}
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
	print('\nEstimator: %s' % grid_dict[idx])	
	gs.fit(X_train, y_train)
	print('Best params: %s' % gs.best_params_)
	print('Best training accuracy: %.3f' % gs.best_score_)
	y_pred = gs.predict(X_test)
	print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
	if accuracy_score(y_test, y_pred) > best_acc:
		best_acc = accuracy_score(y_test, y_pred)
		best_gs = gs
		best_clf = idx
print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])
# Make predictions on validation dataset
lr = gs.best_estimator_
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test,predictions)
group_names = ['True Neg','False Pos','False Neg','True Pos']
categories = ['No', 'Yes']


# import one function: make_confusion_matrix
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if  len(group_names) ==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent: 
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf[1])]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=(10,7))
    sns.heatmap(cf,annot=box_labels,fmt="",cmap="bone",cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


make_confusion_matrix(cm, 
                      group_names=group_names,
                      categories=categories, 
                      cmap='bone')
#Feature Importance for LR

from sklearn.inspection import permutation_importance

results = permutation_importance(lr,X_train, y_train, scoring='neg_root_mean_squared_error')

plt.figure(figsize=(10,8))

#get importance
importance = results.importances_mean
sorted_idx = np.argsort(importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, importance[sorted_idx],height=0.4,align='center')

plt.yticks(pos, X_train.columns[sorted_idx],fontsize=10)
plt.xlabel('Permutation Feature Importance Scores', fontsize=10)
#plt.xticks(fontsize=100)
plt.title('Permutation Feature Importance for Logistic Regression', fontsize=20)

plt.tight_layout()

plt.show()
#Feature Importance for LR

from sklearn.inspection import permutation_importance

results = permutation_importance(lr,X_train, y_train, scoring='neg_root_mean_squared_error')

plt.figure(figsize=(10,8))

#get importance
importance = results.importances_mean
sorted_idx = np.argsort(importance)[:30]
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos[:30], importance[sorted_idx],height=0.4,align='center')

plt.yticks(pos[:30], X_train.columns[sorted_idx],fontsize=10)
plt.xlabel('Permutation Feature Importance Scores', fontsize=10)
#plt.xticks(fontsize=100)
plt.title('Permutation Feature Importance for Logistic Regression', fontsize=20)

plt.tight_layout()

plt.show()