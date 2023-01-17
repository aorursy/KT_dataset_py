import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import *
import seaborn as sns
%matplotlib inline

## Input data files are available in the "../input/" directory.

data_path = "../input/"
data = pd.read_csv(data_path+"WA_Fn-UseC_-Telco-Customer-Churn.csv", dtype='unicode', encoding="utf-8-sig")
data.head()
data.dtypes


# Changing datatypes to category

#data["gender"] = data["gender"].astype('category')
#data["SeniorCitizen"] = data["SeniorCitizen"].astype('category')

obj = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
for j in obj:
    data[j] = data[j].astype('category')
#check for missing data
missing_data = data.isnull().sum(axis=0).reset_index()
data.dtypes
# check count of yes / no churns in the dataset

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 5

plt.rcParams["figure.figsize"] = fig_size
sns.countplot(x='Churn', data=data)
# check count of yes / no churns in the dataset

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 5

plt.rcParams["figure.figsize"] = fig_size
sns.countplot(x='SeniorCitizen', data=data)
fig, axes = plt.subplots(3, 4, figsize=(25, 25))
sns.countplot('gender',data=data, ax=axes[0,0])
sns.countplot('PhoneService',data=data, ax=axes[0,1])
sns.countplot('MultipleLines',data=data, ax=axes[0,2])
sns.countplot('InternetService',data=data, ax=axes[0,3])
sns.countplot('OnlineSecurity',data=data, ax=axes[1,0])
sns.countplot('OnlineBackup',data=data, ax=axes[1,1])
sns.countplot('DeviceProtection',data=data, ax=axes[1,2])
sns.countplot('TechSupport',data=data, ax=axes[1,3])
sns.countplot('StreamingTV',data=data, ax=axes[2,0])
sns.countplot('StreamingMovies',data=data, ax=axes[2,1])
sns.countplot('Contract',data=data, ax=axes[2,2])
sns.countplot('PaperlessBilling',data=data, ax=axes[2,3])

from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.palettes import Spectral4 
from bokeh.models import ColumnDataSource, LabelSet
import warnings
warnings.filterwarnings('ignore')
from bokeh.io import save, push_notebook, output_notebook, curdoc
output_notebook()

x = list(data.PaymentMethod.unique())#['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
counts = data['PaymentMethod'].value_counts()

hover = HoverTool(
        tooltips=[
            ("Type", "@x"),
            ("Count", "@counts{int}")
            ]
    )

source = ColumnDataSource(data=dict(x=x, counts=counts, color=Spectral4))
p = figure(x_range=x, y_range=(0,2500), plot_height=400, plot_width = 800, tools=[hover])
p.vbar(x='x', top='counts', width=0.9, color='color', legend="x", source=source)

p.xgrid.grid_line_color = None
p.legend.orientation = "horizontal"
p.legend.location = "top_center"

show(p)
# -- convert to numeric 

data.TotalCharges=pd.to_numeric(data.TotalCharges,errors='coerce')
data.MonthlyCharges=pd.to_numeric(data.MonthlyCharges,errors='coerce')
data.tenure=pd.to_numeric(data.tenure,errors='coerce')
# Tenure 
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
a = data['tenure']
x = np.log(1/data['tenure'])
y = np.exp(data['tenure'])
z = np.sqrt(data['tenure'])
a.plot.hist()
plt.xlabel('Tenure')
#sns.distplot(a, kde = False)
print("The skewness of SalePrice is {}".format(data['tenure'].skew()))
# Monthly Charges

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
m = data['MonthlyCharges']
xm = np.log(1/data['MonthlyCharges'])
ym = np.exp(data['MonthlyCharges'])
zm = np.sqrt(data['MonthlyCharges'])

plt.xlabel('Monthly Charges')
m.plot.hist()
print("The skewness of SalePrice is {}".format(data['MonthlyCharges'].skew()))
# Total Charges
import warnings
warnings.filterwarnings('ignore')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
t = data['TotalCharges']
xt = np.log(1/data['TotalCharges'])
yt = np.exp(data['TotalCharges'])
zt = np.sqrt(data['TotalCharges'])

plt.xlabel('Total Charges')
t.plot.hist()
print("The skewness of SalePrice is {}".format(data['TotalCharges'].skew()))
# check count of yes / no churns among gender in the dataset

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
sns.countplot(x='Churn', hue = 'gender', data=data)
srchurn = data[((data['Churn']=='Yes' ) & (data['SeniorCitizen']== '0'))|((data['Churn']=='No') & (data['SeniorCitizen']== '0'))].groupby(['Churn'])['Churn'].count()
srchurn
labels = (np.array(srchurn.index))
sizes = (np.array((srchurn / srchurn.sum())*100))
colors = ['Green', 'lightskyblue']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("% of Churns versus No-churns among non Senior citizens")
plt.show()
srcchurn = data[((data['Churn']=='Yes' ) & (data['SeniorCitizen']== '1'))|((data['Churn']=='No') & (data['SeniorCitizen']== '1'))].groupby(['Churn'])['Churn'].count()
srcchurn
labels = (np.array(srcchurn.index))
sizes = (np.array((srcchurn / srcchurn.sum())*100))
colors = ['Pink', 'Gold']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("% of Churns versus No-churns among Senior citizens")
plt.show()
# Males and Females who have not left Telco vs who left

gchurn = data[((data['Churn']=='Yes' ) & (data['gender']== 'Male'))|((data['Churn']=='Yes') & (data['gender']== 'Female'))].groupby(['gender'])['gender'].count()
gchurn
labels = (np.array(gchurn.index))
sizes = (np.array((gchurn / gchurn.sum())*100))
colors = ['Pink', 'Violet']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("% of Churns among Gender")
plt.show()
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.palettes import Spectral5 
from bokeh.models import ColumnDataSource, LabelSet
import warnings
warnings.filterwarnings('ignore')
from bokeh.io import save, push_notebook, output_notebook, curdoc
output_notebook()

x = list(data.Churn.unique())
counts = data.groupby(['Churn'])['tenure'].mean()

hover = HoverTool(
        tooltips=[
            ("Churn", "@x"),
            ("Mean", "@counts{int}")
            ]
    )

source = ColumnDataSource(data=dict(x=x, counts=counts, color=Spectral5))
p = figure(x_range=x, y_range=(0,50), plot_height=400, plot_width = 800, tools=[hover])
p.vbar(x='x', top='counts', width=0.9, color='color', legend="x", source=source)

p.xgrid.grid_line_color = None
p.legend.orientation = "horizontal"
p.legend.location = "top_center"
show(p)
#plt.scatter(data['tenure'], data['TotalCharges'], s=data['tenure'], c =data['TotalCharges'], marker = '.');

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models import HoverTool
from bokeh.io import save, push_notebook, output_notebook, curdoc
output_notebook()

source = ColumnDataSource(data=dict(
            x=data['tenure'],
            y=data['TotalCharges']            
        )
    )

p = figure(title="Bokeh Markers", toolbar_location=None)
p.grid.grid_line_color = None
p.background_fill_color = "#eeeeee"

hover = HoverTool(
        tooltips=[
            ("Tenure", "@x"),
            ("Total Charges", "@y{int}")
            ]
    )

p = figure(plot_width=700, plot_height=700, tools=[hover],
           title="Mouse over the dots")

p.circle('x', 'y', size=7, source=source)
p.xaxis.axis_label = 'tenure'
p.yaxis.axis_label = 'Total Charges'

show(p)

# sns.regplot(data.tenure, data.TotalCharges)
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models import HoverTool, axes 
from bokeh.io import save, push_notebook, output_notebook, curdoc
output_notebook()

source = ColumnDataSource(data=dict(
            x=data['tenure'],
            y=data['MonthlyCharges']            
        )
    )

p = figure(title="Bokeh Markers", toolbar_location=None)
p.grid.grid_line_color = None
p.background_fill_color = "#eeeeee"

hover = HoverTool(
        tooltips=[
            ("Tenure", "@x"),
            ("Monthly Charges", "@y{int}")
            ]
    )

p = figure(plot_width=700, plot_height=700, tools=[hover],
           title="Mouse over the dots")

p.circle('x', 'y', size=6, source=source)
p.xaxis.axis_label = 'tenure'
p.yaxis.axis_label = 'Monthly Charges'
show(p)
numdata = (data[['MonthlyCharges','TotalCharges','tenure']].corr())
mask = np.zeros_like(numdata, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(12, 10))

# colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(numdata, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5 , cbar_kws={"shrink": .5})
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 11
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

sns.set(style="whitegrid")
ax = sns.boxplot(x=data['TotalCharges'], y=data['Churn'])
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 11
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

sns.set(style="whitegrid")
ax = sns.boxplot(x=data['MonthlyCharges'], y=data['Churn'])
crosstab = pd.crosstab(data['Churn'], data['gender'])
crosstab
#important
from scipy import stats
stats.chi2_contingency(crosstab)
crosstab1 = pd.crosstab(data['Churn'], data['InternetService'])
from scipy import stats
stats.chi2_contingency(crosstab1)
crosstab2 = pd.crosstab(data['Churn'], data['Contract'])
from scipy import stats
stats.chi2_contingency(crosstab2)
crosstab3 = pd.crosstab(data['Churn'], data['SeniorCitizen'])
from scipy import stats
stats.chi2_contingency(crosstab3)
# missing value check

data.isnull().sum() 

# treating missing values in total charges column

data['TotalCharges'] = data['TotalCharges'].fillna((data['TotalCharges'].median()))
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size

sns.set(style="whitegrid")
ax = sns.boxplot(x=data['tenure'])
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size

sns.set(style="whitegrid")
ax = sns.boxplot(x=data['TotalCharges'], palette="Set2")
data_model=data
data_model=data_model.drop(columns=['customerID'])
data_dummy=pd.get_dummies(data_model, drop_first=True)
X=data_dummy.iloc[:,0:30]
X.dtypes
Y=data_dummy.iloc[:,30]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
X_test.shape
X_train.shape
y_test.shape
y_train.shape
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lgmodel = LogisticRegression()
lgmodel.fit(X_train, y_train)
y_pred = lgmodel.predict(X_test)
print('Accuracy of logistic regression model on test data: {:.2f}'.format(lgmodel.score(X_test, y_test)))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
results = model_selection.cross_val_score(modelCV, X, Y, cv=kfold, scoring='accuracy')
print("10-fold cross validation average accuracy: %.2f" % (results.mean()))
lgmodel.coef_
import numpy as np
coefs=lgmodel.coef_[0]
top_three = np.argpartition(coefs, -10)[-10:]
top_three_sorted=top_three[np.argsort(coefs[top_three])]
print(data_dummy.columns.values[top_three_sorted])
data['Churn']= data.Churn.map(dict(Yes=1, No=0))
Y=data['Churn']
Y
import statsmodels.api as sm
logit = sm.Logit(Y,X)

# fit the model
#result = logit.fit()
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

#print(result.summary())
#np.exp(result.params)
from xgboost import XGBClassifier
xgb1 = XGBClassifier()
xgb1.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
xgb1_pred = xgb1.predict(X_test)
xgb1_pred_prob = xgb1.predict_proba(X_test)
accuracy = accuracy_score(y_test, xgb1_pred)
print('Accuracy = {:0.2f}%.'.format(accuracy))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)

feature_importances_xgb = pd.DataFrame(xgb1.feature_importances_,
                                  index = X_train.columns,
                                  columns=['importance']).sort_values('importance', ascending=False)
feature_importances_xgb
params = {
        'objective': ['binary:logistic'],
        'min_child_weight': range(1,8,2),
        'gamma':[i/10.0 for i in range(0,5)],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate' : [0.1, 0.2, 0.01],
        'n_estimators' : [1000, 2000],
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)],
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
        }
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rs = GridSearchCV(xgb1,
                  params,
                  cv=5,
                  scoring="accuracy",
                  n_jobs=1,
                  verbose=2)
#rs.fit(X_train, y_train)
#best_est = rs.best_estimator_
#print(best_est)
xgb2 = XGBClassifier(colsample_bylevel= 0.6,
 colsample_bytree = 0.8,
 max_depth = 9,
 min_child_weight = 2, gamma= 1,
 n_estimators = 600, learning_rate=0.01, nthread = 1, reg_alpha = 0.1)
xgb2.get_params
xgb2.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
xgb2_pred = xgb2.predict(X_test)
xgb2_pred_prob = xgb2.predict_proba(X_test)
accuracy = accuracy_score(y_test, xgb2_pred)
print('Accuracy = {:0.2f}%.'.format(accuracy))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
results = model_selection.cross_val_score(xgb2, X, Y, cv=kfold, scoring='accuracy')
print("10-fold cross validation average accuracy: %.2f" % (results.mean()))
rfc = RandomForestClassifier(n_estimators=1000, max_depth=None)
rfc = rfc.fit(X_train, y_train)
#check for missing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
missing_data1 = X_train.isnull().sum(axis=0).reset_index()
missing_data1
X_train['TotalCharges'] = X_train['TotalCharges'].fillna((X_train['TotalCharges'].median()))
#check for missing data
missing_data_test = X_test.isnull().sum(axis=0).reset_index()
missing_data_test
X_test['TotalCharges'] = X_test['TotalCharges'].fillna((X_test['TotalCharges'].median()))
bigrfc_predictions = rfc.predict(X_test)
bigrfc_predictions_prob = rfc.predict_proba(X_test)
accuracy_rf = accuracy_score(y_test, bigrfc_predictions)
print('Accuracy = {:0.2f}%.'.format(accuracy_rf))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
results = model_selection.cross_val_score(rfc, X, Y, cv=kfold, scoring='accuracy')
print("10-fold cross validation average accuracy: %.2f" % (results.mean()))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, lgmodel.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lgmodel.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
#xgb_roc_auc = roc_auc_score(y_test, xgb2.predict(X_test))
rf_roc_auc = roc_auc_score(y_test, rfc.predict(X_test))
rf_roc_auc
fpr, tpr, thresholds = roc_curve(y_test, rfc.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='RF (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
# Random Forest

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, bigrfc_predictions)
print(confusion_matrix)
# XGB

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, xgb2_pred)
print(confusion_matrix)
# Logistic model

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)