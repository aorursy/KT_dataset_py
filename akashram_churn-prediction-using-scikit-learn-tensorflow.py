import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

data_path = "../input/"
data = pd.read_csv(data_path+"WA_Fn-UseC_-Telco-Customer-Churn.csv", dtype='unicode', encoding="utf-8-sig")
data.head()
data.dtypes
# Changing datatypes to category

#data["gender"] = data["gender"].astype('category')
#data["SeniorCitizen"] = data["SeniorCitizen"].astype('category')

#Replacing spaces with null values in total charges column
data['TotalCharges'] = data["TotalCharges"].replace(" ",np.nan)

#Dropping null values from total charges column which contain .15% missing data 
data = data[data["TotalCharges"].notnull()]
data = data.reset_index()[data.columns]

replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in replace_cols : 
    data[i]  = data[i].replace({'No internet service' : 'No'})
    
data["SeniorCitizen"] = data["SeniorCitizen"].replace({1:"Yes",0:"No"})

obj = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
'PaperlessBilling', 'PaymentMethod', 'Churn']

data[obj] = data[obj].astype('category')
data['tenure'] = data['tenure'].astype('int')

def tenure_bracket(data) :
    if data["tenure"] <= 12 :
        return "0-12"
    elif (data["tenure"] > 12) & (data["tenure"] <= 24 ):
        return "12-24"
    elif (data["tenure"] > 24) & (data["tenure"] <= 48) :
        return "24-48"
    elif (data["tenure"] > 48) & (data["tenure"] <= 60) :
        return "48-60"
    elif data["tenure"] > 60 :
        return "> 60"

data["tenure_group"] = data.apply(lambda data:tenure_bracket(data), axis = 1)
data.head(10)
Idcol     = ['customerID']
target_col = ["Churn"]
cat_cols   = data.nunique()[data.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in data.columns if x not in cat_cols + target_col + Idcol]
#  Check for missing data

missing_data = data.isnull().sum(axis=0).reset_index()
missing_data
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
sns.countplot(x='Churn', data=data)
no = len(data[data['Churn'] == 'No'])
yes = len(data[data['Churn'] == 'Yes'])

d = {'Churn': ['Yes', 'No'], 'Perc': [yes/(yes+no) * 100, no/(yes+no) * 100 ]}
pd.DataFrame(data=d)
label = data["Churn"].value_counts().keys().tolist()
values = data["Churn"].value_counts().values.tolist()

trace = go.Pie(labels = label ,
               values = values ,
               marker = dict(colors =  [ 'green' ,'red'],
                             line = dict(color = "white",
                                         width =  1.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Customer Churn",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data1 = [trace]
fig = go.Figure(data = data1,layout = layout)
py.iplot(fig)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size

s = data['SeniorCitizen'].value_counts().head(10)
print (s)

ax=s.plot.bar(width=.8) 

for i, v in s.reset_index().iterrows():
    ax.text(i, v.SeniorCitizen + 0.2 , v.SeniorCitizen, color='red')
label = data["SeniorCitizen"].value_counts().keys().tolist()
values = data["SeniorCitizen"].value_counts().values.tolist()
label[0] = 'No'
label[1] = 'Yes'

trace = go.Pie(labels = label ,
               values = values ,
               marker = dict(colors =  ['blue' ,'pink'],
                             line = dict(color = "white",
                                         width =  1.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Senior Citizen",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data2 = [trace]
fig = go.Figure(data = data2,layout = layout)
py.iplot(fig)
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
sns.countplot('StreamingMovies',data=data, ax=axes[2,0])
sns.countplot('Contract',data=data, ax=axes[2,1])
sns.countplot('PaperlessBilling',data=data, ax=axes[2,2])
sns.countplot('PaymentMethod',data=data, ax=axes[2,3])
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.palettes import Spectral6 
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

source = ColumnDataSource(data=dict(x=x, counts=counts, color=Spectral6))
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

import plotly.figure_factory as ff

a = data['tenure']

hist_data = [a]
group_labels = ['Distribution']

fig = ff.create_distplot(hist_data, group_labels)
py.iplot(fig, filename='Basic Distplot')
print("The skewness of SalePrice is {}".format(data['tenure'].skew()))
# Monthly Charges

m = data['MonthlyCharges']

import plotly.figure_factory as ff

hist_data = [m]
group_labels = ['Distribution of Monthly Charges']

fig = ff.create_distplot(hist_data, group_labels)
py.iplot(fig, filename='Basic Distplot')
print("The skewness of SalePrice is {}".format(data['MonthlyCharges'].skew()))
# Total Charges

t = data['TotalCharges']

import plotly.figure_factory as ff

hist_data = [t]
group_labels = ['Distribution of Total Charges']

fig = ff.create_distplot(hist_data, group_labels)
py.iplot(fig, filename='Basic Distplot')
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
churn     = data[data["Churn"] == "Yes"]
not_churn = data[data["Churn"] == "No"]
cat_cols_ch = data.nunique()[data.nunique() < 3].keys()
cat_cols_ch
df = data[cat_cols_ch]
# Males and Females who have not left Telco vs who left
#for i in range(12):
 #   gsschurn = data[((data['Churn']=='Yes' ) & (data[data.columns[i]]== pd.unique(data[data.columns[i]]).unique()[0]))|((data['Churn']=='Yes') & (data[data.columns[i]]== pd.unique(data[data.columns[i]]).unique()[1]))].groupby([data.columns[i]])[data.columns[i]].count()
  #  gsschurn
   # labels = (np.array(gsschurn.index))
    #sizes = (np.array((gsschurn / gsschurn.sum())*100))
    #colors = ['Red', 'Green']
    ##plt.subplots(figsize=(10, 8))
    ##plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ##plt.title("% of Churns")
   # plt.show()
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.palettes import Spectral6 
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

source = ColumnDataSource(data=dict(x=x, counts=counts, color=Spectral6))
p = figure(x_range=x, y_range=(0,50), plot_height=400, plot_width = 800, tools=[hover])
p.vbar(x='x', top='counts', width=0.9, color='color', legend="x", source=source)

p.xgrid.grid_line_color = None
p.legend.orientation = "horizontal"
p.legend.location = "top_center"

show(p)
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

p.circle('x', 'y', size=10, source=source)

show(p)

# sns.regplot(data.tenure, data.TotalCharges)
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models import HoverTool
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

p.circle('x', 'y', size=10, source=source)
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
fig_size[0] = 12
fig_size[1] = 12
plt.rcParams["figure.figsize"] = fig_size

sns.set(style="whitegrid")
ax = sns.boxplot(x=data['MonthlyCharges'], y=data['Churn'])
crosstab = pd.crosstab(data['Churn'], data['gender'])
crosstab
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
data.head(5)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

Idcol     = ['customerID']

#Target columns
target_col = ["Churn"]

#categorical columns
cat_col   = data.nunique()[data.nunique() < 6].keys().tolist()
cat_col   = [x for x in cat_col if x not in target_col]

#numerical columns
num_col   = [x for x in data.columns if x not in cat_col + target_col + Idcol]
              
#Binary columns with 2 values
bin_col = data.nunique()[data.nunique() == 2].keys().tolist()
              
#Columns more than 2 values
multi_col = [i for i in cat_col if i not in bin_col]

#Label encoding Binary columns
              
le = LabelEncoder()
for i in bin_col :
    data[i] = le.fit_transform(data[i])
    
#Duplicating columns for multi value columns
              
data = pd.get_dummies(data = data,columns = multi_col )

#Scaling Numerical columns
              
std = StandardScaler()
scaled = std.fit_transform(data[num_col])
scaled = pd.DataFrame(scaled,columns=num_col)

#dropping original values merging scaled values for numerical columns

df_data = data.copy()
data = data.drop(columns = num_col,axis = 1)
data = data.merge(scaled,left_index=True,right_index=True,how = "left")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score,recall_score

train,test = train_test_split(data,test_size = .25 ,random_state = 111)
    
cols    = [i for i in data.columns if i not in Idcol + target_col]
X_train = train[cols]
Y_train = train[target_col]
X_test  = test[cols]
Y_test  = test[target_col]
X_test.shape
X_train.shape
Y_test.shape
Y_train.shape
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_lg=sc.fit_transform(X_train)
X_test_lg=sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lgmodel = LogisticRegression()
lgmodel.fit(X_train_lg, Y_train)
y_pred = lgmodel.predict(X_test_lg)
print('Accuracy of logistic regression model on test data: {:.2f}'.format(lgmodel.score(X_test, Y_test)))
X = data[cols]
Y = data[target_col]
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
top_ten_sorted=top_three[np.argsort(coefs[top_three])]
print(data.columns.values[top_ten_sorted])
#data['Churn']= data.Churn.map(dict(Yes=1, No=0))
Y=data['Churn']
import statsmodels.api as sm
logit = sm.Logit(Y,X)

# fit the model
result = logit.fit()
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

print(result.summary())
np.exp(result.params)
import tensorflow as tf
learning_rate = 0.01
training_epochs = 1000

tf.reset_default_graph()
num_features = X_train.shape[1]
X = tf.placeholder(tf.float32, [None, num_features], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")
# Initialize our weigts & bias

W = tf.get_variable("W", [num_features, 1], initializer = tf.contrib.layers.xavier_initializer())
b = tf.get_variable("b", [1], initializer = tf.zeros_initializer())
Z = tf.add(tf.matmul(X, W), b)
prediction = tf.nn.sigmoid(Z)
# Calculate the cost
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
cost_history = np.empty(shape=[1],dtype=float)

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
               "W=", sess.run(W), "b=", sess.run(b))
        cost_history = np.append(cost_history, c)
        
        
    # Calculate predictions
    correct_prediction = tf.to_float(tf.greater(prediction, 0.5))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(Y, correct_prediction)))

    print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
from xgboost import XGBClassifier
xgb1 = XGBClassifier()
xgb1.fit(X_train, Y_train)
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
xgb1_pred = xgb1.predict(X_test)
xgb1_pred_prob = xgb1.predict_proba(X_test)
accuracy = accuracy_score(Y_test, xgb1_pred)
print('Accuracy = {:0.2f}%.'.format(accuracy))
importances = xgb1.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

plt.figure(figsize=(12,12))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.show()
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
#rs.fit(X_train, Y_train)
#best_est = rs.best_estimator_
#print(best_est)
xgb2 = XGBClassifier(colsample_bylevel= 0.6,
 colsample_bytree = 0.8,
 max_depth = 9,
 min_child_weight = 2, gamma= 1,
 n_estimators = 600, learning_rate=0.01, nthread = 1, reg_alpha = 0.1)
xgb2.get_params
xgb2.fit(X_train, Y_train)
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
xgb2_pred = xgb2.predict(X_test)
xgb2_pred_prob = xgb2.predict_proba(X_test)
accuracy = accuracy_score(Y_test, xgb2_pred)
print('Accuracy = {:0.2f}%.'.format(accuracy))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
results = model_selection.cross_val_score(xgb2, X_train, Y_train, cv=kfold, scoring='accuracy')
print("10-fold cross validation average accuracy: %.2f" % (results.mean()))
print("10-fold cross validation average accuracy: %.2f" % (results.mean()))
rfc = RandomForestClassifier(n_estimators=1000, max_depth=None)
rfc = rfc.fit(X_train, Y_train)
# Check for missing data

#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
#missing_data1 = X_train.isnull().sum(axis=0).reset_index()
#missing_data1
#X_train['TotalCharges'] = X_train['TotalCharges'].fillna((X_train['TotalCharges'].median()))
#check for missing data
#missing_data_test = X_test.isnull().sum(axis=0).reset_index()
#missing_data_test
#X_test['TotalCharges'] = X_test['TotalCharges'].fillna((X_test['TotalCharges'].median()))
bigrfc_predictions = rfc.predict(X_test)
bigrfc_predictions_prob = rfc.predict_proba(X_test)
accuracy_rf = accuracy_score(Y_test, bigrfc_predictions)
print('Accuracy = {:0.2f}%.'.format(accuracy_rf))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
results = model_selection.cross_val_score(rfc, X_train, Y_train, cv=kfold, scoring='accuracy')
print("10-fold cross validation average accuracy: %.2f" % (results.mean()))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(Y_test, lgmodel.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, lgmodel.predict_proba(X_test)[:,1])
plt.figure(figsize=(20,10))
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
rf_roc_auc = roc_auc_score(Y_test, rfc.predict(X_test))
rf_roc_auc
fpr, tpr, thresholds = roc_curve(Y_test, rfc.predict_proba(X_test)[:,1])
plt.figure(figsize=(20,10))
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
rf_confusion_matrix = confusion_matrix(Y_test, bigrfc_predictions)
print(rf_confusion_matrix)
import plotly.graph_objs as go

trace = go.Heatmap(z=rf_confusion_matrix,
                   x = ["Not churn","Churn"],
                   y = ["Not churn","Churn"],)
data1=[trace]
py.iplot(data1, filename='rf_heatmap')
# XGB

from sklearn.metrics import confusion_matrix
xgb_confusion_matrix = confusion_matrix(Y_test, xgb2_pred)
print(xgb_confusion_matrix)
import plotly.graph_objs as go

trace = go.Heatmap(z=xgb_confusion_matrix,
                   x = ["Not churn","Churn"],
                   y = ["Not churn","Churn"],)
data1=[trace]
py.iplot(data1, filename='xgb_heatmap')
# Logistic model

from sklearn.metrics import confusion_matrix
lg_confusion_matrix = confusion_matrix(Y_test, y_pred)
print(lg_confusion_matrix)
import plotly.graph_objs as go

trace = go.Heatmap(z=lg_confusion_matrix,
                   x = ["Not churn","Churn"],
                   y = ["Not churn","Churn"],)
data1=[trace]
py.iplot(data1, filename='lr_heatmap')