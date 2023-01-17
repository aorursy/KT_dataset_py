#import basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
init_notebook_mode(connected=True)

plt.style.use('fivethirtyeight')

#import the dataset
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#snapshot of data
data.head()
#column datatypes
data.info()
#counts of customer churn cases vs not churn in dataset
target= data['Churn'].value_counts()
levels = ['No','Yes']
trace = go.Pie(labels=target.index,values=target.values,
               marker=dict(colors=('orange','green')))
layout = dict(title="Telco Customer Churn Ratio", margin=dict(l=150), width=500, height=500)
figdata = [trace]
fig = go.Figure(data=figdata, layout=layout)
iplot(fig)
#print target class counts
print(target)

#Let's visualize the churn on the basis of Gender
def bar_plot(col,data,barmode='group',width=800,height=600,color1='orange',color2='purple'):
    values = list(data[col].value_counts().keys())
    if values ==[0,1]:
        data[col].replace(0,'No',inplace=True)
        data[col].replace(1,'Yes',inplace=True)
        values = list(data[col].value_counts().keys())
    tr1 = data[data[col]==values[0]]['Churn'].value_counts().to_dict()
    tr2 = data[data[col]==values[1]]['Churn'].value_counts().to_dict()
    xx = ['Male', 'Female']
    trace1 = go.Bar(y=[tr1['No'], tr2['No']], name="Not Churn", x=values, marker=dict(color=color1))
    trace2 = go.Bar(y=[tr1['Yes'], tr2['Yes']], name="Churn", x=values, marker=dict(color=color2))
    data = [trace1, trace2]
    layout = go.Layout(
        barmode=barmode,xaxis = dict(title=col),yaxis=dict(title='Count'),
    title='Effect of '+ col + ' on Customer Churn',width=width,height=height)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

#Comparison of churn between male and female
bar_plot('gender',data)
#Let's visualize the churn ratio for senior citizens
bar_plot('SeniorCitizen',data,barmode='stack',width=600,height=400,color1='orange',color2='green')
#let's visualize the impact of having partner on customer churn
bar_plot('Partner',data,barmode='stack',width=600,height=400,color1='blue',color2='pink')
#effect of having dependents on churn
bar_plot('Dependents',data,barmode='stack',width=600,height=400)
#effect of phone service on churn
bar_plot('PhoneService',data)
#let's check effect of PaperlessBilling
bar_plot('PaperlessBilling',data)
# values = list(data['gender'].value_counts().keys())
# tr1 = data[data['gender']==values[0]]['Churn'].value_counts().to_dict()
# tr1['No']
#counts of billing frequency or contacts
fig = plt.gcf()
fig.set_size_inches( 7, 5)
plt.title('Counts of billing frequencies')
sns.countplot(data['Contract'])

#Use of differnt Internet service lines
fig = plt.gcf()
fig.set_size_inches( 7, 5)
plt.title('Counts of different interent service lines')
sns.countplot(data['InternetService'])
#Churn ratio with respect to internet service type
fig = plt.gcf()
plt.title('Churn ratio with respect to internet service type')
fig.set_size_inches( 8, 6)
sns.countplot(data['InternetService'],hue=data['Churn'])
#counts of different bill payment methods using pie chart
target= data['PaymentMethod'].value_counts()
levels = ['Electronic check','Mailed check','Bank transfer','Credit card']
trace = go.Pie(labels=target.index,values=target.values
               )
layout = dict(title="Telco Customer Payment Method", margin=dict(l=50), width=800, height=500)
figdata = [trace]
fig = go.Figure(data=figdata, layout=layout)
iplot(fig)
#Churn ratio analysis for different bill payment method
fig = plt.gcf()
fig.set_size_inches( 12, 8)
plt.title('Churn ratio analysis for different bill payment method')
sns.countplot(data['PaymentMethod'],hue=data['Churn'])
data['OnlineSecurity'].value_counts()
internet_features = ['OnlineSecurity','OnlineBackup' ,'DeviceProtection' ,
                     'TechSupport' ,'StreamingTV' ,'StreamingMovies','InternetService']
#replace No internet service with No
data[internet_features]=data[internet_features].replace('No internet service','No')
#let's verify it
data['OnlineSecurity'].value_counts()
#churn ratio for column Online security
bar_plot('OnlineSecurity',data)
#Churn ratio for StreamingTV
bar_plot('StreamingTV',data)
#churn ratio for people having StreamingMovie service
bar_plot('StreamingMovies',data)
#Churn ratio for feature tech support
bar_plot('TechSupport',data)
#churn ratio for column onlinebackup
bar_plot('OnlineBackup',data)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='Churn', y = 'tenure', data=data)
ax.set_title('Effect of Tenure length on Churn', fontsize=18)
ax.set_ylabel('Tenure', fontsize = 15)
ax.set_xlabel('Churn', fontsize = 15)
plt.figure(figsize=(10,6))
ax = sns.boxplot(x='Churn', y = 'MonthlyCharges', data=data)
ax.set_title('Effect of Monthly Charges on Churn', fontsize=18)
ax.set_ylabel('Charges', fontsize = 15)
ax.set_xlabel('Churn', fontsize = 15)
# Converting Total Charges to a numerical data type.
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')
#Let's check for nulls first
nulls = data.isnull().sum()
nulls[nulls > 0]
#impute missing values with 0
data.fillna(0,inplace=True)
#new feature - Internet(Yes- have internet service, No- do not have internet service)
data['Internet'] = data['InternetService'].apply(lambda x : x if x=='No' else 'Yes')
data['Internet'].value_counts()
data['MultipleLines'].value_counts()
#replace No phone service with No
data['MultipleLines'].replace('No phone service','No',inplace=True)
#train and target
y = data['Churn'].map({'Yes':1,'No':0})
X = data.drop(labels=['Churn','customerID'],axis=1).copy()
#find list of categorical columns for encoding
cat_cols = []
for column in X.columns:
    if column not in ['tenure','MonthlyCharges','TotalCharges']:
        cat_cols.append(column)
#Convert categorical columns to binary
X= pd.get_dummies(X,columns=cat_cols)

#shape after conversion of categorical features
X.head()
#import ML models and metrics
from sklearn.metrics import f1_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#create seperate train and test splits for validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#create function for validation and return accuracy and roc-auc score
def evaluate_model(model):
    model.fit(X_train,y_train)
    prediction_test = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction_test)
    rocauc = metrics.roc_auc_score(y_test, prediction_test)
    return accuracy,rocauc,prediction_test
# Running logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.1)
acc,rocauc,testpred_lr  = evaluate_model(lr)
print('Logistic Regression...')
print('Accuracy score :',acc)
print('ROC-AUC score :',rocauc)
rf =RandomForestClassifier()
rf.fit(X_train,y_train)
acc,rocauc,testpred_rf  = evaluate_model(rf)
print('Random Forest...')
print('Accuracy score :',acc)
print('ROC-AUC score :',rocauc)
#set up search grid
#Number of search trees
n_estimators=range(50,100)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = range(4,8)
# Minimum number of samples required to split a node
min_samples_split = range(2,6)
# Minimum number of samples required at each leaf node
min_samples_leaf = range(1,5)
# Method of selecting samples for training each tree
bootstrap = [True, False]
#criterion
criterion=['gini','entropy']
#create the random grid
random_grid = {'n_estimators':n_estimators,
              'max_features':max_features,
              'max_depth':max_depth,
              'min_samples_split':min_samples_split,
              'min_samples_leaf':min_samples_leaf,
              'bootstrap':bootstrap,
              'criterion':criterion}
print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(random_state=2018)

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 3, verbose=2, n_iter=100,random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train,y_train)
#best params
params = rf_random.best_params_
params
rfc = RandomForestClassifier(**params,random_state=42)
rfc.fit(X_train,y_train)
acc,rocauc,testpred_rfc  = evaluate_model(rfc)
print('Random Forest Optimized...')
print('Accuracy score :',acc)
print('ROC-AUC score :',rocauc)
indices = np.argsort(rfc.feature_importances_)[::-1]
indices = indices[:45]

# Visualise these with a barplot
plt.subplots(figsize=(20, 15))
g = sns.barplot(y=X.columns[indices], x = rfc.feature_importances_[indices], orient='h')
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("RandomForest feature importance");
#we define a plot_multiple_roc to visualise all the model curves together

def plot_multiple_roc(y_preds, y_test, model_names):
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    
    for i in range (0, len(y_preds)):
        false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_preds[i])
        label = ""
        if len(model_names) > i:
            label = model_names[i]
        ax.plot(false_positive_rate, true_positive_rate, label=label)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)
    ax.grid(True)
    
    ax.set(title='ROC Curves for telecom customer churn problem',
           xlabel = 'False positive Rate', ylabel = 'True positive rate')
        
    if len(model_names) > 0:
        plt.legend(loc=4)
validation_probs_fs = []
validation_probs_fs.append(testpred_lr)
validation_probs_fs.append(testpred_rfc)
all_models_names = ['logistic reg', 'Random_forest']
plot_multiple_roc(validation_probs_fs, y_test, all_models_names)
