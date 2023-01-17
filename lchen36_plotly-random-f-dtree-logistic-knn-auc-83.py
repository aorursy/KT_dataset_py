#'''Importing Data Manipulation Modules'''
import numpy as np                 # Linear Algebra
import pandas as pd                # Data Processing, CSV file I/O (e.g. pd.read_csv)

#'''Seaborn and Matplotlib Visualization'''
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import seaborn as sns              # Python Data Visualization Library based on matplotlib
plt.style.use('fivethirtyeight')
%matplotlib inline

#'''Plotly Visualizations'''
import plotly as plotly                # Interactive Graphing Library for Python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.offline as py
init_notebook_mode(connected=True)
import os
%pylab inline
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")
df.head()
df.shape
df.describe()
df.drop('id', axis = 1, inplace = True)
df.info()
df.isnull().sum()
response_0 = df[df['Response'] == 0]
response_1 = df[df['Response'] == 1]
labels = sorted(df.Response.unique())
values = df.Response.value_counts().sort_index()
colors = ['DarkGrey', 'HotPink']


fig = go.Figure(data=[go.Pie(labels=labels,
                             values=values, pull=[0, 0.06])])
fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Distribution of Response")
fig.show()
labels = sorted(df.Gender.unique())
values = df.Gender.value_counts().sort_index()
colors = ['Aqua', 'Peru']


fig = go.Figure(data=[go.Pie(labels=labels,
                             values=values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Distribution of Gender")
fig.show()
trace1 = go.Histogram(
    x=response_0.Gender,
    opacity=0.85,
    name = "No response",
    marker=dict(color='DarkGrey',line=dict(color='#000000', width=2)))
trace2 = go.Histogram(
    x=response_1.Gender,
    opacity=0.85,
    name = "Response",
    marker=dict(color='Crimson',line=dict(color='#000000', width=2)))

data = [trace1, trace2]
layout = go.Layout(barmode='stack',
                   title='Gender - Response',
                   xaxis=dict(title='Gender'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
print('Min age: ', df['Age'].max())
print('Max age: ', df['Age'].min())
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'Age', data = df)
ax.set_xlabel('Age', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Age Count Distribution', fontsize=15)
sns.despine()
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'Age', data = df, orient = 'v', ax = ax1)
ax1.set_xlabel('People Age', fontsize=15)
ax1.set_ylabel('Age', fontsize=15)
ax1.set_title('Age Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

sns.distplot(df['Age'], ax = ax2)
sns.despine(ax = ax2)
ax2.set_xlabel('Age', fontsize=15)
ax2.set_ylabel('Occurence', fontsize=15)
ax2.set_title('Age x Ocucurence', fontsize=15)
ax2.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout()
print('1º Quartile: ', df['Age'].quantile(q = 0.25))
print('2º Quartile: ', df['Age'].quantile(q = 0.50))
print('3º Quartile: ', df['Age'].quantile(q = 0.75))
print('4º Quartile: ', df['Age'].quantile(q = 1.00))
#Calculate the outliers:
  # Interquartile range, IQR = Q3 - Q1
  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
    
print('Ages above: ', df['Age'].quantile(q = 0.75) + 
                      1.5*(df['Age'].quantile(q = 0.75) - df['Age'].quantile(q = 0.25)), 'are outliers')
print('Numerber of outliers: ', df[df['Age'] >= 85.0]['Age'].count())
print('Number of clients: ', len(df))
#Outliers in %
print('Outliers are:', round(df[df['Age'] >= 85.0]['Age'].count()*100/len(df),10), '%')
print('MEAN:', round(df['Age'].mean(), 1))

print('STD :', round(df['Age'].std(), 1))

#coefficient variation: (STD/MEAN)*100
#    cv < 15%, low dispersion
#    cv > 30%, high dispersion

print('CV  :',round(df['Age'].std()*100/df['Age'].mean(), 1), ', High dispersion')

#High dispersion means we have people with all ages and all of them are likely subscrib the service.
trace1 = go.Histogram(
    x=response_0.Age,
    opacity=0.85,
    name = "No response",
    marker=dict(color='DarkGrey',line=dict(color='#000000', width=2)))
trace2 = go.Histogram(
    x=response_1.Age,
    opacity=0.85,
    name = "Response",
    marker=dict(color='Crimson',line=dict(color='#000000', width=2)))

data = [trace1, trace2]
layout = go.Layout(barmode='stack',
                   title='Age - Response',
                   xaxis=dict(title='Age'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Histogram(
    x=response_0.Driving_License,
    opacity=0.85,
    name = "No response",
    marker=dict(color='DarkGrey',line=dict(color='#000000', width=2)))
trace2 = go.Histogram(
    x=response_1.Driving_License,
    opacity=0.85,
    name = "Response",
    marker=dict(color='Crimson',line=dict(color='#000000', width=2)))

data = [trace1, trace2]
layout = go.Layout(barmode='stack',
                   title='Driving License - Response',
                   xaxis=dict(title='Driving License'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Histogram(
    x=response_0.Region_Code,
    opacity=0.85,
    name = "No response",
    marker=dict(color='DarkGrey',line=dict(color='#000000', width=2)))
trace2 = go.Histogram(
    x=response_1.Region_Code,
    opacity=0.85,
    name = "Response",
    marker=dict(color='Crimson',line=dict(color='#000000', width=2)))

data = [trace1, trace2]
layout = go.Layout(barmode='stack',
                   title='Region - Response',
                   xaxis=dict(title='Region Code'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Histogram(
    x=response_0.Previously_Insured,
    opacity=0.85,
    name = "No response",
    marker=dict(color='DarkGrey',line=dict(color='#000000', width=2)))
trace2 = go.Histogram(
    x=response_1.Previously_Insured,
    opacity=0.85,
    name = "Response",
    marker=dict(color='Crimson',line=dict(color='#000000', width=2)))

data = [trace1, trace2]
layout = go.Layout(barmode='stack',
                   title='Previously Insured - Response',
                   xaxis=dict(title='Previously Insured'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
nodamage = df[df['Vehicle_Damage'] == 'No']
yesdamage = df[df['Vehicle_Damage'] == 'Yes']
trace1 = go.Histogram(
    x=response_0.Vehicle_Age,
    opacity=0.85,
    name = "No response",
    marker=dict(color='DarkGrey',line=dict(color='#000000', width=2)))
trace2 = go.Histogram(
    x=response_1.Vehicle_Age,
    opacity=0.85,
    name = "Response",
    marker=dict(color='Crimson',line=dict(color='#000000', width=2)))

data = [trace1, trace2]
layout = go.Layout(barmode='stack',
                   title='Vehicle Age - Response',
                   xaxis=dict(title='Vehicle Age'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Histogram(
    x=nodamage.Vehicle_Age,
    opacity=0.85,
    name = "No Damage",
    marker=dict(color='LightCyan',line=dict(color='#000000', width=2)))
trace2 = go.Histogram(
    x=yesdamage.Vehicle_Age,
    opacity=0.85,
    name = "Damaged",
    marker=dict(color='OrangeRed',line=dict(color='#000000', width=2)))

data = [trace1, trace2]
layout = go.Layout(barmode='stack',
                   title='Vehicle Damage - Vehicle Age',
                   xaxis=dict(title='Vehicle Age'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Histogram(
    x=nodamage.Gender,
    opacity=0.85,
    name = "No Damage",
    marker=dict(color='LightCyan',line=dict(color='#000000', width=2)))
trace2 = go.Histogram(
    x=yesdamage.Gender,
    opacity=0.85,
    name = "Damaged",
    marker=dict(color='DeepPink',line=dict(color='#000000', width=2)))

data = [trace1, trace2]
layout = go.Layout(barmode='stack',
                   title='Vehicle Damage - Gender',
                   xaxis=dict(title='Gender'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
sns.catplot(x = 'Vehicle_Age', y = 'Annual_Premium', hue = 'Vehicle_Damage',data = df)
#function to group gender into 0,1
def gender(dataframe):
    dataframe.loc[dataframe['Gender'] == 'Male', 'Gender'] = 0
    dataframe.loc[dataframe['Gender'] == 'Female', 'Gender'] = 1
    
    return dataframe

gender(df);

df['Gender'].value_counts()
#function to devide Age into 4 groups.
def age(dataframe):
    dataframe.loc[dataframe['Age'] <= 33, 'Age'] = 1
    dataframe.loc[(dataframe['Age'] > 33) & (dataframe['Age'] <= 52), 'Age'] = 2
    dataframe.loc[(dataframe['Age'] > 52) & (dataframe['Age'] <= 66), 'Age'] = 3
    dataframe.loc[(dataframe['Age'] > 66) & (dataframe['Age'] <= 85), 'Age'] = 4
           
    return dataframe

age(df)

df['Age'].value_counts()
rcParams['figure.figsize'] = 11.7,8.27
sns.distplot(df['Annual_Premium'])
print('1º Quartile: ', df['Annual_Premium'].quantile(q = 0.25))
print('2º Quartile: ', df['Annual_Premium'].quantile(q = 0.50))
print('3º Quartile: ', df['Annual_Premium'].quantile(q = 0.75))
print('4º Quartile: ', df['Annual_Premium'].quantile(q = 1.00))
#Calculate the outliers:
  # Interquartile range, IQR = Q3 - Q1
  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
    
print('Annual Premium above: ', df['Annual_Premium'].quantile(q = 0.75) + 
                      1.5*(df['Annual_Premium'].quantile(q = 0.75) - df['Annual_Premium'].quantile(q = 0.25)), 'are outliers')
print('Numerber of outliers: ', df[df['Annual_Premium'] >= 61892.5]['Annual_Premium'].count())
print('Number of clients: ', len(df))
#Outliers in %
print('Outliers are:', round(df[df['Annual_Premium'] >= 61892.5]['Annual_Premium'].count()*100/len(df),2), '%')
#function to devide Annual Premium into 4 groups.
def Premium(dataframe):
    dataframe.loc[dataframe['Annual_Premium'] <= 24405.0, 'Annual_Premium'] = 1
    dataframe.loc[(dataframe['Annual_Premium'] > 24405.0) & (dataframe['Annual_Premium'] <= 39400.0), 'Annual_Premium'] = 2
    dataframe.loc[(dataframe['Annual_Premium'] > 39400.0) & (dataframe['Annual_Premium'] <= 55000), 'Annual_Premium'] = 3
    dataframe.loc[(dataframe['Annual_Premium'] > 55000) & (dataframe['Annual_Premium'] <= 540165.0), 'Annual_Premium'] = 4
           
    return dataframe

Premium(df)

df['Annual_Premium'].value_counts()
print('1º Quartile: ', df['Vintage'].quantile(q = 0.25))
print('2º Quartile: ', df['Vintage'].quantile(q = 0.50))
print('3º Quartile: ', df['Vintage'].quantile(q = 0.75))
print('4º Quartile: ', df['Vintage'].quantile(q = 1.00))
#Calculate the outliers:
  # Interquartile range, IQR = Q3 - Q1
  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
    
print('Vintage above: ', df['Vintage'].quantile(q = 0.75) + 
                      1.5*(df['Vintage'].quantile(q = 0.75) - df['Vintage'].quantile(q = 0.25)), 'are outliers')

#function to devide Annual Premium into 4 groups.
def Vintage(dataframe):
    dataframe.loc[dataframe['Vintage'] <= 82.0, 'Vintage'] = 1
    dataframe.loc[(dataframe['Vintage'] > 82.0) & (dataframe['Vintage'] <= 154.0), 'Vintage'] = 2
    dataframe.loc[(dataframe['Vintage'] > 154.0) & (dataframe['Vintage'] <= 227.0), 'Vintage'] = 3
    dataframe.loc[(dataframe['Vintage'] > 227.0) & (dataframe['Vintage'] <= 450), 'Vintage'] = 4
           
    return dataframe

Vintage(df)

df['Vintage'].value_counts()
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

df['Vehicle_Age']  = labelencoder_X.fit_transform(df['Vehicle_Age']) 
df['Vehicle_Damage']  = labelencoder_X.fit_transform(df['Vehicle_Damage']) 
df['Gender'] = pd.to_numeric(df['Gender'])
df.dtypes
df.head()
len(df)
df_no_response = df[df['Response'] == 0]
df_response = df[df['Response'] == 1]
from sklearn.utils import resample

df_no_response_downsampled = resample(df_no_response,
                                      replace = False,
                                      n_samples=2500,
                                      random_state = 42)
len(df_no_response_downsampled)
df_response_downsampled = resample(df_response,
                                   replace = False,
                                   n_samples=2500,
                                   random_state = 42)
len(df_response_downsampled)
df_downsample = pd.concat([df_no_response_downsampled,df_response_downsampled])
len(df_downsample)
x = df_downsample.drop('Response', axis = 1)
y = df_downsample['Response']
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap='Purples')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
accuracies = {}
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_1 = RandomForestClassifier(random_state = 42) 
rfecv = RFECV(estimator=clf_rf_1, step=1, cv=k_fold,scoring='accuracy')   #10-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x.columns[rfecv.support_])
x_1 = df_downsample[['Region_Code','Previously_Insured','Vehicle_Damage','Policy_Sales_Channel']]
x_train, x_test, y_train, y_test = train_test_split(x_1,y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
clf_rf_2 = RandomForestClassifier(random_state=43)      
clr_rf_2 = clf_rf_2.fit(x_train,y_train)
ac = accuracy_score(y_test,clf_rf_2.predict(x_test))
accuracies['Random_Forest'] = ac

print('Accuracy is: ',ac, '\n')
cm = confusion_matrix(y_test,clf_rf_2.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('RFC Reports\n',classification_report(y_test, clf_rf_2.predict(x_test)))
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression() 
logmodel.fit(x_train,y_train)

ac = accuracy_score(y_test,logmodel.predict(x_test))
accuracies['Logistic regression'] = ac

print('Accuracy is: ',ac, '\n')
cm = confusion_matrix(y_test,logmodel.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('Logistic regression Reports\n',classification_report(y_test, logmodel.predict(x_test)))
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

#Neighbors
neighbors = np.arange(0,25)

#Create empty list that will hold cv scores
cv_scores = []

#Perform 10-fold cross validation on training set for odd values of k:
for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
    kfold = model_selection.KFold(n_splits=10, random_state=123)
    scores = model_selection.cross_val_score(knn, x_train, y_train, cv=k_fold, scoring='accuracy')
    cv_scores.append(scores.mean()*100)
    print("k=%d %0.2f (+/- %0.2f)" % (k_value, scores.mean()*100, scores.std()*100))

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print ("The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_scores[optimal_k]))

plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Train Accuracy')
plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=24)
knn.fit(x_train, y_train)

ac = accuracy_score(y_test,knn.predict(x_test))
accuracies['KNN'] = ac


print('Accuracy is: ',ac, '\n')
cm = confusion_matrix(y_test,knn.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('KNN Reports\n',classification_report(y_test, knn.predict(x_test)))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini
dtree.fit(x_train, y_train)

ac = accuracy_score(y_test,dtree.predict(x_test))
accuracies['decisiontree'] = ac

print('Accuracy is: ',ac, '\n')
cm = confusion_matrix(y_test,dtree.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('DecisionTree Reports\n',classification_report(y_test, dtree.predict(x_test)))
from sklearn.svm import SVC
svc = SVC()

svc1= SVC(random_state = 42,kernel = 'rbf')
svc1.fit(x_train, y_train)

ac = accuracy_score(y_test,svc1.predict(x_test))
accuracies['SVM'] = ac


print('Accuracy is: ',ac, '\n')
cm = confusion_matrix(y_test,svc1.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('SVM report\n',classification_report(y_test, svc1.predict(x_test)))
from sklearn.naive_bayes import GaussianNB
gaussiannb= GaussianNB()
gaussiannb.fit(x_train, y_train)

ac = accuracy_score(y_test,gaussiannb.predict(x_test))
accuracies['GaussianNB'] = ac


print('Accuracy is: ',ac,'\n')
cm = confusion_matrix(y_test,gaussiannb.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('GaussianNB report\n',classification_report(y_test, gaussiannb.predict(x_test)))
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

plt.rcParams['figure.figsize'] = (18,8)

x=list(accuracies.keys())
y=list(accuracies.values())

bars = plt.bar(x, height=y, width=.4, color = colors)

xlocs, xlabs = plt.xticks()

xlocs=[i for i in x]
xlabs=[i for i in x]

plt.xlabel('Algorithms', size = 20)
plt.ylabel('Accuracy %', size = 20)
plt.xticks(xlocs, xlabs, size = 15)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + .1, yval + .005, yval, size = 15)

plt.show()
fig, ax_arr = plt.subplots(nrows = 2, ncols = 3, figsize = (20,15))
from sklearn import metrics

#RandomForest
probs = clf_rf_2.predict_proba(x_test)
preds = probs[:,1]
fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds)
roc_aucrfc = metrics.auc(fprrfc, tprrfc)

ax_arr[0,0].plot(fprrfc, tprrfc, 'b', label = 'AUC = %0.2f' % roc_aucrfc)
ax_arr[0,0].plot([0, 1], [0, 1],'r--')
ax_arr[0,0].set_title('ROC Random Forest ',fontsize=20)
ax_arr[0,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,0].legend(loc = 'lower right', prop={'size': 16})

#LOGMODEL
probs = logmodel.predict_proba(x_test)
preds = probs[:,1]
fprlog, tprlog, thresholdlog = metrics.roc_curve(y_test, preds)
roc_auclog = metrics.auc(fprlog, tprlog)

ax_arr[0,1].plot(fprlog, tprlog, 'b', label = 'AUC = %0.2f' % roc_auclog)
ax_arr[0,1].plot([0, 1], [0, 1],'r--')
ax_arr[0,1].set_title('ROC Logistic ',fontsize=20)
ax_arr[0,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,1].legend(loc = 'lower right', prop={'size': 16})

#KNN
probs = knn.predict_proba(x_test)
preds = probs[:,1]
fprknn, tprknn, thresholdknn = metrics.roc_curve(y_test, preds)
roc_aucknn = metrics.auc(fprknn, tprknn)

ax_arr[0,2].plot(fprknn, tprknn, 'b', label = 'AUC = %0.2f' % roc_aucknn)
ax_arr[0,2].plot([0, 1], [0, 1],'r--')
ax_arr[0,2].set_title('ROC KNN ',fontsize=20)
ax_arr[0,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,2].legend(loc = 'lower right', prop={'size': 16})

#DECISION TREE
probs = dtree.predict_proba(x_test)
preds = probs[:,1]
fprdtree, tprdtree, thresholddtree = metrics.roc_curve(y_test, preds)
roc_aucdtree = metrics.auc(fprdtree, tprdtree)

ax_arr[1,0].plot(fprdtree, tprdtree, 'b', label = 'AUC = %0.2f' % roc_aucdtree)
ax_arr[1,0].plot([0, 1], [0, 1],'r--')
ax_arr[1,0].set_title('ROC Decision Tree ',fontsize=20)
ax_arr[1,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,0].legend(loc = 'lower right', prop={'size': 16})


#Gaussiannb

probs = gaussiannb.predict_proba(x_test)
preds = probs[:,1]
fprgau, tprgau, thresholdgau = metrics.roc_curve(y_test, preds)
roc_aucgau = metrics.auc(fprgau, tprgau)

ax_arr[1,1].plot(fprgau, tprgau, 'b', label = 'AUC = %0.2f' % roc_aucgau)
ax_arr[1,1].plot([0, 1], [0, 1],'r--')
ax_arr[1,1].set_title('ROC Gaussian ',fontsize=20)
ax_arr[1,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,1].legend(loc = 'lower right', prop={'size': 16})

#All plots
ax_arr[1,2].plot(fprrfc, tprrfc, 'b', label = 'rfc', color='black')
ax_arr[1,2].plot(fprlog, tprlog, 'b', label = 'Logistic', color='blue')
ax_arr[1,2].plot(fprknn, tprknn, 'b', label = 'Knn', color='brown')
ax_arr[1,2].plot(fprdtree, tprdtree, 'b', label = 'Decision Tree', color='green')
ax_arr[1,2].plot(fprgau, tprgau, 'b', label = 'Gaussiannb', color='grey')
ax_arr[1,2].set_title('Receiver Operating Comparison ',fontsize=20)
ax_arr[1,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,2].legend(loc = 'lower right', prop={'size': 16})