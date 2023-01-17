#importing libraries pandas and numpy

import pandas as pd
import numpy as np
#Read the dataset,make a copy,view top 5 rows of the copy.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/telecom-customer/Telecom_customer churn.csv')
df_new=df.copy()
df_new.head()
# Get the rows, columns and the feature list
print("Rows:", df_new.shape[0])
print("Columns:", df_new.shape[1])
features=df.columns.to_list()
print(features)
#Get the list of all columns and their number of unique values. 
unique_values=df.nunique().sort_values(ascending=True)
for key,value in unique_values.items():
    if unique_values[key]<50:
        print(key,value)
#Find the list of all categorical columns
cat_cols   = df_new.nunique()[df_new.nunique() < 7].keys().tolist()
cat_cols   = [x for x in cat_cols + ['crclscod','ethnic','area','dwllsize'] ]
cat_cols
#Find null value percentages in categorical variables. create a new data frame for df_cat
df_cat=df_new[cat_cols]
null_value_counts=(df_cat.isnull().sum()/1000).sort_values(ascending=False)
null_value_counts
# unique value for each cat column

for col in df_cat.columns:
    print(col, df_cat[col].unique())
#Relace nan with UNKW in hnd_webcap
df.hnd_webcap=df.hnd_webcap.replace(np.nan,'UNKW')
df_cat.hnd_webcap=df_cat.hnd_webcap.replace(np.nan,'UNKW')
df_cat.head()
#Imporg libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# definea function where you can pass feature as parameter to the function automatically to plot the graph for each categorical variable
def plot_columnwise(column):
    x,y = column, 'churn'
    df=df_cat[df_cat[column].notnull()]
    ax=sns.countplot(x=column,data=df,hue='churn')
    ax.set_title('{} Vs Churn'.format(column))
    ax.set_ylabel('Percentage of %{}'.format(column))
    ax.set_xlabel('{}'.format(column))
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height +300,'{:1.2f}%'.format((height/len(df)*100)),
            ha="center")
    plt.show()
#Vertical bar charts for features with categories < 5
cat_cols_bar   = df_new.nunique()[df_new.nunique() < 5].keys().tolist()
cat_cols_bar=[x for x in cat_cols_bar if x not in ['churn']]
for col in cat_cols_bar:
    plot_columnwise(col)
#Funciton for horizontal barchart when categorical features are too many to fit on vertical bar chart
def plot_columnwise_h(column):
    plt.figure(figsize=(10,8))
    x,y = column, 'churn'
    df=df_cat[df_cat[column].notnull()]
    ax=sns.countplot(y=column,data=df,hue='churn',orient="h")
    ax.set_title('{} Vs Churn'.format(column))
    ax.set_ylabel('Percentage of %{}'.format(column))
    ax.set_xlabel('{}'.format(column))
    total=len(df)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
    plt.show()
#Horizontal barchart except for crclscod
cat_cols_horizontal  = df_new.nunique()[df_new.nunique() < 4].keys().tolist()
cat_cols_horizontal=[x for x in cat_cols_horizontal if x not in ['churn','crclscod']]

for col in cat_cols_horizontal:
    plot_columnwise_h(col)
import plotly.graph_objects as go
from plotly.offline import iplot,init_notebook_mode
init_notebook_mode
import plotly.figure_factory as ff
#Plot churn value

values=df_cat.churn.value_counts().to_list()               
labels = ['Yes','No']
layout={'title':"Churn counts",'legend_title_text':'Churn','width':500,'height':400}
trace=go.Pie(labels=labels, values=values, hole=.3)
data=[trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
#horizontal barchart for crclscod
def plot_columnwise1(column):
    plt.figure(figsize=(10,50))
    x,y = column, 'churn'
    df=df_cat[df_cat[column].notnull()]
    ax=sns.countplot(y=column,data=df,hue='churn',orient="h")
    ax.set_title('{} Vs Churn'.format(column))
    ax.set_ylabel('Percentage of %{}'.format(column))
    ax.set_xlabel('{}'.format(column))
    total=len(df)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
        
            
    plt.show()

plot_columnwise1('crclscod')
#Remove the insignificant percentages and replace with other to have lesser bins/categories
crclscod_unique=df_cat.crclscod.unique().tolist()
crclscod_retain=['A','EA','C','B','CA','AA','U','E','E4','DA','D4','ZA','Z4','A2']
for key,value in df['crclscod'].items():
    if value not in crclscod_retain:
         df['crclscod'][key]='OTHER'
#Final categorical features
cat_features=['asl_flag','crclscod','refurb_new','hnd_webcap','area','ethnic','marital']
cat_features
# get the number of numerical columns
num_cols= [x for x in features if x not in (cat_cols+['Customer_ID'])]
num_cols.append('churn')

print(len(num_cols))
#create data set for churn and no churn

churn=df[df['churn']==1]
no_churn=df[df['churn']==0]
#Function to plot KDE plot 
def kdeplot(feature):
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(no_churn[feature].dropna(), color= 'navy', label= 'Churn: No')
    ax1 = sns.kdeplot(churn[feature].dropna(), color= 'red', label= 'Churn: Yes')
kdeplot('income')
kdeplot('lor')
# remove the >25% null values
num_cols= [x for x in num_cols if x not in ['income','lor']]
#Get null values. Create new data set df_2
df_2=df[cat_features+num_cols+['Customer_ID']]
print((df_2.isnull().sum()/1000).sort_values(ascending=False).head(20))
# add some interesting ratios and then start numerical analysis
#Ratios with revenues. Created 3 excel tabs with each bucket - revenue mou and qty and looked at relevant ratios
df_2['chngavg_rev_3moavg']=(df_2['avg3rev']-df_2['avgrev'])*100/df_2['avgrev']
df_2['chngavg_rev_6moavg']=(df_2['avg6rev']-df_2['avgrev'])*100/df_2['avgrev']
df_2['rev_adj_total_ratio']=df_2['adjrev']/df_2['totrev']

#Ratios with MOUS

df_2['chngavg_mou_3moavg']=(df_2['avg3mou']-df_2['avgmou'])*100/df_2['avgmou']
df_2['chngavg_mou_6moavg']=(df_2['avg6mou']-df_2['avgmou'])*100/df_2['avgmou']

#ratios with no of calls

df_2['chngavg_qty_3moavg']=(df_2['avg3rev']-df_2['avgrev'])*100/df_2['avgrev']
df_2['chngavg_qty_6moavg']=(df_2['avg6rev']-df_2['avgrev'])*100/df_2['avgrev']
df_2['qty_adj_total_ratio']=df_2['adjqty']/df_2['totcalls']
#Start with numerical analysis

numerical_cols=[x for x in df_2.columns if x not in cat_features + ['Customer_ID']]
corr=df_2[numerical_cols].corr()
plt.figure(figsize=(20, 20))
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 
                 linewidths=.2, cmap="YlGnBu")
#Arrange top correlaton by descending order and print the first 30. write it to a file for manual analysis
correlation=corr.abs().unstack().sort_values(ascending=False)
correlation=correlation[correlation!=1]
print(correlation[0:30])
correlation.to_csv('correlation.csv')
#Remove correlation>0.9


#Manually deciding which ones to remove. Automation removes but we have to make sure which ones we are removing to have more consistency
#keeping totals and remocing adj, or break overs like voie and data
to_remove=['adjqty','adjmou','adjrev','attempt_Mean','comp_vce_Mean','vceovr_Mean','comp_dat_Mean',
          'ccrndmou_Mean','inonemin_Mean','avg3qty','avg3mou','opk_dat_Mean','avg3rev','totmou','plcd_vce_Mean',
           'ovrmou_Mean','mou_opkd_Mean','peak_vce_Mean','peak_dat_Mean','avg6mou','avg6qty']

numerical_cols=[x for x in numerical_cols if x not in to_remove]
#removing 11 columns and running correlation again
corr=df_2[numerical_cols].corr()



plt.figure(figsize=(20, 20))
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 
                 linewidths=.2, cmap="YlGnBu")
len(numerical_cols)

#Arrange top correlaton by descending order and print the first 30. write it to a file for manual analysis
correlation1=corr.abs().unstack().sort_values(ascending=False)
correlation1=correlation1[correlation1!=1]
print(correlation1[0:30])
correlation.to_csv('correlation1.csv')
#create  churn and no churn datasets for df_2
churn=df_2[df_2['churn']==1]
no_churn=df_2[df_2['churn']==0]
#Plot for each and every numerical feature vs churn
fig, axes = plt.subplots(ncols=5, nrows=12, figsize=(25,60))

for feature, ax in zip(numerical_cols, axes.flat):
    sns.kdeplot( no_churn[feature].dropna(), color= 'navy', label= 'Churn: No',ax=ax,bw=0.1)
    sns.kdeplot(churn[feature].dropna(), color= 'red', label= 'Churn: Yes',ax=ax,bw=0.1)
    ax.set_title("KDE for {}".format(feature))
plt.show()
# get the rows and columns
df_2.shape
#dropping numerical columns which have same kde distribution for churn vs non churn

to_drop_cols=['mou_Mean','unan_vce_Mean','mou_rvce_Mean','avgmou','avgqty','avg6rev','models','rev_adj_total_ratio',
              'qty_adj_total_ratio','totrev','opk_vce_Mean','complete_vce_ratio']

numerical_cols=[x for x in numerical_cols if x not in to_drop_cols]
len(numerical_cols)
#Get % of null value rows
null_value_counts=(df_2.isnull().sum()/1000).sort_values(ascending=False)

print(null_value_counts)
null_values=list(null_value_counts[null_value_counts>0].keys())
#Drop the null values and validate
df_2=df_2.dropna(axis=0)

null_value_counts=(df_2.isnull().sum()/1000).sort_values(ascending=False)
null_value_counts
#Merge all features that are selected
all_cols=numerical_cols+ cat_features+['Customer_ID']
all_cols=list(dict.fromkeys(all_cols))
len(all_cols)
print(all_cols)
#Copy only the interesting folders to a new data frame
df_3=df_2[all_cols]
print(df_3.shape)
#Validate that there are no nulls in the data frame
null_value_counts=(df_3.isnull().sum()/1000).sort_values(ascending=False)
null_values=list(null_value_counts[null_value_counts>0].keys())
null_values
#Reser the index as we removed some columns. Make a copy into df_4
df_3=df_3.reset_index()
df_4=df_3.copy()
# Label encoding for binary features

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


le=LabelEncoder()
df_4['asl_flag']=le.fit_transform(df_4['asl_flag'])
df_4['refurb_new']=le.fit_transform(df_4['refurb_new'])


#Hot encoding for remaining categorical features
df_4= pd.get_dummies(data = df_4, drop_first=True,columns =['crclscod','hnd_webcap','area','ethnic','marital'])
#Scaling Numerical columns
numerical_cols=[x for x in numerical_cols if x not in ['churn','Customer_ID']]
std = StandardScaler()
scaled = std.fit_transform(df_4[numerical_cols])
scaled = pd.DataFrame(scaled,columns=numerical_cols)
#Drop numerical columns as we have created a scaled copy in scaled dataframe
df_4= df_4.drop(columns = numerical_cols,axis = 1)
#Merge scaled copy and df_4 into a new dataframe. We will use this dataframe for our training, testing and model implmentation
df_5 = df_4.merge(scaled,left_index=True,right_index=True,how = "left")
df_5=df_5.drop('Customer_ID',axis=1)
df_5.shape
# Split data set into training set and test set
from sklearn.model_selection import train_test_split
train,test = train_test_split(df_5,test_size = .25 ,random_state = 0)
X_train=train.drop(['churn'],axis=1)
X_test=test.drop(['churn'],axis=1)

y_train=train['churn']
y_test=test['churn']
#Import the libraries for all the models needed
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.metrics import classification_report,roc_curve
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
#create an empty dataframe to store metrics from each model
metrics=pd.DataFrame(columns=['accuracy','roc_auc_score','f1_score','precision_score','recall_score'])
#Baseline model with graphs
def classification_model(classifier,X_train,y_train,X_test,y_test,name):
    global metrics
    # Fitting the training set to model and predicting on the test set
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)
    
    # Create cpnfusion matrix and store other metrics in variables. Write the metrics to data frame
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy=accuracy_score(y_test, y_pred)
    roc_auc= roc_auc_score(y_test,y_pred)
    f1score=f1_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    
    #Writing to metrics data frame
    metrics.loc[name]=[accuracy,roc_auc,f1score,precision,recall]

    # calculate roc curves
    
    fpr, tpr, _ = roc_curve(y_test, y_prob[:,1], pos_label=1)

    
    #Print the Classification report
    print(name,' Metrics')
    print('Confusion Matrix: \n' , cm)
    print('Classificaiton Report:\n',classification_report(y_test,y_pred))
    
    
    #Plot ROC AUC and Confusion Matrix
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(15,5))
    sns.heatmap(cm,annot=True,xticklabels=['Churn No','Churn Yes'], ax=axes[0],
                    yticklabels=['Churn No','Churn Yes'],cmap='viridis',fmt='g'
                )
    axes[0].set_title('{} Confusion Matrix'.format(name))
    
    sns.scatterplot(fpr,tpr,color='r',label = "AUC = " + str(np.around(roc_auc_score(y_test,y_prob[:,1]),3)),ax=axes[1])
    sns.lineplot(x=[0,1], y=[0,1],ax=axes[1])
    axes[1].lines[0].set_linestyle("--")
    axes[1].set_title('{} ROC Curve'.format(name))
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    
    #Get the top features
    
    if name in ['Logistic Regression','SVC Linear','Decision Tree', 'Random Forest'] :
        if name in ['Logistic Regression','SVC Linear']:
            coeff_df=[]
            coeff_df=pd.DataFrame(classifier.coef_.ravel())
        elif name in ['Decision Tree', 'Random Forest'] :
            coeff_df=[]
            coeff_df  = pd.DataFrame(classifier.feature_importances_)
        coeff_df.columns=['coefficients']
        df_cols=pd.DataFrame((X_train.columns))
        df_cols.columns=['features']

        coeff_df=pd.merge(coeff_df,df_cols,left_index=True,right_index=True,how='left')
        coeff_df=pd.DataFrame.sort_values(coeff_df,axis=0,by='coefficients')
        print('Feature importance - top 6 positive coefficients' )
        print(coeff_df.tail(6))
        print('Feature importance top 6 negative coefficients')
        print(coeff_df.head(6))
    
    



#SVC models take a long time to run on huge data sets. so ignoring this for now
'''
classifiers = {
        'SVC Linear': SVC(kernel = 'linear', random_state = 0),
        'SVC rbf': SVC(kernel = 'rbf', random_state = 0)


}

for index, (name, classifier) in enumerate(classifiers.items()):
    classification_model(classifier,X_train,y_train,X_test,y_test,name,)
    '''
#All Classifiers Except SVC because it takes a lot of time to run on a 100,000 dataset

classifiers = {
        'Logistic Regression': LogisticRegression(random_state=0,solver='liblinear'),
    

    
        'KNN Classifier': KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                           weights='uniform'),

        'Decision Tree': DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
    
        'Random Forest': RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
        'XGBoost': XGBClassifier()
}

for index, (name, classifier) in enumerate(classifiers.items()):
    classification_model(classifier,X_train,y_train,X_test,y_test,name,)
# Artificial Neural Network model

# Initializing the ANN
ann = tf.keras.models.Sequential()

from keras.layers.core import Dropout
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=1024, activation='relu'))
ann.add(tf.keras.layers.Dropout(0.2))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=512, activation='relu'))
ann.add(tf.keras.layers.Dropout(0.2))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=256, activation='relu'))
ann.add(tf.keras.layers.Dropout(0.2))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=128, activation='relu'))
ann.add(tf.keras.layers.Dropout(0.2))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
fit_keras=ann.fit(X_train, y_train, batch_size = 32, epochs = 20,validation_data=(X_test, y_test))

ann.summary()
# Part 4 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
# Predicting the result of a single observation

accuracy = ann.evaluate(X_test, y_test, verbose=False)
print("Testing Score: {:.4f}".format(accuracy[0]))
print("Testing Accuracy: {:.4f}".format(accuracy[1]))


accuracy = ann.evaluate(X_train, y_train, verbose=False)
print("Training Score: {:.4f}".format(accuracy[0]))
print("Training Accuracy: {:.4f}".format(accuracy[1]))
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train1=X_train
X_test1=X_test
X_train1 = pca.fit_transform(X_train1)
X_test1 = pca.transform(X_test1)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classification_model(classifier,X_train1,y_train,X_test1,y_test,'PCA')
metrics
