## Data Analysis Phase
## Main aim is to understand more about the data
import pandas as pd
import numpy as np

import sklearn 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#Metrics Libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as pylab
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
customer=pd.read_csv("../input/online-shoppers-intention/online_shoppers_intention.csv")
customer_copy=customer.copy()

## print shape of dataset with rows and columns and information 
print ("The shape of the  data is (row, column):"+ str(customer_copy.shape))
print (customer_copy.info())
customer_copy.head()
customer_copy.describe()
import missingno as msno 
msno.matrix(customer_copy)
print('Data columns with null values:',customer_copy.isnull().sum(), sep = '\n')
for cols in ['Administrative','Informational','ProductRelated']:
    customer_copy[cols].replace(0, np.nan, inplace= True)
for cols in ['Administrative','Informational','ProductRelated']:
    print('{} null values:'.format(cols),customer_copy[cols].isnull().sum(), sep = '\n')
for cols in ['Administrative','Informational','ProductRelated']:
    median_value=customer_copy[cols].median()
    customer_copy[cols]=customer_copy[cols].fillna(median_value)
for cols in ['Administrative','Informational','ProductRelated']:
    print('{} null values:'.format(cols),customer_copy[cols].isnull().sum(), sep = '\n')
for cols in ['Administrative_Duration','Informational_Duration','ProductRelated_Duration']:
    customer_copy[cols].replace(0, np.nan, inplace= True)
for cols in ['Administrative_Duration','Informational_Duration','ProductRelated_Duration']:
    customer_copy[cols].replace(-1, np.nan, inplace= True)
for cols in ['Administrative_Duration','Informational_Duration','ProductRelated_Duration']:
    print('{} null values:'.format(cols),customer_copy[cols].isnull().sum(), sep = '\n')
for cols in ['Administrative_Duration','Informational_Duration','ProductRelated_Duration','BounceRates','ExitRates']:
    mean_value=customer_copy[cols].mean()
    customer_copy[cols]=customer_copy[cols].fillna(mean_value)
for cols in ['Administrative_Duration','Informational_Duration','ProductRelated_Duration','BounceRates','ExitRates']:
    print('{} null values:'.format(cols),customer_copy[cols].isnull().sum(), sep = '\n')
plt.figure(figsize = (15, 5))
#plt.style.use('seaborn-white')
plt.subplot(131)
sns.scatterplot(x="Administrative", y="Administrative_Duration",hue="Revenue", data=customer_copy)
plt.subplot(132)
sns.scatterplot(x="Informational", y="Informational_Duration",hue="Revenue", data=customer_copy)
plt.subplot(133)
sns.scatterplot(x="ProductRelated", y="ProductRelated_Duration",hue="Revenue", data=customer_copy)
fig, ax1 = plt.subplots(figsize=(10,6))
color = 'tab:green'
ax1.set_title('Average Page Value by Month', fontsize=16)
ax1.set_xlabel('Month', fontsize=16)
ax1.set_ylabel('Avg Temp', fontsize=16, color=color)
ax2 = sns.barplot(x='Month', y='PageValues', data = customer_copy, palette='summer',hue='Revenue')
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Avg Percipitation %', fontsize=16, color=color)
ax2 = sns.lineplot(x='Month', y='PageValues', data = customer_copy, sort=False, color=color)
ax2.tick_params(axis='y', color=color)
plt.show()
sns.relplot(x="BounceRates", y="ExitRates",col="Revenue",hue="Revenue",style="Weekend", data=customer_copy)
sns.catplot(x="VisitorType", y="ExitRates",
                hue="Weekend", col="Revenue",
                data=customer_copy, kind="box");
sns.heatmap(customer_copy.corr(),annot=True,fmt='.1g',cmap='Greys')
feature_customer=customer_copy.copy()
plt.figure(figsize = (15, 10))
plt.style.use('seaborn-white')
ax=plt.subplot(231)
plt.boxplot(feature_customer['BounceRates'])
ax.set_title('BounceRates')
ax=plt.subplot(232)
plt.boxplot(feature_customer['ExitRates'])
ax.set_title('ExitRates')
ax=plt.subplot(233)
plt.boxplot(feature_customer['Administrative_Duration'])
ax.set_title('Administrative_Duration')
ax=plt.subplot(234)
plt.boxplot(feature_customer['Informational_Duration'])
ax.set_title('Informational_Duration')
ax=plt.subplot(235)
plt.boxplot(feature_customer['ProductRelated_Duration'])
ax.set_title('ProductRelated_Duration')
ax=plt.subplot(236)
plt.boxplot(feature_customer['PageValues'])
ax.set_title('PageValues')
numerical_features=['BounceRates','ExitRates','Administrative_Duration','ProductRelated_Duration']
for cols in numerical_features:
    Q1 = feature_customer[cols].quantile(0.25)
    Q3 = feature_customer[cols].quantile(0.75)
    IQR = Q3 - Q1     

    filter = (feature_customer[cols] >= Q1 - 1.5 * IQR) & (feature_customer[cols] <= Q3 + 1.5 *IQR)
    feature_customer=feature_customer.loc[filter]
plt.figure(figsize = (15, 10))
plt.style.use('seaborn-white')
ax=plt.subplot(221)
plt.boxplot(feature_customer['BounceRates'])
ax.set_title('BounceRates')
ax=plt.subplot(222)
plt.boxplot(feature_customer['ExitRates'])
ax.set_title('ExitRates')
ax=plt.subplot(223)
plt.boxplot(feature_customer['Administrative_Duration'])
ax.set_title('Administrative_Duration')
ax=plt.subplot(224)
plt.boxplot(feature_customer['ProductRelated_Duration'])
ax.set_title('ProductRelated_Duration')
feature_customer.loc[feature_customer['SpecialDay'] > 0.4, 'SpecialDay'] = 1
feature_customer.loc[feature_customer['SpecialDay'] <= 0.4, 'SpecialDay'] = 0
feature_customer['SpecialDay'].value_counts()
feature_customer['SpecialDay']=feature_customer['SpecialDay'].astype('bool')
feature_customer['SpecialDay'].value_counts()
for cols in ['Administrative','Informational','ProductRelated','OperatingSystems','Browser',
             'Region','TrafficType','VisitorType']:
    feature_customer[cols] = feature_customer[cols].astype('category')
feature_customer.dtypes
Categorical_variables=['Weekend','Revenue','Administrative','Informational','ProductRelated','SpecialDay',
 'OperatingSystems','Browser','Region','Month','TrafficType','VisitorType']


feature_scale=[feature for feature in feature_customer.columns if feature not in Categorical_variables]


scaler=StandardScaler()
scaler.fit(feature_customer[feature_scale])
scaled_data = pd.concat([feature_customer[['Weekend','Revenue','Administrative','Informational',
                                    'ProductRelated','SpecialDay','OperatingSystems',
                                    'Browser','Region','Month','TrafficType','VisitorType']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(feature_customer[feature_scale]), columns=feature_scale)],
                    axis=1)
scaled_data.head()
encoded_features=['Month','VisitorType']

label_data = scaled_data.copy()
label_encoder = LabelEncoder()
for col in encoded_features:
    label_data[col] = label_encoder.fit_transform(scaled_data[col])
    
label_data.head()
X=label_data.drop(['Revenue'],axis=1)
y=label_data.Revenue

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(17).plot(kind='barh')
plt.show()
X=label_data.drop(['SpecialDay','VisitorType','Weekend','Revenue'],axis=1)
y=label_data.Revenue

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=1)
print("Input Training:",X_train.shape)
print("Input Test:",X_test.shape)
print("Output Training:",y_train.shape)
print("Output Test:",y_test.shape)
#creating the objects
logreg_cv = LogisticRegression(random_state=0)
dt_cv=DecisionTreeClassifier()
rt_cv=RandomForestClassifier()
knn_cv=KNeighborsClassifier()
svc_cv=SVC(kernel='linear')
nb_cv=BernoulliNB()
cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest',3:'KNN',4:'SVC',5:'Naive Bayes'}
cv_models=[logreg_cv,dt_cv,rt_cv,knn_cv,svc_cv,nb_cv]


for i,model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model, X, y, cv=10, scoring ='accuracy').mean()))
#Creating the pipeline
pipeline_lr=Pipeline([('lr_classifier',LogisticRegression(random_state=0))])
pipeline_dt=Pipeline([('dt_classifier',DecisionTreeClassifier())])
pipeline_randomforest=Pipeline([('rf_classifier',RandomForestClassifier())])
pipeline_knn=Pipeline([('knn_classifier',KNeighborsClassifier())])
pipeline_svc=Pipeline([('svc_classifier',SVC(kernel='linear'))])
pipeline_nb=Pipeline([('nb_classifier',BernoulliNB())])

#Assigning the pipeline and relevant outcome variable
pipelines = [pipeline_lr, pipeline_dt, pipeline_randomforest,pipeline_knn,pipeline_svc,pipeline_nb]
best_accuracy=0.0
best_classifier=0
best_pipeline=""

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest',3:'KNN',4:'SVC',5:'Naive Bayes'}

# Fit the pipelines
for pipe in pipelines:
	pipe.fit(X_train, y_train)

#Evaluating each model
for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))
#Choosing the best model for our problem
for i,model in enumerate(pipelines):
    if model.score(X_test,y_test)>best_accuracy:
        best_accuracy=model.score(X_test,y_test)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))
# Create a pipeline
pipe = make_pipeline((RandomForestClassifier()))
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [{"randomforestclassifier": [RandomForestClassifier()],
                 "randomforestclassifier__n_estimators": [10, 100, 1000],
                 "randomforestclassifier__max_depth":[5,8,15,25,30,None],
                 "randomforestclassifier__min_samples_leaf":[1,2,5,10,15,100],
                 "randomforestclassifier__max_leaf_nodes": [2, 5,10]}]
#Gridsearch of the pipeline, the fit the best model
gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1) # Fit grid search
best_model = gridsearch.fit(X_train,y_train)
print(best_model.best_estimator_)
print("The mean accuracy of the model is:",best_model.score(X_test,y_test))
rt=RandomForestClassifier(max_depth=30, max_leaf_nodes=10,min_samples_leaf=15)
rt.fit(X_train,y_train)
y_pred=rt.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
svc_classifier = SVC(kernel='linear',random_state = 0)
svc_classifier.fit(X_train,y_train)
y_pred=svc_classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))