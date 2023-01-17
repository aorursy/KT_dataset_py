import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

# Import stats from scipy

from scipy import stats
df = pd.read_csv("../input/insurance-data/insurance_part2_data (2).csv")
df.head()
df.info()
# Are there any missing values ?

df.isnull().sum()
df.describe().T
## Intital descriptive analysis of the data



df.describe(percentiles=[.25,0.50,0.75,0.90]).T
df.describe(include='all').T
df.head(10)
df.tail(10)
### data dimensions



df.shape
for column in df[['Agency_Code', 'Type', 'Claimed', 'Channel', 

                  'Product Name', 'Destination']]:

    print(column.upper(),': ',df[column].nunique())

    print(df[column].value_counts().sort_values())

    print('\n')
# Are there any duplicates ?

dups = df.duplicated()

print('Number of duplicate rows = %d' % (dups.sum()))

df[dups]
print('Range of values: ', df['Age'].max()-df['Age'].min())
#Central values 

print('Minimum Age: ', df['Age'].min())

print('Maximum Age: ',df['Age'].max())

print('Mean value: ', df['Age'].mean())

print('Median value: ',df['Age'].median())

print('Standard deviation: ', df['Age'].std())

print('Null values: ',df['Age'].isnull().any())
#Quartiles



Q1=df['Age'].quantile(q=0.25)

Q3=df['Age'].quantile(q=0.75)

print('spending - 1st Quartile (Q1) is: ', Q1)

print('spending - 3st Quartile (Q3) is: ', Q3)

print('Interquartile range (IQR) of Age is ', stats.iqr(df['Age']))
#Outlier detection from Interquartile range (IQR) in original data



# IQR=Q3-Q1

#lower 1.5*IQR whisker i.e Q1-1.5*IQR

#upper 1.5*IQR whisker i.e Q3+1.5*IQR

L_outliers=Q1-1.5*(Q3-Q1)

U_outliers=Q3+1.5*(Q3-Q1)

print('Lower outliers in Age: ', L_outliers)

print('Upper outliers in Age: ', U_outliers)
print('Number of outliers in Age upper : ', df[df['Age']>57.0]['Age'].count())

print('Number of outliers in Age lower : ', df[df['Age']<17.0]['Age'].count())

print('% of Outlier in Age upper: ',round(df[df['Age']>57.0]['Age'].count()*100/len(df)), '%')

print('% of Outlier in Age lower: ',round(df[df['Age']<17.0]['Age'].count()*100/len(df)), '%')
plt.title('Age')

sns.boxplot(df['Age'],orient='horizondal',color='purple')
fig, (ax2,ax3)=plt.subplots(1,2,figsize=(13,5))



#distplot

sns.distplot(df['Age'],ax=ax2)

ax2.set_xlabel('Age', fontsize=15)

ax2.tick_params(labelsize=15)



#histogram

ax3.hist(df['Age'])

ax3.set_xlabel('Age', fontsize=15)

ax3.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout()
print('Range of values: ', df['Commision'].max()-df['Commision'].min())
#Central values 

print('Minimum Commision: ', df['Commision'].min())

print('Maximum Commision: ',df['Commision'].max())

print('Mean value: ', df['Commision'].mean())

print('Median value: ',df['Commision'].median())

print('Standard deviation: ', df['Commision'].std())

print('Null values: ',df['Commision'].isnull().any())
#Quartiles



Q1=df['Commision'].quantile(q=0.25)

Q3=df['Commision'].quantile(q=0.75)

print('Commision - 1st Quartile (Q1) is: ', Q1)

print('Commision - 3st Quartile (Q3) is: ', Q3)

print('Interquartile range (IQR) of Commision is ', stats.iqr(df['Commision']))
#Outlier detection from Interquartile range (IQR) in original data



# IQR=Q3-Q1

#lower 1.5*IQR whisker i.e Q1-1.5*IQR

#upper 1.5*IQR whisker i.e Q3+1.5*IQR

L_outliers=Q1-1.5*(Q3-Q1)

U_outliers=Q3+1.5*(Q3-Q1)

print('Lower outliers in Commision: ', L_outliers)

print('Upper outliers in Commision: ', U_outliers)
print('Number of outliers in Commision upper : ', df[df['Commision']>43.0875]['Commision'].count())

print('Number of outliers in Commision lower : ', df[df['Commision']<-25.8525]['Commision'].count())

print('% of Outlier in Commision upper: ',round(df[df['Commision']>43.0875]['Commision'].count()*100/len(df)), '%')

print('% of Outlier in Commision lower: ',round(df[df['Commision']<-25.8525]['Commision'].count()*100/len(df)), '%')
plt.title('Commision')

sns.boxplot(df['Commision'],orient='horizondal',color='purple')
fig, (ax2,ax3)=plt.subplots(1,2,figsize=(13,5))



#distplot

sns.distplot(df['Commision'],ax=ax2)

ax2.set_xlabel('Commision', fontsize=15)

ax2.tick_params(labelsize=15)



#histogram

ax3.hist(df['Commision'])

ax3.set_xlabel('Commision', fontsize=15)

ax3.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout()
print('Range of values: ', df['Duration'].max()-df['Duration'].min())
#Central values 

print('Minimum Duration: ', df['Duration'].min())

print('Maximum Duration: ',df['Duration'].max())

print('Mean value: ', df['Duration'].mean())

print('Median value: ',df['Duration'].median())

print('Standard deviation: ', df['Duration'].std())

print('Null values: ',df['Duration'].isnull().any())
#Quartiles



Q1=df['Duration'].quantile(q=0.25)

Q3=df['Duration'].quantile(q=0.75)

print('Duration - 1st Quartile (Q1) is: ', Q1)

print('Duration - 3st Quartile (Q3) is: ', Q3)

print('Interquartile range (IQR) of Duration is ', stats.iqr(df['Duration']))
#Outlier detection from Interquartile range (IQR) in original data



# IQR=Q3-Q1

#lower 1.5*IQR whisker i.e Q1-1.5*IQR

#upper 1.5*IQR whisker i.e Q3+1.5*IQR

L_outliers=Q1-1.5*(Q3-Q1)

U_outliers=Q3+1.5*(Q3-Q1)

print('Lower outliers in Duration: ', L_outliers)

print('Upper outliers in Duration: ', U_outliers)
print('Number of outliers in Duration upper : ', df[df['Duration']>141.0]['Duration'].count())

print('Number of outliers in Duration lower : ', df[df['Duration']<-67.0]['Duration'].count())

print('% of Outlier in Duration upper: ',round(df[df['Duration']>141.0]['Duration'].count()*100/len(df)), '%')

print('% of Outlier in Duration lower: ',round(df[df['Duration']<-67.0]['Duration'].count()*100/len(df)), '%')
plt.title('Duration')

sns.boxplot(df['Duration'],orient='horizondal',color='purple')
fig, (ax2,ax3)=plt.subplots(1,2,figsize=(13,5))



#distplot

sns.distplot(df['Duration'],ax=ax2)

ax2.set_xlabel('Duration', fontsize=15)

ax2.tick_params(labelsize=15)



#histogram

ax3.hist(df['Duration'])

ax3.set_xlabel('Duration', fontsize=15)

ax3.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout()
print('Range of values: ', df['Sales'].max()-df['Sales'].min())
#Central values 

print('Minimum Sales: ', df['Sales'].min())

print('Maximum Sales: ',df['Sales'].max())

print('Mean value: ', df['Sales'].mean())

print('Median value: ',df['Sales'].median())

print('Standard deviation: ', df['Sales'].std())

print('Null values: ',df['Sales'].isnull().any())
#Quartiles



Q1=df['Sales'].quantile(q=0.25)

Q3=df['Sales'].quantile(q=0.75)

print('Sales - 1st Quartile (Q1) is: ', Q1)

print('Sales - 3st Quartile (Q3) is: ', Q3)

print('Interquartile range (IQR) of Sales is ', stats.iqr(df['Sales']))
#Outlier detection from Interquartile range (IQR) in original data



# IQR=Q3-Q1

#lower 1.5*IQR whisker i.e Q1-1.5*IQR

#upper 1.5*IQR whisker i.e Q3+1.5*IQR

L_outliers=Q1-1.5*(Q3-Q1)

U_outliers=Q3+1.5*(Q3-Q1)

print('Lower outliers in Sales: ', L_outliers)

print('Upper outliers in Sales: ', U_outliers)
print('Number of outliers in Sales upper : ', df[df['Sales']>142.5]['Sales'].count())

print('Number of outliers in Sales lower : ', df[df['Sales']<-53.5]['Sales'].count())

print('% of Outlier in Sales upper: ',round(df[df['Sales']>142.5]['Sales'].count()*100/len(df)), '%')

print('% of Outlier in Sales lower: ',round(df[df['Sales']<-53.5]['Sales'].count()*100/len(df)), '%')
plt.title('Sales')

sns.boxplot(df['Sales'],orient='horizondal',color='purple')
fig, (ax2,ax3)=plt.subplots(1,2,figsize=(13,5))



#distplot

sns.distplot(df['Sales'],ax=ax2)

ax2.set_xlabel('Sales', fontsize=15)

ax2.tick_params(labelsize=15)



#histogram

ax3.hist(df['Sales'])

ax3.set_xlabel('Sales', fontsize=15)

ax3.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout()
sns.countplot(data = df, x = 'Agency_Code')
sns.boxplot(data = df, x='Agency_Code',y='Sales', hue='Claimed')
sns.swarmplot(data = df, x='Agency_Code',y='Sales')
sns.violinplot(data = df, x='Agency_Code',y='Sales')

sns.swarmplot(data = df, x='Agency_Code',y='Sales', color = 'k', alpha = 0.6)
sns.countplot(data = df, x = 'Type')
sns.boxplot(data = df, x='Type',y='Sales', hue='Claimed')
sns.swarmplot(data = df, x='Type',y='Sales')
sns.violinplot(data = df, x='Type',y='Sales')

sns.swarmplot(data = df, x='Type',y='Sales', color = 'k', alpha = 0.6)
sns.countplot(data = df, x = 'Channel')
sns.boxplot(data = df, x='Channel',y='Sales', hue='Claimed')
sns.swarmplot(data = df, x='Channel',y='Sales')
sns.violinplot(data = df, x='Channel',y='Sales')

sns.swarmplot(data = df, x='Channel',y='Sales', color = 'k', alpha = 0.6)
sns.countplot(data = df, x = 'Product Name')
sns.boxplot(data = df, x='Product Name',y='Sales', hue='Claimed')
sns.swarmplot(data = df, x='Product Name',y='Sales')
sns.violinplot(data = df, x='Product Name',y='Sales')

sns.swarmplot(data = df, x='Product Name',y='Sales', color = 'k', alpha = 0.6)
sns.countplot(data = df, x = 'Destination')
sns.boxplot(data = df, x='Destination',y='Sales', hue='Claimed')
sns.swarmplot(data = df, x='Destination',y='Sales')
sns.violinplot(data = df, x='Destination',y='Sales')

sns.swarmplot(data = df, x='Destination',y='Sales', color = 'k', alpha = 0.6)
sns.pairplot(df[['Age', 'Commision', 

    'Duration', 'Sales']])
# construct heatmap with only continuous variables

plt.figure(figsize=(10,8))

sns.set(font_scale=1.2)

sns.heatmap(df[['Age', 'Commision', 

    'Duration', 'Sales']].corr(), annot=True)
for feature in df.columns: 

    if df[feature].dtype == 'object': 

        print('\n')

        print('feature:',feature)

        print(pd.Categorical(df[feature].unique()))

        print(pd.Categorical(df[feature].unique()).codes)

        df[feature] = pd.Categorical(df[feature]).codes
df.info()
df.head()
df.Claimed.value_counts(normalize=True)
X = df.drop("Claimed", axis=1)



y = df.pop("Claimed")



X.head()
# prior to scaling 

plt.plot(X)

plt.show()
# Scaling the attributes.



from scipy.stats import zscore

X_scaled=X.apply(zscore)

X_scaled.head()
# prior to scaling 

plt.plot(X_scaled)

plt.show()
from sklearn.model_selection import train_test_split



X_train, X_test, train_labels, test_labels = train_test_split(X_scaled, y, test_size=.30, random_state=5)
print('X_train',X_train.shape)

print('X_test',X_test.shape)

print('train_labels',train_labels.shape)

print('test_labels',test_labels.shape)
param_grid_dtcl = {

    'criterion': ['gini'],

    'max_depth': [10,20,30,50],

    'min_samples_leaf': [50,100,150], 

    'min_samples_split': [150,300,450],

}



dtcl = DecisionTreeClassifier(random_state=1)



grid_search_dtcl = GridSearchCV(estimator = dtcl, param_grid = param_grid_dtcl, cv = 10)
grid_search_dtcl.fit(X_train, train_labels)

print(grid_search_dtcl.best_params_)

best_grid_dtcl = grid_search_dtcl.best_estimator_

best_grid_dtcl

#{'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 50, 'min_samples_split': 450}
param_grid_dtcl = {

    'criterion': ['gini'],

    'max_depth': [3, 5, 7, 10,12],

    'min_samples_leaf': [20,30,40,50,60], 

    'min_samples_split': [150,300,450],

}



dtcl = DecisionTreeClassifier(random_state=1)



grid_search_dtcl = GridSearchCV(estimator = dtcl, param_grid = param_grid_dtcl, cv = 10)
grid_search_dtcl.fit(X_train, train_labels)

print(grid_search_dtcl.best_params_)

best_grid_dtcl = grid_search_dtcl.best_estimator_

best_grid_dtcl

#{'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 50, 'min_samples_split': 450}
param_grid_dtcl = {

    'criterion': ['gini'],

    'max_depth': [3.5,4.0,4.5, 5.0,5.5],

    'min_samples_leaf': [40, 42, 44,46,48,50,52,54], 

    'min_samples_split': [250, 270, 280, 290, 300,310],

}



dtcl = DecisionTreeClassifier(random_state=1)



grid_search_dtcl = GridSearchCV(estimator = dtcl, param_grid = param_grid_dtcl, cv = 10)
grid_search_dtcl.fit(X_train, train_labels)

print(grid_search_dtcl.best_params_)

best_grid_dtcl = grid_search_dtcl.best_estimator_

best_grid_dtcl

#{'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 50, 'min_samples_split': 450}
param_grid_dtcl = {

    'criterion': ['gini'],

    'max_depth': [4.85, 4.90,4.95, 5.0,5.05,5.10,5.15],

    'min_samples_leaf': [40, 41, 42, 43, 44], 

    'min_samples_split': [150, 175, 200, 210, 220, 230, 240, 250, 260, 270],

}



dtcl = DecisionTreeClassifier(random_state=1)



grid_search_dtcl = GridSearchCV(estimator = dtcl, param_grid = param_grid_dtcl, cv = 10)
grid_search_dtcl.fit(X_train, train_labels)

print(grid_search_dtcl.best_params_)

best_grid_dtcl = grid_search_dtcl.best_estimator_

best_grid_dtcl

#{'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 50, 'min_samples_split': 450}
train_char_label = ['no', 'yes']

tree_regularized = open('tree_regularized.dot','w')

dot_data = tree.export_graphviz(best_grid_dtcl, out_file= tree_regularized ,

                                feature_names = list(X_train),

                                class_names = list(train_char_label))



tree_regularized.close()

dot_data
print (pd.DataFrame(best_grid_dtcl.feature_importances_, columns = ["Imp"], 

                    index = X_train.columns).sort_values('Imp',ascending=False))
ytrain_predict_dtcl = best_grid_dtcl.predict(X_train)

ytest_predict_dtcl = best_grid_dtcl.predict(X_test)
ytest_predict_dtcl

ytest_predict_prob_dtcl=best_grid_dtcl.predict_proba(X_test)

ytest_predict_prob_dtcl

pd.DataFrame(ytest_predict_prob_dtcl).head()
param_grid_rfcl = {

    'max_depth': [4,5,6],#20,30,40

    'max_features': [2,3,4,5],## 7,8,9

    'min_samples_leaf': [8,9,11,15],## 50,100

    'min_samples_split': [46,50,55], ## 60,70

    'n_estimators': [290,350,400] ## 100,200

}



rfcl = RandomForestClassifier(random_state=1)



grid_search_rfcl = GridSearchCV(estimator = rfcl, param_grid = param_grid_rfcl, cv = 5)
grid_search_rfcl.fit(X_train, train_labels)

print(grid_search_rfcl.best_params_)

best_grid_rfcl = grid_search_rfcl.best_estimator_

best_grid_rfcl

#{'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 50, 'min_samples_split': 450}
ytrain_predict_rfcl = best_grid_rfcl.predict(X_train)

ytest_predict_rfcl = best_grid_rfcl.predict(X_test)
ytest_predict_rfcl

ytest_predict_prob_rfcl=best_grid_rfcl.predict_proba(X_test)

ytest_predict_prob_rfcl

pd.DataFrame(ytest_predict_prob_rfcl).head()
# Variable Importance

print (pd.DataFrame(best_grid_rfcl.feature_importances_, 

                    columns = ["Imp"], 

                    index = X_train.columns).sort_values('Imp',ascending=False))
param_grid_nncl = {

    'hidden_layer_sizes': [50,100,200], # 50, 200

    'max_iter': [2500,3000,4000], #5000,2500

    'solver': ['adam'], #sgd

    'tol': [0.01], 

}



nncl = MLPClassifier(random_state=1)



grid_search_nncl = GridSearchCV(estimator = nncl, param_grid = param_grid_nncl, cv = 10)
grid_search_nncl.fit(X_train, train_labels)

grid_search_nncl.best_params_

best_grid_nncl = grid_search_nncl.best_estimator_

best_grid_nncl
ytrain_predict_nncl = best_grid_nncl.predict(X_train)

ytest_predict_nncl = best_grid_nncl.predict(X_test)
ytest_predict_nncl

ytest_predict_prob_nncl=best_grid_nncl.predict_proba(X_test)

ytest_predict_prob_nncl

pd.DataFrame(ytest_predict_prob_nncl).head()
# predict probabilities

probs_cart = best_grid_dtcl.predict_proba(X_train)

# keep probabilities for the positive outcome only

probs_cart = probs_cart[:, 1]

# calculate AUC

cart_train_auc = roc_auc_score(train_labels, probs_cart)

print('AUC: %.3f' % cart_train_auc)

# calculate roc curve

cart_train_fpr, cart_train_tpr, cart_train_thresholds = roc_curve(train_labels, probs_cart)

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(cart_train_fpr, cart_train_tpr)
# predict probabilities

probs_cart = best_grid_dtcl.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs_cart = probs_cart[:, 1]

# calculate AUC

cart_test_auc = roc_auc_score(test_labels, probs_cart)

print('AUC: %.3f' % cart_test_auc)

# calculate roc curve

cart_test_fpr, cart_test_tpr, cart_testthresholds = roc_curve(test_labels, probs_cart)

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(cart_test_fpr, cart_test_tpr)
confusion_matrix(train_labels, ytrain_predict_dtcl)
#Train Data Accuracy

cart_train_acc=best_grid_dtcl.score(X_train,train_labels) 

cart_train_acc
print(classification_report(train_labels, ytrain_predict_dtcl))
cart_metrics=classification_report(train_labels, ytrain_predict_dtcl,output_dict=True)

df=pd.DataFrame(cart_metrics).transpose()

cart_train_f1=round(df.loc["1"][2],2)

cart_train_recall=round(df.loc["1"][1],2)

cart_train_precision=round(df.loc["1"][0],2)

print ('cart_train_precision ',cart_train_precision)

print ('cart_train_recall ',cart_train_recall)

print ('cart_train_f1 ',cart_train_f1)
confusion_matrix(test_labels, ytest_predict_dtcl)
#Test Data Accuracy

cart_test_acc=best_grid_dtcl.score(X_test,test_labels)

cart_test_acc
print(classification_report(test_labels, ytest_predict_dtcl))
cart_metrics=classification_report(test_labels, ytest_predict_dtcl,output_dict=True)

df=pd.DataFrame(cart_metrics).transpose()

cart_test_precision=round(df.loc["1"][0],2)

cart_test_recall=round(df.loc["1"][1],2)

cart_test_f1=round(df.loc["1"][2],2)

print ('cart_test_precision ',cart_test_precision)

print ('cart_test_recall ',cart_test_recall)

print ('cart_test_f1 ',cart_test_f1)
confusion_matrix(train_labels,ytrain_predict_rfcl)
rf_train_acc=best_grid_rfcl.score(X_train,train_labels) 

rf_train_acc
print(classification_report(train_labels,ytrain_predict_rfcl))
rf_metrics=classification_report(train_labels, ytrain_predict_rfcl,output_dict=True)

df=pd.DataFrame(rf_metrics).transpose()

rf_train_precision=round(df.loc["1"][0],2)

rf_train_recall=round(df.loc["1"][1],2)

rf_train_f1=round(df.loc["1"][2],2)

print ('rf_train_precision ',rf_train_precision)

print ('rf_train_recall ',rf_train_recall)

print ('rf_train_f1 ',rf_train_f1)
rf_train_fpr, rf_train_tpr,_=roc_curve(train_labels,best_grid_rfcl.predict_proba(X_train)[:,1])

plt.plot(rf_train_fpr,rf_train_tpr,color='green')

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

rf_train_auc=roc_auc_score(train_labels,best_grid_rfcl.predict_proba(X_train)[:,1])

print('Area under Curve is', rf_train_auc)
confusion_matrix(test_labels,ytest_predict_rfcl)
rf_test_acc=best_grid_rfcl.score(X_test,test_labels)

rf_test_acc
print(classification_report(test_labels,ytest_predict_rfcl))
rf_metrics=classification_report(test_labels, ytest_predict_rfcl,output_dict=True)

df=pd.DataFrame(rf_metrics).transpose()

rf_test_precision=round(df.loc["1"][0],2)

rf_test_recall=round(df.loc["1"][1],2)

rf_test_f1=round(df.loc["1"][2],2)

print ('rf_test_precision ',rf_test_precision)

print ('rf_test_recall ',rf_test_recall)

print ('rf_test_f1 ',rf_test_f1)
rf_test_fpr, rf_test_tpr,_=roc_curve(test_labels,best_grid_rfcl.predict_proba(X_test)[:,1])

plt.plot(rf_test_fpr,rf_test_tpr,color='green')

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

rf_test_auc=roc_auc_score(test_labels,best_grid_rfcl.predict_proba(X_test)[:,1])

print('Area under Curve is', rf_test_auc)
confusion_matrix(train_labels,ytrain_predict_nncl)
nn_train_acc=best_grid_nncl.score(X_train,train_labels) 

nn_train_acc
print(classification_report(train_labels,ytrain_predict_nncl))
nn_metrics=classification_report(train_labels, ytrain_predict_nncl,output_dict=True)

df=pd.DataFrame(nn_metrics).transpose()

nn_train_precision=round(df.loc["1"][0],2)

nn_train_recall=round(df.loc["1"][1],2)

nn_train_f1=round(df.loc["1"][2],2)

print ('nn_train_precision ',nn_train_precision)

print ('nn_train_recall ',nn_train_recall)

print ('nn_train_f1 ',nn_train_f1)
nn_train_fpr, nn_train_tpr,_=roc_curve(train_labels,best_grid_nncl.predict_proba(X_train)[:,1])

plt.plot(nn_train_fpr,nn_train_tpr,color='black')

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

nn_train_auc=roc_auc_score(train_labels,best_grid_nncl.predict_proba(X_train)[:,1])

print('Area under Curve is', nn_train_auc)
confusion_matrix(test_labels,ytest_predict_nncl)
nn_test_acc=best_grid_nncl.score(X_test,test_labels)

nn_test_acc
print(classification_report(test_labels,ytest_predict_nncl))
nn_metrics=classification_report(test_labels, ytest_predict_nncl,output_dict=True)

df=pd.DataFrame(nn_metrics).transpose()

nn_test_precision=round(df.loc["1"][0],2)

nn_test_recall=round(df.loc["1"][1],2)

nn_test_f1=round(df.loc["1"][2],2)

print ('nn_test_precision ',nn_test_precision)

print ('nn_test_recall ',nn_test_recall)

print ('nn_test_f1 ',nn_test_f1)
nn_test_fpr, nn_test_tpr,_=roc_curve(test_labels,best_grid_nncl.predict_proba(X_test)[:,1])

plt.plot(nn_test_fpr,nn_test_tpr,color='black')

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

nn_test_auc=roc_auc_score(test_labels,best_grid_nncl.predict_proba(X_test)[:,1])

print('Area under Curve is', nn_test_auc)
index=['Accuracy', 'AUC', 'Recall','Precision','F1 Score']

data = pd.DataFrame({'CART Train':[cart_train_acc,cart_train_auc,cart_train_recall,cart_train_precision,cart_train_f1],

        'CART Test':[cart_test_acc,cart_test_auc,cart_test_recall,cart_test_precision,cart_test_f1],

       'Random Forest Train':[rf_train_acc,rf_train_auc,rf_train_recall,rf_train_precision,rf_train_f1],

        'Random Forest Test':[rf_test_acc,rf_test_auc,rf_test_recall,rf_test_precision,rf_test_f1],

       'Neural Network Train':[nn_train_acc,nn_train_auc,nn_train_recall,nn_train_precision,nn_train_f1],

        'Neural Network Test':[nn_test_acc,nn_test_auc,nn_test_recall,nn_test_precision,nn_test_f1]},index=index)

round(data,2)
plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(cart_train_fpr, cart_train_tpr,color='red',label="CART")

plt.plot(rf_train_fpr,rf_train_tpr,color='green',label="RF")

plt.plot(nn_train_fpr,nn_train_tpr,color='black',label="NN")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(cart_test_fpr, cart_test_tpr,color='red',label="CART")

plt.plot(rf_test_fpr,rf_test_tpr,color='green',label="RF")

plt.plot(nn_test_fpr,nn_test_tpr,color='black',label="NN")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right')