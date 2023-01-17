import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#custom function to display dataframes    

from IPython.display import display_html

def displayoutput(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline;margin-left:50px !important;margin-right: 40px !important"'),raw=True)
#setting up for customized printing

from IPython.display import Markdown, display

from IPython.display import HTML

def printoutput(string, color=None):

    colorstr = "<span style='color:{}'>{}</span>".format(color, string)

    display(Markdown(colorstr))

    
#load and read the data 

origionaldata = pd.read_csv("../input/personal-loan-bank-modelling/Bank_Personal_Loan_Modelling.csv")
#check the data sie

origionaldata.shape
# we have 5k records and 14 columns
# Lets check the columns and their data types

origionaldata.info()
#check the columns 



print(origionaldata.columns)
origionaldata.describe().transpose()
## total number of negative values



print(origionaldata.agg(lambda x: sum(x<0)).sum())
def missing_check(df):

    total = df.isnull().sum().sort_values(ascending=False)   # total number of null values  

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)  # percentage of values that are null

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 

    return missing_data # return the dataframe

missing_check(origionaldata)
## skwness of each data

origionaldata.skew()
f, axes = plt.subplots(2, 3, figsize=(20,8))



#Age

sns.distplot(origionaldata["Age"],ax=axes[0,0],color='blue')

#Experience

sns.distplot(origionaldata["Experience"],ax=axes[0,1],color="brown")

#Income

sns.distplot(origionaldata["Income"],ax=axes[0,2],color='green')

#CCAvg

sns.distplot(origionaldata["CCAvg"],ax=axes[1,0])

#Mortgage

sns.distplot(origionaldata["Mortgage"],ax=axes[1,1],color="teal")

#ZIP Code

sns.distplot(origionaldata["ZIP Code"],ax=axes[1,2])
f, axes = plt.subplots(1, 3, figsize=(12, 10))

income = sns.boxplot(origionaldata['Income'], color="darkblue", ax=axes[0], orient='v')

income.set_xlabel("Income",fontsize=15)



ccavg = sns.boxplot(origionaldata['CCAvg'], color='green', ax=axes[1], orient='v')

ccavg.set_xlabel("CCAvg",fontsize=15)



mort = sns.boxplot(origionaldata['Mortgage'], color='red', ax=axes[2], orient='v')

mort.set_xlabel("Mortgage",fontsize=15)

# Create a figure instance

f, axes = plt.subplots(1, 3, figsize=(20, 6))



p1 = sns.boxplot('Family', 'Income', data=origionaldata, hue='Personal Loan', palette='Set1', ax=axes[0])

p1.set_xlabel("Family",fontsize=20)

p1.set_ylabel("Income",fontsize=20)



p2 = sns.boxplot('Family', 'CCAvg', data=origionaldata, hue='Personal Loan', palette='YlOrBr_r', ax=axes[1])

p2.set_xlabel("Family",fontsize=20)

p2.set_ylabel("CCAvg",fontsize=20)



p3 = sns.boxplot('Family', 'Mortgage', data=origionaldata, hue='Personal Loan', palette='viridis', ax=axes[2])

p3.set_xlabel("Family",fontsize=20)

p3.set_ylabel("Mortgage",fontsize=20)
fig ,axarr = plt.subplots(2,2,figsize=(15, 15))



sns.barplot(x='Personal Loan',y='Family',data=origionaldata,ax=axarr[0,0],hue='Family');

sns.barplot(x='Personal Loan',y='Education',data=origionaldata,ax=axarr[0,1],hue='Education');

sns.barplot(x='Personal Loan',y='Securities Account',data=origionaldata,ax=axarr[1,0],hue='Securities Account')

sns.barplot(x='Personal Loan',y='CreditCard',data=origionaldata,ax=axarr[1,1],hue='CreditCard')
sns.catplot('Family', 'Income', data=origionaldata, hue='Personal Loan', col='Education', kind='box', palette='viridis' )



sns.catplot('Family', 'CCAvg', data=origionaldata, hue='Personal Loan', col='Education', kind='box', palette='YlOrBr_r')



sns.catplot('Family', 'Mortgage', data=origionaldata, hue='Personal Loan', col='Education', kind='box', palette='Set1')
sns.pairplot(origionaldata,palette="husl",height=5.0)

plt.show()
plt.figure(figsize=(10,10))



c = origionaldata.corr()

sns.heatmap(c,annot=True,fmt='.1f', linecolor='white', linewidths=0.3, square=True)

plt.xticks(rotation=45) 
origionaldata.cov()  
printoutput('Removing **"Experience"** column due to **Multicollinearity**', color='green')



redefinedData = origionaldata.drop(['Experience','ID','ZIP Code'], axis = 1)  ## Dropping irrelevant columns



redefinedData.info()

from sklearn.preprocessing import OneHotEncoder



onehotencoder = OneHotEncoder(categories='auto')

encodedData = onehotencoder.fit_transform(redefinedData[['Family','Education']]).toarray() 

encodedFeatures = pd.DataFrame(encodedData, columns= onehotencoder.get_feature_names(['Family','Education']))

encodedFeatures.head(2)



encodedFeatures.drop(['Family_4', 'Education_3'], axis=1, inplace=True)
redefinedData.drop(['Family', 'Education'], axis=1, inplace=True)

encodedFeatures.head(2)
from sklearn.model_selection import train_test_split





X = redefinedData.loc[:, redefinedData.columns != 'Personal Loan']

y = redefinedData['Personal Loan']





X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size =.30, random_state=1)



printoutput('**Training and Testing Set Distribution**', color='green')



print(f'Training set has {X_train.shape[0]} rows and {X_train.shape[1]} columns')

print(f'Testing set has {X_test.shape[0]} rows and {X_test.shape[1]} columns')



printoutput('**Original Set Target Value Distribution**', color='green')



print("Original Personal Loan '1' Values    : {0} ({1:0.2f}%)".format(len(redefinedData.loc[redefinedData['Personal Loan'] == 1]), (len(redefinedData.loc[redefinedData['Personal Loan'] == 1])/len(redefinedData.index)) * 100))

print("Original Personal Loan '0' Values   : {0} ({1:0.2f}%)".format(len(redefinedData.loc[redefinedData['Personal Loan'] == 0]), (len(redefinedData.loc[redefinedData['Personal Loan'] == 0])/len(redefinedData.index)) * 100))



printoutput('**Training Set Target Value Distribution**', color='green')



print("Training Personal Loan '1' Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))

print("Training Personal Loan '0' Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))



printoutput('**Testing Set Target Value Distribution**', color='green')

print("Test Personal Loan '1' Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))

print("Test Personal Loan '0' Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)

X_test_scaled = scalar.fit_transform(X_test)
f, axes = plt.subplots(1, 1, figsize=(8, 7))

sns.heatmap(pd.DataFrame(X_train_scaled).corr(),cmap='YlGnBu', ax=axes, annot=True, fmt=".2f",xticklabels=X_train.columns, yticklabels=X_train.columns, linecolor='white', linewidths=0.3, square=True)

plt.xticks(rotation=45)
from sklearn import metrics

from sklearn.linear_model import LogisticRegression



# Fit the model on train

logRegModel = LogisticRegression()

logRegModel.fit(X_train_scaled, y_train)

#predict on train and test

y_train_pred = logRegModel.predict(X_train_scaled)

y_test_pred = logRegModel.predict(X_test_scaled)



#predict the probabilities on train and test

y_train_pred_proba = logRegModel.predict_proba(X_train_scaled) 

y_test_pred_proba = logRegModel.predict_proba(X_test_scaled)



#get Accuracy Score for train and test

accuracy_train = metrics.accuracy_score(y_train, y_train_pred)

accuracy_test = metrics.accuracy_score(y_test, y_test_pred)

accdf = pd.DataFrame([[accuracy_train, accuracy_test, ]], columns=['Training', 'Testing'], index=['Accuracy'])



#get Mean Squared Error on train and test

mse_train = metrics.mean_squared_error(y_train, y_train_pred)

mse_test = metrics.mean_squared_error(y_test, y_test_pred)

msedf = pd.DataFrame([[mse_train, mse_test, ]], columns=['Training', 'Testing'], index=['Mean Squared Error'])



#get Precision Score on train and test

precision_train = metrics.precision_score(y_train, y_train_pred)

precision_test = metrics.precision_score(y_test, y_test_pred)

precdf = pd.DataFrame([[precision_train, precision_test, ]], columns=['Training', 'Testing'], index=['Precision'])



#get Recall Score on train and test

recall_train = metrics.recall_score(y_train, y_train_pred)

recall_test = metrics.recall_score(y_test, y_test_pred)

recdf = pd.DataFrame([[recall_train, recall_test, ]], columns=['Training', 'Testing'], index=['Recall'])



#get F1-Score on train and test

f1_score_train = metrics.f1_score(y_train, y_train_pred)

f1_score_test = metrics.f1_score(y_test, y_test_pred)

f1sdf = pd.DataFrame([[f1_score_train, f1_score_test, ]], columns=['Training', 'Testing'], index=['F1 Score'])



#get Area Under the Curve (AUC) for ROC Curve on train and test

roc_auc_score_train = metrics.roc_auc_score(y_train, y_train_pred)

roc_auc_score_test = metrics.roc_auc_score(y_test, y_test_pred)

rocaucsdf = pd.DataFrame([[roc_auc_score_train, roc_auc_score_test, ]], columns=['Training', 'Testing'], index=['ROC AUC Score'])



#get Area Under the Curve (AUC) for Precision-Recall Curve on train and test

precision_train, recall_train, thresholds_train = metrics.precision_recall_curve(y_train, y_train_pred_proba[:,1])

precision_recall_auc_score_train = metrics.auc(recall_train, precision_train)

precision_test, recall_test, thresholds_test = metrics.precision_recall_curve(y_test,y_test_pred_proba[:,1])

precision_recall_auc_score_test = metrics.auc(recall_test, precision_test)

precrecaucsdf = pd.DataFrame([[precision_recall_auc_score_train, precision_recall_auc_score_test]], columns=['Training', 'Testing'], index=['Precision Recall AUC Score'])



#calculate the confusion matrix 

#print('tn, fp, fn, tp')

confusion_matrix_test = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])



#display confusion matrix in a heatmap

f, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(confusion_matrix_test, cmap='YlGnBu', annot=True, fmt=".0f", ax=axes[0])



#plotting the ROC Curve and Precision-Recall Curve

fpr, tpr, threshold = metrics.roc_curve(y_test,y_test_pred_proba[:,1])

plt.plot(fpr, tpr, marker='.', label='ROC Curve')

plt.plot(recall_test, precision_test, marker='.', label='Precision Recall Curve')

plt.axes(axes[1])

plt.title('Logistic Regression')



# axis labels

plt.xlabel('ROC Curve - False Positive Rate \n Precision Recall Curve - Recall')

plt.ylabel('ROC Curve - True Positive Rate \n Precision Recall Curve - Precision')

# show the legend

plt.legend()

# show the plot

plt.show()



#concatenating all the scores and displaying as single dataframe

consolidatedDF= pd.concat([accdf, msedf,precdf,recdf,f1sdf, rocaucsdf, precrecaucsdf])



printoutput('**Confusion Matrix**', color='green')

displayoutput(confusion_matrix_test, consolidatedDF)
from sklearn.naive_bayes import GaussianNB



# Fit the model on train

gnb = GaussianNB()

gnb.fit(X_train_scaled, y_train)

#predict on train and test

y_train_pred = gnb.predict(X_train_scaled)

y_test_pred = gnb.predict(X_test_scaled)



#predict the probabilities on train and test

y_train_pred_proba = gnb.predict_proba(X_train_scaled) 

y_test_pred_proba = gnb.predict_proba(X_test_scaled)



#get Accuracy Score for train and test

accuracy_train = metrics.accuracy_score(y_train, y_train_pred)

accuracy_test = metrics.accuracy_score(y_test, y_test_pred)

accdf = pd.DataFrame([[accuracy_train, accuracy_test, ]], columns=['Training', 'Testing'], index=['Accuracy'])



#get Mean Squared Error on train and test

mse_train = metrics.mean_squared_error(y_train, y_train_pred)

mse_test = metrics.mean_squared_error(y_test, y_test_pred)

msedf = pd.DataFrame([[mse_train, mse_test, ]], columns=['Training', 'Testing'], index=['Mean Squared Error'])



#get Precision Score on train and test

precision_train = metrics.precision_score(y_train, y_train_pred)

precision_test = metrics.precision_score(y_test, y_test_pred)

precdf = pd.DataFrame([[precision_train, precision_test, ]], columns=['Training', 'Testing'], index=['Precision'])



#get Recall Score on train and test

recall_train = metrics.recall_score(y_train, y_train_pred)

recall_test = metrics.recall_score(y_test, y_test_pred)

recdf = pd.DataFrame([[recall_train, recall_test, ]], columns=['Training', 'Testing'], index=['Recall'])



#get F1-Score on train and test

f1_score_train = metrics.f1_score(y_train, y_train_pred)

f1_score_test = metrics.f1_score(y_test, y_test_pred)

f1sdf = pd.DataFrame([[f1_score_train, f1_score_test, ]], columns=['Training', 'Testing'], index=['F1 Score'])



#get Area Under the Curve (AUC) for ROC Curve on train and test

roc_auc_score_train = metrics.roc_auc_score(y_train, y_train_pred)

roc_auc_score_test = metrics.roc_auc_score(y_test, y_test_pred)

rocaucsdf = pd.DataFrame([[roc_auc_score_train, roc_auc_score_test, ]], columns=['Training', 'Testing'], index=['ROC AUC Score'])



#get Area Under the Curve (AUC) for Precision-Recall Curve on train and test

precision_train, recall_train, thresholds_train = metrics.precision_recall_curve(y_train, y_train_pred_proba[:,1])

precision_recall_auc_score_train = metrics.auc(recall_train, precision_train)

precision_test, recall_test, thresholds_test = metrics.precision_recall_curve(y_test,y_test_pred_proba[:,1])

precision_recall_auc_score_test = metrics.auc(recall_test, precision_test)

precrecaucsdf = pd.DataFrame([[precision_recall_auc_score_train, precision_recall_auc_score_test]], columns=['Training', 'Testing'], index=['Precision Recall AUC Score'])



#calculate the confusion matrix 

#print('tn, fp, fn, tp')

confusion_matrix_test = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])



#display confusion matrix in a heatmap

f, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(confusion_matrix_test, cmap='YlGnBu', annot=True, fmt=".0f", ax=axes[0])



#plotting the ROC Curve and Precision-Recall Curve

fpr, tpr, threshold = metrics.roc_curve(y_test,y_test_pred_proba[:,1])

plt.plot(fpr, tpr, marker='.', label='ROC Curve')

plt.plot(recall_test, precision_test, marker='.', label='Precision Recall Curve')

plt.axes(axes[1])

plt.title('Naive Bayes')

# axis labels

plt.xlabel('ROC Curve - False Positive Rate \n Precision Recall Curve - Recall')

plt.ylabel('ROC Curve - True Positive Rate \n Precision Recall Curve - Precision')

# show the legend

plt.legend()

# show the plot

plt.show()



#concatenating all the scores and displaying as single dataframe

consolidatedDF= pd.concat([accdf, msedf,precdf,recdf,f1sdf, rocaucsdf, precrecaucsdf])



printoutput('**Confusion Matrix**', color='brown')

displayoutput(confusion_matrix_test, consolidatedDF)
from sklearn.neighbors import KNeighborsClassifier



# Fit the model on train

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train_scaled, y_train)

#predict on train and test

y_train_pred = knn.predict(X_train_scaled)

y_test_pred = knn.predict(X_test_scaled)



#predict the probabilities on train and test

y_train_pred_proba = knn.predict_proba(X_train_scaled) 

y_test_pred_proba = knn.predict_proba(X_test_scaled)



#get Accuracy Score for train and test

accuracy_train = metrics.accuracy_score(y_train, y_train_pred)

accuracy_test = metrics.accuracy_score(y_test, y_test_pred)

accdf = pd.DataFrame([[accuracy_train, accuracy_test, ]], columns=['Training', 'Testing'], index=['Accuracy'])



#get Mean Squared Error on train and test

mse_train = metrics.mean_squared_error(y_train, y_train_pred)

mse_test = metrics.mean_squared_error(y_test, y_test_pred)

msedf = pd.DataFrame([[mse_train, mse_test, ]], columns=['Training', 'Testing'], index=['Mean Squared Error'])



#get Precision Score on train and test

precision_train = metrics.precision_score(y_train, y_train_pred)

precision_test = metrics.precision_score(y_test, y_test_pred)

precdf = pd.DataFrame([[precision_train, precision_test, ]], columns=['Training', 'Testing'], index=['Precision'])



#get Recall Score on train and test

recall_train = metrics.recall_score(y_train, y_train_pred)

recall_test = metrics.recall_score(y_test, y_test_pred)

recdf = pd.DataFrame([[recall_train, recall_test, ]], columns=['Training', 'Testing'], index=['Recall'])



#get F1-Score on train and test

f1_score_train = metrics.f1_score(y_train, y_train_pred)

f1_score_test = metrics.f1_score(y_test, y_test_pred)

f1sdf = pd.DataFrame([[f1_score_train, f1_score_test, ]], columns=['Training', 'Testing'], index=['F1 Score'])



#get Area Under the Curve (AUC) for ROC Curve on train and test

roc_auc_score_train = metrics.roc_auc_score(y_train, y_train_pred)

roc_auc_score_test = metrics.roc_auc_score(y_test, y_test_pred)

rocaucsdf = pd.DataFrame([[roc_auc_score_train, roc_auc_score_test, ]], columns=['Training', 'Testing'], index=['ROC AUC Score'])



#get Area Under the Curve (AUC) for Precision-Recall Curve on train and test

precision_train, recall_train, thresholds_train = metrics.precision_recall_curve(y_train, y_train_pred_proba[:,1])

precision_recall_auc_score_train = metrics.auc(recall_train, precision_train)

precision_test, recall_test, thresholds_test = metrics.precision_recall_curve(y_test,y_test_pred_proba[:,1])

precision_recall_auc_score_test = metrics.auc(recall_test, precision_test)

precrecaucsdf = pd.DataFrame([[precision_recall_auc_score_train, precision_recall_auc_score_test]], columns=['Training', 'Testing'], index=['Precision Recall AUC Score'])



#calculate the confusion matrix 

#print('tn, fp, fn, tp')

confusion_matrix_test = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])



#display confusion matrix in a heatmap

f, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(confusion_matrix_test, cmap='YlGnBu', annot=True, fmt=".0f", ax=axes[0])



#plotting the ROC Curve and Precision-Recall Curve

fpr, tpr, threshold = metrics.roc_curve(y_test,y_test_pred_proba[:,1])

plt.plot(fpr, tpr, marker='.', label='ROC Curve')

plt.plot(recall_test, precision_test, marker='.', label='Precision Recall Curve')

plt.axes(axes[1])

plt.title('KNN')

# axis labels

plt.xlabel('ROC Curve - False Positive Rate \n Precision Recall Curve - Recall')

plt.ylabel('ROC Curve - True Positive Rate \n Precision Recall Curve - Precision')

# show the legend

plt.legend()

# show the plot

plt.show()



#concatenating all the scores and displaying as single dataframe

consolidatedDF= pd.concat([accdf, msedf,precdf,recdf,f1sdf, rocaucsdf, precrecaucsdf])



printoutput('**Confusion Matrix**', color='brown')

displayoutput(confusion_matrix_test, consolidatedDF)