
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
hdr10 = pd.read_csv(os.path.join(dirname, filename))
hdr10.columns = hdr10.columns.str.lower()
hdr10.head()
hdr10.set_index('gameid',inplace=True)
hdr10.info()
hdr10.bluewins.value_counts()
#plt.figure(figsize=[15,15])
hdr10.hist(figsize=[15,18]);
#plt.show()
#creating minion kill difference attribute to show the difference in number of minions killed by blue and red team in a given game
hdr10['minionskilldifference'] = hdr10["bluetotalminionskilled"]-hdr10["redtotalminionskilled"]

#selecting rows where the blue team has killed more minions than the red team in first 10 mins
minionkillpositive_temp = hdr10.query("minionskilldifference>0")['bluewins']
minionkillpositive_pie = minionkillpositive_temp.value_counts()
minionkillpositive_pie.rename(index={1: "Won", 0:"Lost"}, inplace=True)

#selecting rows where the blue team has killed less minions than the red team in first 10 mins
minionkillnegative_temp = hdr10.query("minionskilldifference<=0")['bluewins']
minionkillnegative_pie = minionkillnegative_temp.value_counts()
minionkillnegative_pie.rename(index={1: "Won", 0:"Lost"}, inplace=True)
fig = plt.figure(figsize=(10,8), dpi=100)
ax1 = plt.subplot2grid((2,1),(0,0))
plt.pie(minionkillpositive_pie, labels=minionkillpositive_pie.index, explode= (0,0.05),
       colors=['YellowGreen', 'Coral'],
       autopct='%1.1f%%', startangle=70)
plt.title("Blue Team Win Percentage with More Minion Kills than Red Team", fontsize=10, color='Black')

ax1 = plt.subplot2grid((2,1),(1,0))
plt.pie(minionkillnegative_pie, labels=minionkillnegative_pie.index, explode= (0,0.05),
       colors=['Coral', 'Yellowgreen'],
       autopct='%1.1f%%', startangle=70)
plt.title("Blue Team Win Percentage with Less Minion Kills than Red Team", fontsize=10, color='Black');
#creating chanmpion kill difference attribute to show the difference in number of champions killed by blue and red team in a given game
hdr10["championkilldifference"] = hdr10["bluekills"] - hdr10["redkills"]

#selecting rows where the blue team has killed more champions than the red team in first 10 mins
champkillpositive_temp = hdr10.query("championkilldifference>0")['bluewins']
champkillpositive_pie = champkillpositive_temp.value_counts()
champkillpositive_pie.rename(index={1: "Won", 0:"Lost"}, inplace=True)

#selecting rows where the blue team has killed less champions than the red team in first 10 mins
champkillnegative_temp = hdr10.query("championkilldifference<=0")['bluewins']
champkillnegative_pie = champkillnegative_temp.value_counts()
champkillnegative_pie.rename(index={1: "Won", 0:"Lost"}, inplace=True)
fig = plt.figure(figsize=(10,8), dpi=100)
ax1 = plt.subplot2grid((2,1),(0,0))
plt.pie(champkillpositive_pie, labels=champkillpositive_pie.index, explode= (0,0.05),
       colors=['YellowGreen', 'Coral'],
       autopct='%1.1f%%', startangle=70)
plt.title("Blue Team Win Percentage with More Champion Kills than Red Team in First 10 Minutes", fontsize=10, color='Black')

ax1 = plt.subplot2grid((2,1),(1,0))
plt.pie(champkillnegative_pie, labels=champkillnegative_pie.index, explode= (0,0.05),
       colors=['Coral', 'Yellowgreen'],
       autopct='%1.1f%%', startangle=70)
plt.title("Blue Team Win Percentage with Less Champion Kills than Red Team in First 10 Minutes", fontsize=10, color='Black');
hdr10['wardsdifference'] = hdr10["bluewardsplaced"]-hdr10["redwardsplaced"]

wardsdifferencepositive_temp = hdr10.query("wardsdifference>0")['bluewins']
wardsdifferencepositive_pie = wardsdifferencepositive_temp.value_counts()
wardsdifferencepositive_pie.rename(index={1: "Won", 0:"Lost"}, inplace=True)

wardsdifferencenegative_temp = hdr10.query("wardsdifference<=0")['bluewins']
wardsdifferencenegative_pie = wardsdifferencenegative_temp.value_counts()
wardsdifferencenegative_pie.rename(index={1: "Won", 0:"Lost"}, inplace=True)
fig = plt.figure(figsize=(10,8), dpi=100)
ax1 = plt.subplot2grid((2,1),(0,0))
plt.pie(wardsdifferencepositive_pie, labels=wardsdifferencepositive_pie.index, explode= (0,0.05),
       colors=['YellowGreen', 'Coral'],
       autopct='%1.1f%%', startangle=70)
plt.title("Blue Team Win Percentage with More Wards Placed than Red Team in First 10 Minutes", fontsize=10, color='Black')

ax1 = plt.subplot2grid((2,1),(1,0))
plt.pie(wardsdifferencenegative_pie, labels=wardsdifferencenegative_pie.index, explode= (0,0.05),
       colors=['Coral', 'Yellowgreen'],
       autopct='%1.1f%%', startangle=70)
plt.title("Blue Team Win Percentage with Less Wards Placed than Red Team in First 10 Minutes", fontsize=10, color='Black');
hdr10_scaled = hdr10.copy()

# Create the scaler with object range of 0-1
scaler = MinMaxScaler()

# Fit and transform using the training data
hdr10_scaled[hdr10_scaled.columns] = scaler.fit_transform(hdr10[hdr10.columns])
#Making series with all the independent features for the train dataset
features=hdr10_scaled.columns[1:]
X_train_temp, X_test, y_train_temp, y_test = train_test_split(hdr10_scaled[features], 
                                                              hdr10_scaled["bluewins"], 
                                                              test_size=0.2, 
                                                              random_state=20)
# Perform the second split
X_train, X_valid, y_train, y_valid = train_test_split(X_train_temp, y_train_temp, 
                                                      test_size=0.25, random_state=20)
# Create the model
lasso = Lasso(alpha=0.01)

# Fit the model to the training data
lasso.fit(X_train, y_train)

# Extract the coefficients
lasso_coef = lasso.coef_

# Plot the coefficients
plt.figure(figsize=(20, 10))
plt.plot(range(len(X_train.columns)), lasso_coef)
plt.xticks(range(len(X_train.columns)), X_train.columns, rotation=60)
plt.axhline(0.0, linestyle='--', color='r')
plt.ylabel('Coefficients')
temp = pd.DataFrame(X_train.columns.values, lasso_coef)
temp.reset_index(inplace=True)
temp.columns=['lasso_coef','features']
print(temp[temp['lasso_coef']!=0])
#selecting all the features with non-zero coefficient to train and test our model
lasso_selected_features = temp[(temp['lasso_coef']>0) | (temp['lasso_coef']<-0.052)]['features']
lasso_selected_features
#Dataset with features selected by Lasso
lasso_Xtrain = X_train.loc[:,lasso_selected_features]
lasso_Xvalid = X_valid.loc[:,lasso_selected_features]
lasso_Xtest = X_test.loc[:,lasso_selected_features]
#Training logistic regression model on features selected by lasso
lr_valid= LogisticRegression()
lr_valid.fit(lasso_Xtrain,y_train)
y_valid_pred = valid_lr.predict(lasso_Xvalid)
cm_valid = confusion_matrix(y_valid,yvalid_pred)
#creating confusion matrix for validation dataset prediction
plt.axes()
sns.heatmap(cm_valid/np.sum(cm_valid), annot=True, 
            fmt='.2%', cmap='Greens',
           xticklabels=['Loss','Wins'],
           yticklabels=['Loss','Wins'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label");
#checking accuracy of regression model on validation data set with features selected by lasso 
print('Accuracy of model on validation dataset: ',metrics.accuracy_score(y_valid, yvalid_pred))
#Training logistic regression model on test data set to check the accuracy
lr_test= LogisticRegression()
lr_test.fit(lasso_Xtrain,y_train)
ytest_pred=lr_test.predict(lasso_Xtest)
cm_test = confusion_matrix(y_test,ytest_predict)
#creating confusion matrix for the test dataset prediction
plt.axes()
sns.heatmap(cm_test/np.sum(cm_test),annot=True,
           fmt='.2%', cmap='Greens',
           xticklabels=['Loss','Wins'],
           yticklabels=['Loss','Wins'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label");
print('Accuracy of model on test dataset :',metrics.accuracy_score(y_test,lasso_ypredtest))