import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import random

import os

print(os.listdir("../input"))
#Importing Dataset

df = pd.read_csv("../input/bank-full.csv")
#Creating User Columns

df_user = pd.DataFrame(np.arange(0,len(df)), columns=['user'])

df = pd.concat([df_user, df], axis=1)
df.info()
df.head(5)
df.tail()
df.columns.values
df.describe()
df.groupby('Target').mean()
df['Target'].value_counts()
countNo = len(df[df.Target == 'no'])

countYes = len(df[df.Target == 'yes'])

print('Percentage of "No": {:.3f}%'. format((countNo/(len(df.Target))*100)))

print('Percentage of "Yes": {:.3f}%'. format((countYes/(len(df.Target))*100)))
df.groupby('poutcome').mean()
df['poutcome'].value_counts()
countunknown = len(df[df.poutcome == 'unknown'])

countfailure = len(df[df.poutcome == 'failure'])

countother = len(df[df.poutcome == 'other'])

countsuccess = len(df[df.poutcome == 'success'])

print('Percentage of "unknown": {:.3f}%'. format((countunknown/(len(df.poutcome))*100)))

print('Percentage of "failure": {:.3f}%'. format((countfailure/(len(df.poutcome))*100)))

print('Percentage of "other": {:.3f}%'. format((countother/(len(df.poutcome))*100)))

print('Percentage of "success": {:.3f}%'. format((countsuccess/(len(df.poutcome))*100)))
#Verifying null values

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df.isna().any()
df.isna().sum()
#Define X and y

X = df.drop(['Target','user','job','marital', 'education', 'contact', 

             'housing', 'loan', 'day', 'month', 'poutcome' ], axis=1)

y = df['Target']
X = pd.get_dummies(X)

y = pd.get_dummies(y)
X.columns

X = X.drop(['default_no'], axis= 1)

X = X.rename(columns = {'default_yes': 'default'})

y.columns

y = y.drop(['yes'], axis=1)

y = y.rename(columns= {'no': 'y'})
#Age group

bins = range(0, 100, 10)

ax = sns.distplot(df.age[df.Target=='yes'],

              color='red', kde=False, bins=bins, label='Have Subscribed')

sns.distplot(df.age[df.Target=='no'],

         ax=ax,  # Overplots on first plot

         color='blue', kde=False, bins=bins, label="Haven't Subscribed")

plt.legend()

plt.show()
#Age

pd.crosstab(df.age,df.Target).plot(kind="bar",figsize=(20,6))

plt.title('Client Subscribed Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
pd.crosstab(df.marital,df.Target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

plt.title('Client Subscribed Frequency for Maritial')

plt.xlabel('marital')

plt.xticks(rotation=0)

plt.legend(["Haven't Subscribed", "Have Subscribed"])

plt.ylabel('Frequency')

plt.show()
plt.scatter(x=df.age[df.Target=='yes'], y=df.duration[(df.Target=='yes')], c="red")

plt.scatter(x=df.age[df.Target=='no'], y=df.duration[(df.Target=='no')])

plt.legend(["Have Subscribed", "Haven't Subscribed"])

plt.xlabel("Age")

plt.ylabel("Duration")

plt.show()

sns.pairplot(data=df, hue='Target', vars= ['age', 'balance', 'duration'])
sns.countplot(x='Target', data=df, label='Count')
sns.scatterplot(x='age', y='balance',hue='Target', data=df)
plt.figure(figsize=(20,10))

sns.heatmap(data=df.corr(), annot=True, cmap='viridis')
sns.distplot(df.age, bins = 20) 
sns.distplot(df.balance, bins = 20) 
sns.distplot(df.duration, bins = 20) 
df2 = X

fig = plt.figure(figsize=(15, 12))

plt.suptitle('Histograms of Numerical Columns', fontsize=20)

for i in range(df2.shape[1]):

    plt.subplot(6, 3, i + 1)

    f = plt.gca()

    f.set_title(df2.columns.values[i])



    vals = np.size(df2.iloc[:, i].unique())

    if vals >= 100:

        vals = 100

    

    plt.hist(df2.iloc[:, i], bins=vals, color='#3F5D7D')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
## Correlation with independent Variable 

df2.corrwith(y.y).plot.bar(figsize = (10, 10), title = "Correlation with Y", fontsize = 15, rot = 45, grid = True)
## Correlation Matrix

sns.set(style="white")



# Compute the correlation matrix

corr = df2.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
## Pie Plots

df.columns

df2 = df[['Target','job','marital', 'education', 'default', 'housing','loan', 'contact', 'month', 'poutcome']]

fig = plt.figure(figsize=(15, 12))

plt.suptitle('Pie Chart Distributions', fontsize=20)

for i in range(1, df2.shape[1] + 1):

    plt.subplot(6, 3, i)

    f = plt.gca()

    f.axes.get_yaxis().set_visible(False)

    f.set_title(df2.columns.values[i - 1])

   

    values = df2.iloc[:, i - 1].value_counts(normalize = True).values

    index = df2.iloc[:, i - 1].value_counts(normalize = True).index

    plt.pie(values, labels = index, autopct='%1.1f%%')

    plt.axis('equal')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
y_train['y'].value_counts()
pos_index = y_train[y_train.values == 1].index

neg_index = y_train[y_train.values == 0].index



if len(pos_index) > len(neg_index):

    higher = pos_index

    lower = neg_index

else:

    higher = neg_index

    lower = pos_index



random.seed(0)

higher = np.random.choice(higher, size=len(lower))

lower = np.asarray(lower)

new_indexes = np.concatenate((lower, higher))



X_train = X_train.loc[new_indexes]

y_train = y_train.loc[new_indexes]
y_train['y'].value_counts()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train2 = pd.DataFrame(sc.fit_transform(X_train))

X_test2 = pd.DataFrame(sc.transform(X_test))

X_train2.columns = X_train.columns.values

X_test2.columns = X_test.columns.values

X_train2.index = X_train.index.values

X_test2.index = X_test.index.values

X_train = X_train2

X_test = X_test2
## Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0, penalty = 'l1')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



results = pd.DataFrame([['Logistic Regression (Lasso)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
## K-Nearest Neighbors (K-NN)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p= 2)

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['K-Nearest Neighbors (minkowski)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## SVM (Linear)

from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'linear', probability= True)

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## SVM (rbf)

from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'rbf', probability= True)

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Naive Bayes

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Naive Bayes (Gaussian)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Decision Tree

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

classifier.fit(X_train, y_train)



#Predicting the best set result

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Decision Tree', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Gini (n=100)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,

                                    criterion = 'gini')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Gini (n=100)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Gini (n=200)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 200,

                                    criterion = 'gini')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Gini (n=200)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Gini (n=300)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 300,

                                    criterion = 'gini')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Gini (n=300)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Entropy (n=100)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,

                                    criterion = 'entropy')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Entropy (n=100)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Entropy (n=200)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 200,

                                    criterion = 'entropy')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Entropy (n=200)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Entropy (n=300)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 300,

                                    criterion = 'entropy')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Entropy (n=300)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
results
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train,cv=10)

accuracies.mean()

accuracies.std()
print("SVM (Linear) Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))
#Confusion Matrix

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, annot=True, fmt='g')

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) 
#Plotting Cumulative Accuracy Profile (CAP)

y_pred_proba = classifier.predict_proba(X=X_test)

import matplotlib.pyplot as plt

from scipy import integrate

def capcurve(y_values, y_preds_proba):

    num_pos_obs = np.sum(y_values)

    num_count = len(y_values)

    rate_pos_obs = float(num_pos_obs) / float(num_count)

    ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})

    xx = np.arange(num_count) / float(num_count - 1)

    

    y_cap = np.c_[y_values,y_preds_proba]

    y_cap_df_s = pd.DataFrame(data=y_cap)

    y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index(level = y_cap_df_s.index.names, drop=True)

    

    print(y_cap_df_s.head(20))

    

    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)

    yy = np.append([0], yy[0:num_count-1]) #add the first curve point (0,0) : for xx=0 we have yy=0

    

    percent = 0.5

    row_index = int(np.trunc(num_count * percent))

    

    val_y1 = yy[row_index]

    val_y2 = yy[row_index+1]

    if val_y1 == val_y2:

        val = val_y1*1.0

    else:

        val_x1 = xx[row_index]

        val_x2 = xx[row_index+1]

        val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)

    

    sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1

    sigma_model = integrate.simps(yy,xx)

    sigma_random = integrate.simps(xx,xx)

    

    ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)

    

    fig, ax = plt.subplots(nrows = 1, ncols = 1)

    ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')

    ax.plot(xx,yy, color='red', label='User Model')

    ax.plot(xx,xx, color='blue', label='Random Model')

    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)

    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1, label=str(val*100)+'% of positive obs at '+str(percent*100)+'%')

    

    plt.xlim(0, 1.02)

    plt.ylim(0, 1.25)

    plt.title("CAP Curve - a_r value ="+str(ar_value))

    plt.xlabel('% of the data')

    plt.ylabel('% of positive obs')

    plt.legend()
capcurve(y_test,y_pred_proba[:,1])
# Recursive Feature Elimination

from sklearn.feature_selection import RFE

from sklearn.svm import SVC



# Model to Test

classifier = SVC(random_state = 0, kernel = 'linear', probability= True)



# Select Best X Features

rfe = RFE(classifier, n_features_to_select=None)

rfe = rfe.fit(X_train, y_train)
# summarize the selection of the attributes

print(rfe.support_)

print(rfe.ranking_)
X_train.columns[rfe.support_]
# New Correlation Matrix

sns.set(style="white")



# Compute the correlation matrix

corr = X_train[X_train.columns[rfe.support_]].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})    
# Fitting Model to the Training Set

classifier = SVC(random_state = 0, kernel = 'linear', probability= True)

classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test[X_train.columns[rfe.support_]])

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM RFE (Linear)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
results
# Evaluating Results

#Making the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(data=cm, annot=True)
#Making the classification report

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
# Applying k-Fold Cross Validation (RFE)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier,

                             X = X_train[X_train.columns[rfe.support_]],

                             y = y_train, cv = 10)
print("SVM RFE (Linear) Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))
# Analyzing Coefficients

pd.concat([pd.DataFrame(X_train[X_train.columns[rfe.support_]].columns, columns = ["features"]),

           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])

           ],axis = 1)   
#CAP Curve

y_pred_proba = classifier.predict_proba(X=X_test[X_train.columns[rfe.support_]])

capcurve(y_test,y_pred_proba[:,1])   
### End of Model ####



# Formatting Final Results

user_identifier = df['user']

final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()

final_results['predicted'] = y_pred

final_results = final_results[['user', 'y', 'predicted']].reset_index(drop=True)
final_results.head(10)