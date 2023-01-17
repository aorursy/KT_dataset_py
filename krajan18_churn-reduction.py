import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn
import random
import time
import os
print(os.listdir("../input"))
train = pd.read_csv("../input/Train_data.csv")
test = pd.read_csv("../input/Test_data.csv")
train.head()
print(train.shape)
test.shape
print("Number of rows: ", train.shape[0])
counts = train.describe().iloc[0]
display(
    pd.DataFrame(
        counts.tolist(), 
        columns=["Count of values"], 
        index=counts.index.values
    ).transpose()
)
# Remaning the Train columns Name:
train.rename(columns = {'account length': 'account_length','area code':'area_code','phone number':'phone_number',
                       'international plan':'international_plan','voice mail plan':'voice_mail_plan',
                      'number vmail messages':'number_vmail_messages','total day minutes':'total_day_minutes',
                      'total day calls':'total_day_calls','total day charge':'total_day_charge',
                        'total eve minutes':'total_eve_minutes','total eve calls':'total_eve_calls',
                      'total eve charge':'total_eve_charge','total night minutes':'total_night_minutes',
                       'total night calls':'total_night_calls','total night charge':'total_night_charge',
                       'total intl minutes':'total_intl_minutes','total intl calls':'total_intl_calls',
                       'total intl charge':'total_intl_charge','number customer service calls':'number_customer_service_calls'}
               ,inplace= True)
# Remaning the Test columns Name:
test.rename(columns = {'account length': 'account_length','area code':'area_code','phone number':'phone_number',
                       'international plan':'international_plan','voice mail plan':'voice_mail_plan',
                      'number vmail messages':'number_vmail_messages','total day minutes':'total_day_minutes',
                      'total day calls':'total_day_calls','total day charge':'total_day_charge',
                        'total eve minutes':'total_eve_minutes','total eve calls':'total_eve_calls',
                      'total eve charge':'total_eve_charge','total night minutes':'total_night_minutes',
                       'total night calls':'total_night_calls','total night charge':'total_night_charge',
                       'total intl minutes':'total_intl_minutes','total intl calls':'total_intl_calls',
                       'total intl charge':'total_intl_charge','number customer service calls':'number_customer_service_calls'}
                        ,inplace= True)
numcol = ['account_length','number_vmail_messages','total_day_minutes','total_day_calls',
          'total_day_charge', 'total_eve_minutes','total_eve_calls','total_eve_charge','total_night_minutes',
           'total_night_calls','total_night_charge','total_intl_minutes','total_intl_calls',
           'total_intl_charge','number_customer_service_calls']
train.describe()
train.info()
# Removing NaN
train.isna().any()

test.isna().any()

#converting variables into Labels:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train[['Churn','international_plan','voice_mail_plan']] = train[['Churn','international_plan','voice_mail_plan']].apply(le.fit_transform)
test[['Churn','international_plan','voice_mail_plan']] = test[['Churn','international_plan','voice_mail_plan']].apply(le.fit_transform)

## Histograms

fig = plt.figure(figsize=(20, 18))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, train[numcol].shape[1] + 1):
    plt.subplot(5, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(True)
    f.set_title(train[numcol].columns.values[i - 1])

    vals = np.size(train[numcol].iloc[:, i - 1].unique())
    
    plt.hist(train[numcol].iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig = plt.figure(figsize=(12,3))

for i in numcol:
    
    facet = sn.FacetGrid(train, hue = 'Churn', size=3, aspect=4)
    facet.map(sn.kdeplot,i, shade=True)
        #facet.set(xlim=(df[col_name].min(), df[col_name].max()))
    facet.add_legend()
fig = plt.figure(figsize=(20,18))
plt.suptitle('', fontsize = 20)
for i in range(1, train[numcol].shape[1] + 1):
    plt.subplot(5,3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(train[numcol].columns.values[i - 1])
    sn.barplot(y=train[numcol].iloc[:, i - 1],x = train['Churn'],hue=train['Churn'])
     
plt.tight_layout()
## Pie Plots
train3 = train[['state','Churn','area_code','international_plan','voice_mail_plan']]
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, train3.shape[1] + 1):
    plt.subplot(3, 2, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(train3.columns.values[i - 1])
   
    values = train3.iloc[:, i - 1].value_counts(normalize = True).values
    index = train3.iloc[:, i - 1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


## Exploring Uneven Features
train[train.international_plan == 1].Churn.value_counts()
train[train.voice_mail_plan == 1].Churn.value_counts()
#Outliers Analysys

#train2 = train.drop(columns = ['state','area_code','Churn','phone_number','international_plan','voice_mail_plan'])
fig = plt.figure(figsize=(15, 25))
plt.suptitle('Churn Boxplot of Each Predictor', fontsize=20)
for i in range(1, train[numcol].shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(True)
    #f.set_title(train2.columns.values[i - 1])

    #vals = np.size(train.iloc[:, i - 1].unique())
    
    sn.boxplot(y=train[numcol].iloc[:, i - 1])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# #Detect and delete outliers from data
for i in train[numcol].columns:
    print(i)
    q75, q25 = np.percentile(train[numcol].loc[:,i], [75 ,25])
    iqr = q75 - q25

    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print(min)
    print(max)
    
    train3 = train.drop(train[numcol][train.loc[:,i] < min].index)
    train3 = train.drop(train[numcol][train.loc[:,i] > max].index)

#train2 = train.drop(columns = ['state','area_code','Churn','phone_number','international_plan','voice_mail_plan'])
fig = plt.figure(figsize=(15, 25))
plt.suptitle('Churn Boxplot of Each Predictor', fontsize=20)
for i in range(1, train3[numcol].shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(True)
    #f.set_title(train2.columns.values[i - 1])

    #vals = np.size(train.iloc[:, i - 1].unique())
    
    sn.boxplot(y=train3[numcol].iloc[:, i - 1])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
## Correlation with Response Variable
train3.drop(columns=['Churn','phone_number','state'],axis = 1).corrwith(train.Churn).plot.bar(figsize=(20,10),
              title = 'Correlation with Response variable',
              fontsize = 15, rot = 45,
              grid = True)





#here we can see that the account_length,area_code,total_eve_calls and total_night_call are very less correlated with Response Variable.So we drop these features.
train3 = train3.drop(columns = ['account_length','area_code','total_eve_calls','total_night_calls','phone_number','state'],axis = 1) 
test = test.drop(columns = ['account_length','area_code','total_eve_calls','total_night_calls','phone_number','state'],axis = 1) 

## Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = train3.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=1,vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = True,fmt = ".2f")

# Removing Correlated Fields
train3 = train3.drop(columns = ['voice_mail_plan','total_day_minutes','total_eve_minutes','total_intl_minutes'],axis = 1)
test = test.drop(columns = ['voice_mail_plan','total_day_minutes','total_eve_minutes','total_intl_minutes'],axis = 1)
y_train = train3['Churn']
train3.drop('Churn',axis = 1, inplace = True)
y_test = test['Churn']
test.drop('Churn',axis = 1, inplace = True)
train3.head()
test.head()
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = pd.DataFrame(sc_X.fit_transform(train3.values))
X_test = pd.DataFrame(sc_X.transform(test.values))
X_train.columns = train3.columns.values
X_test.columns = test.columns.values
X_train.index = train3.index.values
X_test.index = test.index.values

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

results = pd.DataFrame([['Linear Regression (Lasso)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

classifier.get_params()
## SVM (Linear)
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'linear')
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
classifier = SVC(random_state = 0, kernel = 'rbf')
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
##Random Forest 
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

model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
results
## K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X= X_train, y = y_train,
                             cv = 10)
print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))
# Applying Grid Search

# Round 1: Entropy
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["entropy"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters
# Round 2: Entropy
parameters = {"max_depth": [None],
              "max_features": [3, 5, 7],
              'min_samples_split': [8, 10, 12],
              'min_samples_leaf': [1, 2, 3],
              "bootstrap": [True],
              "criterion": ["entropy"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters
# Predicting Test Set
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100, GSx2 + Entropy)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
# Round 1: Gini
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["gini"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters

# Round 2: Gini
parameters = {"max_depth": [None],
              "max_features": [8, 10],
              'min_samples_split': [2, 3, 4],
              'min_samples_leaf': [8, 10, 12],
              "bootstrap": [True],
              "criterion": ["gini"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters
# Predicting Test Set
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100, GSx2 + Gini)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
## EXTRA: Confusion Matrix
cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
Print(results)
importances = classifier.feature_importances_
# Print the feature ranking
print("Feature importance ranking by Random Forest Model:")
for k,v in sorted(zip(map(lambda x: round(x, 4), importances), X_train.columns), reverse=True):
    print(v + ": " + str(k))

