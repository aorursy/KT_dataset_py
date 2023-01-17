# ADD library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Read DataSet
data = pd.read_excel("../input/loan-defaulter-prediction/case_study_data.xlsx",sheet_name ="Defaulters_Data_FNL_V2")

data.head()
# finding  basis stats of each column

def dataExploration(df): 
    eda_df = {}
    eda_df['null_sum'] = df.isnull().sum()
    eda_df['null_pct'] = df.isnull().mean()
    eda_df['dtypes'] = df.dtypes
    eda_df['count'] = df.count()
    eda_df['mean'] = df.mean()
    eda_df['median'] = df.median()
    eda_df['min'] = df.min()
    eda_df['max'] = df.max()
    
    return pd.DataFrame(eda_df)
dataExploration(data)
## Over numerical columns only
corr = data.corr()# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
## with respect to status (Traget Varible)
correlations = data.corrwith(data['status']).iloc[:-1].to_frame()
correlations['abs'] = correlations[0].abs()
sorted_correlations = correlations.sort_values('abs', ascending=False)[0]
fig, ax = plt.subplots(figsize=(10,20))
sns.heatmap(sorted_correlations.to_frame(), cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=ax)
sns.set_style('darkgrid')
plt.figure(figsize = (10,5))
sns.countplot(data['status'], alpha =.80, palette= ['grey','lightgreen'])
plt.title('Good vs Bad')
plt.ylabel('# Customers')
plt.show()
#  Numeric parameter wise distribution
# Identify numeric features
print('Continuous Variables')

# Subplots of Numeric Features
sns.set_style('darkgrid')
fig = plt.figure(figsize = (20,16))
fig.subplots_adjust(hspace = .30)

ax1 = fig.add_subplot(331)
ax1.hist(data['amount'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')
ax1.set_xlabel('amount', fontsize = 15)
ax1.set_ylabel('# Customer',fontsize = 15)
ax1.set_title('amount Class',fontsize = 15)

ax2 = fig.add_subplot(333)
ax2.hist(data['age'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')
ax2.set_xlabel('age',fontsize = 15)
ax2.set_ylabel('# Customer',fontsize = 15)
ax2.set_title('Age of Passengers',fontsize = 15)

ax3 = fig.add_subplot(334)
ax3.hist(data['duration'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')
ax3.set_xlabel('duration',fontsize = 15)
ax3.set_ylabel('# Customer',fontsize = 15)
ax3.set_title('duration',fontsize = 15)

ax4 = fig.add_subplot(335)
ax4.hist(data['num_credits'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')
ax4.set_xlabel('insta_rate',fontsize = 15)
ax4.set_ylabel('# Customer',fontsize = 15)
ax4.set_title('Insta Rate',fontsize = 15)

ax5 = fig.add_subplot(336)
ax5.hist(data['dependents'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')
ax5.set_xlabel('dependents',fontsize = 15)
ax5.set_ylabel('# Customer',fontsize = 15)
ax5.set_title('depentents per customer',fontsize = 15)

ax6 = fig.add_subplot(332)
ax6.hist(data['inst_rate'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')
ax6.set_xlabel('inst_rate',fontsize = 15)
ax6.set_ylabel('# Customer',fontsize = 15)
ax6.set_title('insta_rates',fontsize = 15)


plt.show()
# Caterogical vs Target Analysis


# Suplots of categorical features v Status

sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (40,20))

gender = data.groupby(['checkin_acc','status']).checkin_acc.count().unstack()
print (gender)
p1 = gender.plot(kind = 'bar', stacked = True, 
                   title = 'checkin_acc:Good vs Bad', 
                   color = ['lightgreen','grey'], alpha = .80, ax = axes[0,0])
p1.set_xlabel('checkin_acc',size =30)
p1.set_ylabel('# Customers',size =30)
p1.legend(['Good','Bad'],prop={"size":30})


embarked = data.groupby(['credit_history','status']).credit_history.count().unstack()
print (embarked)
p2 = embarked.plot(kind = 'bar', stacked = True, 
                    title = 'credit_history:Good Vs Bad', 
                    color = ['lightgreen','grey'], alpha = .90, ax = axes[0,1])
p2.set_xlabel('credit_history',size =30)
p2.set_ylabel('# Customers',size =30)
p2.legend(['Good','Bad'],prop={"size":30})

purpose = data.groupby(['purpose','status']).purpose.count().unstack()
print (purpose)
p3 = purpose.plot(kind = 'bar', stacked = True, 
                   title = 'purpose:Good vs Bad', 
                   color = ['lightgreen','grey'], alpha = .60, ax = axes[1,0])
p3.set_xlabel('purpose',size =30)
p3.set_ylabel('# Customers',size =30)
p3.legend(['Good','Bad'],prop={"size":30})


svaing_acc = data.groupby(['svaing_acc','status']).svaing_acc.count().unstack()
print (svaing_acc)
p4 = svaing_acc.plot(kind = 'bar', stacked = True, 
                    title = 'svaing_acc:Good Vs Bad', 
                    color = ['lightgreen','grey'], alpha = .60, ax = axes[1,1])
p4.set_xlabel('svaing_acc',size =30)
p4.set_ylabel('# Customers',size =30)
p4.legend(['Good','Bad'],prop={"size":30})


plt.show()


# Caterogical vs Target Analysis
sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (40,20))

gender = data.groupby(['present_emp_since','status']).present_emp_since.count().unstack()
print (gender)
p1 = gender.plot(kind = 'bar', stacked = True, 
                   title = 'present_emp_since:Good vs Bad', 
                   color = ['lightgreen','grey'], alpha = .60, ax = axes[0,0])
p1.set_xlabel('present_emp_since',size = 30)
p1.set_ylabel('# Customers',size = 30)
p1.legend(['Good','Bad'],prop={"size":30})


embarked = data.groupby(['personal_status','status']).personal_status.count().unstack()
print (embarked)
p2 = embarked.plot(kind = 'bar', stacked = True, 
                    title = 'personal_status:Good Vs Bad', 
                    color = ['lightgreen','grey'], alpha = .60, ax = axes[0,1])
p2.set_xlabel('personal_status',size = 30)
p2.set_ylabel('# Customers',size = 30)
p2.legend(['Good','Bad'],prop={"size":30})

purpose = data.groupby(['other_debtors','status']).other_debtors.count().unstack()
print (purpose)
p3 = purpose.plot(kind = 'bar', stacked = True, 
                   title = 'other_debtors:Good vs Bad', 
                   color = ['lightgreen','grey'], alpha = .60, ax = axes[1,0])
p3.set_xlabel('other_debtors',size = 30)
p3.set_ylabel('# Customers',size = 30)
p3.legend(['Good','Bad'],prop={"size":30})


svaing_acc = data.groupby(['property','status']).property.count().unstack()
print (svaing_acc)
p4 = svaing_acc.plot(kind = 'bar', stacked = True, 
                    title = 'property:Good Vs Bad', 
                    color = ['lightgreen','grey'], alpha = .60, ax = axes[1,1])
p4.set_xlabel('property',size = 30)
p4.set_ylabel('# Customers',size = 30)
p4.legend(['Good','Bad'],prop={"size":30})

plt.show()

# Caterogical vs Target Analysis

sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (40,20))

gender = data.groupby(['inst_plans','status']).inst_plans.count().unstack()
print (gender)
p1 = gender.plot(kind = 'bar', stacked = True, 
                   title = 'inst_plans:Good vs Bad', 
                   color = ['lightgreen','grey'], alpha = .60, ax = axes[0,0])
p1.set_xlabel('inst_plans')
p1.set_ylabel('# Customers')
p1.legend(['Good','Bad'])


embarked = data.groupby(['housing','status']).housing.count().unstack()
print (embarked)
p2 = embarked.plot(kind = 'bar', stacked = True, 
                    title = 'housing:Good Vs Bad', 
                    color = ['lightgreen','grey'], alpha = .60, ax = axes[0,1])
p2.set_xlabel('housing')
p2.set_ylabel('# Customers')
p2.legend(['Good','Bad'])

purpose = data.groupby(['job','status']).job.count().unstack()
print (purpose)
p3 = purpose.plot(kind = 'bar', stacked = True, 
                   title = 'job:Good vs Bad', 
                   color = ['lightgreen','grey'], alpha = .60, ax = axes[1,0])
p3.set_xlabel('job')
p3.set_ylabel('# Customers')
p3.legend(['Good','Bad'])


svaing_acc = data.groupby(['telephone','status']).telephone.count().unstack()
print (svaing_acc)
p4 = svaing_acc.plot(kind = 'bar', stacked = True, 
                    title = 'telephone:Good Vs Bad', 
                    color = ['lightgreen','grey'], alpha = .60, ax = axes[1,1])
p4.set_xlabel('telephone')
p4.set_ylabel('# Customers')
p4.legend(['Good','Bad'])

plt.show()
sns.set_style('darkgrid')
f, axes = plt.subplots(1,1, figsize = (30,10))

gender = data.groupby(['foreign_worker','status']).foreign_worker.count().unstack()
print (gender)
p1 = gender.plot(kind = 'bar', stacked = True, 
                   title = 'foreign_worker:Good vs Bad', 
                   color = ['lightgreen','grey'], alpha = 1, ax = axes)
p1.set_xlabel('foreign_worker',size=30)
p1.set_ylabel('# Customers',size=30)
p1.legend(['Good','Bad'],prop={"size":30})
plt.show()
# Build for training Model
data.head()

# Convert categorical variables into 'dummy' or indicator variables


original_data = data
checkin_acc = pd.get_dummies(data['checkin_acc'], drop_first = True) # drop_first prevents multi-collinearity
credit_history = pd.get_dummies(data['credit_history'], drop_first = True)
purpose = pd.get_dummies(data['purpose'], drop_first = True)
svaing_acc = pd.get_dummies(data['svaing_acc'], drop_first = True)
present_emp_since = pd.get_dummies(data['present_emp_since'], drop_first = True)
personal_status = pd.get_dummies(data['personal_status'], drop_first = True)
other_debtors = pd.get_dummies(data['other_debtors'], drop_first = True)
property = pd.get_dummies(data['property'], drop_first = True)
inst_plans = pd.get_dummies(data['inst_plans'], drop_first = True)
housing = pd.get_dummies(data['housing'], drop_first = True)
job = pd.get_dummies(data['job'], drop_first = True)
telephone = pd.get_dummies(data['telephone'], drop_first = True)
foreign_worker = pd.get_dummies(data['foreign_worker'], drop_first = True)

data = pd.concat([data, checkin_acc, credit_history,purpose,svaing_acc,present_emp_since,personal_status,other_debtors,property,inst_plans,housing,job,telephone,foreign_worker], axis = 1)
data.head()
data.drop(["checkin_acc", "credit_history","purpose","svaing_acc","present_emp_since","personal_status"], axis = 1, inplace = True)

data.drop(["other_debtors","property","inst_plans","housing","job","telephone","foreign_worker"], axis = 1, inplace = True)
data.head()
data.to_excel(r'Name.xlsx', index = False)
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create instance of standard scaler
scaler = StandardScaler()

# Fit scaler object to feature columns
scaler.fit(data.drop('status', axis = 1)) # Everything but target variable 

# Use scaler object to do a transform columns
scaled_features = scaler.transform(data.drop('status', axis = 1)) # performs the standardization by centering and scaling

# Use scaled features variable to re-create a features dataframe
df_feat = pd.DataFrame(scaled_features, columns = data.columns[:-1])

# Split
# Import
from sklearn.model_selection import train_test_split

# Create matrix of features
x = df_feat

# Create target variable
y = data['status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .15, random_state = 101)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Create instance of model
lreg = LogisticRegression()

# Pass training data into model
lreg.fit(x_train, y_train)
# Predict
y_pred_lreg = lreg.predict(x_test)

# Score It
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score

# Confusion Matrix
print('Logistic Regression')
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_lreg))
print('--'*40)

# Classification Report
print('Classification Report')
print(classification_report(y_test,y_pred_lreg))

# Accuracy
print('--'*40)
logreg_accuracy = round(accuracy_score(y_test, y_pred_lreg) * 100,2)
print('Accuracy', logreg_accuracy,'%')


# Import model
from sklearn.svm import SVC

# Instantiate the model
svc = SVC()

# Fit the model on training data
svc.fit(x_train, y_train)
# Predict
y_pred_svc = svc.predict(x_test)

# Score It
print('Support Vector Classifier')
print('\n')

# Confusion matrix
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_svc))
print('--'*40)

# Classification report
print('Classification Report')
print(classification_report(y_test, y_pred_svc))

# Accuracy
print('--'*40)
svc_accuracy = round(accuracy_score(y_test, y_pred_svc)*100,2)
print('Accuracy', svc_accuracy,'%')


# Create parameter grid
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

# Fit
# Import
from sklearn.model_selection import GridSearchCV

# Instantiate grid object
grid = GridSearchCV(SVC(),param_grid, refit = True, verbose = 1)#verbose is the text output describing the process

# Fit to training data
grid.fit(x_train,y_train)

# Call best_params attribute
print(grid.best_params_)
print('\n')
# Call best_estimators attribute
print(grid.best_estimator_)

y_pred_grid = grid.predict(x_test)

# Score It
# Confusion Matrix
print('SVC with GridSearchCV')
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_grid))
print('--'*40)
# Classification Report
print('Classification Report')
print(classification_report(y_test, y_pred_grid))

# Accuracy
print('--'*40)
svc_grid_accuracy = round(accuracy_score(y_test, y_pred_grid)*100,2)
print('Accuracy',svc_grid_accuracy,'%')


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# Fit
# Import model
from sklearn.svm import SVC

# Instantiate model object
ksvc= SVC(kernel = 'rbf', random_state = 0)

# Fit on training data
ksvc.fit(x_train_sc, y_train)
# Predict
y_pred_ksvc = ksvc.predict(x_test_sc)

# Score it
print('Kernel SVC')

# Confusion Matrix
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_ksvc))

# Classification Report
print('--'*40)
print('Classification Report')
print(classification_report(y_test, y_pred_ksvc))

# Accuracy
print('--'*40)
ksvc_accuracy = round(accuracy_score(y_test,y_pred_ksvc)*100,1)
print('Accuracy',ksvc_accuracy,'%')



# Fit
# Import model
from sklearn.tree import DecisionTreeClassifier

# Create model object
dtree = DecisionTreeClassifier()

# Fit to training sets
dtree.fit(x_train,y_train)

y_pred_dtree = dtree.predict(x_test)


# Score It
print('Decision Tree')
# Confusion Matrix
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_dtree))

# Classification Report
print('--'*40)
print('Classification Report',classification_report(y_test, y_pred_dtree))

# Accuracy
print('--'*40)
dtree_accuracy = round(accuracy_score(y_test, y_pred_dtree)*100,2)
print('Accuracy',dtree_accuracy,'%')
# Fit
# Import model object
from sklearn.ensemble import RandomForestClassifier

# Create model object
rfc = RandomForestClassifier(n_estimators = 200)

# Fit model to training data
rfc.fit(x_train,y_train)

# Predict
y_pred_rfc = rfc.predict(x_test)

# Score It
print('Random Forest')
# Confusion matrix
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_rfc))

# Classification report
print('--'*40)
print('Classification Report')
print(classification_report(y_test, y_pred_rfc))

# Accuracy
print('--'*40)
rf_accuracy = round(accuracy_score(y_test, y_pred_rfc)*100,2)
print('Accuracy', rf_accuracy,'%')


## KNN

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create instance of standard scaler
scaler = StandardScaler()

# Fit scaler object to feature columns
scaler.fit(data.drop('status', axis = 1)) # Everything but target variable 

# Use scaler object to do a transform columns
scaled_features = scaler.transform(data.drop('status', axis = 1)) # performs the standardization by centering and scaling

# Use scaled features variable to re-create a features dataframe
df_feat = pd.DataFrame(scaled_features, columns = data.columns[:-1])

# Split
# Import
from sklearn.model_selection import train_test_split

# Create matrix of features
x = df_feat

# Create target variable
y = data['status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 101)




# Function
error_rate = []
# Import model
from sklearn.neighbors import KNeighborsClassifier

for i in range (1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

# Plot error rate
plt.figure(figsize = (10,6))
plt.plot(range(1,40), error_rate, color = 'blue', linestyle = '--', marker = 'o', 
        markerfacecolor = 'green', markersize = 10)

plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
knn = KNeighborsClassifier(n_neighbors = 10)

# Fit new KNN on training data
knn.fit(x_train, y_train)
# Predict new KNN
y_pred_knn_op = knn.predict(x_test)

# Score it with new KNN
print('K-Nearest Neighbors(KNN)')
print('k = 5')

# Confusion Matrix
print('\n')
print(confusion_matrix(y_test, y_pred_knn_op))

# Classification Report
print('--'*40)
print('Classfication Report',classification_report(y_test, y_pred_knn_op))

# Accuracy
print('--'*40)
knn_op_accuracy =round(accuracy_score(y_test, y_pred_knn_op)*100,2)
print('Accuracy',knn_op_accuracy,'%')


import tensorflow as tf # Import tensorflow library
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create instance of standard scaler
scaler = StandardScaler()

# Fit scaler object to feature columns
scaler.fit(data.drop('status', axis = 1)) # Everything but target variable 

# Use scaler object to do a transform columns
scaled_features = scaler.transform(data.drop('status', axis = 1)) # performs the standardization by centering and scaling

# Use scaled features variable to re-create a features dataframe
df_feat = pd.DataFrame(scaled_features, columns = data.columns[:-1])

# Split
# Import
from sklearn.model_selection import train_test_split

# Create matrix of features
x = df_feat



from sklearn.preprocessing import normalize
x1 = normalize(x)

data.loc[data.status ==2, 'status'] = 0
y = data[["status"]]
y2 = np.array(y)
print (y2)
x_train, x_test, y_train, y_test = train_test_split(x1, y2, test_size = .15, random_state = 101)
#Build the model object
model = tf.keras.models.Sequential()
# Add the Flatten Layer
model.add(tf.keras.layers.Flatten())
# Build the input and the hidden layers
# model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
# Build the output layer
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x=x_train, y=y_train, epochs=75)
# Evaluate the model performance
test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
# Print out the model accuracy 
print('\nTest accuracy:', test_acc)
