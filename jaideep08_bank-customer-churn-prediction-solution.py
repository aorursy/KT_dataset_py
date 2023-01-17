
import pandas as pd
import warnings
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
import numpy as np
import random as rd
import seaborn as sns
warnings.filterwarnings("ignore")
#!kaggle datasets download -d jaideep08/bank-customer-churn-prediction -p /content
#!unzip \*.zip
data = pd.read_csv(r"../input/churn_prediction.csv")
data.head()

data.shape
data.dtypes
data.describe()
data['churn'].value_counts()
fig = plt.figure(figsize = (20,20))
ax = fig.gca()
data.hist(ax =  ax, bins = 25, color='orangered')
# dropping customer id variable
data.drop('customer_id', axis = 1, inplace = True)

data.isnull().sum()
data['gender'].value_counts()
data['gender'].fillna(value='Male',inplace=True)
data.isnull().sum()
data.boxplot('dependents')
data['dependents'] = np.where(data['dependents'] >10, data['dependents'].median(),data['dependents'])
data.boxplot('dependents')
temp = ['gender','occupation','dependents']
for i in temp:
    print('************ Value Count in', i, '************')
    print(data[i].value_counts())
    print('')
data.plot.scatter('age','dependents', color='royalblue')
# no strong pattern found though
# so imputing missing dependents values using mode i.e. 0
data['dependents'].fillna(0,inplace=True)
data['occupation'].fillna(value = 'self_employed', inplace=True)
data['occupation'].value_counts()
data.isnull().sum()
data = data[data['city'].notna()]
# taken rows where city is not null, not dropped
data.plot.scatter('days_since_last_transaction','dependents', color='b')
data.plot.scatter('days_since_last_transaction','age', color='k')
# for days since last transaction imputing them with mode i.e. 0 because majority have 0
data['days_since_last_transaction'].fillna(0,inplace=True)
data.isnull().sum()

# target variable is churn
data['churn'].value_counts().plot(kind = 'bar', color='r')
plt.xlabel('Churn')
plt.ylabel('Frequency')
data['age'].plot.hist(bins = 20, color='c')
plt.xlabel('age', fontsize=12)
plt.xlabel('Age')
data['age'].value_counts()
data['occupation'].value_counts()
data['gender'].value_counts()/len(data)
plt.figure(figsize=(6,3))
data['vintage'].plot.hist(bins=30,color='0.25')

plt.figure(figsize=(6,3))
data['days_since_last_transaction'].plot.hist(bins=30,color='0.25')
data['vintage'] = data['vintage'].transform(func='sqrt')
data['days_since_last_transaction'] = data['days_since_last_transaction'].transform(lambda x:x**0.5)
plt.figure(figsize=(6,3))
data['vintage'].plot.hist(bins=30,color='orangered')

plt.figure(figsize=(6,3))
data['days_since_last_transaction'].plot.hist(bins=30,color='orangered')

data.columns
data.plot.scatter('average_monthly_balance_prevQ','churn', color='#ba9723')
plt.ylabel('Churn')
fig, ax = plt.subplots()
colors = {'self_employed':'red', 'salaried':'blue', 'student':'green', 'retired':'yellow', 'company':'black'}
ax.scatter(data['customer_nw_category'], data['churn'], c=data['occupation'].apply(lambda x: colors[x]))
plt.title('plot between customer_nw_category, occupation and churn value')
plt.xlabel('Customer Net Worth Category')
plt.ylabel('Churn value')
plt.legend()
plt.show()
fig, ax = plt.subplots()
colors = {'self_employed':'red', 'salaried':'blue', 'student':'green', 'retired':'yellow', 'company':'black'}
ax.scatter(data['age'], data['churn'], c=data['occupation'].apply(lambda x: colors[x]))
plt.title('Plot between age, occupation and churn value')
plt.xlabel('Age')
plt.ylabel('Churn Value')
#plt.legend(['self_employed','salaried','student','retired','company'])
plt.show()
pd.crosstab(data['churn'],data['occupation']).plot.bar()
plt.ylabel('Frequency')
plt.xlabel('Churn')
data.groupby('occupation')['vintage'].mean()
data.groupby('churn')['vintage'].mean().plot.bar(color='#b81c8b')
plt.ylabel('Frequency')
plt.xlabel('Churn')
data.groupby('occupation')['vintage'].mean().plot.bar(color='hotpink')
plt.xlabel('Occupation')
plt.ylabel('Vintage')
temp_data = data.loc[(data['occupation']=='self_employed')&(data['age']<20)]
temp_data['churn'].plot.hist(bins=50, color='orangered')
plt.xlabel('Churn')
plt.title('For self employed younger than 20')
temp_data = data.loc[(data['occupation']=='salaried')&(data['age']>20)&(data['age']<60)]
temp_data['churn'].plot.hist(bins=50, color='r')
plt.title('For Salaried between 20-60 Age Group')
plt.xlabel('Churn Value')
temp_data = data.loc[(data['occupation']=='retired')&(data['age']>60)]
temp_data['churn'].plot.hist(bins=50, color='orangered')
plt.title('For retired')
plt.xlabel('Churn Value')
fig, ax = plt.subplots()
colors = {0:'green', 1:'red'}
ax.scatter(data['vintage'], data['average_monthly_balance_prevQ'], c=data['churn'].apply(lambda x: colors[x]))
plt.title('plot between vintage, average_monthly_balance_prevQ and churn value')
plt.xlabel('Vintage')
plt.ylabel('Average Monthly Balance Previous Quarter')
plt.legend()
plt.show()
data['vintage'].corr(data['churn'])
data['average_monthly_balance_prevQ2'].corr(data['churn'])
#data.corr()
plt.rcParams['figure.figsize'] = (30, 20)
sns.heatmap(data[['vintage','age','dependents','customer_nw_category','days_since_last_transaction','current_balance','previous_month_end_balance','average_monthly_balance_prevQ','average_monthly_balance_prevQ2','current_month_credit','previous_month_credit','current_month_debit','previous_month_debit','current_month_balance','previous_month_balance','churn',]].corr(), annot = True)

plt.title('Histogram of the Dataset', fontsize = 30)
plt.show()

data = pd.get_dummies(data,dtype = 'int')
x = data.drop('churn',axis=1)
y = data['churn']
x.shape, y.shape
#from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y= train_test_split(x, y, test_size = 0.20, stratify = y, random_state = 42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
valid_x = scaler.fit_transform(valid_x)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_x, train_y)
pred_train = lr.predict_proba(train_x)
pred_valid = lr.predict_proba(valid_x)
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, plot_roc_curve, f1_score
from sklearn.metrics import plot_confusion_matrix, plot_precision_recall_curve, recall_score, precision_score
print('ROC score for predict_proba w.r.t train data: ',roc_auc_score(train_y, pred_train[:,1]))
print('ROC score for predict_proba w.r.t validation data: ',roc_auc_score(valid_y, pred_valid[:,1]))
valid_prediction = lr.predict(valid_x)
print('Accuracy score for predict: ',accuracy_score(valid_prediction, valid_y))
confusion_matrix(valid_prediction, valid_y)
without_city_branch = data.drop(['city','branch_code'], axis = 1)
without_city_branch.columns
x_without_city_branch = without_city_branch.drop('churn',axis=1)
y_without_city_branch = without_city_branch['churn']
train_x_wc, valid_x_wc, train_y_wc, valid_y_wc= train_test_split(x_without_city_branch, y_without_city_branch, test_size = 0.20, stratify = y_without_city_branch, random_state = 42)
train_x_wc = scaler.fit_transform(train_x_wc)
valid_x_wc = scaler.fit_transform(valid_x_wc)
lr = LogisticRegression()
lr.fit(train_x_wc, train_y_wc)
pred_train_wc = lr.predict_proba(train_x_wc)
pred_valid_wc = lr.predict_proba(valid_x_wc)
#probs = pred_valid_wc[:,1]
valid_prediction_wc = lr.predict(valid_x_wc)
print('ROC score for predict_proba w.r.t train data without city: ',roc_auc_score(train_y_wc, pred_train_wc[:,1]))
print('ROC score for predict_proba w.r.t validation data without city: ',roc_auc_score(valid_y_wc, pred_valid_wc[:,1]))
print('ROC score for predict w.r.t validation data without city: ',roc_auc_score(valid_prediction_wc, valid_y_wc))

print('Accuracy score for predict without city ',accuracy_score(valid_prediction_wc, valid_y_wc))
print('Recall score for predict without city ',recall_score(valid_prediction_wc, valid_y_wc))
confusion_matrix(valid_prediction_wc, valid_y_wc)
# plot roc curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(valid_prediction_wc,valid_y_wc)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, color='royalblue')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--', color='#8B0000')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.rcParams['figure.figsize'] = (6, 5)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y= train_test_split(x, y, test_size = 0.20, stratify = y, random_state = 42)

train_x = scaler.fit_transform(train_x)
valid_x = scaler.fit_transform(valid_x)
train_x.shape, valid_x.shape
dt = DecisionTreeClassifier(criterion="gini", max_depth = 3,splitter="random")
dt.fit(train_x,train_y)
dt_pred = dt.predict(valid_x)
print("Decision Trees Accuracy: ", accuracy_score(dt_pred,valid_y))
print("F1 Score: ", f1_score(valid_y, dt_pred, average='weighted'))
dt.score(train_x,train_y), dt.score(valid_x,valid_y)
x_without_city_branch = without_city_branch.drop('churn',axis=1)
y_without_city_branch = without_city_branch['churn']
train_x_wc, valid_x_wc, train_y_wc, valid_y_wc= train_test_split(x_without_city_branch, y_without_city_branch, test_size = 0.20, stratify = y_without_city_branch, random_state = 42)
train_x_wc = scaler.fit_transform(train_x_wc)
valid_x_wc = scaler.fit_transform(valid_x_wc)
valid_acc_score = []
train_acc_score = []
trainscore = []
validscore = []
for md in range(2,10):
    dt = DecisionTreeClassifier(criterion="gini", max_depth = md, splitter="random")
    dt.fit(train_x_wc,train_y_wc)
    dt_pred_wc = dt.predict(valid_x_wc)
    valid_acc_score.append(accuracy_score(dt_pred_wc, valid_y_wc))
    trainscore.append(dt.score(train_x_wc, train_y_wc))
    validscore.append(dt.score(valid_x_wc, valid_y_wc))

plt.plot(valid_acc_score, color='orangered')
plt.ylabel('Accuracy score')
plt.xlabel('Values for maximum depth')
plt.figure(figsize = (15,10))
plt.show()
frame = pd.DataFrame({'max_depth':range(2,10), 'train_acc':trainscore, 'valid_acc':validscore})
plt.figure(figsize=(12,6))
plt.plot(frame['max_depth'], frame['train_acc'], marker='o', label='train_acc')
plt.plot(frame['max_depth'], frame['valid_acc'], marker='o', label='valid_acc')
plt.xlabel('Depth of tree')
plt.ylabel('performance')
plt.legend()
train_x_wc, valid_x_wc, train_y_wc, valid_y_wc= train_test_split(x_without_city_branch, y_without_city_branch, test_size = 0.20, stratify = y_without_city_branch, random_state = 42)
train_x_wc = scaler.fit_transform(train_x_wc)
valid_x_wc = scaler.fit_transform(valid_x_wc)
dt = DecisionTreeClassifier(criterion="gini", max_depth = 2, splitter="random")
dt.fit(train_x_wc,train_y_wc)
dt_pred_wc = dt.predict(valid_x_wc)
print('Accuracy score for decisiontree-predict without city and branch code: ',accuracy_score(dt_pred_wc, valid_y_wc))
print("F1 Score without city and branch code: ", f1_score(dt_pred_wc, valid_y_wc, average='weighted'))
dt.score(train_x_wc,train_y_wc), dt.score(valid_x_wc,valid_y_wc)
from sklearn.model_selection import cross_val_score
score = cross_val_score(DecisionTreeClassifier(criterion="gini", max_depth = 5), X=train_x_wc, y=train_y_wc, cv=10)
score
score.mean()*100, score.std()*100
dtc = DecisionTreeClassifier(criterion="gini", max_depth = 5, splitter='random')
dtc.fit(train_x_wc,train_y_wc)
score = dtc.score(train_x_wc,train_y_wc)
score1 = dtc.score(valid_x_wc,valid_y_wc)
score,score1
from sklearn import tree
!pip install graphviz
decision_tree_image = tree.export_graphviz(dtc,out_file='tree.dot',feature_names=x_without_city_branch.columns,max_depth=2,filled=True)
#!dot -Tpng tree.dot -o tree.png
image = plt.imread('../input/tree.png')
plt.figure(figsize=(15,15))
plt.imshow(image)

model1 = LogisticRegression()
model1.fit(train_x_wc, train_y_wc)
pred1 = model1.predict(valid_x_wc)
model1.score(valid_x_wc, valid_y_wc)
#pred1[:10], valid_y_wc[:10]
model2 = DecisionTreeClassifier(criterion="gini", max_depth = 2, splitter='random')
model2.fit(train_x_wc, train_y_wc)
pred2 = model2.predict(valid_x_wc)
model2.score(valid_x_wc, valid_y_wc)
#pred2[:10], valid_y_wc[:10]
from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(max_depth=5,n_estimators=30,random_state = 42, max_leaf_nodes=2)
model3.fit(train_x_wc, train_y_wc)
pred3 = model3.predict(valid_x_wc)
model3.score(valid_x_wc, valid_y_wc)
#pred3[:10], valid_y_wc[:10]
from statistics import mode
final_pred_mode = np.array([])
for i in range(0,len(valid_x_wc)):
    final_pred_mode = np.append(final_pred_mode, mode([pred1[i], pred2[i], pred3[i]]))
from sklearn.metrics import accuracy_score
accuracy_score(valid_y_wc, pred1), accuracy_score(valid_y_wc, pred2), accuracy_score(valid_y_wc, pred3), accuracy_score(valid_y_wc, final_pred_mode)
df = pd.DataFrame(columns=['M1', 'M2', 'M3', 'Final_mode', 'Actual'])
df['M1'] = pred1
df['M2'] = pred2
df['M3'] = pred3
df['Final_mode'] = final_pred_mode
df['Actual'] = np.array(valid_y)
df.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
train_x_rf, valid_x_rf, train_y_rf, valid_y_rf= train_test_split(x_without_city_branch, y_without_city_branch, test_size = 0.2, stratify = y_without_city_branch)

train_x_rf = scaler.fit_transform(train_x_rf)
valid_x_rf = scaler.fit_transform(valid_x_rf)
lnodes_vals = []
train_x_rf, valid_x_rf, train_y_rf, valid_y_rf= train_test_split(x_without_city_branch, y_without_city_branch, test_size = 0.20, stratify = y_without_city_branch)
train_x_rf = scaler.fit_transform(train_x_rf)
valid_x_rf = scaler.fit_transform(valid_x_rf)
for n in range(2,7):
    RF = RandomForestClassifier(max_depth=22,n_estimators=30,random_state = 42, max_leaf_nodes=n)
    RF.fit(train_x_rf,train_y_rf)
    pred_RF = RF.predict(valid_x_rf)
    lnodes_vals.append(accuracy_score(valid_y_rf, pred_RF))
    
plt.plot(lnodes_vals, color='orangered')
plt.ylabel('Accuracy score')
plt.xlabel('Values for max leaf nodes')
plt.xlim(0,6)
plt.show()
RF = RandomForestClassifier(max_depth=22,n_estimators=30,random_state = 42, max_leaf_nodes=2)
RF.fit(train_x_rf,train_y_rf)
pred_RF = RF.predict(valid_x_rf)
print("Random Forest's Accuracy: ", accuracy_score(valid_y_rf, pred_RF))
print("F1 SCORE: ", f1_score(valid_y_rf, pred_RF, average='weighted'))
RF.score(train_x_rf,train_y_rf),RF.score(valid_x_rf,valid_y_rf)

importances = RF.feature_importances_
std = np.std([tree.feature_importances_ for tree in RF.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1][:40]

# Print the feature ranking
print("Feature Ranking:")

for f in range(train_x_rf.shape[1]):
    print("%d. Feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
fig = plt.figure(figsize=(20, 10));
plt.title("Relative Feature Importances")
plt.bar(range(train_x_rf.shape[1]), importances[indices],
       color="#FF6347", yerr=std[indices], align="center", ecolor='k')
plt.xticks(range(train_x_rf.shape[1]), indices)
plt.xlim([-1, train_x_rf.shape[1]])
plt.show()