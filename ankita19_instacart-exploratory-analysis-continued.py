import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
order_products_train_df = pd.read_csv("../input/order_products__train.csv")
order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")
orders_df = pd.read_csv("../input/orders.csv")
products_df = pd.read_csv("../input/products.csv")
aisles_df = pd.read_csv("../input/aisles.csv")
departments_df = pd.read_csv("../input/departments.csv")

orders_df_2 = orders_df[orders_df['eval_set'] == 'prior']
orders_df_2 = orders_df_2[orders_df_2['days_since_prior_order'] == 2]
orders_df_3 = pd.merge(left=orders_df_2, right=order_products_prior_df, how='inner', on=['order_id'])
orders_df_3 = orders_df_3[orders_df_3['reordered'] == 1]
orders_df_3= pd.merge(left=orders_df_3, right=products_df, how='inner', on=['product_id'])
orders_df_3.head()
orders_df_3.to_csv('prior_orders_with_all_information.csv')
grouped =order_products_prior_df.groupby("product_id")["reordered"].aggregate({'Tot_reorders': 'count'}).reset_index()
grouped = pd.merge(grouped, products_df[['product_id', 'product_name']], how='inner', on=['product_id'])
grouped = grouped.sort_values(by='Tot_reorders', ascending=False).head(10)
grouped  = grouped.groupby(['product_name']).sum()['Tot_reorders'].sort_values(ascending=False)
fig, axes = plt.subplots(figsize=(24, 24))
sns.barplot(grouped.index, grouped.values)

order_products_prior_df_2 = order_products_prior_df[order_products_prior_df['reordered'] == 0]
order_products_prior_df_2 = order_products_prior_df_2.groupby(["reordered","product_id"])["order_id"].aggregate({'Least reordered products': 'count'}).reset_index()
order_products_prior_df_2 = order_products_prior_df_2.sort_values(by='Least reordered products', ascending=True).head(15)
order_products_prior_df_2 = pd.merge(left=products_df, right=order_products_prior_df_2, how='inner', on=['product_id'])
order_products_prior_df_2.sort_values(by='Least reordered products', ascending=True)

order_products_prior_df_3 = order_products_prior_df[order_products_prior_df['reordered'] == 1]
order_products_prior_df_3 = order_products_prior_df_3.groupby(["reordered","product_id"])["order_id"].aggregate({'Most reordered products': 'count'}).reset_index()
order_products_prior_df_3 = order_products_prior_df_3.sort_values(by='Most reordered products', ascending=False).head(15)
order_products_prior_df_3 = pd.merge(left=products_df, right=order_products_prior_df_3, how='inner', on=['product_id'])
order_products_prior_df_3 = order_products_prior_df_3.sort_values(by='Most reordered products', ascending=False)
order_products_prior_df_3.plot(kind='bar', y='Most reordered products', x='product_name');
items  = pd.merge(left =pd.merge(left=products_df, right=departments_df, how='inner'), right=aisles_df, how='inner')
items = items.groupby("aisle")["product_name"].aggregate({'Famous_aisle_count': 'count'}).reset_index()
items = items.sort_values(by='Famous_aisle_count', ascending=False).head(15)
items.plot(kind='bar',x='aisle', y='Famous_aisle_count')
order_products_prior_df_3 = order_products_prior_df[order_products_prior_df['reordered'] == 1]
order_products_prior_df_3 = order_products_prior_df_3.groupby(["reordered","product_id"])["order_id"].aggregate({'Most reordered aisles': 'count'}).reset_index()
order_products_prior_df_3 = order_products_prior_df_3.sort_values(by='Most reordered aisles', ascending=False)
order_products_prior_df_3 = pd.merge(left=products_df, right=order_products_prior_df_3, how='inner', on=['product_id'])
order_products_prior_df_3 = pd.merge(left=aisles_df, right=order_products_prior_df_3, how='inner', on=['aisle_id'])
order_products_prior_df_3 = order_products_prior_df_3.groupby(["reordered","aisle"])["Most reordered aisles"].aggregate({'Most reordered aisles': 'sum'}).reset_index()
order_products_prior_df_3 = order_products_prior_df_3.sort_values(by='Most reordered aisles', ascending=False).head(15)
order_products_prior_df_3.plot(kind='bar', y='Most reordered aisles', x='aisle');

items  = pd.merge(left =pd.merge(left=products_df, right=departments_df, how='inner'), right=aisles_df, how='inner')
items = items.groupby("department")["product_name"].aggregate({'Famous_department_count': 'count'}).reset_index()
items = items.sort_values(by='Famous_department_count', ascending=False).head(15)
f, ax = plt.subplots(figsize=(12, 10))
plt.xticks(rotation='vertical')
sns.barplot(items.department, items.Famous_department_count)
plt.ylabel('Famous department with highest number of products')
plt.xlabel('Department Name')
order_products_prior_df_3 = order_products_prior_df[order_products_prior_df['reordered'] == 1]
order_products_prior_df_3 = order_products_prior_df_3.groupby(["reordered","product_id"])["order_id"].aggregate({'Most reordered department': 'count'}).reset_index()
order_products_prior_df_3 = order_products_prior_df_3.sort_values(by='Most reordered department', ascending=False)
order_products_prior_df_3= pd.merge(left=products_df, right=order_products_prior_df_3, how='inner', on=['product_id'])
order_products_prior_df_3 = pd.merge(left=departments_df, right=order_products_prior_df_3, how='inner', on=['department_id'])
order_products_prior_df_3 = order_products_prior_df_3.groupby(["reordered","department"])["Most reordered department"].aggregate({'Most reordered department': 'sum'}).reset_index()
order_products_prior_df_3 = order_products_prior_df_3.sort_values(by='Most reordered department', ascending=False).head(15)
order_products_prior_df_3.plot(kind='bar', y='Most reordered department', x='department');
items  = pd.merge(left =pd.merge(left=products_df, right=departments_df, how='inner'), right=aisles_df, how='inner')
items.head()

complete_train = pd.merge(left = items, right = order_products_train_df, how = 'inner', on=['product_id'])
complete_train.head()


#orders_df.head()
orders_df_2 = orders_df[orders_df['eval_set'] == 'train']
orders_df_2.head()
complete_train_2 = pd.merge(left = complete_train, right = orders_df_2, how = 'inner', on=['order_id'])
complete_train_2.head()
complete_train_3 = complete_train_2.drop(['product_name','department','aisle','eval_set'], axis=1)
complete_train_3.head()

X = complete_train_3[['product_id', 'aisle_id', 'department_id', 'order_id', 'order_dow', 'order_hour_of_day','days_since_prior_order']]
y = complete_train_3['reordered']

X.head()
y.head()

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(
#X, y, test_size=0.3, random_state=101)
#training, validation = train_test_split(complete_train_3, train_size=.60)
from sklearn.linear_model import LogisticRegression
logModel = LogisticRegression(
    penalty='l2', 
    fit_intercept=True,
    class_weight="balanced"
)
logModel.fit(X, y)
#Calculate score for the Linear Regression model
#logModel.score(X_train, y_train)
#Predict whether an item is reordered or not on test data set 
predictions = logModel.predict(X)
#print (predictions)
probs = logModel.predict_proba(X) 
print(probs)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

# generate evaluation metrics
print (metrics.accuracy_score(y, predictions)) 
print (metrics.roc_auc_score(y, probs[:, 1]))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y, predictions)
confusion_matrix
from sklearn.metrics import classification_report
print(classification_report(y, predictions))
#print (metrics.confusion_matrix(y_test, predicted)) 
#print( metrics.classification_report(y_test, predicted))
#MODEL EVALUATION USING CROSS-VALIDATION
#evaluate model using 10-fold cross-validation, to see if the accuracy holds up more rigorously.
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10) 
print (scores) 
print (scores.mean())
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


##Computing false and true positive rates
fpr, tpr,_=roc_curve(logModel.predict(X),y,drop_intermediate=False)

import matplotlib.pyplot as plt
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()
roc_auc_score(logModel.predict(X),y)
numeric_cols = complete_train_3.columns[complete_train_3.dtypes != 'object']
numeric_cols
# Make test set predictions
predictions = logModel.predict(X)
print (predictions)
import statsmodels.api as sm
X = complete_train_3[['product_id', 'aisle_id', 'department_id', 'order_id', 'order_dow', 'order_hour_of_day','days_since_prior_order']]
y = complete_train_3['reordered']

logit = sm.Logit(y, X)

# fit the model
result = logit.fit()
print (result.summary())
# look at the confidence interval of each coeffecient
print (result.conf_int())
# odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print (np.exp(conf))
