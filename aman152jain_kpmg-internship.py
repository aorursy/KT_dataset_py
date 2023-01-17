import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/econdse'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
transactions = pd.read_excel("/kaggle/input/econdse/B1863B00.xlsx", sheet_name='Transactions')

new_customer_lists = pd.read_excel("/kaggle/input/econdse/B1863B00.xlsx", sheet_name='NewCustomerList')

customer_demographic = pd.read_excel("/kaggle/input/econdse/B1863B00.xlsx", sheet_name='CustomerDemographic')

customer_add = pd.read_excel("/kaggle/input/econdse/B1863B00.xlsx", sheet_name='CustomerAddress')
transactions.info()
transactions.isnull().sum()
new_customer_lists.head()
temp1 = pd.merge(customer_add,customer_demographic, how = 'outer', on = 'customer_id')

df = pd.merge(transactions,temp1, how ='outer', on = 'customer_id')
df.head()
df.info()
df.isnull().sum()
df.columns
new_customer_lists.columns
df= df.drop(['online_order', 'address', 'postcode', 'country','first_name', 'last_name','DOB','job_title','deceased_indicator','transaction_id', 'product_id','transaction_date',

       'order_status', 'brand', 'product_line', 'product_class',

       'product_size', 'list_price', 'standard_cost',

       'product_first_sold_date','Unnamed: 15',

       'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19'] , axis= 1)
new_customer_lists= new_customer_lists.drop([ 'address', 'postcode', 'country','first_name', 'last_name','DOB','job_title','deceased_indicator', 'Rank', 'Value'] , axis= 1)
new_customer_lists.columns
df.columns
df.head()
new_customer_lists['state']= new_customer_lists['state'].replace('NSW','New South Wales')

new_customer_lists['state']= new_customer_lists['state'].replace('VIC','Victoria')
new_customer_lists.head()
df1= pd.read_excel ("../input/econdsekpmg-virtual-internship-2020/export_df1 (9).xlsx", sheet_name= 'final')

aa= pd.read_excel (r"../input/econdsekpmg-virtual-internship-2020/new (2).xlsx", sheet_name= 'Sheet2')
df2= pd.get_dummies(df1, drop_first= True)

nc= pd.get_dummies(aa, drop_first= True)
df2.head()
nc.head()
X = df2.drop('Range', axis=1)

y = df2['Range']
nc1= nc.drop('customer_id', axis=1)
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
logreg = LogisticRegression()

a= logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)

print(cm)

acc_log

y_prob_train= logreg.predict_proba(X_train)[:, 1]

y_prob_train.reshape(1,-1)

y_prob_train
y_prob= logreg.predict_proba(X_test)[:, 1]

y_prob.reshape(1,-1)

y_prob
from sklearn.metrics import classification_report

print(classification_report(y_test,Y_pred))



tn,fp,fn,tp = cm.ravel()

print('TRUE NEGATIVE:', tn)

print('FALSE POSITIVE:', fp)

print('FALSE NEGATIVE:', fn)

print('TRUE POSITIVE:', tn)



specificity = tn/(tn+fp)

print('specificity {:0.2f}'.format(specificity))

sensitivity = tp/(tp+fn)

print('sensitivity {:0.2f}'.format(sensitivity))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc

log_roc_auc= roc_auc_score(y_train,y_prob_train )

fpr, tpr, thresholds = roc_curve(y_train, y_prob_train )

roc_auc= auc(fpr,tpr)
plt.figure()

plt.plot(fpr, tpr, color= "blue" , label = 'ROC CURVE (area= %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0 ,1.0])

plt.ylim([0.0 ,1.05])  

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

plt.title('ROC CURVE')

plt.legend(loc= 'lower right')

plt.show()
log_roc_auc= roc_auc_score(y_test,y_prob)

fpr1, tpr1, thresholds = roc_curve(y_test,y_prob)

roc_auc= auc(fpr1,tpr1)
plt.figure()

plt.plot(fpr1, tpr1, color= "blue" , label = 'ROC CURVE (area= %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0 ,1.0])

plt.ylim([0.0 ,1.05])  

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

plt.title('ROC CURVE')

plt.legend(loc= 'lower right')

plt.show()
svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)

print(cm)

acc_svc

from sklearn.metrics import classification_report

print(classification_report(y_test,Y_pred))



tn,fp,fn,tp = cm.ravel()

print('TRUE NEGATIVE:', tn)

print('FALSE POSITIVE:', fp)

print('FALSE NEGATIVE:', fn)

print('TRUE POSITIVE:', tn)



specificity = tn/(tn+fp)

print('specificity {:0.2f}'.format(specificity))

sensitivity = tp/(tp+fn)

print('sensitivity {:0.2f}'.format(sensitivity))
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)

print(cm)

acc_knn
from sklearn.metrics import classification_report

print(classification_report(y_test,Y_pred))



tn,fp,fn,tp = cm.ravel()

print('TRUE NEGATIVE:', tn)

print('FALSE POSITIVE:', fp)

print('FALSE NEGATIVE:', fn)

print('TRUE POSITIVE:', tn)



specificity = tn/(tn+fp)

print('specificity {:0.2f}'.format(specificity))

sensitivity = tp/(tp+fn)

print('sensitivity {:0.2f}'.format(sensitivity))
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)

print(cm)

acc_linear_svc
from sklearn.metrics import classification_report

print(classification_report(y_test,Y_pred))



tn,fp,fn,tp = cm.ravel()

print('TRUE NEGATIVE:', tn)

print('FALSE POSITIVE:', fp)

print('FALSE NEGATIVE:', fn)

print('TRUE POSITIVE:', tn)



specificity = tn/(tn+fp)

print('specificity {:0.2f}'.format(specificity))

sensitivity = tp/(tp+fn)

print('sensitivity {:0.2f}'.format(sensitivity))
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)

print(cm)

acc_decision_tree
from sklearn.metrics import classification_report

print(classification_report(y_test,Y_pred))



tn,fp,fn,tp = cm.ravel()

print('TRUE NEGATIVE:', tn)

print('FALSE POSITIVE:', fp)

print('FALSE NEGATIVE:', fn)

print('TRUE POSITIVE:', tn)



specificity = tn/(tn+fp)

print('specificity {:0.2f}'.format(specificity))

sensitivity = tp/(tp+fn)

print('sensitivity {:0.2f}'.format(sensitivity))
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)

print(cm)

acc_random_forest
from sklearn.metrics import classification_report

print(classification_report(y_test,Y_pred))



tn,fp,fn,tp = cm.ravel()

print('TRUE NEGATIVE:', tn)

print('FALSE POSITIVE:', fp)

print('FALSE NEGATIVE:', fn)

print('TRUE POSITIVE:', tn)



specificity = tn/(tn+fp)

print('specificity {:0.2f}'.format(specificity))

sensitivity = tp/(tp+fn)

print('sensitivity {:0.2f}'.format(sensitivity))
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(nc1)
y_pred
submission = pd.DataFrame({

        "customer_id": nc["customer_id"],

        "Target": y_pred

    })
submission
submission['Target'].value_counts()
lr = pd.read_excel("../input/econdsekpmg-virtual-internship-2020/linear regression.xlsx")
import matplotlib as mpl

import statsmodels.formula.api as sm

from sklearn.linear_model import LinearRegression

from scipy import stats

import seaborn as sns
lr.columns
df= pd.get_dummies(lr, drop_first= True)
df.columns
df = df.rename(columns = {'brand_Norco Bicycles':'brand_Norco_Bicycles'})

df = df.rename(columns = {'brand_OHM Cycles':'brand_OHM_Cycles'})

df = df.rename(columns = {'brand_Trek Bicycles':'brand_Trek_Bicycles'})
df.columns
df.head(10)
df.info()
df.describe()
X = df[['online_order', 'brand_Norco_Bicycles', 'brand_OHM_Cycles',

       'brand_Solex', 'brand_Trek_Bicycles', 'brand_WeareA2B',

       'product_line_Road', 'product_line_Standard', 'product_line_Touring',

       'product_class_low', 'product_class_medium', 'product_size_medium','product_size_small']]

y = df['profit']
plt.figure(figsize= (10,10))

sns.heatmap(df.corr(),annot= True, cmap= "coolwarm")
from statsmodels.stats.anova import anova_lm

from statsmodels.formula.api import ols


reg= ols(formula= "profit ~ online_order + brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)

fit1= reg.fit()

print(fit1.summary())
reg= ols(formula= "profit ~ brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)

fit1= reg.fit(cov_type='HC1')

print(fit1.summary())
reg= ols(formula= "profit ~ brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)

fit1= reg.fit()

print(fit1.summary())
from statsmodels.formula.api import gls

reg= gls(formula= "profit ~ online_order + brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)

fit1= reg.fit()

print(fit1.summary())
reg= gls(formula= "profit ~ online_order + brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)

fit1= reg.fit(cov_type='HC1')

print(fit1.summary())
reg= ols(formula= "profit ~ brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)

fit1= reg.fit(cov_type='HC1')

print(fit1.summary())