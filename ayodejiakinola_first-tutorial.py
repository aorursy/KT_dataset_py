# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
transaction_data = pd.read_csv('/kaggle/input/first-data/Transaction.csv',sep='|')
transaction_data.head()
transaction_data.dtypes
product_data = pd.read_csv('/kaggle/input/first-data/Product.csv',sep='|')
product_data.head()
product_data.dtypes
customer_data = pd.read_csv('/kaggle/input/first-data/Customer.csv',sep=',')
customer_data.head()
customer_data.dtypes
customer_data.describe()
customer_data['INCOME'] = customer_data['INCOME'].map(lambda x : x.replace('$',''))
customer_data.head()
customer_data['INCOME']=customer_data['INCOME'].map(lambda x : int(x.replace(',','')))
customer_data.dtypes
customer_data.describe()
customer_data["MARITAL STATUS"].unique()
from datetime import datetime
#customer_data['ENROLLMENT DATE']=[customer_data['ENROLLMENT DATE'].fillna("0/0/0")]
#customer_data['ENROLLMENT DATE'] = customer_data['ENROLLMENT DATE'][customer_data['ENROLLMENT DATE'].notnull()].map(lambda x:datetime.strptime(x,'%d-%m-%y'))
print("has customer_data any null value?",customer_data.isnull().values.any())
print("has transaction_data any null value?",transaction_data.isnull().values.any())
print("has product_data any null value?",product_data.isnull().values.any())
#customer_data.columns[customer_data.columns.isnull().any()].tolist()
customer_data.apply(lambda x : sum(x.isnull()),axis=0)
product_data.apply(lambda x : sum(x.isnull()),axis=0)
transaction_data.apply(lambda x : sum(x.isnull()),axis=0)
import matplotlib.pyplot as plt
customer_data['MARITAL STATUS'].value_counts().plot(kind='bar')
plt.xlabel("Marital Status")
plt.ylabel("Frequency")
plt.show()
customer_data['AGE'].hist(bins=10)
plt.show()
plt.figure(figsize=(8,8))
plt.boxplot(customer_data.AGE, 0,'rs',1)
plt.grid(linestyle='-',linewidth=1)
plt.show()
trans_product = transaction_data.merge(product_data, how ='inner', left_on='PRODUCT NUM', right_on='PRODUCT CODE')
trans_product.head()
trans_product['UNIT LIST PRICE'] = trans_product['UNIT LIST PRICE'].map(lambda x : float(x.replace('$','')))
trans_product ['Total_Price'] = \
trans_product['UNIT LIST PRICE'] * trans_product['QUANTITY PURCHASED']*(1-trans_product['DISCOUNT TAKEN'])
#from datetime import datetime
#trans_product['TRANSACTION DATE']=trans_product['TRANSACTION DATE'].map(lambda x : datetime.strptime(x,'%d/%m/%y'))
trans_product.head()
Income_by_product = \
trans_product.groupby('PRODUCT CATEGORY').agg({'Total_Price':'sum'}).sort_values('Total_Price',ascending=False)
Income_by_product.head()
Revenue_by_product = Income_by_product.rename(columns={'Total_Price':'Total_Revenue'})
Revenue_by_product.head()
Revenue_by_product['Total_Revenue'].plot(kind='pie',autopct='%1.1f%%',legend=True)
customer_prod_categ = trans_product.groupby(['CUSTOMER NUM','PRODUCT CATEGORY']).agg({'Total_Price':'sum'})
customer_prod_categ.head()
customer_prod_categ = customer_prod_categ.reset_index()
customer_pivot = customer_prod_categ.pivot(index='CUSTOMER NUM',columns='PRODUCT CATEGORY',values='Total_Price')
customer_pivot.head()
recent_trans_total_spent= trans_product.groupby('CUSTOMER NUM').agg({'TRANSACTION DATE':'max','Total_Price':'sum'}).rename(columns={'TRANSACTION DATE':'RECENT TRANSACTION DATE','Total_Price':'TOTAL SPENT'})
recent_trans_total_spent.head()
customer_KPIs=customer_pivot.merge(recent_trans_total_spent,how='inner',left_index=True, right_index=True)
customer_KPIs=customer_KPIs.fillna(0)
customer_KPIs.head()
customer_all_view=customer_data.merge(customer_KPIs,how='inner',left_on='CUSTOMERID',right_index=True)
customer_all_view.head()
table=pd.crosstab(customer_all_view['GENDER'], customer_all_view['LOYALTY GROUP'])
table.head()
table.plot(kind='bar',stacked=True,figsize=(6,6))
plt.show()
table=pd.crosstab(customer_all_view['EXPERIENCE SCORE'], customer_all_view['LOYALTY GROUP'])
table.head()
table.plot(kind='bar',stacked=True,figsize=(6,6))
plt.show()
table=pd.crosstab(customer_all_view['MARITAL STATUS'], customer_all_view['LOYALTY GROUP'])
table.head()
table.plot(kind='bar',stacked=True,figsize=(6,6))
plt.show()
customer_all_view['AGE_BINNED']=pd.cut(customer_all_view['AGE'],10)
table=pd.crosstab(customer_all_view['AGE_BINNED'], customer_all_view['LOYALTY GROUP'])
table.head()
table.plot(kind='bar',stacked=True,figsize=(6,6))
plt.show()
customer_all_view.groupby(['LOYALTY GROUP']).agg({'AGE':'mean'})
fig = plt.figure(1, figsize = (9,6))
ax = fig.add_subplot(111)
plot1=customer_all_view['AGE'][customer_all_view['LOYALTY GROUP']=="enrolled"]
plot2=customer_all_view['AGE'][customer_all_view['LOYALTY GROUP']=="notenrolled"]
list1=[plot1,plot2]
ax.boxplot(list1,0,'rs',1)
ax.set_xticklabels(['Enrolled','Not Enrolled'])
plt.show()
customer_all_view['TOTAL_SPENT_BINNED']=pd.cut(customer_all_view['TOTAL SPENT'],10)
table=pd.crosstab(customer_all_view['TOTAL_SPENT_BINNED'], customer_all_view['LOYALTY GROUP'])
table.head()
table.plot(kind='bar',stacked=True,figsize=(6,6))
plt.show()
plt.scatter(customer_all_view['AGE'],customer_all_view['TOTAL SPENT'])
plt.xlabel("AGE")
plt.ylabel("TOTAL SPENT")
plt.show()
from scipy.stats import pearsonr
pearsonr(customer_all_view['AGE'],customer_all_view['TOTAL SPENT'])
plt.scatter(customer_all_view['INCOME'],customer_all_view['TOTAL SPENT'])
plt.xlabel("INCOME")
plt.ylabel("TOTAL SPENT")
plt.show()
from scipy.stats import pearsonr
pearsonr(customer_all_view['INCOME'],customer_all_view['TOTAL SPENT'])
table=customer_all_view.groupby(['EXPERIENCE SCORE']).agg({'TOTAL SPENT':'mean'})
table.plot(kind='bar')
plt.xlabel("EXPERIENCE SCORE")
plt.ylabel("Average Total Spent Per Score")
plt.show()
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
cluster_input = customer_all_view[['INCOME','TOTAL SPENT']]
cluster_input.head()
Kmeans_model=KMeans(n_clusters=4)
Kmeans_model
cluster_output = Kmeans_model.fit_predict(cluster_input)
cluster_output
cluster_output_pd=pd.DataFrame(cluster_output,columns=['segment'])
cluster_output_pd.head()
segment_DF = pd.concat([cluster_input,cluster_output_pd],axis=1)
segment_DF.head()
Kmeans_model.cluster_centers_
segment_DF[segment_DF.segment==0].head()
import matplotlib.pyplot as plt
plt.scatter(segment_DF[segment_DF.segment==0]['INCOME'],segment_DF[segment_DF.segment==0]['TOTAL SPENT'],s=50, c='purple', label='Cluster1')

plt.scatter(segment_DF[segment_DF.segment==1]['INCOME'],segment_DF[segment_DF.segment==1]['TOTAL SPENT'],s=50, c='cyan', label='Cluster2')

plt.scatter(segment_DF[segment_DF.segment==2]['INCOME'],segment_DF[segment_DF.segment==2]['TOTAL SPENT'],s=50, c='blue', label='Cluster3')

plt.scatter(segment_DF[segment_DF.segment==3]['INCOME'],segment_DF[segment_DF.segment==3]['TOTAL SPENT'],s=50, c='green', label='Cluster4')

plt.scatter(Kmeans_model.cluster_centers_[:,0],Kmeans_model.cluster_centers_[:,1], s=200,marker='s', c='red', alpha=0.7, label='centroids')

plt.title("Customer Segement using Kmeans - K=4")
plt.xlabel("INCOME")
plt.ylabel("TOTAL SPENT")
plt.legend()
plt.show()
customer_demography=pd.concat([customer_all_view,cluster_output_pd],axis=1)
customer_demography.head()
customer_demography.groupby('segment').agg({'AGE':'mean','HOUSEHOLD SIZE':'median'})
def percentage_loyalty(series):
    percent=100*series.value_counts()['enrolled']/series.count()
    return percent
customer_demography.groupby('segment').agg({'AGE':'mean','HOUSEHOLD SIZE':'median','LOYALTY GROUP': percentage_loyalty})
DF_classification = customer_all_view[['INCOME','AGE','EXPERIENCE SCORE','TOTAL SPENT','LOYALTY GROUP']]
DF_classification.head()
target_feature = pd.DataFrame(DF_classification['LOYALTY GROUP'])
target_feature.head()
DF_classification = DF_classification.drop(['LOYALTY GROUP'],axis=1)
DF_classification.head()
from sklearn import preprocessing
DF_classification_column_names = DF_classification.columns.values
DF_classification = preprocessing.minmax_scale(DF_classification)
DF_classification = pd.DataFrame(DF_classification, columns=DF_classification_column_names)
DF_classification.head()
DF_classification = pd.concat([DF_classification,target_feature],axis=1)
DF_classification.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(DF_classification[['EXPERIENCE SCORE','TOTAL SPENT','INCOME']],DF_classification['LOYALTY GROUP'],test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
clf_kNN = KNeighborsClassifier(n_neighbors=8)
clf_kNN
clf_kNN.fit(X_train,Y_train)
predicted = clf_kNN.predict(X_test)
from sklearn import metrics
acc = metrics.accuracy_score(Y_test,predicted)
print('accuracy = '+str(acc*100)+'%')
print(metrics.classification_report(Y_test,predicted))
from sklearn.tree import DecisionTreeClassifier
clf_Tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
clf_Tree
clf_Tree.fit(X_train,Y_train)
predicted = clf_Tree.predict(X_test)
from sklearn import metrics
acc = metrics.accuracy_score(Y_test,predicted)
print('accuracy = '+str(acc*100)+'%')
print(metrics.classification_report(Y_test,predicted))
clf_Tree.feature_importances_
from sklearn.model_selection import KFold
def Training_Testing_Accuracy_Only(model,train_data,train_labels,test_data,test_labels):
    model.fit(train_data,train_labels)
    predicted = model.predict(test_data)
    acc = metrics.accuracy_score(test_labels,predicted)
    print('accuracy = '+str(acc*100)+'%')
    return(acc)
kf = KFold(n_splits=10)
clf_Tree = DecisionTreeClassifier(criterion='entropy')
accuracy_list=[]
for train_index,test_index in kf.split(DF_classification[['EXPERIENCE SCORE','TOTAL SPENT','INCOME']]):
    X=DF_classification[['EXPERIENCE SCORE','TOTAL SPENT','INCOME']]
    Y=DF_classification['LOYALTY GROUP']
    X_train,X_test=X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test=Y.iloc[train_index],Y.iloc[test_index]
    accuracy=Training_Testing_Accuracy_Only(clf_Tree,X_train,Y_train,X_test,Y_test)
accuracy_list_for_each_K_neighbours=[]
for k_neighbours in range(1,15):
    clf_NN = KNeighborsClassifier(n_neighbors=k_neighbours)
    accuracy_list_k_fold = []
    accuracy_list_k_fold.append(accuracy)
    
for train_index,test_index in kf.split(DF_classification[['EXPERIENCE SCORE','TOTAL SPENT','INCOME']]):  
    X=DF_classification[['EXPERIENCE SCORE','TOTAL SPENT','INCOME']]
    Y=DF_classification['LOYALTY GROUP']
    X_train,X_test=X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test=Y.iloc[train_index],Y.iloc[test_index]
    accuracy=Training_Testing_Accuracy_Only(clf_Tree,X_train,Y_train,X_test,Y_test)
    
  
#accuracy_list_for_each_K_neighbours.append(100*sum(accuracy_list_k_fold)/len(accuracy_list_k_fold))
    
#print("Overall Accuracy for K_neighbours=",k_neighbours,"is",accuracy_list_for_each_K_neighbours[k_neighbours-1])
    
from matplotlib import pyplot as plt
#plt.plot(range(1,15),accuracy_list_for_each_K_neighbours)

#plt.xlabel(['Number of Neighbours'],loc='upper left')
#plt.ylabel("10 k-Fold Accuracy")
#plt.legend('Accuracy Per K neighbours')
#plt.show()
DF_input = customer_all_view[['GENDER','AGE','INCOME','EXPERIENCE SCORE','LOYALTY GROUP','HOUSEHOLD SIZE','MARITAL STATUS']]
DF_input.head()
def encode_loyalty(value):
    if value == "enrolled":
        return 1
    else:
        return 0
DF_input['LOYALTY GROUP'] = DF_input['LOYALTY GROUP'].apply(encode_loyalty)
DF_input.head()
DF_input = pd.get_dummies(DF_input)
DF_input.head()
from sklearn import preprocessing
DF_input_column_names = DF_input.columns.values
DF_input_np = preprocessing.minmax_scale(DF_input)
Reg_input_scaled=pd.DataFrame(DF_input_np,columns=DF_input_column_names)
Reg_input_scaled.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(Reg_input_scaled,customer_all_view['TOTAL SPENT'],test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train,Y_train)
Y_pred = regr.predict(X_test)
print('Coefficients: \n',regr.coef_)
Reg_input_scaled.columns.values
print('intercept: \n', regr.intercept_)
from sklearn.metrics import mean_squared_error
print("Mean squared error: %.2f" % mean_squared_error(Y_test,Y_pred))
