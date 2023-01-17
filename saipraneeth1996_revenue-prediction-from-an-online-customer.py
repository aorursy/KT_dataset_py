#Importing DATA calucation and manipulation modules

import numpy as np

import pandas as pd



#Importing neccessary plotting libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#to filter all warnings

import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/online_shoppers_intention.csv')

data.shape
data.head()
data.info()
print('Descriptive statistics of Data')

data.describe().T
data.columns = ['admin_pages','admin_duration','info_pages','info_duration','product_pages', 'prod_duration',

                'avg_bounce_rate', 'avg_exit_rate','avg_page_value','spl_day','month','os','browser','region',

                'traffic_type','visitor_type','weekend','revenue']
data1 = data.copy()
#Replacing boolean values with binary values(1,0) 

data1.weekend = np.where(data.weekend == True,1,0)

data1.revenue = np.where(data.revenue == True,1,0)
data.month.unique()
#mapping months with numerical values

data1['month'] = data1['month'].map({'Feb':2,'Mar':3 ,'May':5,'June':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12})    
data['visitor_type'].value_counts()
#mapping months with numerical values

data1['visitor_type'] = data1['visitor_type'].map({'Returning_Visitor':0,'New_Visitor':1,'Other':2})
data1.head(10)
data1.info()
#Checking for missing values

pd.isnull(data1).sum()
cat = ['admin_pages','info_pages','spl_day', 'month','os', 'browser','region','traffic_type', 

       'visitor_type', 'weekend']



cont = ['admin_duration', 'info_duration','product_pages','prod_duration','avg_bounce_rate', 'avg_exit_rate','avg_page_value']
print('Correlation Heat map of the data')

plt.figure(figsize=(15,10))

mask = np.array(data1[cont].corr())

mask[np.tril_indices_from(data1[cont].corr())] = False

sns.heatmap(data1[cont].corr(),annot=True,mask = mask, fmt='.2f',vmin=-1,vmax=1)

plt.show()
def cat_data(i):

        sns.countplot(data[i])

        print('--'*60)

        plt.title("Count plot of "+str(i))

        plt.show()

        

for i in cat:

    cat_data(i)        
sns.countplot(data.revenue)
from scipy.stats import skew

sns.set() #Sets the default seaborn plotting style



def continous_data(i):

        sns.boxplot(data1[i])

        print('--'*60)

        plt.title("Boxplot of "+str(i))

        plt.show()

        plt.title("histogram of "+str(i))        

        sns.distplot(data1[i],bins=40,kde=True,color='blue')

        plt.show()

        print('skewness :',skew(data1[i]))

        

for i in cont:

    continous_data(i)        
for i in cont:

    Q1 = data1[i].quantile(0.25)

    Q3 = data1[i].quantile(0.75)

    IQR = Q3 - Q1

    upper = Q3 + 1.5*IQR

    lower = Q1 - 1.5*IQR

    outlier_count = data1[i][(data1[i] < lower) | (data1[i] > upper)].count()

    total = data1[i].count()

    percent = (outlier_count/total)*100

    print('Percentage of Outliers in {} column :: {}%'.format(i,np.round(percent,2)))
def cat_bivar(i):

        sns.barplot(data[i],data1.revenue)

        print('--'*60)

        plt.title("Bar-plot of Revenue against "+str(i))

        plt.show()

        

for i in cat:

    cat_bivar(i)        
sns.boxplot(x=data1.revenue, y=data1.avg_exit_rate)
sns.scatterplot(data1.admin_duration,data1.admin_pages)
sns.scatterplot(data1.prod_duration,data1.product_pages)
sns.scatterplot(data1.avg_bounce_rate,data1.avg_exit_rate)
sns.barplot(data1.revenue,data1.admin_duration)
sns.barplot(data1.revenue,data1.info_duration)
sns.barplot(data1.revenue,data1.prod_duration)
def f(x):

    return pd.Series(dict(avg_time_on_admintrative_pages=x['admin_duration'].mean(),

                         avg_time_on_info_pages=x['info_duration'].mean(),

                         avg_time_on_products=x['prod_duration'].mean(),

                         avg_bounce_rate =x['avg_bounce_rate'].mean(),

                         avg_exit_rate =x['avg_exit_rate'].mean(),

                         avg_page_value =x['avg_page_value'].mean()))





by_visitor_type = data1.groupby('visitor_type').apply(f)

by_visitor_type
data1.pivot_table(index='revenue',columns = 'visitor_type',values ='product_pages', aggfunc='mean')
data1.groupby('revenue')[['admin_duration','info_duration','product_pages','prod_duration','avg_page_value']].mean()
data1.groupby('revenue')[['avg_bounce_rate','avg_exit_rate']].mean()
# Importing train-test-split 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,classification_report
df = data1.copy()

y = df['revenue']

x = df.drop(['revenue'],axis=1)

# Splitting the data into train and test

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.30, random_state = 127)
y_pred = []

for i in range(0,ytest.shape[0]):

    y_pred.append(ytest.mode()[0])
y_pred = pd.Series(y_pred)

score = accuracy_score(ytest,y_pred)



pd.DataFrame({'Classifier':['Baseline model'], 'Accuracy': [score]})
df_cat = df.copy()
def categorize(col):

    l = []

    for i in col:

        if i == 0:

            l.append(0)

        elif i > 0 and i < 300:

            l.append(1)

        else: 

            l.append(2)

    return l               
def cat_pd(col):

    l = []

    for i in col:

        if i <= 300:

            l.append(0)

        elif i > 300 and i < 3000:

            l.append(1)

        else: 

            l.append(2)

    return l        
def cat_rates(col):

    l = []

    for i in col:

        if i <= 0.05:

            l.append(0)

        elif i > 0.05 and i < 0.15:

            l.append(1)

        else: 

            l.append(2)

    return l        
def cat_pv(col):

    l = []

    for i in col:

        if i == 0:

            l.append(0)

        elif i > 0 and i < 20:

            l.append(1)

        else: 

            l.append(2)

    return l      
cat_admin_duration = list(categorize(df.admin_duration)) 

cat_info_duration = list(categorize(df.info_duration))

cat_prod_duration = list(cat_pd(df.prod_duration)) 

cat_bounce_rate = list(cat_rates(df.avg_bounce_rate)) 

cat_exit_rate = list(cat_rates(df.avg_exit_rate)) 

cat_avg_page_value = list(cat_pv(df.avg_page_value)) 

#cat_product_pages = list(cat_pp(df.product_pages)) 
df_cat.admin_duration = cat_admin_duration

df_cat.info_duration = cat_info_duration 

df_cat.prod_duration = cat_prod_duration

df_cat.avg_bounce_rate = cat_bounce_rate

df_cat.avg_exit_rate = cat_exit_rate

df_cat.avg_page_value = cat_avg_page_value

#df_cat.product_pages = cat_product_pages
from scipy.stats import boxcox



dt = boxcox(df_cat['product_pages']+1,lmbda = 0.01)

df_cat.product_pages = pd.Series(dt)

sns.boxplot(df_cat.product_pages)

plt.show()

sns.distplot(df_cat.product_pages)

plt.show()
#replacing outlier with the upper whisker values

IQR = (np.percentile(df_cat.product_pages,75) - np.percentile(df_cat.product_pages,25))

upper = (np.percentile(df_cat.product_pages,75) + 1.5*IQR)

df_cat.product_pages = np.where(df_cat.product_pages > upper,upper,df_cat.product_pages)

sns.boxplot(df_cat.product_pages)
df_cat.head()
#One hot encoding

df_cat1 = pd.get_dummies(df_cat,columns = ['spl_day','month','os','browser','traffic_type','visitor_type'],drop_first=True)

df_cat1.head()
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



features = df_cat1.drop('revenue',axis =1)

scaler = MinMaxScaler()

x_std = scaler.fit_transform(features)

x_std = pd.DataFrame(x_std,columns=features.columns)

x_std.head()
cat1 = cat+['admin_duration','info_duration','prod_duration','avg_bounce_rate','avg_exit_rate','avg_page_value']
from scipy.stats import chisquare,chi2_contingency



cat_col = []

chi_pvalue = []

chi_name = []



def chi_sq(i):

    ct = pd.crosstab(df_cat['revenue'],df_cat[i])

    chi_pvalue.append(chi2_contingency(ct)[1])

    chi_name.append(i)



for i in cat1:

    chi_sq(i)



chi_data = pd.DataFrame()

chi_data['Pvalue'] = chi_pvalue

chi_data.index = chi_name



plt.figure(figsize=(11,8))

plt.title('P-Values of Chisquare with ''REVENUE'' as Target Categorical Attribute',fontsize=16)

x = chi_data.Pvalue.sort_values().plot(kind='barh')

x.set_xlabel('P-Values',fontsize=15)

x.set_ylabel('Independent Categorical Attributes',fontsize=15)

plt.show()
from scipy.stats import ttest_ind



cont_col = []

pvalue = []

name = []

t_stat = []



def t_test(i):

    sample_0 = df[df.revenue==0][i]

    sample_1 = df[df.revenue==1][i]

    t_statistic , p_value  =  ttest_ind(sample_0,sample_1)

    t_stat.append(t_statistic)

    pvalue.append(p_value)

    name.append(i)



for i in cont:

    t_test(i)



t_data = pd.DataFrame()

t_data['Pvalue'] = pvalue

t_data.index = name



plt.figure(figsize=(10,7))

plt.title('P-Values of T-test with ''REVENUE'' as Target Categorical Attribute',fontsize=16)

x = t_data.Pvalue.sort_values().plot(kind='barh')

x.set_xlabel('P-Values',fontsize=14)

x.set_ylabel('Independent Continuous Attributes',fontsize=14)

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc
X = df_cat1.drop(['revenue','region'],axis=1)

Y = df_cat1['revenue']



# Splitting the data into train and test

xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.30, random_state = 25)



# Splitting the STANDARDIZED data into train and test

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_std,Y,test_size=0.30, random_state = 25)
from sklearn.model_selection import GridSearchCV

#for decision tree: 

#parameter={'max_depth':np.arange(1,15),'criterion':['gini','entropy']}

#gs = GridSearchCV(dt,parameter,cv=4)

#gs.fit(xtrain,ytrain)

#gs.best_params_
dt = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=25)

dt.fit(xtrain,ytrain)

y_pred = dt.predict(xtest)



print(confusion_matrix(ytest,y_pred))

print(classification_report(ytest,y_pred))
importance =  pd.DataFrame()

importance['features'] =  xtrain.columns

importance['importance'] = dt.feature_importances_

importance[importance.importance>0].sort_values(by='importance',ascending = False)
training = dt.score(xtrain,ytrain)

score = accuracy_score(ytest,y_pred)

recall = recall_score(ytest,y_pred)

precision = precision_score(ytest,y_pred)

f1score = f1_score(ytest,y_pred)

fpr,tpr, _ = roc_curve(ytest, y_pred)

roc_auc = auc(fpr,tpr)





results = pd.DataFrame({'Classifier':['Decision Tree'],'Training Accuracy':[training],

                          'Test Accuracy': [score],'Recall':[recall],

                          'Precision':[precision],'F1 Score':[f1score],'AUC':[roc_auc]},index=[0])

results
#for KNN:

#parameter={'n_neighbors':np.arange(1,16,2)}

#gs = GridSearchCV(knn,parameter,cv=4,scoring='f1')

#gs.fit(xtrain,ytrain)

#gs.best_params_
knn=KNeighborsClassifier(n_neighbors=11)

knn.fit(xtrain,ytrain)

y_pred = knn.predict(xtest)



print(classification_report(ytest, y_pred))

print(confusion_matrix(ytest,y_pred))
training = knn.score(xtrain,ytrain)

score = accuracy_score(ytest,y_pred)

recall = recall_score(ytest,y_pred)

precision = precision_score(ytest,y_pred)

f1score = f1_score(ytest,y_pred)

fpr,tpr, _ = roc_curve(ytest, y_pred)

roc_auc = auc(fpr,tpr)



knn_results = pd.DataFrame({'Classifier':['KNN'], 'Training Accuracy':[training],

                          'Test Accuracy': [score],'Recall':[recall],

                          'Precision':[precision],'F1 Score':[f1score],'AUC':[roc_auc]},index=[1])



results = pd.concat([results,knn_results])

results
logistic = LogisticRegression()

logistic.fit(xtrain,ytrain)

y_pred = logistic.predict(xtest)



print(classification_report(ytest, y_pred))

print(confusion_matrix(ytest,y_pred))
score = accuracy_score(ytest,y_pred)

recall = recall_score(ytest,y_pred)

precision = precision_score(ytest,y_pred)

f1score = f1_score(ytest,y_pred)

training = logistic.score(xtrain,ytrain)

fpr,tpr, _ = roc_curve(ytest, y_pred)

roc_auc = auc(fpr,tpr)



log_results = pd.DataFrame({'Classifier':['Logistic'], 'Training Accuracy':[training],

                          'Test Accuracy': [score],'Recall':[recall],

                          'Precision':[precision],'F1 Score':[f1score],'AUC':[roc_auc]},index=[2])



results = pd.concat([results,log_results])

results
gnb = GaussianNB()

mnb = MultinomialNB()

bnb = BernoulliNB()



gnb.fit(xtrain,ytrain)

mnb.fit(xtrain,ytrain)

bnb.fit(xtrain,ytrain)



y_pred1 = gnb.predict(xtest)

y_pred2 = mnb.predict(xtest)

y_pred3 = bnb.predict(xtest)
score1 = accuracy_score(ytest,y_pred1)

score2 = accuracy_score(ytest,y_pred2)

score3 = accuracy_score(ytest,y_pred3)



recall1 = recall_score(ytest,y_pred1)

recall2 = recall_score(ytest,y_pred2)

recall3 = recall_score(ytest,y_pred3)



precision1 = precision_score(ytest,y_pred1)

precision2 = precision_score(ytest,y_pred2)

precision3 = precision_score(ytest,y_pred3)



f1score1 = f1_score(ytest,y_pred1)

f1score2 = f1_score(ytest,y_pred2)

f1score3 = f1_score(ytest,y_pred3)



training1 = gnb.score(xtrain,ytrain)

training2 = mnb.score(xtrain,ytrain)

training3 = bnb.score(xtrain,ytrain)



fpr1,tpr1, _ = roc_curve(ytest, y_pred1)

roc_auc1 = auc(fpr1,tpr1)

fpr2,tpr2, _ = roc_curve(ytest, y_pred2)

roc_auc2 = auc(fpr2,tpr2)

fpr3,tpr3, _ = roc_curve(ytest, y_pred3)

roc_auc3 = auc(fpr3,tpr3)



nb_results = pd.DataFrame({'Classifier':['Gaussian NB','Multinomial NB','Bernoulli NB'], 

                           'Training Accuracy':[training1,training2,training3],

                           'Test Accuracy': [score1,score2,score3],'Recall':[recall1,recall2,recall3],

                           'Precision':[precision1,precision2,precision3],

                           'F1 Score':[f1score1,f1score2,f1score3],'AUC':[roc_auc1,roc_auc2,roc_auc3]},index=[3,4,5])



results = pd.concat([results,nb_results])

results
### Class Imbalance in Training set

plt.title('Class Imbalance in Target Variable-Revenue')

plt.pie(ytrain.value_counts(),autopct='%.2f%%',labels=['Negative','Positive'])

plt.show()

print(ytrain.value_counts())
from imblearn.over_sampling import SMOTE



print("Before UpSampling, counts of label '1': {}".format(sum(ytrain==1)))

print("Before UpSampling, counts of label '0': {} \n".format(sum(ytrain==0)))



smt = SMOTE(random_state = 25)   #Synthetic Minority Over Sampling Technique

xtrain_bal, ytrain_bal = smt.fit_sample(xtrain,ytrain)



print("After UpSampling, counts of label '1': {}".format(sum(ytrain_bal==1)))

print("After UpSampling, counts of label '0': {} \n".format(sum(ytrain_bal==0)))



plt.title('Class Imbalance in Target Variable-Revenue after over sampling')

plt.pie(pd.Series(ytrain_bal).value_counts(),autopct='%.2f%%',labels=['Negative','Positive'])

plt.show()
#for decision tree balanced: 

#parameter={'max_depth':np.arange(1,18),'criterion':['gini','entropy']}

#gs = GridSearchCV(dt_bal,parameter,cv=4,scoring='f1')

#gs.fit(xtrain_bal,ytrain_bal)

#gs.best_params_
dt_bal = DecisionTreeClassifier(criterion='entropy',max_depth = 5,random_state=25)

dt_bal.fit(xtrain_bal,ytrain_bal)

y_pred = dt_bal.predict(xtest)



print(confusion_matrix(ytest,y_pred))

print(classification_report(ytest,y_pred))
score = accuracy_score(ytest,y_pred)

recall = recall_score(ytest,y_pred)

precision = precision_score(ytest,y_pred)

f1score = f1_score(ytest,y_pred)

training = dt_bal.score(xtrain,ytrain)

fpr,tpr, _ = roc_curve(ytest, y_pred)

roc_auc = auc(fpr,tpr)





bl_results = pd.DataFrame({'Classifier':['DT Balanced'], 'Training Accuracy':[training],

                          'Test Accuracy': [score],'Recall':[recall],

                          'Precision':[precision],'F1 Score':[f1score],'AUC':[roc_auc]},index=[0])



bl_results
#for RF balanced: 

#parameter={'n_estimators':np.arange(1,101)}

#gs = GridSearchCV(rfc,parameter,cv=3,scoring='roc_auc')

#gs.fit(xtrain_bal,ytrain_bal)

#gs.best_params_
from sklearn.ensemble import RandomForestClassifier 



rfc = RandomForestClassifier(n_estimators=51,random_state=25)

rfc.fit(xtrain_bal,ytrain_bal)

y_pred = rfc.predict(xtest)



print(confusion_matrix(ytest,y_pred))

print(classification_report(ytest,y_pred))
score = accuracy_score(ytest,y_pred)

recall = recall_score(ytest,y_pred)

precision = precision_score(ytest,y_pred)

f1score = f1_score(ytest,y_pred)

training = rfc.score(xtrain,ytrain)

fpr,tpr, _ = roc_curve(ytest, y_pred)

roc_auc = auc(fpr,tpr)





rf_results = pd.DataFrame({'Classifier':['RF Balanced'], 'Training Accuracy':[training],

                          'Test Accuracy': [score],'Recall':[recall],

                          'Precision':[precision],'F1 Score':[f1score],'AUC':[roc_auc]},index=[1])



bl_results = pd.concat([bl_results,rf_results])

bl_results
#parameter={'n_estimators':np.arange(1,101) }

#gs= GridSearchCV(abc,parameter,cv=3,scoring='f1')

#gs.fit(xtrain_bal,ytrain_bal)

#gs.best_params_
from sklearn.ensemble import AdaBoostClassifier



abc = AdaBoostClassifier(n_estimators=71,random_state=25)

abc.fit(xtrain_bal,ytrain_bal)

y_pred = abc.predict(xtest)



print(confusion_matrix(ytest,y_pred))

print(classification_report(ytest,y_pred))
score = accuracy_score(ytest,y_pred)

recall = recall_score(ytest,y_pred)

precision = precision_score(ytest,y_pred)

f1score = f1_score(ytest,y_pred)

training = abc.score(xtrain,ytrain)

fpr,tpr, _ = roc_curve(ytest, y_pred)

roc_auc = auc(fpr,tpr)





abc_results = pd.DataFrame({'Classifier':['ADA-boost Balanced'], 'Training Accuracy':[training],

                          'Test Accuracy': [score],'Recall':[recall],

                          'Precision':[precision],'F1 Score':[f1score],'AUC':[roc_auc]},index=[2])



bl_results = pd.concat([bl_results,abc_results])

bl_results
#parameter={'n_estimators':np.arange(1,101) }

#gs= GridSearchCV(gbc,parameter,cv=3,scoring='f1')

#gs.fit(xtrain_bal,ytrain_bal)
from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier(n_estimators=51,random_state=25)

gbc.fit(xtrain_bal,ytrain_bal)

y_pred = gbc.predict(xtest)



print(confusion_matrix(ytest,y_pred))

print(classification_report(ytest,y_pred))
score = accuracy_score(ytest,y_pred)

recall = recall_score(ytest,y_pred)

precision = precision_score(ytest,y_pred)

f1score = f1_score(ytest,y_pred)

training = gbc.score(xtrain,ytrain)

fpr,tpr, _ = roc_curve(ytest, y_pred)

roc_auc = auc(fpr,tpr)



gbc_results = pd.DataFrame({'Classifier':['Gradient-boost Balanced'],'Training Accuracy':[training], 

                          'Test Accuracy': [score],'Recall':[recall],

                          'Precision':[precision],'F1 Score':[f1score],'AUC':[roc_auc]},index=[3])



bl_results = pd.concat([bl_results,gbc_results])

bl_results
print('Area under Curve :',roc_auc)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristics of Gradient boost classifier')

plt.show()