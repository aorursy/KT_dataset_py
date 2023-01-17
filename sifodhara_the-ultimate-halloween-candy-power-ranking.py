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
import warnings
warnings.filterwarnings('ignore')
candy_data = pd.read_csv('/kaggle/input/the-ultimate-halloween-candy-power-ranking/candy-data.csv')
candy_data.head()
## lets check shape of the dataframe
candy_data.shape
## lets check some info about candy data 
candy_data.info()
## lets visualise the data
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,5), dpi=80)
plt.subplot(1,4,1)
sns.countplot(x=candy_data['chocolate'])
plt.title("candy contains chocolate")

plt.subplot(1,4,2)
sns.countplot(x=candy_data['fruity'])

plt.title("candy contains fruity")

plt.subplot(1,4,3)
sns.countplot(x=candy_data['caramel'])
plt.title("candy contains caramel")


plt.subplot(1,4,4)
sns.countplot(x=candy_data['peanutyalmondy'])
plt.title("candy contains peanutyalmondy")
#ax.set_yscale('log')
plt.tight_layout()
plt.show()
def map_type(x,y):
    if x == 1 and y==0:
        return("hard")
    elif x==0 and y==1:
        return ("bar")
    elif x==0 and y==0:
        return("soft")
    elif x==1 and y==1:
        return("soft")

candy_data['type'] = candy_data[['hard','bar']].apply(lambda x: map_type(x['hard'],x['bar']) , axis=1)
plt.figure(figsize=(15,5), dpi=80)
plt.subplot(1,4,1)
sns.countplot(x=candy_data['nougat'])
plt.title("candy contains nougat")

plt.subplot(1,4,2)
sns.countplot(x=candy_data['crispedricewafer'])

plt.title("candy contains crispedricewafer")

plt.subplot(1,4,3)
sns.countplot(x=candy_data['type'])
plt.title("type of candy")


plt.subplot(1,4,4)
sns.countplot(x=candy_data['pluribus'])
plt.title("pluribus")
#ax.set_yscale('log')
plt.tight_layout()
plt.show()
plt.figure(figsize=(15,5), dpi=80)
plt.subplot(1,3,1)
sns.distplot(candy_data['sugarpercent'])
plt.title("distribution of sugarpercent")

plt.subplot(1,3,2)
sns.distplot(candy_data['pricepercent'])
plt.title("distribution of pricepercent")




plt.subplot(1,3,3)
sns.distplot(candy_data['winpercent'])
plt.title("distribution of winpercent")
#ax.set_yscale('log')
plt.tight_layout()
plt.show()
## creating a feature that tells us one certain candy what type of features contains
candy_data['features'] = candy_data['chocolate']+candy_data['fruity']+candy_data['caramel']+candy_data['peanutyalmondy']+candy_data['nougat']+candy_data['crispedricewafer']
plt.figure(figsize=(15,20), dpi=80)
#plt.subplot(1,4,1)

sns.barplot(x="features",y="competitorname",data=candy_data)
plt.show()
top_candies_win = candy_data.sort_values(by='winpercent',ascending=False)
top_candies_win.head(10)
## top 10 voted candies
## let's look at least voted candies
top_candies_win.tail(10)
top_candies_sugary= candy_data.sort_values(by='sugarpercent',ascending=False)
top_candies_sugary.head(10)
top_candies_costly = candy_data.sort_values(by='pricepercent',ascending = False)
top_candies_costly.head(10)
## divided the data set into train and test
from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(candy_data,train_size=0.7,test_size=0.3,random_state=100)

## checking heat map of variables
candy_data_corr = df_train[['chocolate','fruity','caramel','peanutyalmondy','nougat','crispedricewafer','hard','bar','pluribus','sugarpercent','pricepercent','winpercent','features']]
sns.heatmap(candy_data_corr.corr(),annot=True)
plt.show()
## lets scale some variables for use in our predictive model
import sklearn
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scale_var = ['winpercent','features']
candy_data_corr[scale_var] = scaler.fit_transform(candy_data_corr[scale_var])
## lets check the head once 
candy_data_corr.head()
## divided train set into x and y 
##x - predictors
##y - we are going to predict this variable
y_train = candy_data_corr.pop('chocolate')
X_train = candy_data_corr
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

## using RFE for feature selection in our model
from sklearn.feature_selection import RFE
rfe = RFE(logreg,7)
rfe = rfe.fit(X_train,y_train)
cols = X_train.columns[rfe.support_]
cols
## feature selected by rfe 
## lets run one logistic regression model using features selected by rfe
import statsmodels.api as sm 
model1 = sm.GLM(y_train,sm.add_constant(X_train[cols]),family=sm.families.Binomial())
res = model1.fit()
res.summary()
## lets look for variance inflation factor of features 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['features'] = X_train[cols].columns
vif['vif'] = [variance_inflation_factor(X_train[cols].values,i) for i in range(X_train[cols].shape[1])]
vif['vif'] = round(vif['vif'], 2)
vif = vif.sort_values(by='vif',ascending=False)
vif
X_train_new = X_train[cols]
X_train_new.drop('crispedricewafer',axis=1,inplace=True)
## drop crispedricewafer due to its high p value that means its not statistically fit in our data
## lets check our new feature set
X_train_new.columns
model2 = sm.GLM(y_train,sm.add_constant(X_train_new),family=sm.families.Binomial())
model2 = model2.fit()
model2.summary()
## fit our model to new data set after removing one feature
## lets look for variance inflation factor of features 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['features'] = X_train_new.columns
vif['vif'] = [variance_inflation_factor(X_train_new.values,i) for i in range(X_train_new.shape[1])]
vif['vif'] = round(vif['vif'], 2)
vif = vif.sort_values(by='vif',ascending=False)
vif
## remove feature hard
X_train_new.drop('hard',axis=1,inplace=True)
## drop hard due to its high p value that means its not statistically fit in our data
model3 = sm.GLM(y_train,sm.add_constant(X_train_new),family=sm.families.Binomial())
model3 = model3.fit()
model3.summary()
## fit model to our revised dataset
## lets look for variance inflation factor of features 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['features'] = X_train_new.columns
vif['vif'] = [variance_inflation_factor(X_train_new.values,i) for i in range(X_train_new.shape[1])]
vif['vif'] = round(vif['vif'], 2)
vif = vif.sort_values(by='vif',ascending=False)
vif
X_train_new.drop('winpercent',axis=1,inplace=True)
## drop winpercent due to its high p value that means its not statistically fit in our data
model4 = sm.GLM(y_train,sm.add_constant(X_train_new),family=sm.families.Binomial())
model4 = model4.fit()
model4.summary()
## fit our model in revised data set
## lets look for variance inflation factor of features 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['features'] = X_train_new.columns
vif['vif'] = [variance_inflation_factor(X_train_new.values,i) for i in range(X_train_new.shape[1])]
vif['vif'] = round(vif['vif'], 2)
vif = vif.sort_values(by='vif',ascending=False)
vif
X_train_new.drop('pricepercent',axis=1,inplace=True)
## drop pricepercent due to its high p value that means its not statistically fit in our data
model5 = sm.GLM(y_train,sm.add_constant(X_train_new),family=sm.families.Binomial())
model5 = model5.fit()
model5.summary()
## again fit our model to revised data set
## lets look for variance inflation factor of features 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['features'] = X_train_new.columns
vif['vif'] = [variance_inflation_factor(X_train_new.values,i) for i in range(X_train_new.shape[1])]
vif['vif'] = round(vif['vif'], 2)
vif = vif.sort_values(by='vif',ascending=False)
vif
X_train_new.drop('bar',axis=1,inplace=True)
## drop bar due to its high p value that means its not statistically fit in our data
model6 = sm.GLM(y_train,sm.add_constant(X_train_new),family=sm.families.Binomial())
model6 = model6.fit()
model6.summary()
## again fit our model
## lets look for variance inflation factor of features 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['features'] = X_train_new.columns
vif['vif'] = [variance_inflation_factor(X_train_new.values,i) for i in range(X_train_new.shape[1])]
vif['vif'] = round(vif['vif'], 2)
vif = vif.sort_values(by='vif',ascending=False)
vif
## lets predict some candies whether they have chocolates or not
y_train_pred = model6.predict(sm.add_constant(X_train_new)).values.reshape(-1)
y_train_pred[:10]
final_pred = pd.DataFrame({'competitorname':df_train['competitorname'].values,'chocolate':y_train.values,'pred':y_train_pred})
final_pred.head()
## created a new dataframe with candies name and their predictio values
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve( final_pred.chocolate, final_pred.pred, drop_intermediate = False )
draw_roc(final_pred.chocolate, final_pred.pred)
numbers = [float(x/10) for x in range(10)]
for i in numbers:
    final_pred[i] = final_pred['pred'].map(lambda x:1 if x>i else 0)
final_pred.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(final_pred.chocolate, final_pred[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
final_pred['result'] = final_pred['pred'].apply(lambda x:1 if x>0.5 else 0)
final_pred.head()
confusion = confusion_matrix(final_pred.chocolate,final_pred.result)
confusion
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))
# Positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
