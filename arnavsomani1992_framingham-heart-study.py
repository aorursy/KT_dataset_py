import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/datasets_4123_6408_framingham.csv')

df.head()


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

df.describe()
print (df.shape)

print (df.info())
round((df.isnull().sum()/len(df.index))*100,2)
print (df['glucose'].describe())

df['glucose'].median()
plt.figure(figsize = (10,5))



sns.boxplot(x=df['glucose'])
df = df.dropna()
print (df.shape)

print (df.info())
df.head()
def new(dataframe, features, rows, col):

    fig = plt.figure(figsize = (20,20))

    for i,feature in enumerate(features):

        ax = fig.add_subplot(rows,col, i+1)

        dataframe[feature].hist(bins=20,ax=ax)

        ax.set_title(feature+" Distribution")



    fig.tight_layout()  

    plt.show()

        

new(df,df.columns,6,3)

        


for i in df.columns:

     print ('Unique values in ', i , 'is: ' , df[i].unique())
df['BPMeds'] = df['BPMeds'].map({1.:1,0.:0}) 

df['BPMeds'].unique()
X = df.drop('TenYearCHD',axis = 1)

y = df['TenYearCHD']
from sklearn.model_selection import train_test_split



X_train,X_test,y_train, y_test = train_test_split (X,y, train_size=0.7 , test_size =0.3 , random_state = 100)

print (X_train.shape)

#y_train = y_train.values.reshape(-1,1)

print (y_train.shape)

print (X_test.shape)

#y_test = y_test.values.reshape(-1,1)

print (y_test.shape)
from sklearn.preprocessing import StandardScaler
import numpy as np



from sklearn.base import BaseEstimator

from sklearn.base import TransformerMixin





class MyScaler(TransformerMixin, BaseEstimator):



    def fit(self, X, y=None):

        self.means_ = X.mean(axis=0)

        self.std_dev_ = X.std(axis=0)

        return self



    def transform(self, X, y=None):

        return (X - self.means_[:X.shape[1]]) / self.std_dev_[:X.shape[1]]

varlist = ['age','education','cigsPerDay',

          'totChol','sysBP','diaBP','BMI','heartRate',

          'glucose']

X_train.head()
scaler = StandardScaler()



X_train[varlist] = scaler.fit_transform(X_train[varlist])

X_test[varlist] = scaler.transform(X_test[varlist])
#print (X_train.head())

X_test.head()
plt.figure(figsize = (16,10))

ax = sns.heatmap(X_train.corr(),annot = True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

import statsmodels.api as sm
X_train_lr = sm.add_constant(X_train)
log1 = sm.GLM(y_train, X_train_lr, family= sm.families.Binomial())

log1 = log1.fit()

log1.summary()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE

rfe = RFE (logreg, 10)

rfe = rfe.fit(X_train, y_train)
print (list(zip(X_train.columns, rfe.support_, rfe.ranking_)))
col = X_train.columns[rfe.support_]

print (col)
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

logm2 = logm2.fit()

logm2.summary()
y_train_pred = logm2.predict(X_train_sm)

y_train_pred
y_train_pred_final = pd.DataFrame({'TenYearCHD':y_train,'Prob':y_train_pred})

y_train_pred_final.head()

y_train_pred_final['predicted'] = y_train_pred_final['Prob'].map(lambda x: 1 if x>0.2 else 0)

y_train_pred_final.head()

from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final['TenYearCHD'], y_train_pred_final['predicted'] )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final['TenYearCHD'], y_train_pred_final['predicted']))

def back_feature_elem (df,dependent_var,col_list):



    while len(col_list)>0 :

        df1 = sm.add_constant(df[col_list])

        model=sm.GLM(dependent_var,df1,family=sm.families.Binomial())

        result=model.fit()

        largest_pvalue=round(result.pvalues,3).nlargest(1)

        if largest_pvalue[0]<(0.05):

            return result

            break

        else:

            col_list=col_list.drop(largest_pvalue.index)



result=back_feature_elem(X_train,y_train,col)

result.summary()


X_train_sm = sm.add_constant(X_train[['male','age','cigsPerDay','prevalentHyp','diabetes']])

model=sm.GLM(y_train,X_train_sm,family=sm.families.Binomial())

res=model.fit()

res.summary()
y_train_pred = res.predict(X_train_sm)

y_train_pred[:5]


y_train_pred_final = pd.DataFrame({'TenYearCHD':y_train.values,'prob':y_train_pred })

y_train_pred_final.head()
y_train_pred_final['prediction'] = y_train_pred_final['prob'].map(lambda x: 1 if x>0.2 else 0)

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final['TenYearCHD'], y_train_pred_final['prediction'] )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final['TenYearCHD'], y_train_pred_final['prediction']))

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_sm[['male','age','cigsPerDay','prevalentHyp','diabetes']].columns

vif['VIF'] = [variance_inflation_factor(X_train_sm[['male','age','cigsPerDay','prevalentHyp','diabetes']].values, i) for i in range(X_train_sm[['male','age','cigsPerDay','prevalentHyp','diabetes']].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
TP = int(confusion[1,1]) # true positive 

TN = int(confusion[0,0]) # true negatives

FP = int(confusion[0,1]) # false positives

FN = int(confusion[1,0]) # false negatives

from sklearn.metrics import classification_report



classification_report(y_train_pred_final['TenYearCHD'], y_train_pred_final['prediction'])
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_pred_final['TenYearCHD'], y_train_pred_final['prob'])

plt.plot(fpr,tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for Heart disease classifier')

plt.xlabel('False positive rate (1-Specificity)')

plt.ylabel('True positive rate (Sensitivity)')

plt.grid(True)

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    y_train_pred_final[i] = y_train_pred_final['prob'].map(lambda x: 1 if x > i else 0)

    

y_train_pred_final.head()
cutoff_df = pd.DataFrame(columns = ['prob', 'accuracy', 'sensi', 'speci'])



from sklearn.metrics import confusion_matrix



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    confusion = metrics.confusion_matrix(y_train_pred_final['TenYearCHD'], y_train_pred_final[i] )

    TP = int(confusion[1,1]) # true positive 

    TN = int(confusion[0,0]) # true negatives

    FP = int(confusion[0,1]) # false positives

    FN = int(confusion[1,0]) # false negatives



    total1=TP+FP+TN+FN

    accuracy = (TP+TN)/total1

    speci = TN/(TN+FP)

    sensi = TP/(TP+FN)

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print (cutoff_df)
ax = cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.grid()



plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final['prob'].map( lambda x: 1 if x > 0.165 else 0)



y_train_pred_final.head()
metrics.accuracy_score(y_train_pred_final['TenYearCHD'], y_train_pred_final.final_predicted)







X_test_1 = X_test[['male','age','cigsPerDay','prevalentHyp','diabetes']]

X_test_1.head()

X_test_sm = sm.add_constant(X_test_1)

logtest = sm.GLM(y_test, X_test_sm, family= sm.families.Binomial())

logtest = logtest.fit()

logtest.summary()
y_test_pred = logtest.predict(X_test_sm)

y_test_pred[:5]
y_test_pred_final = pd.DataFrame({'TenYearCHD':y_test,'prob':y_test_pred })

y_test_pred_final.head()

y_test_pred_final['prediction'] = y_test_pred_final['prob'].map(lambda x: 1 if x > 0.165 else 0)

y_test_pred_final.head()                   

                  
metrics.accuracy_score(y_test_pred_final.TenYearCHD, y_test_pred_final.prediction)


