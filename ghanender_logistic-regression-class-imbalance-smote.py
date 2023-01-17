from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as ss
import itertools
df = pd.read_json(r"nnDataSet.json").T
df.head()
df.Party.replace({'I':'Indep'}, inplace=True)
df.head()
df['contrib'] = df.Contributions.apply(lambda x: int(x.replace('$','').replace(',','')))/1e6
df.head()
fig, ax = plt.subplots(figsize=(8, 8))
sns.boxplot(x='Vote', y='contrib', data=df, ax=ax); ax.set_ylabel("Contribution ($, Millions)");
ax.set_title("Contributions by Expected Vote");

res = pd.crosstab(df.Party, df.Vote)
res = res.div(res.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.heatmap(res, annot=True, ax=ax); ax.set_title("Voting Split by Party");
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
cols = ["Party", "Vote", "contrib"]
corrM = np.zeros((len(cols),len(cols)))
# there's probably a nice pandas way to do this
for col1, col2 in itertools.combinations(cols, 2):
    idx1, idx2 = cols.index(col1), cols.index(col2)
    corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
    corrM[idx2, idx1] = corrM[idx1, idx2]
corr = pd.DataFrame(corrM, index=cols, columns=cols)
fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
from sklearn.linear_model import LogisticRegression
df['Vote_i'] = df.Vote.replace({'Yes':1, 'No':0, 'Unknown':np.NaN})
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(df[['Party','Position','State']]).toarray())
enc_df.head()
df['Name']=df.index
df =df.reset_index(drop=True)
df.head()
final_df = df.join(enc_df)

final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.fillna(0)
final_df.head()
#final_df.drop(['Contributions','Party','Position','State','Vote','Vote_i','Name'],axis=1)&
X_train=final_df[(final_df['Vote']!='Unknown')].reset_index(drop=True)
y_train=final_df.Vote_i[(final_df['Vote']!='Unknown')].reset_index(drop=True)


X_train=X_train.drop(['Contributions','Party','Position','State','Vote','Vote_i','Name'],axis=1)
X_train.head()
X_test=final_df[(final_df['Vote']=='Unknown')].reset_index(drop=True)
X_test=X_test.drop(['Contributions','Party','Position','State','Vote','Vote_i','Name'],axis=1)
#y_test=final_df.Vote_i[(final_df['Vote']=='Unknown')].reset_index(drop=True)
X_test.head()
X_train=X_train.fillna(0)
y_train=y_train.fillna(0)
from sklearn.linear_model import LogisticRegression
#create an instance and fit the model 
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
y_test = logmodel.predict(X_test)
y_test
from sklearn.model_selection import train_test_split
df_new=final_df[(final_df['Vote']!='Unknown')].reset_index(drop=True)
X=df_new.drop(['Contributions','Party','Position','State','Vote','Vote_i','Name'],axis=1)
y=df_new['Vote_i']
# Standard Scaling
# Feature Extraction - PCA [Dimensionality Reduction]
# Label encoder & One Hot Encoding
# imballance classes [ are the number of 0 & 1 in proportion]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30, shuffle=False)
len(X_train)
X_train.head()
y_train.groupby(y_train).count()
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2)
X_train1,y_train1 = sm.fit_sample(X_train,y_train)
len(y_train)
len(y_train1)
from imblearn.under_sampling import NearMiss 
nr = NearMiss()
X_train2,y_train2 = nr.fit_sample(X_train,y_train)
len(X_train2)
from sklearn.metrics import confusion_matrix, classification_report
lr = LogisticRegression()
lr.fit(X_train, y_train) 
predictions = lr.predict(X_test) 
print(classification_report(y_test, predictions))

data = pd.read_csv(r"creditcard.csv") 
data.head()
from sklearn.preprocessing import StandardScaler 
data['normAmount'] = StandardScaler().fit_transform(np.array(data['Amount']).reshape(-1, 1))
X = data.drop(['Time','Amount','Class'], axis = 1) 
y=data['Class']
data['Class'].value_counts() 
from sklearn.model_selection import train_test_split 
  
# split into 70:30 ration 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 
  
# describes info about train and test set 
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape) 
# logistic regression object 
lr = LogisticRegression() 
  
# train the model on train set 
lr.fit(X_train, y_train.ravel()) 
  
predictions = lr.predict(X_test) 
  
# print classification report 
print(classification_report(y_test, predictions)) 
y_test.value_counts()
from sklearn.metrics import confusion_matrix
fig,ax= plt.subplots(figsize=(12, 4))
labels = ['Abnormal','Normal']
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, ax = ax,fmt='.2f'); #annot=True to annotate cells
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 
  
# import SMOTE module from imblearn library 
# pip install imblearn (if you don't have imblearn in your system) 
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 
  
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 
lr1 = LogisticRegression() 
lr1.fit(X_train_res, y_train_res.ravel()) 
predictions = lr1.predict(X_test) 
  
# print classification report 
print(classification_report(y_test, predictions)) 
y_test.value_counts()
from sklearn.metrics import confusion_matrix
fig,ax= plt.subplots(figsize=(12, 4))
labels = ['Abnormal','Normal']
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, ax = ax,fmt='.2f'); #annot=True to annotate cells

print("Before Undersampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before Undersampling, counts of label '0': {} \n".format(sum(y_train == 0))) 
  
# apply near miss 
from imblearn.under_sampling import NearMiss 
nr = NearMiss() 
  
X_train_miss, y_train_miss = nr.fit_sample(X_train, y_train.ravel()) 
  
print('After Undersampling, the shape of train_X: {}'.format(X_train_miss.shape)) 
print('After Undersampling, the shape of train_y: {} \n'.format(y_train_miss.shape)) 
  
print("After Undersampling, counts of label '1': {}".format(sum(y_train_miss == 1))) 
print("After Undersampling, counts of label '0': {}".format(sum(y_train_miss == 0))) 

# train the model on train set 
lr2 = LogisticRegression() 
lr2.fit(X_train_miss, y_train_miss.ravel()) 
predictions = lr2.predict(X_test) 
  
# print classification report 
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
fig,ax= plt.subplots(figsize=(12, 4))
labels = ['Abnormal','Normal']
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, ax = ax,fmt='.2f'); #annot=True to annotate cells







from sklearn.decomposition import PCA
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(X)
X_pca = pd.DataFrame(data = principalComponents
             , columns = ['pc1','pc2','pc3','pc4']) #,'pc5','pc6','pc7','pc8','pc9','pc10'])
from sklearn.model_selection import train_test_split 
  
# split into 70:30 ration 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.3, random_state = 0) 
  
# describes info about train and test set 
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape) 

# logistic regression object 
lr = LogisticRegression() 
  
# train the model on train set 
lr.fit(X_train, y_train.ravel()) 
  
predictions = lr.predict(X_test) 
  
# print classification report 
print(classification_report(y_test, predictions)) 
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 
  
# import SMOTE module from imblearn library 
# pip install imblearn (if you don't have imblearn in your system) 
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 
  
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 
lr1 = LogisticRegression() 
lr1.fit(X_train_res, y_train_res.ravel()) 
predictions = lr1.predict(X_test) 
  
# print classification report 
print(classification_report(y_test, predictions)) 
