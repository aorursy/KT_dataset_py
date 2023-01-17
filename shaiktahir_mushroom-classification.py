import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)
df=pd.read_csv('../input/mushroom-classification/mushrooms.csv')

df.head()
print('The Mushroom data set has rows :{} and columns :{}'.format(df.shape[0],df.shape[1]))
df.info()
df.describe()  # 5 point summary is not avaialble as all features here are of object type
pd.DataFrame({'Count':df.isnull().sum(),'% Missing':df.isnull().sum()/df.shape[0]})*100 # Checking for null values
for i in df.select_dtypes(include='O'):        

    print(i,'unique values are:',df[i].nunique(),'--- they are:',df[i].unique(),'---and the % of observation in each feaure is: \n',df[i].value_counts(normalize=True)*100)

    print('*'*50)
df.replace('?',np.nan,inplace=True)  # replacing as Nan value
pd.DataFrame({'Count':df.isnull().sum(),'% Missing':df.isnull().sum()/df.shape[0]*100}) # Checking for null values
df['stalk-root']=df['stalk-root'].fillna(df['stalk-root'].value_counts().index[0])
# Checking for outliers in the dataset 



for i in df.describe(include='O').columns:

    plt.subplots()

    plt.title(i)

    print(sns.boxplot(x=df[i].value_counts(),hue='class',data=df)) # have to use value counts 

for i in df.columns:

    plt.subplots()            # using value counts to count each observation in each feature

    sns.countplot(df[i],hue=df['class'],palette='hot')
for i in df.columns:        # veil-type has only one value p. we can eliminate it.

    if df[i].nunique()==1:

        print(i)
df = df.drop('veil-type', axis=1)  
df1=pd.get_dummies(data=df,drop_first=True)

df1.head()
plt.figure(figsize=(25,10))

sns.heatmap(df1.corr(),annot=True,fmt='0.2f')

plt.show()
corr=df1.corr()                              # Top 15 features that high high correlation with class 

cols=corr.nlargest(15,'class_p').index

cm = np.corrcoef(df1[cols].values.T)

plt.figure(figsize=(20,12))

sns.heatmap(cm,annot=True, yticklabels = cols.values, xticklabels = cols.values)

plt.show()
from scipy.stats import chi2_contingency,f_oneway
cat_cols=df.describe(include='O').columns   # Used df here as all features are categorical



chi_stat=[]

p_value=[]

for i in cat_cols:

    chi_res=chi2_contingency(np.array(pd.crosstab(df[i],df['class'])))

    chi_stat.append(chi_res[0])

    p_value.append(chi_res[1])

chi_square=pd.DataFrame([chi_stat,p_value])

chi_square=chi_square.T

col=['Chi Square Value','P-Value']

chi_square.columns=col

chi_square.index=cat_cols
chi_square  # Need to interpret the Pvalue correctly here 
features_p = list(chi_square[chi_square["P-Value"]==0.00].index)  # Selected 13 important features based on chi2 test 

print("Significant categorical Features:\n\n",features_p)
num_cols=df1.describe(exclude='O')    # After Typecasting the data using pd.get_dummies and then using ANOVA

num_cols.columns



f_stat=[]

p_val=[]

for i in num_cols:

    edible=df1[df['class']=='e'][i]  # make sure to use df as target variable is categorical and only then can use ANOVA

    poison=df1[df['class']=='p'][i]

    a=f_oneway(edible,poison)

    f_stat.append(a[0])

    p_val.append(a[1])

anova=pd.DataFrame([f_stat,p_val])

anova=anova.T

cols=['F-STAT','P-VALUE']

anova.columns=cols

anova.index=num_cols.columns
anova
features_p_n = list(anova[anova["P-VALUE"]==0.00].index)  # Anova chose 4 significant features

print("Significant numerical Features:\n\n",features_p_n)
X=df1.drop('class_p',axis=1)

y=df1['class_p']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.metrics import confusion_matrix , accuracy_score , roc_auc_score , roc_curve,classification_report,precision_score,recall_score,f1_score

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score,KFold







lr = LogisticRegression(fit_intercept=True)

gnb= GaussianNB()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier(ccp_alpha=0.01) # to increase pruning and avoid overfitting

svm= SVC(probability=True)

lda=LinearDiscriminantAnalysis()
models = []

models.append(('LR', LogisticRegression(random_state=123)))

models.append(('GNB', GaussianNB()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DecisionTree', DecisionTreeClassifier(random_state=123)))

models.append(('SVM', SVC(probability=True,random_state=123)))

models.append(('LDA', LinearDiscriminantAnalysis()))
results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=1)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# compare algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison',fontsize=20)

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

fig.set_size_inches(20,8)

plt.show()
def model_eval(algo,X_train,X_test,y_train,y_test):

    algo.fit(X_train,y_train)



    y_train_pred=algo.predict(X_train)             # Finding the positives and negatives 

    y_train_prob=algo.predict_proba(X_train)[:,1]  # we are intersted only in the second column



    #overall accuracy for train model

    print('Confusion Matrix- Train:','\n',confusion_matrix(y_train,y_train_pred))

    print('Overall Accuracy-Train:',accuracy_score(y_train,y_train_pred))

    print('AUC-Train',roc_auc_score(y_train,y_train_prob))



    

    y_test_pred=algo.predict(X_test)

    y_test_prob=algo.predict_proba(X_test)[:,1]

    print('*'*50)

    

    #overall accuracy of test model

    print('Confusion matrix - Test :','\n',confusion_matrix(y_test,y_test_pred))

    print('Overall Accuracy - Test :',accuracy_score(y_test,y_test_pred))

    print('AUC - Test:',roc_auc_score(y_test,y_test_prob))



    print('*'*50)

    scores=cross_val_score(algo,X,y,cv=3,scoring='roc_auc')

    print('3 Fold Cross Validation Scores')

    print(scores)

    print('Bias Error:',100-scores.mean()*100)

    print('Variance Error:',scores.std()*100)

          

          

    print('*'*50)

    print('Classification Report for test model: \n', classification_report(y_test,y_test_pred))

          

    fpr,tpr,threshold=roc_curve(y_test,y_test_prob,pos_label=[2])

    plt.figure(figsize=(15,8))

    plt.plot(fpr,tpr)

    plt.plot(fpr,fpr,color='r')

    plt.xlabel('Fpr')

    plt.ylabel('Tpr')
model_eval(lr,X_train,X_test,y_train,y_test)  
lr.fit(X_train,y_train)

y_test_pred=lr.predict(X_test)
from sklearn.dummy import DummyClassifier

dummy=DummyClassifier(random_state=123)

dummy.fit(X_train,y_train)

print('The dummy classifier gives us a basic score',dummy.score(y_test,y_test_pred))
print('The Precision Score for Logistic Regression Model is:',precision_score(y_test,y_test_pred))

print('The Recall Score for Logistic Regression Model is:',recall_score(y_test,y_test_pred))

print('The F1 Score for Logistic Regression Model is:',f1_score(y_test,y_test_pred)) 
coeff_df = pd.DataFrame(X.columns) # Variables that are significant in prediction based on their coef vals

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(lr.coef_[0])

pd.Series(lr.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
from sklearn.metrics import cohen_kappa_score  # Very Good Score

cohen_kappa_score(y_test,y_test_pred)
df2=df1.copy(deep=True)
X=df2.drop('class_p',axis=1)

y=df2['class_p']
import statsmodels.api as sm

X_1=sm.add_constant(X)

model=sm.OLS(y,X_1).fit()

model.pvalues 
cols=list(X.columns)

pmax=1

while (len(cols)>0):            # Using Backwad Elimination 

    p=[]

    X_1=X[cols]

    X_1=sm.add_constant(X_1)

    model=sm.OLS(y,X_1).fit()

    p=pd.Series(model.pvalues.values[1:],index=cols)

    pmax=max(p)

    feature_with_p_max=p.idxmax()

    if(pmax > 0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE=cols

print(selected_features_BE)
df2=df2[['cap-shape_k',

 'odor_c',

 'odor_f',

 'odor_m',

 'odor_n',

 'odor_p',

 'odor_s',

 'odor_y',

 'gill-size_n',

 'gill-color_e',

 'gill-color_g',

 'gill-color_h',

 'gill-color_k',

 'gill-color_n',

 'gill-color_o',

 'gill-color_p',

 'gill-color_r',

 'gill-color_u',

 'gill-color_w',

 'gill-color_y',

 'stalk-shape_t',

 'stalk-root_c',

 'stalk-root_e',

 'stalk-root_r',

 'stalk-surface-above-ring_y',

 'stalk-surface-below-ring_y',

 'stalk-color-above-ring_c',

 'stalk-color-above-ring_o',

 'stalk-color-below-ring_o',

 'stalk-color-below-ring_w',

 'veil-color_w',

 'ring-number_o',

 'ring-number_t',

 'ring-type_f',

 'ring-type_l',

 'ring-type_n',

 'ring-type_p',

 'spore-print-color_h',

 'spore-print-color_r',

 'spore-print-color_w',

 'habitat_w',

 'class_p']]
X=df2.drop('class_p',axis=1)

y=df2['class_p']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.metrics import confusion_matrix , accuracy_score , roc_auc_score , roc_curve,classification_report

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score,KFold



lr2 = LogisticRegression(fit_intercept=True)

gnb= GaussianNB()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier(ccp_alpha=0.01) # to increase pruning and avoid overfitting

svm= SVC(probability=True)

lda=LinearDiscriminantAnalysis()
models = []

models.append(('LR2', LogisticRegression(random_state=123)))

models.append(('GNB', GaussianNB()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DecisionTree', DecisionTreeClassifier(random_state=123)))

models.append(('SVM', SVC(probability=True,random_state=123)))

models.append(('LDA', LinearDiscriminantAnalysis()))
results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=1)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
model_eval(lr2,X_train,X_test,y_train,y_test)
plt.figure(figsize=(15,8))

df2.corr()['class_p'].sort_values().plot(kind='barh')

plt.show()
corr=df2.corr()                   # Top 15 features that high high correlation with class after feature selection 

cols=corr.nlargest(15,'class_p').index

cm = np.corrcoef(df2[cols].values.T)

plt.figure(figsize=(20,12))

sns.heatmap(cm,annot=True, yticklabels = cols.values, xticklabels = cols.values)

plt.show()
df3=df2.copy(deep=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor



def calc_vif(X):



    # Calculating VIF

    vif = pd.DataFrame()

    vif["variables"] = X.columns

    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



    return(vif)
X = df3.iloc[:,:-1]

calc_vif(X).T
df3.drop(['odor_c','odor_f','odor_m','odor_n','spore-print-color_w','gill-size_n','gill-color_e','gill-color_g','gill-color_h','gill-color_k','gill-color_n','gill-color_o','gill-color_p','gill-color_r'],axis=1,inplace=True)
X = df3.iloc[:,:-1]

calc_vif(X).T
df3.drop(['stalk-color-above-ring_c','stalk-color-above-ring_o','ring-type_n'],inplace=True,axis=1)
X = df3.iloc[:,:-1]

calc_vif(X).T
df3.drop(['veil-color_w','stalk-color-below-ring_o'],inplace=True,axis=1)
X = df3.iloc[:,:-1]

calc_vif(X).T
X=X

y=y
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.metrics import confusion_matrix , accuracy_score , roc_auc_score , roc_curve,classification_report

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score,KFold



lr3 = LogisticRegression(fit_intercept=True)

gnb= GaussianNB()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier(ccp_alpha=0.01) # to increase pruning and avoid overfitting

svm= SVC(probability=True)

lda=LinearDiscriminantAnalysis()
models = []

models.append(('LR3', LogisticRegression(random_state=123)))

models.append(('GNB', GaussianNB()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DecisionTree', DecisionTreeClassifier(random_state=123)))

models.append(('SVM', SVC(probability=True,random_state=123)))

models.append(('LDA', LinearDiscriminantAnalysis()))
results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=1)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
model_eval(lr3,X_train,X_test,y_train,y_test)
df4=df2.copy(deep=True)



X=df4.drop('class_p',axis=1)

y=df4['class_p']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.decomposition import PCA # Scaling not required here 

pca=PCA(n_components=25)       # Applying PCA to Logistic Regression

X_train_pca=pca.fit_transform(X_train) 

X_test_pca=pca.transform(X_test)   
explained_variance=pca.explained_variance_ratio_

explained_variance
cov_matirx=np.cov(X.T)

eig_vals,eig_vectors=np.linalg.eig(cov_matirx)

eig_vals   
tot=sum(eig_vals)

var_exp=[(i/tot)*100 for i in sorted(eig_vals,reverse=True)]

cum_var_exp=np.cumsum(var_exp)

print('Cumulative variance Explained:',(cum_var_exp))
plt.figure(figsize=(15,4))

plt.bar(range(X.shape[1]),var_exp,alpha=0.5,align='center',label='Individual explained variance')

plt.step(range(X.shape[1]),cum_var_exp,where='mid',label='cummulative explained variance')

plt.ylabel("explained variance ratio")

plt.xlabel("principal components")

plt.legend(loc='best')

plt.tight_layout()

plt.show()
from sklearn.linear_model import LogisticRegression

lr4=LogisticRegression(fit_intercept=True)



model_eval(lr4,X_train_pca,X_test_pca,y_train,y_test)