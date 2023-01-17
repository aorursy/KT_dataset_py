import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns 

%matplotlib inline

import warnings 

warnings.filterwarnings('ignore')
import io

df= pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')

df.rename(columns={'Dataset':'Target'},inplace=True)

df.head()
df.info()
df.describe(exclude='O')
print(df.describe(include='O'))

print('% diff')

print(df.Gender.value_counts(normalize=True)*100)
df[df['Albumin_and_Globulin_Ratio'].isnull()]
# We can perform Knn Imputation 

from sklearn.impute import KNNImputer

imputer=KNNImputer()

df['Albumin_and_Globulin_Ratio']=imputer.fit_transform(df[['Albumin_and_Globulin_Ratio']])
for i in df.select_dtypes(exclude='O'):

    if df[i].skew() > 0.95:                 # Checking for skewness of each feature above 0.95 

        print(i,':',df[i].skew())  
df.hist(figsize=(15,10))                   #Transformation is required 

plt.tight_layout()
for i in df.describe(include='O').columns:

    for j in df.describe(exclude='O').columns:

        plt.subplots()

        print(sns.boxplot(x=i,y=j,hue='Target',data=df))
nlp,lp=df.Target.value_counts()  # for target variable

print(sns.countplot(df.Target))

print('The Number of patients diagnosed with liver disease:',lp)

print('The Number of patients diagnosed with no liver disease:',nlp)
sns.countplot(df['Gender']) # for Gender column
df.head()
df['Gender']=df['Gender'].replace({'Female':1,'Male':0})
df.head()
sns.pairplot(df)
df.corr()['Target']
plt.figure(figsize=(15,8))

sns.heatmap(df.corr(),annot=True,fmt= '.2f')
corr=df.corr()

cols=corr.nlargest(15,'Target').index

cm = np.corrcoef(df[cols].values.T)

plt.figure(figsize=(20,12))

sns.heatmap(cm,annot=True, yticklabels = cols.values, xticklabels = cols.values)
from scipy.stats import chi2_contingency,f_oneway
df['Target']=df['Target'].astype('O')           # done only for statistical analysis
df.info()
cat_cols=df.describe(include='O').columns

cat_cols
chi_stat=[]

p_value=[]

for i in cat_cols:

    chi_res=chi2_contingency(np.array(pd.crosstab(df[i],df['Target'])))

    chi_stat.append(chi_res[0])

    p_value.append(chi_res[1])

chi_square=pd.DataFrame([chi_stat,p_value])

chi_square=chi_square.T

col=['Chi Square Value','P-Value']

chi_square.columns=col

chi_square.index=cat_cols
chi_square
features_p = list(chi_square[chi_square["P-Value"]<0.05].index)

print("Significant categorical Features:\n\n",features_p)
num_cols=df.describe(exclude='O')

num_cols.columns
f_stat=[]

p_val=[]

for i in num_cols:

    liver_0=df[df['Target']==1][i]

    liver_1=df[df['Target']==2][i]

    a=f_oneway(liver_0,liver_1)

    f_stat.append(a[0])

    p_val.append(a[1])

anova=pd.DataFrame([f_stat,p_val])

anova=anova.T

cols=['F-STAT','P-VALUE']

anova.columns=cols

anova.index=num_cols.columns
anova
anova[anova["P-VALUE"]<0.05]
features_p_n = list(anova[anova["P-VALUE"]<0.05].index)

print("Significant numerical Features:\n\n",features_p_n)
from sklearn.preprocessing import StandardScaler   #Scaling the data here improved the overall test accuracy

ss=StandardScaler()

cols=list(df.columns)

cols.remove('Target')

for col in cols:

    df[[col]]=ss.fit_transform(df[[col]])

df['Target']=pd.to_numeric(df['Target'],downcast='integer')
df.head()
plt.rcParams['figure.figsize']=[5,10]

df.corr()['Target'].sort_values().plot(kind='barh')
X=df.drop('Target',axis=1)

y=df['Target']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB,BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score,KFold





lr = LogisticRegression(fit_intercept=True,random_state=1)

gnb= GaussianNB()

bnb= BernoulliNB()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier(random_state=1)

rfc= RandomForestClassifier(random_state=1)

svm= SVC(random_state=1)

lda=LinearDiscriminantAnalysis()
models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('SVM', SVC()))
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

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

fig.set_size_inches(8,6)

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

    kfold = KFold(n_splits=10, random_state=1)

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
model_eval(lr,X_train,X_test,y_train,y_test)  # scaling the data has helped reduce the variance error 
coeff_df = pd.DataFrame(X.columns)

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(lr.coef_[0])

pd.Series(lr.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
from sklearn.metrics import cohen_kappa_score

y_test_pred=lr.predict(X_test)

cohen_kappa_score(y_test,y_test_pred)
df1=df.copy(deep=True) # Analysing df1 in the end of the notebook
df1.head()
for i in df1.select_dtypes(exclude='O'):

    if df1[i].skew() > 0.95:                 # Checking for skewness of each feature above 0.95 

        print(i,':',df1[i].skew())  
for i in df1.select_dtypes(exclude='O'): 

    print(i,':',df1[i].skew())                 # Checking for skewness of each feature above 0.95              
# Performing Log Transformation on the highly skewed features 

for i in df1.describe(exclude='O').columns:

    if df1[i].skew()>0.95:

        df1[i]=np.log1p(df1[i])
df1.hist(figsize=(15,10))

plt.tight_layout()
df1.head()
# We can perform Knn Imputation 

from sklearn.impute import KNNImputer

imputer=KNNImputer()

df1['Albumin_and_Globulin_Ratio']=imputer.fit_transform(df1[['Albumin_and_Globulin_Ratio']])
from scipy.stats import zscore

X=df1.drop('Target',axis=1)                     # scaling increased overall accuracy 

y=df1['Target']
df1['Target'].value_counts() # slightly imbalanced therefore using stratify=y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB,BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score,KFold





lr1 = LogisticRegression(fit_intercept=True,random_state=1)

gnb= GaussianNB()

bnb= BernoulliNB()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier(random_state=1)

rfc= RandomForestClassifier(random_state=1)

svm= SVC(random_state=1)

lda=LinearDiscriminantAnalysis()
models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('SVM', SVC()))
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

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

fig.set_size_inches(8,6)

plt.show()
model_eval(lr1,X_train,X_test,y_train,y_test)
df2=df.copy(deep=True)

df2.head()
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
for i in df2.describe(exclude='O').columns:

    if df2[i].skew()> 0.95:

        df2[i]=pt.fit_transform(df2[[i]])
for i in df2.describe(exclude='O').columns:

    print(i,df2[i].skew())
X=df2.drop('Target',axis=1)

y=df2['Target']
df2['Target'].value_counts() # slightly imbalanced therefore using stratify=y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB,BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score,KFold





lr2 = LogisticRegression(fit_intercept=True,random_state=1)

gnb= GaussianNB()

bnb= BernoulliNB()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier(random_state=1)

rfc= RandomForestClassifier(random_state=1)

svm= SVC(random_state=1)

lda=LinearDiscriminantAnalysis()
models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('SVM', SVC()))
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
df3=df.copy(deep=True)

df3.head()
X=df3.drop('Target',axis=1)

y=df3['Target']
import statsmodels.api as sm

X_1=sm.add_constant(X)

model=sm.OLS(y,X_1).fit()

model.pvalues 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123,stratify=y)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
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
df3=df3[['Age', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Total_Protiens', 'Albumin','Target']]

df3.head()
X=df3.drop('Target',axis=1)

y=df3['Target']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB,BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score,KFold





lr3 = LogisticRegression(fit_intercept=True,random_state=1)

gnb= GaussianNB()

bnb= BernoulliNB()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier(random_state=1)

rfc= RandomForestClassifier(random_state=1)

svm= SVC(probability=True, random_state=1)

lda=LinearDiscriminantAnalysis()
models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('SVM', SVC()))
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

df4.head()
df4['Taret']=df4['Target'].transform(lambda x:np.log1p(x))
df4=df4[['Age', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Total_Protiens', 'Albumin','Target']]

df4.head()
for i in df4.columns:

    print(i,df4[i].skew())
X=df4.drop('Target',axis=1)

y=df4['Target']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB,BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score,KFold





lr4 = LogisticRegression(fit_intercept=True,random_state=1)

gnb= GaussianNB()

bnb= BernoulliNB()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier(random_state=1)

rfc= RandomForestClassifier(random_state=1)

svm= SVC(probability=True, random_state=1)

lda=LinearDiscriminantAnalysis()
models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('SVM', SVC()))
results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=1)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
model_eval(lr4,X_train,X_test,y_train,y_test)