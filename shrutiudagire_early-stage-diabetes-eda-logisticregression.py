import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# from google.colab import drive
# drive.mount('/content/drive')
# df=pd.read_csv("/content/drive/My Drive/Supervised_Categorical/Practice/EarlyStageDiab/diabetes_data_upload.csv")
df=pd.read_csv("../input/early-stage-diabetes-risk-prediction-datasets/diabetes_data_upload.csv")
df
sns.countplot(df['class'])
df['class'].value_counts()
pd.crosstab(df['class'],df['Age'])
neg=df[df['class']=='Negative']
pos=df[df['class']=='Positive']
plt.figure(figsize=(20,5))
neg['Age'].plot(kind='kde')
pos['Age'].plot(kind='kde',color='red')
plt.xticks(np.arange(15,94,2))
plt.figure(figsize=(20,5))
sns.countplot(df['Age'],hue=df['class'],order=pos['Age'].value_counts().sort_values(ascending=False).index)
df['Gender'].value_counts()
pd.crosstab(df['class'],df['Gender'])
# Total females=192 out of which 173 are diabetic.

# In Above dataset 90% of feamles have diabetes
df['Polyuria'].value_counts()
pd.crosstab(df['class'],df['Polyuria'])
pd.crosstab(df['class'],df['Polyuria'])/df['Polyuria'].value_counts()*100
sns.countplot(df['Polyuria'],hue=df['class'])
df['Polydipsia'].value_counts()
pd.crosstab(df['class'],df['Polydipsia'])
pd.crosstab(df['class'],df['Polydipsia'])/df['Polydipsia'].value_counts()*100
sns.countplot(df['Polydipsia'],hue=df['class'])
df['sudden weight loss'].value_counts()
pd.crosstab(df['class'],df['sudden weight loss'])
pd.crosstab(df['class'],df['sudden weight loss'])/df['sudden weight loss'].value_counts()*100
sns.countplot(df['sudden weight loss'],hue=df['class'])
# People who have sudden weight loss and are diabetic is 86%.

# People who dont have sudden weight loss are are diabetic is 56% and People who dont have sudden weight loss are not diabetic is 43%,which we can say is almost 50-50% in both case.

# People who have sudden weight loss have a higher tendency to be diabetic
pd.crosstab(df['class'],df['weakness'])/df['weakness'].value_counts()*100
sns.countplot(df['weakness'],hue=df['class'])
# 71% of patient who have weakness are diabetic.

#patient who have weakness have higher tendency to be diabetic
pd.crosstab(df['class'],df['Polyphagia'])/df['Polyphagia'].value_counts()*100
sns.countplot(df['Polyphagia'],hue=df['class'])
# 79% of patient who have Polyphagia are diabetic.

#patient who have Polyphagia have higher tendency to be diabetic
pd.crosstab(df['class'],df['Genital thrush'])/df['Genital thrush'].value_counts()*100
df['Genital thrush'].value_counts()
sns.countplot(df['Genital thrush'],hue=df['class'])
# patient  who have Gential thrush have high tendency of diabetics.

# People who dnt have Gential thrush have 50% tendency of being Diabetic.
pd.crosstab(df['class'],df['visual blurring'])/df['visual blurring'].value_counts()*100
sns.countplot(df['visual blurring'],hue=df['class'])
# People who dnt have visual blurring have have 50% chance of being diabetic.


# 75% of people have visual blurring  and are diabetic.

# People who have visual blurring have higher tendency of being diabetic.
pd.crosstab(df['class'],df['Itching'])/df['Itching'].value_counts()*100
sns.countplot(df['Itching'],hue=df['class'])
# we can see 62% of people who dont have itching are diabetic whereas 60% of people who have itching are diabetic.

# Have 'itching' feature is not providing significant insights.
pd.crosstab(df['class'],df['Irritability'])/df['Irritability'].value_counts()*100
df['Irritability'].value_counts()
sns.countplot(df['Irritability'],hue=df['class'])
# 87% of people having irritability issue have diabetics.

# WE have 394 who dont have irritability issue out of which 53% are diabetic whereas as 43% are not diabetic.


pd.crosstab(df['class'],df['delayed healing'])/df['delayed healing'].value_counts()*100
df['delayed healing'].value_counts()
sns.countplot(df['delayed healing'],hue=df['class'])
# # Have 'delayed healing' feature is not providing significant insights.
df['partial paresis'].value_counts()
pd.crosstab(df['class'],df['partial paresis'])/df['partial paresis'].value_counts()*100
sns.countplot(df['partial paresis'],hue=df['class'])
#85% of people are having diabetes and are partial paresis

df['muscle stiffness'].value_counts()
pd.crosstab(df['class'],df['muscle stiffness'])/df['muscle stiffness'].value_counts()*100
sns.countplot(df['muscle stiffness'],hue=df['class'])
#  Have 'muscle stiffness' feature is not providing significant insights.
df['Alopecia'].value_counts()
pd.crosstab(df['class'],df['Alopecia'])/df['Alopecia'].value_counts()*100
sns.countplot(df['Alopecia'],hue=df['class'])
#  'Alopecia' feature is not providing significant insights.
df['Obesity'].value_counts()
pd.crosstab(df['class'],df['Obesity'])/df['Obesity'].value_counts()*100
sns.countplot(df['Obesity'],hue=df['class'])
# 'Obesity' feature is not providing significant insights.
cat=['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity']
df['Gender'].value_counts()
pd.crosstab(df['Polyuria'],df['Gender'])/df['Gender'].value_counts()*100
pd.crosstab(df['Polydipsia'],df['Gender'])/df['Gender'].value_counts()*100
pd.crosstab(df['Polyphagia'],df['Gender'])/df['Gender'].value_counts()*100
pd.crosstab(df['visual blurring'],df['Gender'])/df['Gender'].value_counts()*100
pd.crosstab(df['partial paresis'],df['Gender'])/df['Gender'].value_counts()*100
pd.crosstab(df['Alopecia'],df['Gender'])/df['Gender'].value_counts()*100
pd.crosstab(df['Obesity'],df['Gender'])/df['Gender'].value_counts()*100
# Percentage of female having polydipsia, polyuria,sudden weight loss,Polyphagia,visual blurring,partial paresis
# is very high as compared to male


# Percentage of Alopecia is more in male
pd.crosstab(df['Polyuria'],df['Polydipsia'])/df['Polyuria'].value_counts()*100
# 84% of people dnt have Polydipsia and Polyuria whereas 74% of people have Polydipsia and Polyuria.

# There is a high tendency that a peroson who doent have Polyuria will not have Polydipsia
pd.crosstab(df['Polyuria'],df['weakness'])/df['Polyuria'].value_counts()*100
# 71% of people have polyuria and have weakness.

# There is a high tendency that people who have polyuria will have weakness

pd.crosstab(df['Polyuria'],df['Polyphagia'])/df['Polyuria'].value_counts()*100
# There are 72% of people who dont have Polyuria and Polyphagia. 
# There is a high tendency that a person doesnt have Polyuria will not have polyphagia
pd.crosstab(df['Polyuria'],df['partial paresis'])/df['Polyuria'].value_counts()*100
# There are 78% of people who dont have Polyuria and partial paresis. 
# There is a high tendency that a person doesnt have Polyuria will not have partial paresis
pd.crosstab(df['Polydipsia'],df['partial paresis'])/df['Polydipsia'].value_counts()*100
#People who dont have Polydipsia have high tendency that they dont have partial paresis
pd.crosstab(df['sudden weight loss'],df['weakness'])/df['sudden weight loss'].value_counts()*100
#sudden weight loss can cause weakness-75% have people have sudden weight loss and weakness
pd.crosstab(df['Polyphagia'],df['partial paresis'])/df['Polyphagia'].value_counts()*100
# 73% of people are dont have Polyphagia and partial paresis
for i in cat:
    df[i]=df[i].replace({'No':0,'Yes':1})
df['class']=df['class'].replace({'Negative':0,'Positive':1})
df['Gender']=df['Gender'].replace({'Male':0,'Female':1})
plt.figure(figsize=(20,10)) 
sns.heatmap(df.corr(),annot=True)
from sklearn.preprocessing import MinMaxScaler
X=df.drop('class',axis=1)
y=df['class']
minmax=MinMaxScaler()
df1=minmax.fit_transform(X)
dfscaled=pd.DataFrame(df1,columns=X.columns)
dfscaled.head()
import statsmodels.api as sm
X_constant=sm.add_constant(dfscaled)
model_base=sm.Logit(y,X_constant,random_state=3).fit()
model_base.summary()
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
lr=LogisticRegression()
kfold = model_selection.KFold(n_splits = 3,random_state = 3,shuffle = True)
cv_results = cross_val_score(lr,dfscaled,y,cv = kfold,scoring='roc_auc')
print("Logistic Regression base model",' : ',np.mean(cv_results),' -- ',np.var(cv_results,ddof = 1))
cols=list(dfscaled.columns)
while(len(cols)>0):
    X_1=dfscaled[cols]
    X_constant=sm.add_constant(X_1)
    model=sm.Logit(y,X_constant,random_state=3).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
cols
X_final=dfscaled[cols]
X_constant=sm.add_constant(X_final)
model_base=sm.Logit(y,X_constant,random_state=3).fit()
model_base.summary()
lr=LogisticRegression()
kfold = model_selection.KFold(n_splits = 3,random_state = 3,shuffle = True)
cv_results = cross_val_score(lr,X_final,y,cv = kfold,scoring='roc_auc')
print("Logistic Regression base model",' : ',np.mean(cv_results),' -- ',np.var(cv_results,ddof = 1))
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

DT=DecisionTreeClassifier()
params={'criterion':['entropy','gini'],
    'max_depth':np.arange(5,250),
       'min_samples_leaf':np.arange(5,150),
       'min_samples_split':np.arange(5,150)
       }
gsearch=RandomizedSearchCV(DT,param_distributions=params,cv=3,scoring='roc_auc',random_state=3)
gsearch.fit(X_final,y)
gsearch.best_params_
LR=LogisticRegression()
DT=DecisionTreeClassifier(**gsearch.best_params_,random_state=3)
models = []
models.append(('Logistic Regression',LR))
models.append(('Decision Tree',DT))

from sklearn.ensemble import BaggingClassifier
for name,model in models: 
    auc_var = []
    for val in np.arange(1,150):
        model_bag = BaggingClassifier(base_estimator = model,n_estimators = val,random_state = 3,n_jobs=-1)
        kfold = model_selection.KFold(n_splits = 3,random_state = 0,shuffle = True)
        results = cross_val_score(model_bag,X_final,y,cv = kfold,scoring='roc_auc')
        auc_var.append(np.var(results,ddof = 1))
    print(name,np.argmin(auc_var)+1)
LR=LogisticRegression()
FGDT=DecisionTreeClassifier()
DT=DecisionTreeClassifier(**gsearch.best_params_,random_state=3)
bag_LR=BaggingClassifier(base_estimator = LR,n_estimators = 7,random_state = 3)
bag_DT=BaggingClassifier(base_estimator = DT,n_estimators = 21,random_state = 3)
models = []
models.append(('Logistic Regression ',LR))
models.append(('Fully grown Decision Tree',FGDT))
models.append(('Decision Tree',DT))
models.append(('Bag Decision Tree',bag_DT))
models.append(('Bag LR',bag_LR))


results = []
names = []
means=[]
variance=[]
for name, model in models:
    kfold = model_selection.KFold(n_splits = 5,random_state = 3,shuffle = True)
    cv_results = cross_val_score(model,X_final,y,cv = kfold,scoring='roc_auc')
    results.append(cv_results)
    names.append(name)
    means.append(np.mean(cv_results))
  #bias.append(100-(np.mean(cv_results)))

    variance.append(np.var(cv_results,ddof = 1))
    print(name,' : ',np.mean(cv_results),' -- ',np.var(cv_results,ddof = 1))

dfresults=pd.DataFrame({"Names":names,"Roc Auc":means,"Variance":variance})
fig = plt.figure(figsize=(10,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
dfresults
