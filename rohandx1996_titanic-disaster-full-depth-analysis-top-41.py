# importing  necesary files & libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as ss
from statsmodels.formula.api import ols
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
%timeit 
%matplotlib inline
dftrain=pd.read_csv("../input/train.csv")
dftest=pd.read_csv("../input/test.csv")
test=dftest.copy()
dftrain.info()
dftrain.head()
dftrain.info()
# analysis of dtypes 
plt.figure(figsize=(5,5))
sns.set(font_scale=2)
sns.countplot(y=dftrain.dtypes ,data=dftrain)
plt.xlabel("count of each data type")
plt.ylabel("data types")
plt.show()
import missingno as msno
msno.bar(dftrain.sample(890))
msno.matrix(dftrain)
msno.heatmap(dftrain)
msno.dendrogram(dftrain)
df=dftrain.copy()
df.head()

male1=df.loc[(df.Survived==1) &(df.Sex=='male'),:].count()
female1=df.loc[(df.Survived==1) & (df.Sex=='female'),:].count()

print(male1)
print(female1)
sns.factorplot(x="Sex",col="Survived", data=df , kind="count",size=6, aspect=.7,palette=['crimson','lightblue'])
malecount=pd.value_counts((df.Sex == 'male') & (df.Survived==1))
femalecount=pd.value_counts((df.Sex=='female') & (df.Survived==1))
totalmale,totalfemale=pd.value_counts(df.Sex)
print("male survived {} , female survived {}".format(malecount/totalmale,femalecount/totalfemale))
plt.figure(figsize=(12,12))
sns.swarmplot(x="Sex",y="Age",hue='Pclass',data=df,size=10 ,palette=['pink','lightgreen','purple'])
plt.figure(figsize=(12,12))
sns.swarmplot(x="Sex",y="Age",hue='Survived',data=df,size=10)

sns.factorplot(x="Sex", hue = "Pclass" , col="Survived", data=df , kind="count",size=7, aspect=.7,palette=['crimson','orange','lightblue'])
pd.crosstab([df.Sex,df.Survived],df.Pclass, margins=True).style.background_gradient(cmap='autumn_r')
pd.crosstab([df.Survived,df.Pclass],df.Age,margins=True).style.background_gradient(cmap='autumn_r')
sns.factorplot(x="Survived",col="Embarked",data=df ,hue="Pclass", kind="count",size=8, aspect=.7,palette=['crimson','darkblue','purple'])
pd.crosstab([df.Survived],[df.Sex,df.Pclass,df.Embarked],margins=True).style.background_gradient(cmap='autumn_r')
sns.factorplot(x="Sex", y="Survived",col="Embarked",data=df ,hue="Pclass",kind="bar",size=7, aspect=.7)
context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
df['Sex_bool']=df.Sex.map(context1)
df["Embarked_bool"] = df.Embarked.map(context2)
plt.figure(figsize=(20,20))
correlation_map = df[['PassengerId', 'Survived', 'Pclass', 'Sex_bool', 'Age', 'SibSp',
       'Parch', 'Fare' , 'Embarked_bool']].corr()
sns.heatmap(correlation_map,vmax=.7, square=True,annot=True,fmt=".2f")
df.groupby("Pclass").Age.mean()
df.isnull().sum()

for x in [dftrain, dftest,df]:
    x['Age_bin']=np.nan
    for i in range(8,0,-1):
        x.loc[ x['Age'] <= i*10, 'Age_bin'] = i
df[['Age','Age_bin']].head(20)
plt.figure(figsize=(20,20))
sns.set(font_scale=1)
sns.factorplot('Age_bin','Survived', col='Pclass' , row = 'Sex',kind="bar", data=df)
df.describe()
for x in [dftrain, dftest , df]:
    x['Fare_bin']=np.nan
    for i in range(12,0,-1):
        x.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i
fig, axes = plt.subplots(2,1)
fig.set_size_inches(20, 18)
sns.kdeplot(df.Age_bin , shade=True, color="red" , ax= axes[0])
sns.kdeplot(df.Fare , shade=True, color="red" , ax= axes[1])
df.isnull().sum()


model= ols('Age~ Pclass + Survived + SibSp',data=df).fit()
print(model.summary())
dftrain.info()
dftest.info()
np.where(dftrain["Embarked"].isnull())[0]
sns.factorplot(x='Embarked',y='Fare', hue='Pclass', kind="box",order=['C', 'Q', 'S'],data=dftrain, size=7,aspect=2)

# ... and median fare
plt.axhline(y=80, color='r', ls='--')
dftrain.loc[[61,829],"Embarked"] = 'C'
dftrain.info()
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# plot original Age values
# NOTE: drop all null values, and convert to int
dftrain['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# get average, std, and number of NaN values
average_age = dftrain["Age"].mean()
std_age = dftrain["Age"].std()
count_nan_age = dftrain["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_age = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# fill NaN values in Age column with random values generated
age_slice = dftrain["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age

# plot imputed Age values
age_slice.astype(int).hist(bins=70, ax=axis2)
dftrain["Age"] = age_slice
dftrain.info()
dftrain=dftrain.drop('Age_bin',axis=1)
dftrain.info()
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# plot original Age values
# NOTE: drop all null values, and convert to int
dftest['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# get average, std, and number of NaN values
average_age = dftest["Age"].mean()
std_age = dftest["Age"].std()
count_nan_age = dftest["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_age = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# fill NaN values in Age column with random values generated
age_slice = dftest["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age

# plot imputed Age values
age_slice.astype(int).hist(bins=70, ax=axis2)
dftest["Age"] = age_slice
dftest.info()
dftest.info()
plt.figure(figsize=(20,20))
sns.factorplot(x='Fare',y='Cabin',data=dftrain,size=20)
family_df = dftrain.loc[:,["Parch", "SibSp", "Survived"]]

# Create a family size variable including the passenger themselves
family_df["Fsize"] = family_df.SibSp + family_df.Parch + 1

family_df.head()
plt.figure(figsize=(15,5))

# visualize the relationship between family size & survival
sns.countplot(x='Fsize', hue="Survived", data=family_df)
dftrain['Fsize']=family_df['Fsize']
dftrain.info()

family_df_t= dftest.loc[:,["Parch", "SibSp", "Survived"]]

# Create a family size variable including the passenger themselves
family_df_t["Fsize"] = family_df_t.SibSp + family_df_t.Parch + 1

family_df_t.head()
dftest['Fsize']=family_df_t['Fsize']
dftest.info()
#dftest=dftest.drop('Cabin',axis=1)
dftest.info()
np.where(dftest["Fare"].isnull())[0]
dftest.ix[[152]]
dftest.loc[[152],"Fare"] = 10
dftest.ix[[152]]
dftest.info()
dftrain.info()
family_df_tr= dftrain.loc[:,["Parch", "SibSp", "Survived"]]

# Create a family size variable including the passenger themselves
family_df_tr["Fsize"] = family_df_tr.SibSp + family_df_tr.Parch + 1

family_df_tr.head()
dftrain['Fsize']=family_df_tr['Fsize']

dftrain['Fsize'].dtype
dftrain.info()
dftest.info()
import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

#Initialize ChiSquare Class
cT = ChiSquare(dftrain)

#Feature Selection
testColumns = ['Embarked','Cabin','Pclass','Age','Name','Fare','Fare_bin','Fsize']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="Survived" )  
# Make a copy of the titanic data frame
dftrain['Title'] = dftrain['Name']

# Grab title from passenger names
dftrain["Title"].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)
rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
dftrain['Title'].replace(rare_titles, "Rare title", inplace=True)

# Also reassign mlle, ms, and mme accordingly
dftrain['Title'].replace(["Mlle","Ms", "Mme"], ["Miss", "Miss", "Mrs"], inplace=True)
dftrain.info()
cT = ChiSquare(dftrain)

#Feature Selection
testColumns = ['Embarked','Cabin','Pclass','Age','Name','Fare','Fare_bin','Fsize','Title','SibSp','Parch']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="Survived" )  
dftest.info()
dftrain.info()
dftest=dftest.drop(['Ticket','PassengerId'],axis=1)

dftest.info()
dftest['Title'] = dftest['Name']

# Grab title from passenger names
dftest["Title"].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)

dftest.info()
dftrain.info()
dftrain.head()
dftrain=dftrain.drop('Name',axis=1)
dftrain.head()
context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
dftrain['Sex_bool']=dftrain.Sex.map(context1)
dftrain["Embarked_bool"] = dftrain.Embarked.map(context2)
dftrain.head()
#dftrain=dftrain.drop(['Sex','Embarked'],axis=1)
context3= {"Mr":0 , "Mrs":1 , "Miss":2,'Master':3}
dftrain['Title']=dftrain.Title.map(context3)
dftrain.head()
dftrain=dftrain.drop(['PassengerId','Cabin','Ticket'],axis=1)
plt.figure(figsize=(14,4))
sns.boxplot(data=dftrain)
reserve=dftrain.copy()
reserve.shape
dftrain.head()
dftrain=dftrain.drop(['Embarked','Sex'],axis=1)
dftrain.head()
#dftrain=dftrain[np.abs(zscore(dftrain)<3).all(axis=1)]
dftest.head()
context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
dftest['Sex_bool']=dftest.Sex.map(context1)
dftest["Embarked_bool"] = dftest.Embarked.map(context2)
context3= {"Mr":0 , "Mrs":1 , "Miss":2,'Master':3}
dftest['Title']=dftest.Title.map(context3)
dftest.head()
dftest=dftest.drop(['Name','Sex','Embarked'],axis=1)
dftest.head()
for x in [dftrain, dftest,df]:
    x['Age_bin']=np.nan
    for i in range(8,0,-1):
        x.loc[ x['Age'] <= i*10, 'Age_bin'] = i
dftrain.head()
dftest.head()
#dftrain=dftrain.drop(['Fare_bin'],axis=1)
#dftest=dftest.drop(['Fare_bin'],axis=1)
for x in [dftrain, dftest,df]:
    x['Fare_bin']=np.nan
    for i in range(12,0,-1):
        x.loc[ x['Fare'] <= i*10, 'Fare_bin'] = i
dftrain.head()
dftest.head()
dftrain=dftrain.drop('Age',axis=1)
dftest=dftest.drop('Age',axis=1)
dftrain.head()
dftrain=dftrain.convert_objects(convert_numeric=True)

def change_type(df):
    float_list=list(df.select_dtypes(include=["float"]).columns)
    print(float_list)
    for col in float_list:
        df[col]=df[col].fillna(0).astype(np.int64)
        
    return df    
change_type(dftrain)    
dftrain.dtypes
#dftrain=dftrain.drop(['Fare'],axis=1)
#dftest=dftest.drop(['Fare','Cabin'],axis=1)
x=dftrain.iloc[:,1:].values
y=dftrain.iloc[:,0].values
print(dftrain.columns)
print(dftest.columns)

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=101)
dftest=dftest.convert_objects(convert_numeric=True)
change_type(dftest)    
dftest.dtypes
MLA = []
Z = [LinearSVC() , DecisionTreeClassifier() , LogisticRegression() , KNeighborsClassifier() , GaussianNB() ,
    RandomForestClassifier() , GradientBoostingClassifier()]
X = ["LinearSVC" , "DecisionTreeClassifier" , "LogisticRegression" , "KNeighborsClassifier" , "GaussianNB" ,
    "RandomForestClassifier" , "GradientBoostingClassifier"]

for i in range(0,len(Z)):
    model = Z[i]
    model.fit( X_train , y_train )
    pred = model.predict(X_test)
    MLA.append(accuracy_score(pred , y_test))
MLA
sns.kdeplot(MLA , shade=True, color="red")
d = { "Accuracy" : MLA , "Algorithm" : X }
dfm = pd.DataFrame(d)
dfm
sns.barplot(x="Accuracy", y="Algorithm", data=dfm)
# imporvsing the model first logistic Regression
params={'C':[1,100,0.01,0.1,1000],'penalty':['l2','l1']}
logreg=LogisticRegression()
gscv=GridSearchCV(logreg,param_grid=params,cv=10)
%timeit gscv.fit(x,y)
gscv.best_params_
logregscore=gscv.best_score_
print(logregscore)
gscv.predict(X_test)
gscv.score(X_test,y_test)
rfcv=RandomForestClassifier(n_estimators=500,max_depth=6)
rfcv.fit(X_train,y_train)
rfcv.predict(X_test)
rfcv.score(X_test,y_test)

gbcv=GradientBoostingClassifier(learning_rate=0.001,n_estimators=2000,max_depth=5)
gbcv.fit(X_train,y_train)
gbcv.predict(X_test)
gbcv.score(X_test,y_test)
param={'n_neighbors':[3,4,5,6,8,9,10],'metric':['euclidean','manhattan','chebyshev','minkowski'] }       
knn = KNeighborsClassifier()
gsknn=GridSearchCV(knn,param_grid=param,cv=10)
gsknn.fit(x,y)                         
                                                
gsknn.best_params_
gsknn.best_score_
gsknn.predict(X_test)
gsknn.score(X_test,y_test)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, gscv.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rfcv.predict_proba(X_test)[:,1])
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, gsknn.predict_proba(X_test)[:,1])
gbc_fpr, gbc_tpr, ada_thresholds = roc_curve(y_test, gbcv.predict_proba(X_test)[:,1])

plt.figure(figsize=(9,9))
log_roc_auc = roc_auc_score(y_test, gscv.predict(X_test))
print ("logreg model AUC = {} " .format(log_roc_auc))
rf_roc_auc = roc_auc_score(y_test, rfcv.predict(X_test))
print ("random forest model AUC ={}" .format(rf_roc_auc))
knn_roc_auc = roc_auc_score(y_test, gsknn.predict(X_test))
print ("KNN model AUC = {}" .format(knn_roc_auc))
gbc_roc_auc = roc_auc_score(y_test, gbcv.predict(X_test))
print ("GBC Boost model AUC = {}" .format(gbc_roc_auc))
# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression')

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest')

# Plot Decision Tree ROC
plt.plot(knn_fpr, knn_tpr, label=' KnnClassifier')

# Plot GradientBooseting Boost ROC
plt.plot(gbc_fpr, gbc_tpr, label='GradientBoostingclassifier')

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()

