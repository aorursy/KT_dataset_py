import numpy as np

import pandas as  pd

import matplotlib.pyplot as plt

import seaborn as sns

import  warnings 

warnings.filterwarnings('ignore')

%matplotlib inline
df=pd.read_csv('/kaggle/input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv')
df.head(10)
df.info()
num_cols=numerical_df = ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)',

                'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)']





cat_cols=['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis','STDs:cervical condylomatosis',

                  'STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',

                  'STDs:pelvic inflammatory disease', 'STDs:genital herpes','STDs:molluscum contagiosum', 'STDs:AIDS', 

                  'STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis','Dx:Cancer', 'Dx:CIN', 

                  'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']
#we are dropping the STDs: column because these are of not of much use for our analysis bcz of having a lot of missing values



df.drop(['STDs: Time since last diagnosis','STDs: Time since first diagnosis'],inplace=True,axis=1)
df.head()
df=df.replace('?',np.NaN)
# now we  fill missing values in numerical columns with mean of numerical data 

for feature in num_cols:

    print(feature,'',df[feature].astype(float).mean())

    feature_mean = round(df[feature].astype(float).mean(),1)

    df[feature] = df[feature].fillna(feature_mean)
for features in cat_cols:

    df[features]=df[features].astype(float).fillna(1.0)
for feature in cat_cols:

    sns.factorplot(feature,data=df,kind='count')
g = sns.PairGrid(df,

                 y_vars=['Hormonal Contraceptives'],

                 x_vars= cat_cols,

                 aspect=2.75, size=10.5)

g.map(sns.barplot, palette="pastel");
df['Number of sexual partners'] = round(df['Number of sexual partners'].astype(float))

df['First sexual intercourse'] = df['First sexual intercourse'].astype(float)

df['Num of pregnancies']=round(df['Num of pregnancies'].astype(float))

df['Smokes'] =df['Smokes'].astype(float)

df['Smokes (years)'] =df['Smokes (years)'].astype(float)

df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].astype(float)

df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].astype(float)

df['IUD (years)'] = df['IUD (years)'].astype(float)



print('minimum:',min(df['Hormonal Contraceptives (years)']))

print('maximum:',max(df['Hormonal Contraceptives (years)']))
plt.hist(df['Age'].mean(),bins=20)

plt.xlabel('Age')                                 # age estimation for the risk of cervical cancer 

plt.ylabel('count')

print('mean age of woman facing the cervical cancer ',df['Age'].mean())

for feature in cat_cols:

    fig=sns.FacetGrid(df,hue=feature)

    fig.map(sns.kdeplot,'Age',shade=True)

    max_age=df['Age'].max()

    fig.set(xlim=(0,max_age))

    fig.add_legend()
for feature in cat_cols:

    plt.figure(figsize=(10,8))

    sns.factorplot(x='Number of sexual partners',y='Age',hue=feature,data=df,kind='bar')
sns.distplot(df['First sexual intercourse'].astype(float))
sns.scatterplot(x='First sexual intercourse',hue='Dx:Cancer',y='Num of pregnancies',data=df)
df['Dx:Cancer'].value_counts()
A1=df.groupby(['Dx:Cancer'])
cancer_patients=A1.get_group(1)

cancer_free_patients=A1.get_group(0)
age_affected_cancer=cancer_patients['Age']

age_notaffected=cancer_free_patients['Age']
age_affected_cancer.describe()
age_notaffected.describe()
plt.hist(age_affected_cancer)
df.boxplot(column='Age',by='Dx:Cancer')
from scipy.stats import shapiro

print(shapiro(age_affected_cancer))

print(shapiro(age_notaffected))
from scipy.stats import levene
levene(age_affected_cancer,age_notaffected)
from scipy.stats import ttest_ind
ttest_ind(age_affected_cancer,age_notaffected)
df['Dx:Cancer'].value_counts()
s=df.groupby(['Dx:Cancer'])
s1=s.get_group(1)

s2=s.get_group(0)
sexual_intercourse=s1['First sexual intercourse']

no_sexual_intercourse=s2['First sexual intercourse']
sexual_intercourse.describe()
no_sexual_intercourse.describe()
print(shapiro(sexual_intercourse))   #H0:data follows normality   H1:data not following normality

shapiro(no_sexual_intercourse)
levene (sexual_intercourse,no_sexual_intercourse)  #H0:variance of (sexual_intercourse)having cancer=var(no_sexual_intercourse)having cancer

                                                    #H1:variance of (sexual_intercourse)having cancer!=var(no_sexual_intercourse)having cancer
ttest_ind(sexual_intercourse,no_sexual_intercourse)
from sklearn.model_selection import cross_val_score # Cross Validation Score

from sklearn.model_selection import GridSearchCV # Parameters of the Model

from sklearn.model_selection import RandomizedSearchCV # Tuning the Parameters

from sklearn.tree import DecisionTreeClassifier # Decision Tree Algo

from sklearn.ensemble import RandomForestClassifier # Random Forest Algo.

from sklearn.model_selection import train_test_split # helps in spliting the data in train and test set

from sklearn.metrics import accuracy_score # Calculating the Accuracy Score againts the Classes Predicted vs Actuals.

from sklearn.ensemble import BaggingClassifier
#defining my Xs and Ys

x=df.drop('Dx:Cancer',axis=1) #dropping the target

y=df['Dx:Cancer']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=32)
#creatig our first model called Decisiontree

tree=DecisionTreeClassifier()



#defining tree params for grid based search

tree_params={

    "criterion":["gini", "entropy"],

    "splitter":["best", "random"],

    "max_depth":[3,4,5,6],

    "max_features":["auto","sqrt","log2"],

    "random_state": [123]

}
# apply grid search algorithm



grid=GridSearchCV(tree,tree_params,cv=10)

grid

#lets fit into data so that it can giuve the best params



best_param_search=grid.fit(x_train,y_train)



best_param_search.best_params_
#creating our first model called decision tree after hypertuning

 

tree2=DecisionTreeClassifier(criterion='entropy',max_depth=5,max_features= 'auto',random_state= 123,splitter='best')



#Developiug the model

model_tree=tree.fit(x_train,y_train)

pred_tree=tree.predict(x_test)

accuracy_score(y_test,pred_tree)
Rf_model=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0) 
bag_model=BaggingClassifier(n_estimators=10,random_state=0)  #fully grown decision tree
bag_model2=BaggingClassifier(n_estimators=10,random_state=0,base_estimator=tree2)## Regularised decision tree
models=[]

models.append(('Decision tree',tree))

models.append(('Random Forest',Rf_model))

models.append(('Bagged_DT',bag_model))

models.append(('bagged_regularized',bag_model2))
models
from sklearn import model_selection

results = []

names = []



for name, model in models:

    kfold = model_selection.KFold(n_splits=5,random_state=2)

    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring='recall')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, np.mean(cv_results), np.var(cv_results,ddof=1))

    print(msg)
fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax=fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# #lets import adaboost and bagging classifier

# from sklearn.ensemble import BaggingClassifier

# bagg=BaggingClassifier()

# bagmodel=bagg.fit(x_train,y_train)

# #make prediction

# pred_bagged=bagg.predict(x_test)
# accuracy_score(y_test,pred_bagged)
# tree=DecisionTreeClassifier()

# cross_val_score(tree,x,y,cv=10).mean()