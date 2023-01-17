import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
#read Train and test data
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

#concat train and Test data
all_Data = pd.concat([train.drop('Survived', axis=1), test], axis=0, sort=True)
group_all_data = all_Data.groupby(['Sex'])['PassengerId'].count()

print('how many people have by sex in all dataset', group_all_data )
print()
print('Total of: \n',all_Data.isnull().sum())

print('Sample of Data \n')
print(all_Data.head(5))
print()
#visualize the null in all dataset 
all_Data_total = all_Data.isnull().sum().sum()
print('Quantity persons on Training and Test Data:', all_Data_total)

print('% People by Sex: \n', group_all_data*100/all_Data_total)


#show all the columns
pd.options.display.max_columns = None

#print Train Dataset 
print(train.info())
print(train.describe())
print(train.head(5))
#from pandas.plotting import scatter_matrix
col_obj = ['Survived', 'Pclass', 'Age', 'SibSp','Parch', 'Fare']
#scatter_matrix(train[col_obj], figsize=(12,8))
sns.pairplot(train.drop('PassengerId', axis=1).dropna(), hue='Survived')

group_s = train.groupby(['Sex'])['PassengerId'].count()
print('train data has:', group_s.sum() , ' rows \n' )
print("Q by sex (train)\n " , group_s)
print("\n % train data (train)  \n",group_s*100/group_s.sum())
group_svs = train.groupby(['Survived'])['PassengerId'].count()*100/train.groupby(['Survived'])['PassengerId'].count().sum()
print('% people Die/survived \n', group_svs.round(2))
print()
group_svs.plot(kind='bar', title='% of survived')
plt.xticks([0,1], ['Die','Survived'])
plt.ylabel('percentage')
plt.show()

#group_sur_sex = train.groupby(['Survived','Sex'])['PassengerId'].count()*100/train.groupby(['Survived','Sex'])['PassengerId'].count().sum().sum()
#print(group_sur_sex.round(2))
#group_sur_sex.plot(kind='bar',hue=['Survived'])
#p=group_sur_sex.unstack().plot(kind='bar')
#plt.title('% survived based on sex')
#plt.xticks([0,1],['Die','Survived'])
#plt.show()

(pd.crosstab(train.Survived, train.Sex, margins=True, normalize='all').round(4)*100).style.background_gradient(cmap='summer_r')
group_sbc= train.groupby(['Pclass'])['PassengerId'].count()*100/train.groupby(['Pclass'])['PassengerId'].count().sum()
print('% people on each Class \n', group_sbc.round(2))
p = group_sbc.plot(kind='bar')
plt.title('% survived by Class')
plt.show()
ctb = pd.crosstab(train.Pclass, train.Survived,  margins=True , normalize='all').round(4)*100
#ctb.plot(kind='bar')
ctb.style.background_gradient(cmap='summer_r')
group_sur_class = train.groupby(['Survived','Pclass','Sex'])['Survived'].count()
#print(group_sur_class)
#group_sur_class.unstack().plot(kind='bar')

# % survived by Row'
ctb = pd.crosstab(index = [train.Pclass, train.Sex], columns=train.Survived, normalize='index').round(4)*100
ctb.style.background_gradient(cmap='summer_r')
ctb

# % survived by Class & Sex'
ctb_ = pd.crosstab(index = [train.Pclass, train.Sex], columns=train.Survived, normalize='all').round(4)*100
ctb_.style.background_gradient(cmap='summer_r')#ctb.plot(kind='bar')
group_sur_class = train.groupby(['Sex','Survived','Pclass'])['Pclass'].count()
#print(group_sur_class)
p = group_sur_class.unstack().plot(kind='bar')
plt.title('Q survived & sex for each Class ')
plt.show()

p = pd.crosstab([train.Sex, train.Survived], train.Pclass, normalize='columns').plot(kind='bar')
plt.title('% survived & sex in base of the Class ')
plt.show()


pd.crosstab([train.Sex, train.Survived], train.Pclass, normalize='columns').style.background_gradient('summer_r')

#sns.distplot(train.Age.dropna())
#train['Age'].fillna(train['Age'].mean(), inplace=True)
#f, axes = plt.subplots(1,3, figsize=(18,8))
sns.distplot(train.Age.dropna(),label='Age')

plt.legend()
plt.show()
sns.distplot(train.Age.dropna()[train.Survived==True],label='survived', color='green',hist_kws=dict(alpha=0.1))
sns.distplot(train.Age.dropna()[train.Survived==False], label='Die', color='red',hist_kws=dict(alpha=0.1))

plt.legend()
plt.show()
grid = sns.FacetGrid(train.dropna(), col='Pclass', margin_titles=True)
bins = np.linspace(0, 70, 10)
grid.map(sns.distplot, 'Age', bins=bins)
f, axes = plt.subplots(2,3, figsize=(18,8))
plt.subplot(231)
plt.hist(x=train.Age[train.Pclass==1].dropna())
plt.title('distribution of Age on Class I')

plt.subplot(232)
plt.hist(x=train.Age[train.Pclass==2].dropna())
plt.title('Distribution of Age on Class II')

plt.subplot(233)
plt.hist(x=train.Age[train.Pclass==3].dropna())
plt.title('Distribution of Age on Class III')

sns.distplot(train.Age.dropna()[(train.Survived==True) & (train.Pclass==1)],label='surv_class_I', hist=True, color='green', ax=axes[1][0], hist_kws=dict(alpha=0.1))
sns.distplot(train.Age.dropna()[(train.Survived==False) & (train.Pclass==1)],label='Die_class_I',  hist=True, color='Red', ax=axes[1][0], hist_kws=dict(alpha=0.1))


sns.distplot(train.Age.dropna()[(train.Survived==True) & (train.Pclass==2)],label='surv_class_II',  hist=True, color='green', ax=axes[1][1], hist_kws=dict(alpha=0.1))
sns.distplot(train.Age.dropna()[(train.Survived==False) & (train.Pclass==2)],label='Die_class_II',  hist=True, color='Red', ax=axes[1][1], hist_kws=dict(alpha=0.1))

sns.distplot(train.Age.dropna()[(train.Survived==True) & (train.Pclass==3)],label='surv_class_II',  hist=True, color='green', ax=axes[1][2], hist_kws=dict(alpha=0.1))
sns.distplot(train.Age.dropna()[(train.Survived==False) & (train.Pclass==3)],label='Die_class_III',  hist=True, color='red', ax=axes[1][2], hist_kws=dict(alpha=0.1))

plt.legend()
plt.show()
plt.figure(figsize=(15,8))
sns.distplot(train.Fare.dropna(), hist=False, label='All' )
sns.distplot(train.Fare.dropna()[train.Survived==1], color='green', label='survived', hist=False)
sns.distplot(train.Fare.dropna()[train.Survived==0], color='red', label='Die', hist=False)
plt.legend()

train['cFare']= train.Fare.apply(lambda r: 'Low' if r <100 else ('Medium' if (r>=100 and r<200) else ('High' if (r>=200 and r<=300) else 'Ultra')  )).astype('category')
#train.groupby([train.Farex,train.Survived])['PassengerId'].count()
train.cFare.cat.reorder_categories(['Low','Medium','High','Ultra'], inplace=True)

#we create table to CFare cross Survived
ctb = pd.crosstab(train.cFare, train.Survived)

#Normalize data by row, to obtein ratio of survived/die by category of Fare
ctb_Nr = pd.crosstab(train.cFare, train.Survived, normalize='index')
ctb.style.background_gradient('summer_r')
ctb_Nr.plot(kind='bar')
plt.legend(labels=['D', 'S'])
plt.title('% Die/Survive base on Category Fare')

print('table Q values on \n \n', ctb)
print('\n')
# Present the data on %
print('table %  row margin values on \n', ctb_Nr.round(4)*100)

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.pipeline import  Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


import numpy as np
#separar los datos
from sklearn.model_selection import train_test_split
# modelo Lineales
from sklearn import linear_model
# Funcion para procesar data
from sklearn.preprocessing import FunctionTransformer
# create a target array
target = train.Survived
#creta a function to get only the numerical data
get_numeric_data = FunctionTransformer(lambda x: x[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']], validate=False)
#divide the train.csv on Train an Test Data to test models
X_train, X_test,y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=42)
#select the imputer and the Strategy
imp = SimpleImputer(strategy='mean')
# Scale the Data
scl = preprocessing.StandardScaler()

# declare
lg = linear_model.LogisticRegression()
lsgd = linear_model.SGDClassifier()
lper = linear_model.Perceptron()
pipeline_lg = Pipeline([('num',get_numeric_data ),('imputer', imp), ('scale', scl), ('lg', lg)])
pipeline_rc = Pipeline([('num',get_numeric_data ),('imputer', imp), ('scale', scl), ('lsgd', lsgd)])
pipeline_pc = Pipeline([('num',get_numeric_data ),('imputer', imp), ('scale', scl), ('per',lper )])

pipeline_lg.fit(X_train, y_train)
pipeline_rc.fit(X_train, y_train)
pipeline_pc.fit(X_train, y_train)

print(pipeline_lg.score(X_test, y_test))
print(pipeline_rc.score(X_test, y_test))
print(pipeline_pc.score(X_test, y_test))
#create a list of models
#add naive and SVM
from sklearn import svm, naive_bayes
from sklearn.metrics import confusion_matrix

modelo =[
    #linear
        linear_model.LogisticRegression(), 
         linear_model.SGDClassifier(), 
         #linear_model.Perceptron(), 
         linear_model.RidgeClassifier(alpha=0.5),
    #naive
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),
    #suport Vector Machine
        svm.SVC(probability=True),
    
    #Tree
        DecisionTreeClassifier(),
    #Random Forest
        RandomForestClassifier(n_estimators=10),
    #Perceptron
        linear_model.Perceptron(max_iter=5, tol=None)
        ]



pdModelos = pd.DataFrame(columns=['modelo NAME', 'Accuracy', 'Modelo', 'pred', 'confM'])





row = 0
for m in modelo: 
    pipe = Pipeline([('num',get_numeric_data ),('imputer', imp), ('scale', scl), ('model', m )])   
    pipe.fit(X_train, y_train)
    pdModelos.loc[row,'modelo NAME']= m.__class__.__name__
    pdModelos.loc[row,'Accuracy']= pipe.score(X_test, y_test)
    pdModelos.loc[row, 'Modelo'] = pipe.steps[3][1]
    pdModelos.loc[row, 'pred'] = pipe.predict(X_train)
    pdModelos.loc[row, 'confM']= confusion_matrix(y_train, pdModelos.loc[row, 'pred'])
    row+=1
    
pdModelos.sort_values('Accuracy', ascending=False, inplace=True)
true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

m_df = pd.DataFrame(pdModelos.confM[0]/pdModelos.confM[0].sum(axis=1)[:,  np.newaxis], 
                 index=true_class_names,
                 columns= predicted_class_names)

plt.figure(figsize=(15,5))
plt.subplot(121)

sns.heatmap(m_df, annot=True)
print(m)
m = pdModelos.loc[0,'Modelo']
print(pipe.steps[3][1])
pred = pipe.predict(test)

t = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived':pred} )

filename = 'titanic_prediction.csv'

t.to_csv(filename, index=False)

print('Saved file: ' + filename)