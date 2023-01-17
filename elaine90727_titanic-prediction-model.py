#access to system parameters
import sys 

#Data analysis tools
import pandas as pd
import numpy as np

#visualization
import matplotlib
import seaborn as sns

#machine learning
import scipy as sp 
import sklearn

import random
import time

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
data_raw=pd.read_csv('../input/train.csv')  #将被拆分成train和test
data_val=pd.read_csv('../input/test.csv')   #最终的validation 数据

data1=data_raw.copy(deep=True)  #copy data_raw一份，用于做数据可视化分析之类的操作
df = [data1, data_val]        #用于做整体的数据清洗

print(data_raw.info())
print(data_val.info())
data_raw.sample(10)
print('Train columns with null values: \n', data1.isnull().sum())
print('-'*10)

print('Test/Validation columns with null values: \n', data_val.isnull().sum())
print('-'*10)

data_raw.describe(include='all')
for dataset in df:   #有两个dataset，df=[data1,data_val]
	#年龄用中位数填充
	dataset['Age'].fillna(dataset['Age'].median(),inplace=True)

	#Embarked用众数填充（虚拟变量，没有众数）
	dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)

	#Fare船票价格用中位数填充
	dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)
drop_column=['PassengerId','Cabin','Ticket']
data1.drop(drop_column,axis=1,inplace=True)  
print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())  #data_val上没有drop掉cabin列，所以依旧有327个null值
for dataset in df:  #df有两个dataset，df=[data1,data_val]
    
    #IsAlone变量
	dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1  #1是把自己加进去
	dataset['IsAlone']=1  #initialize to yes/1 is alone
	dataset['IsAlone'].loc[dataset['FamilySize']>1]=0 #now update to no/0 if familysize>1

	#名字拆分 Moor, Master. Meier会被拆分留下“Master"
	dataset['Title']=dataset['Name'].str.split(", ", expand=True)[1].str.split(".",expand=True)[0]
    
	#Fare Bins/Buckets 4分位数进行切割,分为4档，每档频次相同
	dataset['FareBin']=pd.qcut(dataset['Fare'],4)  

	#Age Bins/Buckets 5分为5档，按数值分，每档频次不同
	dataset['AgeBin']=pd.cut(dataset['Age'].astype(int),5)  
stat_min=10 

#创建一个True / False序列，选出那些出现次数<10的title
title_names=(data1['Title'].value_counts()<stat_min) 

#将小于10个数的title name用Misc替代，其他的保留
data1['Title']=data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x]==True else x) 

print(data1['Title'].value_counts())
print('-'*10)

data1.info()
data_val.info()
data1.sample(5)
label=LabelEncoder()
for dataset in df:    
    
    #男 1  女 0
	dataset['Sex_Code']=label.fit_transform(dataset['Sex'])  
    #dataset.loc[dataset['Sex']=='male','Sex']=0
    #dataset.loc[dataset['Sex']=='female','Sex']=1
    
    #S 2   Q 1   C 0
	dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked']) 
    
    #Mrs 4 Mr 3 Miss 2  Misc 1 Master 0
	dataset['Title_Code']=label.fit_transform(dataset['Title'])
    
    #年龄的5档切分，从年轻到年长分别对应0-4
	dataset['AgeBin_Code']=label.fit_transform(dataset['AgeBin'])
    
    #票价的4档切分，从便宜到贵分别对应0-3
	dataset['FareBin_Code']=label.fit_transform(dataset['FareBin'])

print(data1.info())
data1.sample(5)
#Y
Target = ['Survived']

#X
#pretty name/values for charts 
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] 

#coded for algorithm calculation  数字特征的变量
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']

data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')


#去除连续变量后的X、y，Age\Fare都用Bin的分位数来替代了  离散变量
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin

print('Bin X Y: ', data1_xy_bin, '\n')
#one hot encoding:
#convert each category value into a new column;assigns 1/0 (True/False) to the column. 
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')
data1_dummy.head()
print('Train columns with null values: \n', data1.isnull().sum())
print('-'*10)
print(data1.info())
print('-'*10)

print('Test/Validation columns with null values: \n', data_val.isnull().sum())
print('-'*10)
print(data_val.info())
print('-'*10)

data_raw.describe(include='all')
#75/25 split 
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)

train1_x_bin, test1_x_bin,train1_y_bin,test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)

train1_x_dummy,test1_x_dummy,train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)

print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x_bin.head()
from sklearn.feature_selection import SelectKBest,f_classif
predictors=['Pclass','Sex_Code','AgeBin_Code','SibSp','Parch','FareBin_Code','Embarked_Code','FamilySize','Title_Code']
selector=SelectKBest(f_classif,k=5)
selector.fit(data1[predictors],data1[Target])
scores=-np.log10(selector.pvalues_)

plt.figure(figsize=(12,6))
sns.barplot(predictors,scores,palette='PiYG')
#plt.xticks(range(len(predictors)),predictors,rotation=90)
plt.title('Importance of variables')
plt.show()
for x in data1_x:
	if data1[x].dtype!='float64':
		print('Survival Correlation by :', x)
		print(data1[[x,Target[0]]].groupby(x,as_index=False).mean())
		print('-'*10,'\n')

print(pd.crosstab(data1['Title'],data1[Target[0]]))
sns.set(context='notebook',style='white',palette='Spectral')
fig,saxis=plt.subplots(2,3,figsize=(12,8))

sns.barplot(x='Embarked',y='Survived',data=data1,ax=saxis[0,0])
sns.barplot(x='Pclass',y='Survived',order=[1,2,3],data=data1,ax=saxis[0,1])
sns.barplot(x='IsAlone',y='Survived',order=[1,0],data=data1,ax=saxis[0,2])

sns.pointplot(x='FareBin',y='Survived',data=data1,ax=saxis[1,0])
sns.pointplot(x='AgeBin',y='Survived',data=data1,ax=saxis[1,1])
sns.pointplot(x='FamilySize',y='Survived',data=data1,ax=saxis[1,2])

plt.show()
plt.figure(figsize=[12,8])

plt.subplot(231)
plt.boxplot(x=data1['Fare'],showmeans=True,meanline=True,)  #虚线为mean，实线是中位数
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(x=data1['Age'],showmeans=True,meanline=True)
plt.title('Age Boxplot')
plt.ylabel('Age (years)')

plt.subplot(233)
plt.boxplot(x=data1['FamilySize'],showmeans=True,meanline=True)
plt.title('FamilySize Boxplot')
plt.ylabel('FamilySize (#)')

plt.subplot(234)
plt.hist(x=[data1[data1['Survived']==1]['Fare'],
			data1[data1['Survived']==0]['Fare']],
			stacked=True,  
			color=['lightseagreen','lightgrey'],
			label=['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x=[data1[data1['Survived']==1]['Age'],
			data1[data1['Survived']==0]['Age']],
			stacked=True,
			color=['lightseagreen','lightgrey'],
			label=['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x=[data1[data1['Survived']==1]['FamilySize'],
			data1[data1['Survived']==0]['FamilySize']],
			stacked=True,   #重合
			color=['lightseagreen','lightgrey'],
			label=['Survived','Dead'])
plt.title('FamilySize Histogram by Survival')
plt.xlabel('FamilySize (#)')
plt.ylabel('# of Passengers')
plt.legend()

plt.show()
fig,(axis1,axis2,axis3)=plt.subplots(1,3,figsize=(14,5))

sns.boxplot(x='Pclass',y='Fare',hue='Survived',data=data1,ax=axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x='Pclass',y='Age',hue='Survived',data=data1,split=True,ax=axis2)
axis1.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x='Pclass',y='FamilySize',hue='Survived',data=data1,ax=axis3)
axis1.set_title('Pclass vs Family Size Survival Comparison')

plt.show()
fig,qaxis=plt.subplots(1,3,figsize=(14,5))

sns.barplot(x='Sex',y='Survived',hue='Embarked',data=data1,ax=qaxis[0],alpha=0.75)
axis1.set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x='Sex',y='Survived',hue='Pclass',data=data1,ax=qaxis[1],alpha=0.75)
axis1.set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x='Sex',y='Survived',hue='IsAlone',data=data1,ax=qaxis[2],alpha=0.75)
axis1.set_title('Sex vs IsAlone Survival Comparison')

plt.show()
fig,(maxis1,maxis2)=plt.subplots(1,2,figsize=(14,5))
sns.pointplot(x='FamilySize',y='Survived',hue='Sex',data=data1,
             palette={'male':'skyblue','female':'salmon'},
             markers=['*','o'],linestyles=['-','--'],ax=maxis1)
maxis1.set_title('FamilySize & Sex Survival Comparison')

sns.pointplot(x='Pclass',y='Survived',hue='Sex',data=data1,
             palette={'male':'skyblue','female':'salmon'},
             markers=['*','o'],linestyles=['-','--'],ax=maxis2)
maxis2.set_title('Pclass & Sex Survival Comparison')
g1=sns.FacetGrid(data1,col='Embarked')
g1.map(sns.pointplot,'Pclass','Survived','Sex',ci=95.0,
      palette={'male':'skyblue','female':'salmon'}) #palette=deep,dark
g1.add_legend()
plt.show()
g2=sns.FacetGrid(data1,row='Sex',col='Pclass',hue='Survived')
g2.map(plt.hist,'Age',alpha=0.75)
g2.add_legend()
plt.show()
g3=sns.FacetGrid(data1,hue='Survived',aspect=4)  #aspect 图的宽度
g3.map(sns.kdeplot,'Age',shade=True)
g3.set(xlim=(0,data1['Age'].max()))
g3.add_legend()
plt.show()
g4 = sns.pairplot(data1, palette = 'deep', size=1.2, 
	diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )

g4.set(xticklabels=[])
plt.show()
def correlation_heatmap(df,a,b):
	_,ax=plt.subplots(figsize=(a,b))
	#colormap=sns.diverging_palette(220,10,as_cmap=True)

	_=sns.heatmap(
		df.corr(),
		cmap='YlGnBu', 
		square=True,
		cbar_kws={'shrink':.9},
		ax=ax,
		annot=True,  #文字
		linewidths=0.1,vmax=1.0,linecolor='white',
		annot_kws={'fontsize':8}

		)

	plt.title('Pearson Correlation of Features',y=1.05,size=12)

correlation_heatmap(data1,12,8)
plt.show()
MLA=[
	#ensemble methods

	ensemble.AdaBoostClassifier(),
	ensemble.BaggingClassifier(),
	ensemble.ExtraTreesClassifier(),
	ensemble.GradientBoostingClassifier(),
	ensemble.RandomForestClassifier(),

	#Gaussian Processes
	gaussian_process.GaussianProcessClassifier(),

	#GLM
	linear_model.LogisticRegressionCV(),
	linear_model.PassiveAggressiveClassifier(),
	linear_model.RidgeClassifierCV(),
	linear_model.SGDClassifier(),
	linear_model.Perceptron(),

	#Navies Bayes
	naive_bayes.BernoulliNB(),
	naive_bayes.GaussianNB(),

	#Nearest Neighbor
	neighbors.KNeighborsClassifier(),

	#SVM
	svm.SVC(probability=True),
	svm.NuSVC(probability=True),
	svm.LinearSVC(),

	#Trees
	tree.DecisionTreeClassifier(),
	tree.ExtraTreeClassifier(),

	#Discriminat Analysis
	discriminant_analysis.LinearDiscriminantAnalysis(),
	discriminant_analysis.QuadraticDiscriminantAnalysis(),

	#xgboost
	XGBClassifier()

]
cv_split=model_selection.ShuffleSplit(n_splits=10,test_size=.3,train_size=.6,random_state=0)

#创建各类机器学习算法的比较列表
MLA_columns=['MLA Name','MLA Parameters','MLA Train Accuracy Mean','MLA Test Accuracy Mean','MLA Test Accuracy 3*STD','MLA Time']
MLA_compare=pd.DataFrame(columns=MLA_columns)

#create table to compare MLA predictions
MLA_predict=data1[Target]

#index through MLA and save performance to table
row_index=0
for alg in MLA:
    
    #获取MLA的名称，填入MLA compare表中的机器学习算法名称列
    MLA_name =alg.__class__.__name__
    MLA_compare.loc[row_index,'MLA Name']=MLA_name
    MLA_compare.loc[row_index,'MLA Parameters']=str(alg.get_params())
    
    cv_results=model_selection.cross_validate(alg,data1[data1_x_bin],data1[Target],cv=cv_split)
    
    MLA_compare.loc[row_index,'MLA Time']=cv_results['fit_time'].mean()
    MLA_compare.loc[row_index,'MLA Train Accuracy Mean']=cv_results['train_score'].mean()
    MLA_compare.loc[row_index,'MLA Test Accuracy Mean']=cv_results['test_score'].mean()
    MLA_compare.loc[row_index,'MLA Test Accuracy 3*STD']=cv_results['test_score'].std()*3
    
    
    alg.fit(data1[data1_x_bin],data1[Target])
    MLA_predict[MLA_name]=alg.predict(data1[data1_x_bin])
    
    row_index+=1
    
    
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'],ascending = False,inplace=True)
MLA_compare
sns.barplot(x='MLA Test Accuracy Mean',y='MLA Name',data=MLA_compare, palette='YlGnBu_r')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
for index,row in data1.iterrows():
    if random.random()>.5:  # Random float x, 0.0 <= x < 1.0 
        data1.set_value(index,'Random_Predict',1)  #predict survived=1
    else:
        data1.set_value(index,'Random_Predict',0)  #predict survived=0

#评估模型准确度
data1['Random_Score']=0 #先初始化为0
data1.loc[(data1['Survived']==data1['Random_Predict']),'Random_Score']=1 #如果survived列也是1，set random-score也是1
score1=data1['Random_Score'].mean()*100
print('Coin Flip Model Accuracy: {:.2f}%'.format(score1))

#用sklearn自带的模型准确度评估方法
score2=metrics.accuracy_score(data1['Survived'],data1['Random_Predict'])*100
print('Coin Flip Model Accuracy w/Scikit: {:.2f}%'.format(score2))
pivot_female=data1[data1.Sex=='female'].groupby(['Sex','Pclass','Embarked','FareBin'])['Survived'].mean()
print('Survival Decision Tree w/Female Node: \n',pivot_female)

pivot_male=data1[data1.Sex=='male'].groupby(['Sex','Title'])['Survived'].mean()
print('\n\nSurvival Decision Tree w/Male Node: \n',pivot_male)
def mytree(df):
    
    model=pd.DataFrame(data={'Predict':[]})
    male_title=['Master']  #survived titles
    
    for index,row in df.iterrows():
        
        # 1）假设所有人都die了，因为大部分人都没有存活
        model.loc[index,'Predict']=0
        
        # 2）按性别划分（性别的重要性影响最高）女的survive，男的die
        if (df.loc[index,'Sex']=='female'):
            model.loc[index,'Predict']=1
            
        # 2.1）女性分支
        if ((df.loc[index,'Sex']=='female')&
           (df.loc[index,'Pclass']==3)&
           (df.loc[index,'Embarked']=='S')&
           (df.loc[index,'Fare']>8)
           ):
            model.loc[index,'Predict']=0
            
        # 2.2）男性分支
        if ((df.loc[index,'Sex']=='male')&
           (df.loc[index,'Title'] in male_title)
           ):
            model.loc[index,'Predict']=1
            
    return model
    
    
tree_predict=mytree(data1)
score3=metrics.accuracy_score(data1['Survived'],tree_predict)*100
print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(score3))
            
print(metrics.classification_report(data1['Survived'], tree_predict))         
import itertools

def plot_confusion_matrix(cm,classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    
    fmt='.2f' if normalize else 'd'
    thresh=cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
            horizontalalignment='center',
            color='white' if cm[i,j]>thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted Label')
    
    
#用混淆矩阵评估mytree的模型准确度

cnf_matrix=metrics.confusion_matrix(data1['Survived'],tree_predict)
np.set_printoptions(precision=2)

class_names=['Dead','Survived']
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix,classes=class_names,
                     title='Confusion Matrix, without normalization'
                     )

plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix,classes=class_names,normalize=True,
                     title='Normalized confusion matrix'
                     )
dtree=tree.DecisionTreeClassifier(random_state=0)
base_results=model_selection.cross_validate(dtree,data1[data1_x_bin],data1[Target],cv=cv_split)
dtree.fit(data1[data1_x_bin],data1[Target])

dtree_train_score=base_results['train_score'].mean()*100
dtree_test_score=base_results['test_score'].mean()*100

print('before DT parameters:',dtree.get_params())
print('before DT Training w/bin score mean: {:.2f}'.format(dtree_train_score))
print('before DT Test w/bin score mean:{:.2f}'.format(dtree_test_score))
print('-'*10)

param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini
              #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best
              'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none
              #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2
              #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all
              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
             }

print(list(model_selection.ParameterGrid(param_grid)))



#用grid_search来选择最优模型参数

tune_model=model_selection.GridSearchCV(tree.DecisionTreeClassifier(),param_grid=param_grid,
                                       scoring='roc_auc',cv=cv_split)
tune_model.fit(data1[data1_x_bin],data1[Target])

tune_train_score=tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100
tune_test_score=tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100
tune_std=tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3

print('after DT parameters:',tune_model.best_params_)
print('after DT Training w/bin score mean: {:.2f}'.format(tune_train_score))
print('after DT Test w/bin score mean:{:.2f}'.format(tune_test_score))
print('after DT Test w/bin score 3*std: +/-{:.2f}'.format(tune_std))
print('-'*10)



#用rfe对显著特征进行选择
dtree_rfe=feature_selection.RFECV(dtree,step=1,scoring='accuracy',cv=cv_split)
dtree_rfe.fit(data1[data1_x_bin],data1[Target])

X_rfe=data1[data1_x_bin].columns.values[dtree_rfe.get_support()]
rfe_results=model_selection.cross_validate(dtree,data1[X_rfe],data1[Target],cv=cv_split)

rfe_train_score=rfe_results['train_score'].mean()*100
rfe_test_score=rfe_results['test_score'].mean()*100
rfe_std=rfe_results['test_score'].std()*100*3

print('after DT Training w/bin score mean: {:.2f}'.format(rfe_train_score))
print('after DT Test w/bin score mean:{:.2f}'.format(rfe_test_score))
print('after DT Test w/bin score 3*std: +/-{:.2f}'.format(rfe_std))
print('-'*10)


rfe_tune_model=model_selection.GridSearchCV(tree.DecisionTreeClassifier(),param_grid=param_grid,
                                            scoring='roc_auc',cv=cv_split)
rfe_tune_model.fit(data1[X_rfe],data1[Target])

rfetune_train_score=rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100
rfetune_test_score=rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100
rfetune_std=rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3


print('after DT parameters:',rfe_tune_model.best_params_)
print('after DT Training w/bin score mean: {:.2f}'.format(rfetune_train_score))
print('after DT Test w/bin score mean:{:.2f}'.format(rfetune_test_score))
print('after DT Test w/bin score 3*std: +/-{:.2f}'.format(rfetune_std))
print('-'*10)

import graphviz 
dot_data = tree.export_graphviz(dtree, out_file=None, 
                                feature_names = data1_x_bin, class_names = True,
                                filled = True, rounded = True)
graph = graphviz.Source(dot_data) 
graph
correlation_heatmap(MLA_predict,20,18)
vote_est=[
    
    ('ada',ensemble.AdaBoostClassifier()),
    ('bc',ensemble.BaggingClassifier()),
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc',ensemble.GradientBoostingClassifier()),
    ('rfc',ensemble.RandomForestClassifier()),
    
    ('gpc',gaussian_process.GaussianProcessClassifier()),
    ('lr',linear_model.LogisticRegressionCV()),
    ('bnb',naive_bayes.BernoulliNB()),
    ('gnb',naive_bayes.GaussianNB()),
    ('knn',neighbors.KNeighborsClassifier()),
    ('svc',svm.SVC(probability=True)),
    ('xgb',XGBClassifier())
    
]

vote_hard=ensemble.VotingClassifier(estimators=vote_est,voting='hard')

vote_hard_cv=model_selection.cross_validate(vote_hard,data1[data1_x_bin],data1[Target],cv=cv_split)
vote_hard.fit(data1[data1_x_bin],data1[Target])

hard_train_score=vote_hard_cv['train_score'].mean()*100
hard_test_score=vote_hard_cv['test_score'].mean()*100
hard_std=vote_hard_cv['test_score'].std()*100*3

print('hard Voting Training w/bin score mean: {:.2f}'. format(hard_train_score))
print('hard Voting Test w/bin score mean: {:.2f}'. format(hard_test_score))
print('hard Voting Test w/bin score 3*std: +/- {:.2f}'. format(hard_std))
print('-'*10)
      
vote_soft=ensemble.VotingClassifier(estimators=vote_est,voting='soft')
vote_soft_cv=model_selection.cross_validate(vote_soft,data1[data1_x_bin],data1[Target],cv=cv_split)
vote_soft.fit(data1[data1_x_bin],data1[Target])

soft_train_score=vote_soft_cv['train_score'].mean()*100
soft_test_score=vote_soft_cv['test_score'].mean()*100
soft_std=vote_soft_cv['test_score'].std()*100*3

print("soft Voting Training w/bin score mean: {:.2f}". format(soft_train_score)) 
print("soft Voting Test w/bin score mean: {:.2f}". format(soft_test_score))
print("soft Voting Test w/bin score 3*std: +/- {:.2f}". format(soft_std))
print('-'*10)
grid_n_estimator=[10,50,100,300]
grid_ratio=[.1,.25,.5,.75,1.0]
grid_learn=[.01,.03,.05,.1,.25]
grid_max_depth=[2,4,6,8,10,None]
grid_min_samples=[5,10,.03,.05,.1]
grid_criterion=['gini','entropy']
grid_bool=[True,False]
grid_seed=[0]


grid_param=[
    [{ #AdaBoostClassifier
        'n_estimators':grid_n_estimator,
        'learning_rate':grid_learn,
        'random_state':grid_seed
    }],
    
    [{ #BaggingClassifier
        'n_estimators':grid_n_estimator,
        'max_samples':grid_ratio,
        'random_state':grid_seed
    }],
    
    [{ #ExtraTreesClassifier
        'n_estimators':grid_n_estimator,
        'criterion':grid_criterion, 
        'max_depth':grid_max_depth,
        'random_state':grid_seed
    }],
    
    [{ #GradientBoostingClassifier
        'learning_rate':[.05],
        'n_estimators':[300],
        'max_depth':grid_max_depth,
        'random_state':grid_seed
    }],

    
    [{ #RandomForestClassifier
        'n_estimators':grid_n_estimator,
        'criterion':grid_criterion,
        'max_depth':grid_max_depth,
        'oob_score':[True],
        'random_state':grid_seed
    }],
    
    [{ #GaussianProcessClassifier
        'max_iter_predict':grid_n_estimator,
        'random_state':grid_seed
    }],
    
    [{ #LogisticRegressionCV
        'fit_intercept':grid_bool,
        'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
        'random_state':grid_seed
    }],

    
    [{ #BernoulliNB
        'alpha':grid_ratio
    }],
    
    [{#GaussianNB
        
    }],
    
    [{ #KNeighborsClassifier
        'n_neighbors':[1,2,3,4,5,6,7],
        'weights':['uniform','distance'],
        'algorithm':['auto','ball_tree','kd_tree','brute']
    }],
    
    
    [{ #SVC
        'C':[1,2,3,4,5],
        'gamma':grid_ratio,
        'decision_function_shape':['ovo','ovr'],
        'probability':[True],
        'random_state':grid_seed
    }],
    
    
    [{ #XGBClassifier
        'learning_rate':grid_learn,
        'max_depth':[1,2,4,6,8,10],
        'n_estimators':grid_n_estimator,
        'seed':grid_seed
    }],
    
    
]


for clf,param in zip(vote_est,grid_param):
    
    best_search=model_selection.GridSearchCV(estimator=clf[1],
                                             param_grid=param,
                                            cv=cv_split,
                                            scoring='roc_auc')
    best_search.fit(data1[data1_x_bin],data1[Target])
    
    best_param=best_search.best_params_
    print('The best parameter for {} is {}'.format(clf[1],best_param))
    clf[1].set_params(**best_param)   #用best_param替代参数设置
grid_hard=ensemble.VotingClassifier(estimators=vote_est,voting='hard')

grid_hard_cv=model_selection.cross_validate(vote_hard,data1[data1_x_bin],data1[Target],cv=cv_split)
grid_hard.fit(data1[data1_x_bin],data1[Target])

gridhard_train_score=vote_hard_cv['train_score'].mean()*100
gridhard_test_score=vote_hard_cv['test_score'].mean()*100
gridhard_std=vote_hard_cv['test_score'].std()*100*3

print('hard Voting Training w/bin score mean: {:.2f}'. format(gridhard_train_score))
print('hard Voting Test w/bin score mean: {:.2f}'. format(gridhard_test_score))
print('hard Voting Test w/bin score 3*std: +/- {:.2f}'. format(gridhard_std))
print('-'*10)
      
grid_soft=ensemble.VotingClassifier(estimators=vote_est,voting='soft')
grid_soft_cv=model_selection.cross_validate(vote_soft,data1[data1_x_bin],data1[Target],cv=cv_split)
grid_soft.fit(data1[data1_x_bin],data1[Target])

gridsoft_train_score=vote_soft_cv['train_score'].mean()*100
gridsoft_test_score=vote_soft_cv['test_score'].mean()*100
gridsoft_std=vote_soft_cv['test_score'].std()*100*3

print("soft Voting Training w/bin score mean: {:.2f}". format(gridsoft_train_score)) 
print("soft Voting Test w/bin score mean: {:.2f}". format(gridsoft_test_score))
print("soft Voting Test w/bin score 3*std: +/- {:.2f}". format(gridsoft_std))
print('-'*10)
print(data_val.info())
print('-'*10)

#data_val['Survived']=mytree(data_val).astype(int)
data_val['Survived']=grid_hard.predict(data_val[data1_x_bin])

submit=data_val[['PassengerId','Survived']]
submit.to_csv('submit,csv',index=False)

print('Validation Data Distribution:\n',data_val['Survived'].value_counts(normalize=True))
