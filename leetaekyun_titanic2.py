import sys                      #단축키 ctrl + z , y  (서로 반대)

sys.version                     #↑: go to cell start(코드의 시작 부분으로 커서 이동)



                                #↓: go to cell end(코드의 마지막 부분으로 커서 이동)

import pandas as pd             # x or dd : 셀삭제

pd.__version__                  # 위에 셀추가: a, 아래에 셀추가: b,

import matplotlib               # shift + M : 선택된 셀들을 merge한다.

matplotlib.__version__          # shif +  ↓ or ↑ : 셀 선택

import numpy as np              # f : find and replace

np.__version__                  # ctrl + <- or -> : go one word right(left)

import scipy as sp              # shift + space : scroll up (길게눌러)

sp.__version__                  # space : scroll down (길게눌러)

import sklearn                  # ctrl + [(left)  or ](right) : 들여쓰기 

sklearn.__version__             # k : select cell above

                                # j : select cell down 

import random                   # ALT + <- or -> : cell의 왼쪽(오른쪽) 끝으로 이동

import time                     # shift + ctrl + <- or -> : 단어단위 선택



import warnings

warnings.filterwarnings('ignore')
from sklearn import svm, tree, linear_model, neighbors , naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



from sklearn.preprocessing import OneHotEncoder , LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

#from pandas.tools.plotting import scatter_matrix



%matplitlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8



# nominal data(명목자료)를 categorical data (범주형 자료)라 부르기도 한다. -> Survived

# 많은 수의 예측변수가 더 좋은 모델을 만들지는 않는다. 올바른 변수가 더 좋은 모델을 만든다.



#Pclass -> ordidnal data (1st class 2nd class...)
data_raw = pd.read_csv('../input/titanic/train.csv')

data_val = pd.read_csv('../input/titanic/test.csv')



data1 = data_raw.copy(deep = True) # 깊은 복사  VS 얕은 복사 -> 깊은 복사값이 변경되도 원본은 변경안됨(깊은 복사)

data_cleaner = [data1, data_val]   # dataframe 자체를 리스트에 넣음 



print(data_raw.info())

data_raw.sample(10) #랜덤하게 데이터 10개 추출





# 질적변수 missing value는 mode(최빈값)를 이용해서 채우는 것이 좋고

# 양적변수 missing value는 mean, median or mean + randomized standard deviation가 좋다.

# class별 평균나이, fare에 따른 embark port로 채워넣는 것도 하나의 방법이다.

print('Train columns with null values:\n', data1.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', data_val.isnull().sum())

print("-"*10)



data_raw.describe(include = 'all') # 'all'하면 top(가장많은것) freq(가장 많은 것의 빈도수)표시



# ctrl + '+' -> 웹페이지 화면 확대
for dataset in data_cleaner : 

    dataset['Age'].fillna(dataset['Age'].median(), inplace =True)

    

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

drop_column = ['PassengerId','Cabin','Ticket']

data1.drop(drop_column, axis =1 , inplace = True)



print(data1.isnull().sum())

print("-"*10)

print(data_val.isnull().sum())
for dataset in data_cleaner :

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    dataset['IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1 ] = 0

    

    dataset['Title'] = dataset['Name'].str.split(", ", expand =True)[1].str.split(".", expand = True)[0] # expand = True : dataframe으로 바꿔줌



    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4) # 동일 개수로 나누어서 범주 만들기, 범주마다 원소의 개수가 같다.

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int),5) # 동일 길이로 나누어서 범주 만들기 

                                                             # float 데이터에 astype(int) 적용하면 소수점만 버린 정수가 나옴
stat_min = 10

title_names = (data1['Title'].value_counts() < stat_min)



data1['Title'] = data1['Title'].apply(lambda x : 'Misc' if title_names.loc[x] == True else x )

print(data1['Title'].value_counts())

print("-"*10)



data1.info()

data_val.info()

data1.sample(10)
'''  Label Encoding 과 One Hot Encoding의 차이

   

   Label Encoding은 한칼럼에서 '사과' : 1 , '배' : 2로 값을 바꾸고

   One Hot Encoding은 '사과'와 '배'를 표현하기 위해 칼럼을 2개만들어서 표현한다.

   

   Label Encoding은 한 칼럼내에서 1,2,3으로 값을 부여하기 때문에 모델이 이 수치들을 순서형(대소관계) 으로 인식할 수 있는 문제가 있다.

   이 문제를 해결하기 위한 것이 One Hot Encoding이다.

   

   

   *1 트리모형 사용시 One Hot Encoding을 써야하는 경우

    - 순서형 변수가 아닌경우 (dog,cat,mouse)

    - 비선형 데이터 (x(설명)끼리는 label상 가까운데 - y(종속)값 끼리는 가깝지 않은 관계)

    

   *1 트리모형 사용시 Label Encoding을 써야하는 경우

    - 범주형 변수가 순서형인 경우( 초등학교 - 중학교 - 고등학교 )

    - 비슷한 카테고리에 비슷한 label을 부여하면 트리 모형 이용시 분할이 적어지고 따라서 실행시간이 줄어든다.

    

   ** 원핫 인코딩 사용시 범주형 변수의 값이 많은경우

   1. 메모리 소비가 큼 (sparse matrix로 해결)

   2. non -categorical data가 학습에 거의 안쓰일 수가 있다. (ex, 9개의 non-categorical data + 100개의 unique한 특성을 갖고있는 1개의 categorical data = 109개의 칼럼)

   -> 해결방법1( In xgboost it is called colsample_bytree, in sklearn's Random Forest max_features. )

   -> 해결방법2( PCA with one hor encoding )

   

   ##1 트리모형이 아닌 경우 one hot encoding을 써야하는 경우

   - 종속변수와 타겟변수 사이의 관계가 비선형인 경우

   

   ##2 트리모형이 아닌 경우 Label Encoding을 써야하는 경우

   - 종속변수와 타겟변수 사이의 관계가 선형적인 경우

    

'''
label = LabelEncoder()

for dataset in data_cleaner :

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

    

Target = ['Survived']



data1_x  = ['Sex','Pclass','Embarked','Title','SibSp','Parch','Age','Fare','FamilySize','IsAlone']

data1_x_calc = ['Sex_Code','Pclass','Embarked_Code','Title_Code','SibSp','Parch','Age','Fare']

data1_xy = Target + data1_x

print('Original X Y: ', data1_xy, '\n')



data1_x_bin = ['Sex_Code','Pclass','Embarked_Code','Title_Code','FamilySize','AgeBin_Code','FareBin_Code']

data1_xy_bin = Target + data1_x_bin

print('Bin X Y: ', data1_xy_bin, '\n')



data1_dummy = pd.get_dummies(data1[data1_x])

data1_x_dummy = data1_dummy.columns.tolist() # tolist() -> list로 변환

data1_xy_dummy = Target + data1_x_dummy

print('Dummy X Y: ', data1_xy_dummy, '\n')



data1_dummy.head()
data1_dummy.columns
# [1,2,3 ] + [4,5] -> [1, 2, 3, 4, 5]
print('Train columns with null values: \n', data1.isnull().sum())

print("-"*10) 

print(data1.info())

print("-"*10)



print('Test/Validation columns with null values: \n', data_val.isnull().sum())

print("-"*10)

print(data_val.info())

print("-"*10)



data_raw.describe(include = 'all')
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target], random_state = 0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)



print("Data1 Shape: {}".format(data1.shape))

print("Train1 Shape: {}".format(train1_x.shape))

print("Test1 Shape: {}".format(test1_x.shape))



train1_x_bin.head()
for x in data1_x :

    if data1[x].dtype != 'float64' :

        print('Survival Correlation by :', x)

        print(data1[[x,Target[0]]].groupby(x, as_index = False).mean())

        print('-'*10 , '\n')





        print(pd.crosstab(data1['Title'], data1[Target[0]]))
plt.figure( figsize = (10,10))  # 이렇게 앞에다가 해주면 그림 크기 조정 가능



plt.subplot(231)    # plt.subplot()  괄호안에 들어가는 숫자는 1과 6사이의 숫자여야만 한다.   // plt .subplot(2,3,1)로도 가능

plt.boxplot(x = data1['Fare'], showmeans = True , meanline = True) 

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(x = data1['Age'], showmeans = True , meanline = True) 

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(x = data1['FamilySize'], showmeans = True , meanline = True) 

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x = [data1[data1['Survived'] ==1 ]['Fare'], data1[data1['Survived'] == 0]['Fare']], stacked = True , color = ['g','r'],

        label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [data1[data1['Survived'] ==1 ]['Age'], data1[data1['Survived'] == 0]['Age']], stacked = True , color = ['g','r'],

        label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [data1[data1['Survived'] ==1 ]['FamilySize'], data1[data1['Survived'] == 0]['FamilySize']], stacked = True , color = ['g','r'],

        label = ['Survived','Dead'])

plt.title('FamilySize Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()

fig ,axes = plt.subplots(2,3, figsize = (16,12))



sns.barplot(x = 'Embarked' , y= 'Survived', order = ['S','Q','C'] , data = data1 , ax = axes[0,0])  # '0'부터 시작!!!

sns.barplot(x = 'Pclass' , y= 'Survived', order = [1,2,3] ,  data = data1 , ax = axes[0,1])  #barplot : 평균값을 보여준다.

sns.barplot(x = 'IsAlone' , y= 'Survived', order = [1,0] ,  data = data1 , ax = axes[0,2])   #pointplot : 평균값을 보여준다. # 신뢰구간?

sns.pointplot(x = 'FareBin' , y= 'Survived',  data = data1 , ax = axes[1,0])

sns.pointplot(x = 'AgeBin' , y= 'Survived',  data = data1 , ax = axes[1,1])

sns.pointplot(x = 'FamilySize' , y= 'Survived',  data = data1 , ax = axes[1,2])

fig, (axes1, axes2 , axes3) = plt.subplots(1,3, figsize = (14,10))    #figsize = (가로로커짐, 세로로커짐)



sns.boxplot(x = 'Pclass', y= 'Fare', hue = 'Survived' , data = data1 , ax = axes1)

axes1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass' , y = 'Age', hue = 'Survived' , data =  data1, split = True,  ax= axes2)

axes2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x = 'Pclass' , y = 'FamilySize', hue = 'Survived' , data =  data1, ax= axes3)

axes3.set_title('Pclass vs Family Size Survival Comparison')
fig, qaxis = plt.subplots(1,3, figsize = (14,12))    #figsize = (가로로커짐, 세로로커짐)



sns.barplot(x = 'Sex', y= 'Survived', hue = 'Embarked' , order = ['female','male'] , data = data1 , ax = qaxis[0])

qaxis[0].set_title('Sex vs Embarked Survival Comparison')



sns.barplot(x = 'Sex' , y = 'Survived', hue = 'Pclass' , data =  data1,  ax= qaxis[1])

qaxis[1].set_title('Sex vs Pclass Survival Comparison')



sns.barplot(x = 'Sex' , y = 'Survived', hue = 'IsAlone' , data =  data1, ax= qaxis[2])

qaxis[2].set_title('Sex vs IsAlone Survival Comparison')
fig, (maxis1, maxis2) = plt.subplots(1,2, figsize = (14,12)) 



sns.pointplot(x = 'FamilySize', y= 'Survived', hue = 'Sex', data = data1, palette ={"male" :"blue", "female" : "pink"},

             markers = ["*","o"], linestyles= ["-","--"], ax= maxis1)



sns.pointplot(x = 'Pclass', y= 'Survived', hue = 'Sex', data = data1, palette ={"male" :"blue", "female" : "pink"},

             markers = ["*","o"], linestyles= ["-","--"], ax= maxis2)
e = sns.FacetGrid(data1 , col = 'Embarked')

e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci =95.0, palette = 'deep')

e.add_legend()
a = sns.FacetGrid(data1, hue = 'Survived' , aspect =4)  # aspect : 가로로 늘림

a.map(sns.kdeplot, 'Age', shade = True)

a.set(xlim = (0,data1['Age'].max()))

a.add_legend()
h = sns.FacetGrid(data1 , row ='Sex', col = 'Pclass', hue= 'Survived')  # sns.FacetGrid : 여기서는 row,col,hue로 칸 자체를 나눔

h.map(plt.hist, 'Age', alpha = 0.75)                                    # h.map(plt.hist, 'Age', alpha = .75) : 여기서는 x축 y축을 나눔

h.add_legend()                                                          # alpha : 투명도
pp = sns.pairplot(data1, hue = 'Survived', palette = 'deep', size = 1.2, diag_kind ='kde', diag_kws = dict(shade =True), plot_kws = dict(s =10))

pp.set(xticklabels = [])
def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(data1)
# 일반적으로 회귀에는 기본 k-겹 교차검증을 사용하고, 분류에는 StratifiedKFold를 사용한다.



MLA = [ 

     ensemble.AdaBoostClassifier(),

     ensemble.BaggingClassifier(),

     ensemble.ExtraTreesClassifier(),

     ensemble.GradientBoostingClassifier(),

     ensemble.RandomForestClassifier(),

    

     gaussian_process.GaussianProcessClassifier(),

    

     linear_model.LogisticRegressionCV(),

     linear_model.PassiveAggressiveClassifier(),

     linear_model.RidgeClassifierCV(),

     linear_model.SGDClassifier(),

     linear_model.Perceptron(),

     

     naive_bayes.BernoulliNB(),

     naive_bayes.GaussianNB(),

    

     neighbors.KNeighborsClassifier(),

    

     svm.SVC(probability = True),

     svm.NuSVC(probability = True),

     svm.LinearSVC(),

    

     tree.DecisionTreeClassifier(),

     tree.ExtraTreeClassifier(),

    

     discriminant_analysis.LinearDiscriminantAnalysis(),

     discriminant_analysis.QuadraticDiscriminantAnalysis(),

    

     XGBClassifier() ]



cv_split = model_selection.ShuffleSplit(n_splits=10, test_size = .3, train_size= .6, 

                                        random_state=0) # StratifiedShuffleSplit : 분류 작업에 더 적합 

                                                        # ShuffleSplit(임의분할 교차검증) : 유연함, 전체데이터의 일부만 사용가능(부분 샘플링)

                                                        # random_state : random_state 값에 따라서 train_set ,test_set 데이터들이 달라짐 

MLA_columns = ['MLA Name', 'MLA Parameters' , 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' , 'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



MLA_predict = data1[Target]



row_index = 0

for alg in MLA : 

    

    MLA_name = alg.__class__.__name__ 

    MLA_compare.loc[row_index , 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index , 'MLA Parameters'] = str(alg.get_params())

    

    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin],

                                               data1[Target], cv = cv_split, return_train_score=True)

    

    MLA_compare.loc[row_index , 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index , 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()  # train_score? test_score?

    MLA_compare.loc[row_index , 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

    MLA_compare.loc[row_index , 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3

    

    alg.fit(data1[data1_x_bin], data1[Target])

    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])

    

    row_index +=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False , inplace = True)

MLA_compare
plt.figure(figsize=(7,7))

sns.barplot(x ='MLA Test Accuracy Mean' , y = 'MLA Name', data = MLA_compare , color ='g')



plt.title('Machine Learning Algotirhm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
for index, row in data1.iterrows(): 

    #random number generator: https://docs.python.org/2/library/random.html

    if random.random() > .5:     # Random float x, 0.0 <= x < 1.0    

        data1.set_value(index, 'Random_Predict', 1) #predict survived/1

    else: 

        data1.set_value(index, 'Random_Predict', 0) #predict died/0

    

data1['Random_Score'] = 0 

data1.loc[(data1['Survived'] == data1['Random_Predict']), 'Random_Score'] =1 



print('Coin Flip Model Accuracy : {:.2f}%'.format(data1['Random_Score'].mean()*100))



print('Coin Flip Model Accuracy w/Scikit: {:.2f}%'.format(metrics.accuracy_score(data1['Survived'], data1['Random_Predict'])*100))



# metrics.accuracy_score()
pivot_female  = data1[data1.Sex == 'female'].groupby(['Sex','Pclass','Embarked','FareBin'])['Survived'].mean()

print('Survival Decision Tree w/Female Node: \n', pivot_female)



pivot_male = data1[data1.Sex =='male'].groupby(['Sex','Title'])['Survived'].mean()

print('\n\nSurvival Decision Tree w/male Node: \n', pivot_male)
#handmade data model using brain power (and Microsoft Excel Pivot Tables for quick calculations)

def mytree(df):

    

    #initialize table to store predictions

    Model = pd.DataFrame(data = {'Predict':[]})

    male_title = ['Master'] #survived titles



    for index, row in df.iterrows():



        #Question 1: Were you on the Titanic; majority died

        Model.loc[index, 'Predict'] = 0



        #Question 2: Are you female; majority survived

        if (df.loc[index, 'Sex'] == 'female'):

                  Model.loc[index, 'Predict'] = 1



        #Question 3A Female - Class and Question 4 Embarked gain minimum information



        #Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0       

        if ((df.loc[index, 'Sex'] == 'female') & 

            (df.loc[index, 'Pclass'] == 3) & 

            (df.loc[index, 'Embarked'] == 'S')  &

            (df.loc[index, 'Fare'] > 8)



           ):

                  Model.loc[index, 'Predict'] = 0



        #Question 3B Male: Title; set anything greater than .5 to 1 for majority survived

        if ((df.loc[index, 'Sex'] == 'male') &

            (df.loc[index, 'Title'] in male_title)

            ):

            Model.loc[index, 'Predict'] = 1

        

        

    return Model





#model data

Tree_Predict = mytree(data1)

print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(metrics.accuracy_score(data1['Survived'], Tree_Predict)*100))



#sklearn.metrics.classification_report 

#sklearn.metrics.recall_score

#sklearn.metrics.f1_score 

print(metrics.classification_report(data1['Survived'], Tree_Predict))    



# metric 공부!!!
import itertools ###############

def plot_confusion_matrix(cm , classes, normalize =False, title = 'Confusion Matrix', cmap = plt.cm.Blues) : 

    if normalize :

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #np.newaxis : 축 추가

        print("Normalized confusion matrix")

    else :

        print('Confusion matrix , without normalization')

        

    print(cm)

    

    plt.imshow(cm, interpolation ='nearest', cmap = cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation =45) 

    plt.yticks(tick_marks, classes)

    

    fmt = '.2f' if normalize else 'd'  ###################################################### 간략

    thresh = cm.max() / 2

    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) : # itertools.product() 조합 칼테시안 

        plt.text(j,i, format(cm[i,j], fmt), horizontalalignment ='center',

                color = 'white' if cm[i,j] > thresh else 'black')

        

        plt.tight_layout()

        plt.ylabel('True label')

        plt.xlabel('predicted label')

        

cnf_matrix = metrics.confusion_matrix(data1['Survived'], Tree_Predict) #  metrics.confusion_matrix(data1['Survived'], Tree_Predict , labels = [1,0 ]) : labels이용해서 구분

np.set_printoptions(precision =2)



class_names = ['Dead','Survived']



plt.figure()

plot_confusion_matrix(cnf_matrix, classes =class_names, title = 'Confusion matrix, without normalization')



plt.figure()

plot_confusion_matrix(cnf_matrix, classes =class_names, normalize = True ,title = 'Normalized confusion matrix')
dtree = tree.DecisionTreeClassifier(random_state = 0)

base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv = cv_split, return_train_score=True)

dtree.fit(data1[data1_x_bin], data1[Target]) 



print('BEFORE DT Patameters: ', dtree.get_params())

print('BEFORE DT Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100))

print('BEFORE DT Test w/bin score mean: {:.2f}'.format(base_results['test_score'].mean()*100))

print('BEFORE DT Test w/bin score 3*std: +/- {:.2f}'.format(base_results['test_score'].std()*100*3))

print('-'*10)

      

param_grid = {'criterion' : ['gini','entropy'], 'max_depth' : [2,4,6,8,10,None], 'random_state':[0]}

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid = param_grid, scoring = 'roc_auc', cv = cv_split,

                                         return_train_score = True)

tune_model.fit(data1[data1_x_bin], data1[Target])

      

print('AFTER DT Parameters: ', tune_model.best_params_)

print('AFTER DT Training w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))

print('AFTER DT Test w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print('AFTER DT Test w/bin score 3*std: +/- {:.2f}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
print('BEFORE DT RFE Training Shape Old: ' , data1[data1_x_bin].shape)

print('BEFORE DT RFE Training Columns Old: ' , data1[data1_x_bin].columns.values)



print('BEFORE DT RFE Traning w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100))

print('BEFORE DT RFE Test w/bin score mean: {:.2f}'.format(base_results['test_score'].mean()*100))

print('BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}'.format(base_results['test_score'].std()*100*3))

print('-'*10)



dtree_rfe = feature_selection.RFECV(dtree, step =1 , scoring = 'accuracy', cv = cv_split)

dtree_rfe.fit(data1[data1_x_bin], data1[Target])



X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()] #  get_support() 선택된 칼럼들에 대한 색인 true or false  형태

rfe_results = model_selection.cross_validate(dtree , data1[X_rfe], data1[Target], cv = cv_split, return_train_score=True )



print('AFTER DT RFE Training Shape New: ' , data1[X_rfe].shape)

print('AFTER DT RFE Training Columns New: ' , X_rfe)



print('AFTER DT RFE Traning w/bin score mean: {:.2f}'.format(rfe_results['train_score'].mean()*100))

print('AFTER DT RFE Test w/bin score mean: {:.2f}'.format(rfe_results['test_score'].mean()*100))

print('AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}'.format(rfe_results['test_score'].std()*100*3))

print('-'*10)



rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid = param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

rfe_tune_model.fit(data1[X_rfe], data1[Target])



print('AFTER DT RFE Tuned Parameters: ' ,rfe_tune_model.best_params_)

print('AFTER DT RFE Tuned Training w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))



print('AFTER DT RFE Tuned Test w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))

print('AFTER DT RFE Tuned Test w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print('AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}'.format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)

# correlation_heatmap(MLA_predict) 모델들간의 상관관계 -> 나중에 결합(combining)

#MLA_predict
vote_est = [

    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html

    ('ada', ensemble.AdaBoostClassifier()),

    ('bc', ensemble.BaggingClassifier()),

    ('etc',ensemble.ExtraTreesClassifier()),

    ('gbc', ensemble.GradientBoostingClassifier()),

    ('rfc', ensemble.RandomForestClassifier()),



    #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc

    ('gpc', gaussian_process.GaussianProcessClassifier()),

    

    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    ('lr', linear_model.LogisticRegressionCV()),

    

    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html

    ('bnb', naive_bayes.BernoulliNB()),

    ('gnb', naive_bayes.GaussianNB()),

    

    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html

    ('knn', neighbors.KNeighborsClassifier()),

    

    #SVM: http://scikit-learn.org/stable/modules/svm.html

    ('svc', svm.SVC(probability=True)),

    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

   ('xgb', XGBClassifier())



]





#Hard Vote or majority rules

vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

vote_hard_cv = model_selection.cross_validate(vote_hard, data1[data1_x_bin], data1[Target], cv  = cv_split, return_train_score=True)

vote_hard.fit(data1[data1_x_bin], data1[Target])



print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 

print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))

print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))

print('-'*10)





#Soft Vote or weighted probabilities

vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

vote_soft_cv = model_selection.cross_validate(vote_soft, data1[data1_x_bin], data1[Target], cv  = cv_split, return_train_score=True)

vote_soft.fit(data1[data1_x_bin], data1[Target])



print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 

print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))

print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))

print('-'*10)
#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

grid_n_estimator = [10, 50, 100, 300]

grid_ratio = [.1, .25, .5, .75, 1.0]

grid_learn = [.01, .03, .05, .1, .25]

grid_max_depth = [2, 4, 6, 8, 10, None]

grid_min_samples = [5, 10, .03, .05, .10]

grid_criterion = ['gini', 'entropy']

grid_bool = [True, False]

grid_seed = [0]





grid_param = [

            [{

            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

            'n_estimators': grid_n_estimator, #default=50

            'learning_rate': grid_learn, #default=1

            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R

            'random_state': grid_seed

            }],

       

    

            [{

            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

            'n_estimators': grid_n_estimator, #default=10

            'max_samples': grid_ratio, #default=1.0

            'random_state': grid_seed

             }],



    

            [{

            #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier

            'n_estimators': grid_n_estimator, #default=10

            'criterion': grid_criterion, #default=”gini”

            'max_depth': grid_max_depth, #default=None

            'random_state': grid_seed

             }],





            [{

            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

            #'loss': ['deviance', 'exponential'], #default=’deviance’

            'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.

            'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.

            #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”

            'max_depth': grid_max_depth, #default=3   

            'random_state': grid_seed

             }],



    

            [{

            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

            'n_estimators': grid_n_estimator, #default=10

            'criterion': grid_criterion, #default=”gini”

            'max_depth': grid_max_depth, #default=None

            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.

            'random_state': grid_seed

             }],

    

            [{    

            #GaussianProcessClassifier

            'max_iter_predict': grid_n_estimator, #default: 100

            'random_state': grid_seed

            }],

        

    

            [{

            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV

            'fit_intercept': grid_bool, #default: True

            #'penalty': ['l1','l2'],

            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs

            'random_state': grid_seed

             }],

            

    

            [{

            #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

            'alpha': grid_ratio, #default: 1.0

             }],

    

    

            #GaussianNB - 

            [{}],

    

            [{

            #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

            'n_neighbors': [1,2,3,4,5,6,7], #default: 5

            'weights': ['uniform', 'distance'], #default = ‘uniform’

            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

            }],

            

    

            [{

            #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r

            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

            'C': [1,2,3,4,5], #default=1.0

            'gamma': grid_ratio, #edfault: auto

            'decision_function_shape': ['ovo', 'ovr'], #default:ovr

            'probability': [True],

            'random_state': grid_seed

             }],



    

            [{

            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html

            'learning_rate': grid_learn, #default: .3

            'max_depth': [1,2,4,6,8,10], #default 2

            'n_estimators': grid_n_estimator, 

            'seed': grid_seed  

             }]   

        ]







start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter

for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip



    #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm

    #print(param)

    

    

    start = time.perf_counter()        

    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')

    best_search.fit(data1[data1_x_bin], data1[Target])

    run = time.perf_counter() - start



    best_param = best_search.best_params_

    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))

    clf[1].set_params(**best_param) # best parameter로 각 모델마다 설정해 놓는다. grid search로 best_param 설정하기 가능



run_total = time.perf_counter() - start_total

print('Total optimization time was {:.2f} minutes.'.format(run_total/60))



print('-'*10)
#Hard Vote or majority rules w/Tuned Hyperparameters

grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

grid_hard_cv = model_selection.cross_validate(grid_hard, data1[data1_x_bin], data1[Target], cv  = cv_split, return_train_score=True)

grid_hard.fit(data1[data1_x_bin], data1[Target])



print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 

print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))

print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))

print('-'*10)



#Soft Vote or weighted probabilities w/Tuned Hyperparameters

grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

grid_soft_cv = model_selection.cross_validate(grid_soft, data1[data1_x_bin], data1[Target], cv  = cv_split, return_train_score=True)

grid_soft.fit(data1[data1_x_bin], data1[Target])



print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 

print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))

print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))

print('-'*10)



#12/31/17 tuned with data1_x_bin

#The best parameter for AdaBoostClassifier is {'learning_rate': 0.1, 'n_estimators': 300, 'random_state': 0} with a runtime of 33.39 seconds.

#The best parameter for BaggingClassifier is {'max_samples': 0.25, 'n_estimators': 300, 'random_state': 0} with a runtime of 30.28 seconds.

#The best parameter for ExtraTreesClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0} with a runtime of 64.76 seconds.

#The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 34.35 seconds.

#The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 76.32 seconds.

#The best parameter for GaussianProcessClassifier is {'max_iter_predict': 10, 'random_state': 0} with a runtime of 6.01 seconds.

#The best parameter for LogisticRegressionCV is {'fit_intercept': True, 'random_state': 0, 'solver': 'liblinear'} with a runtime of 8.04 seconds.

#The best parameter for BernoulliNB is {'alpha': 0.1} with a runtime of 0.19 seconds.

#The best parameter for GaussianNB is {} with a runtime of 0.04 seconds.

#The best parameter for KNeighborsClassifier is {'algorithm': 'brute', 'n_neighbors': 7, 'weights': 'uniform'} with a runtime of 4.84 seconds.

#The best parameter for SVC is {'C': 2, 'decision_function_shape': 'ovo', 'gamma': 0.1, 'probability': True, 'random_state': 0} with a runtime of 29.39 seconds.

#The best parameter for XGBClassifier is {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0} with a runtime of 46.23 seconds.

#Total optimization time was 5.56 minutes.
#prepare data for modeling

print(data_val.info())

print("-"*10)

#data_val.sample(10)



#handmade decision tree - submission score = 0.77990

data_val['Survived'] = mytree(data_val).astype(int)



data_val['Survived'] = grid_hard.predict(data_val[data1_x_bin])





submit = data_val[['PassengerId','Survived']]

submit.to_csv("../working/submit.csv", index=False)



print('Validation Data Distribution: \n', data_val['Survived'].value_counts(normalize = True)) # normalize = True 비율 보여줌

submit.sample(10)  # 무작위 추출 data1.sample(frac =2, replace = True) -> 데이터 10개있으면 2배로 복제함(20개 , 똑같은 로우가 하나 더 생김)