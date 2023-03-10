# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use ('seaborn-whitegrid')



import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_PassengerId = test_df['PassengerId']
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
def bar_plot(variable):

    '''

    input :variable ex: 'Sex'

    output : bar plot& value count

    '''

    

    var = train_df[variable]

    

    varValue = var.value_counts()

  #visulation  

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel('Frequency')

    plt.title(variable)

    plt.show()

    print('{}:\n{}'.format(variable, varValue))
category1 = ['Survived','Sex','Pclass','Embarked','SibSp','Parch']

for c in category1:

    bar_plot(c)
category2 = ['Cabin','Name','Ticket']

for c in category2:

    print('{}\n'.format(train_df[c].value_counts()))

    
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel('Frequency')

    plt.title('{}distribution with hist'.format(variable))

    plt.show()
numericVar = ['Fare','Age','PassengerId']

for n in numericVar:

    plot_hist(n)

    
#Pclass vs Survived

train_df[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean()
train_df[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by= 'Survived', ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by= 'Survived', ascending=False)
train_df[['SibSp','Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by= 'Survived', ascending=False)
train_df[['Parch','Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by= 'Survived', ascending=False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        #1 st quartile

        Q1 = np.percentile(df[c],25)        

        # 3 rd quartile

        Q3 = np.percentile(df[c],75)

        #IQR

        IQR = Q3-Q1

        #outlier step

        otlier_step = IQR*1.5

        #detect outlier and their indeces

        outlier_list_col = df[(df[c]<Q1 - otlier_step) | (df[c] > Q3 + otlier_step)].index

        #store indeces

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)#counter ka?? tane oldu??unu s??yler.

    multiple_outliers = list(i for i, v in outlier_indices.items() if v>2)

    

    return multiple_outliers

        
train_df.loc[detect_outliers(train_df, ['Age','SibSp','Parch','Fare'])]
#drop outliers

train_df = train_df.drop(detect_outliers(train_df, ['Age','SibSp','Parch','Fare']),axis = 0).reset_index(drop = True)
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.head()
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df['Embarked'].isnull()]

#nerenin bo?? oldu??unu g??steriyor.
train_df.boxplot(column='Fare',by = 'Embarked')

plt.show()

#iki bilinmeyenin fare de??eri 80 grafikten 

#bakt??????m??z zaman C ye yak??n o zaman C olarak se??ebiliriz.
train_df['Embarked'] = train_df['Embarked'].fillna('C')

train_df[train_df['Embarked'].isnull()]
train_df[train_df['Fare'].isnull()]
np.mean(train_df[train_df['Pclass']==3]['Fare'])

#3. s??n??fta yolculuk edenler ortalama ne kadar ??demi??ler.
train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass']==3]['Fare']))
train_df[train_df['Fare'].isnull()]
list1= ['SibSp','Parch','Age','Fare','Survived']

sns.heatmap(train_df[list1].corr(), annot = True, fmt = '.2f')
g = sns.factorplot( x = 'SibSp', y = 'Survived', data = train_df, kind ='bar', size = 5)

g.set_ylabels('Survived Probability')

plt.show()

#yan??da karde??i e??i yak??n?? olanlar??n ya??ama oranlar??.
g = sns.factorplot( x = 'Parch', y = 'Survived', data = train_df, kind ='bar', size = 5)

g.set_ylabels('Survived Probability')

plt.show()



# bir yolcunun sahip oldu??u anne baba ??ocuk say??s??

# siyah ??izgi std sapmay?? g??sterir.
g = sns.factorplot( x = 'Pclass', y = 'Survived', data = train_df, kind ='bar', size = 5)

g.set_ylabels('Survived Probability')

plt.show()

# Yolcu seyahat s??n??flar??yla hayatta kalma olas??l??????
g = sns.FacetGrid(train_df, col = 'Survived')

g.map(sns.distplot,'Age', bins = 25)

plt.show()

# ??ocuklar??n hayatta kald??????n?? g??r??yoruz. 

#Ya??l??lar??n kurtulma oran??da y??ksek.

#20-25 ya?? aras??ndakilerin ??lme oran?? y??ksek.

#yolcular??n ya?? aral?????? 15-35 aras??nda fazla.
g = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass')

g.map(plt.hist,'Age', bins = 25)

g.add_legend()

plt.show()

# pclass de??erine g??re ya??ama oranlar??n?? g??steriyor.
g = sns.FacetGrid(train_df, row = 'Embarked', size =2)

g.map(sns.pointplot,'Pclass','Survived','Sex')

g.add_legend()

plt.show()



# kad??nlar erkeklere g??re daha fazla hayatta kalm????t??r.

# erkeklerin c liman??nda hayatta kalma olas??l??klar?? daha y??ksek.
g = sns.FacetGrid(train_df, row = 'Embarked',col = 'Survived', size =2.3)

g.map(sns.barplot,'Sex','Fare')

g.add_legend()

plt.show()

#Daha ??ok para ??dedi??imiz zaman daha fazla hayatta kalma ihtimali oluyor.
train_df[train_df['Age'].isnull()]

#age bo?? olanlar?? ????kard??.
sns.factorplot(x = 'Sex', y= 'Age', data = train_df, kind = 'box')

plt.show()

# Median de??erleri kad??n ve erkeklerde hemen hemen ayn??

#onun i??in buradan tahmin i??in bilgi ????kmad??.
sns.factorplot(x = 'Sex', y= 'Age', hue = 'Pclass' , data = train_df, kind = 'box')

plt.show()

# en ya??l??lar 1. s??n??fta en gen??ler 3. s??n??fta kal??yor.
sns.factorplot(x = 'Parch', y= 'Age',  data = train_df, kind = 'box')

sns.factorplot(x = 'SibSp', y= 'Age',  data = train_df, kind = 'box')

plt.show()

train_df['Sex'] = [1 if i == 'male' else 0 for i in train_df['Sex']]

#sonradan ekledik.
sns.heatmap(train_df[['Age','Sex','SibSp','Parch','Pclass',]].corr(),annot = True)

plt.show()

#burada sex bilgisi yok ????nk?? string say??sal de??ere ??evirmemiz laz??m.

#??st sat??r?? sonradan yazd??k.
index_nan_age = list(train_df['Age'][train_df['Age'].isnull()].index)

#bo?? olan nan lar?? buldu.

for i in index_nan_age:

    age_pred = train_df['Age'][((train_df['SibSp'] == train_df.iloc[i]['SibSp'])&(train_df['Parch'] == train_df.iloc[i]['Parch'])&(train_df['Pclass'] == train_df.iloc[i]['Pclass']))].median()

#hala daha nan olanlar var. onun i??in bu kodlar?? yazd??k.

    age_med = train_df['Age'].median()

    if not np.isnan(age_pred):

        train_df['Age'].iloc[i]=age_pred

    else:

        train_df['Age'].iloc[i]=age_med
train_df['Name'].head(10)

#ilk 10 yolcunun ismini yazar.
name = train_df['Name']

train_df['Title']=[i.split('.')[0].split(',')[-1].strip() for i in name]

#Mr ve Miss Mrs Master olarak isimlerin ??n??n?? al??yor.
train_df['Title'].head(10)
sns.countplot(x='Title', data =train_df)

plt.xticks(rotation = 60)

plt.show()

#hangisinden ka?? tane var g??rmemize yarar.
# Kategorik hale getiriyoruz.

train_df['Title']=train_df['Title'].replace(['Lady','the Countness','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'other')

train_df['Title'] = [0 if i == 'Master' else 1 if i == 'Miss' or i == 'Ms' or i == 'Mlle' or i == 'Mrs' else 2 if i == 'Mr' else 3 for i in train_df['Title']]

train_df['Title'].head(20)
sns.countplot(x='Title', data =train_df)

plt.xticks(rotation = 60)

plt.show()

#hangisinden ka?? tane var g??rmemize yarar.
g = sns.factorplot(x = 'Title', y= 'Survived', data = train_df, kind = 'bar')

g.set_xticklabels(['Master','Mrs','Mr','Other'])

g.set_ylabels('Survival Probability')

plt.show()
train_df.drop(labels = ['Name'], axis =1, inplace = True)

# verimizden name i ????kard??k onun yerine title geldi.

train_df.head()
train_df = pd.get_dummies(train_df, columns = ['Title'])

train_df.head()

#title 0-1-2-3 diye yeni ba??l??k olu??turduk.
train_df.head()
train_df['Fsize'] =train_df['SibSp'] +train_df['Parch'] + 1 

# 1 eklememizin sebebi ikisininde 0 olmas?? durumunda ailenin 0 ki??iden olu??mas??n?? engellemek i??in.
train_df.head()
g = sns.factorplot(x = 'Fsize', y= 'Survived', data = train_df, kind = 'bar')

g.set_ylabels('Survival')

plt.show()



# faily size 4 e kadar hayatta kalma oran?? art??yor. 
train_df['Family Size'] = [1 if i<5 else 0 for i in train_df['Fsize']]

# famly size 5 den k??????kse 1 5 den b??y??kse 0 yapt??k.
train_df.head(10)
sns.countplot(x='Family Size', data = train_df)

plt.show()

# ka?? tane 0 ka?? tane 1 oldu??unu g??rmemizi sa??l??yor.
g = sns.factorplot(x = 'Family Size', y= 'Survived', data = train_df, kind = 'bar')

g.set_ylabels('Survival')

plt.show()

# Hayatta kalmayla ili??kisine bak??yoruz.

# b??y??k ailelerin hayatta kalma ihtimalli daha d??????k. K??????k ailelerin hayatta kalma oran?? daha y??ksek.
# family_size 0 ve 1 olarak ikiye b??lelim.

train_df = pd.get_dummies(train_df, columns = ['Family Size'])

train_df.head()
sns.countplot(x = 'Embarked', data =train_df)

plt.show()

# S liman??ndan C liman??ndan ve Q liman??ndan binen ka?? ki??i oldu??unu g??steriyor.
train_df = pd.get_dummies(train_df, columns = ['Embarked'])

train_df.head()

# Emabark?? ortadan kald??rarak onun yerine 3 tane kategori ekledik.
train_df ['Ticket'].head(20)

# ba??taki yaz??lar?? sondaki yaz??lardan ay??rmam??z gerekiyor.
a =  'A/5. 2151'

a.replace('.','').replace('/','').strip().split(' ')[0]# noktayla bo??luk(hi??bir??ey) yer de??i??tiriyor.

#ikincide / gidiyor.

#strip ekstra bo??luk varsa o ortadan gidiyor.

#split dedi??imizde bo??lu??a g??re ay??r??yor.

# daha sonra A5 i alabilmek i??in bu listenin 0. eleman??n?? al diyoruz.
tickets = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace('.','').replace('/','').strip().split(' ')[0])

    else: 

        tickets.append('x')# ba????nda hi??bir ??ey yoksa x yaz??s??n?? koyuyoruz.

train_df['Ticket'] = tickets

        
train_df['Ticket'].head(20)
train_df = pd.get_dummies(train_df, columns = ['Ticket'], prefix = 'T')# Ticket yazmas?? yerine T yazmas??n?? sa??l??yor.

train_df.head(10)

# herbir ticket?? ayr?? s??tun olarak ekliyoruz.
sns.countplot(x ='Pclass', data =train_df)
train_df['Pclass'] = train_df['Pclass'].astype('category')#??nce categorical yap??yoruz.

train_df = pd.get_dummies(train_df, columns = ['Pclass'])

train_df.head()

#Pclass ortadan kalkt?? onun yerine Pclass1, Pclass2, Pclass 3 oldu.
train_df['Sex'] =train_df['Sex'].astype('category')

train_df =pd.get_dummies(train_df, columns = ['Sex'])

train_df.head()
train_df.drop(labels = ['PassengerId','Cabin'], axis =1, inplace =True)

train_df.columns

#passenger??d ve cabin ba??l??klar?? gitti.
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
test = train_df[train_df_len:]

test.drop(labels =['Survived'], axis =1, inplace =True)

test.head()
train = train_df[:train_df_len]

X_train = train.drop(labels ='Survived', axis =1)#X_train de survived olmayacak

y_train = train['Survived'] # y_train de survived olacak.

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.33, random_state = 42)

print('X_train', len(X_train))

print('X_test', len(X_test))

print('y_train', len(y_train))

print('y_test', len(y_test))

print('test', len(test))

# train i??in 590 test i??in 291 data var ayr??ca hi?? dokunulmama???? 418 tane data var.
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100,2) #y??zde olarak vermesi i??in 100 le ??arpt??k. ,2 demek virg??lden sonra 2 hane g??ster.

acc_log_test = round(logreg.score(X_test, y_test)*100,2)

print('Training Accuracy: %{}'.format(acc_log_train))

print('Testing Accuracy: %{}'.format(acc_log_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state=random_state), 

             SVC(random_state=random_state),

             RandomForestClassifier (random_state=random_state),

             LogisticRegression(random_state=random_state),

             KNeighborsClassifier()]

dt_param_grid = {'min_samples_split': range(10,500,20),

                'max_depth': range(1,20,2)}

svc_param_grid = {'kernel': ['rbf'],

                'gamma': [0.001,0.01,0.1,1],

                'C': [1,10,50,100,200,300,1000]}

rf_param_grid = {'max_features': [1,3,10],

                'min_samples_split': [2,3,10],

                'min_samples_leaf': [1,3,10],

                'bootstrap': [False],

                'n_estimators': [100,300],

                'criterion' : ['gini']}

logreg_param_grid = {'C': np.logspace(-3,3,7),

                    'penalty': ['l1','l2']}



knn_param_grid = {'n_neighbors': np.linspace(1,19,10, dtype = int).tolist(),

                 'weights': ['uniform','distance'],

                 'metric': ['euclidean','manhattan']}



classifier_param =[dt_param_grid,svc_param_grid,rf_param_grid, 

                   logreg_param_grid, knn_param_grid]



# hyper param. tuning; mesela knn algoritmas??nda n de??erinin belirlenmesi

#veya ecludian m?? menhattan m?? onun belirlenmesi i?? i??e for d??ng??leri

#yaz??larak accuracy de??erlerine bak??larak karar verilebilir.

#Grid search ise; grid haline getirerek grid search yap??yor. For

# d??ng??s??yle yapaca????m??z?? yap??yor.

cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid = classifier_param[i], cv= StratifiedKFold(n_splits = 10), scoring = 'accuracy', n_jobs= -1, verbose = 1)  

    clf.fit(X_train, y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])



#classifier listesini s??rayla se??erek dola??acak. param gridi tek tek dola??acak.

#score accuracy e bakacaz. n_jobs -1 yapt??????m??zda h??zl?? ??al??????yor.

#verbose canl?? olarak g??sterecek.



cv_results = pd.DataFrame({'Cross Validation Means': cv_result, 'ML Models':['DecisionTreeClassifier', 

             'SVM', 'RandomForestClassifier' , 'LogisticRegression','KNeighborsClassifier']})

g = sns.barplot('Cross Validation Means', 'ML Models', data = cv_results)

g.set_xlabel('Mean Accuracy')

g.set_title('Cross Validation Scores')

# classifierin ba??ar??lar??n?? g??steriyor.
votingC = VotingClassifier(estimators = [('dt', best_estimators[0]),

                                        ('rfc', best_estimators[2]),

                                        ('lr', best_estimators[3])],

                                          voting = 'soft', n_jobs= -1)

# voting soft yaparsak tek sonu?? vermez her classifiere g??re ya??ama yada ??lme oran??n?? hesaplar ona g??re sonu?? verir.

#voting hard yaparsak classifierleri kar????la??t??r??r. ??ok olan sonuca g??re tek bir oran verir.

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test), y_test))
test_survived = pd.Series(votingC.predict(test), name ='Survived').astype(int)

results = pd.concat([test_PassengerId, test_survived], axis =1)

results.to_csv('titanic_csv' , index = False)#csv format??nda d????ar??ya aktaraca????z.