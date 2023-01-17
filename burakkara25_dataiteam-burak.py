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

    outlier_indices = Counter(outlier_indices)#counter kaç tane olduğunu söyler.

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

#nerenin boş olduğunu gösteriyor.
train_df.boxplot(column='Fare',by = 'Embarked')

plt.show()

#iki bilinmeyenin fare değeri 80 grafikten 

#baktığımız zaman C ye yakın o zaman C olarak seçebiliriz.
train_df['Embarked'] = train_df['Embarked'].fillna('C')

train_df[train_df['Embarked'].isnull()]
train_df[train_df['Fare'].isnull()]
np.mean(train_df[train_df['Pclass']==3]['Fare'])

#3. sınıfta yolculuk edenler ortalama ne kadar ödemişler.
train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass']==3]['Fare']))
train_df[train_df['Fare'].isnull()]
list1= ['SibSp','Parch','Age','Fare','Survived']

sns.heatmap(train_df[list1].corr(), annot = True, fmt = '.2f')
g = sns.factorplot( x = 'SibSp', y = 'Survived', data = train_df, kind ='bar', size = 5)

g.set_ylabels('Survived Probability')

plt.show()

#yanıda kardeşi eşi yakını olanların yaşama oranları.
g = sns.factorplot( x = 'Parch', y = 'Survived', data = train_df, kind ='bar', size = 5)

g.set_ylabels('Survived Probability')

plt.show()



# bir yolcunun sahip olduğu anne baba çocuk sayısı

# siyah çizgi std sapmayı gösterir.
g = sns.factorplot( x = 'Pclass', y = 'Survived', data = train_df, kind ='bar', size = 5)

g.set_ylabels('Survived Probability')

plt.show()

# Yolcu seyahat sınıflarıyla hayatta kalma olasılığı
g = sns.FacetGrid(train_df, col = 'Survived')

g.map(sns.distplot,'Age', bins = 25)

plt.show()

# çocukların hayatta kaldığını görüyoruz. 

#Yaşlıların kurtulma oranıda yüksek.

#20-25 yaş arasındakilerin ölme oranı yüksek.

#yolcuların yaş aralığı 15-35 arasında fazla.
g = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass')

g.map(plt.hist,'Age', bins = 25)

g.add_legend()

plt.show()

# pclass değerine göre yaşama oranlarını gösteriyor.
g = sns.FacetGrid(train_df, row = 'Embarked', size =2)

g.map(sns.pointplot,'Pclass','Survived','Sex')

g.add_legend()

plt.show()



# kadınlar erkeklere göre daha fazla hayatta kalmıştır.

# erkeklerin c limanında hayatta kalma olasılıkları daha yüksek.
g = sns.FacetGrid(train_df, row = 'Embarked',col = 'Survived', size =2.3)

g.map(sns.barplot,'Sex','Fare')

g.add_legend()

plt.show()

#Daha çok para ödediğimiz zaman daha fazla hayatta kalma ihtimali oluyor.
train_df[train_df['Age'].isnull()]

#age boş olanları çıkardı.
sns.factorplot(x = 'Sex', y= 'Age', data = train_df, kind = 'box')

plt.show()

# Median değerleri kadın ve erkeklerde hemen hemen aynı

#onun için buradan tahmin için bilgi çıkmadı.
sns.factorplot(x = 'Sex', y= 'Age', hue = 'Pclass' , data = train_df, kind = 'box')

plt.show()

# en yaşlılar 1. sınıfta en gençler 3. sınıfta kalıyor.
sns.factorplot(x = 'Parch', y= 'Age',  data = train_df, kind = 'box')

sns.factorplot(x = 'SibSp', y= 'Age',  data = train_df, kind = 'box')

plt.show()

train_df['Sex'] = [1 if i == 'male' else 0 for i in train_df['Sex']]

#sonradan ekledik.
sns.heatmap(train_df[['Age','Sex','SibSp','Parch','Pclass',]].corr(),annot = True)

plt.show()

#burada sex bilgisi yok çünkü string sayısal değere çevirmemiz lazım.

#üst satırı sonradan yazdık.
index_nan_age = list(train_df['Age'][train_df['Age'].isnull()].index)

#boş olan nan ları buldu.

for i in index_nan_age:

    age_pred = train_df['Age'][((train_df['SibSp'] == train_df.iloc[i]['SibSp'])&(train_df['Parch'] == train_df.iloc[i]['Parch'])&(train_df['Pclass'] == train_df.iloc[i]['Pclass']))].median()

#hala daha nan olanlar var. onun için bu kodları yazdık.

    age_med = train_df['Age'].median()

    if not np.isnan(age_pred):

        train_df['Age'].iloc[i]=age_pred

    else:

        train_df['Age'].iloc[i]=age_med
train_df['Name'].head(10)

#ilk 10 yolcunun ismini yazar.
name = train_df['Name']

train_df['Title']=[i.split('.')[0].split(',')[-1].strip() for i in name]

#Mr ve Miss Mrs Master olarak isimlerin önünü alıyor.
train_df['Title'].head(10)
sns.countplot(x='Title', data =train_df)

plt.xticks(rotation = 60)

plt.show()

#hangisinden kaç tane var görmemize yarar.
# Kategorik hale getiriyoruz.

train_df['Title']=train_df['Title'].replace(['Lady','the Countness','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'other')

train_df['Title'] = [0 if i == 'Master' else 1 if i == 'Miss' or i == 'Ms' or i == 'Mlle' or i == 'Mrs' else 2 if i == 'Mr' else 3 for i in train_df['Title']]

train_df['Title'].head(20)
sns.countplot(x='Title', data =train_df)

plt.xticks(rotation = 60)

plt.show()

#hangisinden kaç tane var görmemize yarar.
g = sns.factorplot(x = 'Title', y= 'Survived', data = train_df, kind = 'bar')

g.set_xticklabels(['Master','Mrs','Mr','Other'])

g.set_ylabels('Survival Probability')

plt.show()
train_df.drop(labels = ['Name'], axis =1, inplace = True)

# verimizden name i çıkardık onun yerine title geldi.

train_df.head()
train_df = pd.get_dummies(train_df, columns = ['Title'])

train_df.head()

#title 0-1-2-3 diye yeni başlık oluşturduk.
train_df.head()
train_df['Fsize'] =train_df['SibSp'] +train_df['Parch'] + 1 

# 1 eklememizin sebebi ikisininde 0 olması durumunda ailenin 0 kişiden oluşmasını engellemek için.
train_df.head()
g = sns.factorplot(x = 'Fsize', y= 'Survived', data = train_df, kind = 'bar')

g.set_ylabels('Survival')

plt.show()



# faily size 4 e kadar hayatta kalma oranı artıyor. 
train_df['Family Size'] = [1 if i<5 else 0 for i in train_df['Fsize']]

# famly size 5 den küçükse 1 5 den büyükse 0 yaptık.
train_df.head(10)
sns.countplot(x='Family Size', data = train_df)

plt.show()

# kaç tane 0 kaç tane 1 olduğunu görmemizi sağlıyor.
g = sns.factorplot(x = 'Family Size', y= 'Survived', data = train_df, kind = 'bar')

g.set_ylabels('Survival')

plt.show()

# Hayatta kalmayla ilişkisine bakıyoruz.

# büyük ailelerin hayatta kalma ihtimalli daha düşük. Küçük ailelerin hayatta kalma oranı daha yüksek.
# family_size 0 ve 1 olarak ikiye bölelim.

train_df = pd.get_dummies(train_df, columns = ['Family Size'])

train_df.head()
sns.countplot(x = 'Embarked', data =train_df)

plt.show()

# S limanından C limanından ve Q limanından binen kaç kişi olduğunu gösteriyor.
train_df = pd.get_dummies(train_df, columns = ['Embarked'])

train_df.head()

# Emabarkı ortadan kaldırarak onun yerine 3 tane kategori ekledik.
train_df ['Ticket'].head(20)

# baştaki yazıları sondaki yazılardan ayırmamız gerekiyor.
a =  'A/5. 2151'

a.replace('.','').replace('/','').strip().split(' ')[0]# noktayla boşluk(hiçbirşey) yer değiştiriyor.

#ikincide / gidiyor.

#strip ekstra boşluk varsa o ortadan gidiyor.

#split dediğimizde boşluğa göre ayırıyor.

# daha sonra A5 i alabilmek için bu listenin 0. elemanını al diyoruz.
tickets = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace('.','').replace('/','').strip().split(' ')[0])

    else: 

        tickets.append('x')# başında hiçbir şey yoksa x yazısını koyuyoruz.

train_df['Ticket'] = tickets

        
train_df['Ticket'].head(20)
train_df = pd.get_dummies(train_df, columns = ['Ticket'], prefix = 'T')# Ticket yazması yerine T yazmasını sağlıyor.

train_df.head(10)

# herbir ticketı ayrı sütun olarak ekliyoruz.
sns.countplot(x ='Pclass', data =train_df)
train_df['Pclass'] = train_df['Pclass'].astype('category')#önce categorical yapıyoruz.

train_df = pd.get_dummies(train_df, columns = ['Pclass'])

train_df.head()

#Pclass ortadan kalktı onun yerine Pclass1, Pclass2, Pclass 3 oldu.
train_df['Sex'] =train_df['Sex'].astype('category')

train_df =pd.get_dummies(train_df, columns = ['Sex'])

train_df.head()
train_df.drop(labels = ['PassengerId','Cabin'], axis =1, inplace =True)

train_df.columns

#passengerİd ve cabin başlıkları gitti.
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

# train için 590 test için 291 data var ayrıca hiç dokunulmamaış 418 tane data var.
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100,2) #yüzde olarak vermesi için 100 le çarptık. ,2 demek virgülden sonra 2 hane göster.

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



# hyper param. tuning; mesela knn algoritmasında n değerinin belirlenmesi

#veya ecludian mı menhattan mı onun belirlenmesi iç içe for döngüleri

#yazılarak accuracy değerlerine bakılarak karar verilebilir.

#Grid search ise; grid haline getirerek grid search yapıyor. For

# döngüsüyle yapacağımızı yapıyor.

cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid = classifier_param[i], cv= StratifiedKFold(n_splits = 10), scoring = 'accuracy', n_jobs= -1, verbose = 1)  

    clf.fit(X_train, y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])



#classifier listesini sırayla seçerek dolaşacak. param gridi tek tek dolaşacak.

#score accuracy e bakacaz. n_jobs -1 yaptığımızda hızlı çalışıyor.

#verbose canlı olarak gösterecek.



cv_results = pd.DataFrame({'Cross Validation Means': cv_result, 'ML Models':['DecisionTreeClassifier', 

             'SVM', 'RandomForestClassifier' , 'LogisticRegression','KNeighborsClassifier']})

g = sns.barplot('Cross Validation Means', 'ML Models', data = cv_results)

g.set_xlabel('Mean Accuracy')

g.set_title('Cross Validation Scores')

# classifierin başarılarını gösteriyor.
votingC = VotingClassifier(estimators = [('dt', best_estimators[0]),

                                        ('rfc', best_estimators[2]),

                                        ('lr', best_estimators[3])],

                                          voting = 'soft', n_jobs= -1)

# voting soft yaparsak tek sonuç vermez her classifiere göre yaşama yada ölme oranını hesaplar ona göre sonuç verir.

#voting hard yaparsak classifierleri karşılaştırır. çok olan sonuca göre tek bir oran verir.

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test), y_test))
test_survived = pd.Series(votingC.predict(test), name ='Survived').astype(int)

results = pd.concat([test_PassengerId, test_survived], axis =1)

results.to_csv('titanic_csv' , index = False)#csv formatında dışarıya aktaracağız.