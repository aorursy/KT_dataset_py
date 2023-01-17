import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(2018)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_validate

import xgboost as xgb

import matplotlib.pyplot as plt
%matplotlib inline

#wczytujemy nasze dane
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

#sklejamy nasze oba zbiory w jeden
#POCO? za chwilę się dowiesz.
df_all = pd.concat([df_train, df_test])
#możemy zauważyć ciekawostkę, NaN - puste.
#jesli mamy C23, C25 itp itd to zauważamy, że ziomek Mr. Mark Fortune ma żonę i najprawdopodobniej 4 dzieci (za stary na to żeby to byli jego rodzice)

df_train.sample(10)

#patrząc na te dane to już można coś wyciągnąć, ale dla modelu niekoniecznie to jest oczywiste dlategoo.. idziemy dalej z tematem
df_train.select_dtypes(include=[np.int, np.float]).head() #sprawdzamy dla pierwszych 5 wyników
#sprawdzamy jakie wartości mogą zostać przypisane do tej cechy?
df_all['Sex'].unique()
# zamieniamy tekstowe wartości na numeryczne
pd.factorize(df_all['Sex'])

#gdzie 0 - male, 1 - female (w indexie mamy labelki), żeby się tego pozbyć dajemy [0] - tylko pierwsza kolumna nas interesuje

#budujemy tu prosty benchmark, gdzie wyciągamy potrzebne dla nas cechy
# w tym poniższym przypadku są to liczby numeryczne
#olewamy PassengerID - bo to zwyczajnie nie potrzebne :D
#druga kolumna którą ignorujemy to "Survived" bo chcemy to zbadać.

# 1. ignorujemy niepotrzebne cechy (dodajemy do czarnej listy)
def get_feats(df):
    feats = df.select_dtypes(include=[np.int]).columns.values
    black_list = ['PassengerId', 'Survived']
    
    return [feat for feat in feats if feat not in black_list]

# 2. zamieniamy tekstowe wartości na wartości numeryczne
def feature_engineering(df):
    df['sex_cat'] = pd.factorize( df['Sex'] )[0]
    df['embarked_cat'] = pd.factorize( df['Embarked'] )[0]
    return df

# 3. Tworzymy 4 modele 
def get_models():
    return [
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('et', ExtraTreesClassifier())
    ]
#chcemy zbudować funkcję, która pomoże nam zwizualizować wynik (modele)

def plot_result(model_name, result, ylim=(0,1.)):
    mean_train = np.round( np.mean(result['train_score']), 2)
    mean_test = np.round( np.mean(result['test_score']), 2)
    
    plt.title('{0}: cross validation\nmean-train-acc:{1}\nmean-test-acc:{2}'.format(model_name, mean_train, mean_test))
    plt.plot( result['train_score'], 'r-o', label="train" )
    plt.plot( result['test_score'], 'g-o', label="test" )
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel('# of fold')
    plt.ylim(*ylim)
    plt.show()
df = feature_engineering(df_train)
get_feats(df)

X = df_train[ get_feats(df_train) ].values
y = df_train[ 'Survived' ].values

for model_name, model in get_models():
    result = cross_validate(model, X, y, scoring='accuracy', cv=3)
    print(model_name)

    plot_result(model_name, result)


def make_prediction(df_train, df_test, model, output_file_name):
    train = feature_engineering(df_train)
    test = feature_engineering(df_test)

    feats = get_feats(train)

    X_train = train[feats].values
    y_train = train['Survived'].values
    X_test = test[feats]

    print(model)
    model.fit(X_train, y_train)

    test['Survived'] = model.predict(X_test)
    test[ ['PassengerId', 'Survived'] ].to_csv(output_file_name, index=False)


#Linear model

make_prediction(df_train, df_test, LogisticRegression(), 'linear.csv')


for max_depth in range(2, 10, 2):
    model = RandomForestClassifier(max_depth=max_depth, n_estimators = 15)
    result = cross_validate(model, X, y, scoring='accuracy', cv=3)
    plot_result("Devision Tree: max_depth=%s" % max_depth, result)
#Sprawdzamy teraz co jeden
for max_depth in range(3, 6, 1):
    model = RandomForestClassifier(max_depth=max_depth, n_estimators = 15)
    result = cross_validate(model, X, y, scoring='accuracy', cv=3)
    plot_result("Devision Tree: max_depth=%s" % max_depth, result)

model = RandomForestClassifier(max_depth=3, n_estimators=15, random_state=2018)
make_prediction(df_train,df_test, model, 'rf_md3_ne15.csv')
print(df_train.shape)
print(df_test.shape)
print(df_all.shape)
df_train['Name']
df_train['Name'].map(lambda x: x.split(',')[1].split('.')[0].lower().strip()).value_counts()
# sprawdzamy tytuły w danych testowych
df_test['Name'].map(lambda x: x.split(',')[1].split('.')[0].lower().strip()).value_counts()
df_all['Name'].map(lambda x: x.split(',')[1].split('.')[0].lower().strip()).value_counts()
# normalizujemy dane. te co się powtarzają czyli minimum 13 bo to jest 1% (ale 8 też wzięliśmy :)
# pozostałe dajemy do wartości "other" co stanowi troche ponad 1% danych bo 18szt.
popular_titles = ['mr', 'miss', 'mrs', 'master', 'dr', 'rev']
df_all['Name'].map(lambda x: x.split(',')[1].split('.')[0].lower().strip() ).map(lambda x: x if x in popular_titles else "other").value_counts()
#tworzymy funkcję żeby można było korzystać z niej w przyszłości bez powielania kodu

def feature_title(df):
    df['title'] = df['Name'].map(lambda x: x.split(',')[1].split('.')[0].lower().strip())
    popular_titles = ["mr", "mrs", "miss", "master", "dr", "rev"]
    df['title_norm'] = df['title'].map(lambda x: x if x in popular_titles else "other")
    df['title_norm_cat'] = pd.factorize( df['title_norm'])[0]
    
    return df
df = feature_title(df_all)
df['title_norm'].value_counts()
df_all['Cabin'].map(lambda x: 'missing' if str(x) == 'nan' else x[0]).value_counts()
df_all.apply(lambda x: x['Parch'] + x['SibSp'], axis=1).value_counts()
print(df_all.shape)
df_all.Age.describe()
# możemy zobaczyć średni wiek dla poszczególnych tytułów

df = feature_title(df_all)
df.groupby('title_norm')['Age'].agg([np.mean, np.median])
df = feature_title(df_all)
missing_ages = df.groupby('title_norm')['Age'].agg([np.mean, np.median]).to_dict()['median']
df['Age'] = df.apply( lambda x: x['Age'] if str(x['Age']) != 'nan' else missing_ages[x['title_norm']], axis =1 )
age_bins = [0, 1, 3, 5, 9, 15, 20, 40, 60, 100]
pd.cut(df["Age"], bins=age_bins).astype(object).value_counts()
print(df_all.shape)
df_all['Fare'].describe()
df_all['Fare'].hist(bins=100); #średnik usuwa linijke z opisem
np.log2(df_all['Fare'] + 1).hist(bins=100)
def feature_engineering(df):
    df['sex_cat'] = pd.factorize( df['Sex'] )[0]
    df['embarked_cat'] = pd.factorize( df['Embarked'] )[0]
    
    df = feature_title(df)
    
    df['cabin_norm'] = df['Cabin'].map(lambda x: 'missing' if str(x) == 'nan' else x[0] )
    df['cabin_norm_cat'] = pd.factorize( df['cabin_norm'] )[0]
    
    df['family_size'] = df.apply(lambda x: x['Parch'] + x['SibSp'], axis=1)
    df["single"] = df["family_size"].apply(lambda x: 1 if x == 0 else 0)

    df['fare_log'] = np.log2( df['Fare'] + 1 )
    
    missing_ages = df.groupby('title_norm')['Age'].agg([np.mean, np.median]).to_dict()['median']
    df['Age'] = df.apply( lambda x: x['Age'] if str(x['Age']) != 'nan' else missing_ages[x['title_norm']], axis=1 )
    
    age_bins = [0, 1, 3, 5, 9, 15, 20, 40, 60, 100]
    df['age_bin'] = pd.factorize( pd.cut(df["Age"], bins=age_bins).astype(object) )[0]

    return df
df = feature_engineering(df_all)
feats = get_feats(df)
print(feats)

train = df[ ~df.Survived.isnull() ]

X = train[ feats ].values
y = train[ 'Survived' ].values

for model_name, model in get_models():
    result = cross_validate(model, X, y, scoring='accuracy', cv=3)
    print(model_name)

    plot_result(model_name, result)
for max_depth in range(2, 6, 1):
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=20, min_samples_leaf=8, random_state=2018)
    result = cross_validate(model, X, y, scoring='accuracy', cv=3)
    plot_result("Random Forest: max_depth=%s" % max_depth, result, ylim=(0.7, 0.9))
df_train, df_test = df_all[ ~df.Survived.isnull() ].copy(), df[ df.Survived.isnull() ].copy()

model = model = RandomForestClassifier(max_depth=4, n_estimators=20, min_samples_leaf=8, random_state=2018)
make_prediction(df_train, df_test, model, 'rf_md4_ne20_sl8_fe.csv')
