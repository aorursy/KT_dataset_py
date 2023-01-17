import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time,sleep


import nltk
from nltk import tokenize
from string import punctuation
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from unidecode import unidecode

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate,KFold,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scipy.stats import randint
from numpy.random import uniform
# pandas options
pd.options.display.max_columns = 30
pd.options.display.float_format = '{:.2f}'.format

# seaborn options
sns.set(style="darkgrid")

import warnings
warnings.filterwarnings("ignore")

SEED = 42
def treat_words(df,
      col,
      language='english',
      inplace=False,
      tokenizer = tokenize.WordPunctTokenizer(),
      decode = True,
      stemmer = None,
      lower = True,
      remove_words = [],

    ):
    """
    Description:
    ----------------
        Receives a dataframe and the column name. Eliminates
        stopwords for each row of that column and apply stemmer.
        After that, it regroups and returns a list.
        
        tokenizer = tokenize.WordPunctTokenizer()
                    tokenize.WhitespaceTokenizer()
                    
        stemmer =    PorterStemmer()
                     SnowballStemmer()
                     LancasterStemmer()
                     nltk.RSLPStemmer() # in portuguese
    """
    
    
    pnct = [string for string in punctuation] # from string import punctuation 
    wrds = nltk.corpus.stopwords.words(language)
    unwanted_words = pnct + wrds + remove_words

    processed_text = list()

    for element in tqdm(df[col]):

        # starts a new list
        new_text = list()

        # starts a list with the words of the non precessed text
        text_old = tokenizer.tokenize(element)

        # check each word
        for wrd in text_old:

            # if the word are not in the unwanted words list
            # add to the new list
            if wrd.lower() not in unwanted_words:
                
                new_wrd = wrd
                
                if decode: new_wrd = unidecode(new_wrd)
                if stemmer: new_wrd = stemmer.stem(new_wrd)
                if lower: new_wrd = new_wrd.lower()
                    
                if new_wrd not in remove_words:
                    new_text.append(new_wrd)

        processed_text.append(' '.join(new_text))

    if inplace:
        df[col] = processed_text
    else:
        return processed_text

def list_words_of_class(df,
                          col,
                          language='english',
                          inplace=False,
                          tokenizer = tokenize.WordPunctTokenizer(),
                          decode = True,
                          stemmer = None,
                          lower = True,
                          remove_words = []
                         ):
    """
    Description:
    ----------------
    
        Receives a dataframe and the column name. Eliminates
        stopwords for each row of that column, apply stemmer
        and returns a list of all the words.
    
    """
    
    lista = treat_words(
        df,col = col,language = language,
        tokenizer=tokenizer,decode=decode,
        stemmer=stemmer,lower=lower,
        remove_words = remove_words
        )
    
    words_list = []
    for string in lista:
        words_list += tokenizer.tokenize(string)
        
        
    return words_list
def get_frequency(df,
                  col,
                  language='english',
                  inplace=False,
                  tokenizer = tokenize.WordPunctTokenizer(),
                  decode = True,
                  stemmer = None,
                  lower = True,
                  remove_words = []
                 ):
    
    list_of_words = list_words_of_class(
              df,
              col = col,
              decode = decode,
              stemmer = stemmer,
              lower = lower,
              remove_words = remove_words
      )
    
    freq = nltk.FreqDist(list_of_words)
    
    df_freq = pd.DataFrame({
        'word': list(freq.keys()),
        'frequency': list(freq.values())
    }).sort_values(by='frequency',ascending=False)

    n_words = df_freq['frequency'].sum()

    df_freq['prop'] = 100*df_freq['frequency']/n_words

    return df_freq
def common_best_words(df,col,n_common = 10,tol_frac = 0.8,n_jobs = 1):
    list_to_remove = []

    for i in range(0,n_jobs):
        print('[info] Most common words in not survived')
        sleep(0.5)
        df_dead = get_frequency(
                  df.query('Survived == 0'),
                  col = col,
                  decode = False,
                  stemmer = False,
                  lower = False,
                  remove_words = list_to_remove )

        print('[info] Most common words in survived')
        sleep(0.5)
        df_surv = get_frequency(
                  df.query('Survived == 1'),
                  col = col,
                  decode = False,
                  stemmer = False,
                  lower = False,
                  remove_words = list_to_remove )


        words_dead = df_dead.nlargest(n_common, 'frequency')

        list_dead = list(words_dead['word'].values)

        words_surv = df_surv.nlargest(n_common, 'frequency')

        list_surv = list(words_surv['word'].values)

        for word in list(set(list_dead).intersection(list_surv)):
            prop_dead = words_dead[words_dead['word'] == word]['prop'].values[0]
            prop_surv = words_surv[words_surv['word'] == word]['prop'].values[0]
            ratio = min([prop_dead,prop_surv])/max([prop_dead,prop_surv])

            if ratio > tol_frac:
                list_to_remove.append(word)
        
        return list_to_remove
def just_keep_the_words(df,
      col,
      keep_words = [],
      tokenizer = tokenize.WordPunctTokenizer()
    ):
    """
    Description:
    ----------------
        Removes all words that is not in `keep_words`
    """
    
    processed_text = list()

    # para cada avaliação
    for element in tqdm(df[col]):

        # starts a new list
        new_text = list()

        # starts a list with the words of the non precessed text
        text_old = tokenizer.tokenize(element)

        for wrd in text_old:

            if wrd in keep_words: new_text.append(wrd)

        processed_text.append(' '.join(new_text))

    return processed_text

class Classifier:
    '''
    Description
    -----------------
    
    Class to approach classification algorithm
    
    
    Example
    -----------------
        classifier = Classifier(
                 algorithm = ChooseTheAlgorith,
                 hyperparameters_range = {
                    'hyperparameter_1': [1,2,3],
                    'hyperparameter_2': [4,5,6],
                    'hyperparameter_3': [7,8,9]
                 }
             )

        # Looking for best model
        classifier.grid_search_fit(X,y,n_splits=10)
        #dt.grid_search_results.head(3)

        # Prediction Form 1
        par = classifier.best_model_params
        dt.fit(X_trn,y_trn,params = par)
        y_pred = classifier.predict(X_tst)
        print(accuracy_score(y_tst, y_pred))

        # Prediction Form 2
        classifier.fit(X_trn,y_trn,params = 'best_model')
        y_pred = classifier.predict(X_tst)
        print(accuracy_score(y_tst, y_pred))

        # Prediction Form 3
        classifier.fit(X_trn,y_trn,min_samples_split = 5,max_depth=4)
        y_pred = classifier.predict(X_tst)
        print(accuracy_score(y_tst, y_pred))
    '''
    def __init__(self,algorithm, hyperparameters_range={},random_state=42):
        
        self.algorithm = algorithm
        self.hyperparameters_range = hyperparameters_range
        self.random_state = random_state
        self.grid_search_cv = None
        self.grid_search_results = None
        self.hyperparameters = self.__get_hyperparameters()
        self.best_model = None
        self.best_model_params = None
        self.fitted_model = None
        
    def grid_search_fit(self,X,y,verbose=0,n_splits=10,shuffle=True,scoring='accuracy'):
        
        self.grid_search_cv = GridSearchCV(
            self.algorithm(),
            self.hyperparameters_range,
            cv = KFold(n_splits = n_splits, shuffle=shuffle, random_state=self.random_state),
            scoring=scoring,
            verbose=verbose
        )
        
        self.grid_search_cv.fit(X, y)
        
        col = list(map(lambda par: 'param_'+str(par),self.hyperparameters))+[
                'mean_fit_time',
                'mean_test_score',
                'std_test_score',
                'params'
              ]
        
        results = pd.DataFrame(self.grid_search_cv.cv_results_)
        
        self.grid_search_results = results[col].sort_values(
                    ['mean_test_score','mean_fit_time'],
                    ascending=[False,True]
                ).reset_index(drop=True)
        
        self.best_model = self.grid_search_cv.best_estimator_
        
        self.best_model_params = self.best_model.get_params()
    
    def best_model_cv_score(self,X,y,parameter='test_score',verbose=0,n_splits=10,shuffle=True,scoring='accuracy'):
        if self.best_model != None:
            cv_results = cross_validate(
                self.best_model,
                X = X,
                y = y,
                cv=KFold(n_splits = 10,shuffle=True,random_state=self.random_state)
            )
            return {
                parameter+'_mean': cv_results[parameter].mean(),
                parameter+'_std': cv_results[parameter].std()
            }
        
    def fit(self,X,y,params=None,**kwargs):
        model = None
        if len(kwargs) == 0 and params == 'best_model' and self.best_model != None:
            model = self.best_model
            
        elif type(params) == dict and len(params) > 0:
            model = self.algorithm(**params)
            
        elif len(kwargs) >= 0 and params==None:
            model = self.algorithm(**kwargs)
            
        else:
            print('[Error]')
            
        if model != None:
            model.fit(X,y)
            
        self.fitted_model = model
            
    def predict(self,X):
        if self.fitted_model != None:
            return self.fitted_model.predict(X)
        else:
            print('[Error]')
            return np.array([])
            
    def predict_score(self,X_tst,y_tst,score=accuracy_score):
        if self.fitted_model != None:
            y_pred = self.predict(X_tst)
            return score(y_tst, y_pred)
        else:
            print('[Error]')
            return np.array([])
        
    def hyperparameter_info(self,hyperpar):
        str_ = 'param_'+hyperpar
        return self.grid_search_results[
                [str_,'mean_fit_time','mean_test_score']
            ].groupby(str_).agg(['mean','std'])
        
    def __get_hyperparameters(self):
        return [hp for hp in self.hyperparameters_range]
def cont_class_limits(lis_df,n_class):
    ampl = lis_df.quantile(1.0)-lis_df.quantile(0.0)
    ampl_class = ampl/n_class 
    limits = [[i*ampl_class,(i+1)*ampl_class] for i in range(n_class)]
    return limits

def cont_classification(lis_df,limits):
    list_res = []
    n_class = len(limits)
    for elem in lis_df:
        for ind in range(n_class-1):
            if elem >= limits[ind][0] and elem < limits[ind][1]:
                list_res.append(ind+1)
            
        if elem >= limits[-1][0]: list_res.append(n_class)
            
    return list_res

df_trn = pd.read_csv('data/train.csv')
df_tst = pd.read_csv('data/test.csv')

df = pd.concat([df_trn,df_tst])

df_trn = df_trn.drop(columns=['PassengerId'])
df_tst = df_tst.drop(columns=['PassengerId'])
df_tst.info()
sns.barplot(x='Pclass', y="Survived", data=df_trn)
treat_words(df_trn,col = 'Name',inplace=True)
treat_words(df_tst,col = 'Name',inplace=True)
%matplotlib inline

from wordcloud import WordCloud
import matplotlib.pyplot as plt

all_words = ' '.join(list(df_trn['Name']))

word_cloud = WordCloud().generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
common_best_words(df_trn,col='Name',n_common = 10,tol_frac = 0.5,n_jobs = 1)
df_comm = get_frequency(df_trn,col = 'Name',remove_words=['("','")','master', 'william']).reset_index(drop=True)
surv_prob = [ df_trn['Survived'][df_trn['Name'].str.contains(row['word'])].mean() for index, row in df_comm.iterrows()]
df_comm['survival_prob (%)'] = 100*np.array(surv_prob)
print('Survival Frequency related to words in Name')
df_comm.head(10)
df_comm_surv = get_frequency(df_trn[df_trn['Survived']==1],col = 'Name',remove_words=['("','")']).reset_index(drop=True)
sleep(0.5)
print('Most frequent words within those who survived')
df_comm_surv.head(10)
df_comm_dead = get_frequency(df_trn[df_trn['Survived']==0],col = 'Name',remove_words=['("','")']).reset_index(drop=True)
sleep(0.5)
print("Most frequent words within those that did not survive")
df_comm_dead.head(10)
min_occurrences = 2
df_comm = get_frequency(df,col = 'Name',
                        remove_words=['("','")','john','henry', 'william','h','j','jr']
                       ).reset_index(drop=True)
words_to_keep = list(df_comm[df_comm['frequency'] > min_occurrences]['word'])

df_trn['Name'] = just_keep_the_words(df_trn,
                    col = 'Name',
                    keep_words = words_to_keep 
                   )

df_tst['Name'] = just_keep_the_words(df_tst,
                    col = 'Name',
                    keep_words = words_to_keep 
                   )
vectorize = CountVectorizer(lowercase=True,max_features = 4)
vectorize.fit(df_trn['Name'])
bag_of_words = vectorize.transform(df_trn['Name'])

X = pd.DataFrame(vectorize.fit_transform(df_trn['Name']).toarray(),
             columns=list(map(lambda word: 'Name_'+word,vectorize.get_feature_names()))
            )
y = df_trn['Survived']

from sklearn.model_selection import train_test_split
X_trn,X_tst,y_trn,y_tst = train_test_split(
    X,
    y,
    test_size = 0.25,
    random_state=42
)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=100)
classifier.fit(X_trn,y_trn)
accuracy = classifier.score(X_tst,y_tst)
print('Accuracy = %.3f%%' % (100*accuracy))
df_trn = pd.concat([
            df_trn
            ,
            pd.DataFrame(vectorize.fit_transform(df_trn['Name']).toarray(),
                  columns=list(map(lambda word: 'Name_'+word,vectorize.get_feature_names()))
                )
        ],axis=1).drop(columns=['Name'])

df_tst = pd.concat([
            df_tst
            ,
            pd.DataFrame(vectorize.fit_transform(df_tst['Name']).toarray(),
                  columns=list(map(lambda word: 'Name_'+word,vectorize.get_feature_names()))
                )
        ],axis=1).drop(columns=['Name'])
from sklearn.preprocessing import LabelEncoder
Sex_Encoder = LabelEncoder()

df_trn['Sex'] = Sex_Encoder.fit_transform(df_trn['Sex']).astype(int)
df_tst['Sex'] = Sex_Encoder.transform(df_tst['Sex']).astype(int)
mean_age = df['Age'][df['Age'].notna()].mean()
df_trn['Age'].fillna(mean_age,inplace=True)
df_tst['Age'].fillna(mean_age,inplace=True)
age_limits = cont_class_limits(df['Age'],3)
df_trn['Age'] = cont_classification(df_trn['Age'],age_limits)
df_tst['Age'] = cont_classification(df_tst['Age'],age_limits)
# df_trn['FamilySize'] = df_trn['SibSp'] + df_trn['Parch'] + 1
# df_tst['FamilySize'] = df_tst['SibSp'] + df_tst['Parch'] + 1

# df_trn = df_trn.drop(columns = ['SibSp','Parch'])
# df_tst = df_tst.drop(columns = ['SibSp','Parch'])
# df_trn['Cabin'] = df_trn['Cabin'].fillna('N000')

# df_cab = df_trn[df_trn['Cabin'].notna()]

# df_cab = pd.concat(
#     [
#         df_cab,
#         df_cab['Cabin'].str.extract(
#             '([A-Za-z]+)(\d+\.?\d*)([A-Za-z]*)', 
#             expand = True).drop(columns=[2]).rename(
#             columns={0: 'Cabin_Class', 1: 'Cabin_Number'}
#         )
#     ], axis=1)

# df_trn = df_cab.drop(columns=['Cabin','Cabin_Number'])
# df_trn = pd.concat([
#             df_trn.drop(columns=['Cabin_Class']),
#             pd.get_dummies(df_trn['Cabin_Class'],prefix='Cabin').drop(columns=['Cabin_N'])
# #             pd.get_dummies(df_trn['Cabin_Class'],prefix='Cabin')
#         ],axis=1)


# df_tst['Cabin'] = df_tst['Cabin'].fillna('N000')

# df_cab = df_tst[df_tst['Cabin'].notna()]

# df_cab = pd.concat(
#     [
#         df_cab,
#         df_cab['Cabin'].str.extract(
#             '([A-Za-z]+)(\d+\.?\d*)([A-Za-z]*)', 
#             expand = True).drop(columns=[2]).rename(
#             columns={0: 'Cabin_Class', 1: 'Cabin_Number'}
#         )
#     ], axis=1)

# df_tst = df_cab.drop(columns=['Cabin','Cabin_Number'])
# df_tst = pd.concat([
#             df_tst.drop(columns=['Cabin_Class']),
#             pd.get_dummies(df_tst['Cabin_Class'],prefix='Cabin').drop(columns=['Cabin_N'])
# #             pd.get_dummies(df_tst['Cabin_Class'],prefix='Cabin')
#         ],axis=1)
df_trn = df_trn.drop(columns=['Ticket'])
df_tst = df_tst.drop(columns=['Ticket'])
mean_fare = df['Fare'][df['Fare'].notna()].mean()
df_trn['Fare'].fillna(mean_fare,inplace=True)
df_tst['Fare'].fillna(mean_fare,inplace=True)

fare_limits = cont_class_limits(df['Fare'],3)
df_trn['Fare'] = cont_classification(df_trn['Fare'],fare_limits)
df_tst['Fare'] = cont_classification(df_tst['Fare'],fare_limits)
most_frequent_emb = df['Embarked'].value_counts()[:1].index.tolist()[0]
df_trn['Embarked'] = df_trn['Embarked'].fillna(most_frequent_emb)
df_tst['Embarked'] = df_tst['Embarked'].fillna(most_frequent_emb)
df_trn = pd.concat([
            df_trn.drop(columns=['Embarked']),
            pd.get_dummies(df_trn['Embarked'],prefix='Emb').drop(columns=['Emb_C'])
#             pd.get_dummies(df_trn['Embarked'],prefix='Emb')
        ],axis=1)


df_tst = pd.concat([
            df_tst.drop(columns=['Embarked']),
            pd.get_dummies(df_tst['Embarked'],prefix='Emb').drop(columns=['Emb_C'])
#             pd.get_dummies(df_tst['Embarked'],prefix='Emb')
        ],axis=1)
Model_Scores = {}
Model_Scores = {}

def print_model_scores():
    return pd.DataFrame([[
        model,
        Model_Scores[model]['test_accuracy_score'],
        Model_Scores[model]['cv_score_mean'],
        Model_Scores[model]['cv_score_std']
    ] for model in Model_Scores.keys()],
        columns=['model','test_accuracy_score','cv_score','cv_score_std']
    ).sort_values(by='cv_score',ascending=False)


def OptimizeClassification(X,y,
    model,
    hyperparametric_space,
    cv = KFold(n_splits = 10, shuffle=True,random_state=SEED),
    model_description = 'classifier',
    n_iter = 20,
    test_size = 0.25
):
    
    X_trn,X_tst,y_trn,y_tst = train_test_split(
        X,
        y,
        test_size = test_size,
        random_state=SEED
    )
    
    start = time()

    # Searching the best setting
    print('[info] Searching for the best hyperparameter')
    search_cv = RandomizedSearchCV(
                        model,
                        hyperparametric_space,
                        n_iter = n_iter,
                        cv = cv,
                        random_state = SEED)

    search_cv.fit(X, y)
    results = pd.DataFrame(search_cv.cv_results_)

    print('[info] Search Timing: %.2f seconds'%(time() - start))

    # Evaluating Test Score For Best Estimator
    start = time()
    print('[info] Test Accuracy Score')
    gb = search_cv.best_estimator_
    gb.fit(X_trn, y_trn)
    y_pred = gb.predict(X_tst)

    # Evaluating K Folded Cross Validation
    print('[info] KFolded Cross Validation')
    cv_results = cross_validate(search_cv.best_estimator_,X,y,
                    cv = cv )

    print('[info] Cross Validation Timing: %.2f seconds'%(time() - start))
    
    Model_Scores[model_description] = {
        'test_accuracy_score':gb.score(X_tst,y_tst),
        'cv_score_mean':cv_results['test_score'].mean(),
        'cv_score_std':cv_results['test_score'].std(),
        'best_params':search_cv.best_estimator_.get_params()
    }
    
    pd.options.display.float_format = '{:,.5f}'.format

    print('\t\t test_accuracy_score: {:.3f}'.format(gb.score(X_tst,y_tst)))
    print('\t\t cv_score: {:.3f}±{:.3f}'.format(
        cv_results['test_score'].mean(),cv_results['test_score'].std()))


    params_list = ['mean_test_score']+list(map(lambda var: 'param_'+var,search_cv.best_params_.keys()))+['mean_fit_time']
    
    return results[params_list].sort_values(
        ['mean_test_score','mean_fit_time'],
        ascending=[False,True]
    )
scaler = StandardScaler()
# caler = Normalizer()
scaler.fit(df_trn.drop(columns=['Survived','Cabin']))
X = scaler.transform(df_trn.drop(columns=['Survived','Cabin']))
y = df_trn['Survived']
results = OptimizeClassification(X,y,
        model = LogisticRegression(random_state=SEED),
        hyperparametric_space = {
            'solver' : ['newton-cg', 'lbfgs', 'liblinear'],# 
            'C' : uniform(0.075,0.125,200) #10**uniform(-2,2,200)
        },
        cv = KFold(n_splits = 50, shuffle=True,random_state=SEED),
        model_description = 'LogisticRegression',
        n_iter = 20
    )

results.head(5)
results = OptimizeClassification(X,y,
         model = SVC(random_state=SEED),
         hyperparametric_space = {
                'kernel' : ['linear', 'poly','rbf','sigmoid'],
                'C' : 10**uniform(-1,1,200),
                'decision_function_shape' : ['ovo', 'ovr'],
                'degree' : [1,2,3,4]
            },
        cv = KFold(n_splits = 50, shuffle=True,random_state=SEED),
        model_description = 'SVC',
        n_iter = 20
    )

results.head(5)
results = OptimizeClassification(X,y,
         model = DecisionTreeClassifier(),
         hyperparametric_space = {
            'min_samples_split': randint(10,30),
            'max_depth': randint(10,30),
            'min_samples_leaf': randint(1,10)
         },
        cv = KFold(n_splits = 50, shuffle=True,random_state=SEED),
        model_description = 'DecisionTree',
        n_iter = 100
    )

results.head(5)
print_model_scores()
results = OptimizeClassification(X,y,
         model = RandomForestClassifier(random_state = SEED,oob_score=True),
         hyperparametric_space = {
            'n_estimators': randint(190,250),
            'min_samples_split': randint(10,15),
            'min_samples_leaf': randint(1,6)
#             'max_depth': randint(1,100),
#             ,
#             'min_weight_fraction_leaf': uniform(0,1,100),
#             'max_features': uniform(0,1,100),
#             'max_leaf_nodes': randint(10,100),
         },
        cv = KFold(n_splits = 20, shuffle=True,random_state=SEED),
        model_description = 'RandomForestClassifier',
        n_iter = 20
    )

results.head(5)
print_model_scores()
results = OptimizeClassification(X,y,
    model = GradientBoostingClassifier(),
    hyperparametric_space = {
            'loss':               ['exponential'], #'deviance', 
            'min_samples_split':  randint(130,170),
            'max_depth':          randint(6,15),
            'learning_rate':      uniform(0.05,0.15,100),
            'random_state' :      randint(0,10),
            'tol':                10**uniform(-5,-3,100)
         },
    cv = KFold(n_splits = 20, shuffle=True,random_state=SEED),
    model_description = 'GradientBoostingClassifier',
    n_iter = 20
)
results.head(5)
def random_layer(max_depth=4,max_layer=100):
    res = list()
    depth = np.random.randint(1,1+max_depth)
    
    for i in range(1,depth+1):
        res.append(np.random.randint(2,max_layer))
        
    return tuple(res)



results = OptimizeClassification(X,y,
    model = MLPClassifier(random_state=SEED),
    hyperparametric_space  = {
            'hidden_layer_sizes': [random_layer(max_depth=4,max_layer=40) for i in range(10)],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['adaptive'],
            'activation' : ['identity', 'logistic', 'tanh', 'relu']
    },
    cv = KFold(n_splits = 20, shuffle=True,random_state=SEED),
    model_description = 'MLPClassifier',
    n_iter = 20
)
results.head(5)
print_model_scores()
model = GradientBoostingClassifier(**Model_Scores['GradientBoostingClassifier']['best_params'])

X_trn,X_tst,y_trn,y_tst = train_test_split(
    X,
    y,
    test_size = 0.25
)

model.fit(X_trn,y_trn)

y_pred = model.predict(X_tst)

cv_results = cross_validate(model,X,y,
                    cv = KFold(n_splits = 20, shuffle=True)  )
    
print('test_accuracy_score: {:.3f}'.format(model.score(X_tst,y_tst)))
print('cv_score: {:.3f}±{:.3f}'.format(
    cv_results['test_score'].mean(),cv_results['test_score'].std()))

pass_id = pd.read_csv('data/test.csv')['PassengerId']
model = GradientBoostingClassifier(**Model_Scores['GradientBoostingClassifier']['best_params'])
model.fit(X,y)

X_sub = scaler.transform(df_tst.drop(columns=['Cabin']))
y_pred = model.predict(X_sub)

sub = pd.Series(y_pred,index=pass_id,name='Survived')
sub.to_csv('gbc_model_2.csv',header=True)

y_pred
model
