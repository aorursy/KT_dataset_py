# Pandas and numpy for data manipulation

import warnings

import pandas as pd

import numpy as np



# Matplotlib and seaborn for visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Scipy for statistics

from scipy import stats



# os to manipulate files

import os



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score 

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, LabelEncoder



from sklearn.model_selection import cross_validate,KFold,GridSearchCV

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC





from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# from sklearn.metrics import mean_absolute_error,r2_score

# from sklearn import linear_model

# from sklearn.preprocessing import PolynomialFeatures
warnings.filterwarnings("ignore")



# Setting seaborn style

sns.set_style("darkgrid")

sns.set_palette(sns.color_palette(

    ["#7a0723","#ff5b21","#f7cf60", "#ff94c0","#323133"]

))



# Setting Pandas float format

pd.options.display.float_format = '{:,.1f}'.format



SEED = 42   #The Answer to the Ultimate Question of Life, The Universe, and Everything

np.random.seed(SEED)
def print_categories(encoder,num_list,title):

    n_list = np.array(num_list).reshape(-1,)

    res = [[encoder.inverse_transform([[elem]]).reshape(-1,)[0],elem] for elem in n_list]

    

    print('\nEncoding for '+title)

    for elem in res:

        print(' '+str(elem[0])+'\t->\t'+str(elem[1]))

        

        

def cross_validate_interval(results):

    mean = results['test_score'].mean()

    std = results['test_score'].std()

    interval = [mean-2*std,mean+2*std]

    print('A acurácia média é {:.3}%'.format(100*media))

    print('A acurácia está entre {:.3}% e {:.3}%'.format(*(100*np.array(intervalo))))

    

def print_score(scores):

    mean= scores.mean() * 100

    std = scores.std() * 100

    print("Accuracy médio %.2f" % mean)

    print("Intervalo [%.2f, %.2f]" % (mean - std, mean + std))

    

def hyperparameter_mean(results,hyperpar):

    str_ = 'param_'+hyperpar

    return results[[str_,'mean_fit_time','mean_score_time','mean_test_score']].groupby(str_).mean()
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
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv('/kaggle/input/drug-classification/drug200.csv')#.drop('Unnamed: 0',axis=1)



df1.columns = ['age','gender','pressure','cholesterol','na_k_ratio','drug']

df1['drug'] = df1['drug'].str.replace('drug','Drug')

df1.sample(5)
df1.describe(include='all')
# Set up the matplotlib figure

f, axes = plt.subplots(2, 2, figsize=(12, 12))

#sns.despine(left=True)

sns.countplot(x=df1['drug'],data=df1, ax=axes[0,0])

sns.countplot(x=df1['drug'],hue='cholesterol',data=df1, ax=axes[0,1])

sns.countplot(x=df1['drug'],hue='gender',data=df1, ax=axes[1,0])

sns.countplot(x=df1['drug'],hue='pressure',data=df1, ax=axes[1,1])



axes[0,0].set(xlabel='', ylabel='Number of Occurrences',title='Drug Occurrences')

axes[0,1].set(xlabel='', ylabel='Number of Occurrences',title='Relation between Drug and Cholesterol')

axes[1,0].set(xlabel='', ylabel='Number of Occurrences',title='Relation between Drug and Gender')

axes[1,1].set(xlabel='', ylabel='Number of Occurrences',title='Relation between Drug and Blood Pressure')



axes[0,1].legend(loc=1,title='Cholesterol')

axes[1,0].legend(loc=1,title='Gender')

axes[1,1].legend(loc=1,title='Blood Pressure')



plt.show()
ax1 = sns.catplot(x='drug', y='age', data = df1)

ax2 = sns.catplot(x='drug', y='na_k_ratio', data = df1)



ax1.set(xlabel='', ylabel='Age',title="People's age")

ax2.set(xlabel='', ylabel='Na/K Ratio',title="Urinary Na/K ratio")





plt.show()
g = sns.pairplot(df1, hue="drug")
# Splitting target from features

y_column = 'drug'

X_columns = list(df1.drop(columns=y_column).columns)



X = df1[X_columns]

y = df1[y_column]
Ord_Encoder = OrdinalEncoder(categories=[['LOW', 'NORMAL', 'HIGH']])

Gender_Encoder = LabelEncoder()

Drugs_Encoder = LabelEncoder()



pressure_list = [[elem] for elem in X['pressure'].to_list()]

X['pressure'] = Ord_Encoder.fit_transform(pressure_list).astype(int)



cholesterol_list = [[elem] for elem in X['cholesterol'].to_list()]

X['cholesterol'] = Ord_Encoder.fit_transform(cholesterol_list).astype(int)



X['gender'] = Gender_Encoder.fit_transform(X[['gender']]).astype(int)



y = Drugs_Encoder.fit_transform(y)



df_enc = pd.concat([X,pd.DataFrame(y,columns=['drug'])],axis=1)
print_categories(Ord_Encoder,X['pressure'].unique(),'pressure')

print_categories(Ord_Encoder,X['cholesterol'].unique(),'cholesterol')

print_categories(Gender_Encoder,X['gender'].unique(),'gender')

print_categories(Drugs_Encoder,np.unique(y),'drugs')
df_enc.sample(5)
X_trn, X_tst, y_trn, y_tst = train_test_split(

    X, y, test_size = 0.33,stratify = y,random_state=42)
from sklearn.preprocessing import StandardScaler



scaler_trn = StandardScaler()

X_scl_trn = scaler_trn.fit_transform(X_trn)



scaler_tst = StandardScaler()

X_scl_tst = scaler_trn.fit_transform(X_tst)



scaler = StandardScaler()

X_scl = scaler_trn.fit_transform(X)
X_scl = pd.concat([

            pd.DataFrame(X_scl,columns=X.columns),

            pd.DataFrame(y,columns=['drug'])

        ],axis=1)



X_trn_scl = pd.concat([

            pd.DataFrame(X_scl_trn,columns=X.columns),

            pd.DataFrame(y_trn,columns=['drug'])

        ],axis=1)



X_tst_scl = pd.concat([

            pd.DataFrame(X_scl_tst,columns=X.columns),

            pd.DataFrame(y_tst,columns=['drug'])

        ],axis=1)
pd.options.display.float_format = '{:,.4f}'.format



Model_Scores = {}
SEED = 40

model_dummy = DummyClassifier(strategy='stratified',random_state = SEED)



dummy_results = cross_validate(

    model_dummy,

    X = X,

    y = y,

    cv=KFold(n_splits = 200,shuffle=True,random_state = SEED)

)



dummy_score = dummy_results['test_score'].mean()



model_dummy = DummyClassifier(strategy='stratified',random_state = SEED)



model_dummy.fit(X_trn,y_trn)

y_pred = model_dummy.predict(X_tst)

dummy_acc_score = accuracy_score(y_tst,y_pred)





print('Dummy Accuracy Score: %.2f' % (dummy_acc_score))

print('Dummy CV Score: %.1f%%' % (100*dummy_score))





Model_Scores['dummy'] = {

    'model' : model_dummy,

    'best_params' : model_dummy.get_params(),

    'test_accuracy_score' : dummy_acc_score,

    'cv_score' : dummy_results['test_score'].mean(),

    'cv_score_std' : dummy_results['test_score'].std()

}
"""

    From previous tests, it was possible to conclude that sag and saga solvers

    retrieve the worst results. I'll not consider this in GridSearchCV.

"""

SEED = 42



hyperparametric_space = {

    'solver' : ['newton-cg', 'lbfgs', 'liblinear'],

    'C' : [5,7,10,20,30,50,100]

}



grid_search_cv = GridSearchCV(

    LogisticRegression(random_state=SEED),

    hyperparametric_space,

    cv = KFold(n_splits = 10, shuffle=True,random_state=SEED),

    scoring='accuracy',

    verbose=0

)



grid_search_cv.fit(X, y)

results = pd.DataFrame(grid_search_cv.cv_results_)



pd.options.display.float_format = '{:,.5f}'.format



col = ['param_C', 'param_solver','mean_fit_time', 'mean_test_score', 'std_test_score']



results[col].sort_values(

    ['mean_test_score','mean_fit_time'],

    ascending=[False,True]

).head(10)
SEED = 42



best_solver = 'newton-cg'



res_liblinear = results[results['param_solver']==best_solver].sort_values(

    ['mean_test_score','mean_fit_time'], ascending=[False,True] )



logReg_model_params = res_liblinear['params'].iloc[0]



logReg_model = LogisticRegression(**logReg_model_params,random_state=SEED)



logReg_cv_results = cross_validate(

    logReg_model,

    X = X,

    y = y,

    cv=KFold(n_splits = 10,shuffle=True,random_state=SEED)

)



logReg_cv_score = logReg_cv_results['test_score'].mean()

logReg_cv_std = logReg_cv_results['test_score'].std()



print('Logistic Regression Classifier CV Score: %.2f%% ± %.2f%%' % (100*logReg_cv_score,100*logReg_cv_std))
SEED = 42



logReg = LogisticRegression(**logReg_model_params,random_state=SEED)



logReg.fit(X_trn,y_trn)



y_pred = logReg.predict(X_tst)



logReg_score = accuracy_score(y_tst, y_pred)

logReg_score
Model_Scores['logistic_regression'] = {

    'model' : logReg,

    'best_params' : logReg.get_params(),

    'test_accuracy_score' : logReg_score,

    'cv_score' : logReg_cv_score,

    'cv_score_std' : logReg_cv_std

}
# Instantiating the Classifier class

log = Classifier(

         algorithm = LogisticRegression,

         hyperparameters_range = {

            'solver' : ['newton-cg', 'lbfgs', 'liblinear'],

            'C' : [5,7,10,20,30,50,100]

        }

     )



log.grid_search_fit(X,y)



print('\nBest Model:')

print('\n',log.best_model)



sc_dict = log.best_model_cv_score(X,y)

sc_list = list((100*np.array(list(sc_dict.values()))))

print('\nCV Score: %.2f%% ± %.2f%%' % (sc_list[0],sc_list[1]))



log.fit(X_trn,y_trn,params = 'best_model')

psc = log.predict_score(X_tst,y_tst)

print('\nAccuracy Score: %.2f ' % psc)



Model_Scores['logistic_regression'] = {

    'model' : log.best_model,

    'best_params' : log.best_model_params,

    'test_accuracy_score' : psc,

    'cv_score' : 0.01*sc_list[0],

    'cv_score_std' : 0.01*sc_list[1]

}



log.grid_search_results.head(5)
sv = Classifier(

         algorithm = SVC,

         hyperparameters_range = {

                'kernel' : ['linear', 'poly','rbf','sigmoid'],

                'C' : [0.1,0.5,1,3,7,10]

            }

     )



sv.grid_search_fit(X,y)



print('\nBest Model:')

print('\n',sv.best_model)



sc_dict = sv.best_model_cv_score(X,y)

sc_list = list((100*np.array(list(sc_dict.values()))))

print('\nCV Score: %.2f%% ± %.2f%%' % (sc_list[0],sc_list[1]))



sv.fit(X_trn,y_trn,params = 'best_model')

psc = sv.predict_score(X_tst,y_tst)

print('\nAccuracy Score: %.2f ' % (psc))





Model_Scores['svc'] = {

    'model' : sv.best_model,

    'best_params' : sv.best_model_params,

    'test_accuracy_score' : psc,

    'cv_score' : 0.01*sc_list[0],

    'cv_score_std' : 0.01*sc_list[1]

}



sv.grid_search_results.head(5)
dt = Classifier(

         algorithm = DecisionTreeClassifier,

         hyperparameters_range = {

            'min_samples_split': [2,5,10],

            'max_depth': [2,5,10],

            'min_samples_leaf': [1,5,10]

         }

     )



dt.grid_search_fit(X,y)



print('\nBest Model:')

print('\n',dt.best_model)



sc_dict = dt.best_model_cv_score(X,y)

sc_list = list((100*np.array(list(sc_dict.values()))))

print('\nCV Score: %.2f%% ± %.2f%%' % (sc_list[0],sc_list[1]))



dt.fit(X_trn,y_trn,params = 'best_model')

psc = dt.predict_score(X_tst,y_tst)

print('\nAccuracy Score: %.2f ' % (psc))



Model_Scores['decision_tree'] = {

    'model' : dt.best_model,

    'best_params' : dt.best_model_params,

    'test_accuracy_score' : psc,

    'cv_score' : 0.01*sc_list[0],

    'cv_score_std' : 0.01*sc_list[1]

}



dt.grid_search_results.head(5)
gnb = Classifier(

         algorithm = GaussianNB,

         hyperparameters_range = {

            'var_smoothing': [1e-09,1e-08,1e-07,1e-06,1e-05,4e-05,1e-04],

         }

     )



gnb.grid_search_fit(X,y)



print('\nBest Model:')

print('\n',gnb.best_model)



sc_dict = gnb.best_model_cv_score(X,y)

sc_list = list((100*np.array(list(sc_dict.values()))))

print('\nCV Score: %.2f%% ± %.2f%%' % (sc_list[0],sc_list[1]))



gnb.fit(X_trn,y_trn,params = 'best_model')

print('\nAccuracy Score: %.2f ' % (gnb.predict_score(X_tst,y_tst)))





pd.options.display.float_format = '{:,.8f}'.format



Model_Scores['gaussian_nb'] = {

    'model' : gnb.best_model,

    'best_params' : gnb.best_model_params,

    'test_accuracy_score' : psc,

    'cv_score' : 0.01*sc_list[0],

    'cv_score_std' : 0.01*sc_list[1]

}



gnb.grid_search_results.head(9)
knn = Classifier(

         algorithm = KNeighborsClassifier,

         hyperparameters_range = {

            'n_neighbors': [2,5,10,20],

             'weights' : ['uniform', 'distance'],

             'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],

             'p' : [2,3,4,5]

         }

     )



knn.grid_search_fit(X,y)



print('\nBest Model:')

print('\n',knn.best_model)



sc_dict = knn.best_model_cv_score(X,y)

sc_list = list((100*np.array(list(sc_dict.values()))))

print('\nCV Score: %.2f%% ± %.2f%%' % (sc_list[0],sc_list[1]))



knn.fit(X_trn,y_trn,params = 'best_model')

psc = knn.predict_score(X_tst,y_tst)

print('\nAccuracy Score: %.2f ' % (psc))





pd.options.display.float_format = '{:,.3f}'.format





Model_Scores['knn_classifier'] = {

    'model' : knn.best_model,

    'best_params' : knn.best_model_params,

    'test_accuracy_score' : psc,

    'cv_score' : 0.01*sc_list[0],

    'cv_score_std' : 0.01*sc_list[1]

}





knn.grid_search_results.head(9)
rf = Classifier(

         algorithm = RandomForestClassifier,

         hyperparameters_range = {

            'n_estimators': [5,10,15,20,25,30,50,100],

             'random_state': [42]

         }

     )



rf.grid_search_fit(X,y)



print('\nBest Model:')

print('\n',rf.best_model)



sc_dict = rf.best_model_cv_score(X,y)

sc_list = list((100*np.array(list(sc_dict.values()))))

print('\nCV Score: %.2f%% ± %.2f%%' % (sc_list[0],sc_list[1]))



rf.fit(X_trn,y_trn,params = 'best_model')

psc = rf.predict_score(X_tst,y_tst)

print('\nAccuracy Score: %.2f ' % (psc))





pd.options.display.float_format = '{:,.3f}'.format





Model_Scores['random_forest'] = {

    'model' : rf.best_model,

    'best_params' : rf.best_model_params,

    'test_accuracy_score' : psc,

    'cv_score' : 0.01*sc_list[0],

    'cv_score_std' : 0.01*sc_list[1]

}





rf.grid_search_results.head(9)
pd.DataFrame([

    [

        key,

        Model_Scores[key]['test_accuracy_score'],

        Model_Scores[key]['cv_score'],

        Model_Scores[key]['cv_score_std']

    ]

    

    for key in Model_Scores

],columns=['model','accuracy_score','cv_score','cv_score_std'])