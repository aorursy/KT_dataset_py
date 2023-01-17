#import data into dataframe

import pandas as pd

data=pd.read_csv('../input/income_evaluation.csv')

import warnings

warnings.filterwarnings('ignore')
data.columns=[x.strip() for x in data.columns]

data=data.apply(lambda x: x.str.strip() if x.dtype=='object' else x)

data.head()
data.describe()
import seaborn as sns

import matplotlib.pyplot as plt



for i, col in enumerate(data.columns):

    plt.figure(i)

    if (data[col].dtype=='object')|(col=='education-num')|(col=='age'):

        plot=sns.countplot(data[col])

        

        for index, item in enumerate(plot.get_xticklabels()):

            item.set_rotation(45)

            

            if col=='age':

                if index % 5 != 0:

                    item.set_visible(False)

                

            

    else:

        plot=sns.distplot(data[col])

    plot
target=data['income']

target=target.map({'<=50K':False,'>50K':True})

target=target.astype('bool')

data=data.drop(columns=['income'])
#remove redundant features

data=data.drop(columns=['education-num'])

sns.heatmap(data.select_dtypes(exclude='object').assign(income=target).corr(),annot=True)
from dython.nominal import associations

associations(dataset=data.select_dtypes(include='object').assign(age=data['age'],income=target), nominal_columns='all',plot=True)

associations(dataset=data.select_dtypes(include='object').assign(age=data['age'],income=target),theil_u=True, nominal_columns='all',plot=True)

from sklearn.preprocessing import label_binarize



def correct_dtypes(data):

    cat_cols=data.select_dtypes('object').columns.values

    data[cat_cols]=data[cat_cols].astype('category')

    

    data['sex']=data['sex'].map({'Female':0,'Male':1}).astype('bool')

    data=data.rename(columns={'sex':'female'})

    return data



data=correct_dtypes(data)

data.head()
cat=data.select_dtypes(include='category')

cat=pd.get_dummies(cat, columns=cat.columns.values,prefix=cat.columns.values)

data_oh=pd.concat([data.select_dtypes(exclude='category'),cat],axis=1)



data_oh.head()
from sklearn.tree import export_graphviz

from sklearn.tree import DecisionTreeClassifier

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import pydot



#data_oh=data_oh.drop(columns=['fnlwgt'])

tree=DecisionTreeClassifier().fit(data_oh,target)



with open("tree1.dot", 'w') as dot:

    dot = export_graphviz(tree,

                          out_file=dot,

                          max_depth = 5,

                          impurity = True,

                          class_names = ['<=50K','>50K'],

                          feature_names = data_oh.columns.values,

                          rounded = True,

                          filled= True )



    

# Annotating chart with PIL

(graph,) = pydot.graph_from_dot_file('tree1.dot')



graph.write_png('tree1.png')

PImage('tree1.png')

import numpy as np

feat_impt=pd.DataFrame(data=tree.feature_importances_).T

feat_impt.columns=data_oh.columns.values



names = list(data.columns.values)



feature_importances=pd.DataFrame()

for column in names:

    value=feat_impt.filter(regex=column)

    value=value.mean(axis=1)

    feature_importances[column]=value



#feature_importances=pd.melt(feature_importances)

p=sns.barplot(data=feature_importances)

p.set_xticklabels(p.get_xticklabels(),rotation=45)

feature_importances
data=data.drop(columns=['fnlwgt'])

data_oh=data_oh.drop(columns=['fnlwgt'])
#split data into train and test

#we will train our model with data/target evalulate our model at the very end with eval_data/eval_taret



from sklearn.model_selection import train_test_split





data, eval_data, target, eval_target = train_test_split(data,target,test_size=.20)
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.impute import SimpleImputer



class TypeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype):

        self.dtype = dtype

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X.select_dtypes(include=[self.dtype])

    

class StringIndexer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X.apply(lambda s: s.cat.codes.replace(

            {-1: len(s.cat.categories)}

        ))



transformer = Pipeline([

    ('features', FeatureUnion(n_jobs=1, transformer_list=[

        # Part 1

        ('boolean', Pipeline([

            ('selector', TypeSelector('bool')),

        ])),  # booleans close

        

        ('numericals', Pipeline([

            ('selector', TypeSelector(np.number)),

            ('imputer',SimpleImputer()),

            ('scaler', StandardScaler()),

            

        ])),  # numericals close

        

        # Part 2

        ('categoricals', Pipeline([

            ('selector', TypeSelector('category')),

            ('labeler', StringIndexer()),

            ('encoder', OneHotEncoder()),

        ]))  # categoricals close

    ])),  # features close

])  # pipeline close
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import RidgeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier



estimators=[

    ('logistic',LogisticRegression(solver='liblinear',penalty='l2')),

    ('lasso',LogisticRegression(solver='liblinear',penalty='l1')),

    ('ridge',RidgeClassifier()),

    ('elasticnet',SGDClassifier(loss='log', penalty='elasticnet')),

    #('decision_tree',DecisionTreeClassifier()),

    ('random_forest',RandomForestClassifier()),

    ('xgb',XGBClassifier(ojective='reg:logistic')),

    ('svc',LinearSVC()),

    ('deep_nn',MLPClassifier()),

    ('knn',KNeighborsClassifier(n_neighbors=5,weights='distance',algorithm='auto'))

]



pipes={}

for model in estimators:

    pipe=Pipeline(steps=[('data_prep',transformer),model])

    pipe.fit(data,target)

    pipes[pipe.steps[1][0]]=pipe



from sklearn.model_selection import KFold,cross_validate

from sklearn.metrics import make_scorer, f1_score, accuracy_score,roc_auc_score,log_loss



_metrics={'f1':make_scorer(f1_score),'auc':make_scorer(roc_auc_score),

         'accuracy':'accuracy','logloss':make_scorer(log_loss)}



estimator_names=[model[0] for model in estimators]



def plot_estimators(estimators=estimator_names,n_splits=5,metrics=['f1','auc','accuracy','logloss']):

    metrics={key : _metrics[key] for key in metrics}

    scorers=[]

    labels=[]

    for pipe_name in pipes.keys():

        if pipe_name in estimators:

            pipe=pipes[pipe_name]

            labels.append(pipe_name)

            kf=KFold(n_splits)

            model_score=cross_validate(pipe,data,target,scoring=metrics,cv=kf)

            scorers.append(model_score)

    

    score_lists={}

    for metric in metrics:

        score_lists[metric]=[score['test_'+metric] for score in scorers]

    

    for  i,(title, _list) in enumerate(score_lists.items()):

        plt.figure(i)

        plot=sns.boxplot(data=_list).set_xticklabels(labels, rotation=45)

        plt.title(title)

metrics={'f1':make_scorer(f1_score),'auc':make_scorer(roc_auc_score),

         'accuracy':'accuracy','logloss':make_scorer(log_loss)}
plot_estimators()
from sklearn.model_selection import GridSearchCV



def tune_param(model,param_grid,refit='auc',chart=None,data=data,target=target,cv=5):

    

    param_grid={model+'__'+key : param_grid[key] for key in param_grid.keys()}



    xgbcv=GridSearchCV(pipes[model],param_grid,scoring=metrics,refit=refit,cv=cv)

    xgbcv.fit(data,target)



    print('best score: '+str(xgbcv.best_score_))

    print('best params: '+str(xgbcv.best_params_))

    results=pd.DataFrame(xgbcv.cv_results_)

    

    if 'line' in chart:

        for i,param in enumerate(param_grid.keys()):

            graph_data=results[['param_'+param,'mean_test_'+refit,'mean_train_'+refit]]

            graph_data=graph_data.rename(columns={'mean_test_'+refit:'test','mean_train_'+refit:'train'})

            graph_data=graph_data.melt('param_'+param, var_name='type',value_name=refit)

            plt.figure(i)

            plot=sns.lineplot(x='param_'+param,y=refit,hue='type',data=graph_data)

            

    if 'heatmap' in chart:

        assert len(param_grid) == 2,  'heatmap only works with 2 params, {} passed'.format(str(len(param_grid)))

        

        param1=list(param_grid.keys())[0]

        param2=list(param_grid.keys())[1]



        graph_data=results[['param_'+param1,'param_'+param2,'mean_test_'+refit]]

        graph_data=graph_data.pivot(index='param_'+param1,columns='param_'+param2,values='mean_test_'+refit)

        sns.heatmap(graph_data,annot=True,xticklabels=True,yticklabels=True).set(xlabel=param2,ylabel=param1)

pipes['xgb'].named_steps['xgb'].get_params()
param_grid={'n_estimators': [100,200,300,400,500,600,700,800,900]} 

tune_param('xgb',param_grid,chart='line')
pipes['xgb'].set_params(**{'xgb__n_estimators': 1000})
param_grid={'max_depth':[3,5,7,9]}

tune_param('xgb',param_grid,chart='line')
pipes['xgb'].set_params(**{'xgb__max_depth': 3})
param_grid={'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]}

tune_param('xgb',param_grid,chart='line')
pipes['xgb'].set_params(**{'xgb__learning_rate': 0.1})
param_grid={'subsample':[0.8,0.85,0.9,0.95,1]}

tune_param('xgb',param_grid,chart='line')

pipes['xgb'].set_params(**{'xgb__learning_rate': 0.95})
param_grid={'colsample_bytree':[0.3,0.5,0.7,0.9,1]}

tune_param('xgb',param_grid,chart='line')
pipes['xgb'].set_params(**{'xgb__colsample_bytree': 0.3})
param_grid={'gamma':[0,1,5]}

tune_param('xgb',param_grid,chart='line')
pipes['xgb'].set_params(**{'xgb__gamma': 5})

pipes['xgb'].named_steps['xgb'].get_params()
pipes['random_forest'].named_steps['random_forest'].get_params()
pipes['random_forest'].set_params(**{'random_forest__n_estimators': 100})

param_grid={'max_depth':[3,8,13,18]}

tune_param('random_forest',param_grid,chart='line')
#we see that test auc plateaus at around 13-14ish

#for efficiency sake lets set max depth at 13

pipes['random_forest'].set_params(**{'random_forest__max_depth': 13})

param_grid={'max_leaf_nodes':[5,10,15,20,25]}

tune_param('random_forest',param_grid,chart='line')
param_grid={'n_estimators':[10,50,100,150,200,500]}

tune_param('random_forest',param_grid,chart='line')
pipes['random_forest'].set_params(**{'random_forest__n_estimators': 50})
pipes['logistic'].named_steps['logistic'].get_params()
pipes['lasso'].named_steps['lasso'].get_params()
warnings.filterwarnings('ignore')

param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 

           'tol':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

tune_param('logistic',param_grid,chart='heatmap')
pipes['logistic'].set_params(**{'logistic__C': 1000, 'logistic__tol': 0.001})

pipes['lasso'].set_params(**{'lasso__C': 1000, 'lasso__tol': 0.001})

param_grid={'C': [0.001, 0.01, 0.1, 1, 10]}

tune_param('svc',param_grid,chart='linear')
pipes['svc'].set_params(**{'svc__C': 1})
plot_estimators(['xgb','svc','logistic','lasso','random_forest'])
from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



t=transformer

t.fit(data)

t=t.transform(data)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



to_plot=['xgb','random_forest','lasso','ridge']

for name in pipes.keys():

    if name in to_plot:

        g=plot_learning_curve(pipes[name].named_steps[name],name+' learning curves',t,target,cv=KFold(4),n_jobs=4)
tree_based=['random_forest','xgb']

for name in pipes.keys(): 

    if name in tree_based:

        feat_impt=pipes[name].named_steps[name].feature_importances_

        graph_data=pd.DataFrame()



        graph_data['feature']=data_oh.columns.values

        graph_data['importance']=feat_impt

        graph_data_top=graph_data.nlargest(30,'importance')



        plt.figure(figsize=(6.4,6))

        g=sns.barplot(y='feature',x='importance',data=graph_data_top,orient='h')

        g.set_ylabel('Features',fontsize=12)

        g.set_xlabel('Relative Importance')

        g.set_title(name + " feature importance")

        g.tick_params(labelsize=8)

        
ensemble_results=pd.DataFrame()

for name,pipe in pipes.items():

    ensemble_results[name]=pipe.predict(eval_data)

sns.heatmap(ensemble_results.corr(),annot=True)
del pipes['ridge'],pipes['decision_tree'],pipes['logistic']
def print_predictions(target,predictions):

    print('auc: '+str(roc_auc_score(target,predictions)))

    print('f1: '+str(f1_score(target,predictions)))

    print('accuracy: '+str(accuracy_score(target,predictions)))

    print('logloss: '+str(log_loss(target,predictions)))
from sklearn.ensemble import VotingClassifier

from sklearn.base import clone



estimators=[(pipe.steps[1][0],clone(pipe.steps[1][1])) for pipe in pipes.values()] 

vote=Pipeline(steps=[('data_prep',transformer),('voter',VotingClassifier(estimators))])

vote.fit(data,target)

predictions=vote.predict(eval_data)



print_predictions(eval_target,predictions)



for name in pipes.keys():

    print(name)

    predictions=pipes[name].predict(eval_data)

    print_predictions(eval_target,predictions)

    print()
from itertools import combinations



final_estimators=['random_forest','xgb','deep_nn','elasticnet']



combos=[]

for L in range(2, len(final_estimators)+1):

    for subset in combinations(final_estimators, L):

        combos.append(list(subset))





combo_names=[]

auc=[]

f1=[]

logloss=[]

accuracy=[]



for combo in combos:

    estimators=[(name,clone(pipes[name].named_steps[name])) for name in combo] 

    vote=Pipeline(steps=[('data_prep',transformer),('voter',VotingClassifier(estimators))])

    vote.fit(data,target)

    predictions=vote.predict(eval_data)



    auc.append(roc_auc_score(eval_target,predictions))

    accuracy.append(accuracy_score(eval_target,predictions))

    logloss.append(log_loss(eval_target,predictions))

    f1.append(f1_score(eval_target,predictions))

    combo_names.append(str(list(combo)))

    

score=pd.DataFrame()

score['combo']=combo_names

score['auc']=auc

score['f1']=f1

score['accuracy']=accuracy



score


