import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot') # if error, use plt.style.use('ggplot') instead

import seaborn as sns
sns.set_style("whitegrid")     # need grid to plot seaborn plot

import scipy.stats as ss
import math

import sklearn as skl
import sklearn.metrics as sklm
import sklearn.feature_selection as fs
import sklearn.model_selection as ms

import sklearn.preprocessing as prep
import sklearn.decomposition as decomp

import sklearn.pipeline as pipe

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
glbl = {}   # define dict
glbl['show_figs'] = 1      # flag to enable printing figures
glbl['n_jobs'] = 1        # -1 to use all available CPUs
glbl['random_state'] = 5   # = None if we dont need demo mode
glbl['n_iter'] = 100        # how many search iterations
glbl['n_splits']=10            # cross validation splits
path = '../input/train.csv'
data1 = pd.read_csv(path, sep=',', index_col = 'PassengerId')
del(path)

# load the competition test data
path = '../input/test.csv'
data2 = pd.read_csv(path, sep=',', index_col = 'PassengerId')
del(path)

print('data1 shape:', data1.shape)
print('data2 shape:', data2.shape)
data = data1.append(data2, sort=False)  # Append rows of data2 to data1

# clean workspace
del(data1, data2)

print('data shape:', data.shape)
data.dtypes
data.sample(10)     # take a random sample of 10 observations
# drop useless features
data.drop(labels=['Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# see if we have ?s
print ('number of nans in data:', (data.astype(np.object) == '?').any().sum() ) # we have no ?s
data.isnull().sum() # check each column.
for Pcl in data.Pclass.unique():   # 1, 2, 3
    med = data[['Fare']].where(data.Pclass==Pcl).median()    # get median over all data for that class
    data.loc[ ((data.Fare.isnull() == True) & (data.Pclass==Pcl)) , 'Fare'] = med[0] # med is series
# clean workspace:
del(Pcl, med)

# see if we have nans:
data.isnull().sum() # check each column.
def extract_title(x):   # x is entire row
    string=x['Title']
    ix=string.find(".")    # use .find to find the first dot
    for i in range(0,ix):
        if (string[ix-i] == ' '):  # if we find space, then stop iterating
            break                   # break out of for-loop
    return string[(ix-i+1):ix]  # return everything after space up till before the dot

data['Title'] = data.Name  # for now copy name directly
data['Title']=data.apply(extract_title, axis=1)     # axis = 1 : apply function to each row
data.drop(labels=['Name'], axis=1, inplace=True)  # we can even drop the 'Name' column now

data.Title.unique()   # lets see the unique titles in our dataset
# standardize 'Title'
def standardize_title(x):   # x is an entire row
    Title=x['Title']
    
    if x.Sex == 'male':
        if Title != 'Master':   # we can keep 'Master' title, but we want to change all others to Mr
            return 'Mr'
        else:
            return Title
    if x.Sex == 'female':
        if Title in ['Miss', 'Mlle', 'Ms']:
            return 'Miss'
        else:
            return 'Mrs'

data['Title']=data.apply(standardize_title, axis=1)     # axis = 1 : apply function to each row

data.Title.unique()   # lets see the unique titles in our dataset
if glbl['show_figs']:
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    data[['Age', 'Title']].boxplot(by = 'Title', ax=ax)
if glbl['show_figs']:
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    plt.imshow(data.corr(), cmap=plt.cm.Blues, interpolation='nearest') 	# plots the correlation matrix of data
    plt.colorbar()
    tick_marks = [i for i in range(len(data.columns))]
    plt.xticks(tick_marks, data.columns, rotation='vertical')
    plt.yticks(tick_marks, data.columns)
if glbl['show_figs']:
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    data[['Age', 'SibSp']].boxplot(by = 'SibSp', ax=ax)
for titl in data.Title.unique():
    med = data[['Age']].where(data.Title==titl).median()
    data.loc[ (data.Age.isnull() == True) & (data.Title==titl) , 'Age'] = med[0] # med is a series. must be a scalar
del(titl, med)

# let us see if we have nans:
data.isnull().sum() # check each column.
data['FamSize'] = data.SibSp + data.Parch + 1
print(data.columns)
data.dtypes
# nominal
data.Sex = data.Sex.map({'female':0, 'male':1})

data = pd.get_dummies(data,columns=['Title'])       # turn nominal to bool

# ordinal
ordered_categs = [1, 2, 3]  # categories for Pclass
categs = pd.api.types.CategoricalDtype(categories = ordered_categs, ordered=True)
data.Pclass = data.Pclass.astype(categs) # ordinal
#
del(ordered_categs) # clean workspace

data.dtypes
def to_lowest_numeric(x):
    # x is a column
    if (x.apply(np.isreal).all(axis=0)) & ((str(x.dtypes) != 'category')): # if this column is numeric, but NOT categorical
        x = pd.to_numeric(x, errors='coerce', downcast='float') # first downcast floats
        x = pd.to_numeric(x, errors='coerce', downcast='unsigned') # now downcast ints
    
    # now to handle booleans:
    # if x has only the ints 0 and 1  OR  x has only 'True' and 'False' strings
    if set(x.unique()) == set(np.array([0, 1])) :
        x2=x.astype('bool')
        return x2
    elif set(x.unique()) == set(np.array(['True', 'False'])):
        x2 = x=='True'
        return x2
    else:
        return x
#

data = data.apply(to_lowest_numeric, axis=0)

# View Column DataTypes
data.dtypes
data1 = data.iloc[0:891, :]     # iloc is incl:excl
data2 = data.iloc[891:1309, :]

# we will split data1 into out train and test sets.
# we will use data2 for the Kaggle submission at the end

X = data1.drop(labels=['Survived'], axis=1)
y=data1['Survived']    # return a series
data2

# clean up Workspace
del(data, data1)

print('X shape:', X.shape)
print('y shape:', y.shape)
print('data2 shape:', data2.shape)
print ('y dtype:', y.dtypes ) 
y = to_lowest_numeric(y)
print ('y dtype:', y.dtypes ) # = 'bool'     this is correct.
# let us split the train:test ratio at 80:20
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, 
                                                       random_state = glbl['random_state'])

del(X,y) # clean up Workspace

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
X_train2 = X_train.copy()
print('X_train2 shape:', X_train2.shape)
X_train2.describe()
def all_violin(X):  # X is a dataframe
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    ax.set_title('Violin plots of quantitative variables')
    
    # lets store up all the columns that are from Quantitative variables
    # aka we will find columns that are not Nominal (bool dtype) or Ordinal (category dtype)
    dtypes = pd.DataFrame( X.dtypes )
    dtypes = dtypes.astype('str').values.reshape(-1,)
    valid = ( (dtypes != 'bool') & (dtypes != 'category') )
    
    X = X.iloc[:,valid]     # contains only columns that are not bool and not category
    
    ax = sns.violinplot(data=X)     # plot on single axis

if glbl['show_figs']:
    all_violin(X_train2)
def all_box(X):  # X is a dataframe
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    ax.set_title('Boxplots of quantitative variables')
    
    # lets store up all the columns that are from Quantitative variables
    # aka we will find columns that are not Nominal (bool dtype) or Ordinal (category dtype)
    dtypes = pd.DataFrame( X.dtypes )
    dtypes = dtypes.astype('str').values.reshape(-1,)
    valid = ( (dtypes != 'bool') & (dtypes != 'category') )
    
    X = X.iloc[:,valid]     # contains only columns that are not bool and not category
    
    ax = sns.boxplot(data=X)     # plot on single axis

if glbl['show_figs']:
    all_box(X_train2)
norm = prep.QuantileTransformer()
f=X_train2.loc[:,['Age', 'SibSp', 'Parch', 'Fare', 'FamSize']]
norm.fit( np.array(f) )
f = norm.transform(np.array(f))
X_train2.loc[:,['Age', 'SibSp', 'Parch', 'Fare', 'FamSize']]=f

del(f)

if glbl['show_figs']:
    all_violin(X_train2)
X_train2.describe()
scaler = prep.StandardScaler()

f = X_train2.loc[:,['Age','SibSp','Parch','Fare','FamSize']]

scaler.fit( f )
f = scaler.transform( f )
f = pd.DataFrame (f)

X_train2.loc[:,['Age','SibSp','Parch','Fare','FamSize']] = f.values

# clean up workspace
del(f)

X_train2.describe()
if glbl['show_figs']:
    all_violin(X_train2)
if glbl['show_figs']:
    all_box(X_train2)
pca = decomp.PCA()
pca = pca.fit(X_train2)

def plotPCA_explained(mod):
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    plt.suptitle('Scree plot of explained variance per principle component')
    ax.set_xlabel('number of components')   # Set text for the x axis
    ax.set_ylabel('explained variance')   # Set text for y axis
    comps = mod.explained_variance_ratio_
    x = range(len(comps))
    x = [y + 1 for y in x]          
    plt.plot(x,comps)
#
print ('X_train2 shape:', X_train2.shape)
if glbl['show_figs']:
    plotPCA_explained(pca)
pca_6 = decomp.PCA(n_components = 6)
pca_6.fit(X_train2)
X_trainPCA = pca_6.transform(X_train2)
print ('X_train2 shape:' ,X_trainPCA.shape)
def drawPCAVectors(transformed_features, components_, columns):
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    plt.suptitle('Features in principle component space')
    ax.set_xlabel('PC1')   # Set text for the x axis
    ax.set_ylabel('PC2')   # Set text for y axis
    num_columns = len(columns)
    # This funtion will project your *original* feature (columns) onto your principal component feature-space, so that you can visualize how "important" each one was in the multi-dimensional scaling
    # Scale the principal components by the max value in the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])
    ## visualize projections
    # Sort each column by it's length. These are your *original* columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Features by importance:\n", important_features)
    ax = plt.axes()
    for i in range(num_columns):
        # Use an arrow to project each original feature as a labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)
    return ax

if glbl['show_figs']: 
    drawPCAVectors(X_trainPCA, pca_6.components_, X_train2.columns)

# clean workspace
del(X_trainPCA)

# until now we used X_train2, which was a copy of X_train. X_train2 was used to illustrate thet transforms that we will be using in the pipeline. We did not want to alter X_train in any way, but now we can actually implement our transforms as part of our pipeline for real. We can therefore delete X_train2
del(X_train2)
def get_estimator(est, y_train):
    #
    ratio_classes =  pd.Series(y_train).value_counts(normalize=True)
    #
    if (est == 'LogisticRegression') | (est == 'logit'):
        from sklearn.linear_model import LogisticRegression as estimator
        parameter_dist = {
                'penalty' : ['l2'], # 'penalty' : ['l1', 'l2'],     # default l2
                'class_weight' : [{0:ratio_classes[0], 1:ratio_classes[1]}], #class_weight : dict or ‘balanced’, default: None
                'C': ss.expon(scale=100), # must be a positive float
        }
    elif (est == 'KNeighborsClassifier') | (est == 'knc'):
        from sklearn.neighbors import KNeighborsClassifier as estimator
        parameter_dist = {
                'n_neighbors' : ss.randint(1, 11),
                # 'weights' : ['uniform', 'distance'],
                # 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }
    elif (est == 'LinearSVC') | (est == 'lsvc'):
        from sklearn.svm import LinearSVC as estimator
        parameter_dist = {
                'C': ss.expon(scale=10),
                # 'penalty' : ['l1', 'l2'],
                # 'multi_class' : ['ovr', 'crammer_singer'],
                'class_weight' : [{0:ratio_classes[0], 1:ratio_classes[1]}], #class_weight : dict or ‘balanced’
        }
    elif (est == 'SVC') | (est == 'svc'):
        from sklearn.svm import SVC as estimator
        parameter_dist = {
                'C': ss.expon(scale=10),
                'gamma' : ss.expon(scale=0.1), # float, optional (default=’auto’). If gamma is ‘auto’ then 1/n_features will be used instead.
                # 'kernel' : ['rbf', 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                'class_weight' : [{0:ratio_classes[0], 1:ratio_classes[1]}], #class_weight : dict or ‘balanced’
        }
    elif (est == 'DecisionTreeClassifier') | (est == 'dtc'):
        from sklearn.tree import DecisionTreeClassifier as estimator
        parameter_dist = {
                'criterion' : ['entropy'], # 'criterion': ['gini', 'entropy'],
                # 'splitter' : ['best', 'random'],
                # 'max_depth': [None, 3],'min_samples_split': ss.randint(2, 11),
                'min_samples_leaf': ss.randint(1, 11),
                'max_features': ss.uniform(0.0, 1.0), # we have to make this a float
                'class_weight' : [{0:ratio_classes[0], 1:ratio_classes[1]}], #class_weight : dict or ‘balanced’
        }
    elif (est == 'RandomForestClassifier') | (est == 'rfc'):
        from sklearn.ensemble import RandomForestClassifier as estimator
        parameter_dist = {
                # 'max_depth': [None, 3],
                'n_estimators' : ss.randint(8, 20), # 'n_estimators' : integer, optional (default=10)
                'criterion': ['gini', 'entropy'],
                'max_features': ss.uniform(0.0, 1.0), # we have to make this a float
                'min_samples_split': ss.randint(2, 11),
                'min_samples_leaf': ss.randint(1, 11),
                "bootstrap": [True, False],
                'class_weight' : [{0:ratio_classes[0], 1:ratio_classes[1]}], #class_weight : dict or ‘balanced’
        }
    #
    estimator=estimator()
    
    if 'random_state' in estimator.get_params():
        estimator.set_params(random_state=glbl['random_state'])
    #
    return estimator, parameter_dist



def createPipes(y_train, idx=0):
    
    '''
    idx specifies which estimator we want to use.
    In this project we chose to demonstrate 6 classifiers:
        Logistic Regression     idx=0
        k-nearest neighbours    idx=1
        linear SVM              idx=2
        SVM                     idx=3
        Decision Tree           idx=4
        Random Forest           idx=5
    
    Outputs: 
        pipe1 { a pipeline using RFE for dimensionality reduction }
        param_dist1 { parameter distributions of the components in pipe1 }
        pipe2 { a pipeline using PCA for dimensionality reduction }
        param_dist2 { parameter distributions of the components in pipe2 }
    '''
    
    # the normalizer
    norm = prep.QuantileTransformer(random_state=glbl['random_state'])
    # the scaler
    scaler=prep.StandardScaler()
    # the classfier estimator as a function of input 'idx'
    classifiers = ['logit', 'knc', 'lsvc', 'svc', 'dtc', 'rfc'] # let us test these classfiers
    estimator, est_param_dist = get_estimator(classifiers[idx], y_train)
    # dimensionality reduction methods
    rfe = fs.RFE(estimator = estimator)
    rfe_param_dist = {
        'n_features_to_select': ss.randint(1,11),   # since we have 11 features
    }
    pca = decomp.PCA()
    pca_param_dist = {
        'n_components': ss.randint(1,10),   # since we have 11 features
        'random_state' : [glbl['random_state']]
    }
    
    #pipe1 uses RFE with estimator, pipe2 uses PCA with estimator
    
    pipe1 = pipe.Pipeline([
            ('norm', norm),
            ('scaler', scaler),
            ('rfe', rfe),
            ('est', estimator)
    ])
    
    pipe2 = pipe.Pipeline([
            ('norm', norm),
            ('scaler', scaler),
            ('pca', pca),
            ('est', estimator)
    ])
    
    pca_param_dist = {f'pca__{k}': v for k, v in pca_param_dist.items()}    # add 'pca__' string to all keys
    rfe_param_dist = {f'rfe__{k}': v for k, v in rfe_param_dist.items()}    # add 'rfe__' string to all keys
    est_param_dist = {f'est__{k}': v for k, v in est_param_dist.items()}    # add 'est__' string to all keys
    # adding the transformer name in the parameter name is required for pipeline
    
    # merge dictionaries
    param_dist1 = {**rfe_param_dist,  **est_param_dist}
    param_dist2 = {**pca_param_dist,  **est_param_dist}
    
    return pipe1, param_dist1, pipe2, param_dist2


pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=0)  # create pipelines for logit model

print('pipe1\n',pipe1)
print('estimator of pipe1:', pipe1.named_steps['est'])
print('------------------------------------------------')
print('pipe2\n',pipe2)
print('estimator of pipe2:', pipe2.named_steps['est'])
def modelSel(pipeline, param_dist, X_train, y_train):
    
    '''
    This function performs model selection given a pipeline and it's distribution
    inputs:
        pipeline { the pipeline to undergo model selection }
        param_dist { the parameter distributions of the pipeline }
    outputs:
        inner { the model selection object. Contains parameters like best_estimator, best_params, best_index }
        outer { the outer CV results }
        this function also prints out the training score, as well as the testing score for the best estimator
    '''
    
    # define the randomly sampled folds for the inner and outer Cross Validation loops:
    insideCV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
    outsideCV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
    
    ## Perform the Random search over the parameters
    inner = ms.RandomizedSearchCV(estimator = pipeline,
                                param_distributions = param_dist,
                                n_iter=glbl['n_iter'], # Number of models that are tried
                                cv = insideCV, # Use the inside folds
                                scoring = 'accuracy',
                                n_jobs=glbl['n_jobs'],
                                return_train_score = True,
                                random_state=glbl['random_state'])
    # The cross validated random search object, 'inner', has been created.
    
    # Fit the cross validated grid search over the data 
    inner.fit(X_train, y_train)
    # we have now scored each of the n_iter models (hyper-param combo) and we have an average score for each. we can use these scores as a model selection step, or we can feed these scores into an optimization algorithm. we wont use optim algo in this project, so we use best_estimator as our selected estimator.
    
    print('best accuracy on inner (train) set', inner.best_score_)
    
    # -------------------------------------------------
    
    # the inner loop evaluates model performance. we decided to let it do our model selection too.
    # the estimate of the classifier is not reliable though. So we need to have an
    # outer CV where we evaluate the 'best_estimator'
    
    outer = ms.cross_val_score(inner.best_estimator_, X_train, y_train, cv = outsideCV,
                               n_jobs=glbl['n_jobs'])
    
    print('For outer (testing) set:')
    #print('Outcomes by cv fold')
    #for i, x in enumerate(outer):
    #    print('Fold %2d    %4.3f' % (i+1, x))
    print('Mean outer performance metric = %4.7f' % np.mean(outer))
    print('stddev of the outer metric       = %4.7f' % np.std(outer))
    #
        
    return inner, outer
#

inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)
inner=inner2
outer=outer2

# clean worspace
del(pipe1, param_dist1, pipe2, param_dist2)
del(inner1, outer1, inner2, outer2)


# the results of each model in the pipeline
inner_results = pd.DataFrame( inner.cv_results_ )# the score for each model. there are n_iter models
# look at 3 random models in the pipeline
inner_results.sample(3)
# print the parameter values of the best model
print( inner.best_estimator_ )
# print parameters of the best model
print(inner.best_params_)
# 'inner' contains many models. What is the index of the best model?
print( inner.best_index_ ) 
if glbl['show_figs']:
    
    # lets look at the simplest check first: check 3
    innerTest_mean, innerTest_std = inner_results.loc[
        inner.best_index_,['mean_test_score','std_test_score']]
    outerTest_mean = np.mean(outer)
    outerTest_std = np.std(outer)
    
    print('The inner CV mean and standard deviation are, respectively:')
    print( innerTest_mean , innerTest_std)
    print('The outer CV mean and standard deviation are, respectively:')
    print( outerTest_mean , outerTest_std)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = ms.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, random_state=glbl['random_state'],
        train_sizes=train_sizes)
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
             label="Inner CV (Training) score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Outer CV (Testing) score")
    
    plt.legend(loc="best")
    return plt
#
CV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
if glbl['show_figs']:
    # check 1
    # Plot learning curves
    # Learning curves are a good way to see the overfitting effect on the training set and the effect of the training size on the accuracy.
    
    plot_learning_curve(inner.best_estimator_, "logit learning curves", X_train, 
                            y_train, cv=CV, n_jobs=glbl['n_jobs'])
def plot_validation_curve(estimator, X, y, param_name, param_range, cv=10, scoring="accuracy",n_jobs=1):
    plt.figure()
    train_scores, test_scores = ms.validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # the following is used to generate the title (objNmFull) of the figure
    if (  str (type(estimator)) == "<class 'sklearn.pipeline.Pipeline'>"  ):
        objNm = param_name.split('__')   # what is before the '__' ?
        # which object in the pipeline are we considering?
        objStr=str( estimator.named_steps[objNm[0]] )
        # split at '(' and keep what is before
        objNmFull = str( "Validation Curve for Pipeline for " + objStr.split('(')[0] )
        objNm = objNm[1]
    else:
        objNm = param_name
        objStr=str( estimator )
        objNmFull = str( "Validation Curve for " + objStr.split('(')[0] )
    objNmFull = objNmFull + ' for parameter ' + objNm
    
    plt.title(objNmFull)        # make title
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Inner CV (Training) score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Outer CV (Testing) score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
#

# lets first see the effect of 'C' on score
CV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
plot_validation_curve(inner.best_estimator_, X_train, y_train,
                      'est__C', np.logspace(-1, 5, 20), cv=CV,
                      scoring="accuracy", n_jobs=glbl['n_jobs'])
# now lets see the effect of n_components on score
plot_validation_curve(inner.best_estimator_, X_train, y_train,
                      'pca__n_components',
                      np.linspace(1, 10, 10, dtype = int),
                      cv=CV, scoring="accuracy",n_jobs=glbl['n_jobs'])
best_logit = inner.best_estimator_

# clean workspace
del(inner, outer)

print('best_logit\n', best_logit)
pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=1)  # create pipelines for knc model

# inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
# this fails, because RFE wont work with knc:
# RuntimeError: The classifier does not expose "coef_" or "feature_importances_" attributes

print('pipe2')
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)
# lets name rename inner2 as:
best_knc = inner2.best_estimator_

print('\nbest pipeline saved.')
pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=2)  # create pipelines for lsvc model
print('pipe1')
inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
print('pipe2')
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)

# lets name rename inner1 as:
best_lsvc = inner1.best_estimator_
print('\nbest pipeline saved.')
pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=3)  # create pipelines for svc model

#inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
# this fails, because RFE wont work with svc:
# RuntimeError: The classifier does not expose "coef_" or "feature_importances_" attributes

print('pipe2')
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)

# lets name rename inner2 as:
best_svc = inner2.best_estimator_
print('\nbest pipeline saved.')
pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=4)  # create pipelines for dtc model
print('pipe1')
inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
print('pipe2')
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)
# lets name rename inner1 as:
best_dtc = inner1.best_estimator_
print('\nbest pipeline saved.')
pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=5)  # create pipelines for rfc model

print('pipe1')
inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
print('pipe2')
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)

# lets name rename inner1 as:
best_rfc = inner1.best_estimator_
print('\nbest pipeline saved.')

# clean workspace
del(pipe1, param_dist1, pipe2, param_dist2)
del(inner1,outer1,inner2,outer2)
from sklearn.ensemble import VotingClassifier

# hard voting classifier
votingC_hard = VotingClassifier(
        estimators=[('logit', best_logit), ('knc', best_knc), 
                    ('lsvc', best_lsvc), ('svc', best_svc), 
                    ('dtc',best_dtc), ('rfc',best_rfc)],
        voting='hard', n_jobs=glbl['n_jobs'])
#
CV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
cv_estimate2 = ms.cross_val_score(votingC_hard, X_train, y_train, cv = CV, n_jobs=glbl['n_jobs'])
print('Outcomes by cv fold')
for i, x in enumerate(cv_estimate2):
    print('Fold %2d    %4.3f' % (i+1, x))
print('Mean performance metric = %4.3f' % np.mean(cv_estimate2))
print('stddev of the metric       = %4.3f' % np.std(cv_estimate2))
#
# soft voting classifier
votingC_soft = VotingClassifier(
        estimators=[('logit', best_logit), ('knc', best_knc),
                    #('lsvc', best_lsvc),    # AttributeError: 'LinearSVC' object has no attribute 'predict_proba'. we need to take this out.
                    ('svc', best_svc),      # predict_proba is not available when  probability=False   --- we will set this soon ...
                    ('dtc',best_dtc), ('rfc',best_rfc)],
        voting='soft', n_jobs=glbl['n_jobs'])
#

# for soft vote we need probabilities. the SVC can be set to have probabilities...
# probability : boolean, optional (default=False)
#    Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
# lets see the parameters i can set in VotingC_soft:
votingC_soft.get_params().keys()
# we need to set that 'svc__est__probability' to True
votingC_soft.set_params(svc__est__probability=True)

CV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
cv_estimate3 = ms.cross_val_score(votingC_soft, X_train, y_train, cv = CV)
print('Outcomes by cv fold')
for i, x in enumerate(cv_estimate3):
    print('Fold %2d    %4.3f' % (i+1, x))
print('Mean performance metric = %4.3f' % np.mean(cv_estimate3))
print('stddev of the metric       = %4.3f' % np.std(cv_estimate3))
#

del(cv_estimate2, cv_estimate3, i, x)
# First, we fit each model with ENTIRE train-set.
best_logit.fit(X_train, y_train)
best_knc.fit(X_train, y_train)
best_lsvc.fit(X_train, y_train)
best_svc.fit(X_train, y_train)
best_dtc.fit(X_train, y_train)
best_rfc.fit(X_train, y_train)
votingC_hard.fit(X_train, y_train)
votingC_soft.fit(X_train, y_train)


# now lets score each model on the test set

scores=[]
classifiers=pd.DataFrame( ['logit', 'knc', 'lsvc', 'svc', 'dtc', 'rfc', 'vch', 'vcs'] )
classifiers.columns = ['classifier']

scores.append( best_logit.score(X_test, y_test) )
scores.append( best_knc.score(X_test, y_test) )
scores.append( best_lsvc.score(X_test, y_test) )
scores.append( best_svc.score(X_test, y_test) ) 
scores.append( best_dtc.score(X_test, y_test) ) 
scores.append( best_rfc.score(X_test, y_test) )
scores.append( votingC_hard.score(X_test, y_test) ) 
scores.append( votingC_soft.score(X_test, y_test) )

scores = pd.DataFrame( scores )
scores.columns = ['score']

scoresdf = pd.concat([classifiers, scores], axis = 1)   # concatenate columns
# clean workspace
del(classifiers, scores)


# Lets sort the `scoresdf` dataframe
scoresdf.sort_values(by='score', inplace=True)
scoresdf.score = (scoresdf.score*100000).astype(int)/1000   # give each 3 decimal points



# we create a column chart of the scores for each estimator used
fig = plt.figure(figsize=(8,6))   # define (new) plot area
ax = fig.gca()   # define axis
sns.barplot(x = 'score', y='classifier', data = scoresdf, palette="Blues_d", ax=ax)
plt.title('Accuracy Score on the test-set by different classifiers \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Classifier')

for i, v in enumerate( scoresdf.score ):  # give each 3 decimal points
    ax.text(v + 0.5, i + .25, str(v), color='black', fontweight='bold')
# clean workspace
del(i, v, scoresdf)
# lets collect the names of the oringinal features
featureRanks = pd.DataFrame(X_train.columns)
featureRanks.columns=['feature']
# first we need to figure out which features were deleted by the RFE
featureRanks['support'] = best_dtc.named_steps['rfe'].support_
featureRanks['ranking'] = best_dtc.named_steps['rfe'].ranking_

# lets look at the feature importance ranking as per the RFC
featureRanks['importance'] = 0     # initialize
# now set the importance of features that were included
featureRanks.loc[featureRanks.support==True,'importance'] = best_dtc.named_steps['est'].feature_importances_

# lets sort by feature importances
featureRanks.sort_values(by='importance', inplace=True)
featureRanks.importance = (featureRanks.importance*100000).astype(int)/1000   # give each 3 decimal points

# we create a column chart of the importances for each feature
fig = plt.figure(figsize=(8,6))   # define (new) plot area
ax = fig.gca()   # define axis
sns.barplot(x = 'importance', y='feature', data = featureRanks, palette="Blues_d", ax=ax)
plt.title('Importance as per Decision Tree Classifier with RFE \n')
plt.xlabel('Importance [%]')
plt.ylabel('Feature')

for i, v in enumerate( featureRanks.importance ):  # give each 3 decimal points
    ax.text(v + 0.5, i + .25, str(v), color='black', fontweight='bold')

del(i,v, featureRanks)
# we will need to name the features used by the DTC

featureRanks2 = pd.DataFrame(X_train.columns)
featureRanks2.columns=['feature']
# first we need to figure out which features were deleted by the RFE
featureRanks2['support'] = best_dtc.named_steps['rfe'].support_
featureRanks2['ranking'] = best_dtc.named_steps['rfe'].ranking_
# now isolate feature names taht were used
names = featureRanks2.loc[featureRanks2.support==True,['feature']]
names = list(names.feature)

from sklearn.base import clone
dtc = clone ( best_dtc.named_steps['est'] )     # copies estimator wihtout pointing to it

dtc.fit(X_train.loc[:,names], y_train)   # refit the classifier with included data only
# note that DTC does not need transforms on X_train, so we can use it directly

import graphviz 

# Create DOT data
dot_data = skl.tree.export_graphviz(dtc, out_file=None,
                                    max_depth = None,
                                    feature_names = names,
                                    class_names = True,     # in ascending numerical order
                                    impurity=False,
                                    proportion=True,
                                    filled = True, rounded = True)
#

# Draw graph
graph = graphviz.Source(dot_data) 
# show graph
graph

# First, clean workspace
del(featureRanks2, dot_data, names)

# we dont need the train-set or test-set data at all anymore
del(X_train, y_train, X_test, y_test)


# we now use the testing data from Kaggle, which we have named 'X2'

y2 = data2.Survived
X2 = data2.drop(columns=['Survived'])
del(data2)

# we use our best classifier: the VCS
y_pred = votingC_soft.predict(X2)     # make the predictions based on X2

# now lets make the submission

submit = pd.DataFrame(y2)  # just quickly create a new dataFrame called submit
submit['PassengerId'] = X2.index
submit['Survived'] = y_pred

# lets reorder our columns... just to make it look nice
submit = submit[['PassengerId','Survived']]

# lets take a look at what our data looks  like
submit.head(5)
submit['Survived'] = submit.Survived.astype('uint8')
# lets take a look at what our data looks  like
submit.head(5)
#submit file
submit.to_csv("../working/submit.csv", index=False)

print("Submitted to 'Output'")