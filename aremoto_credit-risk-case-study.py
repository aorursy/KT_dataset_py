import pandas as pd

import numpy as np

from pprint import pprint

import matplotlib.pyplot as plt

import seaborn as sns



import holoviews as hv

hv.extension('bokeh', 'matplotlib', logo=False)



# Avoid warnings to show up (trick for the final notebook on kaggle)

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/hmeq.csv', low_memory=False) # No duplicated columns, no highly correlated columns

df.drop('DEBTINC', axis=1, inplace=True) # The meaning of this variable is not clear. Better drop it for the moment

df.dropna(axis=0, how='any', inplace=True)
df[df['BAD']==0].drop('BAD', axis=1).describe().style.format("{:.2f}")
df[df['BAD']==1].drop('BAD', axis=1).describe().style.format("{:.2f}")
df.loc[df.BAD == 1, 'STATUS'] = 'DEFAULT'

df.loc[df.BAD == 0, 'STATUS'] = 'PAID'
g = df.groupby('REASON')

g['STATUS'].value_counts(normalize=True).to_frame().style.format("{:.1%}")
g = df.groupby('JOB')

g['STATUS'].value_counts(normalize=True).to_frame().style.format("{:.1%}")
%%opts Bars[width=700 height=400 tools=['hover'] xrotation=45]{+axiswise +framewise}



# Categorical



cols = ['REASON', 'JOB']



dd={}



for col in cols:



    counts=df.groupby(col)['STATUS'].value_counts(normalize=True).to_frame('val').reset_index()

    dd[col] = hv.Bars(counts, [col, 'STATUS'], 'val') 

    

var = [*dd]

kdims=hv.Dimension(('var', 'Variable'), values=var)    

hv.HoloMap(dd, kdims=kdims)
%%opts Histogram[width=700 height=400 tools=['hover'] xrotation=0]{+axiswise +framewise}



g = df.groupby('STATUS')



cols = ['LOAN',

        'MORTDUE', 

        'VALUE',

        'YOJ',

        'DEROG',

        'DELINQ',

        'CLAGE',

        'NINQ',

        'CLNO']

dd={}



# Histograms

for col in cols:

    

    freq, edges = np.histogram(df[col].values)

    dd[col] = hv.Histogram((edges, freq), label='ALL Loans').redim.label(x=' ')

    

    freq, edges = np.histogram(g.get_group('PAID')[col].values, bins=edges)

    dd[col] *= hv.Histogram((edges, freq), label='PAID Loans').redim.label(x=' ')

    

    freq, edges = np.histogram(g.get_group('DEFAULT')[col].values, bins=edges)

    dd[col] *= hv.Histogram((edges, freq), label='DEFAULT Loans' ).redim.label(x=' ')   

    

var = [*dd]

kdims=hv.Dimension(('var', 'Variable'), values=var)    

hv.HoloMap(dd, kdims=kdims)
%%opts Scatter[width=500 height=500 tools=['hover'] xrotation=0]{+axiswise +framewise}



g = df.groupby('STATUS')



cols = ['LOAN',

        'MORTDUE',

        'VALUE',

        'YOJ',

        'DEROG',

        'DELINQ',

        'CLAGE',

        'NINQ',

        'CLNO']



import itertools

prod = list(itertools.combinations(cols,2))



dd = {}



for p in prod:

    dd['_'.join(p)] = hv.Scatter(g.get_group('PAID')[list(p)], label='PAID Loans').options(size=5)

    dd['_'.join(p)] *= hv.Scatter(g.get_group('DEFAULT')[list(p)], label='DEFAULT Loans').options(size=5, marker='x')

    

var = [*dd]

kdims=hv.Dimension(('var', 'Variable'), values=var)    

hv.HoloMap(dd, kdims=kdims).collate()
g=sns.PairGrid(df.drop('BAD',axis=1), hue='STATUS', diag_sharey=False, palette={'PAID': 'C0', 'DEFAULT':'C1'})

g.map_lower(sns.kdeplot)

g.map_upper(sns.scatterplot)

g.map_diag(sns.kdeplot, lw=3)

g.add_legend()

plt.show()
cols=['YOJ', 'CLAGE', 'NINQ']



for col in cols:

    

    plt.figure(figsize=(15,5))



    sns.violinplot(x='JOB', y=col, hue='STATUS',

                   split=True, inner="quart",  palette={'PAID': 'C0', 'DEFAULT':'C1'},

                   data=df)

    

    sns.despine(left=True)
def compute_corr(df,size=10):

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot'''

    import scipy

    import scipy.cluster.hierarchy as sch

    

    corr = df.corr()

    

    # Clustering

    d = sch.distance.pdist(corr)   # vector of ('55' choose 2) pairwise distances

    L = sch.linkage(d, method='complete')

    ind = sch.fcluster(L, 0.5*d.max(), 'distance')

    columns = [df.select_dtypes(include=[np.number]).columns.tolist()[i] for i in list((np.argsort(ind)))]

    

    # Reordered df upon custering results

    df = df.reindex(columns, axis=1)

    

    # Recompute correlation matrix w/ clustering

    corr = df.corr()

    #corr.dropna(axis=0, how='all', inplace=True)

    #corr.dropna(axis=1, how='all', inplace=True)

    #corr.fillna(0, inplace=True)

    

    #fig, ax = plt.subplots(figsize=(size, size))

    #img = ax.matshow(corr)

    #plt.xticks(range(len(corr.columns)), corr.columns, rotation=45);

    #plt.yticks(range(len(corr.columns)), corr.columns);

    #fig.colorbar(img)

    

    return corr
%%opts HeatMap [tools=['hover'] colorbar=True width=500  height=500 toolbar='above', xrotation=45, yrotation=45]



corr=compute_corr(df)

corr=corr.stack(level=0).to_frame('value').reset_index()

hv.HeatMap(corr).options(cmap='Viridis')
import pandas as pd

import numpy as np

from pprint import pprint

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.metrics import classification_report
df=pd.read_csv('../input/hmeq.csv', low_memory=False) # No duplicated columns, no highly correlated columns

df=pd.get_dummies(df, columns=['REASON','JOB'])

df.drop('DEBTINC', axis=1, inplace=True)

df.dropna(axis=0, how='any', inplace=True)

y = df['BAD']

X = df.drop(['BAD'], axis=1)
def cross_validate_model(model, X, y, 

                         scoring=['f1', 'precision', 'recall', 'roc_auc'], 

                         cv=12, n_jobs=-1, verbose=True):

    

    scores = cross_validate(pipe, 

                        X, y, 

                        scoring=scoring,

                        cv=cv, n_jobs=n_jobs, 

                        verbose=verbose,

                        return_train_score=False)



    #sorted(scores.keys())

    dd={}

    

    for key, val in scores.items():

        if key in ['fit_time', 'score_time']:

            continue

        #print('{:>30}: {:>6.5f} +/- {:.5f}'.format(key, np.mean(val), np.std(val)) )

        name = " ".join(key.split('_')[1:]).capitalize()

        

        dd[name] = {'value' : np.mean(val), 'error' : np.std(val)}

        

    return  pd.DataFrame(dd)    

    #print()

    #pprint(scores)

    #print()
def plot_roc(model, X_test ,y_test, n_classes=0):

    

    from sklearn.metrics import roc_curve, auc

    

    """

    Target scores, can either be probability estimates 

    of the positive class, confidence values, or 

    non-thresholded measure of decisions (as returned 

    by “decision_function” on some classifiers).

    """

    try:

        y_score = model.decision_function(X_test)

    except Exception as e:

        y_score = model.predict_proba(X_test)[:,1]

    

    

    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())

    roc_auc = auc(fpr, tpr)



    # Compute micro-average ROC curve and ROC area

    #fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

    #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    

    #plt.figure()

    lw = 2

    plt.plot(fpr, tpr, color='darkorange',

             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)



    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    #plt.show()

    

# shuffle and split training and test sets

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,

#                                                    random_state=0)
def plot_confusion_matrix(model, X_test ,y_test,

                          classes=[0,1],

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    

    import itertools

    from sklearn.metrics import confusion_matrix

    

    y_pred = model.predict(X_test)

    

    # Compute confusion matrix

    cm = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)

    

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #    print("Normalized confusion matrix")

    #else:

    #    print('Confusion matrix, without normalization')



    #print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
def feature_importance(coef, names, verbose=False, plot=True):

    

    #importances = model.feature_importances_



    

    

    #std = np.std([tree.feature_importances_ for tree in model.estimators_],

    #             axis=0)

    indices = np.argsort(coef)[::-1]

    

    if verbose:

    

        # Print the feature ranking

        print("Feature ranking:")

    

        for f in range(len(names)):

            print("{:>2d}. {:>15}: {:.5f}".format(f + 1, names[indices[f]], coef[indices[f]]))

        

    if plot:

        

        # Plot the feature importances of the forest

        #plt.figure(figsize=(5,10))

        plt.title("Feature importances")

        plt.barh(range(len(names)), coef[indices][::-1], align="center")

        #plt.barh(range(X.shape[1]), importances[indices][::-1],

        #         xerr=std[indices][::-1], align="center")

        plt.yticks(range(len(names)), names[indices][::-1])

        #plt.xlim([-0.001, 1.1])

        #plt.show()
def plot_proba(model, X, y, bins=40, show_class = 1):

    

    from sklearn.calibration import CalibratedClassifierCV

    

    model = CalibratedClassifierCV(model)#, cv='prefit')

    

    model.fit(X, y)

    

    proba=model.predict_proba(X)

    

    if show_class == 0:

        sns.kdeplot(proba[y==0,0], shade=True, color="r", label='True class')

        sns.kdeplot(proba[y==0,1], shade=True, color="b", label='Wrong class')

        plt.title('Classification probability: Class 0')

    elif show_class == 1:

        sns.kdeplot(proba[y==1,1], shade=True, color="r", label='True class')

        sns.kdeplot(proba[y==1,0], shade=True, color="b", label='Wrong class')

        plt.title('Classification probability: Class 1')

    plt.legend()
from sklearn.linear_model import LogisticRegression



steps = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),

         ('model', LogisticRegression(random_state=0))]



pipe = Pipeline(steps)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

pipe.fit(X_train, y_train)
plt.figure(figsize=(15,10))



plt.subplot(221)

plot_roc(pipe, X_test ,y_test)



plt.subplot(222)

plot_confusion_matrix(pipe, X_test ,y_test, normalize=True)



plt.subplot(223)

plot_proba(pipe, X_test, y_test)



plt.subplot(224)

feature_importance(pipe.named_steps['model'].coef_[0], X.columns)



plt.tight_layout()
logit_xval_res = cross_validate_model(pipe, X, y, verbose=False)

logit_xval_res.T[['value','error']].style.format("{:.2f}")
from sklearn.linear_model import SGDClassifier



steps = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),

         ('model', SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, tol=1e-3, random_state=0))]



pipe = Pipeline(steps)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

pipe.fit(X_train, y_train)
plt.figure(figsize=(15,10))



plt.subplot(221)

plot_roc(pipe, X_test ,y_test)



plt.subplot(222)

plot_confusion_matrix(pipe, X_test ,y_test, normalize=True)



plt.subplot(223)

plot_proba(pipe, X_test, y_test)



plt.subplot(224)

feature_importance(pipe.named_steps['model'].coef_[0], X.columns)



plt.tight_layout()
sgd_xval_res = cross_validate_model(pipe, X, y, verbose=False)

sgd_xval_res.T[['value','error']].style.format("{:.2f}")
from sklearn.svm import SVC



steps = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),

         ('model', SVC(random_state=0, kernel='linear', probability=True))]



pipe = Pipeline(steps)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

pipe.fit(X_train, y_train)
plt.figure(figsize=(15,10))



plt.subplot(221)

plot_roc(pipe, X_test ,y_test)



plt.subplot(222)

plot_confusion_matrix(pipe, X_test ,y_test, normalize=True)



plt.subplot(223)

plot_proba(pipe, X_test, y_test)



plt.subplot(224)

feature_importance(pipe.named_steps['model'].coef_[0], X.columns)



plt.tight_layout()
svc_xval_res = cross_validate_model(pipe, X, y, verbose=False)

svc_xval_res.T[['value','error']].style.format("{:.2f}")
from sklearn.ensemble import GradientBoostingClassifier



steps = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),

         ('model', GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, random_state=0))]



pipe = Pipeline(steps)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

pipe.fit(X_train, y_train)
plt.figure(figsize=(15,10))



plt.subplot(221)

plot_roc(pipe, X_test ,y_test)



plt.subplot(222)

plot_confusion_matrix(pipe, X_test ,y_test, normalize=True)



plt.subplot(223)

plot_proba(pipe, X_test, y_test)



plt.subplot(224)

feature_importance(pipe.named_steps['model'].feature_importances_, X.columns)



plt.tight_layout()
gbc_xval_res = cross_validate_model(pipe, X, y, verbose=False)

gbc_xval_res.T[['value','error']].style.format("{:.2f}")
from sklearn.ensemble import RandomForestClassifier



steps = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),

         ('model', RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=0))]



pipe = Pipeline(steps)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

pipe.fit(X_train, y_train)
plt.figure(figsize=(15,10))



plt.subplot(221)

plot_roc(pipe, X_test ,y_test)



plt.subplot(222)

plot_confusion_matrix(pipe, X_test ,y_test, normalize=True)



plt.subplot(223)

plot_proba(pipe, X_test, y_test)



plt.subplot(224)

feature_importance(pipe.named_steps['model'].feature_importances_, X.columns)



plt.tight_layout()
rfc_xval_res = cross_validate_model(pipe, X, y, verbose=False)

rfc_xval_res.T[['value','error']].style.format("{:.2f}")
from sklearn.ensemble import ExtraTreesClassifier



steps = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),

         ('model', ExtraTreesClassifier(n_estimators=250, n_jobs=-1, random_state=0, class_weight='balanced'))]



pipe = Pipeline(steps)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

pipe.fit(X_train, y_train)
plt.figure(figsize=(15,10))



plt.subplot(221)

plot_roc(pipe, X_test ,y_test)



plt.subplot(222)

plot_confusion_matrix(pipe, X_test ,y_test, normalize=True)



plt.subplot(223)

plot_proba(pipe, X_test, y_test)



plt.subplot(224)

feature_importance(pipe.named_steps['model'].feature_importances_, X.columns)



plt.tight_layout()
ert_xval_res = cross_validate_model(pipe, X, y, verbose=False)

ert_xval_res.T[['value','error']].style.format("{:.2f}")
from collections import OrderedDict



res_comp = OrderedDict([

    ('Logistic regression'              , logit_xval_res[1:]),

    ('SGD classifier'                   , sgd_xval_res[1:]  ),

    ('Supporting vector classifier'     , svc_xval_res[1:]  ),

    ('Random forest classifier'         , rfc_xval_res[1:]  ),

    ('Extermely random tree classifier' , ert_xval_res[1:]  ),

    ('Gradient boost classifier'        , gbc_xval_res[1:]  ),

])



new_columns = {'level_0' : 'Model'}



pd.concat(res_comp).reset_index().drop('level_1', axis=1).rename(columns=new_columns).set_index('Model').sort_values('F1', ascending=False).style.format("{:.2f}")