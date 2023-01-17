import time



import pandas as pd

import numpy as np



from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



from sklearn.feature_selection import SelectKBest



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.metrics import make_scorer

from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score



from sklearn import model_selection

from sklearn import metrics

from sklearn import preprocessing



from sklearn.pipeline import Pipeline



from imblearn.metrics import geometric_mean_score as gmean_score



from imblearn.under_sampling import RandomUnderSampler



from imblearn.over_sampling import SMOTE



from bokeh.models import Div

from bokeh import plotting

from bokeh import layouts
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=UserWarning)



plotting.output_notebook()
running_time = time.time()
def fraud_genuine_counter(y_list, names):

    mapper = {1: 'fraudulent', 0: 'genunine'}

    f = lambda y, n: y.map(mapper).value_counts().to_frame(n)

    y_mapped = [ f(y, n) for y, n in zip(y_list, names) ]

    y_counts = pd.concat(y_mapped, axis=1).T

    return y_counts
# from Vitor Gaboardi



def front(self, n):

    return self.iloc[:, :n]

    

def correlation(X,y,n,lim):

  data = pd.concat([X, y],axis=1)

  corr_matrix = data.corr().abs()

  y_corr = corr_matrix.loc[['Class']].sort_values(by=['Class'],axis=1,ascending=False).drop('Class',axis=1)

  n_best_features = list(front(y_corr,n).columns)



  data = data[n_best_features]

  corr_matrix = data.corr().abs()



  X = (corr_matrix < lim)



  return n_best_features
def parse_proba(proba, thr=0.5):

    preds = proba.copy()

    preds[preds >= thr] = 1

    preds[preds  < thr] = 0

    return preds



def compute_metric(truth, proba, thr, metric):

    pred = parse_proba(proba=proba, thr=thr)

    score = metric(truth, pred)

    return score



def compute_precision(truth, proba, thr):

    precision = compute_metric(truth, proba, thr, metrics.precision_score)

    return precision



def compute_recall(truth, proba, thr):

    recall = compute_metric(truth, proba, thr, metrics.recall_score)

    return recall



def compute_roc_auc(truth, proba, thr):

    roc_auc = compute_metric(truth, proba, thr, metrics.roc_auc_score)

    return roc_auc



def compute_gmean(truth, proba, thr):

    gmean = compute_metric(truth, proba, thr, gmean_score)

    return gmean
ylim = [-0.05, 1.05]
def plot_thr_precision_scores(thrs, pt, pv=None):

    plt.title('precision scores')

    plt.plot(thrs, pt, label='train')

    if pv is not None:

        plt.plot(thrs, pv, label='valid')

    plt.xlabel('threshold')

    plt.ylabel('precision')

    plt.ylim(ylim)

    plt.legend()
def plot_thr_recall_scores(thrs, rt, rv=None):

    plt.title('recall scores')

    plt.plot(thrs, rt, label='train')

    if rv is not None:

        plt.plot(thrs, rv, label='valid')

    plt.xlabel('threshold')

    plt.ylabel('recall')

    plt.ylim(ylim)

    plt.legend()
def plot_thr_precision_recall_scores(thrs, 

                                     pt=None, rt=None, 

                                     pv=None, rv=None, 

                                     pc=None, rc=None, # results of cross validation

                                     ylim=[-0.05, 1.05]):

    plt.title('precision/recall scores')



    a1 = 1.0

    a2 = 0.5

    a3 = 0.1



    if pv is not None :

        plt.plot(thrs, pv, label='precision valid', color='red', alpha=a1)

    if pc is not None:

        plt.plot(thrs, pc, label='precision cross', color='red', alpha=a2)

    if pt is not None:

        plt.plot(thrs, pt, label='precision train', color='red', alpha=a3)



    if rv is not None:

        plt.plot(thrs, rv, label='recall valid', color='green', alpha=a1)

    if rc is not None:

        plt.plot(thrs, rc, label='recall cross', color='green', alpha=a2)

    if rt is not None:

        plt.plot(thrs, rt, label='recall train', color='green', alpha=a3)



    plt.xlabel('threshold ({} points)'.format(len(thrs)))

    plt.ylabel('scores')

    plt.xlim([-0.05, 1.05])

    plt.ylim(ylim)

    plt.legend()
def plot_thr_gmean_scores(thrs, gmt=None, gmv=None, labels=False):

    plt.title('geometric mean scores')



    a_light = 0.2

    a_dark = 1.0



    if gmv is not None:

        plt.plot(thrs, gmv, label='gmean valid' if labels else 'valid', 

                 color='blue', alpha=a_dark)



    if gmt is not None:

        plt.plot(thrs, gmt, label='gmean train' if labels else 'train', 

                 color='blue', alpha=a_dark if gmv is None else a_light)



    plt.xlabel('threshold ({} points)'.format(len(thrs)))

    plt.ylabel('gmean score')

    plt.ylim(ylim)

    plt.legend()
def plot_thr_roc_auc_scores(thrs, roc_auc_train, roc_auc_valid=None):

    plt.title('roc auc scores')



    plt.plot(thrs, roc_auc_train, label='train')



    if roc_auc_valid is not None:

        plt.plot(thrs, roc_auc_valid, label='valid')



    plt.xlabel('threshold')

    plt.ylabel('roc auc score')

    plt.ylim(ylim)

    plt.legend()
def plot_roc_auc_curves(fpr_train, tpr_train, fpr_valid=None, tpr_valid=None):

    plt.title('roc curve - {} points'.format(len(fpr_train)))



    auc_t_roc = metrics.auc(fpr_train, tpr_train)

    plt.plot(fpr_train, tpr_train, label='train (auc: {:.3f})'.format(auc_t_roc))



    if fpr_valid is not None and tpr_valid is not None:

        auc_v_roc = metrics.auc(fpr_valid, tpr_valid)

        plt.plot(fpr_valid, tpr_valid, label='valid (auc: {:.3f})'.format(auc_v_roc))



    plt.xlabel('tpr')

    plt.ylabel('fpr')

    plt.ylim(ylim)

    plt.legend()
def plot_precision_recall_curves(rec_train, pre_train, rec_valid=None, pre_valid=None):

    plt.title('precision-recall curve - {} points'.format(len(pre_train)))

    

    auc_t_pr = metrics.auc(rec_train, pre_train)

    plt.plot(rec_train, pre_train, label='train (auc: {:.3f})'.format(auc_t_pr))



    if rec_valid is not None and pre_valid is not None:

        auc_v_pr = metrics.auc(rec_valid, pre_valid)

        plt.plot(rec_valid, pre_valid, label='valid (auc: {:.3f})'.format(auc_v_pr))



    plt.xlabel('recall')

    plt.ylabel('precision')

    plt.ylim(ylim)

    plt.legend()
def bokeh_precision_recall_curves(rt=None, pt=None, rv=None, pv=None):



    a1 = 1.0

    a2 = 0.5

    a3 = 0.1



    title = 'precision-recall curve - {} points'.format( len(pt) if pt is not None else len(rv) )



    f = plotting.figure(title=title, 

                        x_axis_label='threshold', 

                        y_axis_label='scores')



    if rv is not None and pv is not None:

        aucv = metrics.auc(rv, pv)

        f.line(rv, pv, legend='valid (auc: {:.3f})'.format(aucv), line_alpha=a1)



    if rt is not None and pt is not None:

        auct = metrics.auc(rt, pt)

        f.line(rt, pt, legend='train (auc: {:.3f})'.format(auct), line_alpha=a2)



    f.legend.location = 'bottom_center'



    return f
def bokeh_thr_precision_recall_scores(thrs, 

                                      pt=None, rt=None, 

                                      pv=None, rv=None,

                                      pc=None, rc=None):



    a1 = 1.0

    a2 = 0.5

    a3 = 0.1



    f = plotting.figure(title='precision/recall scores', 

                        x_axis_label='threshold ({} points)'.format(len(thrs)), 

                        y_axis_label='scores')



    if pv is not None:

        f.line(thrs, pv, legend='precision valid', line_color='red', line_alpha=a1)

    if pc is not None:

        f.line(thrs, pc, legend='precision cross', line_color='red', line_alpha=a2)

    if pt is not None:

        f.line(thrs, pt, legend='precision train', line_color='red', line_alpha=a3)

        

    if rv is not None:

        f.line(thrs, rv, legend='recall valid', line_color='green', line_alpha=a1)

    if rc is not None:

        f.line(thrs, rc, legend='recall valid', line_color='green', line_alpha=a2)

    if rt is not None:

        f.line(thrs, rt, legend='recall train', line_color='green', line_alpha=a3)



    f.legend.location = 'bottom_center'



    return f
def bokeh_thr_gmean_scores(thrs, gmt=None, gmv=None):



    a1 = 1.0

    a2 = 0.5

    a3 = 0.1



    f = plotting.figure(title='geometric mean scores', 

                        x_axis_label='threshold ({} points)'.format(len(thrs)), 

                        y_axis_label='gmean score')



    if gmv is not None:

        f.line(thrs, gmv, legend='gmean valid', line_alpha=a1)



    if gmt is not None:

        f.line(thrs, gmt, legend='gmean train', line_alpha=a2)



    f.legend.location = 'bottom_center'

    

    return f
def thr_cross_val_score(wrapper, X_train, y_train, thr, metric, kf):

    '''

    Cross Validation with specified threshold

    The wrapper should be created outside the function.

    '''



    def proba_scorer(y_true, y_proba):

        pos_proba = y_proba[:, 1].copy()

        y_pred = parse_proba(pos_proba, thr)

        score = metric(y_true, y_pred)

        return score



    scorer = metrics.make_scorer(proba_scorer)



    scores = cross_val_score(estimator=wrapper,

                             X=X_train,

                             y=y_train,

                             scoring=scorer,

                             cv=kf,)

    

    return scores.min(), scores.max(), scores.mean(), scores.std()
def resampler_cross_val_score(model, params, X, y, metric, resampler=None, kf=5, thr=0.5):

    '''

    Cross Validation with specified threshold and optional resampling techinque.

    '''



    def proba_scorer(y_true, y_proba):

        pos_proba = y_proba[:, 1].copy()

        y_pred = parse_proba(pos_proba, thr)

        score = metric(y_true, y_pred)

        return score



    scorer = metrics.make_scorer(proba_scorer)



    class Wrapper(model):

        

        def fit(self, X, y=None):



            if resampler is None:

                X_res, y_res = X.copy(), y.copy()

            else:

                X_res, y_res = resampler.fit_resample(X, y)



            params = self.get_params()

            

            self._model = model(**params).fit(X_res, y_res)



            return self



        def predict(self, X):

            return self._model.predict_proba(X)



    wrapper = Wrapper()



    if 'random_state' in wrapper.get_params():

        wrapper.set_params(random_state=1, **params)

    else:

        wrapper.set_params(**params)



    scores = cross_val_score(estimator=wrapper,

                             X=X, y=y,

                             scoring=scorer,

                             cv=kf,)

    

    return scores.min(), scores.max(), scores.mean(), scores.std()
def evaluate_params_grid(clf, params_dict, 

                         X_train, y_train, 

                         X_valid, y_valid, 

                         nrows=1, ncols=3, 

                         wid=4.5, hei=4.0,

                         roc_curves=False,

                         thr_num=100,

                         cross_val=False,

                         resampler=None,

                         hide_train=False,

                         bokeh_plots=False,

                         print_results=False,

                         return_results=False):



    '''

    Computes and outputs all metrics and statistics.



    Args:

        clf: Scikit-Learn's classifier class

        params_dict: dictionary of parameters names and values lists

        X_train: training data

        y_train: training target

        X_valid: validation data

        y_valid: validation target

        nrows: number of rows for matplotlib's subplot

        ncols: number of columns for matplotlib's subplot

        wid: width of matplotlib's plot

        hei: height of matplotlib's plot

        roc_curves: if output the roc auc curves

        thr_num: number of threshold points to generate on computations

        cross_val: if to compute cross validation scores

        resampler: resampling entity

        hide_train: if to hide training curves on the plots

        bokeh_plots: if to use bokeh instead of matplotlib

        print_results: if to print the dataframe of the results

        return_results: if to return the dataframe of the results

    '''



    params_grid = model_selection.ParameterGrid(params_dict)



    params_list = list(params_grid)



    X_copy = X_train.copy()

    y_copy = y_train.copy()



    for params in params_list:

        model = clf()



        if 'random_state' in model.get_params():

            model.set_params(random_state=1, **params)

        else:

            model.set_params(**params)



        if resampler is not None:

            X_train, y_train = resampler.fit_resample(X_copy, y_copy)



        t = time.time()

        model.fit(X_train, y_train)

        fit_time = time.time() - t





        t = time.time()



        positive_proba_t = model.predict_proba(X_train)[:, 1]

        positive_proba_v = model.predict_proba(X_valid)[:, 1]





        pre_t, rec_t, thr_t_pr = metrics.precision_recall_curve(y_train, positive_proba_t)

        pre_v, rec_v, thr_v_pr = metrics.precision_recall_curve(y_valid, positive_proba_v)



        if roc_curves:

            fpr_t, tpr_t, thr_t_roc = metrics.roc_curve(y_train, positive_proba_t)

            fpr_v, tpr_v, thr_v_roc = metrics.roc_curve(y_valid, positive_proba_v)





        thrs_tuning = np.linspace(start=0, stop=1, num=thr_num, endpoint=False)



        precisions_t = None if hide_train else [compute_precision(y_train, positive_proba_t, thr) for thr in thrs_tuning]

        precisions_v = [compute_precision(y_valid, positive_proba_v, thr) for thr in thrs_tuning]



        recalls_t = None if hide_train else [compute_recall(y_train, positive_proba_t, thr) for thr in thrs_tuning]

        recalls_v = [compute_recall(y_valid, positive_proba_v, thr) for thr in thrs_tuning]



        gmeans_t = [compute_gmean(y_train, positive_proba_t, thr) for thr in thrs_tuning]

        gmeans_v = [compute_gmean(y_valid, positive_proba_v, thr) for thr in thrs_tuning]



        calc_time = time.time() - t





        if cross_val:



            X_train, y_train = X_copy.copy(), y_copy.copy()



            t = time.time()



            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)



            cvp = [resampler_cross_val_score(model=clf, params=params,

                                             X=X_train, 

                                             y=y_train,

                                             metric=metrics.precision_score, 

                                             resampler=resampler, 

                                             kf=kf,

                                             thr=thr)[2] for thr in thrs_tuning]



            cvr = [resampler_cross_val_score(model=clf, params=params,

                                             X=X_train, 

                                             y=y_train,

                                             metric=metrics.recall_score   , 

                                             resampler=resampler, 

                                             kf=kf,

                                             thr=thr)[2] for thr in thrs_tuning]



            cv_time = time.time() - t



        else:

            cvp = None

            cvr = None



        if bokeh_plots:



            prc = bokeh_precision_recall_curves(rt=rec_t, pt=pre_t, rv=rec_v, pv=pre_v)

            prs = bokeh_thr_precision_recall_scores(thrs=thrs_tuning, 

                                                    pt=precisions_t, rt=recalls_t,

                                                    pv=precisions_v, rv=recalls_v,)

            gms = bokeh_thr_gmean_scores(thrs=thrs_tuning, gmv=gmeans_v, gmt=gmeans_t)



            plots = [prc, prs, gms]



            suptitle  = '<h2>{}</h2>'.format(model.__class__.__name__)

            suptitle += '<p>{:.1f}s + {:.1f}s'.format(fit_time, calc_time)

            suptitle += ' + {:.0f}m{:.0f}s</p>'.format(cv_time // 60, cv_time % 60) if cross_val else '</p>'

            suptitle += '<p>{}</p>'.format(str(params)) if bool(params) else str()



            title = Div(text=suptitle, 

                        style={'width': '100%',

                            'text-align': 'center'})



            grid_items = [[None, title, None], plots]



            wid_scale = 60

            hei_scale = 60

            grid = layouts.gridplot(grid_items, 

                                    plot_width=int(wid*wid_scale), 

                                    plot_height=int(hei*hei_scale), 

                                    merge_tools=True)

            plotting.show(grid)



        else:

        

            figsize = (wid*ncols, hei*nrows)

            suptitle  = model.__class__.__name__

            suptitle += ' ({:.1f}s + {:.1f}s'.format(fit_time, calc_time)

            suptitle += ' - {:.0f}m{:.0f}s)'.format(cv_time // 60, cv_time % 60) if cross_val else ')'

            suptitle += ' - {}'.format(str(params)) if bool(params) else str()



            plt.figure(figsize=figsize)

            plt.suptitle(suptitle, fontsize=14, y=1.1)

            index = 0



            index = index + 1

            plt.subplot(nrows, ncols, index)

            plot_precision_recall_curves(rec_t, pre_t, rec_v, pre_v)



            index = index + 1

            plt.subplot(nrows, ncols, index)

            plot_thr_precision_recall_scores(thrs_tuning, 

                                             precisions_t, recalls_t, 

                                             precisions_v, recalls_v,

                                             cvp, cvr)



            index = index + 1

            plt.subplot(nrows, ncols, index)

            plot_thr_gmean_scores(thrs_tuning, gmeans_t, gmeans_v)



            if roc_curves:

                index = index + 1

                plt.subplot(nrows, ncols, index)

                plot_roc_auc_curves(fpr_t, tpr_t, fpr_v, tpr_v)



            plt.tight_layout()

            plt.show()

            # print()



        results_dict = {'threshold': thrs_tuning,

                        'precision_cross': cvp,

                        'recall_cross': cvr,

                        'precision_valid': precisions_v,

                        'recall_valid': recalls_v,

                        'gmean_valid': gmeans_v,

                        'precision_train': precisions_t,

                        'recall_train': recalls_t,

                        'gmean_train': gmeans_t,

                        }



        results_df = pd.DataFrame(results_dict).dropna(axis=1)

            

        if print_results:

            display(results_df)

        

        if return_results:

            return results_df
csv = '/kaggle/input/creditcardfraud/creditcard.csv'



data = pd.read_csv(csv)

target = data.pop('Class')



display(data.head())



cps = {}

cps['data_original'] = data.copy()

cps['target_original'] = target.copy()
data = cps['data_original'].copy()

target = cps['target_original'].copy()



X_train, X_valid, y_train, y_valid = train_test_split(data,

                                                      target,

                                                      test_size=0.20,

                                                      random_state=1,

                                                      stratify=target)



cps['X_train'] = X_train.copy()

cps['y_train'] = y_train.copy()



cps['X_valid'] = X_valid.copy()

cps['y_valid'] = y_valid.copy()



fraud_genuine_counter([y_train, y_valid], ['Train', 'Valid'])
lr = LogisticRegression(random_state=1, max_iter=500)

lr.fit(X_train, y_train)



y_pred_train = lr.predict(X_train)

y_pred_valid = lr.predict(X_valid)



prec_train = precision_score(y_train, y_pred_train)

acc_train = accuracy_score(y_train, y_pred_train)

logloss_train = metrics.log_loss(y_train, y_pred_train)



prec_valid = precision_score(y_valid, y_pred_valid)

acc_valid = accuracy_score(y_valid, y_pred_valid)

logloss_valid = metrics.log_loss(y_valid, y_pred_valid)



pd.DataFrame({'Train': [prec_train, acc_train, logloss_train],

              'Valid': [prec_valid, acc_valid, logloss_valid]},

             index=['Precision', 'Accuracy', 'Log Loss'])
lr = LogisticRegression(random_state=1, max_iter=500)



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)



precisions = cross_val_score(estimator=lr,

                             X=X_train,

                             y=y_train,

                             scoring='precision',

                             cv=kf,

                             n_jobs=-1)



print('Min : {:.6f}'.format(precisions.min()))

print('Max : {:.6f}'.format(precisions.max()))

print('Mean: {:.6f}'.format(precisions.mean()))

print('Std : {:.6f}'.format(precisions.std()))
lr = LogisticRegression(random_state=1, max_iter=500)

lr.fit(X_train, y_train)



y_proba_train = lr.predict_proba(X_train)[:, 1]

y_proba_valid = lr.predict_proba(X_valid)[:, 1]



fpr_train, tpr_train, roc_thr_train = metrics.roc_curve(y_train, y_proba_train)

fpr_valid, tpr_valid, roc_thr_valid = metrics.roc_curve(y_valid, y_proba_valid)



pre_train, rec_train, pr_thr_train = metrics.precision_recall_curve(y_train, y_proba_train)

pre_valid, rec_valid, pr_thr_valid = metrics.precision_recall_curve(y_valid, y_proba_valid)
nrows = 2

ncols = 2

wid = 6

hei = 4

index = 0

figsize = (wid*ncols, hei*nrows)

plt.figure(figsize=figsize)



index = index + 1

plt.subplot(nrows, ncols, index)

plt.plot(roc_thr_train, tpr_train, label='tpr train')

plt.plot(roc_thr_train, fpr_train, label='fpr train')

plt.title('train tpr/fpr')

plt.xlabel('thr')

plt.ylabel('rate')

plt.legend()



index = index + 1

plt.subplot(nrows, ncols, index)

plt.plot(roc_thr_valid, tpr_valid, label='tpr valid')

plt.plot(roc_thr_valid, fpr_valid, label='fpr valid')

plt.title('valid tpr/fpr')

plt.xlabel('thr')

plt.ylabel('rate')

plt.legend()



index = index + 1

plt.subplot(nrows, ncols, index)

plot_roc_auc_curves(fpr_train, tpr_train, fpr_valid, tpr_valid)



index = index + 1

plt.subplot(nrows, ncols, index)

plot_precision_recall_curves(rec_train, pre_train, rec_valid, pre_valid)



# WARNING causes an error

# plt.subplot(nrows, ncols, 5)

# plt.plot(pr_thr_train, pre_train, label='precision train')

# plt.plot(pr_thr_train, rec_train, label='recall train')

# plt.title('train rates')

# plt.xlabel('thr')

# plt.ylabel('rate')

# plt.legend()



# WARNING causes an error

# plt.subplot(nrows, ncols, 6)

# plt.plot(pr_thr_valid, pre_valid, label='precision valid')

# plt.plot(pr_thr_valid, rec_valid, label='recall valid')

# plt.title('valid rates')

# plt.xlabel('thr')

# plt.ylabel('rate')

# plt.legend()



plt.tight_layout()

plt.show()
'''

Mapping the output of a confusion matrix



[[tn, fp],

 [fn, tp]]



rows: expected values (truths)

cols: predicted values (preds)

'''



truth = [1, 0, 0, 0, 0, 0, 1, 1, 1, 1]

pred  = [1, 0, 0, 1, 1, 1, 0, 0, 0, 0]



tp, tn, fp, fn = 1, 2, 3, 4



cm = metrics.confusion_matrix(truth, pred)

print(cm)



def parse_confusion_matrix(cm):

    tn, fp, fn, tp = cm.ravel()

    return tp, tn, fp, fn



(tp, tn, fp, fn) == parse_confusion_matrix(cm)
lr = LogisticRegression(random_state=1, max_iter=500)

lr.fit(X_train, y_train)



pos_proba_train = lr.predict_proba(X_train)[:, 0]

pos_proba_valid = lr.predict_proba(X_valid)[:, 0]



preds_train = parse_proba(pos_proba_train)

preds_valid = parse_proba(pos_proba_valid)



cm_train = metrics.confusion_matrix(y_train, preds_train)

cm_valid = metrics.confusion_matrix(y_valid, preds_valid)
lr = LogisticRegression(random_state=1, max_iter=500)

lr.fit(X_train, y_train)



y_proba_train = lr.predict_proba(X_train)[:, 1]

y_proba_valid = lr.predict_proba(X_valid)[:, 1]



thrs_metrics = np.linspace(start=0, stop=1, num=110, endpoint=False)



precisions_train = [compute_precision(y_train, y_proba_train, thr) for thr in thrs_metrics]

precisions_valid = [compute_precision(y_valid, y_proba_valid, thr) for thr in thrs_metrics]



recalls_train = [compute_recall(y_train, y_proba_train, thr) for thr in thrs_metrics]

recalls_valid = [compute_recall(y_valid, y_proba_valid, thr) for thr in thrs_metrics]



roc_aucs_train = [compute_roc_auc(y_train, y_proba_train, thr) for thr in thrs_metrics]

roc_aucs_valid = [compute_roc_auc(y_valid, y_proba_valid, thr) for thr in thrs_metrics]



gmeans_train = [compute_gmean(y_train, y_proba_train, thr) for thr in thrs_metrics]

gmeans_valid = [compute_gmean(y_valid, y_proba_valid, thr) for thr in thrs_metrics]
nrows = 3

ncols = 3

wid = 4.5

hei = 4

index = 0

figsize = (wid*ncols, hei*nrows)

plt.figure(figsize=figsize)



index = index + 1

plt.subplot(nrows, ncols, index)

plot_thr_precision_scores(thrs_metrics, precisions_train, precisions_valid)



index = index + 1

plt.subplot(nrows, ncols, index)

plot_thr_recall_scores(thrs_metrics, recalls_train, recalls_valid)



index = index + 1

plt.subplot(nrows, ncols, index)

plot_thr_precision_recall_scores(thrs_metrics, precisions_train, recalls_train, precisions_valid, recalls_valid)



index = index + 1

plt.subplot(nrows, ncols, index)

plot_thr_roc_auc_scores(thrs_metrics, roc_aucs_train, roc_aucs_valid)



index = index + 1

plt.subplot(nrows, ncols, index)

plot_thr_gmean_scores(thrs_metrics, gmeans_train, gmeans_valid)



index = index + 1

plt.subplot(nrows, ncols, index)

plot_thr_precision_recall_scores(thrs_metrics, pv=precisions_valid, rv=recalls_valid)

plot_thr_gmean_scores(thrs_metrics, gmv=gmeans_valid, labels=True)

plt.title('precision/recall/gmean')

plt.ylabel('scores')



plt.tight_layout()

plt.show()
class WrapperLogisticRegression(LogisticRegression):

    '''

    Wrapper used to return probabilities instead of classes in

    the cross val score evaluation below

    '''

    def predict(self, X):

        return LogisticRegression.predict_proba(self, X)
%%time



lr = WrapperLogisticRegression(random_state=1, max_iter=500)



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)



thrs_cv = np.linspace(start=0, stop=1, num=15, endpoint=False)



precis_scores = []

recall_scores = []



for thr in thrs_cv:

    p_scores = thr_cross_val_score(lr, X_train, y_train, thr, metrics.precision_score, kf)

    precis_scores.append(p_scores)

    r_scores = thr_cross_val_score(lr, X_train, y_train, thr, metrics.recall_score, kf)

    recall_scores.append(r_scores)



precis_scores = np.array(precis_scores)

recall_scores = np.array(recall_scores)
nrows = 1

ncols = 2

index = 0

wid = 5.5

hei = 4

figsize = (wid*ncols, hei*nrows)

plt.figure(figsize=figsize)



index = index + 1

plt.subplot(nrows, ncols, index)

plt.title('precision statistics')

plt.plot(thrs_cv, precis_scores[:, 0], label='min')

plt.plot(thrs_cv, precis_scores[:, 1], label='max')

plt.plot(thrs_cv, precis_scores[:, 2], label='mean')

plt.plot(thrs_cv, precis_scores[:, 2] + precis_scores[:, 3], label='mean + std')

plt.plot(thrs_cv, precis_scores[:, 2] - precis_scores[:, 3], label='mean - std')

plt.xlabel('thr')

plt.ylabel('score')

plt.legend()



index = index + 1

plt.subplot(nrows, ncols, index)

plt.title('recall statistics')

plt.plot(thrs_cv, recall_scores[:, 0], label='min')

plt.plot(thrs_cv, recall_scores[:, 1], label='max')

plt.plot(thrs_cv, recall_scores[:, 2], label='mean')

plt.plot(thrs_cv, recall_scores[:, 2] + recall_scores[:, 3], label='mean + std')

plt.plot(thrs_cv, recall_scores[:, 2] - recall_scores[:, 3], label='mean - std')

plt.xlabel('thr')

plt.ylabel('score')

plt.legend()



plt.tight_layout()

plt.show()





plt.figure(figsize=figsize)

plt.title('precision/recall statistics')

plt.plot(thrs_cv, precis_scores[:, 0], label='precision min', color='red', alpha=0.3)

plt.plot(thrs_cv, precis_scores[:, 1], label='precision max', color='red', alpha=0.5)

plt.plot(thrs_cv, precis_scores[:, 2], label='precision mean', color='red')

plt.plot(thrs_cv, recall_scores[:, 0], label='recall min', color='green', alpha=0.3)

plt.plot(thrs_cv, recall_scores[:, 1], label='recall max', color='green', alpha=0.5)

plt.plot(thrs_cv, recall_scores[:, 2], label='recall mean', color='green')

plt.xlabel('thr')

plt.ylabel('score')

plt.legend()



plt.tight_layout()

plt.show()





plt.figure(figsize=figsize)

plt.title('precision/recall std')

plt.plot(thrs_cv, precis_scores[:, 3], label='precision')

plt.plot(thrs_cv, recall_scores[:, 3], label='recall')

plt.xlabel('thr')

plt.ylabel('std')

plt.legend()



plt.tight_layout()

plt.show()
resampler_cross_val_score(model=RandomForestClassifier, params={}, 

                          X=X_train, y=y_train, 

                          metric=metrics.recall_score, 

                          resampler=None, 

                          kf=5, thr=0.5)
resampler_cross_val_score(model=RandomForestClassifier, params={}, 

                          X=X_train, y=y_train, 

                          metric=metrics.recall_score, 

                          resampler=RandomUnderSampler(), 

                          kf=5, thr=0.5)
resampler_cross_val_score(model=RandomForestClassifier, params={}, 

                          X=X_train, y=y_train, 

                          metric=metrics.recall_score, 

                          resampler=SMOTE(), 

                          kf=5, thr=0.5)
models = [

    # KNeighborsClassifier(),

    LogisticRegression(random_state=1),

    DecisionTreeClassifier(random_state=1, min_samples_split=100, max_depth=None),

    # DecisionTreeClassifier(random_state=1, min_samples_split=100, max_depth=10),

    RandomForestClassifier(random_state=1),

    # RandomForestClassifier(random_state=1, n_estimators=500),

    # RandomForestClassifier(random_state=1, min_samples_split=100, max_depth=None),

    RandomForestClassifier(random_state=1, min_samples_split=100, max_depth=10),

    # GradientBoostingClassifier(),

]



nrows = 1

ncols = 3

nfigs = len(models)

wid = 4.8

hei = 4.0



for model in models:

    t = time.time()

    model.fit(X_train, y_train)

    fit_time = time.time() - t





    positive_proba_t = model.predict_proba(X_train)[:, 1]

    positive_proba_v = model.predict_proba(X_valid)[:, 1]





    fpr_t, tpr_t, thr_t_roc = metrics.roc_curve(y_train, positive_proba_t)

    fpr_v, tpr_v, thr_v_roc = metrics.roc_curve(y_valid, positive_proba_v)



    pre_t, rec_t, thr_t_pr = metrics.precision_recall_curve(y_train, positive_proba_t)

    pre_v, rec_v, thr_v_pr = metrics.precision_recall_curve(y_valid, positive_proba_v)

    



    thrs_models = np.linspace(start=0, stop=1, num=90, endpoint=False)



    precisions_t = [compute_precision(y_train, positive_proba_t, thr) for thr in thrs_models]

    precisions_v = [compute_precision(y_valid, positive_proba_v, thr) for thr in thrs_models]



    recalls_t = [compute_recall(y_train, positive_proba_t, thr) for thr in thrs_models]

    recalls_v = [compute_recall(y_valid, positive_proba_v, thr) for thr in thrs_models]



    gmeans_t = [compute_gmean(y_train, positive_proba_t, thr) for thr in thrs_models]

    gmeans_v = [compute_gmean(y_valid, positive_proba_v, thr) for thr in thrs_models]





    figsize = (wid*ncols, hei*nrows)

    suptitle = model.__class__.__name__ + ' ({:.1f}s)'.format(fit_time)



    plt.figure(figsize=figsize)

    plt.suptitle(suptitle, fontsize=14, y=1.1)

    index = 0



    index = index + 1

    plt.subplot(nrows, ncols, index)

    plot_roc_auc_curves(fpr_t, tpr_t, fpr_v, tpr_v)



    index = index + 1

    plt.subplot(nrows, ncols, index)

    plot_precision_recall_curves(rec_t, pre_t, rec_v, pre_v)



    index = index + 1

    plt.subplot(nrows, ncols, index)

    plot_thr_precision_recall_scores(thrs_models, pv=precisions_v, rv=recalls_v)

    plot_thr_gmean_scores(thrs_models, gmv=gmeans_v, labels=True)



    plt.tight_layout()

    plt.show()
param_dict = {'n_estimators': [25, 50, 100]}



evaluate_params_grid(RandomForestClassifier, param_dict, X_train, y_train, X_valid, y_valid)
param_dict = {'min_samples_split': [50, 100, 250, 500]}



evaluate_params_grid(RandomForestClassifier, param_dict, X_train, y_train, X_valid, y_valid)
param_dict = {'min_samples_leaf': [50, 100, 250]}



evaluate_params_grid(RandomForestClassifier, param_dict, X_train, y_train, X_valid, y_valid)
param_dict = {'max_depth': [1, 5, None]}



evaluate_params_grid(RandomForestClassifier, param_dict, X_train, y_train, X_valid, y_valid)
X_train = cps['X_train'].copy()

y_train = cps['y_train'].copy()



X_under, y_under = RandomUnderSampler(random_state=1).fit_resample(X_train, y_train)



X_under = pd.DataFrame(X_under, columns=X_train.columns)

y_under = pd.DataFrame(y_under, columns=['Class'])



cps['X_under'] = X_under.copy()

cps['y_under'] = y_under.copy()
X_train = cps['X_train'].copy()

y_train = cps['y_train'].copy()



X_over, y_over = SMOTE(random_state=1).fit_resample(X_train, y_train)



X_over = pd.DataFrame(X_over, columns=X_train.columns)

y_over = pd.DataFrame(y_over, columns=['Class'])



cps['X_over'] = X_over.copy()

cps['y_over'] = y_over.copy()
display( fraud_genuine_counter([y_train, y_under['Class'], y_over['Class']], 

                               ['Train', 'Under', 'Over']) )
def apply_scaler(X_fit, X_trf, cols, scaler):



    df_fit = X_fit.copy().filter(items=items)

    df_trf = X_trf.copy().filter(items=items)



    items_scaled = pd.DataFrame(scaler.fit(df_fit).transform(df_trf), 

                                columns=items,

                                index=df_trf.index)



    X_scaled = items_scaled.join(X_trf.drop(columns=items))



    return X_scaled
X_train = cps['X_train'].copy()

X_valid = cps['X_valid'].copy()



scaler = preprocessing.StandardScaler()

items = ['Time', 'Amount']



X_train_scaled_std = apply_scaler(X_train, X_train, items, scaler)

X_valid_scaled_std = apply_scaler(X_train, X_valid, items, scaler)



cps['X_train_scaled_std'] = X_train_scaled_std.copy()

cps['X_valid_scaled_std'] = X_valid_scaled_std.copy()
X_train = cps['X_train'].copy()

X_valid = cps['X_valid'].copy()



scaler = preprocessing.RobustScaler()

items = ['Time', 'Amount']



X_train_scaled_rob = apply_scaler(X_train, X_train, items, scaler)

X_valid_scaled_rob = apply_scaler(X_train, X_valid, items, scaler)



cps['X_train_scaled_rob'] = X_train_scaled_rob.copy()

cps['X_valid_scaled_rob'] = X_valid_scaled_rob.copy()
# test cases generation



params_dict = {

    'sampling': ['Original', 'Undersampled', 'SMOTE'],

    # 'scaling': ['none', 'std', 'rob'],

    'selection': ['All', 'K Best', 'PCA', 'Correlation'],

    #  'model': ['RandomForest', 'LogisticRegression'],

}



params_list = list(model_selection.ParameterGrid(params_dict))



print('{} cases'.format(len(params_list)))



for params in params_list:

    s = 'Data: {:15} > Features: {:15}'

    s = s.format(params['sampling'], params['selection'])

    print(s)
# TEST 1 - comparison of RF and LR with undersampled data



evaluate_params_grid(RandomForestClassifier, {}, X_under, y_under, X_valid, y_valid)



evaluate_params_grid(LogisticRegression, {}, X_under, y_under, X_valid, y_valid)



# we can not trust one only metric to evalute a model performance

# in this case the gmean and recall are so good but precision is terrible
# TEST 2 - comparison of not scaled and scaled data with RF



evaluate_params_grid(RandomForestClassifier, {}, X_train, y_train, X_valid, y_valid)



evaluate_params_grid(RandomForestClassifier, {}, X_train_scaled_std, y_train, X_valid_scaled_std, y_valid)



evaluate_params_grid(RandomForestClassifier, {}, X_train_scaled_rob, y_train, X_valid_scaled_rob, y_valid)



# scaling the data does not provide better results
# TEST 3 - comparison of not scaled and scaled data with LR



evaluate_params_grid(LogisticRegression, {}, X_train, y_train, X_valid, y_valid)



evaluate_params_grid(LogisticRegression, {}, X_train_scaled_std, y_train, X_valid_scaled_std, y_valid)



# Logistic Regression is a poor model and can be definitively discarded
# TEST 4 - feature selection with std scaled data



for k in [5, 15, 25]:

    selector = SelectKBest(k=k)

    evaluate_params_grid(RandomForestClassifier, {},

                         selector.fit(X_train_scaled_std, y_train).transform(X_train_scaled_std), y_train,

                         selector.fit(X_valid_scaled_std, y_valid).transform(X_valid_scaled_std), y_valid,

                         thr_num=60)



# the less features selected the best and this is a weird result
# TEST 5 - feature selection with undersampled data



for k in [5, 15, 25]:

    selector = SelectKBest(k=k)

    evaluate_params_grid(RandomForestClassifier, {},

                         selector.fit(X_under, y_under).transform(X_under), y_under,

                         selector.fit(X_valid, y_valid).transform(X_valid), y_valid,

                         thr_num=100)



# again the precision is terrible with undersampled data
# TEST 6 - comparison of PCA features performance between original and undersampled data



pattern = 'V'



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(regex=pattern), y_train,

                     X_valid.filter(regex=pattern), y_valid,

                     thr_num=50)



evaluate_params_grid(RandomForestClassifier, {},

                     X_under.filter(regex=pattern), y_under,

                     X_valid.filter(regex=pattern), y_valid,

                     thr_num=50)



# undersampling continues terrible
# TEST 7 - comparison of ALL vs PCA features



# ERASE because redundant with Test 8



# evaluate_params_grid(RandomForestClassifier, {},

#                      X_train, y_train,

#                      X_valid, y_valid,

#                      hei=4.0, thr_num=70)



# evaluate_params_grid(RandomForestClassifier, {},

#                      X_train.filter(regex='V'), y_train,

#                      X_valid.filter(regex='V'), y_valid,

#                      hei=4.0, thr_num=70)



# no significant impact but PCA seems a few better
# TEST 8 - comparison of feature seleciton with original data



features = ['V10', 'V12', 'V14', 'V17']



ylim_prs = [0.85, 0.95]

thr_points = 70



evaluate_params_grid(RandomForestClassifier, {},

                     X_train, y_train,

                     X_valid, y_valid,

                     thr_num=thr_points)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(like='V'), y_train,

                     X_valid.filter(like='V'), y_valid,

                     thr_num=thr_points)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=features), y_train,

                     X_valid.filter(items=features), y_valid,

                     thr_num=thr_points)



# precision/recall sounds better because of equality in scores for a range in thr

# gmean is better 

# correlation features are better than all
# TEST 8 - comparison of feature selection with original data [REPEATED WITH BOKEH PLOT]



features = ['V10', 'V12', 'V14', 'V17']



ylim_prs = [0.85, 0.95]

thr_points = 70



evaluate_params_grid(RandomForestClassifier, {},

                     X_train, y_train,

                     X_valid, y_valid,

                     hei=5.0, thr_num=thr_points, bokeh_plots=True)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(like='V'), y_train,

                     X_valid.filter(like='V'), y_valid,

                     hei=5.0, thr_num=thr_points, bokeh_plots=True)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=features), y_train,

                     X_valid.filter(items=features), y_valid,

                     hei=5.0, thr_num=thr_points, bokeh_plots=True)



# precision/recall sounds better because of equality in scores for a range in thr

# gmean is better 

# correlation features are better than all
# TEST 9 - comparison of RF and LR with SMOTE



evaluate_params_grid(RandomForestClassifier, {}, 

                     X_over, y_over, 

                     X_valid, y_valid, 

                     thr_num=60)



evaluate_params_grid(LogisticRegression, {}, 

                     X_over, y_over, 

                     X_valid, y_valid, 

                     thr_num=60)
# TEST 10 - comparison of original and SMOTE data with RF



evaluate_params_grid(RandomForestClassifier, {}, 

                     X_train, y_train, 

                     X_valid, y_valid, 

                     thr_num=60)



evaluate_params_grid(RandomForestClassifier, {}, 

                     X_over, y_over, 

                     X_valid, y_valid, 

                     thr_num=60)
# TEST 10 - comparison of original and SMOTE data with RF [REPEATED WITH BOKEH PLOT]



evaluate_params_grid(RandomForestClassifier, {}, 

                     X_train, y_train, 

                     X_valid, y_valid, 

                     hei=6, thr_num=60, bokeh_plots=True)



evaluate_params_grid(RandomForestClassifier, {}, 

                     X_over, y_over, 

                     X_valid, y_valid, 

                     hei=6, thr_num=60, bokeh_plots=True)
# TEST 11 - Random Forest hyperparameters tuning with SMOTE data



evaluate_params_grid(RandomForestClassifier, {'n_estimators': [25, 50, 100]}, 

                     X_over, y_over, 

                     X_valid, y_valid, 

                     thr_num=60)



# almost the same results so model tuning may not improve results
# TEST 11 - Random Forest hyperparameters tuning with SMOTE data  [REPEATED WITH BOKEH PLOT]



evaluate_params_grid(RandomForestClassifier, {'n_estimators': [25, 50, 100]}, 

                     X_over, y_over, 

                     X_valid, y_valid, 

                     hei=6, thr_num=60, bokeh_plots=True)



# almost the same results so model tuning may not improve results
# TEST 12 - comparison of feature selection by correlation with original data



corr_original = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3', 'V7', 

                 'V11', 'V4', 'V18', 'V1', 'V9', 'V5', 'V2', 'V21']



thr_points = 70

ylim_prs = [0.85, 0.95]



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=corr_original[:5]), y_train,

                     X_valid.filter(items=corr_original[:5]), y_valid,

                     thr_num=thr_points, )# ylim=ylim_prs)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=corr_original[:10]), y_train,

                     X_valid.filter(items=corr_original[:10]), y_valid,

                     thr_num=thr_points, )# ylim=ylim_prs)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=corr_original[:15]), y_train,

                     X_valid.filter(items=corr_original[:15]), y_valid,

                     thr_num=thr_points, )# ylim=ylim_prs)



# precision and recall are better with more features
# TEST 13 - comparison of feature selection with undersampled data



features = ['V14', 'V4', 'V12', 'V11', 'V10', 'V16', 'V3', 

            'V9', 'V17', 'V2', 'V7', 'V18', 'V1', 'V6', 'V5']



thr_points = 70

ylim_prs = [0.8, 0.9]



evaluate_params_grid(RandomForestClassifier, {},

                     X_under.filter(items=features[:5]), y_under,

                     X_valid.filter(items=features[:5]), y_valid,

                     thr_num=thr_points, )



evaluate_params_grid(RandomForestClassifier, {},

                     X_under.filter(items=features[:10]), y_under,

                     X_valid.filter(items=features[:10]), y_valid,

                     thr_num=thr_points, )



evaluate_params_grid(RandomForestClassifier, {},

                     X_under.filter(items=features[:15]), y_under,

                     X_valid.filter(items=features[:15]), y_valid,

                     thr_num=thr_points, )



# FORGEEEEEEEEET undersampling
# TEST 14 - comparison of feature selection by correlation with SMOTE data



features = ['V14', 'V4', 'V11', 'V12', 'V10', 'V16', 'V3', 

            'V9', 'V17', 'V2', 'V7', 'V18', 'V6', 'V1', 'V5']



thr_points = 70

ylim_prs = [0.8, 0.9]



evaluate_params_grid(RandomForestClassifier, {},

                     X_over.filter(items=features[:5]), y_over,

                     X_valid.filter(items=features[:5]), y_valid,

                     thr_num=thr_points, )



evaluate_params_grid(RandomForestClassifier, {},

                     X_over.filter(items=features[:10]), y_over,

                     X_valid.filter(items=features[:10]), y_valid,

                     thr_num=thr_points, )



evaluate_params_grid(RandomForestClassifier, {},

                     X_over.filter(items=features[:15]), y_over,

                     X_valid.filter(items=features[:15]), y_valid,

                     thr_num=thr_points, )
features = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3', 'V7', 

            'V11', 'V4', 'V18', 'V1', 'V9', 'V5', 'V2', 'V21']



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=features[:5]), y_train,

                     X_valid.filter(items=features[:5]), y_valid,

                     cross_val=True, thr_num=5, )



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=features[:5]), y_train,

                     X_valid.filter(items=features[:5]), y_valid,

                     cross_val=True, thr_num=15, )



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=features[:5]), y_train,

                     X_valid.filter(items=features[:5]), y_valid,

                     cross_val=True, thr_num=50, )
# feature selection by correlation on original data (1/3)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=corr_original[:5]), y_train,

                     X_valid.filter(items=corr_original[:5]), y_valid,

                     hei=5.0, thr_num=15, cross_val=True, print_results=True)
# feature selection by correlation on original data (2/3)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=corr_original[:10]), y_train,

                     X_valid.filter(items=corr_original[:10]), y_valid,

                     hei=5.0, thr_num=15, cross_val=True, print_results=True)
# feature selection by correlation on original data (3/3)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=corr_original[:15]), y_train,

                     X_valid.filter(items=corr_original[:15]), y_valid,

                     hei=5.0, thr_num=15, cross_val=True, print_results=True)
# 'class_weight' parameter - test 1/3



evaluate_params_grid(RandomForestClassifier, 

                     {'class_weight': [None]},

                     X_train.filter(items=corr_original[:15]), y_train,

                     X_valid.filter(items=corr_original[:15]), y_valid,

                     hei=5.0, thr_num=15, cross_val=True, print_results=True)
# 'class_weight' parameter - test 2/3



evaluate_params_grid(RandomForestClassifier, 

                     {'class_weight': ['balanced']},

                     X_train.filter(items=corr_original[:15]), y_train,

                     X_valid.filter(items=corr_original[:15]), y_valid,

                     hei=5.0, thr_num=15, cross_val=True, print_results=True)
# 'class_weight' parameter - test 3/3



evaluate_params_grid(RandomForestClassifier, 

                     {'class_weight': ['balanced_subsample']},

                     X_train.filter(items=corr_original[:15]), y_train,

                     X_valid.filter(items=corr_original[:15]), y_valid,

                     hei=5.0, thr_num=15, cross_val=True, print_results=True)
# 'class_weight' parameter - test 1/3 - repeated with SMOTE



features = correlation(X_over, y_over, 10, 0.9)



evaluate_params_grid(RandomForestClassifier, 

                     {'class_weight': [None]},

                     X_train.filter(items=features), y_train,

                     X_valid.filter(items=features), y_valid,

                     hei=5.0, thr_num=15, cross_val=True, print_results=True,

                     resampler=SMOTE(random_state=1), )
# TEST 2: 15 best features (Vitor)



features = correlation(X_over, y_over, 15, 0.9)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=features), y_train,

                     X_valid.filter(items=features), y_valid,

                     hei=5.0, thr_num=10, cross_val=True, print_results=True,

                     resampler=SMOTE(random_state=1), )
# TEST 3: 10 best features (Vitor)



features = correlation(X_over, y_over, 10, 0.9)



evaluate_params_grid(RandomForestClassifier, {},

                     X_train.filter(items=features), y_train,

                     X_valid.filter(items=features), y_valid,

                     hei=5.0, thr_num=10, cross_val=True, print_results=True,

                     resampler=SMOTE(random_state=1), )
# all features with SMOTE



evaluate_params_grid(RandomForestClassifier, {},

                     X_train, y_train,

                     X_valid, y_valid,

                     hei=5.0, thr_num=10, cross_val=True, print_results=True,

                     resampler=SMOTE(random_state=1), )
total_seconds = time.time() - running_time



total_minutes = total_seconds // 60

seconds = total_seconds % total_minutes



hours = total_minutes // 60

minutes = hours % total_minutes



print(total_seconds)

print('{}h{}m{}s'.format(hours, seconds, minutes))