import pandas

import numpy as np

import matplotlib.pyplot as plt

import json
hd_Signal=pandas.read_hdf('../input/Signal.h5', 'df')

hd_Background=pandas.read_hdf('../input/Background.h5', 'df')
hd_Signal.head()
variablelist = [

  'nJets_OR_T',

  'nJets_OR_T_MV2c10_70',

  'lep_flavour',

  'Mll01',

  'minDeltaR_LJ_0',

  'minDeltaR_LJ_1',

  'max_eta',

  'lep_Pt_1',

  'MET_RefFinal_et',

  'DRll01',

 ]
fig, ax = plt.subplots(3, 4, figsize=(25, 15))

nbins = 50



varcounter = -1

for i, axobjlist in enumerate(ax):

      for j, axobj in enumerate(axobjlist):

        varcounter+=1

        if varcounter < len(variablelist):

            var = variablelist[varcounter]

        

            p_Signal = pandas.DataFrame({var: hd_Signal[var]})

            p_Background = pandas.DataFrame({var: hd_Background[var]})

            #b.replace([np.inf, -np.inf], np.nan, inplace=True)

            #c.replace([np.inf, -np.inf], np.nan, inplace=True)

            #b = b.dropna()

            #c = c.dropna()            

            minval = np.amin(p_Signal[var])

            maxval = max([np.amax(p_Signal[var]), np.amax(p_Background[var])])*1.4

            binning = np.linspace(minval,maxval,nbins)



            axobj.hist(p_Signal[var],binning,histtype=u'step',label='Signal',density=1) # color='orange',

            axobj.hist(p_Background[var],binning,histtype=u'step', label='Background',density=1) #color='b',

            axobj.legend()

            axobj.set_yscale('log',nonposy='clip')

            axobj.set_title(variablelist[varcounter])



        else:

            axobj.axis('off')



plt.tight_layout()

plt.show()
Signal_vars=hd_Signal[variablelist]

Background_vars=hd_Background[variablelist]
X = np.concatenate((Signal_vars,Background_vars)) # training data                                                                                                                               

y = np.concatenate((np.ones(Signal_vars.shape[0]),np.zeros(Background_vars.shape[0]))) # class lables                                                                                           



from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
bdt_0 = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth=4, max_features='auto', min_samples_split=10, min_samples_leaf=10), n_estimators=100, learning_rate=0.5)
bdt_0.fit(X_train, y_train)
y_predicted_0 = bdt_0.predict(X_test)
print( classification_report(y_test, y_predicted_0, target_names=["signal", "background"]))

print( "Area under ROC curve: %.4f"%(roc_auc_score(y_test, bdt_0.decision_function(X_test))))
import xgboost

#bdt_xgb = xgboost.XGBClassifier(tree_method="hist", thread_count=-1)

bdt_xgb = xgboost.XGBClassifier(tree_method="hist", thread_count=-1,max_depth=3, 

                                learning_rate=0.1, n_estimators=1000, verbosity=1, 

                                objective='binary:logistic', booster='gbtree', n_jobs=1, gamma=0, min_child_weight=1)

#, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, missing=None, gpu_id=-1, **kwargs)



#agrees to the tth default:

# Method_Opt = "!H:!V:NTrees=1000:MinNodeSize=1.5%:BoostType=Grad:Shrinkage=0.10:

# UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=2";
bdt_xgb.fit(X_train, y_train)
y_predicted_xgb = bdt_xgb.predict(X_test)
print( classification_report(y_test, y_predicted_xgb, target_names=["signal", "background"]))

xgb_bdt_ROC=roc_auc_score(y_test, bdt_xgb.predict_proba(X_test)[:, 1])

print("XGBoost ROC AUC = {:.3f}".format( xgb_bdt_ROC))

print( "wrt  BDT: %.4f"%(xgb_bdt_ROC/roc_auc_score(y_test, bdt_0.decision_function(X_test))))
import time

def evaluate_models(models_dict):

  for model_name, model in models_dict.items():

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    print("{}; train time {:.3f} s; ROC AUC = {:.3f}".format(

          model_name,

          end - start,

          roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])))
mods = {"AdaBoost":bdt_0,

       "XGboost":bdt_xgb}

           
evaluate_models(mods)
fpr = dict()

tpr = dict()

roc_auc = dict()

i=0

for model_name, model in mods.items():

#for i in range(len(mods.keys())):

    print(i)

    #fpr[i], tpr[i], _ = roc_curve(y_test, pred_vec[i])

    fpr[i], tpr[i], _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    #roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    roc_auc[i] = auc(fpr[i], tpr[i])

    i+=1
plt.figure()

lw = 2

for i in range(len(mods)):

    plt.plot(fpr[i], tpr[i], 

             lw=lw, label='%s ROC (%0.3f)' % (list(mods.keys())[i], roc_auc[i])) #color='darkorange',



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
from xgboost import plot_tree

plot_tree(bdt_xgb,rankdir='LR')

#plot_tree(bst, num_trees=2),num_trees=1

fig = plt.gcf()

fig.set_size_inches(150, 100)
plot_range = (0,1)

colors=['orange','blue']

class_names=["Signal","Background"]

nbins=40

for i in range(2):

    plt.hist(bdt_xgb.predict_proba(X_test)[:, i],nbins, range=plot_range, label='Test %s' % class_names[i], color=colors[i],alpha=.5,density=True )

    plt.hist(bdt_xgb.predict_proba(X_train)[:, i],nbins, range=plot_range, label='Train %s' % class_names[i], color=colors[i], alpha=.5, histtype='step',density=True )

#x1, x2, y1, y2 = plt.axis()

#plt.axis((x1, x2, y1, y2 * 1.2))

plt.legend(loc='upper right')

plt.ylabel('Samples')

plt.xlabel('Score')

plt.title('Decision Scores')
bdt_xgb_ovf = xgboost.XGBClassifier(tree_method="hist", thread_count=-1,max_depth=20, learning_rate=0.1, n_estimators=100, verbosity=1, objective='binary:logistic', booster='gbtree')
bdt_xgb_ovf.fit(X_train, y_train)
y_predicted_xgb_ovf = bdt_xgb_ovf.predict(X_test)
xgb_bdt_ovf_ROC=roc_auc_score(y_test, bdt_xgb_ovf.predict_proba(X_test)[:, 1])

print("XGBoost ROC AUC = {:.3f}".format( xgb_bdt_ovf_ROC))
plot_range = (0,1)

colors=['orange','blue']

class_names=["Signal","Background"]

nbins=40

for i in range(2):

    plt.hist(bdt_xgb_ovf.predict_proba(X_test)[:, i],nbins, range=plot_range, label='Test %s' % class_names[i], color=colors[i],alpha=.5,density=True )

    plt.hist(bdt_xgb_ovf.predict_proba(X_train)[:, i],nbins, range=plot_range, label='Train %s' % class_names[i], color=colors[i], alpha=.5, histtype='step',density=True )

#x1, x2, y1, y2 = plt.axis()

#plt.axis((x1, x2, y1, y2 * 1.2))

plt.legend(loc='upper right')

plt.ylabel('Samples')

plt.xlabel('Score')

plt.title('Decision Scores')
from scipy import stats

from scipy.stats import ks_2samp



KS_stat = ks_2samp(bdt_xgb.predict_proba(X_test)[:, 1], bdt_xgb.predict_proba(X_train)[:, 1])



KS_stat_ovf = ks_2samp(bdt_xgb_ovf.predict_proba(X_test)[:, 1], bdt_xgb_ovf.predict_proba(X_train)[:, 1])



print("Kolmogorov-Smirnoff statistics for : \n  - shallow tree - ",KS_stat, "\n  - overfitting model",KS_stat_ovf)
def empirical_cdf(sample, plotting=True):

    N = len(sample)

    rng = max(sample) - min(sample)

    if plotting:

        xs = np.concatenate([np.array([min(sample)-rng/3]), np.sort(sample) , np.array([max(sample)+rng/3])])

        ys = np.append(np.arange(N+1)/N, 1)

    else:

        xs = np.sort(sample)

        ys = np.arange(1, N+1)/N

    return (xs, ys)
xs_test, ys_test = empirical_cdf(bdt_xgb.predict_proba(X_test)[:, 1])

xs_train, ys_train = empirical_cdf(bdt_xgb.predict_proba(X_train)[:, 1])
xs_test_ovf, ys_test_ovf = empirical_cdf(bdt_xgb_ovf.predict_proba(X_test)[:, 1])

xs_train_ovf, ys_train_ovf = empirical_cdf(bdt_xgb_ovf.predict_proba(X_train)[:, 1])
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(18, 6))



#fig.suptitle('Horizontally stacked subplots')

ax1.set_title('Normal model')

ax1.plot(xs_test, ys_test,label='Test Signal',linewidth=3,linestyle=':')

ax1.plot(xs_train, ys_train,label='Train Signal')

ax1.set_ylabel('c. d. f.')

ax1.set_xlabel('Score')

ax1.legend()

ax2.set_title('Overfitting model')

ax2.set_ylabel('c. d. f.')

ax2.plot(xs_test_ovf, ys_test_ovf,label='Test Signal')

ax2.plot(xs_train_ovf, ys_train_ovf,label='Train Signal')

ax2.set_xlabel('Score')

ax2.legend()

#plt.step(xs_test, ys_test)

##plt.step(xs_train, ys_train)

#plt.step(xs_test_ovf, ys_test_ovf)

#plt.step(xs_train_ovf, ys_train_ovf)
plot_step = 0.2

x_min, x_max = X[:, 4].min() , X[:, 4].max() 

y_min, y_max = X[:, 6].min(), X[:, 6].max()

xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step*2),

                     np.arange(y_min, y_max, plot_step*2))





# Plot the decision boundaries



plt.subplot(121)

plt.axis("tight")



for i  in range(2):

    idx = np.where(y == i)

    plt.scatter(X[idx, 4], X[idx, 6],

                #c=c, cmap=plt.cm.Paired,

                s=20, edgecolor='k',

                label="Class %s" % i)

plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.legend(loc='upper right')

plt.xlabel('x')

plt.ylabel('y')

plt.title('Decision Boundary')