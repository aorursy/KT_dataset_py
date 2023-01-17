# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import StratifiedKFold



import multiprocessing, time, itertools



from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score



from collections import namedtuple, Counter



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm



def get_model_ml_(params):



  if params.split()[0] == 'svc':

    if params.split()[1] == 'none':

      C_parameter=1.0

    else:

      C_parameter=float(params.split()[1])



    if params.split()[2] == 'none':

      gamma_parameter='scale'

    else:

      gamma_parameter=float(params.split()[2])

      

    clf = svm.SVC(C=C_parameter, gamma=gamma_parameter, kernel='rbf', probability=True)



  elif params.split()[0] == 'rfc':

    if len(params.split()) != 4:

      if params.split()[1].lower()  == 'none':

        clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=int(params.split()[2]))

      else:        

        clf = RandomForestClassifier(n_estimators=100, max_depth=int(params.split()[1]), min_samples_split=int(params.split()[2]))

    else: # aqui eh pra entrar o n_est

      try:

        max_depth = int(params.split()[1])

      except:

        max_depth = None

      try:

        min_samples_split = int(params.split()[2])

      except:

        print ('Erro '+params)

        sys.exit(0)

      try:

        n_estimators = int(params.split()[3])

      except:

        print ('Erro '+params)

        sys.exit(0)

      clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)



  elif params.split()[0] == 'logit':

    if params.split()[1].lower()  == 'none':

      clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', C=1.0)

    else:

      clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', C=float(params.split()[1]))



  elif params.split()[0] == 'ada':

      clf = AdaBoostClassifier(n_estimators=int(params.split()[1]), learning_rate=float(params.split()[2]))



  else:

    print ('Nao foi identificado o classificador. {}'.format(params))

    

  return clf



def cros_val(clf, X, Y, metrics=['accuracy', 'recall'], resampling='undersampling', cv=3, multiclass=False):



  n_classes = len(set(Y))



  # falta instanciar

  return_named_tuple = namedtuple('return_named_tuple', ('clf', 'smote', 'cv', 'accuracy', 'recall', 'auc', 'f1_score'))



  # laco dos folds

  cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(time.time()))



  scores_f1 = list()

  scores_precision = list()

  scores_recall = list()

  scores_auc = list()



  Y_pred_proba_geral = np.zeros(shape=Y.shape)

  Y_pred_geral = np.zeros(shape=Y.shape)



  for train, test in cv_folds.split(X, Y):



    # essa linha h soh pra setar caso o augmented seja None

    if resampling == None:

      X_train_aug, Y_train_aug = X[train], Y[train]



    # agora eh necessario checar o aumento de dados

    if resampling == 'smote':

      X_train_aug, Y_train_aug = SMOTE().fit_resample(X[train], Y[train])



    if resampling == 'undersampling':

      s = s + 'aug check: undersampling\n'

      rus = RandomUnderSampler()

      X_train_aug, Y_train_aug = rus.fit_resample(X[train], Y[train])



    if resampling == 'oversampling':

      s = s + 'aug check: oversampling\n'

      ros = RandomOverSampler()

      X_train_aug, Y_train_aug = ros.fit_resample(X[train], Y[train])

    

    # treino e predicoes

    clf.fit(X_train_aug, Y_train_aug)

    Y_pred = clf.predict(X[test])

    Y_true = Y[test]



    # guarda na matrizona geral

    if multiclass:

      roc_temporario_ = 0

      for i in range(1, n_classes+1):

        Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, i-1] # pegar o proba 1 aqui 

        roc_temporario_ = roc_temporario_ + roc_auc_score((Y_true==i).astype('int'), Y_pred_proba_geral[test])

      roc_temporario_ = roc_temporario_ / n_classes

      scores_auc.append(roc_temporario_)

    else:

      Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, 1] # pegar o proba 1 aqui 

      scores_auc.append(roc_auc_score(Y_true, Y_pred_proba_geral[test]))





    #Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, 1].copy() # pegar o proba 1 aqui 

    Y_pred_geral[test] = clf.predict(X[test]).copy()





    # guardando os scores

    #scores_f1.append(f1_score(Y_true, Y_pred, average=average))

    #scores_precision.append(precision_score(Y_true, Y_pred, average=average))

    #scores_recall.append(recall_score(Y_true, Y_pred, average=average))

    #scores_auc.append(roc_auc_score(Y_true, Y_pred_proba_geral[test]))



    # guardando as confmatrix de cada fold

    confm = confusion_matrix(Y_true, Y_pred) 

    #s = s + str(confm) + '\n'



  scores_f1 = np.array(scores_f1)

  scores_precision = np.array(scores_precision)

  scores_recall = np.array(scores_recall)

  scores_auc = np.array(scores_auc)



  # conf matrix

  #Y_pred = cross_val_predict(clf, X, Y, cv=cv)

  #Y_true = Y.copy()

  #confm = confusion_matrix(Y_true, Y_pred_geral)



  r = return_named_tuple (clf, resampling, cv, scores_precision, scores_recall, scores_auc, scores_f1)

  

  return r





### function to assist parallel nested cross-validation implementation

def f(params, X, Y, cv, resampling):



  print ('rodando params: {}'.format(params))



  clf = get_model_ml_(params)

  

  if len(set(Y)) > 2:

    multiclass = True

  elif len(set(Y)) == 2:

    multiclass = False

  else:

    print ('Erro! flag multiclass.')



  return_ = cros_val(clf, X, Y, metrics=['accuracy', 'recall'], resampling=resampling, cv=cv, multiclass=multiclass)



  return return_.auc.mean()



  print ('params {} finalizado.'.format(params))





def grid_search_nested_parallel(X, Y, cv=3, writefolder=None, n_jobs=30, resampling='undersample'):



  n_classes = len(set(Y))



  if len(set(Y)) > 2:

    multiclass = True

  elif len(set(Y)) == 2:

    multiclass = False

  else:

    print ('Erro! flag multiclass.')



  # laco dos folds

  cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(time.time()))



  Y_pred_proba_geral = np.zeros(shape=Y.shape)

  Y_pred_geral = np.zeros(shape=Y.shape)



  

  # SVC

  C_list = np.logspace(np.log10(0.1), np.log10(1000), num=3)

  C_list = [str(x) for x in C_list]

  gamma_list = np.logspace(np.log10(0.0001), np.log10(1), num=3)

  gamma_list = [str(x) for x in gamma_list]



  svc_kernel = 'rbf'

  svc_params_list = list(itertools.product(C_list, gamma_list))

  svc_params_list = ['svc '+' '.join(x) for x in svc_params_list]

  

  # RF

  max_depth_list = ['1', '2', 'None']

  min_samples_split_list = ['2', '4']



  rfc_params_list = list(itertools.product(max_depth_list, min_samples_split_list))

  rfc_params_list = ['rfc '+' '.join(x) for x in rfc_params_list]



  # Logit

  C_list = np.logspace(np.log10(0.0001), np.log10(10), num=10)

  C_list = [str(x) for x in C_list]

  logit_params_list = ['logit '+' '+x for x in C_list]



  params_list = svc_params_list + rfc_params_list + logit_params_list

   

  params_scores = np.zeros((len(params_list),))

  params_std_scores = np.zeros((len(params_list),))



  s = ''



  if (writefolder != None):

    plt.figure(figsize=(18,10))

    plt.ylabel('AUC score')

    plt.xlabel('Parameter set number')

    plt.title('')





  best_params_all = list()

  best_auc_scores_holdout = list()

  

  #plt.figure(figsize=(12,8))



  for i, (train, test) in enumerate(cv_folds.split(X, Y)):





    parameters_vector_total = [(x, X[train], Y[train], cv, resampling) for x in params_list]



    params_scores_partial = list()

    for parameters_vector in [parameters_vector_total[j:j+n_jobs] for j in range(0, len(parameters_vector_total), n_jobs)]:

      with multiprocessing.Pool(processes=n_jobs) as pool:

        params_scores_partial = params_scores_partial + pool.starmap(f, parameters_vector)



    params_scores = np.array(params_scores_partial)

  

    best_params = params_list[params_scores.argmax()]

    best_params_all.append(best_params)

    best_params_idx = params_scores.argmax()



    clf = get_model_ml_(best_params)

    

    ##

    if resampling == None or resampling == 'None':

      X_train_aug, Y_train_aug = X[train], Y[train]



    # agora eh necessario checar o aumento de dados

    if resampling == 'smote':

      X_train_aug, Y_train_aug = SMOTE().fit_resample(X[train], Y[train])



    if resampling == 'undersampling':

      rus = RandomUnderSampler()

      X_train_aug, Y_train_aug = rus.fit_resample(X[train], Y[train])



    if resampling == 'oversampling':

      ros = RandomOverSampler()

      X_train_aug, Y_train_aug = ros.fit_resample(X[train], Y[train])

    

    # treino e predicoes

    clf.fit(X_train_aug, Y_train_aug)

    Y_pred = clf.predict(X[test])

    #Y_true = Y[test]

    ##

    

    

    #clf.fit(X[train], Y[train])

    Y_true = Y[test]

    



    if writefolder:

      s = s + '####### FOLD {} of {} #####\n'.format(i+1, cv)

      for param, score, std in zip(params_list, params_scores, params_std_scores):

        s = s + 'param: {}, score: {:.3} ({:.4})\n'.format(param, score, std)

      s = s + '* Best params: {}, idx: {} - score: {:.3}\n'.format(best_params, best_params_idx, params_scores[best_params_idx])

      

      s = s + '*** Evaluation phase ***\n'

      

      if multiclass:

        roc_temporario_ = 0

        for j in range(1, n_classes+1):

          Y_pred_proba_geral[test] = clf.predict_proba(X[test])[:, j-1] # pegar o proba 1 aqui 

          roc_temporario_ = roc_temporario_ + roc_auc_score((Y_true==j).astype('int'), Y_pred_proba_geral[test])

        roc_temporario_ = roc_temporario_ / n_classes

        auc_ = roc_temporario_

      else:

        auc_ = roc_auc_score(Y[test], clf.predict_proba(X[test])[:, 1])





      s = s + 'AUC Ev. score: {:.3}\n'.format(auc_)

      s = s + '###########################\n'



    best_auc_scores_holdout.append(auc_)





    if (writefolder != None):

      plt.plot(params_scores, 's-', label='fold {}'.format(i+1))

      plt.plot([0,len(params_scores)], [auc_, auc_], label='auc fold {}: {:.3}'.format(i+1, auc_))



  

  file_ = open(writefolder+'/'+'report_nested_cross_validation_hyperparameter_tuning.txt', 'w')

  file_.write(s)

  file_.close()



  if (writefolder != None):

    plt.legend(loc="lower right")

    plt.savefig(writefolder+'/'+'nested_cross_validation_scores.png', dpi=100)

  



  return best_params_all, best_auc_scores_holdout



# you can test combinations here

features = [

  #'Patient age quantile',

  'SARS-Cov-2 exam result',

  #'Hematocrit', #correlacao 1 com hemoglob

  #'Platelets', # good

  #'Mean platelet volume ', # good

  'Leukocytes', # amazing!

  'Basophils',

  'Eosinophils', #good

  'Monocytes', #good

  #'Rhinovirus_Enterovirus',

  #'Proteina C reativa mg_dL', #good

  #'Creatinine', # good

]



yname = 'SARS-Cov-2 exam result'
# Choose one

#resampling = 'smote'

resampling = 'undersampling'

#resampling = None

#



file_ = '/kaggle/input/covid19/dataset.xlsx'

work_folder = '/kaggle/working'



df = pd.read_excel(file_)[features]
# took this code block https://www.kaggle.com/ossamum/exploratory-data-analysis-and-feature-importance

# changed a little bit

full_null_series = (df.isnull().sum() == df.shape[0])

full_null_columns = full_null_series[full_null_series == True].index

print(full_null_columns.tolist())

df.drop(full_null_columns, axis=1, inplace=True)



contain_null_series = (df.isnull().sum() != 0).index

target = yname

just_one_target = []

for col in contain_null_series:

    i = df[df[col].notnull()][target].nunique()

    if i == 1:

        just_one_target.append(col)    

# columns that only are present when covid is negative        

print(just_one_target)

for col in just_one_target:

    print(df[df[col].notnull()][target].unique())



df.drop(just_one_target, axis=1, inplace=True)



# dataprep categorical

mask_pos_neg = {'positive': 1, 'negative': 0}

mask_detected = {'detected': 1, 'not_detected': 0}

mask_notdone_absent_present = {'not_done': np.nan, 'absent': 0, 'present': 1}

mask_normal = {'normal': 1}

mask_urine_color = {'light_yellow': 1, 'yellow': 2, 'citrus_yellow': 3, 'orange': 4}

mask_urine_aspect = {'clear': 1, 'lightly_cloudy': 2, 'cloudy': 3, 'altered_coloring': 4}

mask_realizado = {'Não Realizado': 0}

mask_urine_leuk = {'<1000': 0}

mask_urine_crys = {'Ausentes': 1, 'Urato Amorfo --+': 0, 'Oxalato de Cálcio +++': 0,

                   'Oxalato de Cálcio -++': 0, 'Urato Amorfo +++': 0}

#df = df.replace(mask_detected)

df = df.replace(mask_pos_neg)

#df = df.replace(mask_notdone_absent_present)

#df = df.replace(mask_normal)

#df = df.replace(mask_realizado)

#df = df.replace(mask_urine_leuk)

#df = df.replace(mask_urine_color)

#df = df.replace(mask_urine_aspect)

#df = df.replace(mask_urine_crys)

#df['Urine - pH'] = df['Urine - pH'].astype('float')

#df['Urine - Leukocytes'] = df['Urine - Leukocytes'].astype('float')



#x = df.drop(['Patient ID', 'SARS-Cov-2 exam result'], axis=1)

#x.fillna(999999, inplace=True)

#y = df['SARS-Cov-2 exam result']

###
# I'll skip the EDA stage. There is a lot of good notebooks about this out there.

df
# randomize here

df = df.sample(frac=1, random_state=int(time.time()))

df = df.reset_index(drop=True)



# I'd like to use data imputation here but in this dataset it is very dangerous. Let's stick to the dropna().

df = df.dropna()



reg_y0 = df[df[yname]==0]

reg_y1 = df[df[yname]==1]



print ('Records with y==1:{}'.format(len(reg_y1)))

print ('Records with y==0:{}'.format(len(reg_y0)))



if resampling == 'undersampling':

  rus = RandomUnderSampler()

  X_resampled, y_resampled = rus.fit_resample(df, df[yname].values)

  df = pd.DataFrame(X_resampled, columns=df.columns)



  resampling_par = None # ugly thing but I do not want to another routine to resample the data

    

  print ('After undersampling:')

  reg_y0 = df[df[yname]==0]

  reg_y1 = df[df[yname]==1]

  print ('Records with y==1:{}'.format(len(reg_y1)))

  print ('Records with y==0:{}'.format(len(reg_y0)))

elif resampling == 'smote':

  print ('The smote procedure will be take place inside cros_val function.')
# standardize 

df2 = (df-df.mean())/df.std()



# data standard

df_X = df2.drop([yname], axis=1)

df_Y = np.rint(df.copy()[yname])



# Here I'll explore a few different hyperparameters due to the cpu processing limits, in my computer I run usually thousands.

# Look at output folder for the complete report file

list_best_models, auc_scores = grid_search_nested_parallel(df_X.values, df_Y.values, cv=3, writefolder=work_folder, n_jobs=1, resampling=resampling_par)
lst_features = list(df.columns)

lst_features.remove(yname)

reg_y0 = df[df[yname]==0]

reg_y1 = df[df[yname]==1]



print ('Features: ' + ', '.join(lst_features))

print ('registros y==1:{}, y==0:{}'.format(len(reg_y0), len(reg_y1)))

print ('resampling: {}'.format(str(resampling)))

for model, auc in zip(list_best_models, auc_scores):

  print ('model:{}, AUC score holdout:{}'.format(model, auc))
