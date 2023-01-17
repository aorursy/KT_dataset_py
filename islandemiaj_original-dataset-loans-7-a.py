!pip install fklearn
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from matplotlib import pyplot as plt
import fklearn, matplotlib
from tqdm import tqdm

from sklearn.feature_selection import SelectKBest, chi2 
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from fklearn.preprocessing.splitting import time_split_dataset
from fklearn.preprocessing.splitting import space_time_split_dataset

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import itertools
from itertools import groupby
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
#from HelperMethods import HelperClass
# https://github.com/jaimemishima/Data-Science-Projects/blob/master/Credit%20Card%20Fraud%20Detection.ipynb
class HelperClass(object):
        
    # helper method
    @staticmethod
    def stars():
        print ("***********************")
    
    # print metrics as dataframe
    @staticmethod
    def print_dataframe(values):
    
        metrics_print = ['True Positive', 'True Negative', 'False Negative', 'False Positive',
    'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'Roc Auc Score']

        df_metrics = pd.DataFrame(
            {'Metrics': metrics_print,
             'Values': values
            })

        print (df_metrics.to_string(header=False, index=False))
        HelperClass.stars()
        
                
        
    # Disclaimer: metodo obtido em:
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    # Metodo para plotar a matrix de confusao
    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix:")
        else:
            print('Confusion matrix, without normalization:')

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
        
        # adjust plot
        
        bottom, top = plt.gca().get_ylim()
        plt.gca().set_ylim(bottom + 0.5, top - 0.5)
        plt.show()




    # ROC curve
    @staticmethod
    def plot_roc_curve(y_true, y_scores):

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        HelperClass.stars()
        print ("Roc Curve:")
        HelperClass.stars()

        plt.plot(fpr, tpr, label = 'ROC Curve', linewidth = 2)
        plt.plot([0,1],[0,1], 'k--', linewidth = 2)
        plt.title('ROC Curve')
        plt.xlim([0.0, 0.001])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()



    # Precision Recall Curve
    @staticmethod
    def plot_precision_recall_curve(y_true, y_scores):

        HelperClass.stars()
        print ("Precision Recall Curve:")
        HelperClass.stars()

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')

        plt.plot(recall, precision, linewidth=2)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.show()



    # Show classification report
    @staticmethod
    def show_full_classification_report(y_true, y_pred, y_scores, classes):

        HelperClass.stars()
        print ("Metrics Report:")
        HelperClass.stars()

        cm = confusion_matrix(y_true, y_pred)

        true_positive = cm[1,1]
        true_negative = cm[0,0]
        false_negative = cm[1,0]
        false_positive = cm[0,1]

        accuracy = ((true_positive + true_negative)/(true_positive + true_negative + false_negative + false_positive))
        precision = (true_positive/(true_positive + false_positive))
        recall = (true_positive/(true_positive + false_negative))  
        sensitivity = (true_positive/(true_positive + false_negative))  
        specificity = (true_negative/(true_negative + false_positive))  
        f1_score = ((2 * precision * recall)/(precision + recall))
        
        roc_auc = roc_auc_score(y_true, y_scores)

        print ("Classification Report:")
        HelperClass.stars()

        values = []

        values.append(true_positive)
        values.append(true_negative)
        values.append(false_negative)
        values.append(false_positive)
        values.append('{:.4f}'.format(accuracy))
        values.append('{:.4f}'.format(precision))
        values.append('{:.4f}'.format(recall))
        values.append('{:.4f}'.format(specificity))
        values.append('{:.4f}'.format(f1_score))
        values.append('{:.4f}'.format(roc_auc))

        HelperClass.print_dataframe(values)

        HelperClass.plot_confusion_matrix(cm, classes)

        HelperClass.plot_roc_curve(y_true, y_scores)

        HelperClass.plot_precision_recall_curve(y_true, y_scores)
pd.set_option('display.max_columns', None) # para mostrar todas as colunas
# raw = pd.read_excel("FOIA - 7(a)(FY2010-Present).xlsx", sheet_name="7A_FY2010_Present")
# raw.to_csv("foia_7a.csv",index = False, header=True)
raw_base = pd.read_csv("foia_7a.csv")
df = raw_base.copy()
df.head()
df['AsOfDate'] = df['AsOfDate'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
df['ApprovalDate'] = df['ApprovalDate'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
df['FirstDisbursementDate'] = df['FirstDisbursementDate'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
df['PaidInFullDate'] = df['PaidInFullDate'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
df['Target'] = np.where(df['PaidInFullDate'].isnull(), 1, 0)  
df['LoanStatus'].value_counts()
# vamos manter somente os emprestimos que foram totalmente pagos (PIF = Paid in Full) e cobrados (CHGOFF = Charged Off)
# COMMIT = não desembolsado, CANCLD = cancelado, EXEMPT = foram desembolsados mas não foram cancelados, cobrados ou pagos.
df = df[df['LoanStatus'].isin(['PIF', 'CHGOFF'])]
df.to_csv("foia_7a_tratado.csv",index = False, header=True)
df = pd.read_csv("foia_7a_tratado.csv")
#df['AsOfDate'] = df['AsOfDate'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
df['ApprovalDate'] = df['ApprovalDate'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
df['FirstDisbursementDate'] = df['FirstDisbursementDate'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
#df['PaidInFullDate'] = df['PaidInFullDate'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
# removing unnused columns
df = df.drop(columns=['AsOfDate', 'Program', 'BorrName', 'BorrStreet', 'BankStreet', 'LoanStatus'])

# removing variables related with target
df = df.drop(columns=['PaidInFullDate', 'ChargeOffDate', 'GrossChargeOffAmount'])

# converting categorical columns
df['NaicsCode'] = df['NaicsCode'].astype(str)
df['CongressionalDistrict'] = df['CongressionalDistrict'].astype(str)
df['BorrZip'] = df['BorrZip'].astype(str)
df['BankZip'] = df['BankZip'].astype(str)

df['Id'] = range(1, 1+len(df))
dados = df.copy()
dados.head()
dados.head(50)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
dados.describe().transpose()
dados.isna().mean()
dados.apply(pd.Series.nunique).sort_values()
target = 'Target'
dados.corr(method='pearson')[target].sort_values()
plt.figure(figsize=(10, 5))
ax = sns.heatmap(dados.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=0.5, linecolor='black')

# rotate xlabel
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.axes.set_title("Matriz de Correlação",fontsize=20)
#ax.set_xlabel("X Label",fontsize=30)
#ax.set_ylabel("Y Label",fontsize=20)
ax.tick_params(labelsize=10)

# adjust plot
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
dados.head()
year_volume = dados['ApprovalDate'].groupby(dados["ApprovalDate"].dt.year).count().to_frame()#.plot(kind="bar")
year_volume = year_volume.rename(columns={"ApprovalDate": "Count Loans"})
year_volume['Share Loans'] = year_volume['Count Loans'].div(year_volume['Count Loans'].sum(), axis=0).multiply(100)
year_volume
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=year_volume.index, y='Count Loans',data=year_volume, dodge=False)
ax.axes.set_title("Volume de Loans anual",fontsize=15)
dados['ApprovalDate'].min()
dados['ApprovalDate'].max()
split_fn = space_time_split_dataset(train_start_date="2009-10-01",
                                    train_end_date="2015-12-31",
                                    holdout_end_date="2019-09-30",
                                    split_seed=42,
                                    space_holdout_percentage=0.2,
                                    space_column="Id",
                                    time_column="ApprovalDate")

dados_amostral = dados.sample(n = 10000)
train_set, out_of_space_ho, out_of_time_ho, out_of_space_time_ho =  split_fn(dados_amostral)
print(train_set.shape)
print(out_of_space_ho.shape)
print(out_of_time_ho.shape)
print(out_of_space_time_ho.shape)
numerical_variables = dados.describe().transpose().index.tolist()
categorical_variables = list(set(dados.columns.tolist()) - set(numerical_variables))
numerical_variables.remove('Target')
numerical_variables.remove('Id')
print('Variveis numericas: ',numerical_variables)
print('Variveis categoricas: ',categorical_variables)
from fklearn.training.imputation import imputer, placeholder_imputer
from toolz import compose

num_impute_learner = imputer(columns_to_impute=numerical_variables,
                             impute_strategy="median")

cat_impute_learner = placeholder_imputer(columns_to_impute=categorical_variables,
                                          placeholder_value="unk")

#tupla que retorna: [0] a funcao, [1] dataset e [2] log
num_impute_fn, _, num_impute_log = num_impute_learner(train_set)
cat_impute_fn, _, cat_impute_log = cat_impute_learner(train_set)

compose(num_impute_fn, cat_impute_fn)(train_set).isnull().mean()
pd.set_option('display.float_format', lambda x: '%.2f' % x)
dados.describe().transpose()
from fklearn.training.transformation import capper, floorer
from fklearn.training.transformation import label_categorizer
from fklearn.training.transformation import onehot_categorizer

# Capping altos
capper_fn = capper(columns_to_cap=['GrossApproval', 'SBAGuaranteedApproval'], 
                   precomputed_caps={'GrossApproval': 5000000,
                                     'SBAGuaranteedApproval': 5250000
                                    })

# One hot encoding (cria dummies)
categorical_features_onehot = ['BusinessType', 'DeliveryMethod', 'subpgmdesc']
oh_encode_learner = onehot_categorizer(columns_to_categorize=categorical_features_onehot,
                                       hardcode_nans=False, # hardcodes an extra column with 1 if nan or unseen else 0
                                       drop_first_column=True)

# Label encoding
categorical_features_label_encoding = list(set(categorical_variables) - set(categorical_features_onehot) 
                                           - set(['ApprovalDate']))
le_encode_learner = label_categorizer(
                                columns_to_categorize=categorical_features_label_encoding,
                                store_mapping=True,
)
from fklearn.training.pipeline import build_pipeline

pipeline_learner = build_pipeline(
    capper_fn,
    num_impute_learner,
    cat_impute_learner,
    oh_encode_learner,
    le_encode_learner
)
## Using the created pipeline I transform my data
_, pre_processed_data, _ = pipeline_learner(dados)
#_, pre_processed_data, _ = pipeline_learner(dados_amostral)
train_set, out_of_space_ho, out_of_time_ho, out_of_space_time_ho =  split_fn(pre_processed_data)
train_set.columns.tolist()
# cols = ['Target', 'fklearn_feat__PerfilEconomico==3', 'fklearn_feat__Sexo==mulher', 'fklearn_feat__PerfilCompra==8','fklearn_feat__RegiaodoPais==Região Sul']
# train_df_copy = train_df
# test_df_copy = test_df
# for col in cols:
#     if col in train_df_copy.columns:
#         train_df_copy = train_df_copy.drop(columns=col, axis=1)
#     if col in test_df_copy.columns:
#         test_df_copy = test_df_copy.drop(columns=col, axis=1)

explicativas_train = train_set[list(set(train_set) - set(['ApprovalFiscalYear', 'ApprovalFiscalYear', 'ApprovalDate', 'FirstDisbursementDate', 'Target']))]
target_train = train_set['Target']

explicativas_test = out_of_time_ho[list(set(out_of_time_ho) - set(['ApprovalFiscalYear', 'ApprovalFiscalYear', 'ApprovalDate', 'FirstDisbursementDate', 'Target']))]
target_test = out_of_time_ho['Target']
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

x_norm = MinMaxScaler().fit_transform(explicativas_train)

# chamada do objeto
chi2_selector = SelectKBest(chi2)

chi2_selector.fit(x_norm, target_train)

chi_s = chi2_selector.get_support()

chi_feature = explicativas_train.loc[:,chi_s].columns.tolist()
print(str(len(chi_feature)), 'variaveis selecionadas')
print(chi_feature)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# chamada do objeto
rfe_selector = RFE(estimator=LogisticRegression(), step=10) #default is half, n_features_to_select=10)

rfe_selector.fit(explicativas_train, target_train)
rfe_support = rfe_selector.get_support()
rfe_feature = explicativas_train.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'variaveis selecionadas:')
print(rfe_feature)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

em_selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
em_selector.fit(explicativas_train, target_train)

em_sup = em_selector.get_support()
em_feature = explicativas_train.loc[:,em_sup].columns.tolist()
print(str(len(em_feature)), 'variaveis selecionadas:')
print(em_feature)
feature_selection_df = pd.DataFrame({'Variaveis':explicativas_train.columns,
                                     'chi2':chi_s,
                                     'RFE': rfe_support,
                                     'Random forest': em_sup})
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
feature_selection_df = feature_selection_df.sort_values(['Total', 'Variaveis'], ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df
var_select = feature_selection_df[feature_selection_df['Total'] > 1]['Variaveis'].tolist()
x_treino = explicativas_train[var_select]
y_treino = target_train

x_teste = explicativas_test[var_select]
y_teste = target_test
from sklearn.linear_model import LogisticRegression
RL = LogisticRegression(random_state=42)
# hyperparameters = {"C":np.logspace(-3,3,7), 
#                    "penalty":["l1","l2"]} # l1 lasso l2 ridge

hyperparameters = {'penalty' : ['l1','l2'], 
                   'class_weight' : ['balanced', None], 
                   'C' : [0.001, 0.01, 0.1, 1, 10]
                  }
%%time
grid_RL = GridSearchCV(RL, 
                  hyperparameters, 
                  cv=10,
                  verbose=0)
grid_RL.fit(x_treino, y_treino)
grid_RL.best_params_
RL = LogisticRegression(**grid_RL.best_params_, random_state=42)
%%time
acuracias_RL_treino = cross_val_score(estimator=RL,
                            X = x_treino,
                            y = y_treino,
                            cv=10)

acuracias_RL_teste = cross_val_score(estimator=RL,
                            X = x_teste,
                            y = y_teste,
                            cv=10)
print('REGRESSAO LOGISTICA - Acuracia de Treino:',round(acuracias_RL_treino.mean()*100,2))
print('REGRESSAO LOGISTICA - Acuracia de Teste:',round(acuracias_RL_teste.mean()*100,2))
classes = ['Pago','Default']
y_pred = grid_RL.predict(x_teste)
y_scores_RL = grid_RL.predict_proba(x_teste)[:,1]

HelperClass.show_full_classification_report(y_teste, y_pred, y_scores_RL, classes)
parametros_grid = {'n_neighbors': [3, 5, 7],
                   'weights': ['uniform', 'distance']
                  }

KNN = KNeighborsClassifier()
%%time
from sklearn.model_selection import GridSearchCV

grid_KNN = GridSearchCV(estimator=KNN,
                    param_grid=parametros_grid,
                    scoring='recall',
                    cv=10)
grid_KNN.fit(x_treino, y_treino)
grid_KNN.best_params_
grid_KNN.best_score_
%%time
KNN = KNeighborsClassifier(**grid_KNN.best_params_)
KNN.fit(x_treino, y_treino)
%%time
from sklearn.model_selection import cross_val_score
acuracias_KNN_treino = cross_val_score(estimator=KNN,
                            X = x_treino,
                            y = y_treino,
                            cv=10)

acuracias_KNN_teste = cross_val_score(estimator=KNN,
                            X = x_teste,
                            y = y_teste,
                            cv=10)
print('RANDOM FOREST - Acuracia de Treino:',round(acuracias_KNN_treino.mean()*100,2))
print('RANDOM FOREST - Acuracia de Teste:',round(acuracias_KNN_teste.mean()*100,2))
classes = ['Pago','Default']
y_pred = grid_KNN.predict(x_teste)
y_scores_KNN = grid_KNN.predict_proba(x_teste)[:,1]

HelperClass.show_full_classification_report(y_teste, y_pred, y_scores_KNN, classes)
%%time
from sklearn.ensemble import RandomForestClassifier

parametros_grid = {
    'n_estimators':[5,10,100],
    'criterion':['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_features': [2],
    'bootstrap':[True, False],
    'min_samples_leaf': [2,3],
    'max_depth':[5]
}

RF = RandomForestClassifier(random_state=123)
%%time
from sklearn.model_selection import GridSearchCV

grid_RF = GridSearchCV(estimator=RF,
                    param_grid=parametros_grid,
                    scoring='accuracy',
                    cv=5)
grid_RF.fit(x_treino, y_treino)
grid_RF.best_params_
grid_RF.best_score_
%%time
RF = RandomForestClassifier(**grid_RF.best_params_, random_state=42)
RF.fit(x_treino, y_treino)
%%time
from sklearn.model_selection import cross_val_score
acuracias_RF_treino = cross_val_score(estimator=RF,
                            X = x_treino,
                            y = y_treino,
                            cv=10)

acuracias_RF_teste = cross_val_score(estimator=RF,
                            X = x_teste,
                            y = y_teste,
                            cv=10)
print('RANDOM FOREST - Acuracia de Treino:',round(acuracias_RF_treino.mean()*100,2))
print('RANDOM FOREST - Acuracia de Teste:',round(acuracias_RF_teste.mean()*100,2))
classes = ['Pago','Default']
y_pred = grid_RF.predict(x_teste)
y_scores_RF = grid_RF.predict_proba(x_teste)[:,1]

HelperClass.show_full_classification_report(y_teste, y_pred, y_scores_RF, classes)
%%time
classes = ['Pago','Default']
voting_clf = VotingClassifier (
        estimators = [('lg', grid_RL), ('knn', grid_KNN), ('rf', grid_RF)], voting='soft')
    
voting_clf.fit(x_treino, y_treino)

y_pred = voting_clf.predict(x_teste)
y_scores_VotingClassifier = voting_clf.predict_proba(x_teste)[:,1]
HelperClass.show_full_classification_report(y_teste, y_pred, y_scores_VotingClassifier, classes)
from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state=42)
parametros_gb = {'min_samples_split': [3, 5],
                 'min_samples_leaf': [3, 5],
                 'max_depth': [3,5],
                 'n_estimators':[2,5],
                 'loss':['deviance'],
                 'learning_rate': [0.05, 0.2],
                 'max_features':["log2","sqrt"],
                 #'criterion': ["friedman_mse",  "mae"],
                 #'subsample':[0.5,0.8, 1.0]
                }
%%time
from sklearn.model_selection import GridSearchCV

# assinatura do objeto
grid_GB = GridSearchCV(estimator=GB,
                       param_grid=parametros_gb,
                       cv=5,
                       n_jobs=-1) # tenta rodar em paralelo, se possível
grid_GB.fit(x_treino, y_treino)
grid_GB.best_score_
grid_GB.best_params_
%%time
GB = GradientBoostingClassifier(**grid_GB.best_params_, random_state=42)
GB.fit(x_treino, y_treino)
%%time
from sklearn.model_selection import cross_val_score
acuracias_GB_treino = cross_val_score(estimator=GB,
                            X = x_treino,
                            y = y_treino,
                            cv=5)

acuracias_GB_teste = cross_val_score(estimator=GB,
                            X = x_teste,
                            y = y_teste,
                            cv=5)
print('GRADIENT BOOSTING - Acuracia de Treino:',round(acuracias_GB_treino.mean()*100,2))
print('GRADIENT BOOSTING - Acuracia de Teste:',round(acuracias_GB_teste.mean()*100,2))
classes = ['Pago','Default']
y_pred = grid_GB.predict(x_teste)
y_scores_GB = grid_GB.predict_proba(x_teste)[:,1]

HelperClass.show_full_classification_report(y_teste, y_pred, y_scores_GB, classes)
# %%time
# from sklearn.metrics import roc_auc_score, log_loss
# # An experiment to understand why weak learners work better:
# max_depth = np.unique(np.random.randint(3, 15, size=7))
# num_estimators = np.unique(np.concatenate((np.random.randint(3, 15, size=7), np.random.randint(15, 100, size=5)),axis=0))

# auc = pd.DataFrame()

# for depth in max_depth:
#     for tot_trees in num_estimators:
        
#         xgb_ = XGBClassifier(max_depth=depth, num_estimators=tot_trees)
#         xgb_.fit(x_treino, y_treino)
        
#         test_xgb  = xgb_.predict_proba(x_teste)[:,1]
#         auc_test  = roc_auc_score(y_teste.values, test_xgb)
        

#         auc = pd.concat([auc, pd.DataFrame(data={'AUC': [auc_test], 'max_depth':[depth] , 'num_estimators':[tot_trees]})], axis=0)
#         print(auc)
from xgboost import XGBClassifier

XGB = XGBClassifier(random_state=42)

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have 
#much fun of fighting against overfit 
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters_xgb = {#'nthread':[4], #when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'],
                  'learning_rate': [0.05,0.1], #so called `eta` value
                  'max_depth': [2],
                  'min_child_weight': [11],
                  #'silent': [1],
                  #'subsample': [0.8],
                  #'colsample_bytree': [0.7],
                  'n_estimators': [1000], #number of trees, change it to 1000 for better results
                  #'missing':[-999],
                  #'seed': [1337],
                  #'lambda':[1.2, 1.3],
                  #'alpha':[1.2, 1.3],

                 }
%%time
from sklearn.model_selection import GridSearchCV

# assinatura do objeto
grid_XGB = GridSearchCV(estimator=XGB,
                       param_grid=parameters_xgb,
                       cv=5,
                       n_jobs=-1) # tenta rodar em paralelo, se possível
grid_XGB.fit(x_treino, y_treino)
grid_XGB.best_params_
%%time
XGB = XGBClassifier(**grid_XGB.best_params_, random_state=42)
XGB.fit(x_treino, y_treino)
%%time
from sklearn.model_selection import cross_val_score
acuracias_XGB_treino = cross_val_score(estimator=XGB,
                            X = x_treino,
                            y = y_treino,
                            cv=5)

acuracias_XGB_teste = cross_val_score(estimator=XGB,
                            X = x_teste,
                            y = y_teste,
                            cv=5)
print('GRADIENT BOOSTING - Acuracia de Treino:',round(acuracias_XGB_treino.mean()*100,2))
print('GRADIENT BOOSTING - Acuracia de Teste:',round(acuracias_XGB_teste.mean()*100,2))
classes = ['Pago','Default']
y_pred = grid_XGB.predict(x_teste)
y_scores_XGB = grid_XGB.predict_proba(x_teste)[:,1]

HelperClass.show_full_classification_report(y_teste, y_pred, y_scores_XGB, classes)
modelos = pd.DataFrame({'Modelo': ['Regressao Logistica',
                                   'Random Forest',
                                   'Gradient Boosting',
                                   'XGboosting'
                                  ],
                        'Acuracia_treino':[round(acuracias_RL_treino.mean()*100,2), 
                                           round(acuracias_RF_treino.mean()*100,2),
                                           round(acuracias_GB_treino.mean()*100,2),
                                           round(acuracias_XGB_treino.mean()*100,2)],
                        'Acuracia_teste':[round(acuracias_RL_teste.mean()*100,2), 
                                          round(acuracias_RF_teste.mean()*100,2),
                                          round(acuracias_GB_teste.mean()*100,2),
                                          round(acuracias_XGB_teste.mean()*100,2)]
                       })

comparacao = modelos.sort_values(by='Acuracia_teste', ascending=False)
comparacao = comparacao[['Modelo', 'Acuracia_treino', 'Acuracia_teste']]
comparacao
grid_GB.best_estimator_
from sklearn.externals import joblib

# salva modelo Gradient Boosting
joblib.dump(grid_GB.best_estimator_, 'modelo_loan_gb_total.pkl', compress = 1)
# salva modelo Random Forest
joblib.dump(grid_RF.best_estimator_, 'modelo_loan_rf_total.pkl', compress = 1)
# salva modelo Regressao Logistica
joblib.dump(grid_RL.best_estimator_, 'modelo_loan_rl_total.pkl', compress = 1)
modelo_treinado = open('modelo_loan_gb.pkl', 'rb')
model = joblib.load(modelo_treinado)
model.predict(x_teste)
model.predict_proba(x_teste)[:,1]


