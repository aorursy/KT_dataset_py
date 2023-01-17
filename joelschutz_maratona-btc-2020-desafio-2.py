import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv('/kaggle/input/desafio-2/dataset_desafio_2.csv')
df
from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# All sklearn Transforms must have the `transform` and `fit` methods
class ColumnsTransformerGOINGLES(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        cl = ColumnTransformer([
        ('go_median', SimpleImputer(missing_values=np.nan,strategy='median'), ['NOTA_GO']),
        ('ingles_contant', SimpleImputer(missing_values=np.nan,strategy='most_frequent'), ['INGLES'])
                          
        ],remainder='drop')
        data['NOTA_GO'] = pd.DataFrame(cl.fit_transform(data))[0]
        data['INGLES'] = pd.DataFrame(cl.fit_transform(data))[1]
        return data
class SimpleImputerCustom(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        si = SimpleImputer(missing_values=np.nan,strategy='median')
        return pd.DataFrame.from_records(data=si.fit_transform(X=data), columns=data.columns)
class CombNotaReprov(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
      
    def comb(self, data):
        return pd.Series([
        data['NOTA_DE']-data['REPROVACOES_DE'],
        data['NOTA_EM']-data['REPROVACOES_EM'],
        data['NOTA_MF']-data['REPROVACOES_MF'],
        data['NOTA_GO']-data['REPROVACOES_GO']], index =['COMB_DE', 'COMB_EM','COMB_MF','COMB_GO']
        )
          
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        data = data.join(data.apply(self.comb, axis=1))
        data.drop(labels=['REPROVACOES_DE', 'REPROVACOES_EM', 'REPROVACOES_MF', 'REPROVACOES_GO'], axis=1, inplace=True)
        data.drop(labels=['NOTA_DE', 'NOTA_EM', 'NOTA_MF', 'NOTA_GO'], axis=1, inplace=True)
        return data
class CombMedias(BaseEstimator, TransformerMixin):
    def __init__(self, columns, name):
        self.columns = columns
        self.name = name

    def fit(self, X, y=None):
        return self
      
    def comb(self, data):
        return pd.Series([
        np.sum([data[nota] for nota in self.columns])/len(self.columns)], index =[f'COMB_{self.name}']
        )
          
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        data = data.join(data.apply(self.comb, axis=1))
        return data
class FillNan(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        self.dataframe = None

    def fit(self, X, y=None):
        self.dataframe = X.copy().join(y) if y is not None else None
        return self
   
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        if self.dataframe is not None:
          data = self.dataframe
          perfis = data['PERFIL'][:]
          medias = data.groupby('PERFIL')[self.column].median()
          data = data.set_index(['PERFIL'])
          data[self.column] = data[self.column].fillna(medias)
          data.reset_index(inplace=True)
          data.drop(['PERFIL'], axis=1, inplace=True)
          self.dataframe = None
        return data
class DropNAGOINGLES(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        data.dropna(inplace=True)
        return data.reset_index(inplace=True)
# Sampling
samples = 5773 
from imblearn.over_sampling import SMOTE

smote_1 = SMOTE(sampling_strategy='all', random_state=500, n_jobs=2)
smote_2 = SMOTE(sampling_strategy='auto', random_state=500, n_jobs=2)
smote_3 = SMOTE(sampling_strategy={'DIFICULDADE':samples, 'EXATAS':samples, 'EXCELENTE':samples, 'HUMANAS':samples, 'MUITO_BOM':samples}, random_state=500, n_jobs=2)
from imblearn.over_sampling import SVMSMOTE

svmsmote_1 = SVMSMOTE(sampling_strategy='all', random_state=500, n_jobs=2)
svmsmote_2 = SVMSMOTE(sampling_strategy='auto', random_state=500, n_jobs=2)
svmsmote_3 = SVMSMOTE(sampling_strategy={'DIFICULDADE':samples, 'EXATAS':samples, 'EXCELENTE':samples, 'HUMANAS':samples, 'MUITO_BOM':samples}, random_state=500, n_jobs=2)
from imblearn.under_sampling import TomekLinks

tlusmote_1 = TomekLinks(sampling_strategy='all', random_state=500, n_jobs=2)
tlusmote_2 = TomekLinks(sampling_strategy='auto', random_state=500, n_jobs=2)
tlusmote_3 = TomekLinks(sampling_strategy='majority', random_state=500, n_jobs=2)
smote_list = {'smote_tradicional_all':smote_1,
              'smote_tradicional_auto':smote_2,
              'smote_tradicional_custom':smote_3,
              'smote_svm_all':svmsmote_1,
              'smote_svm_auto':svmsmote_2,
              'smote_svm_custom':svmsmote_3,
              'smote_tlunder_all':tlusmote_1,
              'smote_tlunder_auto':tlusmote_2,
              'smote_tlunder_majority':tlusmote_3}
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=300, min_samples_split=50, criterion='gini', n_jobs=2)
import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective='multi:softprob', learning_rate = 0.01,
                max_depth = 5, n_estimators =500, n_jobs=2, subsample=0.6, random_state=42)
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier

final_estimator = ExtraTreesClassifier(n_jobs=2)

stack_model = StackingClassifier([('ransom-forest', rf_model), ('xgboost', xgb_model)],final_estimator=final_estimator, n_jobs=2)

misael_model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0, criterion='entropy', n_jobs=2)
clarisse_model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, n_estimators=1600,
                       n_jobs=2, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
rs = RobustScaler()
ss = StandardScaler()
mms = MinMaxScaler()


def classify(model, df_ibm, smote_used=None, scaler=None):
  # preparação das amostras
  X = df_ibm.drop(['PERFIL'], axis=1).to_numpy()
  y = df_ibm['PERFIL'].to_numpy()

  if scaler == 'robust': X = rs.fit_transform(X)
  elif scaler == 'standard': X = ss.fit_transform(X)
  elif scaler == 'minmax': X = mms.fit_transform(X)


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=500)

  if smote_used: X_train, y_train = smote_used.fit_resample(X_train, y_train)

  # Classification Report e Confusion Matrix
  y_predic = model.fit(X_train, y_train).predict(X_test)
  report = classification_report(y_test, y_predic, output_dict=True)
  report = pd.DataFrame.from_dict(report).T
  plt_report = report['precision'].drop(['accuracy', 'macro avg', 'weighted avg'])
  return plt_report, accuracy_score(y_test, y_predic)

def plot_results(model, df, scaler=None):
  accuracies = pd.DataFrame()
  scores = []
  for name, method in smote_list.items():
    accuracies[name], score = classify(model,df, method, scaler=scaler)
    scores.append(score)
  accuracies['standard'], score = classify(model, df, scaler=scaler)
  scores.append(score)

  fig = plt.figure()
  sns.set_style("whitegrid")
  fig.set_size_inches(18, 18)
  fig.suptitle('Acurácia das variações do modelo')
  for index, column in enumerate(accuracies.columns):
    ax = fig.add_subplot(accuracies.shape[1]/3+1, 3, index+1)
    sns.barplot(accuracies.index, accuracies[column], ax=ax)
    ax.set_title(f'{column} - {scores[index]:.2f}')
    ax.set_ylabel('Precision')
    plt.xticks(rotation=45)
    plt.ylim(0.0,1.0)
  fig.subplots_adjust(hspace=0.8)
  plt.show()

# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME'])
si = SimpleImputerGOINGLES()

# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(si.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME'])
cf = ColumnsTransformerGOINGLES()

# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(cf.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME'])
si = SimpleImputerGOINGLES()
cnr = CombNotaReprov()

# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(si.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cnr.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME'])
cf = ColumnsTransformerGOINGLES()
cnr = CombNotaReprov()

# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(cf.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cnr.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME'])
si = SimpleImputerGOINGLES()
cm = CombMedias(['NOTA_DE', 'NOTA_MF', 'NOTA_GO', 'NOTA_EM'], 'NOTAS')

# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(si.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cm.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME'])
cf = ColumnsTransformerGOINGLES()
cm = CombMedias(['NOTA_DE', 'NOTA_MF', 'NOTA_GO', 'NOTA_EM'], 'NOTAS')

# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(cf.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cm.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME', 'NOTA_GO'])
si = SimpleImputerGOINGLES()
cmh = CombMedias(['NOTA_EM', 'NOTA_DE'], 'HUMANAS')

# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(si.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cmh.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME', 'NOTA_GO'])
si = SimpleImputerGOINGLES()
cm = CombMedias(['NOTA_DE', 'NOTA_MF', 'NOTA_EM'], 'NOTAS')
cmh = CombMedias(['NOTA_EM', 'NOTA_DE'], 'HUMANAS')


# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(si.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cm.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cmh.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME', 'INGLES'])
fn = FillNan('NOTA_GO')
cm = CombMedias(['NOTA_DE', 'NOTA_MF', 'NOTA_EM', 'NOTA_GO'], 'NOTAS')
cmh = CombMedias(['NOTA_EM', 'NOTA_DE'], 'HUMANAS')
dn = DropColumns(['NOTA_DE', 'NOTA_EM', 'NOTA_GO'])


# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(fn.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cm.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cmh.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(dn.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME', 'INGLES', 'REPROVACOES_DE',
                          'REPROVACOES_MF', 'REPROVACOES_EM', 'REPROVACOES_GO'])
fn = FillNan('NOTA_GO')
cm = CombMedias(['NOTA_DE', 'NOTA_MF', 'NOTA_EM', 'NOTA_GO'], 'NOTAS')
cmh = CombMedias(['NOTA_EM', 'NOTA_DE'], 'HUMANAS')
dn = DropColumns(['NOTA_DE', 'NOTA_EM', 'NOTA_GO'])


# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(fn.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cm.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cmh.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(dn.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
print(model.feature_importance_)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# objetos utilizados
rm_columns = DropColumns(['MATRICULA', 'NOME', 'INGLES', 'REPROVACOES_DE',
                          'REPROVACOES_MF', 'REPROVACOES_EM', 'REPROVACOES_GO',
                          'FALTAS', 'H_AULA_PRES', 'TAREFAS_ONLINE'])
fn = FillNan('NOTA_GO')
cm = CombMedias(['NOTA_DE', 'NOTA_MF', 'NOTA_EM', 'NOTA_GO'], 'NOTAS')


# limpeza dos dados
df_ibm = pd.DataFrame.from_records(rm_columns.fit_transform(df))
df_ibm = pd.DataFrame.from_records(fn.fit_transform(df_ibm))
df_ibm = pd.DataFrame.from_records(cm.fit_transform(df_ibm))

df_ibm
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = rf_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = xgb_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = stack_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = misael_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm)
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='robust')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='standard')
# modelo utilizado
dtc_model = clarisse_model

# Classificando por smote
plot_results(dtc_model, df_ibm, scaler='minmax')
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

model = xgb.XGBClassifier(objective='multi:softprob',
                          learning_rate=0.5,
                        max_depth=1, 
                        min_child_weight=1, 
                        n_estimators=400, nthread=1, 
                        subsample=0.9000000000000001)

rm_columns = DropColumns(['MATRICULA', 'NOME', 'INGLES', 'REPROVACOES_DE',
                          'REPROVACOES_MF', 'REPROVACOES_EM', 'REPROVACOES_GO',
                          'FALTAS', 'H_AULA_PRES', 'TAREFAS_ONLINE'])
# rm_columns = DropColumns(['MATRICULA', 'NOME'])
fn = FillNan('NOTA_GO')
cm = CombMedias(['NOTA_DE', 'NOTA_MF', 'NOTA_EM', 'NOTA_GO'], 'NOTAS')
ch = CombMedias(['NOTA_DE', 'NOTA_EM'], 'HUMANAS')
ce = CombMedias(['NOTA_MF', 'NOTA_GO'], 'EXATAS')

pipeline = Pipeline(steps=[
                      # ('fill-nan', fn),
                      ('comb_medias', cm),
                      ('comb_humanas', ch),
                      ('comb_exatas', ce),
                      ('rm_columns', rm_columns),
                      ('xgboost', model),
])

# df_ibm = fn.fit_transform(df)
df_ibm = df

X = df_ibm.drop(['PERFIL'], axis=1)
y = df_ibm['PERFIL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=82)


y_predic = pipeline.fit(X_train, y_train).predict(X_test)
print('Desbalanceado', classification_report(y_test, y_predic))
feature_importances = pd.DataFrame(pipeline.steps[-1][-1].feature_importances_,
                                   index = ['NOTA_DE','NOTA_EM','NOTA_GO','COMB_NOTAS', 'COMB_HUMANAS', 'COMB_EXATAS'],
                                   columns=['importance']).sort_values('importance', ascending=False)
feature_importances