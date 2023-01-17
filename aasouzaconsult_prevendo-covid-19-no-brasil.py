import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Data processing, metrics and modeling
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics

# Suppr warning
import warnings
warnings.filterwarnings("ignore")

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import matplotlib.patches as patches

from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
%%time
dataset = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
dataset.columns = [x.lower().strip().replace(' ','_') for x in dataset.columns]
dataset.head()
dataset.info()
dataset.describe()
dataset['sars-cov-2_exam_result'] = dataset['sars-cov-2_exam_result'].replace(['negative','positive'], [0,1])
sns.countplot(dataset['sars-cov-2_exam_result'])
dataset['sars-cov-2_exam_result'].value_counts()
dataset.isnull().sum()
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
# Executando a função...
missing_data(dataset)
# Importando a biblioteca
import missingno as msno
msno.matrix(dataset.head(20000))
hm = msno.heatmap(dataset)
hm
dataset.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
### Consultando essas colunas (features)
## sem dados
#dataset['mycoplasma_pneumoniae'].value_counts()
#dataset['urine_-_sugar'].value_counts()
#dataset['prothrombin_time_(pt),_activity'].value_counts()
#dataset['d-dimer'].value_counts()
#dataset['partial_thromboplastin_time\xa0(ptt)'].value_counts()

## Valor 0.0 (apenas valores zeros)
#dataset['fio2_(venous_blood_gas_analysis)'].value_counts()
#dataset['myeloblasts'].value_counts()
dataset.drop('mycoplasma_pneumoniae',axis=1,inplace=True)
dataset.drop('urine_-_sugar',axis=1,inplace=True)
dataset.drop('prothrombin_time_(pt),_activity',axis=1,inplace=True)
dataset.drop('d-dimer',axis=1,inplace=True)
dataset.drop('partial_thromboplastin_time\xa0(ptt)',axis=1,inplace=True)

# Contem apenas 1 informação e é 0.0
dataset.drop('fio2_(venous_blood_gas_analysis)',axis=1,inplace=True)
dataset.drop('myeloblasts',axis=1,inplace=True)

# A coluna patient_id é muito especifica e atrapalha o modelo, iremos apagar!
dataset.drop('patient_id',axis=1,inplace=True)
# Visualizando o dataframe do dataset 
dataset
%%time
# Usando a média para valores float
for c in dataset.columns:
    if dataset[c].dtype=='float16' or  dataset[c].dtype=='float32' or  dataset[c].dtype=='float64':
        dataset[c].fillna(dataset[c].mean())

# 99999 para Categóricas
dataset = dataset.fillna(99999)

# Label Encoding para as Object
for f in dataset.columns:
    if dataset[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(dataset[f].values))
        dataset[f] = lbl.transform(list(dataset[f].values))
        
print('Discretização Realizada')
dataset
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib import pyplot
svm = SVC()
dt  = tree.DecisionTreeClassifier()
rf  = RandomForestClassifier()
knn = KNeighborsClassifier()
mlp = MLPClassifier()
adb = AdaBoostClassifier()
gnb = GaussianNB()
qda = QuadraticDiscriminantAnalysis()
xgb = xgb.XGBClassifier()
sdg = SGDClassifier()
X = dataset.drop('sars-cov-2_exam_result',axis=1)
y = dataset['sars-cov-2_exam_result']
all_models = [
    ("Modelo: SVM - Support Vector Machine", svm),
    ("Modelo: Decision Tree", dt),
    ("Modelo: Random Forest", rf),
    ("Modelo: KNN Classifier", knn),
    ("Modelo: MLP - Multi Layer Perceptron", mlp),
    ("Modelo: AdaBoost Classifier", adb),
    ("Modelo: Gaussian NB", gnb),
    ("Modelo: Quadratic Discriminant Analysis", qda),
    ("Modelo: XGB Classifier", xgb),
    ("Modelo: Stochastic Gradient Descent", sdg),
]
def benchmark_auc(model, X, y):
    scores = []
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    scores.append(cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1))
    print('-> Média ROC AUC: %.3f' % mean(scores))
    return np.mean(scores)
%%time
for name, model in all_models:
    print(name)
    benchmark_auc(model, X, y)

print("Resultados - ROC AUC - Concluídos")
X_train = dataset.drop('sars-cov-2_exam_result',axis=1)
y = dataset['sars-cov-2_exam_result']
#Visualizando a quantidade de dados por classe antes da execução do SMOTE
np.bincount(y)
from imblearn.over_sampling import SMOTE
smt = SMOTE(k_neighbors=20)
X_smt, y_smt = smt.fit_sample(X_train, y)
#Visualizando a quantidade de dados por classe após a execução do SMOTE
np.bincount(y_smt)
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
import xgboost as xgb
from matplotlib import pyplot
from numpy import mean
svm = SVC()
dt  = tree.DecisionTreeClassifier()
rf  = RandomForestClassifier()
knn = KNeighborsClassifier()
mlp = MLPClassifier()
adb = AdaBoostClassifier()
gnb = GaussianNB()
qda = QuadraticDiscriminantAnalysis()
xgb = xgb.XGBClassifier()
sdg = SGDClassifier()
X = X_smt
y = y_smt
%%time
for name, model in all_models:
    print(name)
    benchmark_auc(model, X, y)

print("Resultados - ROC AUC - Concluídos (Balanceado com SMOTE)")
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y,test_size=0.15,random_state=65)
xgb.fit(X_treino, y_treino)
print (pd.crosstab(y_teste, xgb.predict(X_teste), rownames=['Real'], colnames=['Predito'], margins=True), '')
print (metrics.classification_report(y_teste,xgb.predict(X_teste)))
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(xgb, X_teste, y_teste)
import shap
shap_values = shap.TreeExplainer(xgb).shap_values(X_treino)
shap.summary_plot(shap_values, X_treino, plot_type="bar")
shap.summary_plot(shap_values, X_treino)
shap.dependence_plot('patient_age_quantile', shap_values, X_treino)