# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import pandas as pd
import os

print(os.getcwd())
path_data = os.path.join("..", "input","bank-full.csv")
df = pd.read_csv(path_data, sep=";", converters={"y": lambda y: -1 if y == "no" else 1,
                                                 })
map_jobs = {'management' : "professional",
 'technician' :  "technical",
 "entrepreneur": "professional",
 "blue-collar": "professional",
 "unknown" : "unemployed", 
 "retired" : "unemployed",
 "admin." : "professional",
 "services": "technical",
 "self-employed" : "technical",
 "unemployed" : "unemployed",
 "housemaid": "unemployed",
 "student": "unemployed"}
df.head()
df.describe()
df["balance_log"] = np.log10(np.abs(df.balance) + 1) * np.sign(df.balance)
df_neg = df[df.y == -1]
df_pos = df[df.y == 1]
df.balance_log[pd.isnull(df.balance_log)]
df.balance_log.describe()
df.balance_log.hist()
df_neg.balance_log.hist()
df_pos.balance_log.hist()
campaign_neg = df_neg.campaign 
campaign_pos = df_pos.campaign
campaign_neg[campaign_neg < 10].hist()
campaign_pos[campaign_pos < 10].hist()
duration_neg = df_neg.duration 
duration_pos = df_pos.duration 
duration_neg[duration_neg < 2000].hist()
duration_pos[duration_pos < 2000].hist()
pdays_neg = df_neg.pdays 
pdays_pos = df_pos.pdays 
pdays_neg.hist()
pdays_pos.hist()
days_neg = df_neg.day
days_pos = df_pos.day 
days_neg.hist(bins=range(1, 32),  density=1)
days_pos.hist(bins=range(1, 32),  density=1)
previous_neg = df_neg.previous
previous_pos = df_pos.previous 
previous_neg[previous_neg < 20].hist()
previous_pos[previous_pos < 20].hist()
from sklearn.preprocessing import LabelBinarizer
label_b = LabelBinarizer()

X = df[["age", "job", "balance_log", "loan", "duration", 
         "day", "pdays", "previous", "poutcome", "campaign",
         "education", "marital", "default", "housing", 
        ]].values

y = df.y.values

print(X)
print(y)
X.shape
def age_buckets(age, f=1./10):
    centroides = [20, 30, 40, 50, 60]
    bk = np.array([np.exp(-1 * abs(age[0] - c) * f) for c in centroides])
    bk_n = bk / bk.sum() # Escala manual
    return bk_n
def balance_buckets(balance, f=1./1):
    centroides = [-3, -2, -1, 0, 1, 2, 3]
    bk = np.array([np.exp(-1 * abs(balance[0] - c) * f) for c in centroides])
    bk_n = bk / bk.sum() # Escala manual
    return bk_n
def pdays_transform(pdays, base=365.242):
    pds = pdays[0]
    if pds == -1:
        return np.array([1,0])
    else:
        return np.array([0, pds / base])
def days_transform(day, f=1./5):
    centroides = [5, 10, 15, 20, 25, 30]
    bk = np.array([np.exp(-1 * abs(day[0] - c) * f) for c in centroides])
    bk_n = bk / bk.sum() # Escala manual
    return bk_n
v = np.vectorize(age_buckets, signature="(1)->(5)",)
v(np.array([[4], [38]]))
v = np.vectorize(balance_buckets, signature="(1)->(7)",)
v(np.array([[0.67], [5.0]]))
v = np.vectorize(pdays_transform, signature="(1)->(2)",)
v(np.array([[40], [-1]]))
v = np.vectorize(days_transform, signature="(1)->(6)",)
v(np.array([[4], [17]]))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


job_transformer = FunctionTransformer(np.vectorize(lambda job: map_jobs[job]))
indentity_transformer = FunctionTransformer(np.vectorize(lambda x: x))
age_transformer = FunctionTransformer(np.vectorize(age_buckets, signature="(1)->(5)"))
balance_transformer = FunctionTransformer(np.vectorize(balance_buckets, signature="(1)->(7)"))
pdays_transformer = FunctionTransformer(np.vectorize(pdays_transform, signature="(1)->(2)"))
days_transformer = FunctionTransformer(np.vectorize(days_transform, signature="(1)->(6)"))
job_onehot_ = Pipeline([("JobTransformer", job_transformer),("OneHotEncoder", OneHotEncoder())])

ct = ColumnTransformer([("AgeTransformer", age_transformer , [0]),
                        ("JobTransformer", job_onehot_, [1]),
                        ("BalanceTransformer", balance_transformer, [2]),
                        ("LoanTransformer", OneHotEncoder(),  [3]),
                        ("DurationTransformer", indentity_transformer, [4]),
                        ("DayTransformer", days_transformer, [5]),
                        ("PdaysTransformer", pdays_transformer, [6]),
                        ("PreviousTransformer", indentity_transformer, [7]),
                        ("PoutcomeTransformer", OneHotEncoder(),  [8]),
                        ("CampaignTransformer", indentity_transformer, [9]),
                        ("EducationTransformer", OneHotEncoder(), [10]),
                        ("MaritalTransformer", OneHotEncoder(), [11]),
                        ("DefaultTransformer", OneHotEncoder(), [12]),
                        ("HousingTransformer", OneHotEncoder(), [13]),
                       ],)
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

pca = PCA()

csf_ridge = RidgeClassifier()
csf_logistic = LogisticRegression()
csf_MLPC = MLPClassifier()

piper = Pipeline([("transformer", ct), ("scaler", StandardScaler()), ("pca", pca), ("csf", csf_MLPC)])

param_grid = {"csf__hidden_layer_sizes" : [(5,), (10,), (20,), (50,),  
                                           #(100,), (5,5), (10,10), (20,20),
                                          ],
             "pca__n_components": [30]
             }

search = GridSearchCV(piper, param_grid, n_jobs=-1, verbose=20)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape)
from sklearn.utils import resample

X_y_train = np.concatenate([X_train, np.array([y_train]).T], axis=1)
X_y_pos = X_y_train[X_y_train[:,-1] == 1]
X_y_neg = X_y_train[X_y_train[:,-1] == -1]
X_y_pos_resample = resample(X_y_pos, n_samples=160000) 
X_y_train = np.concatenate([X_y_pos_resample, X_y_neg])

np.random.shuffle(X_y_train)             

print(X_y_pos_resample.shape, X_y_neg.shape, X_y_train.shape,)
X_y_train[:,:-1]
set(X_y_train[:,-1].astype(int))
X_train_ = X_y_train[:,:-1]
y_train_ = X_y_train[:,-1].astype(int)
search.fit(X_train_, y_train_)
search.score(X_train_, y_train_)
search.score(X_test, y_test)
display(search.cv_results_)
from sklearn.metrics import confusion_matrix

y_predict = search.predict(X_test)
confusion_matrix(y_test, y_predict, labels=[1,-1])
import numpy as np
from sklearn import metrics

scores = search.predict_proba(X_test)
print(scores, y_test)
scores[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, scores[:,1], pos_label=1)
print(fpr, tpr, thresholds)
from sklearn import metrics
metrics.auc(fpr, tpr)
plt.scatter(fpr, tpr)
# Primera aplicación/objetivo: Con la condición de que la tasa de falsos positivos este entre cierto umbral, maximizar la tasa de true positive.
from matplotlib import cm

threshold_fpr = (0.0, 0.3)
fpr_tpr = np.column_stack((fpr, tpr))

condition = (fpr_tpr[:,0] >= threshold_fpr[0]) & (fpr_tpr[:,0] <= threshold_fpr[1]) 
fil = fpr_tpr[condition]
others = fpr_tpr[~condition]

colors = cm.get_cmap('Greens', 12)
newcolors = colors(np.linspace(0, 1, len(fil)))

plt.scatter(others[:,0], others[:,1], c="#477DF1")
plt.scatter(fil[:,0], fil[:,1], c=newcolors)
fil[fil[:,1] == max(fil[:,1])]
max([(x,y) for x,y in zip(fpr, tpr) if x  <= 0.3])
# Segunda aplicación/objetivo: Con la condición de la tasa de true positive este en un umbral, encontrar la tasa de falsos positivos que debo tolerar.
from matplotlib import cm

threshold_tpr = (0.8, 1)
fpr_tpr = np.column_stack((fpr, tpr))

condition = (fpr_tpr[:,1] >= threshold_tpr[0]) & (fpr_tpr[:,1] <= threshold_tpr[1]) 
fil = fpr_tpr[condition]
others = fpr_tpr[~condition]

colors = cm.get_cmap('Oranges', 12)
newcolors = colors(np.linspace(0.3, 1, len(fil)))

plt.scatter(others[:,0], others[:,1], c="#477DF1")
plt.scatter(fil[:,0], fil[:,1], c=newcolors[::-1])
min([(x,y) for x,y in zip(fpr, tpr) if y >= 0.8])
# Tercera aplicacion: Minimizar cierto costo o maximizar cierta ganancia.
# (p_positive * n_personas * tpr * (ganancia - cuesta llamada) -  (1-p_positive) * n_personas  * fpr * cuesta llamada)

ganacia_por_compra = 200
cuesta_llamada = 10
p_positive = len(df_pos) / len(df) 
n = len(df)

ganacia_total = lambda fpr_tpr : (p_positive * fpr_tpr[1]) * (ganacia_por_compra - cuesta_llamada) - (1 - p_positive) * (fpr_tpr[0] * cuesta_llamada)
fpr_tpr = np.column_stack((fpr, tpr))

gnc = list(map(ganacia_total, fpr_tpr))
s = np.column_stack((fpr, tpr, gnc))

max_g = max(s[:,2])

s[s[:,2] == max_g]
mm = s[s[:,2] == max_g]
nmm = s[s[:,2] != max_g]
plt.scatter(nmm[:,0], nmm[:,1], c='#477DF1')
plt.scatter(mm[:,0], mm[:,1], c='#010101')

