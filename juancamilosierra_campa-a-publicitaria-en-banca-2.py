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
cols = ["age", "job", "balance_log", "loan", "duration", 
         "day", "pdays", "previous", "poutcome", "campaign",
         "education", "marital", "default", "housing", 
        ]
from sklearn.preprocessing import LabelBinarizer

X = df[cols].values

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

pca = PCA(n_components=30)

csf_ridge = RidgeClassifier()
csf_logistic = LogisticRegression()
csf_MLPC = MLPClassifier(hidden_layer_sizes=(50,))

piper = Pipeline([("transformer", ct), ("scaler", StandardScaler()), ("pca", pca), ("csf", csf_MLPC)])
from sklearn.utils import resample


def re_sample_x_y(X, y, n_samples, label_resample=1,): 
    
    X_y_train = np.concatenate([X, np.array([y]).T], axis=1)
    X_y_pos = X_y_train[X_y_train[:,-1] == label_resample]
    X_y_neg = X_y_train[X_y_train[:,-1] != label_resample]
    X_y_pos_resample = resample(X_y_pos, n_samples=n_samples) 
    X_y_ = np.concatenate([X_y_pos_resample, X_y_neg])            
    
    X_ = X_y_[:,:-1]
    y_ = X_y_[:,-1].astype(int)

    return X_, y_
def resample_df(df_train, n_sample, x_cols=cols, y_cols="y", pos_label=1):
    
    df_train_pos = df_train[df_train[y_cols] == pos_label]
    df_train_neg = df_train[df_train[y_cols] != pos_label]
    
    df_train_pos_resample = df_train_pos.sample(n_sample, replace=True)
    df_resample = pd.concat([df_train_pos_resample, df_train_neg]).sample(frac=1)
    
    X_ = df_resample[cols]
    y_ = df_resample[y_cols]
    
    return X_,y_
negatives =  df[df.y == -1]
positives  =  df[df.y == 1]
p = 1/100
len_pos = len(negatives) / (1/p - 1)
print(len_pos, len_pos / (len_pos + len(negatives)))
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

confusion_matrices = list()

len_positives = [0] + [int(len(negatives) / ( (1 / p) - 1)) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

for i, n_spl in enumerate(len_positives):  
    
    df_train, df_test = train_test_split(df, test_size=0.33)
    
    X_test, y_test = df_test[cols], df_test["y"]
    
    X_train_resample, y_train_resample = resample_df(df_train, n_spl)
    
    piper.fit(X_train_resample, y_train_resample)
    
    y_predic = piper.predict(X_test)
    
    score = piper.score(X_test, y_test)
    
    c_m = confusion_matrix(y_test, y_predic, labels=[-1,1])
    
    confusion_matrices.append(c_m)   
    
    print("iteration done: {}, % positives in sample : {}, score test: {} ".format(i+1, n_spl/(n_spl+len(negatives)), score))
print(confusion_matrices)
# ROC SPACE = x_FPR / y_TPR

FPR = list()
TPR = list()

for c_m in confusion_matrices:
    
    fpr = c_m[0,1] / sum(c_m[0,0:2])
    tpr = c_m[1,1] / sum(c_m[1,0:2])
    
    FPR.append(fpr)
    TPR.append(tpr)   
import matplotlib.pyplot as plt

fig, [ax1, ax2] = plt.subplots(1, 2)
ax1.scatter(FPR, TPR)
ax2.scatter(FPR, TPR)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)








