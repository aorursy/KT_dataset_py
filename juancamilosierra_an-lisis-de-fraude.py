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

path_data = os.path.join("..", "input","PS_20174392719_1491204439457_log.csv")
df = pd.read_csv(path_data)
df.head()
df.describe()
print("Numero de entradas: ", len(df))
df_sorted = df.sort_values(["nameDest", "step"])
df_sorted.head()
null_df = pd.DataFrame(columns=df.columns)
null_df
df_concat = pd.concat([df_sorted, df_sorted.shift(1)], axis=1)
df_concat.columns = null_df.columns.append(null_df.columns.map(lambda x: x + "_1"))
df_concat.head()
df_concat.keys()
df_concat.loc[df_concat.nameDest != df_concat.nameDest_1, ["type_1", "nameOrig_1"]] = "NE"
df_concat.loc[df_concat.nameDest != df_concat.nameDest_1, ['oldbalanceOrg_1', 'newbalanceOrig_1', 'oldbalanceDest_1', 'newbalanceDest_1', 'isFraud_1', 'amount_1', 'step_1', 'isFlaggedFraud_1']] = 0
df_concat.head()
len(df_concat[df_concat.nameDest != df_concat.nameDest_1])
len(df_concat)
fraude = df_concat[df_concat.isFraud == 1]
no_fraude = df_concat[df_concat.isFraud == 0]
l_fraude = len(fraude)
l_no_fraude = len(no_fraude)
l_fraude/len(df), l_no_fraude/len(df)
l_fraude/(l_fraude + l_no_fraude*(5/100))
new_df = pd.concat([fraude, no_fraude.sample(int(l_no_fraude*(5/100)))])
print("numero de entradas en nuestro nueva matriz: ",len(new_df))
delta = 1

new_df["amount_oldbalancaOrg"]  = (((new_df["amount"] + delta) / (new_df["oldbalanceOrg"] + delta) + 1) - abs((new_df["amount"] + delta) / (new_df["oldbalanceOrg"] + delta)-1)) / 2
new_df["amount_newbalanceOrig"] = (((new_df["amount"] + delta) / (new_df["newbalanceOrig"] + delta)  + 1)  - abs((new_df["amount"] + delta) / (new_df["newbalanceOrig"] + delta)-1)) / 2

new_df["amount_oldbalanceDest"] = (((new_df["amount"] + delta) / (new_df["oldbalanceDest"] + delta) + 1) - abs((new_df["amount"] + delta) / (new_df["oldbalanceDest"] + delta)-1)) / 2
new_df["amount_newbalanceDest"] = (((new_df["amount"] + delta) / (new_df["newbalanceDest"] + delta) + 1) - abs((new_df["amount"] + delta) / (new_df["newbalanceDest"] + delta)-1)) / 2


new_df["amount_oldbalancaOrg_ant"]  = (((new_df["amount_1"] + delta) / (new_df["oldbalanceOrg_1"] + delta) + 1) - abs((new_df["amount_1"] + delta) / (new_df["oldbalanceOrg_1"] + delta)-1)) / 2
new_df["amount_newbalanceOrig_ant"] = (((new_df["amount_1"] + delta) / (new_df["newbalanceOrig_1"] + delta) + 1) - abs((new_df["amount_1"] + delta) / (new_df["newbalanceOrig_1"] + delta)-1)) / 2

new_df["amount_oldbalanceDest_ant"] = ((((new_df["amount_1"] + delta) / (new_df["oldbalanceDest_1"] + delta) + 1) - abs((new_df["amount_1"] + delta) / (new_df["oldbalanceDest_1"] + delta) -1))) / 2
new_df["amount_newbalanceDest_ant"] = ((((new_df["amount_1"] + delta) / (new_df["newbalanceDest_1"] + delta) + 1) - abs((new_df["amount_1"] + delta) / (new_df["newbalanceDest_1"] + delta) -1))) / 2
delta = 1.5

new_df["oldbalanceOrg_log"] = np.log(new_df["oldbalanceOrg"] + delta)
new_df["newbalanceOrig_log"] = np.log(new_df["newbalanceOrig"] + delta)
new_df["oldbalanceDest_log"] = np.log(new_df["oldbalanceDest"] + delta)
new_df["newbalanceDest_log"] = np.log(new_df["newbalanceDest"] + delta)

new_df["oldbalanceOrg_ant_log"] = np.log(new_df["oldbalanceOrg_1"] + delta)
new_df["newbalanceOrig_ant_log"] = np.log(new_df["newbalanceOrig_1"] + delta)
new_df["oldbalanceDest_ant_log"] = np.log(new_df["oldbalanceDest_1"] + delta)
new_df["newbalanceDest_ant_log"] = np.log(new_df["newbalanceDest_1"] + delta)
new_df[[ 'amount_oldbalancaOrg', 'amount_newbalanceOrig',
       'amount_oldbalanceDest', 'amount_newbalanceDest',
       'amount_oldbalancaOrg_ant', 'amount_newbalanceOrig_ant',
       'amount_oldbalanceDest_ant', 'amount_newbalanceDest_ant',
       'oldbalanceOrg_log', 'newbalanceOrig_log', 'oldbalanceDest_log',
       'newbalanceDest_log', 'oldbalanceOrg_ant_log', 'newbalanceOrig_ant_log',
       'oldbalanceDest_ant_log', 'newbalanceDest_ant_log']].describe()
fig, [[ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]] = plt.subplots(2,4, figsize=(15, 5))


new_df["oldbalanceOrg_log"].hist(ax=ax1)
new_df["newbalanceOrig_log"].hist(ax=ax2)
new_df["oldbalanceDest_log"].hist(ax=ax3) 
new_df["newbalanceDest_log"].hist(ax=ax4)

new_df["oldbalanceOrg_ant_log"].hist(ax=ax5)
new_df["newbalanceOrig_ant_log"].hist(ax=ax6)
new_df["oldbalanceDest_ant_log"].hist(ax=ax7) 
new_df["newbalanceDest_ant_log"].hist(ax=ax8)

plt.tight_layout()
fig, [[ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]] = plt.subplots(2,4, figsize=(15, 5))

new_df.amount_oldbalancaOrg.hist(ax=ax1, bins=50)
new_df.amount_newbalanceOrig.hist(ax=ax2, bins=50)
new_df.amount_oldbalanceDest.hist(ax=ax3, bins=50)
new_df.amount_newbalanceDest.hist(ax=ax4, bins=50)
new_df.amount_oldbalancaOrg_ant.hist(ax=ax5, bins=50)
new_df.amount_newbalanceOrig_ant.hist(ax=ax6, bins=50)
new_df.amount_oldbalanceDest_ant.hist(ax=ax7, bins=50)
new_df.amount_newbalanceDest_ant.hist(ax=ax8, bins=50)

plt.tight_layout()
def balance_log_buckets(limit_bill, f=1./2):
    centroides = [0, 5, 10, 15, 20]
    bk = np.array([np.exp(-1 * abs(limit_bill[0] - c) * f) for c in centroides])
    bk_n = bk / bk.sum() # Escala manual
    return bk_n
def step_map(step, base=743):
    return np.array([step[0] + 1 / base])
v = np.vectorize(balance_log_buckets, signature="(1)->(5)",)
v(np.array([[0], [5.3]]))
new_df.keys()
cols = ['step', 'type', 
       'isFlaggedFraud', 
       'type_1', 
       'isFlaggedFraud_1', 
       'amount_oldbalancaOrg', 'amount_newbalanceOrig',
       'amount_oldbalanceDest', 'amount_newbalanceDest',
       'amount_oldbalancaOrg_ant', 'amount_newbalanceOrig_ant',
       'amount_oldbalanceDest_ant', 'amount_newbalanceDest_ant',
       'oldbalanceOrg_log', 'newbalanceOrig_log', 'oldbalanceDest_log',
       'newbalanceDest_log', 'oldbalanceOrg_ant_log', 'newbalanceOrig_ant_log',
       'oldbalanceDest_ant_log', 'newbalanceDest_ant_log']

new_df.step.describe()
positivos = new_df[new_df.isFraud == 1]
negativos = new_df[new_df.isFraud == 0]
X = new_df[cols].values
y = new_df.isFraud
X.shape
X
cols
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

balance_transformer = FunctionTransformer(np.vectorize(balance_log_buckets, signature="(1)->(5)"))
step_transformer = FunctionTransformer(np.vectorize(step_map, signature="(1)->(1)"), )
indentity_transformer = FunctionTransformer(np.vectorize(lambda x: x))



ct = ColumnTransformer([("StepTransformer", step_transformer , [0]),
                        ("TypeTransformer", OneHotEncoder(), [1]),
                        ("IsFlaggedFraudTransformer", indentity_transformer, [2]),
                        ("TypeAntTransformer", OneHotEncoder(), [3]),
                        
                        ("IsFlaggedFraudAntTransformer", indentity_transformer, [4]),
                        ("AmountOldbalancaOrgTransformer", indentity_transformer, [5]),
                        ("AmountNewbalanceOrigTransformer", indentity_transformer, [6]),
                        ("AmountOldbalanceDestTransformer", indentity_transformer, [7]),
                        ("AmountNewbalanceDestTransformer", indentity_transformer, [8]),
                        
                        ("AmountOldbalancaOrgAntTransformer", indentity_transformer, [9]),                        
                        ("AmountNewbalanceOrigAntTransformer", indentity_transformer, [10]),
                        ("AmountOldbalanceDestAntTransformer", indentity_transformer, [11]),
                        ("AmountNewbalanceDestAntTransformer", indentity_transformer, [12]),
                        
                        ("OldbalanceOrgLogTransformer", balance_transformer, [13]),
                        ("NewbalanceOrigLogTransformer", balance_transformer, [14]),
                        ("OldbalanceDestLogTransformer", balance_transformer, [15]),
                        ("NewbalanceDestLogTransformer", balance_transformer, [16]),
                        
                        ("OldbalanceOrgAntLogTransformer", balance_transformer, [17]),
                        ("NewbalanceOrigAntLog1Transformer", balance_transformer, [18]),
                        ("OldbalanceDestAnt_logTransformer", balance_transformer, [19]),
                        ("NewbalanceDestAntLogTransformer", balance_transformer, [20]),
                        
                       ], remainder="passthrough")
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

pca = PCA(n_components=30)
csf_MLPC = MLPClassifier(hidden_layer_sizes=(100,))

piper = Pipeline([("transformer", ct), ("scaler", StandardScaler()), ("csf", csf_MLPC)])
piper.fit(X_train, y_train)    
score_train = piper.score(X_train, y_train)
score_test = piper.score(X_test, y_test)
print("score train: {}, socre test: {}".format(score_train, score_test))
from sklearn import metrics

fig, ax = plt.subplots(1, figsize=(15,5))
metrics.plot_roc_curve(piper, X_test, y_test, ax=ax)
y_predic = piper.predict(X_test) 
confusion_matrix(y_test, y_predic, labels=[1,0])
