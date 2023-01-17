import numpy as np 
import pandas as pd
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/diabetes-data-set/diabetes.csv")
df.head()
df.info()
sns.pairplot(df)
y = df.Outcome
x = df.drop(columns=["Outcome"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=0).fit(x_train, y_train)
pred = clf.predict(x_test)
print(accuracy_score(y_test, pred))
plot_confusion_matrix(clf, x_test, y_test)
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0,
                                   max_features = 'auto', max_depth = 10)
ranfor.fit(x_train, y_train)
pred_ranfor = ranfor.predict(x_test)
print(accuracy_score(y_test, pred_ranfor))
plot_confusion_matrix(ranfor, x_test, y_test)
from sklearn import svm
sv = svm.SVC(kernel='linear')
sv.fit(x_train, y_train)
pred_svm = sv.predict(x_test)
print(accuracy_score(y_test, pred_svm))
plot_confusion_matrix(sv, x_test, y_test)
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
import statsmodels.stats.tests.test_influence
res = GLM(y_train, x_train,
          family=families.Binomial()).fit(attach_wls=True, atol=1e-10)
print(res.summary())
pred = np.array(res.predict(x_test), dtype=float)
table = np.histogram2d(y_test, pred, bins=2)[0]
table
print("Statmodel Acc : ", (table[0,0] + table[1,1])/(table[0,0] + table[1,1]+table[1,0] + table[0,1]))
ax = sns.heatmap(table, linewidth=0.5)
plt.show()
import xgboost as xgb
dt = xgb.DMatrix(x_train,label=y_train)
dv = xgb.DMatrix(x_test,label=y_test)
params = {
    "eta": 0.2,
    "max_depth": 4,
    "objective": "binary:logistic",
    "silent": 1,
    "base_score": np.mean(y_train),
    'n_estimators': 1000,
    "eval_metric": "logloss"
}
model = xgb.train(params, dt, 3000, [(dt, "train"),(dv, "valid")], verbose_eval=200)
y_pred = model.predict(dv)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, (y_pred>0.5))
print(cm)
# Calculate the accuracy on test set
predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])
ax = sns.heatmap(cm, linewidth=0.5)
plt.show()
print("xgboost Acc : ", predict_accuracy_on_test_set)
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_confusion_matrix

eval_dataset = Pool(x_test,
                    y_test)

model = CatBoostClassifier(learning_rate=0.0001,
                           eval_metric='AUC')

model.fit(x_train,
          y_train,
          eval_set=eval_dataset,
          verbose=False)

print(model.get_best_score())
cm = get_confusion_matrix(model, eval_dataset)
print(cm)
predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])
ax = sns.heatmap(cm, linewidth=0.5)
plt.show()
print("catboost Acc : ", predict_accuracy_on_test_set)