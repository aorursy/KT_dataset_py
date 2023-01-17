import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
data = pd.read_csv("/kaggle/input/advertising/advertising.csv")
data.head()
data.info()
data.describe()
plt.figure(figsize=(10, 8))
data.Age.hist(bins=data.Age.nunique())
plt.xlabel('Age')
sns.jointplot(data["Area Income"], data.Age)
sns.jointplot(data["Daily Time Spent on Site"], data.Age, kind='kde')
sns.jointplot(data["Daily Time Spent on Site"], data["Daily Internet Usage"])
sns.pairplot(data, hue='Clicked on Ad')
data['Clicked on Ad'].value_counts()
x = np.linspace(-6, 6, num=1000)
plt.figure(figsize=(10, 6))
plt.plot(x, (1 / (1 + np.exp(-x))))
plt.title("Sigmoid Function")
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(true, pred):

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"CONFUSION MATRIX:\n{confusion_matrix(true, pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(true, pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n\tPrecision: {precision:.4f}\n\tRecall: {recall:.4f}\n\tF1_Score: {f1:.4f}")
data["Country"] = data.Country.astype('category').cat.codes
data["City"] = data.City.astype('category').cat.codes
X = data.drop(['Timestamp', 'Clicked on Ad', 'Ad Topic Line'], axis=1)
y = data['Clicked on Ad']
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver='liblinear', penalty='l1')
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
evaluate(y_test, y_pred)
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim(0, 1.5)
plt.style.use("fivethirtyeight")
plt.figure(figsize=(12, 8))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
plt.figure(figsize=(12, 8))
plt.plot(precisions, recalls)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("PR Curve: precisions/recalls tradeoff");
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
plt.figure(figsize=(12,8)); 
plot_roc_curve(fpr, tpr)
plt.show();
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
from sklearn.model_selection import GridSearchCV

penalty = ['l1', 'l2']
C = [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2]
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
solver = ['liblinear', 'saga']

param_grid = dict(penalty=penalty, C=C, class_weight=class_weight, solver=solver)

grid = GridSearchCV(estimator=log_reg, param_grid=param_grid, scoring='accuracy',
                    verbose=1, n_jobs=-1, cv=10, iid=True)
grid_result = grid.fit(X_train, y_train)
grid_result.best_params_
y_pred = grid_result.predict(X_test)

evaluate(y_test, y_pred)