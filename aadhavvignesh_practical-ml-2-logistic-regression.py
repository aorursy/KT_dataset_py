import pandas as pd

df = pd.read_csv("../input/diabetes-dataset/diabetes2.csv")
df.head()
df.info()
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df.Age)
sns.jointplot(df.Age, df['Glucose'], kind = 'kde')
sns.pairplot(df, hue = 'Outcome')
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(true, pred):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Confusion Matrix:\n", confusion_matrix(true, pred))
    cm = pd.crosstab(true, pred)
    sns.heatmap(cm, annot=True)
    print("Accuracy Score:", accuracy_score(true, pred))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
X = df.drop(['Outcome'], axis = 1)
y = df['Outcome']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver = 'liblinear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
evaluate(y_test, y_pred)
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(12, 8))
plt.plot(precisions, recalls)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("PR Curve: precisions/recalls tradeoff");
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], "k--")
plt.axis([0, 1, 0, 1])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
from sklearn.model_selection import GridSearchCV

penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
solver = ['liblinear', 'saga']

param_grid = dict(penalty=penalty, C=C, class_weight=class_weight, solver=solver)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc',
                    verbose=1, n_jobs=-1, cv=10, iid=True)
grid_result = grid.fit(X_train, y_train)
y_pred = grid_result.predict(X_test)

evaluate(y_test, y_pred)