import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve,average_precision_score
import xgboost
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
np.unique(df.Class, return_counts=True)
# Select feature columns
selFeatureColumns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
       'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
       'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
       'Amount']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df[selFeatureColumns], df.Class, test_size=0.5,stratify=df.Class,random_state=42 )
# Parameter grid for XGBoost with below combinations was tried 
# I have pre-selected best estimators obtained from GridSearch

xgb_param_grid = {
        'min_child_weight': [1],       # [1, 5, 10],
        'gamma': [1],                  # [0.01, 0.1, 1, 10],
        'subsample': [1],              # [0.6, 0.8, 1.0],
        'colsample_bytree': [0.7],     # [0.7, 0.8, 0.9],
        'max_depth': [3],              # [3 ,5 ,7]
        'learning_rate' : [0.1],       # [0.01, 0.1, 1, 10],
        'n_estimators': [200]          # [200,300,400]
        }

xgb = xgboost.XGBClassifier(random_state=42,scale_pos_weight=577,silent=True, n_jobs=-1)

gs = GridSearchCV(estimator = xgb, param_grid = xgb_param_grid, cv = 3, n_jobs = -1, scoring='recall', refit=True)

model = gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.best_score_)
y_pred = gs.predict(X_test)
print(classification_report(y_test,y_pred))
y_score_xg = model.predict_proba(X_test)[:,1]
average_precision = average_precision_score(y_test, y_score_xg)
precision, recall, _ = precision_recall_curve(y_test, y_score_xg)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class XGBoost Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
   
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}%".format(cm[i, j]*100),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()
    
# Source - https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm, normalize= True,target_names = ['Not Fraud', 'Fraud'], title = "Confusion Matrix")
