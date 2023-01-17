import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import label_binarize

train_data = pd.read_csv("../input/train.csv")
feature_cols = [x for x in train_data.columns if x not in ["Target", "Id", "idhogar"]]
X = train_data[feature_cols]
y = train_data.Target
y_bin = label_binarize(y, [1,2,3,4])

#Split train-test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=.1, random_state=42)
y_train = list(map(lambda x: np.argmax(x)+1, y_train_bin))
y_test = list(map(lambda x: np.argmax(x)+1, y_test_bin))
#These columns are not numerical. They cannot be feed into any model. (Id and idhogar are obvious)
[(x, y) for (x,y) in X_train.dtypes.to_dict().items() if y=="O"]
#These columns have NaNs
[x for x in X_train.columns if sum(X_train[x].isna())>0]
fill_dict = {"v2a1": X_train.v2a1.median(), #Monthly rent payment
             "v18q1": 0, #number of tablets household owns
             "rez_esc": X_train.rez_esc.median(), #Years behind in school
             "meaneduc": X_train.meaneduc.median(), #average years of education for adults (18+)
            }
X_train = X_train.fillna(fill_dict)
X_test = X_test.fillna(fill_dict)
X_train.SQBmeaned = np.sqrt(X_train.meaneduc)
X_test.SQBmeaned = np.sqrt(X_test.meaneduc)
[x for x in X_train.columns if sum(X_train[x].isna())>0]
def clean_yes_no_column(serie, train=True, train_mean=None):
    _serie = serie.apply(lambda x: 0 if x=="no" else x)
    _serie = _serie.apply(lambda x: float(x) if x!="yes" else x)
    if train:
        mean_value = _serie[_serie != "yes"].mean()
    else:
        mean_value = train_mean
    return _serie.apply(lambda x: mean_value if x=="yes" else x)
#Clean those nasty categorical columns
X_train.dependency = clean_yes_no_column(X_train.dependency)
X_train.edjefe = clean_yes_no_column(X_train.edjefe)
X_train.edjefa = clean_yes_no_column(X_train.edjefa)

X_test.dependency = clean_yes_no_column(X_test.dependency, False, X_train.dependency.mean())
X_test.edjefe = clean_yes_no_column(X_test.edjefe, False, X_train.edjefe.mean())
X_test.edjefa = clean_yes_no_column(X_test.edjefa, False, X_train.edjefa.mean())
from sklearn.metrics import roc_curve, auc, f1_score
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp

def plot_roc_curve(y_score, y_test_bin, y_test, fig_size = (15,10)):
    n_classes = y_train_bin.shape[1]
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure(figsize=fig_size)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    predictions = list(map(lambda x: np.argmax(x)+1, y_score))
    F1 = f1_score(y_test, predictions, average="macro")
    return F1
from sklearn.linear_model import logistic

model = logistic.LogisticRegression()
model.fit(X_train, y_train)
y_score = model.predict_proba(X_test)
plot_roc_curve(y_score, y_test_bin, y_test, (15, 10))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_score = model.predict_proba(X_test)
plot_roc_curve(y_score, y_test_bin, y_test, (15, 10))
train_data = pd.read_csv("../input/train.csv")
feature_cols = [x for x in train_data.columns if x not in ["Target", "Id", "idhogar"]]
X = train_data[feature_cols]
y = train_data.Target
fill_dict = {"v2a1": X.v2a1.median(), #Monthly rent payment
             "v18q1": 0, #number of tablets household owns
             "rez_esc": X.rez_esc.median(), #Years behind in school
             "meaneduc": X.meaneduc.median(), #average years of education for adults (18+)
            }
X = X.fillna(fill_dict)
X.SQBmeaned = np.sqrt(X.meaneduc)

X.dependency = clean_yes_no_column(X.dependency)
X.edjefe = clean_yes_no_column(X.edjefe)
X.edjefa = clean_yes_no_column(X.edjefa)

model =  RandomForestClassifier()
model.fit(X, y)

#Split train-test
X_test = pd.read_csv("../input/test.csv")
X_test = X_test.fillna(fill_dict)
X_test.SQBmeaned = np.sqrt(X_test.meaneduc)

X_test.dependency = clean_yes_no_column(X_test.dependency, False, X_train.dependency.mean())
X_test.edjefe = clean_yes_no_column(X_test.edjefe, False, X_train.edjefe.mean())
X_test.edjefa = clean_yes_no_column(X_test.edjefa, False, X_train.edjefa.mean())
test_id = X_test.Id
y_predict = model.predict(X_test[feature_cols])
pred = pd.DataFrame({"Id": test_id, "Target": y_predict})
pred.to_csv('submission.csv', index=False)
pred.head()