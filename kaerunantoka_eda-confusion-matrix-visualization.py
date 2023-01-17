import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
print('Train')
train = pd.read_csv("../input/train/train.csv")
print(train.shape)

target = train['AdoptionSpeed']
train_id = train['PetID']
train.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)
# drop categorical features to simplify
train.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)
train.shape
# 5 classes classificasion

lgb_params = {'objective':'multiclass',
              'num_class': 5, 
              'learning_rate': 0.1,
              'boosting': 'gbdt',
              'n_estimators': 10000, 
              'random_state': 2019}
from sklearn.model_selection import KFold

folds = KFold(n_splits=3, shuffle=True, random_state=15)

classes = sorted(target.unique())
oof_preds = np.zeros((len(train), len(classes)))

features = [c for c in train.columns if c not in ['target']]

for fold_, (trn_, val_) in enumerate(folds.split(train.values, target.values)):
    trn_x, trn_y = train.iloc[trn_][features], target.iloc[trn_]
    val_x, val_y = train.iloc[val_][features], target.iloc[val_]
    
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        verbose=100,
        early_stopping_rounds=100,
    )
    oof_preds[val_] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
    
print(oof_preds.shape)
oof_preds[:5]
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
unique_y = np.unique(target)
class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i
        
y_map = np.zeros((target.shape[0],))
y_map = np.array([class_map[val] for val in target])
y_map.shape
import itertools
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds,axis=-1))
np.set_printoptions(precision=2)

class_names = classes # list [0, 1, 2, 3, 4]

# Plot non-normalized confusion matrix
plt.figure(figsize=(7,7))
foo = plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                      title='Confusion matrix')
