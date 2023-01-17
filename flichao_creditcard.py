import numpy as np

from imblearn.datasets import fetch_datasets

from kmeans_smote import KMeansSMOTE

import itertools

import pandas as pd



from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from imblearn.metrics import geometric_mean_score

from sklearn.metrics import confusion_matrix

from matplotlib.axes import Axes







from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier





from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)



from imblearn.metrics import geometric_mean_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score
def compute_measure(y_test, y_pred):

    '''依次返回AUC，G-mean, Accuracy'''

    print("AUC={:.6f},  Gmean={:.6f},  Accuracy={:.6f}"

          .format(roc_auc_score(y_test, y_pred), geometric_mean_score(y_test, y_pred), accuracy_score(y_test, y_pred)))

    

    

def plot_confusion_matrix(cm, classes, ax,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    print(cm)

    print('')

    font1 = {'family':'Times New Roman',

             "weight":"normal",

             'size':10,

            }

    font2 = {'family' : 'Times New Roman',

            'weight' : 'normal',

            'size'   : 8,

            }



    ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set_title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, fontproperties = 'Times New Roman', size = 9)

    plt.sca(ax)

    plt.yticks(tick_marks, classes, fontproperties = 'Times New Roman', size = 9)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 

                color="white" if cm[i, j] > thresh else "black",

                fontdict=font1

               )



    ax.set_ylabel('True label',fontdict=font2)

    ax.set_xlabel('Predicted label', fontdict=font2)

    

    

    

def diff_k(nclusters, kneighbours, fname):

    

    #####导入数据#####

    data = pd.read_csv("../input/creditcardfraud/{}.csv".format(fname))

    data = np.array(data)

    np.random.shuffle(data)

    y = data[: :,-1]

    y = y.astype('int')

    X = data[:, : -1]

    [print('Class {} has {} instances after oversampling'.format(label, count)) 

     for label, count in zip(*np.unique(y, return_counts=True))]

    

    #####数据采样#####

    kmeans_smote = KMeansSMOTE(

        kmeans_args={

            'n_clusters': nclusters

        },

        smote_args={

            'k_neighbors': kneighbours

        }

        )



    X_resampled, y_resampled = kmeans_smote.fit_sample(X, y)

    [print('Class {} has {} instances after oversampling'.format(label, count))

         for label, count in zip(*np.unique(y_resampled, return_counts=True))]



    

    #####原始数据集模型训练##########

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    rf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)

    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)



    ######平衡数据集模型训练###########

    X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_resampled, y_resampled, stratify=y_resampled, random_state=0)

    rf_bal = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)

    rf_bal.fit(X_train_bal, y_train_bal)

    y_pred_brf = rf_bal.predict(X_test)

    y_pred_balanced = rf_bal.predict(X_test_bal)

    

    font1 = {'family' : 'Times New Roman',

    'weight' : 'normal',

    'size'   : 10,

    }



    print('不采样:')#############################################################

    compute_measure(y_test, y_pred_rf)

    cm_rf = confusion_matrix(y_test, y_pred_rf)

    fig, ax = plt.subplots(ncols=3)

    plot_confusion_matrix(cm_rf, classes=np.unique(y), ax=ax[0])

    plt.title(u"Result1", fontdict=font1)

   

    font1 = {'family' : 'Times New Roman',

            'weight' : 'normal',

            'size'   : 10,

            }

    print('采样后使用未采样数据测试:')#################################################

    compute_measure(y_test, y_pred_brf)

    cm_brf = confusion_matrix(y_test, y_pred_brf)

    plot_confusion_matrix(cm_brf, classes=np.unique(y), ax=ax[1])

    plt.title(u"Result2",fontdict=font1)

   

    

    

    print('采样后使用采样后数据测试:')###################################################

    compute_measure(y_test_bal, y_pred_balanced)

    cm_brf = confusion_matrix(y_test_bal, y_pred_balanced)

    plot_confusion_matrix(cm_brf, classes=np.unique(y), ax=ax[2])

    plt.title(u"Result3", fontdict=font1)

    

    

    

    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0.7)

#     plt.savefig("figs2/RF_{}_K{}N{}.svg".format(fname, nclusters, kneighbours), bbox_inches='tight')
diff_k(100, 5,"creditcard")