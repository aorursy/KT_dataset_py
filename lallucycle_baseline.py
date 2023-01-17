import pandas as pd

import numpy as np

from scipy import interp

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
df1 = pd.read_csv('../input/baseline2/OOF_preds.csv').drop(columns = 'Unnamed: 0')

df2 = pd.read_csv('../input/baseline1/OOF_preds.csv').drop(columns = 'Unnamed: 0')

df = pd.concat([df1,df2],axis =0)

df = df.sort_values(by = 'row').reset_index(drop=True)

df.to_csv('OOF_preds.csv')
def pT_classes(a):

    if a<=10:

        return 0

    if a>10 and a<=30:

        return 1

    if a>30 and a<=100:

        return 2

    if a>100:

        return 3
df['True_pT'] = 1/df['true_value']
classes = ['0-10 GeV','10-30 GeV','30-100 GeV','100-inf GeV','micro','macro']

for i in range(6):

    try:

        fpr,tpr,_ = roc_curve(df['pT_classes']==i, df.iloc[:,i+3])

        roc_auc = auc(fpr, tpr)

    except:

        pppppppp = 1

    if i==4:

        y_score = df.iloc[:,3:7].to_numpy()

        y_test = np.array([df['pT_classes']==0,df['pT_classes']==1,df['pT_classes']==2,df['pT_classes']==3]).T*1.0

        fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())

        roc_auc = auc(fpr, tpr)

    if i==5:

        y_score = df.iloc[:,3:7].to_numpy()

        y_test = np.array([df['pT_classes']==0,df['pT_classes']==1,df['pT_classes']==2,df['pT_classes']==3]).T*1.0

        all_fpr = np.unique(np.concatenate([roc_curve(df['pT_classes']==i, df.iloc[:,i+3])[0] for i in range(4)]))

        mean_tpr = np.zeros_like(all_fpr)

        for j in range(4):

            A = roc_curve(df['pT_classes']==j, df.iloc[:,j+3])

            mean_tpr += interp(all_fpr, A[0], A[1])

        mean_tpr /= 4

        fpr = all_fpr

        tpr = mean_tpr

        roc_auc = auc(fpr, tpr)

    print(classes[i],'| auc | ',roc_auc)

    plt.plot(fpr, tpr, color='darkorange',

             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic ( '+classes[i]+ ' )')

    plt.legend(loc="lower right")

    plt.show()

    
print(classification_report(df['True_pT'].apply(pT_classes), df.iloc[:,3:7].to_numpy().argmax(axis = 1)))
x = pd.DataFrame(confusion_matrix(df['True_pT'].apply(pT_classes), df.iloc[:,3:7].to_numpy().argmax(axis = 1)))

x.columns = ['pred | '+i for i in ['0-10','10-30','30-100','100-inf']]

x.index = ['true | '+ i for i in ['0-10','10-30','30-100','100-inf']]
## confusion-matrix

x