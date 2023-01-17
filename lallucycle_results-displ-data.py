import os

import pandas as pd

import numpy as np

from scipy import interp

from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")
def pT_classes(a):

    if a<=10:

        return 0

    if a>10 and a<=30:

        return 1

    if a>30 and a<=100:

        return 2

    if a>100:

        return 3
def MAE_pT(df,dx = 0.5,r = 100):

    MAE1 = []

    for i in range(int(2/dx),int(150/dx)):

        P = df[(df['True_pT']>=(i-1)*dx)&(df['True_pT']<=(i+1)*dx)]

        p = mae(P['True_pT'],P['Predicted_pT'])/(i)

        if p<1:

            p=p

        else:

            p=p_

        MAE1.append(p)

        p_=p

    MAE1 = [0]*2*int(1/dx)+MAE1[:r*2-2*int(1/dx)]

    return MAE1
def MAE(df,dx = 0.5,r = 100):

    MAE1 = []

    for i in range(int(2/dx),int(150/dx)):

        P = df[(df['True_pT']>=(i-1)*dx)&(df['True_pT']<=(i+1)*dx)]

        p = mae(P['True_pT'],P['Predicted_pT'])

        if p<100:

            p=p

        else:

            p=p_

        MAE1.append(p)

        p_=p

    MAE1 = [0]*2*int(1/dx)+MAE1[:r*2-2*int(1/dx)]

    return MAE1
def acc_hist(df):

    acc = []

    for i in range(5,121):

        acc.append(accuracy_score(df['True_pT']>=i, df['Predicted_pT']>=i))

    return acc
def f1_upper_hist(df):

    f1 = []

    for i in range(5,121):

        f1.append(f1_score(df['True_pT']>=i, df['Predicted_pT']>=i))

    return f1
def f1_lower_hist(df):

    f1 = []

    for i in range(5,121):

        f1.append(f1_score(df['True_pT']<=i, df['Predicted_pT']<=i))

    return f1
def Frame1(path):

    path = '../input/'+path+'/OOF_preds.csv'

    try:

        df = pd.read_csv(path).drop(columns = 'Unnamed: 0')

    except:

        print('Unnamed: 0 not found in',path)

        df = pd.read_csv(path)

    df['True_pT'] = 1/df['true_value']

    try:

        df['Predicted_pT'] = 1/df['preds']

    except:

        p=1

    return df
def Frame2(path):

    path = '../input/'+path+'/OOF_preds.csv'

    try:

        df = pd.read_csv(path).drop(columns = 'Unnamed: 0')

    except:

        print('Unnamed: 0 not found in',path)

        df = pd.read_csv(path)

    df['True_pT'] = df['true_value']

    try:

        df['Predicted_pT'] = df['preds']

    except:

        p=1

    return df
def generate_classification_report(df):

    print('####################################################################################')

    print('                                      ROC-AUC                                       ')

    print('####################################################################################') 

    print()

    print()

    print()

    classes = ['0-10','10-30','30-100','100-inf','micro','macro']

    for i in range(6):

        try:

            fpr,tpr,_ = roc_curve(df['pT_classes']==i, df[classes[i]])

            roc_auc = auc(fpr, tpr)

        except:

            pppppppp = 1

        if i==4:

            y_score = df[classes[:4]].to_numpy()

            y_test = np.array([df['pT_classes']==0,df['pT_classes']==1,df['pT_classes']==2,df['pT_classes']==3]).T*1.0

            fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())

            roc_auc = auc(fpr, tpr)

        if i==5:

            y_score = df[classes[:4]].to_numpy()

            y_test = np.array([df['pT_classes']==0,df['pT_classes']==1,df['pT_classes']==2,df['pT_classes']==3]).T*1.0

            all_fpr = np.unique(np.concatenate([roc_curve(df['pT_classes']==i, df[classes[i]])[0] for i in range(4)]))

            mean_tpr = np.zeros_like(all_fpr)

            for j in range(4):

                A = roc_curve(df['pT_classes']==j, df[classes[j]])

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

    

    def pT_classes(a):

        if a==0:

            return '0-10 GeV'

        if a==1:

            return '10-30 GeV'

        if a==2:

            return '30-100 GeV'

        if a==3:

            return '100-inf GeV'

    

    print()

    print()

    print()

    print('####################################################################################')

    print('                      Sklearn Classification Report                                 ')

    print('####################################################################################')

    print(classification_report(df['pT_classes'], df[classes[:4]].to_numpy().argmax(axis = 1)))

    

    

    print()

    print()

    print()

    print('####################################################################################')

    print('                              Confusion Matrix                                      ')

    print('####################################################################################')

    x = pd.DataFrame(confusion_matrix(df['pT_classes'], df[classes[:4]].to_numpy().argmax(axis = 1)))

    x.columns = ['pred | '+i for i in ['0-10','10-30','30-100','100-inf']]

    x.index = ['true | '+ i for i in ['0-10','10-30','30-100','100-inf']]

    print(x)
def compare_regression(dfs = [], names = [], no = 0):



    dx = 0.5

    r = 125

    print('####################################################################################')

    print('                                    MAE/pT                                          ')

    print('####################################################################################')

    for i in range(no):

        plt.plot([i*dx for i in range(int(r/dx))][4:],MAE_pT(dfs[i],dx,r)[4:],label = names[i])

    plt.xlabel('pT (in GeV) -->')

    plt.ylabel('MAE/pT -->')

    plt.legend()

    plt.show()

    

    dx = 0.5

    r = 125

    print()

    print()

    print()

    print('####################################################################################')

    print('                                      MAE                                           ')

    print('####################################################################################')

    for i in range(no):

        plt.plot([i*dx for i in range(int(r/dx))][4:],MAE(dfs[i],dx,r)[4:],label = names[i])

    plt.xlabel('pT (in GeV) -->')

    plt.ylabel('MAE -->')

    plt.legend()

    plt.show()

    

    print()

    print()

    print()

    print('####################################################################################')

    print('                            Accuracy @ pT = x cut                                   ')

    print('####################################################################################')

    for i in range(no):

        plt.plot(range(5,121),acc_hist(dfs[i]), label = names[i])

    plt.xlabel('pT (in GeV) -->')

    plt.ylabel('Accuracy -->')

    plt.legend()

    plt.show()

    

    print()

    print()

    print()

    print('####################################################################################')

    print('                            F1 for class pT > x                                     ')

    print('####################################################################################')

    for i in range(no):

        plt.plot(range(5,121),f1_upper_hist(dfs[i]), label = names[i])

    plt.xlabel('pT (in GeV) -->')

    plt.ylabel('F1 (for class pT > x) -->')

    plt.legend()

    plt.show()

    

    print()

    print()

    print()

    print('####################################################################################')

    print('                            F1 for class pT < x                                     ')

    print('####################################################################################')

    for i in range(no):

        plt.plot(range(5,121),f1_lower_hist(dfs[i]), label = names[i])

    plt.xlabel('pT (in GeV) -->')

    plt.ylabel('F1 (for class pT < x) -->')

    plt.legend()

    plt.show()

    

    

    def pT_classes(a):

        if a<=10:

            return '0-10 GeV'

        if a>10 and a<=30:

            return '10-30 GeV'

        if a>30 and a<=100:

            return '30-100 GeV'

        if a>100:

            return '100-inf GeV'



    print()

    print()

    print()

    

    for i in range(no):

        print('####################################################################################')

        print('                      Sklearn Classification Report - ',names[i])

        print('####################################################################################')

        print(classification_report(dfs[i]['True_pT'].apply(pT_classes), dfs[i]['Predicted_pT'].apply(pT_classes)))

        print()

        print()

        print()

        

    for i in range(no):

        print('####################################################################################')

        print('                              Confusion Matrix - ', names[i])

        print('####################################################################################')

        x = pd.DataFrame(confusion_matrix(dfs[i]['True_pT'].apply(pT_classes),  dfs[i]['Predicted_pT'].apply(pT_classes)))

        x.columns = ['pred | '+i for i in ['0-10','10-30','30-100','100-inf']]

        x.index = ['true | '+ i for i in ['0-10','10-30','30-100','100-inf']]

        print(x)

        print()

        print()

        print()
sorted(os.listdir('../input'))
df = pd.read_csv('../input/pt-class-focal-loss-displ-data/OOF_preds.csv')

generate_classification_report(df)
df = pd.read_csv('../input/cnn-pt-class-focal-loss-displ-data/OOF_preds.csv')

generate_classification_report(df)
df0 = Frame1('fcnn-cms-1-displ-data')

df1 = Frame1('1-pt-regression-swiss-activation-displ-data')

df2 = Frame1('1-pt-regression-multi1-25-class-displ-data')

df3 = Frame1('cnn-1-pt-regression-displ-data')

df4 = Frame1('cnn-1-pt-regression-swish-activation-displ')

df5 = Frame1('cnn-1-pt-regression-multi1-25-class-displ')

compare_regression([df0, df1, df2, df3, df4, df5], ['FCNN-relu','FCNN-swish', 'FCNN-multitask','CNN-relu','CNN-swish', 'CNN-multitask'], 6)
df0 = Frame2('pt-regression-new-loss-displ-data')

df1 = Frame2('pt-regression-swiss-activation-displ-data')

df2 = Frame2('pt-regression-multi1-displ-data')

df3 = Frame2('cnn-pt-regression-new-loss-displ-data')

df4 = Frame2('cnn-pt-regression-swiss-activation-displ')

df5 = Frame2('cnn-pt-regression-multi1-displ-data')

compare_regression([df0, df1, df2, df3, df4, df5], ['FCNN-relu','FCNN-swish', 'FCNN-multitask', 'CNN-relu','CNN-swish', 'CNN-multitask'], 6)
df2 = Frame1('1-pt-regression-multi1-25-class-displ-data')

df1 = Frame1('1-pt-regression-swiss-activation-displ-data')

df0 = Frame1('fcnn-cms-1-displ-data')

df3 = Frame2('pt-regression-new-loss-displ-data')

df4 = Frame2('pt-regression-swiss-activation-displ-data')

df5 = Frame2('pt-regression-multi1-displ-data')

compare_regression([df0, df1, df2, df3, df4, df5], ['1/pT-FCNN-relu','1/pT-FCNN-swish', '1/pT-FCNN-multitask', 'pT-FCNN-relu','pT-FCNN-swish', 'pT-FCNN-multitask'], 6)
df0 = Frame1('cnn-1-pt-regression-displ-data')

df1 = Frame1('cnn-1-pt-regression-swish-activation-displ')

df2 = Frame1('cnn-1-pt-regression-multi1-25-class-displ')

df3 = Frame2('cnn-pt-regression-new-loss-displ-data')

df4 = Frame2('cnn-pt-regression-swiss-activation-displ')

df5 = Frame2('cnn-pt-regression-multi1-displ-data')

compare_regression([df0, df1, df2, df3, df4, df5], ['1/pT-CNN-relu','1/pT-CNN-swish', '1/pT-CNN-multitask', 'pT-CNN-relu','pT-CNN-swish', 'pT-CNN-multitask'], 6)