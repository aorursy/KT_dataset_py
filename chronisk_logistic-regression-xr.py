import numpy as np

import pandas as pd

import matplotlib.pyplot as plt # plotting

from sklearn.neural_network import MLPClassifier 

from sklearn.linear_model import LogisticRegression

#from mpl_toolkits.mplot3d import Axes3D

#from sklearn.preprocessing import StandardScaler

#from sklearn.model_selection import train_test_split





data = pd.read_csv('../input/bank.csv',sep=',',header='infer')

data = data.drop(['day','poutcome','contact'],axis=1)



def binaryType_(data):

    

    data.y.replace(('yes', 'no'), (1, 0), inplace=True)

    data.default.replace(('yes','no'),(1,0),inplace=True)

    data.housing.replace(('yes','no'),(1,0),inplace=True)

    data.loan.replace(('yes','no'),(1,0),inplace=True)

    data.marital.replace(('married','single','divorced'),(1,2,3),inplace=True)

    #data.contact.replace(('telephone','cellular','unknown'),(1,2,3),inplace=True)

    #data.putcome.replace(('other','failure','success','unknown'),(1,2,3,4),inplace=True)

    data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)

    data.education.replace(('primary','secondary','tertiary','unknown'),(1,2,3,4),inplace=True)

    data.job.replace(('technician','services','retired','blue-collar','entrepreneur','admin.','housemaid','student','self-employed','management','unemployed','unknown'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True )

    return data



data = binaryType_(data)

data.head()
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
import matplotlib.pyplot as plt # plotting



# === Γραφική αναπαράσταση των χαρακτηριστικών.



plt.style.use('seaborn-whitegrid')



# να σβήσω και το in[2]

#plotPerColumnDistribution(data, 15, 5) 

#plotPerColumnDistribution(data, 20, 5)

#plotPerColumnDistribution(data, 10, 10)



data.hist(bins=20, figsize=(14,10), color='#E14906')

plt.show()



plt.hist((data.y),bins=20)

plt.show()



plt.hist((data.duration),bins=100)

plt.show()



plt.hist((data.age),bins=10) 

plt.show()



plt.hist((data.balance),bins=10) 

plt.show()



# Μετρητής των y σε 0 και 1

data['y'].value_counts()

import matplotlib.pyplot as plt



from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import classification_report



def evaluate_classifier_performance(df_train_class, predicted_train, predicted_prob_train, df_test_class, predicted_test, predicted_prob_test, roc_y_n):

    ### Confusion Matrix

    confusion_matrix_train = confusion_matrix(df_train_class, predicted_train)

    confusion_matrix_test = confusion_matrix(df_test_class, predicted_test)

    print ("\nTraining Confusion Matrix:\n ", confusion_matrix_train)

    print ("\nTesting Confusion Matrix:\n ", confusion_matrix_test)

    

    # Testing Confusion Matrix graph

    import pylab as pl

    cm = confusion_matrix(y_test, predicted_test)

    pl.matshow(cm)

    pl.title('Testing Confusion matrix of the classifier')

    pl.colorbar()

    pl.show()

    

    ### Accuracy score

    score_train = accuracy_score(df_train_class, predicted_train)

    score_test = accuracy_score(df_test_class, predicted_test)

    print ("\nTraining Accuracy Score: ", score_train)

    print ("\nTesting Accuracy Score: ", score_test)

       

    ### Precision, Recall  

    precision_train = precision_score(df_train_class, predicted_train)

    precision_test = precision_score(df_test_class, predicted_test)

    print ("\nTraining Precision: ", precision_train)

    print ("\nTesting Precision: ", precision_test)

    

    recall_train = recall_score(df_train_class, predicted_train)

    recall_test = recall_score(df_test_class, predicted_test)

    print ("\nTraining Recall: ", recall_train)

    print ("\nTesting Recall: ", recall_test)

    

    ### Classification Report

    print ("\nTrain Classification Report: \n",classification_report(df_train_class, predicted_train))

    print ("\nTest Classification Report: \n",classification_report(df_test_class, predicted_test))



    ### F1 Score

    f1score_train = f1_score(df_train_class, predicted_train)#, average='weighted')

    f1score_test = f1_score(df_test_class, predicted_test)#, average='weighted')

    print ("\nTraining F1score: ", f1score_train)

    print ("\nTesting F1score: ", f1score_test)

    

    f1score_train = f1_score(df_train_class, predicted_train, average='weighted')

    f1score_test = f1_score(df_test_class, predicted_test, average='weighted')

    print ("\nTraining Weigted F1score: ", f1score_train)

    print ("\nTesting Weighted F1score: ", f1score_test)

    

    ### ROC-AUC

    if roc_y_n == 'y':

        fpr, tpr, threshold = roc_curve(df_train_class, predicted_prob_train[:,1])

        roc_auc_train = auc(fpr, tpr)

        print ("\nTraining AUC for ROC: ",roc_auc_train)

        plt.figure()

        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_train)

        plt.plot([0, 1], [0, 1],'r--')

        plt.xlim([0, 1])

        plt.ylim([0, 1])

        plt.ylabel('True Positive Rate')

        plt.xlabel('False Positive Rate')

        plt.legend(loc = 'lower right')

        plt.title('Training - Receiver Operating Characteristic')

        

        fpr, tpr, threshold = roc_curve(df_test_class, predicted_prob_test[:,1])

        roc_auc_test = auc(fpr, tpr)

        print ("\nTesting AUC for ROC: ",roc_auc_test)

        plt.figure()

        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_test)

        plt.plot([0, 1], [0, 1],'r--')

        plt.xlim([0, 1])

        plt.ylim([0, 1])

        plt.ylabel('True Positive Rate')

        plt.xlabel('False Positive Rate')

        plt.legend(loc = 'lower right')

        plt.title('Testing - Receiver Operating Characteristic')

        

        return(df_train_class, predicted_train, predicted_prob_train, df_test_class, predicted_test, predicted_prob_test, roc_y_n);
import matplotlib.pyplot as plt 

import seaborn as sns 

import pandas as pd 

import numpy as np

import time



from sklearn.neural_network import MLPClassifier 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score



# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(data, data.y, test_size=0.2)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)



X_train['y'].value_counts()

# ============================

#X_train.info()



df_train = X_train

df_test = X_test



df_train_class = pd.DataFrame(df_train['y'])    

df_train_features = df_train.loc[:, df_train.columns != 'y']



df_test_class = pd.DataFrame(df_test['y'])

df_test_features = df_test.loc[:, df_test.columns != 'y']





print("  ### Logistic Regression  ")

t_start = time.clock()

LR = LogisticRegression()

LR.fit(df_train_features, df_train_class)



log_scores = cross_val_score(LR, X_train, y_train, cv=3)

log_reg_mean = log_scores.mean()

print("Crossval Mean Scores :" , log_reg_mean)



predicted_train = LR.predict(df_train_features)

predicted_test = LR.predict(df_test_features)



predicted_prob_train = LR.predict_proba(df_train_features)

predicted_prob_test = LR.predict_proba(df_test_features)



evaluate_classifier_performance(df_train_class, predicted_train, predicted_prob_train, df_test_class, predicted_test, predicted_prob_test, 'y')



print("Crossval Mean Scores :" , log_reg_mean)



t_end = time.clock()

t_diff = t_end - t_start

print("Trained in {f:.2f} sec".format(f=t_diff))



print("  ###  Τέλος Logistic Regression  ")