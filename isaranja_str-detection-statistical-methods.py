# Loading required libraries



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import matplotlib.font_manager
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(true_y,pred_y):

    # confusion matrix

    LABELS = ["Normal","Fraud"]

    conf_matrix = confusion_matrix(true_y, pred_y)



    plt.figure(figsize=(5, 4))

    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

    plt.title("Confusion matrix")

    plt.ylabel('True class')

    plt.xlabel('Predicted class')

    plt.show()
# loading the log file

df = pd.read_csv('../input/paysim1/PS_20174392719_1491204439457_log.csv')

df_bkp = df.copy()

df.head()
# statistical analysis

df[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']].describe()



#df.shape

#df.nameDest.nunique()



#total    = 6362620

#step     = 743

#type     = 5

#nameOrig = 6353307

#nameDest = 2722362
# Preprocesing 

df = df_bkp.copy()

# numerical columns

f1 = ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']



# log transformation

#df.loc[:,f1] =  np.log(df.loc[:,f1]+1)



# scaling

#from sklearn.preprocessing import MinMaxScaler # normalization

#sc = MinMaxScaler()



#df.loc[:,f1] = sc.fit_transform(df.loc[:,f1])



# one-hot encoding type variable

df = pd.get_dummies(df,columns=['type'],drop_first=True)



df.head()
# train/val/test split

train_x = df.loc[0:100000,['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER']].values

train_y = df.loc[0:100000:,['isFraud']].values
from sklearn.neighbors import LocalOutlierFactor



clf = LocalOutlierFactor(n_neighbors=10, algorithm = 'auto', contamination=0.001)

# predict returns 1 for an inlier and -1 for an outlier

pred_y = clf.fit_predict(train_x)

pred_y[pred_y>0]=0

pred_y[pred_y<0]=1



plot_confusion_matrix(train_y,pred_y)
from sklearn.ensemble import IsolationForest



clf = IsolationForest(n_estimators=10, max_samples = 1000, contamination= 0.001, behaviour='new')

clf.fit(train_x)  # fit 100 trees

pred_y=clf.predict(train_x)

pred_y[pred_y>0] = 0

pred_y[pred_y<0] = 1



plot_confusion_matrix(train_y,pred_y)
from sklearn.covariance import EllipticEnvelope



cov = EllipticEnvelope(random_state=0,contamination=0.001, support_fraction=1).fit(train_x)

# predict returns 1 for an inlier and -1 for an outlier

pred_y = cov.predict(train_x)

pred_y[pred_y>0]=0

pred_y[pred_y<0]=1

plot_confusion_matrix(train_y,pred_y)