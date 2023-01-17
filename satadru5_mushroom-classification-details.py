# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/mushrooms.csv')
df.head(2)
df['class'].unique()
df.columns
sns.countplot(x='class',data=df)
df.groupby('class')['class'].count()
#hot encodding

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

for col in df.columns:

    df[col]=encoder.fit_transform(df[col])

    



df.head(3)    

sns.pairplot(df)
sns.boxplot(x='class',y='cap-shape',data=df)
sns.countplot(x='cap-shape',data=df)
X = df.iloc[:,1:23]  # all rows, all the features and no labels

y = df.iloc[:, 0]  # all rows, label only

X.head(3)

y.head()

#CHECK CO-RELATION

corr=df.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')
#Scale the data to normalize

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X=scaler.fit_transform(X)
#Principle component analysis

from sklearn.decomposition import PCA

pca = PCA()

pa=pca.fit_transform(X)

pa
covariance=pca.get_covariance()

explained_variance=pca.explained_variance_

explained_variance
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))

    

    plt.bar(range(22), explained_variance, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
N=df.values

pca = PCA(n_components=2)

x = pca.fit_transform(N)

plt.figure(figsize = (5,5))

plt.scatter(x[:,0],x[:,1])

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=5)

X_clustered = kmeans.fit_predict(N)



LABEL_COLOR_MAP = {0 : 'g',

                   1 : 'y'

                   

                  }



label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

plt.figure(figsize = (5,5))

plt.scatter(x[:,0],x[:,1], c= label_color)

plt.show()
#Splitting the data into training and testing dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#Logictic Regrassion

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics



model_LR= LogisticRegression()

model_LR.fit(X_train,y_train)

dt_score_train = model_LR.score(X_train, y_train)

print("Training score: ",dt_score_train)

dt_score_test = model_LR.score(X_test, y_test)

print("Testing score: ",dt_score_test)

y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

model_LR.score(X_test, y_pred)
#Logictic Regrassion

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics



model_LR= LogisticRegression()

model_LR.fit(X_train,y_train)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
#Logistic Regression(Tuned model)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics



LR_model= LogisticRegression()



tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,

              'penalty':['l1','l2']

                   }
from sklearn.model_selection import GridSearchCV



LR= GridSearchCV(LR_model, tuned_parameters,cv=10)
LR.fit(X_train,y_train)
print(LR.best_params_)
y_prob = LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

LR.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
#Trying default Random model

from sklearn.ensemble import RandomForestClassifier



model_RR=RandomForestClassifier()



#tuned_parameters = {'min_samples_leaf': range(5,10,5), 'n_estimators' : range(50,200,50),

                    #'max_depth': range(5,15,5), 'max_features':range(5,20,5)

                    #}

               

model_RR.fit(X_train,y_train)
dt_score_train = model_RR.score(X_train, y_train)

print("Training score: ",dt_score_train)

dt_score_test = model_RR.score(X_test, y_test)

print("Testing score: ",dt_score_test)

y_prob = model_RR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

model_RR.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
#Random forest is giving best performance

from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
#Plot data in higher dimention



import pandas as pd

import hypertools as hyp 

%matplotlib inline
data = pd.read_csv('../input/mushrooms.csv')

data.head()
class_labels = data.pop('class')
hyp.plot(data,'o') # if the number of features is greater than 3, the default is to plot in 3d
hyp.plot(data,'o', group=class_labels, legend=list(set(class_labels)))
hyp.plot(data, 'o', n_clusters=23)



# you can also recover the cluster labels using the cluster tool

cluster_labels = hyp.tools.cluster(data, n_clusters=23, ndims=3) 

# hyp.plot(data, 'o', point_colors=cluster_labels, ndims=2)
hyp.plot(data,'o', group=cluster_labels, palette="deep")
#Independent Components Analysis

from sklearn.decomposition import FastICA

ICA_model = FastICA(n_components=3)

reduced_data_ICA = ICA_model.fit_transform(hyp.tools.df2mat(data))

hyp.plot(reduced_data_ICA,'o', group=class_labels, legend=list(set(class_labels)))
#t-SNE

from sklearn.manifold import TSNE

TSNE_model = TSNE(n_components=3)

reduced_data_TSNE = TSNE_model.fit_transform(hyp.tools.df2mat(data))

hyp.plot(reduced_data_TSNE,'o', group=class_labels, legend=list(set(class_labels)))