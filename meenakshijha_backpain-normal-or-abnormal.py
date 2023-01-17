# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import os

# os.chdir("C:\\Users\\DILIP\\Downloads")



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report 

from sklearn.pipeline import make_pipeline

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import statsmodels.api as sm

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/lower-back-pain-symptoms-dataset/Dataset_spine.csv')

df.head()
df.drop('Unnamed: 13',axis=1,inplace=True)
df.columns = ['pelvic_incidence','pelvic tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis','pelvic_slope','Direct_tilt','thoracic_slope','cervical_tilt','sacrum_angle','scoliosis_slope','Status']
df.head()
df.info()
columns=df.columns

for i in columns:

    print(i,'has : ',df[i].nunique(),'Unique Values')
print(df.Status.value_counts())

print(df.Status.describe())
df.isnull().sum()
df.describe().T
sns.set(style="ticks", color_codes=True)

sns.pairplot(df)

plt.show()
df.groupby('Status').mean()
df.groupby('Status').median()
plt.figure(figsize=(10,7))

sns.distplot(df['degree_spondylolisthesis'],bins=100, kde=True, color="blue")
df.skew()
df.kurt()
df["deg1"]=(df["degree_spondylolisthesis"])**2
sns.set(style="darkgrid")

fig,ax=plt.subplots(1,2,figsize=(15,8))

sns.kdeplot(df['degree_spondylolisthesis'],shade=True, color="black",ax=ax[0])

ax[0].set_title('degree_spondylolisthesis')

sns.kdeplot(df['deg1'],shade=True, color="black",ax=ax[1])

ax[1].set_title('degree_spondylolisthesis after applying Square transformation')

plt.show()

df.drop('deg1',axis=1,inplace=True)
correlation=df.corr()

correlation
plt.figure(figsize=(12,10))

sns.heatmap(df.corr(),annot=True,cmap='seismic_r')
plt.figure(figsize=(12,10))

mask = np.zeros_like(correlation, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(correlation, mask=mask, center=0, square=True, linewidths=.5,cmap='gist_ncar')



plt.show()
sns.set(style="darkgrid")

fig,ax=plt.subplots(2,3,figsize=(15,8))

sns.swarmplot("Status","pelvic_incidence",data=df,ax=ax[0,0])

sns.swarmplot("Status","pelvic tilt",data=df,ax=ax[0,1])

sns.swarmplot("Status","sacral_slope",data=df,ax=ax[0,2])

sns.swarmplot("Status","degree_spondylolisthesis",data=df,ax=ax[1,0])

sns.swarmplot("Status","lumbar_lordosis_angle",data=df,ax=ax[1,1])

sns.swarmplot("Status","pelvic_radius",data=df,ax=ax[1,2])

plt.show()
def draw_histograms(dataframe, features, rows, cols):

    fig=plt.figure(figsize=(20,20))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins=20,ax=ax,facecolor='k')

        ax.set_title(feature+" Distribution",color='DarkRed')

        

    fig.tight_layout()  

    plt.show()

draw_histograms(df,df.columns,6,3)
fig, axes = plt.subplots(3, 4, figsize = (15,20))

axes = axes.flatten()

for i in range(0,len(df.columns)-1):

   

    sns.kdeplot(df.iloc[:,i], ax=axes[i],shade=True, color="r")



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(3, 4, figsize = (15,20))

axes = axes.flatten()



for i in range(0,len(df.columns)-1):

    sns.boxenplot(x="Status", y=df.iloc[:,i], data=df, orient='v', ax=axes[i],palette="magma_r")



plt.tight_layout()

plt.show()
plt.figure(figsize=(8,5))

sns.countplot('Status',data=df,palette="viridis_r")

plt.show()

df['Status'].value_counts()
fig, ax = plt.subplots(figsize=(8,5))



sns.swarmplot(x ='Status', y='degree_spondylolisthesis',ax=ax,size=4,data=df, palette="PuRd") 

sns.boxplot(x ='Status', y='degree_spondylolisthesis',ax=ax,data=df)
import plotly

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

col = "Status"

grouped = df[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})



## plot

colors = ['magenta', 'purplr']

trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0],marker=dict(colors=colors, line=dict(color='#000000', width=2)))

layout = {'title': 'Status of Spine(Normal, Abnormal)'}

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
df["Status"]=pd.get_dummies(df["Status"],drop_first=True)
from statsmodels.tools import add_constant as add_constant

df_constant = add_constant(df)

df_constant.head()
cols=df_constant.columns[:-1]

model=sm.Logit(df.Status,df_constant[cols])

result=model.fit()

result.summary()
def back_feature_elem (data_frame,dep_var,col_list):





    while len(col_list)>0 :

        model=sm.Logit(dep_var,data_frame[col_list])

        result=model.fit(disp=0)

        largest_pvalue=round(result.pvalues,3).nlargest(1)

        if largest_pvalue[0]<(0.05):

            return result

            break

        else:

            col_list=col_list.drop(largest_pvalue.index)



result=back_feature_elem(df_constant,df.Status,cols)
result.summary()
params = np.exp(result.params)

conf = np.exp(result.conf_int())

conf['OR'] = params

pvalue=round(result.pvalues,3)

conf['pvalue']=pvalue

conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']

print ((conf))

import sklearn

new_features=df[['pelvic tilt','sacral_slope','pelvic_radius','degree_spondylolisthesis',"Status"]]

x=new_features.iloc[:,:-1]

y=new_features.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
sklearn.metrics.accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap='gist_ncar')
TN=cm[0,0]

TP=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print('The acuuracy of the model = TP+TN / (TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n\n',



'The Miss-classification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n\n',



'Sensitivity or True Positive Rate = TP / (TP+FN) = ',TP/float(TP+FN),'\n\n',



'Specificity or True Negative Rate = TN / (TN+FP) = ',TN/float(TN+FP),'\n\n',



'Positive Predictive value = TP / (TP+FP) = ',TP/float(TP+FP),'\n\n',



'Negative predictive Value = TN / (TN+FN) = ',TN/float(TN+FN),'\n\n',



'Positive Likelihood Ratio = Sensitivity / (1-Specificity) = ',sensitivity/(1-specificity),'\n\n',



'Negative likelihood Ratio = (1-Sensitivity) / Specificity = ',(1-sensitivity)/specificity)

y_pred_prob=logreg.predict_proba(x_test)[:,:]

y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of abnormal spine (0)','Prb of normal spine (1)'])

y_pred_prob_df.head()
from sklearn.preprocessing import binarize

for i in range(1,5):

    cm2=0

    y_pred_prob_yes=logreg.predict_proba(x_test)

    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]

    cm2=confusion_matrix(y_test,y_pred2)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

    
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])

plt.plot(fpr,tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for Heart disease classifier')

plt.xlabel('False positive rate (1-Specificity)')

plt.ylabel('True positive rate (Sensitivity)')

plt.grid(True)
sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])
print('Abnormal', round(df['Status'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Normal', round(df['Status'].value_counts()[1]/len(df) * 100,2), '% of the dataset')