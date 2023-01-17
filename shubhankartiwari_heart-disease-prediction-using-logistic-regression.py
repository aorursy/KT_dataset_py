import pandas as pd

import numpy as np

import scipy.stats as st

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

import matplotlib.mlab as mlab
df = pd.read_csv("../input/framingham.csv")

df.drop(['education'],axis = 1,inplace = True)

df.head()
df.rename(columns = {'male' : 'Sex_male'},inplace = True)
df.isnull().sum()
count = 0

for i in df.isnull().sum(axis = 1):

    count = count + 1

print("Total number of missing values:",count)
df.dropna(axis = 0,inplace = True)
def draw_histograms(dataframe,features,rows,cols):

    fig = plt.figure(figsize = (20,20))

    for i, feature in enumerate(features):

        a = fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins = 20,ax=a,facecolor = 'green')

        a.set_title(feature + "Distribution",color = 'blue')

    fig.tight_layout()

    plt.show()

draw_histograms(df,df.columns,6,3)
df.TenYearCHD.value_counts()
sns.countplot(x = 'TenYearCHD',data = df)
sns.pairplot(df)
df.describe()
from statsmodels.tools import add_constant

df_constant = add_constant(df)

df_constant.head()
st.chisqprob = lambda chisq,df: st.chi2.sf(chisq,df)

cols = df_constant.columns[:-1]

model = sm.Logit(df.TenYearCHD,df_constant[cols])

r = model.fit()

r.summary()
p = np.exp(r.params)

conf = np.exp(r.conf_int())

conf['OR'] = p

pv = round(r.pvalues,3)

conf['pvalue'] = pv

conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']

print((conf))
import sklearn
new_frs = df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]

x = new_frs.iloc[:,:-1]

y = new_frs.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .20,random_state = 5)
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()

lgr.fit(x_test,y_test)

y_pred = lgr.predict(x_test)
sklearn.metrics.accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix,annot = True,fmt = 'd',cmap = 'YlGnBu')
TN=cm[0,0]

TP=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print('The accuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',



'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',



'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',



'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',



'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',



'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',



'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',



'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)
y_pred_prob=lgr.predict_proba(x_test)[:,:]

y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['No Heart Disease (0)','Heart Disease (1)'])

y_pred_prob_df.head()
from sklearn.preprocessing import binarize
for i in range(1,5):

    cm2=0

    y_pred_prob_yes=lgr.predict_proba(x_test)

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