import pandas as pd

import numpy as np

import statsmodels.api as sm

import seaborn as sn

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab
diabetes_df=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

diabetes_df.head(10)
diabetes_df.describe()
#Checking for Null Values in each Feature

diabetes_df.isnull().sum()
#Drawing a histogram of each feature

def draw_histograms(dataframe, features, rows, cols):

    fig=plt.figure(figsize=(20,20))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins=20,ax=ax,facecolor='deepskyblue')

        ax.set_title(feature+" Distribution",color = 'black')

        

    fig.tight_layout()

    plt.show()



draw_histograms(diabetes_df, diabetes_df.columns,4,3)
#Checking the outcome counts

diabetes_df.Outcome.value_counts()
sn.countplot(x='Outcome', data=diabetes_df)
#Looking for correlation between the different features

sn.pairplot(data=diabetes_df)
plt.figure(figsize=(12,10))

sn.heatmap(diabetes_df.corr(), annot=True,cmap ='coolwarm', vmax=.6)
import statsmodels.api as sm



from statsmodels.stats.outliers_influence import variance_inflation_factor



# Get variables for which to compute VIF and add intercept term

X = diabetes_df[['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

X['Intercept'] = 1



# Compute and view VIF

vif = pd.DataFrame()

vif["variables"] = X.columns

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



# View results using print

print(vif[0:-1])
#adding a constant value to the dataframe. This will not influence the accuracy of our model.

from statsmodels.tools import add_constant as add_constant

diabetes_df_constant = add_constant(diabetes_df)
#running a logistic regression model in order to look at the p values of each feature

import scipy.stats as st

st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)

cols=diabetes_df_constant.columns[:-1]

model = sm.Logit(diabetes_df.Outcome,diabetes_df_constant[cols])

result = model.fit()

result.summary()
#dropping the SkinThickness feature

diabetes_df_drop = diabetes_df_constant.drop(['SkinThickness'], axis=1)
#Replacing the 0 values of the Glucose, Insulin, BMI, and BloodPressure features with their median values

median_glucose = diabetes_df_drop['Glucose'].median(skipna=True)

median_Insulin = diabetes_df_drop['Insulin'].median(skipna=True)

median_BMI = diabetes_df_drop['BMI'].median(skipna=True)

median_bp = diabetes_df_drop['BloodPressure'].median(skipna=True)

diabetes_df_drop['Glucose']=diabetes_df_drop.Glucose.mask(diabetes_df_drop.Glucose == 0,median_glucose)

diabetes_df_drop['Insulin']=diabetes_df_drop.Insulin.mask(diabetes_df_drop.Insulin == 0,median_Insulin)

diabetes_df_drop['BMI']=diabetes_df_drop.BMI.mask(diabetes_df_drop.BMI == 0,median_BMI)

diabetes_df_drop['BloodPressure']=diabetes_df_drop.BloodPressure.mask(diabetes_df_drop.BloodPressure == 0,median_bp)
#Feature histograms after replacing missing data

draw_histograms(diabetes_df_drop, diabetes_df_drop.columns,4,3)
st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)

cols=diabetes_df_drop.columns[:-1]

model = sm.Logit(diabetes_df.Outcome,diabetes_df_drop[cols])

result = model.fit()

result.summary()
def back_feature_elem (data_frame,dep_var,col_list):

    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eleminating feature with the highest

    P-value above alpha one at a time and returns the regression summary with all p-values below alpha"""



    while len(col_list)>0 :

        model=sm.Logit(dep_var,data_frame[col_list])

        result=model.fit(disp=0)

        largest_pvalue=round(result.pvalues,3).nlargest(1)

        if largest_pvalue[0]<(0.05):

            return result

            break

        else:

            col_list=col_list.drop(largest_pvalue.index)
#using feature elimination function given above

result=back_feature_elem(diabetes_df_drop,diabetes_df_drop.Outcome,cols)
result.summary()
params = np.exp(result.params)

conf = np.exp(result.conf_int())

conf['OR'] = params

pvalue=round(result.pvalues,3)

conf['pvalue']=pvalue

conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']

print ((conf))
import sklearn

new_features = diabetes_df[['Pregnancies','Glucose','BMI','DiabetesPedigreeFunction','Outcome']]
#Creating the training and test sets

x=new_features.iloc[:,:-1]

y=new_features.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
#Running our machine learning model

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
#Our Accuracy score

sklearn.metrics.accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

hm = sn.heatmap(conf_matrix, annot=True,fmt='d',cmap='coolwarm',vmax=50, cbar=False)

hm.set_title("Model Confusion Matrix")

hm
TN=cm[0,0]

TP=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',



'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',



'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',



'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',



'Positive predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',



'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',



'Positive likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',



'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)
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

plt.title('ROC curve for Diabetes classifier')

plt.xlabel('False positive rate (1-Specificity)')

plt.ylabel('True positive rate (Sensitivity)')

plt.grid(True)
sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])