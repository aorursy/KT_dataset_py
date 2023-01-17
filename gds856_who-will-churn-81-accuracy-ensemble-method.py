import pandas as pd

import numpy as np

import pandas_profiling



import matplotlib.pyplot as plt#visualization

from PIL import  Image

%matplotlib inline

import pandas as pd

import seaborn as sns#visualization

import itertools

import warnings

warnings.filterwarnings("ignore")

import io

import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization





pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 500)

pd.set_option('float_format', '{:f}'.format)

pd.options.display.float_format = '{:.4f}'.format









from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE, BorderlineSMOTE

from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve



from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

from collections import Counter

from imblearn.combine import SMOTETomek, SMOTEENN

from imblearn.over_sampling import BorderlineSMOTE

from IPython.display import Markdown, display



def printmd(string, color=None):

    colorstr = "<span style='color:{}'>{}</span>".format(color, string)

    display(Markdown(colorstr))
printmd("**bold and blue**", color="blue")

print('dkjvkjfs')

printmd('dkjvkjfs')
df=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
pandas_profiling.ProfileReport(df)
from IPython.display import display, HTML, display_html

display(df.head())

display(df.sample(5))

display(df.shape)

display(df.dtypes)

display(df.nunique())

display(df.describe())

display(df.describe(include = 'O'))

display(df.isna().sum())
# Data to plot

labels =df['Churn'].value_counts().index

sizes = df['Churn'].value_counts()



plt.pie(sizes, explode=(0.1,0), labels=labels, colors=["whitesmoke","red"], autopct='%1.1f%%', shadow=True, startangle=270,)



plt.title('Percent of churn in customer')

plt.show()
df['gender'] = df['gender'].map( {'Female': 0, 'Male': 1} ).astype(int)                 #

df['Partner'] = df['Partner'].map( {'No': 0, 'Yes': 1} ).astype(int)                    #

df['Dependents'] = df['Dependents'].map( {'No': 0, 'Yes': 1} ).astype(int)              #

df['PhoneService'] = df['PhoneService'].map( {'No': 0, 'Yes': 1} ).astype(int)          #

df['OnlineSecurity'] = df['OnlineSecurity'].map( {'No': 0, 'Yes': 1, 'No internet service': -1} ).astype(int)  #

df['Churn'] = df['Churn'].map( {'No': 0, 'Yes': 1} ).astype(int)                        #

df['OnlineBackup'] = df['OnlineBackup'].map( {'No': 0, 'Yes': 1, 'No internet service': -1} ).astype(int)   #

df['DeviceProtection'] = df['DeviceProtection'].map( {'No': 0, 'Yes': 1, 'No internet service': -1} ).astype(int)

df['TechSupport'] = df['TechSupport'].map( {'No': 0, 'Yes': 1, 'No internet service': -1} ).astype(int)

df['StreamingTV'] = df['StreamingTV'].map( {'No': 0, 'Yes': 1, 'No internet service': -1} ).astype(int)  #

df['StreamingMovies'] = df['StreamingMovies'].map( {'No': 0, 'Yes': 1, 'No internet service': -1} ).astype(int)

df['PaperlessBilling'] = df['PaperlessBilling'].map( {'No': 0, 'Yes': 1} ).astype(int)   #

df['MultipleLines'] = df['MultipleLines'].map( {'No': 0, 'Yes': 1, 'No phone service': -1} ).astype(int)

df['InternetService'] = df['InternetService'].map( {'DSL': 0, 'Fiber optic': 1, 'No': -1} ).astype(int)

df.PaymentMethod.value_counts()
df.dtypes
# deleting emply values in Total charge column

df = df.drop(df[df.TotalCharges == ' '].index)
df.TotalCharges = df.TotalCharges.astype('float64')
churn= df[df.Churn == 1]

nochurn= df[df.Churn == 0]
def kdeplot(feature):

    plt.figure(figsize=(12, 6))

    sns.kdeplot(churn[feature], color= 'navy', label= 'Churn: Yes')

    sns.kdeplot(nochurn[feature], color= 'orange', label= 'Churn: No')
kdeplot('MonthlyCharges')

kdeplot('TotalCharges')

kdeplot('tenure')
plt.subplots(figsize=(20,15))

plt.subplot(321)

sns.distplot(churn.MonthlyCharges, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'})

plt.title('churn.MonthlyCharges')



plt.subplot(322)

sns.distplot(nochurn.MonthlyCharges, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'});

plt.title('NO churn.MonthlyCharges')



plt.subplot(323)

sns.distplot(churn.TotalCharges, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'})

plt.title('churn Total charge')



plt.subplot(324)

sns.distplot(nochurn.TotalCharges, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'});

plt.title('NO churn Total charge')



plt.subplot(325)

sns.distplot(churn.tenure, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'})

plt.title('churn tenure')



plt.subplot(326)

sns.distplot(nochurn.tenure, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'});

plt.title('NO churn tenure')

plt.tight_layout()



plt.show()
df.head()
plt.subplots(figsize=(10,25))

plt.subplot(921)

plt.pie(churn.gender.value_counts(), labels=churn.gender.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('Churn Gender')



plt.subplot(922)

plt.pie(nochurn.gender.value_counts(), labels=nochurn.gender.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('NO churn Gender')

#-----------------------------------------------------------------------------------------#

plt.subplot(923)

plt.pie(churn.SeniorCitizen.value_counts(), labels=churn.SeniorCitizen.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('Churn Senior Citizen')



plt.subplot(924)

plt.pie(nochurn.SeniorCitizen.value_counts(), labels=nochurn.SeniorCitizen.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('NO Churn Senior Citizen')



plt.subplot(925)

plt.pie(churn.Partner.value_counts(), labels=churn.Partner.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('churn Partner')



plt.subplot(926)

plt.pie(nochurn.Partner.value_counts(), labels=nochurn.Partner.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('NO churn Partner')

plt.tight_layout()



plt.subplot(927)

plt.pie(churn.Dependents.value_counts(), labels=churn.Dependents.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('churn Dependents')



plt.subplot(928)

plt.pie(nochurn.Dependents.value_counts(), labels=nochurn.Dependents.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('NO churn Dependents')

plt.tight_layout()



plt.subplot(929)

plt.pie(churn.PhoneService.value_counts(), labels=churn.PhoneService.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('churn PhoneService')



plt.subplot(9,2,10)

plt.pie(nochurn.PhoneService.value_counts(), labels=nochurn.PhoneService.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('NO churn PhoneService')

plt.tight_layout()



plt.subplot(9,2,11)

plt.pie(churn.MultipleLines.value_counts(), labels=churn.MultipleLines.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('churn MultipleLines')



plt.subplot(9,2,12)

plt.pie(nochurn.MultipleLines.value_counts(), labels=nochurn.MultipleLines.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('NO churn MultipleLines')

plt.tight_layout()



plt.subplot(9,2,13)

plt.pie(churn.InternetService.value_counts(), labels=churn.InternetService.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('churn InternetService')



plt.subplot(9,2,14)

plt.pie(nochurn.InternetService.value_counts(), labels=nochurn.InternetService.value_counts().index, autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

plt.title('NO churn InternetService')

plt.tight_layout()



# plt.subplot(325)

# plt.pie(churn.Dependents.value_counts(), labels=df.Churn.value_counts().index, autopct='%1.1f%%')

# p=plt.gcf()

# p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

# plt.title('churn tenure')



# plt.subplot(326)

# plt.pie(nochurn.Dependents.value_counts(), labels=df.Churn.value_counts().index, autopct='%1.1f%%')

# p=plt.gcf()

# p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

# plt.title('NO churn tenure')

# plt.tight_layout()



# plt.subplot(325)

# plt.pie(churn.Dependents.value_counts(), labels=df.Churn.value_counts().index, autopct='%1.1f%%')

# p=plt.gcf()

# p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

# plt.title('churn tenure')



# plt.subplot(326)

# plt.pie(nochurn.Dependents.value_counts(), labels=df.Churn.value_counts().index, autopct='%1.1f%%')

# p=plt.gcf()

# p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))

# plt.title('NO churn tenure')

# plt.tight_layout()

plt.figure(figsize=(15,10)) 

sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r')
df.groupby('tenure')['Churn'].mean()
plt.figure(figsize=(25,10))



sns.barplot(df.tenure, df.Churn)
sns.countplot(x="Churn", hue="Contract", data=df)
sns.factorplot(y="TotalCharges",x="Churn",data=df,kind="boxen", palette = "Pastel2")
df.drop('customerID', axis=1, inplace=True)
df=pd.concat([pd.get_dummies(df['Contract'], drop_first = True),df],axis=1).drop('Contract',axis=1)

df=pd.concat([pd.get_dummies(df['PaymentMethod'], drop_first = True),df],axis=1).drop('PaymentMethod',axis=1)
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import Perceptron



X = df.drop(['Churn'], axis=1)

Y = df.Churn



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=32, stratify= Y)



feature_cols = X.columns
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, random_state=22, max_depth= 10, class_weight={0:1,1:1})

classifier.fit(X_train, y_train)

x_predrf = classifier.predict(X_train)

y_predrf = classifier.predict(X_test)



print(f'Accuracy Test  : {accuracy_score(y_test, y_predrf):0.4f}; || Accuracy Train : {accuracy_score(y_train, x_predrf):0.4f}')

print(f'Precision Test : {precision_score(y_test, y_predrf):0.4f}; || Precision Train: {precision_score(y_train, x_predrf):0.4f}')

print(f'Recall Test    : {recall_score(y_test, y_predrf):0.4f}; || Recall Train   : {recall_score(y_train, x_predrf):0.4f}')

print(f'Cohen Kappa    : {cohen_kappa_score(y_test, y_predrf):0.4f}; || Avg Precesion  : {average_precision_score(y_train, x_predrf):0.4f}; || AUC:{roc_auc_score(y_test,y_predrf):0.4f}')

printmd("**CONFUSION MATRIX**", color="blue")

print(pd.crosstab(y_test, y_predrf, margins = True))

printmd("**Classification Report**", color="red")

print(classification_report(y_test,y_predrf ))

printmd("** Feature Importances**", color="green")



importances = classifier.feature_importances_

weights = pd.Series(importances, index=X.columns.values).sort_values(ascending=False)

plt.figure(figsize=(10,12))

plt.title("Feature importance")

ax = sns.barplot(y=weights.index, x=weights.values, palette="Blues_d", orient='h')
%%time

from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, precision_score, recall_score



LR = LogisticRegression('l2')

LR.fit(X_train, y_train)



x_predrf = LR.predict(X_train)

y_predrf = LR.predict(X_test)



print(f'Accuracy Test  : {accuracy_score(y_test, y_predrf):0.4f}; || Accuracy Train : {accuracy_score(y_train, x_predrf):0.4f}')

print(f'Precision Test : {precision_score(y_test, y_predrf):0.4f}; || Precision Train: {precision_score(y_train, x_predrf):0.4f}')

print(f'Recall Test    : {recall_score(y_test, y_predrf):0.4f}; || Recall Train   : {recall_score(y_train, x_predrf):0.4f}')

print(f'Cohen Kappa    : {cohen_kappa_score(y_test, y_predrf):0.4f}; || Avg Precesion  : {average_precision_score(y_train, x_predrf):0.4f}; || AUC:{roc_auc_score(y_test,y_predrf):0.4f}')

printmd("**CONFUSION MATRIX**", color="blue")

print(pd.crosstab(y_test, y_predrf, margins = True))

printmd("**Classification Report**", color="red")



print(classification_report(y_test,y_predrf ))
%%time



xgb = XGBClassifier()

xgb.fit(X_train, y_train)



x_predrf = xgb.predict(X_train)

y_predrf = xgb.predict(X_test)



print(f'Accuracy Test  : {accuracy_score(y_test, y_predrf):0.4f}; || Accuracy Train : {accuracy_score(y_train, x_predrf):0.4f}')

print(f'Precision Test : {precision_score(y_test, y_predrf):0.4f}; || Precision Train: {precision_score(y_train, x_predrf):0.4f}')

print(f'Recall Test    : {recall_score(y_test, y_predrf):0.4f}; || Recall Train   : {recall_score(y_train, x_predrf):0.4f}')

print(f'Cohen Kappa    : {cohen_kappa_score(y_test, y_predrf):0.4f}; || Avg Precesion  : {average_precision_score(y_train, x_predrf):0.4f}; || AUC:{roc_auc_score(y_test,y_predrf):0.4f}')

printmd("**CONFUSION MATRIX**", color="blue")

print(pd.crosstab(y_test, y_predrf, margins = True))

printmd("**Classification Report**", color="red")



print(classification_report(y_test,y_predrf ))
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.tree import ExtraTreeClassifier

from catboost import CatBoostClassifier

from sklearn.ensemble import VotingClassifier

clf1 = RandomForestClassifier()

clf2 = LogisticRegression()

clf3 = XGBClassifier()

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

eclf1.fit(X_train, y_train)

x_predrf = eclf1.predict(X_train)

y_predrf = eclf1.predict(X_test)



print(f'Accuracy Test  : {accuracy_score(y_test, y_predrf):0.4f}; || Accuracy Train : {accuracy_score(y_train, x_predrf):0.4f}')

print(f'Precision Test : {precision_score(y_test, y_predrf):0.4f}; || Precision Train: {precision_score(y_train, x_predrf):0.4f}')

print(f'Recall Test    : {recall_score(y_test, y_predrf):0.4f}; || Recall Train   : {recall_score(y_train, x_predrf):0.4f}')

print(f'Cohen Kappa    : {cohen_kappa_score(y_test, y_predrf):0.4f}; || Avg Precesion  : {average_precision_score(y_train, x_predrf):0.4f}; || AUC:{roc_auc_score(y_test,y_predrf):0.4f}')

printmd("**CONFUSION MATRIX**", color="blue")

print(pd.crosstab(y_test, y_predrf, margins = True))

printmd("**Classification Report**", color="red")



print(classification_report(y_test,y_predrf ))