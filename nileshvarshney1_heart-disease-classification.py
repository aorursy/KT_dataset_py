import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats # for stats 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_df = pd.read_csv("../input/heart.csv")

data_df.sample(10)
def convert_to_objects(df , feature):

    return df[feature].astype('category')



for feature in ['sex','cp','fbs','restecg','exang','slope','ca','thal','target']:

    data_df[feature] = convert_to_objects(data_df, feature)

    

data_df.info()
sns.set_palette("Set1")

sns.set_style("whitegrid")

cat_features = ['sex','cp','fbs','restecg','exang','slope','ca','thal','target']

fig,ax = plt.subplots(3,3, figsize=(18,12))

for row in range(3):

    for col in range(3):

        feature_location = row * 3 + col

        # print(cat_features[feature_location])

        sns.countplot(data= data_df, x = cat_features[feature_location], ax = ax[row][col])

        ax[row][col].set_title(cat_features[feature_location].capitalize())

        ax[row][col].set(xlabel='',ylabel='Counts')

plt.suptitle('Categorical Features Distributions',color='b',fontsize = 20);
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

num_features = ['trestbps','chol','thalach','oldpeak']

def continuous_variables_plot(num_features):  

    fig,ax = plt.subplots(2,2, figsize=(18,12))

    for row in range(2):

        for col in range(2):

            feature_location = row * 2 + col

            # print(cat_features[feature_location])

            sns.distplot(data_df[num_features[feature_location]], ax = ax[row][col])

            ax[row][col].set_title(

                '{} (skew = {} kurtosis = {})'.format(

                    num_features[feature_location].capitalize(),

                    data_df[num_features[feature_location]].skew(),

                    data_df[num_features[feature_location]].kurtosis()

                ))

            ax[row][col].set(xlabel='')

            #ax[row][col].annotate('actual group', xy=(100,200), xytext=(100, 300))

    plt.suptitle('Continous Features Distributions',color='b',fontsize = 20);

    

continuous_variables_plot(num_features)
data_df['chol'] = data_df.apply(lambda row: 420 if (row['chol'] > 420) 

                                else row['chol'],axis = 1)

data_df['oldpeak'] = data_df.apply(lambda row: 0.01 if (row['oldpeak'] <= 0) 

                                   else row['oldpeak'],axis = 1)

data_df['trestbps'] = data_df.apply(lambda row: 165 if (row['trestbps'] > 165) 

                                   else row['trestbps'],axis = 1)



data_df['xt_thalach'], t_thalach = stats.boxcox(data_df['thalach'])

data_df['xt_chol'], t_chol = stats.boxcox(data_df['chol'])

data_df['xt_oldpeak'], t_oldpeak = stats.boxcox(data_df['oldpeak'])

data_df['xt_trestbps'], t_trestbps = stats.boxcox(data_df['trestbps'])

#data_df.drop(num_features,axis=1,inplace=True)
def skew_test(feature):

    stat, pvalue = stats.skewtest(data_df[feature])

    if pvalue > 0.05:

        print('{} - Not Skewed Feature  p-value -{:.4} skewness {:.4}'.format(feature,pvalue,stat))

    else:

        print('{} - Skewed feature : p-value{:.4} skewness - {:.4}'.format(feature,pvalue,stat))



skew_test('xt_thalach')

skew_test('xt_chol')

skew_test('xt_trestbps')

skew_test('xt_oldpeak')
data_df['age_bucket'] = pd.cut(data_df['age'],bins=5)

sns.countplot(data_df['age_bucket']);

# data_df['age'].hist();
num_features = ['xt_thalach','xt_chol','xt_trestbps','xt_oldpeak']

def continuous_variables_boxplot(num_features):  

    fig,ax = plt.subplots(2,2, figsize=(18,12))

    for row in range(2):

        for col in range(2):

            feature_location = row * 2 + col

            # print(cat_features[feature_location])

            sns.boxplot(data=data_df, y = data_df[num_features[feature_location]],x='target',

                        ax = ax[row][col])

            ax[row][col].set_title(num_features[feature_location].capitalize())

            ax[row][col].set(xlabel='',ylabel='')

            #ax[row][col].annotate('actual group', xy=(100,200), xytext=(100, 300))

    plt.suptitle('Continous Features Boxplot',color='b',fontsize = 20);



continuous_variables_boxplot(num_features)
def feature_independent(feature):

    chi2, p , dof, expected = stats.chi2_contingency(pd.crosstab(data_df['target'],data_df[feature]))

    #print("feature : {}".format(feature))

    #print('Chi : {:.4}\np-value :{:.4} \nDOF :{}'.format(chi2,p,dof))

    print(

        '{} is not associated. p-value {}'.format(feature,p)) if p > 0.05 else print(

        '{} is associated.  p-value {}'.format(feature,p))

    print('_'*55)

        

for feature in ['sex','cp','fbs','restecg','exang','slope','ca','thal']:

    feature_independent(feature)
age_dummies = pd.get_dummies(data_df['age_bucket'],drop_first=True,prefix="age")

cp_dummies = pd.get_dummies(data_df['cp'],drop_first=True,prefix="cp")

ca_dummies = pd.get_dummies(data_df['ca'],drop_first=True,prefix="ca")

slope_dummies = pd.get_dummies(data_df.slope,drop_first=True,prefix="slope")

thal_dummies = pd.get_dummies(data_df.thal,drop_first=True,prefix="thal")

restecg_dummies = pd.get_dummies(data_df.restecg,drop_first=True,prefix="restecg")

data = pd.concat([age_dummies,cp_dummies,ca_dummies,slope_dummies,

           thal_dummies,restecg_dummies,data_df[num_features],data_df['target']],axis = 1)
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import numpy as np

scale = preprocessing.StandardScaler()

data['xt_chol'] = scale.fit_transform(data['xt_chol'].values.reshape(-1, 1))

data['xt_thalach'] = scale.fit_transform(data['xt_thalach'].values.reshape(-1, 1))

data['xt_trestbps'] = scale.fit_transform(data['xt_trestbps'].values.reshape(-1, 1))

data['xt_oldpeak'] = scale.fit_transform(data['xt_oldpeak'].values.reshape(-1, 1))



X = data.iloc[:,0:-1]

y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 42)
def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    plt.xlabel("Threshold")

    plt.legend(loc="upper left")

    plt.ylim([0, 1])
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,roc_curve,recall_score, precision_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import precision_recall_curve



lr = LogisticRegression(solver='liblinear', random_state=42, C=25, max_iter=50)

lr_cvs = cross_val_score(lr, X_train, y_train, cv=5)

print('Mean Score: {:.3}\t Std Deviation: {:.3}'.format(lr_cvs.mean(), lr_cvs.std()))

lr_predict_train = cross_val_predict(lr,X_train, y_train, cv=5)

print('Precision Score: {:.3} \t Recall Score: {:.3f}'.format(precision_score(y_train,lr_predict_train),

                                                         recall_score(y_train,lr_predict_train)))







y_scores = cross_val_predict(lr,X= X_train,y = y_train, cv = 3,method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()

fpr, tpr, thresholds = roc_curve(y_train, y_scores)

plot_roc_curve(fpr, tpr)

plt.show()

# lr.fit(X_train, y_train)

# lr_predict = lr.predict(X_test)

# lr_cm = confusion_matrix(y_test,lr_predict )

# print(classification_report(y_test,lr_predict))

# print('ROC AUC Score :{:.3}'.format(roc_auc_score(y_test,lr_predict)))