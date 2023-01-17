# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")



from scipy.stats import norm

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, learning_curve, cross_validate, train_test_split, KFold, cross_val_score

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import OrdinalEncoder

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
data=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

data.head(2)

test_df=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
data.drop(columns='id',inplace=True)
def count_plot(df,feat,palette='rainbow'):

    plt.style.use('seaborn')

    sns.set_style('whitegrid')



    labels=df[feat].value_counts().index

    values=df[feat].value_counts().values

    

    plt.figure(figsize=(15,5))



    ax = plt.subplot2grid((1,2),(0,0))

    sns.barplot(x=labels, y=values,palette=palette, alpha=0.75)

    for i, p in enumerate(ax.patches):

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2., height + 0.1, values[i],ha="center")

    plt.title('Response of Customer', fontsize=15, weight='bold')    

    plt.show()
sns.set_style("whitegrid")
count_plot(data,'Response')
missing = data.isnull().sum()

missing
count_plot(data,'Gender','Purples')

plt.show()
import plotly.express as px

fig=px.histogram(data, x="Age", color="Response", marginal="violin",title ="Distribution of Age vs Response", 

                   labels={"Age": "Age"},

                   template="plotly_dark",

                   color_discrete_map={"0": "Not Buy", "1": "Buy"})

fig.show()
bins = [20, 30, 40, 50, 60, 70, 80,90]

labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79','80+']

data['AgeClass']=pd.cut(data.Age, bins, labels = labels,include_lowest = True)



test_df['AgeClass']=pd.cut(test_df.Age, bins, labels = labels,include_lowest = True)



data[['Age','AgeClass']].head(5)
with sns.axes_style(style='ticks'):

    g = sns.factorplot("Vehicle_Damage", "Age", "Gender", data=data, kind="box")

    g.set_axis_labels("Vehicle_Damage", "Age");
data_cats=['Gender','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Policy_Sales_Channel','Vintage','AgeClass']

data_nums=['Age','Annual_Premium']

data_all=data_cats+data_nums
def detect_outliers(df,feat):

    Q1 = data[feat].quantile(0.25)

    Q3 = data[feat].quantile(0.75)

    IQR = Q3 - Q1

    #data[~ ((data['Annual_Premium'] < (Q1 - 1.5 * IQR)) |(data['Annual_Premium'] > (Q3 + 1.5 * IQR))) ]

    return df[((df[feat] < (Q1 - 1.5 * IQR)) |(data[feat] > (Q3 + 1.5 * IQR))) ].shape[0]



def clean_outliers(df,feat):

    Q1 = data[feat].quantile(0.25)

    Q3 = data[feat].quantile(0.75)

    IQR = Q3 - Q1

    return df[~ ((df[feat] < (Q1 - 1.5 * IQR)) |(data[feat] > (Q3 + 1.5 * IQR))) ]
for feat in data_nums:

    res=detect_outliers(data,feat)

    if (res>0):

        print('%d Outlier detected in feature %s' % (res,feat))
clean_data=clean_outliers(data,'Annual_Premium')

clean_data.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(clean_data[data_cats+data_nums], clean_data.Response, test_size=0.33, random_state=1)
def prepare_inputs(train):

    oe = OrdinalEncoder()

    oe.fit(train)

    return oe
oe=prepare_inputs(data[data_cats])



X_train_enc=oe.transform(X_train[data_cats])

X_test_enc=oe.transform(X_test[data_cats])



# there is 2 unknown new Policy_Sales_Channel values in test 141 and 142

# we replace them with 140



test_df.loc[test_df['Policy_Sales_Channel']==141.0, 'Policy_Sales_Channel']=140.0

test_df.loc[test_df['Policy_Sales_Channel']==142.0, 'Policy_Sales_Channel']=140.0



test_df_enc=oe.transform(test_df[data_cats])

all_train_enc=np.concatenate((X_train_enc, X_train[data_nums].values), axis=1)

all_test_enc=np.concatenate((X_test_enc, X_test[data_nums].values), axis=1)



all_test_df_enc=np.concatenate((test_df_enc, test_df[data_nums].values), axis=1)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2, mutual_info_classif



# chi2 for categorical variables

# mutual_info_classif for mixed variables

   

fs = SelectKBest(score_func=mutual_info_classif, k='all')

fs.fit(all_train_enc, y_train)

X_train_fs = fs.transform(all_train_enc)







for i in range(len(fs.scores_)):

    print('%s: %f' % (data_all[i], fs.scores_[i]))



plt.figure(figsize=(18,8))

sns.barplot(data_all, fs.scores_, orient='v')

plt.title('Categorical Feature Selection with mutual_info_classif')

plt.show()
from imblearn.over_sampling import RandomOverSampler 

from imblearn.over_sampling import ADASYN



#ros = RandomOverSampler(random_state=42, sampling_strategy='minority')

#all_train_enc_over_sampled, y_train_over_sampled = ros.fit_resample(all_train_enc, y_train)



ada = ADASYN(random_state=42)

all_train_enc_over_sampled, y_train_over_sampled = ada.fit_resample(all_train_enc, y_train)



y_train=y_train_over_sampled
import plotly.express as px

from sklearn.decomposition import PCA

n_components = 2



pca = PCA(n_components=n_components)

components = pca.fit_transform(all_train_enc_over_sampled)



total_var = pca.explained_variance_ratio_.sum() * 100





fig = px.scatter(components, x=0, y=1, color=y_train, title=f'Total Explained Variance: {total_var:.2f}%',)

fig.show()

from sklearn import preprocessing



scaler = preprocessing.StandardScaler()

scaler.fit(all_train_enc)

X_train_transformed = scaler.transform(all_train_enc_over_sampled)

X_test_transformed = scaler.transform(all_test_enc)

all_test_df_transformed = scaler.transform(all_test_df_enc)
from collections import Counter 



#calculate class weight for XGBoost

counter = Counter(y_train)

weight_estimate = counter[0] / counter[1]

print('Estimate: %.3f' % weight_estimate)

# this is mainly for scale_pos_weight in xgboost since it's not support class_weight='balanced' like option

# weights is manual in xgboost

# eg. xgtest=XGBClassifier(random_state=55,  scale_pos_weight=weight_estimate)
rf=RandomForestClassifier(random_state=55, n_jobs=-1)

lr=LogisticRegression(random_state=55, n_jobs=-1)

sv = SVC(probability=True,random_state=55,)

logreg = LogisticRegression(solver='newton-cg',random_state=55, n_jobs=-1) 

gb = GradientBoostingClassifier(random_state=55)

gnb = GaussianNB()

xgb = XGBClassifier(random_state=55, nthread=-1)
models=[rf, lr, logreg, gb, gnb, xgb]

cv = StratifiedKFold(5, shuffle=True, random_state=42)
model_results = pd.DataFrame()

row_number = 0

results = []

names = []



for ml in models:

    model_name=ml.__class__.__name__

    print('Training %s model ' % model_name)

    cv_results = cross_validate(ml, X_train_transformed, y_train, cv=cv, scoring='roc_auc', return_train_score=True, n_jobs=-1 )

    model_results.loc[row_number,'Model Name']=model_name

    model_results.loc[row_number, 'Train roc_auc  Mean']=cv_results['train_score'].mean()

    model_results.loc[row_number, 'Test roc_auc  Mean']=cv_results['test_score'].mean()

    model_results.loc[row_number, 'Fit Time Mean']=cv_results['fit_time'].mean()

    results.append(cv_results)

    names.append(model_name)

    

    row_number+=1
cv_results_array = []

for tt in results:

    cv_results_array.append(tt['test_score'])



fig = plt.figure(figsize=(18, 6))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(cv_results_array)

ax.set_xticklabels(names)

plt.show()
display(model_results.style.background_gradient(cmap='summer_r'))
eval_set = [(X_train_transformed, y_train), (X_test_transformed,y_test)]

xgtest=XGBClassifier(random_state=55, nthread=-1)

xgtest.fit(X_train_transformed, y_train, eval_metric=["auc", "logloss", "error"], eval_set=eval_set, verbose=False)
y_scores=xgtest.predict(X_test_transformed)

roc_auc_score(y_test, y_scores)
from matplotlib import pyplot

results = xgtest.evals_result()

epochs = len(results['validation_0']['error'])

x_axis = range(0, epochs)

# plot log loss

fig, ax = pyplot.subplots()

ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax.plot(x_axis, results['validation_1']['logloss'], label='Test')

ax.legend()

pyplot.ylabel('Log Loss')

pyplot.title('XGBoost Log Loss')

pyplot.show()

# plot classification error

fig, ax = pyplot.subplots()

ax.plot(x_axis, results['validation_0']['error'], label='Train')

ax.plot(x_axis, results['validation_1']['error'], label='Test')

ax.legend()

pyplot.ylabel('Classification Error')

pyplot.title('XGBoost Classification Error')

# plot auc

fig, ax = pyplot.subplots()

ax.plot(x_axis, results['validation_0']['auc'], label='Train')

ax.plot(x_axis, results['validation_1']['auc'], label='Test')

ax.legend()

pyplot.ylabel('AUC')

pyplot.title('XGBoost AUC Score')

pyplot.show()
gb_proba=xgtest.predict_proba(X_test_transformed)[:,1]
fpr, tpr, thresholds  = roc_curve(y_test, gb_proba)





plt.title('XGBoost ROC curve')

plt.xlabel('FPR (Precision)')

plt.ylabel('TPR (Recall)')



plt.plot(fpr,tpr)

plt.plot((0,1), ls='dashed',color='black')

plt.show()

print ('Area under curve (AUC): ', format(round(auc(fpr,tpr),5)))
from yellowbrick.classifier import ClassificationReport





def view_report(model,X,y):

    visualizer = ClassificationReport(

        model, classes=['0', '1'],

        cmap="YlGn", size=(600, 360)

    )

    visualizer.fit(X,y)

    visualizer.score(X,y)

    visualizer.show()

model = xgtest

view_report(model,X_train_transformed, y_train)
from yellowbrick.classifier import ClassPredictionError



def show_errors(model, X_train,y_train,X_test,y_test):

    classes=['Not Responded','Responded']

    visualizer = ClassPredictionError(model)



    visualizer.fit(X_train, y_train)

    visualizer.score(X_test, y_test)

    visualizer.show()
model = xgtest

show_errors(model, X_train_transformed, y_train,X_test_transformed,y_test)
from yellowbrick.classifier import DiscriminationThreshold



model = xgtest



visualizer = DiscriminationThreshold(model, n_trials=1,excludestr=['queue_rate'],random_state=55)

visualizer.fit(X_train_transformed, y_train)

visualizer.show()
tt=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv')

id=tt.id
best_model=XGBClassifier(random_state=55)

best_model.fit(X_train_transformed, y_train)
preds=best_model.predict_proba(all_test_df_transformed)[:,1]
submission = pd.DataFrame(data = {'id': id, 'Response': preds})

submission.to_csv('vehicle_insurance.csv', index = False)

submission.head()