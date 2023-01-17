# Calculations and manipulation

import numpy as np

import pandas as pd



# Data visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Training and testing division

from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split



# ML algorithms

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import LinearSVC

!pip install xgboost

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



# Evaluation of algorithms

from sklearn.metrics import accuracy_score

from sklearn.metrics import make_scorer

from sklearn.metrics import confusion_matrix



# Testing an ensemble

from sklearn.ensemble import VotingClassifier



# Hyperparameter optimization

from sklearn.model_selection import GridSearchCV



# Export the model

!pip install joblib

import joblib



# Silence warnings

import warnings

warnings.filterwarnings("ignore")
path = '../input/cardiovascular-disease-dataset/cardio_train.csv'

data = pd.read_csv(path, sep=';')
data
sns.set_style('darkgrid')



background_color = ['#eaeaf2']



my_palette = ['#02c39a',

              '#ff006e',

              '#ffe74c',

              '#8338ec',

              '#3a86ff',

              '#e07a5f',

              '#525252']



sns.set_palette(my_palette, 7)
miss_values = pd.DataFrame(data.isna().sum())

miss_values.T
cols_useless = ['id']

data.drop(cols_useless, axis=1, inplace= True)
data.drop_duplicates(inplace= True)
def elimOutlier(dataframe, col, min, max):

    # Remove values below min and above max

    dataframe = dataframe[dataframe[col] > min]

    dataframe = dataframe[dataframe[col] < max]

    return dataframe
#turning into years

data.loc[:, 'age'] = data.loc[:, 'age'].apply(lambda x: int(x/365))



fig, ax = plt.subplots(figsize = (10, 6))



sns.distplot(data['age'])

plt.xlim(data['age'].min(),

         data['age'].max())



plt.subplots_adjust(hspace= 0.6)

plt.show()
fig, ax = plt.subplots(figsize = (10, 6))



plt.subplot(2,2,1)

sns.distplot(data['height'])

plt.title('Distribtion')



plt.subplot(2,2,2)

sns.distplot(data['height'], kde= False, )

plt.xlim(140, 190)

plt.title('Zoom / No KDE')



plt.subplot(2,2,3)

sns.distplot(data['weight'])

plt.title('Distribtion')



plt.subplot(2,2,4)

sns.distplot(data['weight'], kde= False)

plt.xlim(40, 130)

plt.title('Zoom / No KDE')



plt.subplots_adjust(hspace= 0.6)

plt.show()
fig, ax = plt.subplots(figsize = (10, 6))



plt.subplot(2,2,1)

sns.distplot(data['ap_hi'], kde= False)

plt.title('Dist total - ap_hi | selection without outliers (blue)')



plt.subplot(2,2,2)

sns.distplot(data['ap_hi'], kde= False)

plt.xlim(0, 175)

plt.title('Zoom')



plt.subplot(2,2,3)

sns.distplot(data['ap_lo'], kde= False)

plt.title('Dist total - ap_lo | selection without outliers (blue)')



plt.subplot(2,2,4)

sns.distplot(data['ap_lo'], kde= False)

plt.xlim(0, 150)

plt.title('Zoom')



plt.subplots_adjust(hspace= 0.6)

plt.show()
data.drop(data[data["ap_lo"] > data["ap_hi"]].index, inplace=True)



data = elimOutlier(data, 'ap_hi', 0, 175)

data = elimOutlier(data, 'ap_lo', 0, 150)
fig, ax = plt.subplots(figsize = (10, 6))



plt.subplot(2,1,1)

sns.countplot('cholesterol', data= data)

plt.title('Cholesterol')



plt.subplot(2,1,2)

sns.countplot('gluc', data= data)

plt.title('Glucose')



plt.subplots_adjust(hspace= 0.6)

plt.show()
fig, ax = plt.subplots(figsize = (10, 6))



plt.subplot(2,2,1)

sns.countplot('smoke', data= data)

plt.title('Smoke')



plt.subplot(2,2,2)

sns.countplot('alco', data= data)

plt.title('Alcohol')



plt.subplot(2,2,3)

sns.countplot('active', data= data)

plt.title('Activity')



plt.subplot(2,2,4)

sns.countplot('cardio', data= data)

plt.title('Cardio disease')



plt.subplots_adjust(hspace= 0.6)

plt.show()
fig, ax = plt.subplots(figsize = (10, 6))

sns.countplot(x= 'age', hue= 'cardio', data= data)

plt.title('CD -> cardio disease')

plt.legend(['CD (─)', 'CD (+)'],

           loc= 'upper right')



plt.subplots_adjust(hspace= 0.3)

plt.show()
data.loc[(data['age'] < 45), 'age'] = 1

data.loc[(data['age'] >= 45) & (data['age'] < 55), 'age'] = 2

data.loc[(data['age'] >= 55) & (data['age'] < 60), 'age'] = 3

data.loc[(data['age'] >= 60), 'age'] = 4
# Calculation of BMI

data['bmi'] = data['weight'] / ((data['height']/100) ** 2)

data.drop(['weight', 'height'], axis=1, inplace= True)
data.loc[(data['bmi'] < 18.5), 'bmi'] = 1

data.loc[(data['bmi'] >= 18.5) & (data['bmi'] < 25), 'bmi'] = 2

data.loc[(data['bmi'] >= 25) & (data['bmi'] < 30), 'bmi'] = 3

data.loc[(data['bmi'] >= 30), 'bmi'] = 4
data['bpc'] = 0



data.loc[(data['ap_hi'] < 120) & (data['ap_lo'] < 80), 'bpc'] = 1



data.loc[((data['ap_hi'] >= 120) & (data['ap_hi'] < 130)) &

         ((data['ap_lo'] < 80)), 'bpc'] = 2



data.loc[((data['ap_hi'] >= 130) & (data['ap_hi'] < 140)) |

         ((data['ap_lo'] >= 80) & (data['ap_lo'] < 90)), 'bpc'] = 3



data.loc[((data['ap_hi'] >= 140) & (data['ap_hi'] < 180)) |

         ((data['ap_lo'] >= 90) & (data['ap_lo'] < 120)), 'bpc'] = 4



data.loc[(data['ap_hi'] >= 180) | (data['ap_lo'] >= 120), 'bpc'] = 5



cols_ap_ = ['ap_hi', 'ap_lo']

data.drop(cols_ap_, axis= 1, inplace= True)
data
def habitPlot(dataframe, col):

    sns.countplot(x= col,

                  hue= 'cardio',

                  data= data)

    plt.title('Comparation - {}'.format(col))

    plt.legend(['CD (─)', 'CD (+)'],

               loc= 'upper right')



fig, ax = plt.subplots(figsize = (10, 12))

fig.suptitle('CD -> Cardio Disease')



plt.subplot(4,2,1)

habitPlot(data, 'gender')



plt.subplot(4,2,2)

habitPlot(data, 'cholesterol')



plt.subplot(4,2,3)

habitPlot(data, 'gluc')



plt.subplot(4,2,4)

habitPlot(data, 'smoke')



plt.subplot(4,2,5)

habitPlot(data, 'alco')



plt.subplot(4,2,6)

habitPlot(data, 'active')



plt.subplot(4,2,7)

habitPlot(data, 'bmi')



plt.subplot(4,2,8)

habitPlot(data, 'bpc')



plt.subplots_adjust(hspace= 0.6, wspace= 0.3)

plt.show()
fig, ax = plt.subplots(figsize = (10, 6))

sns.heatmap(data.corr(), annot= True,

            cmap= 'YlGn')

plt.show()
# Cross validation

x = data.drop(['cardio'], axis= 1)

y = data['cardio']



# Confusion matrix

x_train, x_test, y_train, y_test = train_test_split(x, y,

                                                    test_size=0.20,

                                                    random_state= 1)
# Cross validation evaluators

scoring = ['accuracy', 'precision', 'recall', 'roc_auc']



def scorAlg(clf, x, y):

    # Cross validation

    scores = cross_validate(clf, x, y,

                            scoring= scoring,

                            cv= 10)

    

    # Confusion matrix

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)



    return (pd.DataFrame(scores), 

            y_pred,

            pd.DataFrame(confusion_matrix(y_test, y_pred)))

    

def plotScor(scores, y_pred, conf_matrix):

    # Plot the results

    fig, ax = plt.subplots(figsize = (10, 6))



    plt.subplot(2,1,1)

    sns.lineplot(scores['test_accuracy'].index,

                 scores['test_accuracy'],

                 label= 'test_accuracy')



    sns.lineplot(scores['test_precision'].index,

                 scores['test_precision'],

                 label= 'test_precision')



    sns.lineplot(scores['test_recall'].index,

                 scores['test_recall'],

                 label= 'test_recall')



    plt.xticks(range(0, len(scores), 1))

    plt.title('Means -> Accuracy : {acc_mean} | Precision: {pre_mean} | Recall: {rec_mean}'.

              format(acc_mean= round(scores['test_accuracy'].mean(),2),

                     pre_mean= round(scores['test_precision'].mean(),2),

                     rec_mean= round(scores['test_recall'].mean(),2)))



    plt.subplot(2,2,3)

    sns.lineplot(scores['fit_time'].index,

                 scores['fit_time'],

                 label= 'fit_time')



    plt.subplot(2,2,3)

    sns.lineplot(scores['score_time'].index,

                 scores['score_time'],

                 label= 'score_time')



    plt.title('Fit and Score time')



    plt.subplot(2,2,4)

    sns.heatmap(conf_matrix,

                annot= True,

                fmt= '0',

                cmap= background_color,

                cbar= False)

    

    plt.title('Matrix Confusion - Train_Test_Split (20%)')



    plt.subplots_adjust(hspace= 0.3)

    plt.show()
rfc = RandomForestClassifier()



scores_rfc, y_pred_rfc, conf_matrix_rfc = scorAlg(rfc, x, y)



plotScor(scores_rfc, y_pred_rfc, conf_matrix_rfc)
sgd = SGDClassifier()



scores_sgd, y_pred_sgd, conf_matrix_sgd = scorAlg(sgd, x, y)



plotScor(scores_sgd, y_pred_sgd, conf_matrix_sgd)
gbc = GradientBoostingClassifier()



scores_gbc, y_pred_gbc, conf_matrix_gbc = scorAlg(gbc, x, y)



plotScor(scores_gbc, y_pred_gbc, conf_matrix_gbc)
abc = AdaBoostClassifier()



scores_abc, y_pred_abc, conf_matrix_abc = scorAlg(abc, x, y)



plotScor(scores_abc, y_pred_abc, conf_matrix_abc)
lsv = LinearSVC()



scores_lsv, y_pred_lsv, conf_matrix_lsv = scorAlg(lsv, x, y)



plotScor(scores_lsv, y_pred_lsv, conf_matrix_lsv)
xbc = XGBClassifier()



scores_xbc, y_pred_xbc, conf_matrix_xbc = scorAlg(xbc, x, y)



plotScor(scores_xbc, y_pred_xbc, conf_matrix_xbc)
lgb = LGBMClassifier()



scores_lgb, y_pred_lgb, conf_matrix_lgb = scorAlg(lgb, x, y)



plotScor(scores_lgb, y_pred_lgb, conf_matrix_lgb)
fig, ax = plt.subplots(figsize = (10, 12))



plt.subplot(2,1,1)

sns.lineplot(scores_rfc['test_accuracy'].index,

             scores_rfc['test_accuracy'],

             label= 'RandomForestClassifier')



sns.lineplot(scores_lgb['test_accuracy'].index,

             scores_lgb['test_accuracy'],

             label= 'LGBMClassifier')



sns.lineplot(scores_sgd['test_accuracy'].index,

             scores_sgd['test_accuracy'],

             label= 'SGDClassifier')



sns.lineplot(scores_gbc['test_accuracy'].index,

             scores_gbc['test_accuracy'],

             label= 'GradientBoostingClassifier')



sns.lineplot(scores_abc['test_accuracy'].index,

             scores_abc['test_accuracy'],

             label= 'AdaBoostClassifier')



sns.lineplot(scores_lsv['test_accuracy'].index,

             scores_lsv['test_accuracy'],

             label= 'LinearSVC')



sns.lineplot(scores_xbc['test_accuracy'].index,

             scores_xbc['test_accuracy'],

             label= 'XGBClassifier')



plt.subplot(2,1,2)

sns.lineplot(scores_rfc['test_precision'].index,

             scores_rfc['test_precision'],

             label= 'RandomForestClassifier')



sns.lineplot(scores_lgb['test_precision'].index,

             scores_lgb['test_precision'],

             label= 'LGBMClassifier')



sns.lineplot(scores_sgd['test_precision'].index,

             scores_sgd['test_precision'],

             label= 'SGDClassifier')



sns.lineplot(scores_gbc['test_precision'].index,

             scores_gbc['test_precision'],

             label= 'GradientBoostingClassifier')



sns.lineplot(scores_abc['test_precision'].index,

             scores_abc['test_precision'],

             label= 'AdaBoostClassifier')



sns.lineplot(scores_lsv['test_precision'].index,

             scores_lsv['test_precision'],

             label= 'LinearSVC')



sns.lineplot(scores_xbc['test_precision'].index,

             scores_xbc['test_precision'],

             label= 'XGBClassifier')



plt.subplots_adjust(hspace= 0.3)

plt.show()
model1 = GradientBoostingClassifier()

model2 = AdaBoostClassifier()



model = VotingClassifier(estimators=[('gbc', model1), 

                                     ('abc', model2)],

                         voting='soft')



scores_model, y_pred_scores_model, conf_matrix_scores_model = scorAlg(model, x, y)



plotScor(scores_model, y_pred_scores_model, conf_matrix_scores_model)
parameters = {'n_estimators': [50, 100, 300, 1000],

              'learning_rate': [1.0, 0.5, 0.3],

              'algorithm': ('SAMME.R', 'SAMME'),

              'random_state': [1]}



clf = AdaBoostClassifier()



abc_tun = GridSearchCV(clf, parameters, scoring= 'precision')



abc_tun.fit(x, y)



abc_tun.best_estimator_
abc_final = AdaBoostClassifier(algorithm='SAMME',

                               learning_rate=0.3,

                               random_state=1)



scores_abc_final, y_pred_abc_final, conf_matrix_abc_final = scorAlg(abc_final, x, y)



plotScor(scores_abc_final, y_pred_abc_final, conf_matrix_abc_final)
joblib.dump(abc_final, 'model.pkl')
abc_final.predict_proba([[1, 1, 1, 2, 0, 1, 0, 2, 2]])