%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # drawing graph

import warnings; warnings.filterwarnings("ignore") 

import os; os.environ['OMP_NUM_THREADS'] = '4' # speed up using 4 cpu

from fastai.tabular import *

from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE
def auroc_score(input, target):

    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()

    return roc_auc_score(target, input)



class AUROC(Callback):

    _order = -20 #Needs to run before the recorder



    def __init__(self, learn, **kwargs): self.learn = learn

    def on_train_begin(self, **kwargs): self.learn.recorder.add_metric_names(['AUROC'])

    def on_epoch_begin(self, **kwargs): self.output, self.target = [], []

    

    def on_batch_end(self, last_target, last_output, train, **kwargs):

        if not train:

            self.output.append(last_output)

            self.target.append(last_target)

                

    def on_epoch_end(self, last_metrics, **kwargs):

        if len(self.output) > 0:

            output = torch.cat(self.output)

            target = torch.cat(self.target)

            preds = F.softmax(output, dim=1)

            metric = auroc_score(preds, target)

            return add_metrics(last_metrics, [metric])
dtypes={

    'Age':                         'int64',

    'Attrition':                   'category',

    'BusinessTravel':              'category',

    'DailyRate':                   'int64',

    'Department':                  'category',

    'DistanceFromHome':            'int64',

    'Education':                   'int64',

    'EducationField':              'category',

    'EmployeeCount':               'int64',

    'EmployeeNumber':              'int64',

    'EnvironmentSatisfaction':     'int64',

    'Gender':                      'category',

    'HourlyRate':                  'int64',

    'JobInvolvement':              'int64',

    'JobLevel':                    'int64',

    'JobRole':                     'category',

    'JobSatisfaction':             'int64',

    'MaritalStatus':               'category',

    'MonthlyIncome':               'int64',

    'MonthlyRate':                 'int64',

    'NumCompaniesWorked':          'int64',

    'Over18':                      'category',

    'OverTime':                    'category',

    'PercentSalaryHike':           'int64',

    'PerformanceRating':           'int64',

    'RelationshipSatisfaction':    'int64',

    'StandardHours':               'int64',

    'StockOptionLevel':            'int64',

    'TotalWorkingYears':           'int64',

    'TrainingTimesLastYear':       'int64',

    'WorkLifeBalance':             'int64',

    'YearsAtCompany':              'int64',

    'YearsInCurrentRole':          'int64',

    'YearsSinceLastPromotion':     'int64',

    'YearsWithCurrManager':        'int64',}
# source : https://www.ibm.com/communities/analytics/watson-analytics-blog/hr-employee-attrition/

df = pd.read_excel('../input/WA_Fn-UseC_-HR-Employee-Attrition.xlsx', sheet_name=0,dtype=dtypes)

print("Shape of dataframe is: {}".format(df.shape))
# preprocessing : categorical encoding

df['Attrition']=df.Attrition.eq('Yes').mul(1) # change target from Yes/no to 1/0

cont=[]

cat=[]

for key, value in dtypes.items():

    if key!='Attrition':

        if value == "int64":

            cont.append(key)

        else:

            cat.append(key)

df = pd.get_dummies(df, columns=cat)
# get train data

col = df.columns

cont=[]

for i in range(0,len(col)):

    if col[i]!='Attrition':

        cont.append(col[i])
#save the column name

x_col = cont

y_col = 'Attrition'



X = df.drop('Attrition', axis=1)

Y = df.Attrition

X_res, Y_res = SMOTE().fit_resample(X, Y)
smote_df = pd.DataFrame(X_res, columns = x_col)
smote_df = smote_df.assign(Attrition = Y_res)
smote_df.Attrition.value_counts()
smote_df.shape
dep_var='Attrition'

procs=[ Normalize]

data = (TabularList.from_df(smote_df, cont_names=col , procs=procs,)

                .split_subsets(train_size=0.8, valid_size=0.2, seed=34)

                .label_from_df(cols=dep_var)

                .databunch())
learn = tabular_learner(data, layers=[200,100],metrics=accuracy, callback_fns=AUROC)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit(3,lr=1e-3)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2,max_lr=1e-6)
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()