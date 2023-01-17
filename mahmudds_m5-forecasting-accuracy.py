from IPython.display import HTML

HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/bn8rVBuIcFg?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import matplotlib.pyplot as plt
submission0 = pd.read_csv('/kaggle/input/m5-final-models/submission_LSTM.csv') 

submission1 = pd.read_csv('/kaggle/input/m5-final-models/submission_XGBoost.csv') 

submission2 = pd.read_csv('/kaggle/input/m5-final-models/submission_LGBM.csv')

submission3 = pd.read_csv('/kaggle/input/m5-final-models/submission_prophet.csv')

submission4 = pd.read_csv('/kaggle/input/m5-final-models/submission_SARIMAX.csv')
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv') 

sample_sub = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
def perfect_sub(): # returns a perfect submition just to be sure that we are validating on the right window of values

    submission = OperateBaseModels(submission0,submission1, a=0, b=1)

    diference = validation.merge(submission, how='right')

    shift= 56

    perfect_submission = diference[diference.columns[-shift:-shift+28]]





    #

    col = { 'd_'+str(1914+i):'F'+str(i+1) for i in range(28)}

    perfect_submission = perfect_submission.rename(columns=col)

    perfect_submission.insert(loc=0, column='id', value= submission0['id']) 

    perfect_submission = perfect_submission.fillna(0)

    return perfect_submission





def OperateBaseModels(ModelA,ModelB, a,b):

    submissionMerge = pd.merge(ModelA, ModelB, on='id', how='left', suffixes=('_x', '_y'))#.mean(level=0)

    submissionMerge

    submission = pd.DataFrame()

    submission.insert(loc=0, column='id', value= ModelB['id']) 



    for j in range(28):

        i =j+1

        

        submission.insert(loc=i, column='F'+str(i), value= ((submissionMerge[submissionMerge.columns[i]]*a)+(submissionMerge[submissionMerge.columns[i+28]])*b)/(a+b)) 

     



    return submission
from typing import Union



import numpy as np

import pandas as pd

from tqdm.notebook import tqdm_notebook as tqdm

class WRMSSEEvaluator(object):



    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):

        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]

        train_target_columns = train_y.columns.tolist()

        weight_columns = train_y.iloc[:, -28:].columns.tolist()



        train_df['all_id'] = 0  # for lv1 aggregation



        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()

        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()



        if not all([c in valid_df.columns for c in id_columns]):

            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)



        self.train_df = train_df

        self.valid_df = valid_df

        self.calendar = calendar

        self.prices = prices



        self.weight_columns = weight_columns

        self.id_columns = id_columns

        self.valid_target_columns = valid_target_columns



        weight_df = self.get_weight_df()



        self.group_ids = (

            'all_id',

            'state_id',

            'store_id',

            'cat_id',

            'dept_id',

            ['state_id', 'cat_id'],

            ['state_id', 'dept_id'],

            ['store_id', 'cat_id'],

            ['store_id', 'dept_id'],

            'item_id',

            ['item_id', 'state_id'],

            ['item_id', 'store_id']

        )



        for i, group_id in enumerate(tqdm(self.group_ids)):

            train_y = train_df.groupby(group_id)[train_target_columns].sum()

            scale = []

            for _, row in train_y.iterrows():

                series = row.values[np.argmax(row.values != 0):]

                scale.append(((series[1:] - series[:-1]) ** 2).mean())

            setattr(self, f'lv{i + 1}_scale', np.array(scale))

            setattr(self, f'lv{i + 1}_train_df', train_y)

            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())



            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)

            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())



    def get_weight_df(self) -> pd.DataFrame:

        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()

        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])

        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})

        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)



        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])

        weight_df['value'] = weight_df['value'] * weight_df['sell_price']

        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']

        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)

        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)

        return weight_df



    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:

        valid_y = getattr(self, f'lv{lv}_valid_df')

        score = ((valid_y - valid_preds) ** 2).mean(axis=1)

        scale = getattr(self, f'lv{lv}_scale')

        return (score / scale).map(np.sqrt)



    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:

        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape



        if isinstance(valid_preds, np.ndarray):

            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)



        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)



        all_scores = []

        for i, group_id in enumerate(self.group_ids):

            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)

            weight = getattr(self, f'lv{i + 1}_weight')

            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)

            all_scores.append(lv_scores.sum())



        return np.mean(all_scores)

df_train_full =  pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")

df_calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

df_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")



df_train = df_train_full.iloc[:, :-28]

df_valid = df_train_full.iloc[:, -28:]

evaluator = WRMSSEEvaluator(df_train, df_valid, df_calendar, df_prices)





def WRMSSEE(sub): # Add a function to take the standard submission format 

    val_preds = sample_sub[['id']].merge(sub, on = 'id')  #Order values like the submission example

    

    val_preds.columns = ['id'] + list(df_valid.columns)  # Rename columns

    valid_preds = val_preds.iloc[:30490, -28:] #Take just validation data from the submition dataframe

    return evaluator.score(valid_preds)
def function(solution):

    solution = np.abs(solution )

    submission = OperateBaseModels(submission0,submission1, a=solution[0], b=solution[1])

    submission = OperateBaseModels(submission,submission2, a=1, b=solution[2]) 

    submission = OperateBaseModels(submission,submission3, a=1, b=solution[3]) 

    submission = OperateBaseModels(submission,submission4, a=1, b=solution[4])

                                   

                                   

    #submission = OperateBaseModels(submission,submission2, a=1, b=solution[2]) #I just added this for fun to see how much it affects to have more models into consideration 

    #submission = OperateBaseModels(submission,submission3, a=1, b=solution[3]) #00

    #submission = OperateBaseModels(submission,submission4, a=1, b=solution[4])#00

    #submission = OperateBaseModels(submission,submission5, a=1, b=solution[5])#00

    #submission = OperateBaseModels(submission,submission6, a=1, b=solution[3])

    #submission = OperateBaseModels(submission,submission7, a=1, b=solution[4])

    #submission = OperateBaseModels(submission,submission8, a=1, b=solution[8])#00

    #submission = OperateBaseModels(submission,submission9, a=1, b=solution[5])

    #submission = OperateBaseModels(submission,submission10, a=1, b=solution[10])#00



    #Mean_error = np.mean(np.mean(np.abs(perfect_sub._get_numeric_data()-submission._get_numeric_data())))

    error = WRMSSEE(submission) #+Mean_error/2

    return error 
perfect_sub = perfect_sub()
from scipy.optimize import OptimizeResult

from scipy.optimize import minimize

hist = []

def custmin(fun, x0, args=(), maxfev=None, stepsize=0.1,  # Then we optimize the coeficients to minimize error

        maxiter=500, callback=None, **options):

    bestx = x0

    besty = fun(x0)

    funcalls = 1

    niter = 0

    improved = True

    stop = False



    while improved and not stop and niter < maxiter:

        improved = False

        niter += 1

        print('Iteration number',niter,'WRMSSEE',besty, 'using ', str(np.abs(np.array(bestx)) ))

        hist.append(besty)

        for dim in range(np.size(x0)):

            for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:

                testx = np.copy(bestx)

                testx[dim] = s

                testy = fun(testx, *args)

                funcalls += 1

                if testy < besty:

                    besty = testy

                    bestx = testx

                    improved = True

            if callback is not None:

                callback(bestx)

            if maxfev is not None and funcalls >= maxfev:

                stop = True

                break



    return OptimizeResult(fun=besty, x=bestx, nit=niter,

                          nfev=funcalls, success=(niter > 1))

x0 = np.random.rand(5)

res = minimize(function, x0, method=custmin, options=dict(stepsize=0.05))
Plot_solution =np.abs(res.x)

# [9.35946038e-04, 1.00239378e+01, 1.67410230e-02, 4.01652749e+00,   1.16401096e-02, 9.17120136e-01]

sns.set()

plt.plot(hist)

plt.ylabel('WRMSSEE')

plt.xlabel('Iteration')



Plot_solution
plt.figure()





ax = sns.barplot( x=pd.DataFrame(Plot_solution).index,y=0, data=pd.DataFrame(Plot_solution));

ax.set(xlabel='Model', ylabel='Coeficients form weighted average ');

ax.set_xticklabels(['DNN Embedings','XGBoost','LGBM','SARIMAX','Prophet'], rotation=30);
solution = Plot_solution



solution = np.abs(solution )

submission = OperateBaseModels(submission0,submission1, a=solution[0], b=solution[1])

submission = OperateBaseModels(submission,submission2, a=1, b=solution[2]) 

submission = OperateBaseModels(submission,submission3, a=1, b=solution[3]) 

submission = OperateBaseModels(submission,submission4, a=1, b=solution[4])





#submission = OperateBaseModels(submission,submission2, a=1, b=solution[2]) #I just added this for fun to see how much it affects to have more models into consideration 

#submission = OperateBaseModels(submission,submission3, a=1, b=solution[3]) #00

#submission = OperateBaseModels(submission,submission4, a=1, b=solution[4])#00

#submission = OperateBaseModels(submission,submission5, a=1, b=solution[5])#00

#submission = OperateBaseModels(submission,submission6, a=1, b=solution[3])

#submission = OperateBaseModels(submission,submission7, a=1, b=solution[4])

#submission = OperateBaseModels(submission,submission8, a=1, b=solution[8])#00

#submission = OperateBaseModels(submission,submission9, a=1, b=solution[5])

#submission = OperateBaseModels(submission,submission10, a=1, b=solution[10])#00
WRMSSEE(submission)
#error

#perfect_sub

diference = validation.merge(submission,how='left')                                      

error = np.mean(np.mean(np.abs(perfect_sub._get_numeric_data()-submission._get_numeric_data())))
submission.to_csv("submission.csv", index=False)

submission