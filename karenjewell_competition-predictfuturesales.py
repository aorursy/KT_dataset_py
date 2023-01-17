import os
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import all_estimators
from sklearn import base

### Allow multi-line results
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

### See all dataframe output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# Dynamically load data from filenames
folder = "/Users/Karen/OneDrive/GitHub/Kaggle_PredictFutureSales/"
datafolder = "competitive-data-science-predict-future-sales/"
for dirname, _, filenames in os.walk(folder + datafolder):
    for filename in filenames:
        #file = os.path.join(dirname, filename)
        globals()[filename.split('.')[0]] = pd.read_csv(folder + datafolder + filename)
        print(filename)
        
print(sales_train.head(5))
len(sales_train)
print(test.head(5))
print(shops.head(5))
print(items.head(5))
print(item_categories.head(5))
type(sales_train['date'][0])
### the date column contains value as str format, we assume formatting is consistent dd.mm.yyyy
print(len(set(sales_train['date'])))
# print(set(sales_train['date']))
### 1034 unique dates

### extract the month + year
sales_train['month'] = sales_train.apply(lambda x: x['date'].split('.',1)[1],axis=1)
print(set(sales_train['month']))

### Calculate monthly items sold and prepare for training
### however doesn't account for trends over time.
sales_train_t = sales_train\
                .groupby(['shop_id','item_id','month'])\
                ['item_cnt_day'].sum()\
                .reset_index()\
                .drop(columns={'month'})\
                .rename(columns={'item_cnt_day':'item_cnt_month'})
sales_train_t
### Pick regressor models
estimators = all_estimators()
dict_classifiers = {}

shortlist = ['ARDRegression','BaggingRegressor','DecisionTreeRegressor','ExtraTreesRegressor', \
             'GradientBoostingRegressor','KNeighborsRegressor','LassoLars','LinearRegression', \
             'Linear SVR', 'PoissonRegressor','RandomForestRegressor']

for name, class_ in estimators:
    ### use one: ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin
    if issubclass(class_, base.RegressorMixin):
        if name in shortlist:
            ### excluding any models which require additional parameters
            try:
                dict_classifiers[name] = class_()
            except:
                pass
        
len(dict_classifiers.items())
pprint.pprint(dict_classifiers)
### Split train-test data
x = sales_train_t.copy().drop(columns={'item_cnt_month'})
y = sales_train_t['item_cnt_month']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
xy_train = list(zip(x_train, y_train))
xy_test = list(zip(x_test, y_test))
### Cycle train through all models
results = ([['base',0]])

for model, model_inst in dict_classifiers.items():
    try: 
        model_inst.fit(x_train, y_train)
        pred = np.array(model_inst.predict(x_test))
        score = metrics.mean_squared_error(y_test, pred)
        print(model, "mse: ",score)
        results.append([model,score])
    except:
        pass
### Results of all trained models
### For MSE: a lower MSE is desired
print(pd.DataFrame(results, columns={'model','score'}).sort_values(by='score', ascending=True))
for name, class_ in estimators:
    if name == 'BaggingRegressor':
        print(class_)
### Prepare test set + final model
test_t = test.drop(columns={"ID"})

final_clf = sklearn.ensemble._forest.RandomForestRegressor()
final_clf.fit(x, y)
pred = np.array(final_clf.predict(test_t))
finalresults = pd.DataFrame()
finalresults['item_cnt_month'] = list(pred)
finalresults = finalresults.join(test[['ID']])
print(finalresults)
### Export predictions to .csv
finalresults.to_csv(folder + 'results_RandomForest_NoProcessing.csv',index=False)
%%script false --no-raise-error
from pycaret.regression import *
exp_reg101 = setup(data = sales_train_t, target = 'item_cnt_day', session_id=1)