%matplotlib inline

import numpy as np

import pandas as pd

import json

import codecs

import seaborn as sns



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('../input/data.csv')

data.set_index('Unnamed: 0')

del data['Unnamed: 0']

data['Start_time'] = pd.to_datetime(data['Start_time'])

data['End_time'] = pd.to_datetime(data['End_time'])

data['Date'] = pd.to_datetime(data['Date'])



test_data = pd.read_csv('../input/test.csv')

test_data.set_index('Unnamed: 0')

del test_data['Unnamed: 0']

test_data['Start_time']= pd.to_datetime(test_data['Start_time'])

test_data['End_time']= pd.to_datetime(test_data['End_time'])

test_data['Date'] = pd.to_datetime(test_data['Date'])
data.head(3)
print("Data.csv has {} rows & {} Columns".format(data.shape[0],data.shape[1]))

print("Test.csv has {} rows & {} Columns".format(test_data.shape[0],test_data.shape[1]))
null_cols = data.isnull().sum()

null_cols = null_cols[null_cols > 0]

print(null_cols)
null_cols = test_data.isnull().sum()

null_cols = null_cols[null_cols > 0]

print(null_cols)
data      = data[ data['Start_time'].notna() ]

test_data = test_data[ test_data['Start_time'].notna() ]
data['Month'] = data['Start_time'].dt.month

data['Hour']  = data['Start_time'].dt.hour

data['Day']   = data['Start_time'].dt.day



test_data['Month'] = test_data['Start_time'].dt.month

test_data['Hour']  = test_data['Start_time'].dt.hour

test_data['Day']   = test_data['Start_time'].dt.day
del data['Name of episode']

del data['Start_time']

del data['End_time']

del data['Date']



del test_data['Name of episode']

del test_data['Start_time']

del test_data['End_time']

del test_data['Date']
cols = data.columns.tolist()

cols.insert(len(cols), cols.pop(cols.index('Market Share_total')))

data = data.reindex(columns = cols)
data.head()
data.interpolate(inplace = True)

data.dropna(inplace = True)



test_data.interpolate(inplace = True)

test_data.dropna(inplace = True)
print("Data.csv has {} rows & {} Columns".format(data.shape[0],data.shape[1]))

print("Test.csv has {} rows & {} Columns".format(test_data.shape[0],test_data.shape[1]))
from sklearn.model_selection import train_test_split

from catboost import Pool, CatBoostRegressor, cv



X=data.drop(columns=['Market Share_total'])



print(X.columns)
categorical_features_indices =[0,1,2,3,5,7,8,9,10,11,12]



y=data['Market Share_total']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 

                                                    random_state=42)



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 

                                                    random_state=52)
def perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test):

    model = CatBoostRegressor(

        random_seed = 400,

        loss_function = 'RMSE',

        iterations=400,

    )

    

    model.fit(

        X_train, y_train,

        cat_features = categorical_features_indices,

        eval_set=(X_valid, y_valid),

        verbose=False

    )

    

    print("RMSE on training data: "+ model.score(X_train, y_train).astype(str))

    print("RMSE on test data: "+ model.score(X_test, y_test).astype(str))

    

    return model
model=perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test)
feature_score = pd.DataFrame(list(zip(X.dtypes.index, model.get_feature_importance(Pool(X, label=y, cat_features=categorical_features_indices)))),

                columns=['Feature','Score'])



feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
plt.rcParams["figure.figsize"] = (12,7)

ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')

ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)

ax.set_xlabel('')



rects = ax.patches



labels = feature_score['Score'].round(2)



for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')



plt.show()
object_cols = [col for col in data.columns if data[col].dtype == "object"]
labeled_data      = data.copy()

labeled_test_data = test_data.copy()
encoders = {}

encoders.clear()



for col in object_cols:

    le = LabelEncoder()

    labeled_data[col] = le.fit_transform(labeled_data[col])

    le.classes_ = np.append(le.classes_, '<unknown>')

    

    labeled_test_data[col] = labeled_test_data[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)

    labeled_test_data[col] = le.transform(labeled_test_data[col])

    

    encoders[col] = le
cols = labeled_data.columns.tolist()
Var_Corr = labeled_data.corr()

sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
scaler = MinMaxScaler()

labeled_data_scaled = scaler.fit_transform(labeled_data)

labeled_data_scaled = pd.DataFrame(labeled_data_scaled,columns = cols)

labeled_data_scaled['Market Share_total'] = labeled_data['Market Share_total']
labeled_data_scaled.head()
plt.figure(figsize=(10,10))

sns.boxplot(data=labeled_data_scaled[labeled_data_scaled.columns[:-1]])

plt.xticks(rotation=90)
x_features = labeled_data_scaled[labeled_data_scaled.columns[:-1]]

sns.jointplot(x_features.loc[:,'Episode'], x_features.loc[:,'Name of show'], kind="regg", color="#ce1414")
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, SGDRegressor,ARDRegression,RANSACRegressor,PassiveAggressiveRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor 

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import json

import codecs



class Model:



    def __init__(self,esitimator):

        self.estimator = esitimator



    def fit(self,data,evaluation = True, cv=5):

        x , y =  data[data.columns[:-1]], data["Market Share_total"]

        self.x_train, self.x_dev, self.y_train, self.y_dev = train_test_split(x, y, test_size=0.3, random_state=42)

        self.estimator.fit(self.x_train ,self.y_train)

        if evaluation:            

            y_pred = self.predict(self.x_dev)

            self._r2_score , self.mae = self.evaluate(self.y_dev, y_pred)

            self.r2_score_cv, self.mae_cv = self.perform_cv(cv)



    def predict(self,data):

        return self.estimator.predict(data)



    def save_results(self, path):

        result = { "Cross-validation": { "R2 Squared": str(self.r2_score_cv), "MAE":str(self.mae_cv) },

                    "Dev-Set" : { "R2 Squared":str(self._r2_score),    "MAE":str(self.mae) }

                }

        with codecs.open( path , "w", encoding= "utf-8") as J:

            json.dump(result,J, indent=4)

        print("results saved to {}".format(path))



    def evaluate(self,y_truth,y_pred):

        _r2_score = self.calculate_r2_score(y_truth, y_pred)

        mae = self.calculate_mae(y_truth, y_pred)

        return _r2_score , mae



    def calculate_mae(self,y_truth, y_pred):

        return mean_absolute_error(y_truth, y_pred)



    def calculate_r2_score(self,y_truth, y_pred):

        return r2_score(y_truth, y_pred)



    def perform_cv(self, cv = 5):

        r2_score_cv = self.perform_cv_r2_score(self.x_train , self.y_train, cv)

        mae_cv = self.perform_cv_mae(self.x_train , self.y_train, cv)

        return r2_score_cv, mae_cv



    def perform_cv_r2_score(self,x,y,cv = 5):

        cv_r2_scores = cross_val_score( self.estimator, x , y , cv = cv , scoring='r2')

        mean, std = self.get_mean_std(cv_r2_scores)

        return { str(cv) + "-Fold":cv_r2_scores , "Mean":mean, "Std":std }



    def perform_cv_mae(self,x ,y ,cv = 5):

        mse_scorer = make_scorer(mean_absolute_error)

        cv_maes = cross_val_score( self.estimator, x , y , cv = cv , scoring=mse_scorer)

        mean, std = self.get_mean_std(cv_maes)

        return { str(cv) + "-Fold":cv_maes , "Mean":mean, "Std":std }



    def get_mean_std(self, lst):

        return np.mean(lst), np.std(lst)

res_cols=['Model', 'R2 Score' , 'MAE']

results = pd.DataFrame(columns=res_cols)



import os

os.makedirs('results')
def run_model(estimator, data):

    model_name = type(estimator).__name__

    model = Model(estimator)

    model.fit(data)

    model.save_results("results/" + model_name + "_result.json")

    with codecs.open("results/" + model_name + "_result.json","r",encoding="utf-8") as j:

        result = json.load(j)

    #print(result['Cross-validation']['R2 Squared'])

    #print(result['Cross-validation']['MAE'])

    return model.estimator, pd.DataFrame([[model_name, result['Dev-Set']['R2 Squared'] , result['Dev-Set']['MAE'] ]],columns = res_cols)
!ls
model_lin, res = run_model(LinearRegression(), labeled_data)

results = results.append(res)
model_rfr, res = run_model(RandomForestRegressor(n_estimators=10), labeled_data)

results = results.append(res)
model_ada, res = run_model(AdaBoostRegressor( DecisionTreeRegressor(max_depth=100)), labeled_data)

results = results.append(res)
ransac = RANSACRegressor(loss='absolute_loss', max_trials=500, random_state=0)

model_ran, res = run_model(ransac, labeled_data)

results = results.append(res)
model_knn, res = run_model(KNeighborsRegressor(), labeled_data)

results = results.append(res)
model_dtr, res = run_model(DecisionTreeRegressor(), labeled_data)

results = results.append(res)
results
objects = results['Model'].values

y_pos = np.arange(len(objects))

performance1 = results['R2 Score'].astype(float).values

performance2 = results['MAE'].astype(float).values



fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle('Errors')



ax1.set_ylabel('R2 Score')

ax1.set_xlabel('Models')

ax1.set_xticklabels(objects,rotation=40)

ax1.bar(objects, performance1)



ax2.set_ylabel('MAE')

ax2.set_xlabel('Models')

ax2.set_xticklabels(objects,rotation=40)

ax2.bar(objects, performance2)
predictions = model_ada.predict(labeled_test_data)
preds_data = test_data

preds_data['Market Share_total'] = predictions
preds_data.head()
preds_data.to_csv('predictions.csv')
import shutil

shutil.rmtree("catboost_info")