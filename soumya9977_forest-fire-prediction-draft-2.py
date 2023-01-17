import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from plotly.offline import iplot

import plotly as py

import plotly.tools as tls

import cufflinks as cf



from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error
pd.pandas.set_option('display.max_columns',None)

pd.pandas.set_option('display.max_rows',None)
cm = sns.light_palette("green", as_cmap=True)
py.offline.init_notebook_mode(connected = True)

cf.go_offline()

cf.set_config_file(theme='solar')

plt.style.use('ggplot')
df = pd.read_csv('../input/forest-fire-prediction/forestfires.csv')

df.head().style.set_properties(**{'background-color': 'black',

                           'color': 'lawngreen',

                           'border-color': 'white'})

# .style.background_gradient(cmap='Reds')



df.info()
numerical_features = [features for features in df.columns if df[features].dtypes != 'O']

print('Number of Numerical variables are: ', len(numerical_features))

print('Numerical features are: ', numerical_features)

df[numerical_features].head().style.background_gradient(cmap=cm)
discrete_feature = [features for features in numerical_features if len(df[features].unique())< 20]

print(f"length of discrete numerical variables are: {len(discrete_feature)}")

print(f"And the discreate features are: {discrete_feature}")

# lets see the head of the data frame consists of discrete numerical values

df[discrete_feature].head().style.background_gradient(cm).highlight_null('green')
df['X'].value_counts()
df['Y'].value_counts()
df['rain'].value_counts()
# lets see the different values in each discreate variables

print(df['X'].value_counts())

print('\n')

print(df['Y'].value_counts())

print('\n')

print(df['rain'].value_counts())
#  lets search for year feature

year_feature = [features for features in numerical_features if 'Yr' in features or 'Year' in features or 'yr' in features or 'year' in features]

print(f"year features are : {year_feature}")
continuous_feature=[features for features in numerical_features if features not in discrete_feature]

print(f"Continuous feature Count {len(continuous_feature)}")

print(f"Continuous feature are: {continuous_feature}")



# lets see the head

df[continuous_feature].head().style.background_gradient(cmap=cm)
categorical_features = [features for features in df.columns if df[features].dtypes =='O']

print(f"Now categorical variables are: {categorical_features}")

print(f"number of categorical variables are: {categorical_features}")



# see the head

# CANT COLOR A CATEGORICAL VARIABLE

df[categorical_features].head()
df['month'].describe()
df['day'].describe()
# lets see the different values in each categorical variables

print(df['month'].value_counts())

print('\n')

print(df['day'].value_counts())

print('\n')
df.describe().style.background_gradient(cmap='Reds')
df['area'].iplot(kind = 'scatter' , mode = 'markers',title="Scatter plot of area",

                            yTitle='area',xTitle = 'id')
df['area'].iplot(title="Line plot of area",

                            yTitle='area',xTitle = 'id')
import plotly.figure_factory as ff

import numpy as np

np.random.seed(1)





x = np.array(df['area'])

hist_data = [x]

group_labels = ['area'] 



fig = ff.create_distplot(hist_data, group_labels)

fig.show()
pd.DataFrame(df["area"]).iplot(kind="histogram", 

                bins=40, 

                theme="solar",

                title="Histogram of area",

                xTitle='area', 

                yTitle='Count',

                asFigure=True)
import plotly.express as px



# df = px.data.tips()

fig = px.violin(df, y="area", box=True, # draw box plot inside the violin

                points='all', # can be 'outliers', or False

               )

fig.show()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



for i in categorical_features:

    df[i]=label_encoder.fit_transform(df[i])
all_fe = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
corr_new_train=df.corr()

plt.figure(figsize=(10,20))

sns.heatmap(corr_new_train[['area']].sort_values(by=['area'],ascending=False).head(60),vmin=-1, cmap='seismic', annot=True)

plt.ylabel('features')

plt.xlabel('Target')

plt.title("Corelation of different fitures with target")

plt.show()
fs1 = ['X','Y','month','FFMC','DMC','DC','temp','area'] 
df_fs1 = df[fs1]

df_fs1.head()
SEED = 42



data = df_fs1.copy()

y = data['area']

x = data.drop(['area'],axis=1)





from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(x,y,test_size = 0.2,random_state = SEED)
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

# from sklearn.metrics import mean_absolute_percentage_error

from sklearn.ensemble import RandomForestRegressor
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]







random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
reg_rf_rscv = RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

random_search_rf = RandomizedSearchCV(reg_rf_rscv, random_grid,n_iter=5, n_jobs=1, cv=5,verbose=2)
random_search_rf.fit(x_train,y_train)
random_search_rf.best_params_
base_model = RandomForestRegressor(n_estimators= 1200,

                                     min_samples_split= 10,

                                     min_samples_leaf= 2,

                                     max_features= 'auto',

                                     max_depth= 20,

                                     bootstrap= True,

                                    random_state = SEED)

base_model.fit(x_train, y_train)

# base_accuracy = evaluate(base_model, x_val,y_val)
y_pred_rf_rscv = base_model.predict(x_val)
def MSE(model_preds, ground_truths):

  return mean_squared_error(model_preds, ground_truths)



def MAE(model_preds, ground_truths):

  return mean_absolute_error(model_preds, ground_truths)



def Other_Err(model_preds, ground_truths):

  return r2_score( ground_truths,model_preds)



def RMSE(model_preds, ground_truths):

  return np.sqrt(mean_squared_error(model_preds, ground_truths))
print(f"mean squared error: {MSE(y_pred_rf_rscv,y_val)}")

print(f"mean absolute error: {MAE(y_pred_rf_rscv,y_val)}")

print(f"r2 error: {Other_Err(y_pred_rf_rscv,y_val)}")

print(f"root mean squared error: {RMSE(y_pred_rf_rscv,y_val)}")
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
score = cross_val_score(reg_rf_rscv, x_train, y_train, cv=k_fold, n_jobs=1, scoring='r2')

print(score)
print('train r2 %2f' %(1 * score.mean()))
score_val = cross_val_score(reg_rf_rscv, x_val, y_val, cv=k_fold, n_jobs=1, scoring='r2')

print(score)
print('train r2 %2f' %(1 * score_val.mean()))
pd.DataFrame(score).iplot(title="R2 score of diferent CV for training data",xTitle = "count",yTitle="R2 Score")
pd.DataFrame(score_val).iplot(title="R2 score of diferent CV for validation data",xTitle = "count",yTitle="R2 Score")
import pickle

filename = 'finalized_model.pkl'

pickle.dump(reg_rf_rscv,open(filename,'wb'))