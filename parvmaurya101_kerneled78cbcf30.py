import pandas as pd



import matplotlib.pyplot as plt

import numpy as np
data = pd.read_csv('../input/diamonds.csv')
data = data.drop(['Unnamed: 0'], axis=1)

data.head()

#cut, color and clarity seem like categorical attributes
data.info()

#from observation we can see that there are no empty attributes

#hence we won't have to worry about the working strategies to tackle them
data["cut"].value_counts()

#cut indeed is categorical
data["color"].value_counts()

#color is also categorical
data["clarity"].value_counts()

#clarity is also categorical
data.describe()
cond = data["z"]!=0.0

data = data[cond]
data["size"] = (data['x'])*(data['y'])*(data['z'])

data["depth_by_z"] = data['depth']/data['z']

data["table_times_y"] = data["table"]*data['y']

corr = data.corr()

print(corr["price"])

#there is definitely a high correlation between the price of the diamond and the carat 

#attribute and the size of the diamond

#depth_by_z also showing very promissing correlation 

#another interesting and very strong correlation is shown by table_times_y 

data.info()
from pandas.plotting import scatter_matrix



scatter_matrix(data[['price','carat','size', 'x']], alpha=0.05)
import seaborn as sns



sns.heatmap(corr,

           xticklabels = corr.columns.values,

           yticklabels = corr.columns.values,

           annot=True

           ) 
map_categorical = {

                    "cut" : {'Fair': -2, 'Good': -1, 'Very Good':0, 'Premium':1, 'Ideal':2},

                    "clarity" : {'I1': -3, 'SI2': -2, 'SI1':-1, 'VS2':0, 'VS1':1, 'VVS2':2, 'VVS1':3, 'IF':4},

                    "color": {'D':3,'E':2,'F':1,'G':0,'H':-1,'I':-2,'J':-3}

                  }



data.replace(map_categorical, inplace=True) 
data = data.sample(frac=1)

X = data.drop(["price","depth","table"], axis = 1)

y = data["price"]



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state= 42)
X_train.head(3)
y_train.head(3)
X_train.columns
X_train.info()
from sklearn.base import BaseEstimator, TransformerMixin



class DFSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]

    







#class map_cut():

#    def __init__(self, cut):

#        self.cut = cut

#    def fit(self, X, y=None):

#        return self

#    def transform(self,X):

#        X[self.cut] = X[self.cut].map({'Fair': -2, 'Good': -1, 'Very Good':0, 'Premium':1, 'Ideal':2})

#        return X[self.cut]

    

#class map_clarity():

#    def __init__(self,clarity):

#        self.clarity = clarity

#    def fit(self, X, y=None):

#        return self

#    def transform(self,X):

#        X[self.clarity] = X[self.clarity].map({'I1': -3, 'SI2': -2, 'SI1':-1, 'VS2':0, 'VS1':1, 'VVS2':2, 'VVS1':3, 'IF':4})

#        return X[self.clarity]

    

#class map_color():

#    def __init__(self,color):

#        self.color = color

#    def fit(self, X, y=None):

#        return self



#    def transform(self,X):

#        X[self.color] = X[self.color].map({'D':3,'E':2,'F':1,'G':0,'H':-1,'I':-2,'J':-3})

#        return X[self.color] 
num_attributes = list(X_train)



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction import DictVectorizer 



num_pipeline = Pipeline([

    ('selector', DFSelector(num_attributes)),

    ('scaler', StandardScaler())

])



#cut_pipeline = Pipeline([

#    ('selector', DFSelector(['cut'])),

#    ('map_cut', DictVectorizer(sparse=False).fit([cut_map]))

#])



#clarity_pipeline = Pipeline([

#    ('selector',DFSelector(['clarity'])),

#    ('map_clarity', DictVectorizer(sparse = False).fit([clarity_map])),

#])



#color_pipeline = Pipeline([

#    ('selector', DFSelector(['color'])),

#    ('map_color', DictVectorizer(sparse = False).fit([color_map])),

#]) 
from sklearn.pipeline import FeatureUnion



full_pipeline = FeatureUnion(transformer_list=[

    ("num_pipeline", num_pipeline),

    #("cut_pipeline", cut_pipeline)

    #("clarity_pipeline", clarity_pipeline),

    #("color_pipeline", color_pipeline)

])

X_train_prepared = num_pipeline.fit_transform(X_train,y_train)

X_test_prepared = num_pipeline.fit_transform(X_test,y_test)

X_train_prepared
def percent_data_with_given_accuracy(diff, val):

    percent_arr = (abs(diff["Predicted"]-diff["Actual"])/diff["Actual"])*100

    per = 0

    for x in percent_arr:

        if(x<=val):

            per = per+1

    return (per/len(percent_arr))*100
def draw_predictions_actual(test_arr, pred_arr):

    t = np.linspace(0,18500,1000)

    leng = len(pred_arr)

    plt.plot(t,t,c='red')

    plt.scatter(test_arr, pred_arr, alpha=0.05, c='blue')

    plt.xlabel('Actual')

    plt.ylabel('Prediction')

    plt.ylim(0,20000)

    plt.xlim(0,20000)

    plt.show()

    
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



import numpy as np



def get_cross_val_score(model, x, y, folds, scoring_type):

    scores = cross_val_score(model, x, y, cv=folds, scoring = scoring_type)

    val = np.sqrt(-scores)

    print('The root mean square error scores: ', val)

    print('\nThe mean of the rmse error is: ', val.mean())

    print('\nThe standard deviation of the rmse eroor is: ', val.std())
from sklearn.linear_model import LinearRegression



model_lr = LinearRegression()

model_lr.fit(X_train_prepared, y_train) 
get_cross_val_score(model_lr, X_train_prepared, y_train, 10,"neg_mean_squared_error")
pred_lr_test = model_lr.predict(X_test_prepared)



get_cross_val_score(model_lr, X_test_prepared, y_test, 10,"neg_mean_squared_error")



draw_predictions_actual(y_test.values, pred_lr_test)

plt.show()
pred_lr_train = model_lr.predict(X_train_prepared)



import numpy as np



scores_lr = cross_val_score(model_lr, X_train_prepared, pred_lr_train, cv=10, scoring="neg_mean_squared_error")

print(np.sqrt(-scores_lr))



print(np.sqrt(-scores_lr).mean())



draw_predictions_actual(y_train.values, pred_lr_train)

plt.show() 
temp_df = pd.DataFrame({'Actual':y_test.values, 'Predicted': pred_lr_test})

temp_df.head(10) 
from sklearn.linear_model import SGDRegressor



model_sgd = SGDRegressor(random_state=42)

model_sgd.fit(X_train_prepared, y_train)
get_cross_val_score(model_sgd, X_train_prepared, y_train, 5,"neg_mean_squared_error")
get_cross_val_score(model_sgd, X_test_prepared, y_test, 5,"neg_mean_squared_error") 
pred_sgd_train = model_sgd.predict(X_train_prepared)



draw_predictions_actual(y_train.values, pred_sgd_train)

plt.show() 
random_data  = X_train_prepared[0:10,:]

random_label = y_train.iloc[:10]



print('Predictions', model_sgd.predict(random_data))

print('\nActual', list(random_label))
model_sgd.score(X_test_prepared, y_test) 
from sklearn.tree import DecisionTreeRegressor



reg_tree = DecisionTreeRegressor()

reg_tree.fit(X_train_prepared, y_train)
get_cross_val_score(reg_tree, X_train_prepared, y_train, 5,"neg_mean_squared_error")
reg_tree.score(X_test_prepared, y_test) 
pred_tree_train = reg_tree.predict(X_train_prepared)



draw_predictions_actual(y_train.values, pred_tree_train) 
pred_tree_test = reg_tree.predict(X_test_prepared)



draw_predictions_actual(y_test.values, pred_tree_test) 
random_data  = X_test_prepared[0:10,:]

random_label = y_test.iloc[:10]



print('Predictions', reg_tree.predict(random_data))

print('\nActual', list(random_label)) 
from sklearn.linear_model import LinearRegression



regr_linear = LinearRegression()

regr_linear.fit(X_train_prepared, y_train)
regr_linear.score(X_train_prepared,y_train)

#the score is more promising than SVM 
pred_lr = regr_linear.predict(X_test_prepared) 
draw_predictions_actual(y_test.values, pred_lr)

plt.show() 
random_data  = X_test_prepared[0:10,:]

random_label = y_test.iloc[:10]



print('Predictions', reg_tree.predict(random_data))

print('\nActual', list(random_label)) 