import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import sklearn.metrics as metrics
from tpot import TPOTRegressor
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("../input/automobileprice/datasets_1291_2355_Automobile_data.csv")
data.info()

dtypes = pd.DataFrame([list(data.columns),list(data.dtypes)]).T
dtypes.columns = ["column","type"]
enccol = []
for index,row in dtypes.iterrows():
    if row["type"] == "object":
        enccol.append(row["column"])
data[enccol] = data[enccol].astype("string")
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
data.dtypes
data = MultiColumnLabelEncoder(columns = enccol).fit_transform(data)
data.dtypes
data = data.drop("normalized-losses",axis = 1)
data.info()
data.shape
X_train, X_test, y_train, y_test = train_test_split(data.drop("price",axis = 1), data["price"], test_size=0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
def evaluate(model,X_test,y_test):
    pred = model.predict(X_test)
    explained_variance_score = metrics.explained_variance_score(y_test,pred)
    max_error = metrics.max_error(y_test,pred)
    mean_absolute_error = metrics.mean_absolute_error(y_test,pred)
    mean_squared_error = metrics.mean_squared_error(y_test,pred)
    root_mean_squared_error = metrics.mean_squared_error(y_test,pred)**0.5
    r2_score = metrics.r2_score(y_test,pred)
    print("explained_variance_score: ", explained_variance_score)
    print("\nmax_error: ", max_error)
    print("\nmean_absolute_error: ", mean_absolute_error)
    print("\nmean_squared_error: ", mean_squared_error)
    print("\nroot_mean_squared_error: ", root_mean_squared_error)
    print("\nr2_score: ", r2_score)
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
pipeline_optimizer = TPOTRegressor(generations=100, population_size=50, cv=5,
                                    random_state=42, verbosity=2, max_time_mins = 20)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
evaluate(pipeline_optimizer,X_test,y_test)