# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/train.csv')

y=data.pop('Survived')

data['Age'].fillna(data['Age'].mean(),inplace=True)

data.describe()
num_var=list(data.dtypes[data.dtypes != 'object'].index)

data[num_var].head()
model=RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

model.fit(data[num_var],y)
model.oob_score_
y_oob=model.oob_prediction_

print ('c-stat:' , roc_auc_score(y,y_oob))
def describe_categorical(data):

    from IPython.display import display,HTML

    display(HTML(data[data.columns[data.dtypes=='object']].describe().to_html()))
describe_categorical(data)
data.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)
def clean_cabin(data):

    try:

        return data[0]

    except TypeError:

        return 'None'

data['Cabin']=data.Cabin.apply(clean_cabin)

         
categorical_variable=['Sex','Cabin','Embarked']

for variable in categorical_variable:

    data[variable].fillna('Missing',inplace=True)

    dummies=pd.get_dummies(data[variable], prefix=variable)

    data=pd.concat([data ,dummies], axis=1)

    data.drop([variable], axis=1, inplace=True)
def printall(data,max_rows=10):

    from IPython.display import display,HTML

    display(HTML(data.to_html(max_rows=max_rows)))

printall(data)
model= RandomForestRegressor(100, oob_score=True,n_jobs=-1, random_state=42)

model.fit(data,y)

print ('c-stat:', roc_auc_score(y,model.oob_prediction_))
model.feature_importances_
feature_importances=pd.Series(model.feature_importances_,index=data.columns)

feature_importances.sort()

feature_importances.plot(kind='barh', figsize=(8,6))
def graph_feature_importances(model, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):

    if autoscale:

        x_scale=model.feature_importances_.max()+headroom

    else:

        x_scale=1

        

    feature_dict=dict(zip(feature_names, model.feature_importances_))

    if summarized_columns:

        for col_name in summarized_columns:

            sum_value=sum(x for i, x in feature_dict.items() if col_name in i)

            #print (sum_value)

            

            keys_to_remove=[i for i in feature_dict.keys() if col_name in i]

            #print (keys_to_remove)

            for i in keys_to_remove:

                feature_dict.pop(i)

            #print (keys_to_remove)

            feature_dict[col_name]=sum_value

            #print (feature_dict[col_name])

            #print (feature_dict[col_name])

    results=pd.Series(feature_dict.values(), index=feature_dict.keys())

    results.sort_values(axis=0)

    

    results.plot(kind='barh',figsize=(width,len(results)/4), xlim=(0,x_scale))

    

graph_feature_importances(model, data.columns, summarized_columns=categorical_variable)

            
def graph_feature_importances(model, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):

    """

    By Mike Bernico 

    """

    if autoscale:

        x_scale = model.feature_importances_.max()+headroom

    else:

        x_scale = 1

        

    feature_dict = dict(zip(feature_names, model.feature_importances_))

    

    if summarized_columns:

        # some dummy columns need to be summarized

        for col_name in summarized_columns:

            sum_value = sum(x for i, x in feature_dict.items() if col_name in i)

            

            keys_to_remove = [i for i in feature_dict.keys() if col_name in i]

            for i in keys_to_remove:

                feature_dict.pop(i)

            feature_dict[col_name] = sum_value

    

    results = pd.Series(feature_dict.values(), index=feature_dict.keys())

    results.sort_values(axis=0)

    results.plot(kind="barh", figsize=(width, len(results)/4), xlim=(0,x_scale))
graph_feature_importances(model, data.columns, summarized_columns=categorical_variable)