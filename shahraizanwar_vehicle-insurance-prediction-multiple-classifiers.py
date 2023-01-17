# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go

from sklearn.feature_selection import chi2, SelectKBest

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit
train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')

test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')

sample_sub = pd.read_csv('../input/health-insurance-cross-sell-prediction/sample_submission.csv')



train.head()
train.isnull().any()
print('Length of train data: {}'.format(len(train)))
data = train.copy()



gender_dist = data['Gender'].value_counts()

driveL_dist = data['Driving_License'].value_counts().rename(index={0:'No',1:'Yes'})

prev_insur_dist = data['Previously_Insured'].value_counts().rename(index={0:'No',1:'Yes'})

vehicle_damg_dist = data['Vehicle_Damage'].value_counts().rename(index={0:'No',1:'Yes'})

vehicle_age_dist = data['Vehicle_Age'].value_counts()

response_dist = data['Response'].value_counts().rename(index={0:'No',1:'Yes'})



############



fig = make_subplots(

    rows=6, cols=2,

    subplot_titles=("Gender","Driving License","Previously Insured","Vehicle Damage",

                    "Vehicle Age", "Response", "Annual_Premium", "Age", "Vintage"),

    specs=[[{}, {}],[{}, {}],[{},{}],

           [{"colspan": 2}, None],

          [{"colspan": 2}, None],

          [{"colspan": 2}, None]],

    horizontal_spacing=0.25,

    vertical_spacing=0.075

)



############



fig.add_trace(

    go.Bar(x = gender_dist.index, y = gender_dist.values),

    col=1, row=1

)



fig.add_trace(

    go.Bar(x = driveL_dist.index, y = driveL_dist.values),

    col=2, row=1

)



fig.add_trace(

    go.Bar(x = prev_insur_dist.index, y = prev_insur_dist.values),

    col=1, row=2

)



fig.add_trace(

    go.Bar(x = vehicle_damg_dist.index, y = vehicle_damg_dist.values),

    col=2, row=2

)



fig.add_trace(

    go.Bar(x = response_dist.index, y = response_dist.values),

    col=2, row=3

)



fig.add_trace(

    go.Bar(x = vehicle_age_dist.index, y = vehicle_age_dist.values),

    col=1, row=3

)



fig.add_trace(

    go.Histogram(x = data['Annual_Premium']),

    row=4, col=1

)



fig.add_trace(

    go.Histogram(x = data['Age']),

    row=5, col=1

)



fig.add_trace(

    go.Histogram(x = data['Vintage']),

    row=6, col=1

)





###########



fig.update_layout(

    showlegend=False,

    title_text='Distributions',

    height=2100

)



fig.show()
data = train.query('Response == 1').copy()



gender_dist = data['Gender'].value_counts()

driveL_dist = data['Driving_License'].value_counts().rename(index={0:'No',1:'Yes'})

prev_insur_dist = data['Previously_Insured'].value_counts().rename(index={0:'No',1:'Yes'})

vehicle_damg_dist = data['Vehicle_Damage'].value_counts().rename(index={0:'No',1:'Yes'})

vehicle_age_dist = data['Vehicle_Age'].value_counts()

response_dist = data['Response'].value_counts().rename(index={0:'No',1:'Yes'})



############



fig = make_subplots(

    rows=6, cols=2,

    subplot_titles=("Gender","Driving License","Previously Insured","Vehicle Damage",

                    "Vehicle Age", "Response", "Annual_Premium", "Age", "Vintage"),

    specs=[[{}, {}],[{}, {}],[{},{}],

           [{"colspan": 2}, None],

          [{"colspan": 2}, None],

          [{"colspan": 2}, None]],

    horizontal_spacing=0.25,

    vertical_spacing=0.075

)



############



fig.add_trace(

    go.Bar(x = gender_dist.index, y = gender_dist.values),

    col=1, row=1

)



fig.add_trace(

    go.Bar(x = driveL_dist.index, y = driveL_dist.values),

    col=2, row=1

)



fig.add_trace(

    go.Bar(x = prev_insur_dist.index, y = prev_insur_dist.values),

    col=1, row=2

)



fig.add_trace(

    go.Bar(x = vehicle_damg_dist.index, y = vehicle_damg_dist.values),

    col=2, row=2

)



fig.add_trace(

    go.Bar(x = response_dist.index, y = response_dist.values),

    col=2, row=3

)



fig.add_trace(

    go.Bar(x = vehicle_age_dist.index, y = vehicle_age_dist.values),

    col=1, row=3

)



fig.add_trace(

    go.Histogram(x = data['Annual_Premium']),

    row=4, col=1

)



fig.add_trace(

    go.Histogram(x = data['Age']),

    row=5, col=1

)



fig.add_trace(

    go.Histogram(x = data['Vintage']),

    row=6, col=1

)





###########



fig.update_layout(

    showlegend=False,

    title_text='Distributions',

    height=2100

)



fig.show()
data = train.query('Response == 0').copy()



gender_dist = data['Gender'].value_counts()

driveL_dist = data['Driving_License'].value_counts().rename(index={0:'No',1:'Yes'})

prev_insur_dist = data['Previously_Insured'].value_counts().rename(index={0:'No',1:'Yes'})

vehicle_damg_dist = data['Vehicle_Damage'].value_counts().rename(index={0:'No',1:'Yes'})

vehicle_age_dist = data['Vehicle_Age'].value_counts()

response_dist = data['Response'].value_counts().rename(index={0:'No',1:'Yes'})



############



fig = make_subplots(

    rows=6, cols=2,

    subplot_titles=("Gender","Driving License","Previously Insured","Vehicle Damage",

                    "Vehicle Age", "Response", "Annual_Premium", "Age", "Vintage"),

    specs=[[{}, {}],[{}, {}],[{},{}],

           [{"colspan": 2}, None],

          [{"colspan": 2}, None],

          [{"colspan": 2}, None]],

    horizontal_spacing=0.25,

    vertical_spacing=0.075

)



############



fig.add_trace(

    go.Bar(x = gender_dist.index, y = gender_dist.values),

    col=1, row=1

)



fig.add_trace(

    go.Bar(x = driveL_dist.index, y = driveL_dist.values),

    col=2, row=1

)



fig.add_trace(

    go.Bar(x = prev_insur_dist.index, y = prev_insur_dist.values),

    col=1, row=2

)



fig.add_trace(

    go.Bar(x = vehicle_damg_dist.index, y = vehicle_damg_dist.values),

    col=2, row=2

)



fig.add_trace(

    go.Bar(x = response_dist.index, y = response_dist.values),

    col=2, row=3

)



fig.add_trace(

    go.Bar(x = vehicle_age_dist.index, y = vehicle_age_dist.values),

    col=1, row=3

)



fig.add_trace(

    go.Histogram(x = data['Annual_Premium']),

    row=4, col=1

)



fig.add_trace(

    go.Histogram(x = data['Age']),

    row=5, col=1

)



fig.add_trace(

    go.Histogram(x = data['Vintage']),

    row=6, col=1

)





###########



fig.update_layout(

    showlegend=False,

    title_text='Distributions',

    height=2100

)



fig.show()
def normalize_Data(dfs, Scaler , cols=None):

    train, test = dfs

    

    all_cols = train.columns

    

    if not cols: 

        selected_cols = train.columns

        temp_train = train

        temp_test = test

    else:

        ## Separating columns to be standardize from others

        selected_cols = cols

        temp_train = train[selected_cols].copy()

        temp_test = test[selected_cols].copy()

        train = train.drop(selected_cols,axis=1)

        test = test.drop(selected_cols,axis=1)

    

    

    ## Scaling

    scaler = Scaler()

    

    temp_train = pd.DataFrame(

        scaler.fit_transform(temp_train),

        columns=selected_cols

    )

    

    temp_test = pd.DataFrame(

        scaler.transform(temp_test),

        columns=selected_cols

    )

    

    if not cols:

        return (temp_train, temp_test)

    else:

        return (

            pd.concat([temp_train,train], axis=1)[all_cols],

            pd.concat([temp_test,test], axis=1)[all_cols]

        )



def feature_selector(score_func, k, X, y):

    

    feature_selector = SelectKBest(score_func, k=k)

    feature_selector.fit_transform(X,y)

    features = feature_selector.get_support()

        

    return [b for a, b in zip(features, X.columns) if a]
## Categorical and numeric features respectively

catg_features = ['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage']

numeric_features = ['Age', 'Region_Code','Annual_Premium', 'Policy_Sales_Channel', 'Vintage']



train_X = train.copy()

test_X = test.copy()



del train_X['id']

del test_X['id']



train_y = train_X.pop("Response").values
encodings = {"Vehicle_Age": {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2},

             "Gender": {'Male': 1, 'Female': 0},

             "Vehicle_Damage": {'Yes': 1, 'No': 0}

            }



train_X.replace(encodings, inplace=True)

test_X.replace(encodings, inplace=True)



train_X.head()
train_X, test_X = normalize_Data(

    (train_X, test_X), Scaler=StandardScaler, cols=numeric_features

)



train_X.head()
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score, recall_score, roc_auc_score



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import PassiveAggressiveClassifier



classifiers = [

    KNeighborsClassifier(5),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=125),

    LogisticRegression(),

    PassiveAggressiveClassifier(max_iter=2500),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    GaussianNB()

]



def precision(x,y):

    return precision_score(x,y,zero_division=0)

    

def recall(x,y):

    return recall_score(x,y,zero_division=0)



metrics = [

    accuracy_score,

    log_loss,

    f1_score,

    precision,

    recall,

    roc_auc_score

]
SPLITS = 3



sss = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.2)



report = pd.DataFrame(

    data = 0,

    index = [x.__class__.__name__ for x in classifiers],

    columns = [x.__name__ for x in metrics]

)



k=1



for train_index, test_index in sss.split(train_X.values, train_y):



    print('Fold:{} is running...'.format(k))

    X_train, X_test = train_X.values[train_index], train_X.values[test_index]

    y_train, y_test = train_y[train_index], train_y[test_index]

    

    print('  Classifiers:'.format(k))

    for clf in classifiers:

        print('     {}...'.format(clf.__class__.__name__))

        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)



        for metric in metrics:

            score = metric(y_test, pred)

            report.loc[

                clf.__class__.__name__, metric.__name__

            ] += score       

    

    print()

    k+=1

            

report = report.applymap(lambda x: x/SPLITS)
report