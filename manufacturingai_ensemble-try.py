%load_ext autoreload

%autoreload 2



%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestClassifier

from IPython.display import display

from sklearn import metrics
PATH = '../input'



!ls {PATH}
df_raw = pd.read_csv(f'{PATH}/train.csv', low_memory=False)
def display_all(df):

    with pd.option_context('display.max_rows',1000):

        with pd.option_context('display.max_columns', 1000):

            display(df);

            

display_all(df_raw.tail().transpose())
df_raw.info()
def feature_engineering(df):

    df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']

    df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])

    df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])

    df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])

    df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])

    df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])

    df['ele_vert'] = df.Elevation-df.Vertical_Distance_To_Hydrology



    df['slope_hyd'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5

    df.slope_hyd=df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any



    #Mean distance to Amenities 

    df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 

    #Mean Distance to Fire and Water 

    df['Mean_Fire_Hyd']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2 

    

    df['Neg_Elevation_Vertical'] = df['Elevation']-df['Vertical_Distance_To_Hydrology']

    df['Elevation_Vertical'] = df['Elevation']+df['Vertical_Distance_To_Hydrology']



    df['mean_hillshade'] =  (df['Hillshade_9am']  + df['Hillshade_Noon'] + df['Hillshade_3pm'] ) / 3



    df['Mean_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points'])/2

    df['Mean_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])/2

    df['Mean_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])/2



    df['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])/2

    df['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])/2

    df['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])/2



    df['Slope2'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)

    df['Mean_Fire_Hydrology_Roadways']=(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways']) / 3

    df['Mean_Fire_Hyd']=(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Hydrology']) / 2 



    df["Vertical_Distance_To_Hydrology"] = abs(df['Vertical_Distance_To_Hydrology'])



    df['Neg_EHyd'] = df.Elevation-df.Horizontal_Distance_To_Hydrology*0.2

    

    return df
df_raw = feature_engineering(df_raw)
import matplotlib.pyplot as plt

df_raw.hist(bins=50, figsize=(24,20))

plt.show()
df_raw.info()
y = df_raw['Cover_Type']

df = df_raw.drop(columns='Cover_Type', axis=1)
from sklearn.model_selection  import train_test_split



X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=32)
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression
rand_forest_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features=0.5, random_state=42)

extra_tree_clf=ExtraTreesClassifier(n_estimators=100, min_samples_leaf=1, max_features=0.5, bootstrap=True, random_state=42)

svm_clf = LinearSVC(random_state=42)

log_reg_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
estimators = [rand_forest_clf, extra_tree_clf, svm_clf, log_reg_clf]



for estimator in estimators:

    print ("Training estimator: ", estimator)

    estimator.fit(X_train, y_train)
[estimator.score(X_val, y_val) for estimator in estimators ]
from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone
df_test = pd.read_csv(f'{PATH}/test.csv')
df_test = feature_engineering(df_test)
#score1, score2, score3, score4
preds1 = rand_forest_clf.predict(df);

preds2= extra_tree_clf.predict(df);

preds3 = svm_clf.predict(df);

preds4 = log_reg_clf.predict(df);
X_final = pd.DataFrame({'rf':preds1,'et':preds2,'svm':preds3,'log_reg':preds4})

X_final.head(5)
ID_test = df_test.Id
X_final.tail(), y.tail()
preds1 = rand_forest_clf.predict(df_test);

preds2= extra_tree_clf.predict(df_test);

preds3 = svm_clf.predict(df_test);

preds4= log_reg_clf.predict(df_test);
X_test_final = pd.DataFrame({'rf':preds1,'et':preds2,'svm':preds3,'log_reg':preds4})

X_test_final.head(5)
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

from xgboost import XGBClassifier
rf = RandomForestClassifier(n_estimators = 700)

et=ExtraTreesClassifier(n_estimators=700)

svm = LinearSVC(random_state=42)

log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)

xgb = XGBClassifier()
voting_clf = VotingClassifier(estimators=[('rf', rf),('et', et), ('lr', log_reg),('xgb', xgb)], voting='soft')



voting_clf.fit(X_final, y)

preds = voting_clf.predict(X_test_final)
submission = pd.DataFrame({

    "ID": df_test.Id,

    "Cover_Type": preds

})

submission.to_csv('my_submission.csv', index=False)
submission.head()