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
!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887



!apt update && apt install -y libsm6 libxext6
from fastai.imports import *

from fastai.structured import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn import metrics
PATH = '../input'
df_raw = pd.read_csv(f'{PATH}/train.csv', low_memory=False)

df_raw.sample(5)
def display_all(df):

    with pd.option_context('display.max_rows',1000):

        with pd.option_context('display.max_columns',1000):

            display(df)

display_all(df_raw.tail().transpose())
df, y, nas = proc_df(df_raw, 'Cover_Type', max_n_cat=6)
df.head()
def add_feats(df):

    df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']

    df['HF2'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])

    df['HR1'] = (df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])

    df['HR2'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])

    df['FR1'] = (df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])

    df['FR2'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])

    df['EV1'] = df.Elevation+df.Vertical_Distance_To_Hydrology

    df['EV2'] = df.Elevation-df.Vertical_Distance_To_Hydrology

    df['Mean_HF1'] = df.HF1/2

    df['Mean_HF2'] = df.HF2/2

    df['Mean_HR1'] = df.HR1/2

    df['Mean_HR2'] = df.HR2/2

    df['Mean_FR1'] = df.FR1/2

    df['Mean_FR2'] = df.FR2/2

    df['Mean_EV1'] = df.EV1/2

    df['Mean_EV2'] = df.EV2/2    

    df['Elevation_Vertical'] = df['Elevation']+df['Vertical_Distance_To_Hydrology']    

    df['Neg_Elevation_Vertical'] = df['Elevation']-df['Vertical_Distance_To_Hydrology']

    

    # Given the horizontal & vertical distance to hydrology, 

    # it will be more intuitive to obtain the euclidean distance: sqrt{(verticaldistance)^2 + (horizontaldistance)^2}    

    df['slope_hyd_sqrt'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5

    df.slope_hyd_sqrt=df.slope_hyd_sqrt.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    

    df['slope_hyd2'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)

    df.slope_hyd2=df.slope_hyd2.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    

    #Mean distance to Amenities 

    df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 

    #Mean Distance to Fire and Water 

    df['Mean_Fire_Hyd1']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2

    df['Mean_Fire_Hyd2']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Roadways) / 2

    

    #Shadiness

    df['Shadiness_morn_noon'] = df.Hillshade_9am/(df.Hillshade_Noon+1)

    df['Shadiness_noon_3pm'] = df.Hillshade_Noon/(df.Hillshade_3pm+1)

    df['Shadiness_morn_3'] = df.Hillshade_9am/(df.Hillshade_3pm+1)

    df['Shadiness_morn_avg'] = (df.Hillshade_9am+df.Hillshade_Noon)/2

    df['Shadiness_afternoon'] = (df.Hillshade_Noon+df.Hillshade_3pm)/2

    df['Shadiness_mean_hillshade'] =  (df['Hillshade_9am']  + df['Hillshade_Noon'] + df['Hillshade_3pm'] ) / 3    

    

    # Shade Difference

    df["Hillshade-9_Noon_diff"] = df["Hillshade_9am"] - df["Hillshade_Noon"]

    df["Hillshade-noon_3pm_diff"] = df["Hillshade_Noon"] - df["Hillshade_3pm"]

    df["Hillshade-9am_3pm_diff"] = df["Hillshade_9am"] - df["Hillshade_3pm"]



    # Mountain Trees

    df["Slope*Elevation"] = df["Slope"] * df["Elevation"]

    # Only some trees can grow on steep montain

    

    ### More features

    df['Neg_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])

    df['Neg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])

    df['Neg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])

    

    df['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])/2

    df['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])/2

    df['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])/2   

        

    df["Vertical_Distance_To_Hydrology"] = abs(df['Vertical_Distance_To_Hydrology'])

    

    df['Neg_Elev_Hyd'] = df.Elevation-df.Horizontal_Distance_To_Hydrology*0.2

    

    # Bin Features

    bin_defs = [

        # col name, bin size, new name

        ('Elevation', 200, 'Binned_Elevation'), # Elevation is different in train vs. test!?

        ('Aspect', 45, 'Binned_Aspect'),

        ('Slope', 6, 'Binned_Slope'),

        ('Horizontal_Distance_To_Hydrology', 140, 'Binned_Horizontal_Distance_To_Hydrology'),

        ('Horizontal_Distance_To_Roadways', 712, 'Binned_Horizontal_Distance_To_Roadways'),

        ('Hillshade_9am', 32, 'Binned_Hillshade_9am'),

        ('Hillshade_Noon', 32, 'Binned_Hillshade_Noon'),

        ('Hillshade_3pm', 32, 'Binned_Hillshade_3pm'),

        ('Horizontal_Distance_To_Fire_Points', 717, 'Binned_Horizontal_Distance_To_Fire_Points')

    ]

    

    for col_name, bin_size, new_name in bin_defs:

        df[new_name] = np.floor(df[col_name]/bin_size)

        

    print('Total number of features : %d' % (df.shape)[1])

    return df
df = add_feats(df)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid=1512

n_trn=len(df)-n_valid







X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)





X_train.shape, X_valid.shape, y_train.shape
#df_test = add_feats(df_test)
display_all(df.tail().transpose())
def print_score(m):

    res = [m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_train[:10]
m=RandomForestClassifier(n_jobs=-1, n_estimators=80, bootstrap=True, max_features=0.5, min_samples_leaf=10, oob_score=True, random_state=1)

%time m.fit(X_train, y_train)

print_score(m)
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(10,16), legend=False)
fi = rf_feat_importance(m, df)
plot_fi(fi)
to_keep = fi[fi.imp > 0.005].cols
to_keep
df_keep = df[to_keep].copy()
df_keep.columns
X_train, X_valid = split_vals(df_keep, n_trn)

#X_train = sc.fit_transform(X_train)

X_train.shape, X_valid.shape, y_train.shape
m=RandomForestClassifier(n_jobs=-1, n_estimators=80, bootstrap=True, max_features=0.5, min_samples_leaf=10, oob_score=True, random_state=1)

%time m.fit(X_train, y_train)

print_score(m)
from scipy.cluster import hierarchy as hc
corr=np.round(scipy.stats.spearmanr(df_keep).correlation,4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(20,16))

dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='top', leaf_font_size=16)
to_keep
df_keep=df[to_keep]
df_keep.sample()
to_drop = ['Id','Mean_EV2','slope_hyd_sqrt','HF1', 'Mean_HF1', 'slope_hyd2','EV2','HR1','HF2','HR2','FR2']
df_keep.drop(columns=to_drop, axis=1, inplace=True)
df_ext = df_keep.copy()
df_ext['is_valid'] = 1
df_ext[:n_trn] = 0
x, y, nas = proc_df(df_ext, 'is_valid')
m=RandomForestClassifier(n_jobs=-1, n_estimators=80, bootstrap=True, oob_score=True, random_state=1)

%time m.fit(x, y)

m.oob_score_
rf = rf_feat_importance(m, x); rf[rf.imp > 0.03]
X_train, X_valid = split_vals(df_keep, n_trn)

m=RandomForestClassifier(n_jobs=-1, n_estimators=80, bootstrap=True, oob_score=True, random_state=1)

%time m.fit(X_train, y_train)

print_score(m)
feats = rf[rf.imp > 0.03]['cols']
for f in feats:

    df_subs = df_keep.drop(columns=f, axis=1)

    X_train, X_valid = split_vals(df_subs, n_trn)

    m=RandomForestClassifier(n_jobs=-1, n_estimators=80, bootstrap=True, oob_score=True, random_state=1)

    %time m.fit(X_train, y_train)

    print(f)

    print_score(m)
X_train, X_valid = split_vals(df_keep, n_trn)



X_train.shape, X_valid.shape, y_train.shape
#X_train = sc.fit_transform(X_train)
m=RandomForestClassifier(n_jobs=-1, n_estimators=480, bootstrap=True, oob_score=True, random_state=1)

%time m.fit(X_train, y_train)

print_score(m)
df_test = pd.read_csv(f'{PATH}/test.csv', low_memory=False)
df_test = add_feats(df_test)
X_test = df_test[to_keep].copy()
#df_test = proc_df(test, max_n_cat=6, do_scale=True, mapper=mapper, na_dict=nas)
#X_test = test[to_keep].copy()
X_test.drop(columns=to_drop, axis=1, inplace=True)


test = sc.transform(X_test)
%time preds = m.predict(X_test)
preds
#df_test.head()
out = pd.DataFrame()

out['ID'] = df_test['Id'].copy()

out['Cover_Type'] = preds

out.to_csv('my_submission.csv', index=False)

out.head(5)