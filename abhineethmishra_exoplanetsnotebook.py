# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
alldf=pd.read_csv('../input/exoooo/phl_hec_all_confirmed.csv')
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
alldf.head()
#Creating a feature set consisting of useful features. Dropping irrelevant data

feature_df=alldf.drop(columns=['P. Name','P. Name Kepler','P. Disc. Year','S. Name HD','S. Name HIP','P. Disc. Method','P. Min Mass (EU)','P. Max Mass (EU)','P. Name KOI','P. SFlux Min (EU)','P. SFlux Max (EU)','P. Habitable Class','P. Teq Min (K)','P. Teq Max (K)','P. Ts Min (K)','P. Ts Max (K)'])

feature_df.head()
feature_df.describe()
feature_df.corr()['P. Habitable'].sort_values(ascending=False)
#remove gaseous planets

feature_df=feature_df[feature_df['P. Composition Class']!='gas']

feature_df=feature_df.reset_index()

feature_df.head()
#keeping only rocky planets

count=0

for idx,val in enumerate(feature_df['P. Composition Class']):

    if 'rocky' in str(val):

        count+=1

    else:

        feature_df=feature_df.drop(idx,axis=0)

feature_df.describe()
#removing planets with 0 mass

for idx,i in enumerate(feature_df['P. Mass (EU)']):

    if i==0:

        feature_df=feature_df.drop(idx,axis=0)

#removing unnecassary features

feature_df=feature_df.drop(columns=['index','S. HabCat','P. Hab Moon','S. [Fe/H]','S. Age (Gyrs)','S. No. Planets HZ','S. Distance (pc)','P. Inclination (deg)','S. Appar Mag'])
feature_df.head()
feature_df.describe()
import math

for idx,i in enumerate(feature_df['P. Teq Mean (K)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['P. Ts Mean (K)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['P. Mag']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['P. Period (days)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['P. Sem Major Axis (AU)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['P. Mean Distance (AU)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['S. Mass (SU)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['S. Radius (SU)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['S. Teff (K)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['S. Luminosity (SU)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['S. Mag from Planet']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['S. Size from Planet (deg)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['S. Hab Zone Min (AU)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['S. Hab Zone Max (AU)']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['P. HZD']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['P. HZA']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['P. HZI']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')

for idx,i in enumerate(feature_df['P. ESI']):

    if math.isnan(i)==True:

        feature_df=feature_df.drop(idx,axis=0)

feature_df=feature_df.reset_index().drop(columns='index')
X=feature_df.drop(columns=['P. Habitable'])

y=feature_df['P. Habitable']

X=X.reset_index().drop(columns='index')

y=y.reset_index().drop(columns='index')
#encoding text attributes

from sklearn.preprocessing import LabelEncoder

#zone class

lb=LabelEncoder()

X['P. Zone Class']=(lb.fit_transform(X['P. Zone Class'].astype(str)))

#composition class

composition=LabelEncoder()

X['P. Composition Class']=composition.fit_transform(X['P. Composition Class'].astype(str))

#atmosphere

atmosphere=LabelEncoder()

X['P. Atmosphere Class']=atmosphere.fit_transform(X['P. Atmosphere Class'].astype(str))

#mass class

mclass=LabelEncoder()

X['P. Mass Class']=mclass.fit_transform(X['P. Mass Class'].astype(str))

#star name

sname=LabelEncoder()

X['S. Name']=sname.fit_transform(X['S. Name'].astype(str))

#constellation

cons=LabelEncoder()

X['S. Constellation']=cons.fit_transform(X['S. Constellation'].astype(str))

#stype

stype=LabelEncoder()

X['S. Type']=stype.fit_transform(X['S. Type'].astype(str))
X.describe()
X.head()
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN',strategy='median',axis=0)  

altX=X.copy()

imputer = imputer.fit(altX)

altX=pd.DataFrame(imputer.transform(altX))

#altX.describe()

imputer = Imputer(missing_values=0,strategy='median',axis=0)  

imputer = imputer.fit(altX)

impX=pd.DataFrame(imputer.transform(altX))

#altX.describe()

impX=pd.DataFrame(altX,columns=X.columns)

impX.describe()
'''from sklearn.base import TransformerMixin



class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with mean of column.



        """

    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)



data = [

    ['a', 1, 2],

    ['b', 1, 1],

    ['b', 2, 2],

    [np.nan, np.nan, np.nan]

]



DataFrameImputer().fit_transform(X).head()'''
import math

for i in X['P. Teq Mean (K)']:

    if math.isnan(i)==True:

        print(i)
X.describe()
X.head()
#splitting train and test sets



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#from sklearn.cluster import KMeans

#kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train,y_train)
#type(X['S. Name'].iloc[0])
count=0

for i in X['P. Teq Mean (K)']:

    if isinstance(i, float)==True:

        #print(i)

        count+=1

    #if(count>5)

count
X.describe()
import math

for i,v in enumerate(X['P. Mass Class']):

    if math.isnan(i)==True:

        X=X.drop(i,axis=0)
X.describe()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(random_state=42)

clf.fit(X_train,np.array(y_train))

score = accuracy_score(y_test,clf.predict(X_test))

score
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train,y_train)