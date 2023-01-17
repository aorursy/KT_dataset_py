# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing libraries

import pandas as pd

import numpy as np
#load the training data set

planet_data=pd.read_csv("../input/soil-sample/soil_samples_training.csv")

planet_data.head()
planet=planet_data.copy()
planet.shape
planet.info()
#check the 'object' type columns

print(planet['origin'].value_counts())

print(planet['grain_shape'].value_counts())

print(planet['grain_color'].value_counts())

print(planet['particle_attached'].value_counts())

print(planet['particle_spacing'].value_counts())

print(planet['particle_width'].value_counts())

print(planet['particle_color'].value_counts())

print(planet['grain_color'].value_counts())

print(planet['particle_color'].value_counts())

print(planet['particle_distribution'].value_counts())

print(planet['organics'].value_counts())

print(planet['solubles'].value_counts())

print(planet['isotope_diversity'].value_counts())
planet['particle_width'].isnull().sum()
planet.columns


planet.describe()
planet.isnull().sum().sort_values(ascending=False)
#check for duplicates

planet[planet.duplicated()]
#building histogram to check how the samples are spread 

%matplotlib inline

import matplotlib.pyplot as plt

planet.hist(bins=50, figsize=(20,15))

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
features=planet.columns.values

features






mask = np.zeros_like(planet[features].corr(), dtype = np.bool) 

mask[np.triu_indices_from(mask)] = True 



f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix',fontsize=25)



sns.heatmap(planet[features].corr(),linewidths = 0.25,vmax = 0.7,square = True,cmap = "BuGn", 

            linecolor = 'w',annot = True,annot_kws = {"size":8},mask = mask,cbar_kws = {"shrink": 0.9})

planet['origin']


sns.boxplot( x=planet["origin"], y=planet["pH"], width = 0.5, notch = True)
sns.boxplot( x=planet["origin"], y=planet["optical_density"], width = 0.5, notch = True)
sns.boxplot( x=planet["origin"], y=planet["chlorate"], width = 0.5, notch = True)
sns.boxplot( x=planet["origin"], y=planet["chloride"], width = 0.5, notch = True)
sns.boxplot( x=planet["origin"], y=planet["nitrate"], width = 0.5, notch = True)
sns.boxplot( x=planet["origin"], y=planet["nitrite"], width = 0.5, notch = True)
sns.boxplot( x=planet["origin"], y=planet["sulphate"], width = 0.5, notch = True)
sns.boxplot( x=planet["origin"], y=planet["sulphite"], width = 0.5, notch = True)
sns.boxplot( x=planet["origin"], y=planet["phosphate"], width = 0.5, notch = True)
sns.boxplot( x=planet["origin"], y=planet["radioactivity"], width = 0.5, notch = True)
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(planet, test_size=0.2, random_state=42)
train_set.shape
test_set.shape
features
train_new=train_set
train_new.shape
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['grain_shape'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['grain_shape'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['grain_shape'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['grain_shape'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['grain_color'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['grain_color'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['grain_color'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['grain_color'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['grain_surface'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['grain_surface'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['grain_surface'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['grain_surface'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['particle_attached'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['particle_attached'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['particle_attached'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['particle_attached'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['particle_spacing'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['particle_spacing'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['particle_spacing'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['particle_spacing'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['particle_width'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['particle_width'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['particle_width'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['particle_width'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['particle_color'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['particle_color'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['particle_color'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['particle_color'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['particle_distribution'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['particle_distribution'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['particle_distribution'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['particle_distribution'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['organics'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['organics'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['organics'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['organics'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['solubles'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['solubles'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['solubles'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['solubles'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
train_new[train_new['origin']=='earth'].groupby(['solubles'])['sample_id'].agg(np.size).values
55/(49+55+2389)
train_new[train_new['origin']=='alien'].groupby(['solubles'])['sample_id'].agg(np.size).values
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='alien'].groupby(['isotope_diversity'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='alien'].groupby(['isotope_diversity'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
from matplotlib.pyplot import figure

figure(num=None, figsize=(12,4))

y = train_new[train_new['origin']=='earth'].groupby(['isotope_diversity'])['sample_id'].agg(np.size).values

x = train_new[train_new['origin']=='earth'].groupby(['isotope_diversity'])['sample_id'].agg(np.size).index

sns.barplot(x=x, y=y, data=train_new)
train_new_final=train_new
train_new_final.shape
train_new_final.isnull().sum()
train_set1 = train_new_final.drop("origin", axis=1) # drop labels for training set

train_labels1 = train_new_final["origin"].copy()
train_labels=pd.DataFrame(train_labels1, columns=['origin'])
train_labels
train_set1.shape
#train_final=train_set1.loc[~train_new_final['grain_surface'].isnull()]

#train_final=train_final.loc[~train_new_final['particle_width'].isnull()]

#df_p1 = train.loc[~train['handpump_age'].isnull()]

train_set1['grain_surface']=train_set1['grain_surface'].ffill(axis=0)

train_set1['particle_width']=train_set1['particle_width'].ffill(axis=0)
train_final=train_set1
train_final.shape
train_final=train_final.drop(['chlorate','phosphate','particle_attached','solubles'],axis=1)
train_final.shape
train_final.info()
num_attributes=['grain_shape','grain_surface','grain_color','particle_spacing','particle_width','particle_color',

           'particle_distribution','organics','isotope_diversity']

train_final_num = train_final.drop(num_attributes, axis=1)
train_final_num.shape
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

imputer.fit(train_final_num)
imputer.statistics_
X = imputer.transform(train_final_num)
train_final_num_tr = pd.DataFrame(X, columns=train_final_num.columns,

                         index=train_final_num.index)
train_final_num_tr.shape
train_final_num_tr.isnull().sum()
train_final.isnull().sum()
#train_final_cat=train_final-train_final_num
#train_final=train_set1.loc[~train_new_final['grain_surface'].isnull()]

#df_final['permit'] = df_final['permit'].map({'True': 1, 'False': 0, 'N/A': 0})
cat_attributes=['sample_id','optical_density','pH','chloride','nitrate','nitrite',

           'sulphate','sulphite','radioactivity']

#cat_attributes=train_final.select_dtypes(include=['object']).columns

train_final_cat= train_final.drop(cat_attributes, axis=1)
train_final_cat['grain_shape'].value_counts()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_final_cat = train_final_cat.apply(LabelEncoder().fit_transform)

train_final_cat

le.fit_transform(train_final_cat['particle_width'].values)
train_final_cat['particle_width']= le.fit_transform(train_final_cat['particle_width'].values)

data = train_final_cat.drop_duplicates('particle_width')

print(data)
train_final=pd.concat([train_final_num_tr,train_final_cat],axis=1)
train_final
train_final.shape
train_final.isnull().sum()
#type(train_final_prepared)

train_labels.columns
#df_final['permit'] = df_final['permit'].map({'True': 1, 'False': 0, 'N/A': 0})

train_labels['origin']=train_labels['origin'].map({'earth': 1, 'alien': 0})
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import fbeta_score
model = RandomForestClassifier()
test_set
test_set_x = test_set.drop("origin", axis=1) # drop labels for training set

test_set_labels = test_set["origin"].copy()
test_set_labels=pd.DataFrame(test_set_labels, columns=['origin'])
test_set_x
model.fit(train_final,train_labels)
train_final
test=pd.read_csv("../input/soil-sample/soil_samples_test.csv")

test.head()
type(test)
# prepare the test data

def fit_the_data(test_set_x):

    test_set_x['grain_surface']=test_set_x['grain_surface'].ffill(axis=0)

    test_set_x['particle_width']=test_set_x['particle_width'].ffill(axis=0)

    test_final=test_set_x

    test_final=test_final.drop(['chlorate','phosphate','particle_attached','solubles'],axis=1)



    num_attributes=['grain_shape','grain_surface','grain_color','particle_spacing','particle_width','particle_color',

               'particle_distribution','organics','isotope_diversity']

    test_final_num = test_final.drop(num_attributes, axis=1)



    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="mean")

    imputer.fit(test_final_num)



    X = imputer.transform(test_final_num)

    test_final_num_tr = pd.DataFrame(X, columns=test_final_num.columns,

                             index=test_final_num.index)



    cat_attributes=['sample_id','optical_density','pH','chloride','nitrate','nitrite',

               'sulphate','sulphite','radioactivity']

    test_final_cat= test_final.drop(cat_attributes, axis=1)





    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    test_final_cat = test_final_cat.apply(LabelEncoder().fit_transform)





    test_final=pd.concat([test_final_num_tr,test_final_cat],axis=1)



  

    

    return test_final
#test_set_labels['origin']=test_set_labels['origin'].map({'earth': 1, 'alien': 0})
Y_test=fit_the_data(test)
test
Y_test
y_pred = model.predict(Y_test)
type(y_pred)
Y_test
y_pred
y_ans=pd.DataFrame(y_pred,columns=['origin'])
y_ans
ans=pd.concat([Y_test,y_ans],axis=1)
ans
print(y_pred)

ans['origin']=ans['origin'].map({1:'earth',0:'alien'})
ans
final_ans=pd.DataFrame(ans['sample_id'],columns=['sample_id'],dtype=int)

final_ans['predicted_origin']=ans['origin']

final_ans
final_ans.to_csv("mahima_lalwani.csv",index=False)
#from sklearn.metrics import accuracy_score

#from sklearn.metrics import fbeta_score
print((accuracy_score(test_set_labels, y_pred)))

print(fbeta_score(test_set_labels, y_pred, beta = 1))
#from sklearn.metrics import make_scorer,classification_report,confusion_matrix

#from sklearn.model_selection import GridSearchCV,ShuffleSplit
def fit_model(X, y):

    """ Performs grid search over the 'max_depth' parameter for a 

        decision tree regressor trained on the input data [X, y]. """

    

    # Create cross-validation sets from the training data

    cv_sets = ShuffleSplit(n_splits=10, random_state=0, test_size=0.2, train_size=None)



    # Create a random forest classifier object

    model = RandomForestClassifier()



    # Create a dictionary for the parameters 'max_depth',min_samples_split and min_samples_leaf

    params = {'max_depth':range(2,12,2),

              'min_samples_split':range(2,12,2),

              'min_samples_leaf':range(2,12,2)}



    # Transform 'performance_metric' into a scoring function using 'make_scorer' 

    scoring_fnc = make_scorer(fbeta_score, beta = 1)



    # Create the grid search cv object --> GridSearchCV()

    grid = GridSearchCV(model, params, scoring_fnc, cv = cv_sets)



    # Fit the grid search object to the data to compute the optimal model

    grid = grid.fit(X, y)



    # Return the optimal model after fitting the data

    return grid.best_estimator_
#reg = fit_model(train_final,train_labels)

#reg.score

#rf_clf = RandomForestClassifier(max_depth = 10, min_samples_split = 8, min_samples_leaf = 2)

#rf_clf.fit(train_final,train_labels)

#y_pred = rf_clf.predict(Y_test)