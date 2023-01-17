import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

import pandas_profiling as pp

from sklearn.compose import ColumnTransformer

import  tensorflow as tf

import keras

from tpot import TPOTClassifier
df = pd.read_csv('../input/learn-together/train.csv')

df.head()
df_test = pd.read_csv('../input/learn-together/test.csv')

df_test.head()
print('Train size: ',df.shape)

print('Test size: ', df_test.shape)
df.info()
df.isnull().mean()
df.describe().T
colormap = plt.cm.RdBu

plt.figure(figsize=(50,35))

plt.title('Pearson Correlation of Features', y=1.05, size=50)

sns.heatmap(df.corr(),linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
df.duplicated().sum() #No Duplicate Data
## Checking if we have a balance dataset

target = df.Cover_Type.value_counts()

sns.countplot(x='Cover_Type', data=df)

plt.title('Class Distribution');

print(target) # Balanced Train Dataset.
pp.ProfileReport(df)
#Soil_Type7 and  Soil_Type15 have zero values, so removing them. 

df.drop(['Soil_Type7', 'Id','Soil_Type15'], axis=1, inplace=True)
df.Cover_Type.value_counts()
# separate intro train and test set



X_train, X_test, y_train, y_test = train_test_split(

    df.drop(['Cover_Type'], axis=1),  # just the features

    df['Cover_Type'],  # the target

    test_size=0.2,  # the percentage of obs in the test set

    random_state=42)  # for reproducibility



X_train.shape, X_test.shape
plt.figure(figsize=(10,10))

plt.scatter(y=df.Hillshade_9am, x=df.Hillshade_3pm)

plt.xlabel("Hillshade_3pm")

plt.ylabel("Hillshade_9am")

plt.title("Hillshade_3pm VS Hillshade_9am")
plt.figure(figsize=(10,10))

plt.scatter(y=df.Hillshade_Noon, x=df.Slope)

plt.xlabel("Slope")

plt.ylabel("Hillshade_Noon")

plt.title("Slope VS Hillshade_Noon")
plt.figure(figsize=(10,10))

plt.scatter(x=df.Hillshade_9am, y=df.Aspect)

plt.ylabel("Aspect")

plt.xlabel("Hillshade_9am")

plt.title("Aspect VS Hillshade_9am")
plt.figure(figsize=(10,10))

plt.scatter(x=df.Hillshade_3pm, y=df.Aspect)

plt.ylabel("Aspect")

plt.xlabel("Hillshade_3pm")

plt.title("Aspect VS Hillshade_3pm")
plt.figure(figsize=(10,10))

plt.scatter(x=df.Vertical_Distance_To_Hydrology, y=df.Horizontal_Distance_To_Hydrology)

plt.ylabel("Horizontal_Distance_To_Hydrology")

plt.xlabel("Vertical_Distance_To_Hydrology")

plt.title("Horizontal_Distance_To_Hydrology VS Vertical_Distance_To_Hydrology")
plt.figure(figsize=(12,12))

cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',

       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points', 'Cover_Type']

sns.pairplot(df[cols][df.Cover_Type==4])
#tpot = TPOTClassifier(generations=5,population_size=10,verbosity=2, n_jobs=-1)

#tpot.fit(X_train, y_train)

#print(tpot.score(X_test, y_test))

#print(tpot.score(X_train, y_train))

#tpot.export('tpot_tree_classification_pipeline.py')

#!cat tpot_tree_classification_pipeline.py

#tpot.evaluated_individuals_

#tpot.fitted_pipeline_

#print(classification_report(y_test, tpot.predict(X_test)))

#print(confusion_matrix(y_test, tpot.predict(X_test)))
import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline, make_union

from tpot.builtins import StackingEstimator





# Average CV score on the training set was:0.8458153791211421

exported_pipeline = make_pipeline(

    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.5, max_depth=9, max_features=0.25, min_samples_leaf=17, min_samples_split=6, n_estimators=100, subsample=0.8)),

    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.25, min_samples_leaf=18, min_samples_split=17, n_estimators=100)

)



exported_pipeline.fit(X_train, y_train)

results = exported_pipeline.predict(X_test)
print(exported_pipeline.score(X_test, y_test))

print(exported_pipeline.score(X_train, y_train))
print(classification_report(y_test, exported_pipeline.predict(X_test)))

print(confusion_matrix(y_test, exported_pipeline.predict(X_test)))
result_final = exported_pipeline.predict(df_test.drop(['Soil_Type7', 'Id', 'Soil_Type15'], axis=1))

result_final_proba = exported_pipeline.predict_proba(df_test.drop(['Soil_Type7', 'Id', 'Soil_Type15'], axis=1))

#df_test.drop(['Soil_Type7', 'Id', 'Soil_Type15'], axis=1, inplace=True)
result_final_proba[0]
# Save test predictions to file

#output = pd.DataFrame({'ID': df_test.Id,

#                       'Cover_Type': result_final})

#output.to_csv('submission.csv', index=False)
#pd.DataFrame(output).iloc[0]
#result_final_proba[0]
import seaborn as sns

cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points', 'Cover_Type']

sns.pairplot(df[cols], hue="Cover_Type")
train = df.copy()

del df
test = df_test.copy()

del df_test
# train.head()

train['HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])

train['Neg_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])

train['HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])

train['Neg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])

train['HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])

train['Neg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])



train['Neg_Elevation_Vertical'] = train['Elevation']-train['Vertical_Distance_To_Hydrology']

train['Elevation_Vertical'] = train['Elevation']+train['Vertical_Distance_To_Hydrology']



train['mean_hillshade'] =  (train['Hillshade_9am']  + train['Hillshade_Noon'] + train['Hillshade_3pm'] )



train['Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])

train['Mean_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])

train['Mean_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])



train['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])

train['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])

train['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])



train['Slope2'] = np.sqrt(train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)

train['Mean_Fire_Hydrology_Roadways']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology'] + train['Horizontal_Distance_To_Roadways'])

train['Mean_Fire_Hyd']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology'])



train["Vertical_Distance_To_Hydrology"] = abs(train['Vertical_Distance_To_Hydrology'])



train['Neg_EHyd'] = train.Elevation-train.Horizontal_Distance_To_Hydrology





test['HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])

test['Neg_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])

test['HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])

test['Neg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])

test['HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])

test['Neg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])



test['Neg_Elevation_Vertical'] = test['Elevation']-test['Vertical_Distance_To_Hydrology']

test['Elevation_Vertical'] = test['Elevation'] + test['Vertical_Distance_To_Hydrology']



test['mean_hillshade'] = (test['Hillshade_9am']  + test['Hillshade_Noon']  + test['Hillshade_3pm'] )



test['Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])

test['Mean_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])

test['Mean_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])



test['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])

test['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])

test['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])



test['Slope2'] = np.sqrt(test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)

test['Mean_Fire_Hydrology_Roadways']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology'] + test['Horizontal_Distance_To_Roadways'])

test['Mean_Fire_Hyd']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology'])





test['Vertical_Distance_To_Hydrology'] = abs(test["Vertical_Distance_To_Hydrology"])



test['Neg_EHyd'] = test.Elevation-test.Horizontal_Distance_To_Hydrology
train.head()
from sklearn.model_selection import train_test_split

x = train.drop(['Cover_Type'], axis = 1)



y = train['Cover_Type']

print( y.head() )



x_train, x_test, y_train, y_test = train_test_split( x.values, y.values, test_size=0.05, random_state=42 )
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

    

from sklearn import decomposition



scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline, make_union

from tpot.builtins import StackingEstimator





# Average CV score on the training set was:0.8458153791211421

exported_pipeline = make_pipeline(

    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.5, max_depth=9, max_features=0.25, min_samples_leaf=17, min_samples_split=6, n_estimators=100, subsample=0.8)),

    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.25, min_samples_leaf=18, min_samples_split=17, n_estimators=100)

)



exported_pipeline.fit(x_train, y_train)

results = exported_pipeline.predict(x_test)
print(exported_pipeline.score(x_test, y_test))

print(exported_pipeline.score(x_train, y_train))
print(classification_report(y_test, exported_pipeline.predict(x_test)))

print(confusion_matrix(y_test, exported_pipeline.predict(x_test)))
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.ensemble import GradientBoostingClassifier



#uncomment the commented code and uncomment the commented to perform gridsearchCV

from xgboost import XGBClassifier as xgb



clf = ExtraTreesClassifier(n_estimators=950, random_state=0)

from sklearn.svm import LinearSVC

from mlxtend.classifier import StackingCVClassifier



c1 = ExtraTreesClassifier(n_estimators=500,bootstrap=True) 

c2= RandomForestClassifier(n_estimators=500,bootstrap=True)

c3=xgb();

meta = LinearSVC()

sclf = StackingCVClassifier(classifiers=[c1,c2,c3],use_probas=True,meta_classifier=meta)
sclf.fit(x_train, y_train)

print('Accuracy of classifier on training set: {:.2f}'.format(sclf.score(x_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(sclf.score(x_test, y_test) * 100))
test.head()
test.columns
test.head()



id = test['Id']

test.drop(['Id', 'Soil_Type7', 'Soil_Type15'] , inplace = True , axis = 1)

test = scaler.transform(test)
predictions = sclf.predict(test)
out = pd.DataFrame({'Id': id,'Cover_Type': predictions})

out.to_csv('submission.csv', index=False)

out.head(5)