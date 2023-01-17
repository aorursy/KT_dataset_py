### EDA on the human resrouce dataset

### Why are the employees leaving
import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

###------------------------------ LOADING THE DATA

df = pd.read_csv('../input/HR_comma_sep.csv')

###------------------------------ LOADING THE DATA

df.head()

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()  ### Encode the salary

df['salary'] = lb.fit_transform(df['salary'])
### Make the sales columns dummies

temp = pd.get_dummies(df['sales'])

df = pd.concat([df,temp],axis=1)

df = df.drop('sales',axis=1)

df.head()
corr = df.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.8, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr['left']

### The highest correlation is satisfaction_level make sense
sns.boxplot(df['left'],df['satisfaction_level'])

### More satisfied in the one who stayed but more variance in the one who left
corr['satisfaction_level']  ### Too many project and too many hours decrease the satisfaction
### Looking at the average montly hours difference between those who left and stayed

df.groupby('left')['average_montly_hours'].mean()
### Creating a new variable Time repartition

df['time_repartition'] = df['average_montly_hours']/df['number_project']

df.corr()['left']
df.tail()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



### I will keep all the model is a list for further analysis

modelList = []

modelNames = []



### Fillna with 0

df = df.fillna(0)



    

X = df.drop('left',axis=1)

y = df['left']



### Split in val and train

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 10)



### Create classifier  and quick grid search

clf = RandomForestClassifier()

rfparams = {"max_depth": [3, 5,8],

              "max_features": [1, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}

rf_G = GridSearchCV(clf,rfparams,cv=5,verbose=1)

rf_G.fit(X_train,y_train)



modelList.append(rf_G)

modelNames.append('RandomForest')
y_pred = rf_G.predict(X_val)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, y_pred)

print(cm)

acc = (2262+663)/(2262+4+71+663)



print('RandomForest : The accuracy is of %f' %acc)
### Not bad 97.4667 of accuracy can we do better ?
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgbparams ={'max_depth': [2,4,6],

            'n_estimators': [50,100,200]}

xgb_G = GridSearchCV(xgb,xgbparams,cv=5,verbose=1)

xgb_G.fit(X_train,y_train)
y_pred = xgb_G.predict(X_val)

cm = confusion_matrix(y_val, y_pred)

print(cm)

acc = (2260+688)/(2260+688+46+6)

print('XGB : The accuracy is of %f' %acc)



modelList.append(xgb_G)

modelNames.append('XGBoost')



### Better 98.2667
### Is there some people that leave that are Anomaly ???

from sklearn import svm



model = svm.OneClassSVM(nu=0.02, kernel='rbf', gamma=0.005)  

model.fit(X)  

df['anomaly'] = model.predict(X)

df.groupby('anomaly').mean()
### We lower satisfactoion level for the outlier

### Higher montly hours

### More time spend in the comoany

### More Work Accident

### Less left...

### More promotion last 5 years

### Lower salary

### Time more concentrate on one project.







# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.8, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
### Weak correlation of anomaly with the salary



### Before splitting keeping index

df= df.reset_index()

### Let's rerun the xgb with this new columns

### Without outlier 

dff = df.loc[df['anomaly'] > 0,:]



X = dff.drop('left',axis=1)

y = dff['left']



### Split in val and train

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 10)



xgb_G.fit(X_train,y_train)
y_pred = xgb_G.predict(X_val)

cm = confusion_matrix(y_val, y_pred)

print(cm)

acc = (2215+716)/(2215+716)

print('XGB : The accuracy is of %f' %acc)



modelList.append(xgb_G)

modelNames.append('XGBoost_Without_anomaly')



### 98.6012 without the outlier



### Interesting with the outlier was 98.2

### We could try to run a model for each group ""hybrid""
### Run the model for the other

dfo = df.loc[df['anomaly'] < 0,:]



X = dfo.drop('left',axis=1)

y = dfo['left']



### Split in val and train

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 10)



xgb_G.fit(X_train,y_train)
y_pred = xgb_G.predict(X_val)

cm = confusion_matrix(y_val, y_pred)

print(cm)



acc = (59+10)/(59+10+1)

print('XGB : The accuracy is of %f' %acc)



modelList.append(xgb_G)

modelNames.append('XGBoost_anomaly')

### Bringing it together



def Predicting_with_outlier(data,outlier=True):

    if outlier == True:

        from sklearn import svm

        svm = svm.OneClassSVM()  

        svm.fit(data)  

        data['anomaly'] = svm.predict(data)

        data = data.reset_index()

        outlier = data.loc[data['anomaly']<0,:]

        normal = data.loc[data['anomaly']>0,:]

        

        def fit(df):

            X = df.drop('left',axis=1)

            y = df['left']

            ### Split in val and train

            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 10)

            from xgboost import XGBClassifier

            xgb = XGBClassifier()

            xgbparams ={'max_depth': [2,4,6],

                        'n_estimators': [50,100,200]}

            xgb_G = GridSearchCV(xgb,xgbparams,cv=5,verbose=1)

            xgb_G.fit(X_train,y_train)

            df['predictions'] = xgb_G.predict(X)

            return df

        

        outlier = fit(outlier)

        normal = fit(normal)

        df_p = normal.append(outlier)

        df_p = df_p.sort_values(['index'])

    else:

        print('Not Training')

    return df_p

            



df_p = Predicting_with_outlier(df)

df_p.head()
def is_the_same_value(col1,col2):

    boolean = col1==col2

    return boolean.sum()/len(col1)



acc = is_the_same_value(df_p['left'],df_p['predictions'])



print('the accuracy of the new model is %f'%acc)



    