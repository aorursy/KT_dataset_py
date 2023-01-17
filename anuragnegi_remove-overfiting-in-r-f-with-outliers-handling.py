

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train=pd.read_csv('../input/learn-together/train.csv')

test=pd.read_csv('../input/learn-together/test.csv')

sample_submission=pd.read_csv('../input/learn-together/sample_submission.csv')
plt.figure(figsize=(20,30))

plt.subplot(5, 2, 1)

fig = train.boxplot(column='Elevation')

fig.set_title('')

fig.set_ylabel('Elevation')

 

plt.subplot(5, 2, 2)

fig = train.boxplot(column='Aspect')

fig.set_title('')

fig.set_ylabel('Aspect')



plt.subplot(5, 2, 3)

fig = train.boxplot(column='Slope')

fig.set_title('')

fig.set_ylabel('Slope')

 

plt.subplot(5, 2, 4)

fig = train.boxplot(column='Horizontal_Distance_To_Hydrology')

fig.set_title('')

fig.set_ylabel('Horizontal_Distance_To_Hydrology')



plt.subplot(5, 2, 5)

fig = train.boxplot(column='Vertical_Distance_To_Hydrology')

fig.set_title('')

fig.set_ylabel('Vertical_Distance_To_Hydrology')

 

plt.subplot(5, 2, 6)

fig = train.boxplot(column='Horizontal_Distance_To_Roadways')

fig.set_title('')

fig.set_ylabel('Horizontal_Distance_To_Roadways')



plt.subplot(5, 2, 7)

fig = train.boxplot(column='Hillshade_9am')

fig.set_title('')

fig.set_ylabel('Hillshade_9am')

 

plt.subplot(5, 2, 8)

fig = train.boxplot(column='Hillshade_Noon')

fig.set_title('')

fig.set_ylabel('Hillshade_Noon')



plt.subplot(5, 2, 9)

fig = train.boxplot(column='Hillshade_3pm')

fig.set_title('')

fig.set_ylabel('Hillshade_3pm')

 

plt.subplot(5, 2, 10)

fig = train.boxplot(column='Horizontal_Distance_To_Fire_Points')

fig.set_title('')

fig.set_ylabel('Horizontal_Distance_To_Fire_Points')
X_train=train.copy()

x_test=test.copy()


IQR = X_train.Slope.quantile(0.75) - X_train.Slope.quantile(0.25) #finding interquantile range

Lower_fence = X_train.Slope.quantile(0.25) - (IQR * 1.5)          #lower value

Upper_fence = X_train.Slope.quantile(0.75) + (IQR * 1.5)          #upper value

print('slope number outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

Slope = len(X_train[X_train.Slope>Upper_fence]) / np.float(len(X_train))

print('Number of Slope with values higher than {upperboundary}: {slope}'.format(upperboundary=Upper_fence, slope=Slope))
def top_code(df, variable, top):                            #function to top_code values

    return np.where(df[variable]>top, top, df[variable])

X_train['Slope']= top_code(X_train, 'Slope', 40)            #train_set is top_coded

x_test['Slope']= top_code(x_test,'Slope',40)        #test set is top_coded


IQR2 = X_train.Hillshade_3pm.quantile(0.75) - X_train.Hillshade_3pm.quantile(0.25)

Lower_fence = X_train.Hillshade_3pm.quantile(0.25) - (IQR2 * 1.5)

Upper_fence = X_train.Hillshade_3pm.quantile(0.75) + (IQR2 * 1.5)

print('Hillshade_3pm number outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

Hillshades_3pm = len(X_train[X_train.Hillshade_3pm<Lower_fence]) / np.float(len(X_train))

print('Number of Hillshade_3pm with values lower than {lowerboundary}: {Hillshade_3pm}'.format(lowerboundary=Lower_fence, Hillshade_3pm=Hillshades_3pm))
def bottom_code(df, variable, bottom):                      #function to bottom values

    return np.where(df[variable]<bottom, bottom, df[variable])



#bottom coding Hillshade_3pm

X_train['Hillshade_3pm']=bottom_code(X_train,'Hillshade_3pm',14)

x_test['Hillshade_3pm']=bottom_code(x_test,'Hillshade_3pm',14)
def discretise(var,X_train,x_test):

    #ploting figure before discritisation

    fig = plt.figure()

    fig = X_train.groupby([var])['Cover_Type'].mean().plot(figsize=(12,6))

    fig.set_title('relationship between variable and target before discretisation')

    fig.set_ylabel('Cover_Type')

    

    # find quantiles and discretise train set

    

    X_train[var], bins = pd.qcut(x=X_train[var], q=8, retbins=True, precision=3, duplicates='raise')

    x_test[var] = pd.cut(x = x_test[var], bins=bins, include_lowest=True)

    

    t1 = X_train.groupby([var])[var].count() / np.float(len(X_train))

    t3 = x_test.groupby([var])[var].count() / np.float(len(x_test))

    

    

    #plot to show distribution of values

    temp = pd.concat([t1,t3], axis=1)

    temp.columns = ['train', 'test']

    temp.plot.bar(figsize=(12,6))

    

    #plot after discretisation

    fig = plt.figure()

    fig = X_train.groupby([var])['Cover_Type'].mean().plot(figsize=(12,6))

    fig.set_title('Normal relationship between variable and target after discretisation')

    fig.set_ylabel('Cover_Type')

    

#     return X_train,x_test
try:

    X_train,x_test=discretise('Horizontal_Distance_To_Hydrology',X_train,x_test)

except TypeError:

    pass
try:

    X_train,x_test=discretise('Vertical_Distance_To_Hydrology',X_train,x_test)

except TypeError:

    pass
try:

    X_train,x_test=discretise('Horizontal_Distance_To_Roadways',X_train,x_test)

except TypeError:

    pass
try:

    X_train,x_test=discretise('Hillshade_9am',X_train,x_test)

except TypeError:

    pass
try:

    X_train,x_test=discretise('Hillshade_Noon',X_train,x_test)

except TypeError:

    pass
try:

    X_train,x_test=discretise('Horizontal_Distance_To_Fire_Points',X_train,x_test)

except TypeError:

    pass
def inpute(var,X_train,x_test):

    if x_test[var].isnull().sum()>0:

        x_test.loc[x_test[var].isnull(), var] = X_train[var].unique()[0]

        print("inputed column :: ",var)

    
inpute('Hillshade_3pm',X_train,x_test)
for col in X_train.columns:

    if col !='Cover_Type':

        inpute(col,X_train,x_test)

X_train.head()
for df in [X_train, x_test]:

    df.Horizontal_Distance_To_Hydrology = df.Horizontal_Distance_To_Hydrology.astype('O')

    df.Vertical_Distance_To_Hydrology = df.Vertical_Distance_To_Hydrology.astype('O')

    df.Horizontal_Distance_To_Roadways = df.Horizontal_Distance_To_Roadways.astype('O')

    df.Hillshade_9am = df.Hillshade_9am.astype('O')

    df.Hillshade_Noon = df.Hillshade_Noon.astype('O')

    df.Horizontal_Distance_To_Fire_Points = df.Horizontal_Distance_To_Fire_Points.astype('O')
def encode_categorical_variables(var, target):

        # make label to risk dictionary

        ordered_labels = X_train.groupby([var])[target].mean().to_dict()

        

        # encode variables

        X_train[var] = X_train[var].map(ordered_labels)

        x_test[var] = x_test[var].map(ordered_labels)
for var in ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Horizontal_Distance_To_Fire_Points']:

    print(var)

    encode_categorical_variables(var, 'Cover_Type')
X_train.head()
import warnings

warnings.filterwarnings("ignore")



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

import gc
def model_training(model,X_train,y_train):

    scores =  cross_val_score(model, X_train, y_train,

                              cv=5)

    return scores.mean()
X_train.drop(['Id','Cover_Type'],axis=1,inplace=True)

x_test.drop(['Id'],axis=1,inplace=True)

y_train=train['Cover_Type']
model_training(RandomForestClassifier(n_estimators=885,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=110,bootstrap=False),X_train,y_train)
my_model=RandomForestClassifier(n_estimators=885,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=110,bootstrap=False)

my_model.fit(X_train,y_train)
test_preds=my_model.predict(x_test)
output = pd.DataFrame({'Id': test.Id,

                      'Cover_Type': test_preds})

output.to_csv('sample_submission.csv', index=False)