# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dressdf = pd.read_csv('../input/spoolOut.csv')

print(dressdf.head(5))
print(dressdf.info())
for i in dressdf: # loops through column names

    count = 0

    for j in dressdf[i]:

        if j is np.nan:

            count += 1

    print('column: '+i+' number of nulls: '+str(count))
def bargraphs(df,feature):

    totalCountYesRec = df[df['recommendation']==1].shape[0] # total number of rows for both recommended and not recommended dresses

    totalCountNotRec = df[df['recommendation'] ==0].shape[0]

    

    resultdf = pd.DataFrame()

    df = df.dropna(axis=0,how='any')

    

    allFeatureValues = df[feature].unique()

    allFeatureValues = np.sort(allFeatureValues)

    resultdf[feature] = allFeatureValues # defining the necessary columns for the temporary data frame

    resultdf['pctStyleRec'] = np.nan

    resultdf['pctStyleNotRec'] = np.nan

    

    # loop will populate the columns with calculated field values that represent the percent of each style makes up of all dresses

    # in either recommended or not recommended dresses

    for col in allFeatureValues:

        resultdf.loc[resultdf[feature] == col,['pctStyleRec']] = (df[(df[feature] == col) & (df['recommendation'] == 1)][feature].count() / totalCountYesRec) 

        #print(df[df['style'] == col & df['recommendation'] == 1]['style'].count() / totalCountYesRec)

        resultdf.loc[resultdf[feature] == col,['pctStyleNotRec']] =(df[(df[feature] == col) & (df['recommendation'] == 0)][feature].count() /totalCountNotRec)

        

        

    fig, ax = plt.subplots(1,1,figsize=(9,6))

    

    #resultdf[['pctStyleRec','pctStyleNotRec']].plot(kind='bar',subplots=False)

    print(resultdf)

    

    xTicks = len(allFeatureValues)

    

    ax.bar(np.arange(xTicks),resultdf['pctStyleRec'],width=0.45,color='b',align='edge',label='recommended dress: '+feature+' frequency',tick_label=allFeatureValues)

    ax.bar(np.arange(xTicks),resultdf['pctStyleNotRec'],width=-0.45,color='r',align='edge',label='non-rec dress: '+feature+' frequency',tick_label=allFeatureValues)

    plt.xticks(rotation=70)

    plt.title('percent of dresses of feature: '+feature+' in recommend or non-recommended class')

    plt.legend(loc='best')

    plt.show()

   

bargraphs(dressdf,'style')

bargraphs(dressdf,'price')

bargraphs(dressdf,'rating')

bargraphs(dressdf,'dress_size')

bargraphs(dressdf,'season')

bargraphs(dressdf,'neckline')

bargraphs(dressdf,'sleeve_length')

bargraphs(dressdf,'waistline')
df3 = dressdf.drop(['material','pattern_type','waistline','dress_id'],axis=1)



df3 = pd.get_dummies(df3) # convert the catagorical columns into dummies



feature_data = df3.drop('recommendation',axis=1) # split the data into raw data and labels

label_data = df3['recommendation']



# test set will be for the eventual model. train set is for the cross validation 

x_test, x_train, y_test, y_train = train_test_split(feature_data,label_data,test_size=0.33,random_state=177)



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV      # will allow testing for optimal combination of model parameters



paramDict = { 'n_estimators' : [100,500,1000],

             'max_depth' : [3,10,20]}



gridSearch = GridSearchCV(RandomForestClassifier(),paramDict,cv=5) 

gridSearch.fit(x_train,y_train)

print('best parameters found via grid search %s\n'% gridSearch.best_params_)

print('scoring the test data with the random forest and best paramters %s' % gridSearch.score(x_test,y_test))

from sklearn.metrics import confusion_matrix, classification_report



predictedLabels = gridSearch.predict(x_test)



matrix = confusion_matrix(y_test,predictedLabels)

print('printing confusion matrix for random forest')

print(matrix)

print(classification_report(y_test,predictedLabels,target_names=['not recommended','recommended']))
from sklearn.metrics import roc_curve

 

PosProEstimates = gridSearch.predict_proba(x_test)[:,1] # return the positive class' prediction probability esitmates

fpr, tpr, thresholds = roc_curve(y_test, PosProEstimates)

 

plt.plot(fpr,tpr, label='ROC curve',c='blue')

plt.grid(b=True,alpha=0.7,color='gray')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate ")

close_zero = np.argmin(np.abs(np.array(thresholds) - 0.5)) # default classification threshold is 0.5. So finding the index for

# the calculated threshold closest to 0.5 requires subtracting 0.5 and finding the minimum absolute value. 

plt.plot(fpr[close_zero],tpr[close_zero],marker='X',label='zero threshold',ms=15,c='black')

plt.legend(loc='best')


