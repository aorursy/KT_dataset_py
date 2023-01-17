#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
def findDr(st):
    found = 0
    for word in st.lower().split():
        if word == 'dr.':
            found += 1
    if found > 0:
        return 1
    else:
        return 0
def CabinCat(cabin):
    returnedPrefix = 'U'
    if pd.notnull(cabin):
        CabinPrefix = cabin[0].upper()
        knownCabinTypes = ['A','B','C','D','E','F','G','T']
        
        if CabinPrefix in knownCabinTypes:
            returnedPrefix = CabinPrefix   
    
    return returnedPrefix
def age_model_build(df):
    #function uses linear regression to predict age.
    
    #drop the null ages for the purpose of building a model.  Drop columns which don't seem relevant to age
    df = df.dropna()
    X = df.drop(columns=['Age','PassengerId','Survived','Embarked_Q','Embarked_C','CabinCategory_T',
                         'CabinCategory_D','CabinCategory_E','CabinCategory_C','CabinCategory_B',
                         'CabinCategory_U','CabinCategory_F','CabinCategory_G'])
    y = df['Age']
    
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    lm = LinearRegression()
    lm.fit(X,y)
    
    predictions = lm.predict(X)
    coeffecients = pd.DataFrame(lm.coef_,X.columns)
    coeffecients.columns = ['Coeffecient']
    coeffecients.to_csv('Coefficients.csv')
    
    #Write out a scatterplot to check model performance
    plt.figure(figsize=(8, 6))
    plt.scatter(y,predictions)
    plt.grid()
    plt.xlabel('Actual Y Values')
    plt.ylabel('Predicted Values')
    plt.rcParams['axes.axisbelow'] = True
    plt.rc('axes', axisbelow=True)
    
    print('MAE:', metrics.mean_absolute_error(y, predictions))
    print('MSE:', metrics.mean_squared_error(y, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y, predictions)))

    
    return lm
def findRev(st):
    found = 0
    for word in st.lower().split():
        if word == 'rev.':
            found += 1
    if found > 0:
        return 1
    else:
        return 0
def findMstr(st):
    found = 0
    for word in st.lower().split():
        if word == 'master.':
            found += 1
    if found > 0:
        return 1
    else:
        return 0
def guess_missing(cols):
    #if data is missing use dummy column else use actual data
    if pd.isnull(cols[0]):
        return cols[1]
    else:
        return cols[0]
def populate_missing(df, model = None, dropPredicted = None):
    
    if(dropPredicted):
        delcolumns=['Age','PassengerId','Avg_Fare','Survived','Embarked_Q','Embarked_C','CabinCategory_T',
                         'CabinCategory_D','CabinCategory_E','CabinCategory_C','CabinCategory_B',
                         'CabinCategory_U','CabinCategory_F','CabinCategory_G']
    else:
        delcolumns=['Age','PassengerId','Avg_Fare','Embarked_Q','Embarked_C','CabinCategory_T',
                         'CabinCategory_D','CabinCategory_E','CabinCategory_C','CabinCategory_B',
                         'CabinCategory_U','CabinCategory_F','CabinCategory_G']
    
    #First a calculation for average Fare
    by_Class = df.groupby("Pclass")
    AvgFare = pd.DataFrame(by_Class.mean()['Fare'])
    AvgFare['Pclass'] = AvgFare.index
    AvgFare = AvgFare.rename(columns={'Fare': 'Avg_Fare'})
    df = df.merge(AvgFare, on='Pclass',how='left')
    
    #There isn't any missing Fare data in the Train dataset but there is missing fare data in the test dataset
    #For the fare I will use the mean fare for the particular class
    df['Fare'] = df[['Fare','Avg_Fare']].apply(guess_missing,axis=1)

    #Next a calculation for age using a linear regression model
    predicted_ages = pd.DataFrame(model.predict(df.drop(columns=delcolumns)))
    predicted_ages = predicted_ages.rename(columns={0: 'Pred_Age'})
    
    df = df.join(predicted_ages)
    df['Age'] = df[['Age','Pred_Age']].apply(guess_missing,axis=1)
    df = df.drop(columns=['Pred_Age','Avg_Fare'])
    
    return df
    
#dataclean up function
def dataCleanUp(df):

    #simplified strategy:  Drop Cabin & Ticket.  
    #From Name, only derive Master, Dr, Rev
    
    df['CabinCategory'] = df['Cabin'].apply(CabinCat)
    df = df.drop(columns='Cabin')
    
    df = df.drop(columns='Ticket')

    df['IsDR'] = df['Name'].apply(findDr)
    df['IsRev'] = df['Name'].apply(findRev)
    df['IsMstr'] = df['Name'].apply(findMstr)

    df = df.drop(columns='Name')

    df = pd.get_dummies(df,columns=['Sex'])
    df = pd.get_dummies(df,columns=['Embarked'])
    df = pd.get_dummies(df,columns=['CabinCategory'])

    df =df.drop(columns=['Sex_female','Embarked_S','CabinCategory_A'])
    #df =df.drop(columns=['Sex_female','Embarked_S'])
    
    return df
def buildRandomForest(df):
    X = df.drop(columns='Survived')
    y= df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    #parameters come from GridSearchCV which is commented out below 
    rfc = RandomForestClassifier(n_estimators=600,max_depth=7,max_features='sqrt',criterion='gini')
    rfc.fit(X_train,y_train)
    #rfc.fit(X,y)
    predictions = rfc.predict(X_test)
    #predictions = rfc.predict(X)
    from sklearn.metrics import confusion_matrix,classification_report
    print("The confusion Matrix")
    print(confusion_matrix(y_test,predictions))
    #print(confusion_matrix(y,predictions))
    #print("The classification report")
    print(classification_report(y_test,predictions))
    #print(classification_report(y,predictions))
    
    return rfc
def buildLogReg(df):
    X = df.drop(columns='Survived')
    y= df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    #parameters come from GridSearchCV which is commented out below 
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    #rfc.fit(X,y)
    predictions = logmodel.predict(X_test)
    #predictions = rfc.predict(X)
    from sklearn.metrics import confusion_matrix,classification_report
    print("The confusion Matrix")
    print(confusion_matrix(y_test,predictions))
    #print(confusion_matrix(y,predictions))
    #print("The classification report")
    print(classification_report(y_test,predictions))
    #print(classification_report(y,predictions))
    
    return logmodel
df = pd.read_csv('../input/train.csv')
#First step is to build a model to predict the age of each passenger.  This will be used to populate missing ages later
df = dataCleanUp(df)
df_headers = set(df.columns.drop('Survived'))
age_model = age_model_build(df)
df = populate_missing(df,age_model,True)
rfc = buildRandomForest(df.drop(columns='PassengerId'))
# #Determine the best parameters for Random Forest
# from sklearn.model_selection import GridSearchCV
# param_grid = { 
#   'n_estimators': [500, 600],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'max_depth' : [4,5,6,7,8],
#    'criterion' :['gini', 'entropy']
# }

# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# CV_rfc.fit(df.drop(columns='Survived'),df['Survived'])
# CV_rfc.best_params_
test_df = pd.read_csv('../input/test.csv')
test_df_rows = test_df.count()[0]
test_df = dataCleanUp(test_df)
test_df_headers = set(test_df.columns)
missing_from_test =  df_headers-test_df_headers
number_of_columns_missing = len(list(missing_from_test))
#Append dummy data
dummy_columns = np.zeros((test_df_rows,number_of_columns_missing))
final_dummy_data = pd.DataFrame(dummy_columns,columns=list(missing_from_test))
test_df = test_df.join(final_dummy_data)
#populate the ages using the previously built age model
test_df = populate_missing(test_df,age_model,False)
test_df.head()
finalPredictions = pd.DataFrame(rfc.predict(test_df.drop(columns='PassengerId')))
finalPredictions = finalPredictions.rename(columns={0:'Survived'})
#join back to original dataset
final_predictions_for_csv = pd.DataFrame(test_df['PassengerId']).join(finalPredictions)
#final_predictions_for_csv.to_csv('../input/final_prediction.csv',index=False)
final_predictions_for_csv.head()

