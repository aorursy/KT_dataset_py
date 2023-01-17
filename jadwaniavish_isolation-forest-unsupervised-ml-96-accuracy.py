import pandas as pd 
import numpy as np 
# Author: Avish Jadwani
# Last Updated: 01/10/2020

# Function to remove a column from a data frame manualy
def remove_col(df):
    col = input("Enter the column you want to remove, If you want to remove multiple columns separate it by comma eg. variable1,variable2")
    if col == '':
        print("No column removed")
    else:
        col=col.split(",")
        df.drop(columns=col,axis=1, inplace= True)
    
    print("Removed column {}".format(col))
# Function to remove columns with '0' values 
def  rem_zero_cols(df):
    rem = []
    for col in df:
        if( sum(df[col]==0) >= (0.95 * len(df)) ):
        
            rem.append(col)
    df.drop(rem, axis=1, inplace = True)
    print('Removed columns {} with 0 values'.format(rem))
    
    
# Function to remove columns with null values 
def  rem_null_cols(df):
    import pandas as pd
    import numpy as np
    rem = []
    for col in df:
        if(sum(pd.notna(df[col]))==0):
            rem.append(col)
    df.drop(rem, axis=1, inplace = True)
    print('Removed columns {} with all null values'.format(rem))
    
# Identifying and Removing Unique identifiers
def uniq_iden(df):
    rem = []
    for col in df:
        if df[col].nunique() == len(df):
            rem.append(col)
    df.drop(rem,axis=1, inplace =True)
    print("Removed uinque identifiers {}  ".format(rem))
# Function to replace NA values 
def replace_NA(df,num):
    df.fillna(num, inplace = True)

# Replace NA values with -9999, One Hot encoding for categorical variables, isolation forest, result stored in excel file with name Outlier_result
def hotenc_isof_btrace(df):
    import pandas as pd
    import numpy as np
    lsto=[]
    lstflt=[]
    df2 = pd.DataFrame()
    df_short= pd.DataFrame()
    df_catvar = pd.DataFrame()
    n=0
    #Replace NA values with -9999
    #-9999 can be changed to any value
    replace_NA(df,-9999)
    # Performing one hot encoding by subsettig categorical variables
    for col in df:
        if(df[col].dtype == 'O'):
            lsto.append(col)
        elif(df[col].dtype != 'O'):
            lstflt.append(col)
    df2 =pd.get_dummies(df, prefix_sep="__", columns=lsto)
    
    # Performing isolation forest 
    from sklearn.ensemble import IsolationForest

    isofor = IsolationForest(n_estimators=200,contamination=0.01, behaviour='deprecated')
    isofor.fit(df2)
    predict = isofor.predict(df2)
    # Storing results of isolation forest in 'Outlier' variable in data frame
    df2['Outlier'] = predict
    # Restoring data frame to original format, ie. without one hot encoding
    lstflt.append('Outlier')
    df_catvar = df.loc[:,lsto]
    df_short = df2.loc[:,lstflt]
    result = pd.concat([df_catvar,df_short], axis =1 , sort=False)
    result.replace(-9999, np.nan,inplace =True )
    
    n = n+len(result.columns)
    
    def highlight_lessthan(x):
        if x.Outlier == -1:
            return ['background-color: yellow']*n
        else:
             return ['background-color: white']*n
    result = result.style.apply(highlight_lessthan, axis= 1)
    
    
    result.to_excel("NormOutlier_result.xlsx")



# Complete analysis in one function 

def isoforest(df):
    import pandas as pd
    import numpy as np
    import GCOutlier as gc
    # Select column to remove 
    gc.remove_col(df)
    #remove columns with null values
    gc.rem_null_cols(df)
    # Remove colmns with 0
    gc.rem_zero_cols(df)
    # Remove Unique identifiers
    gc.uniq_iden(df)
    
    lsto=[]
    lstflt=[]
    df2 = pd.DataFrame()
    df_short= pd.DataFrame()
    df_catvar = pd.DataFrame()
    n=0
    #Replace NA values with -9999
    #-9999 can be changed to any value
    replace_NA(df,-9999)
    # Performing one hot encoding by subsettig categorical variables
    for col in df:
        if(df[col].dtype == 'O'):
            lsto.append(col)
        elif(df[col].dtype != 'O'):
            lstflt.append(col)
    df2 =pd.get_dummies(df, prefix_sep="__", columns=lsto)
    
    # Performing isolation forest 
    from sklearn.ensemble import IsolationForest

    isofor = IsolationForest(n_estimators=200,contamination=0.01, behaviour='deprecated')
    isofor.fit(df2)
    predict = isofor.predict(df2)
    # Storing results of isolation forest in 'Outlier' variable in data frame
    df2['Outlier'] = predict
    # Restoring data frame to original format, ie. without one hot encoding
    lstflt.append('Outlier')
    df_catvar = df.loc[:,lsto]
    df_short = df2.loc[:,lstflt]
    result = pd.concat([df_catvar,df_short], axis =1 , sort=False)
    result.replace(-9999, np.nan,inplace =True )
    
    n = n+len(result.columns)
    
    def highlight_lessthan(x):
        if x.Outlier == -1:
            return ['background-color: yellow']*n
        else:
             return ['background-color: white']*n
    result = result.style.apply(highlight_lessthan, axis= 1)
    
    
    result.to_excel("NormOutlier_result.xlsx")
    
# Function for one hot encoding , isolation forest, back tracing and saving outliers as a separate file 
def return_Outlier(df):
    import pandas as pd
    import numpy as np
    lsto=[]
    lstflt=[]
    df2 = pd.DataFrame()
    df_short= pd.DataFrame()
    df_catvar = pd.DataFrame()
    n=0
    #Replace NA values with -9999
    #-9999 can be changed to any value
    replace_NA(df,-9999)
    # Performing one hot encoding by subsettig categorical variables
    for col in df:
        if(df[col].dtype == 'O'):
            lsto.append(col)
        elif(df[col].dtype != 'O'):
            lstflt.append(col)
    df2 =pd.get_dummies(df, prefix_sep="__", columns=lsto)
    
    # Performing isolation forest 
    from sklearn.ensemble import IsolationForest

    isofor = IsolationForest(n_estimators=200,contamination=0.01, behaviour='deprecated')
    isofor.fit(df2)
    predict = isofor.predict(df2)
    # Storing results of isolation forest in 'Outlier' variable in data frame
    df2['Outlier'] = predict
    # Restoring data frame to original format, ie. without one hot encoding
    lstflt.append('Outlier')
    df_catvar = df.loc[:,lsto]
    df_short = df2.loc[:,lstflt]
    result = pd.concat([df_catvar,df_short], axis =1 , sort=False)
    result.replace(-9999, np.nan,inplace =True )
    
    n = n+len(result.columns)
    
    
    #def highlight_lessthan(x):
        #if x.Outlier == -1:
            #return ['background-color: yellow']*n
        #else:
             #return ['background-color: white']*n
    #result = result.style.apply(highlight_lessthan, axis= 1)
    result = result[result.loc[:,'Outlier']==-1]
    result = result.drop('Outlier',axis=1)
    
    result.to_excel("Outlier_result.xlsx")
    
# Function to merge data with polyciynum
def change(df):
    import pandas as pd
    import numpy as np
    #def remove_col(df):
    col = input("Enter the column you want to remove. If you have more than 1 column separate it with comma without any space eg column1,column2")
    if col == '':
        print("No column removed")
    else:
        col=col.split(",")
        removedcol = df.loc[:,col]
        df.drop(columns=col,axis=1, inplace= True)
        print("Removed column {}".format(col))
    
    lsto=[]
    lstflt=[]
    df2 = pd.DataFrame()
    df_short= pd.DataFrame()
    df_catvar = pd.DataFrame()
    n=0
    #Replace NA values with -9999
    #-9999 can be changed to any value
    replace_NA(df,-9999)
    # Performing one hot encoding by subsettig categorical variables
    for col in df:
        if(df[col].dtype == 'O'):
            lsto.append(col)
        elif(df[col].dtype != 'O'):
            lstflt.append(col)
    df2 =pd.get_dummies(df, prefix_sep="__", columns=lsto)
    
    # Performing isolation forest 
    from sklearn.ensemble import IsolationForest

    isofor = IsolationForest(n_estimators=200,contamination=0.01, behaviour='deprecated')
    isofor.fit(df2)
    predict = isofor.predict(df2)
    # Storing results of isolation forest in 'Outlier' variable in data frame
    df2['Outlier'] = predict
    # Restoring data frame to original format, ie. without one hot encoding
    lstflt.append('Outlier')
    df_catvar = df.loc[:,lsto]
    df_short = df2.loc[:,lstflt]
    result = pd.concat([df_catvar,df_short], axis =1 , sort=False)
    result=pd.concat([removedcol,result],axis=1,sort=False)
    result.replace(-9999, np.nan,inplace =True )
    
    n = n+len(result.columns)
    
    
    #def highlight_lessthan(x):
        #if x.Outlier == -1:
            #return ['background-color: yellow']*n
        #else:
             #return ['background-color: white']*n
    #result = result.style.apply(highlight_lessthan, axis= 1)
    result.drop('Outlier', axis=1)
    return result
    #result.to_excel("Outlier_result.xlsx")
# Final Function to perform isoforest- it also removes the input columns,multiple columns can also be removed. Gives try agin #message
def gcoutlier(df):
    import pandas as pd
    import numpy as np
    #def remove_col(df):
    def check_elements(df,col):
        col = col.split(",")
        for i in range(0,len(col)):
            if(col[i] not in list(df.columns)):
                return -1
                break
    while True:
        
        col = input("Enter the column you want to remove. \n If you have more than 1 column separate it with comma(,)")
        if col == '':
            print("No column removed")
            merge = -1
            break
        elif check_elements(df,col)==-1:
            #check = check_elements(df,col)
            #if check==-1:
            print('Incorrect column name, Try Again')
            continue
        else:
            col=col.split(",")
            removedcol = df.loc[:,col]
            df.drop(columns=col,axis=1, inplace= True)
            merge = 1
            print("Removed column {}".format(col))
            break
    
    lsto=[]
    lstflt=[]
    df2 = pd.DataFrame()
    df_short= pd.DataFrame()
    df_catvar = pd.DataFrame()
    n=0
    #Replace NA values with -9999
    #-9999 can be changed to any value
    replace_NA(df,-9999)
    # Performing one hot encoding by subsettig categorical variables
    for col in df:
        if(df[col].dtype == 'O'):
            lsto.append(col)
        elif(df[col].dtype != 'O'):
            lstflt.append(col)
    df2 =pd.get_dummies(df, prefix_sep="__", columns=lsto)
    
    # Performing isolation forest 
    from sklearn.ensemble import IsolationForest

    isofor = IsolationForest(contamination='auto', behaviour='deprecated')
    isofor.fit(df2)
    predict = isofor.predict(df2)
    # Storing results of isolation forest in 'Outlier' variable in data frame
    df2['Outlier'] = predict
    # Restoring data frame to original format, ie. without one hot encoding
    lstflt.append('Outlier')
    df_catvar = df.loc[:,lsto]
    df_short = df2.loc[:,lstflt]
    result = pd.concat([df_catvar,df_short], axis =1 , sort=False)
    if merge == 1:
        result=pd.concat([removedcol,result],axis=1,sort=False)
    result.replace(-9999, np.nan,inplace =True )
    
    n = n+len(result.columns)
    
    
    #def highlight_lessthan(x):
        #if x.Outlier == -1:
            #return ['background-color: yellow']*n
        #else:
             #return ['background-color: white']*n
    #result = result.style.apply(highlight_lessthan, axis= 1)
    result.drop('Outlier', axis=1)
    #return result
    result.to_excel("Outlier_result.xlsx")

    
    
    
    
import os 
os.getcwd()
os.chdir('..')
os.listdir()
df = pd.read_csv("./input/creditcardfraud/creditcard.csv")
df.info()

# Removing columsn which has all the null values
rem_null_cols(df)
uniq_iden(df)
# gcoutlier performs isolation forest 
# This function pops up a dialog box to enter a column name to delete it manually 
# Fucntion stores the result of the Isolation forest in the directory with filename Outlier_results.xlsx
#in our case Class
gcoutlier(df)
df2 = pd.read_excel("Outlier_result.xlsx")
df2.head()
from sklearn.metrics import confusion_matrix
df2['Outlier'].replace(1,0, inplace = True)
df2['Outlier'].replace(-1,1,inplace = True)
actual = df2['Class']
pred = df2['Outlier']
confusion_matrix(actual,pred)
pd.crosstab(actual,pred,rownames=['Actual'],colnames=['Predicted'])
from sklearn.metrics import accuracy_score 
accuracy_score(actual, pred) 
# True Positive Rate

415/(77+415)
