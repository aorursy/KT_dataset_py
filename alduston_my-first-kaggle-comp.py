import pandas as pd

import numpy as np

import os

import seaborn as sns

import matplotlib.pyplot as plt

import numbers

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn import neighbors

import scipy.stats as stats

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score

import math

import random

import shutil

from sklearn import tree







#os.remove("/kaggle/working/prediction1.csv")

#os.remove("/kaggle/working/prediction5.csv")

#os.remove("/kaggle/working/prediction2.csv")

#os.remove("/kaggle/working/prediction0.csv")

#os.remove("/kaggle/working/prediction4.csv")

#os.remove("/kaggle/working/prediction6.csv")

















train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_shape=train.shape

test_shape=test.shape



print("shape of train df is",train.shape)



print("shape of test df is",test.shape)
columns=train.columns



print (columns)
head=train.head()



print(head)


def without(big, little):

    result = []

    

    #loop through big list, return stuff that isnt in smaller list

    

    for item in big:

        if item not in little:

            result.append(item)

    

    # need this for dealing with technichal error with results of len 1

    

    if len(result)==1:



        return result[0]

    else:

        return result





def labelled(df,label_name):

    

    # get number of rows in df 

    

    l=len(df)

    label_col=[]



    # loop over rows append label name to list for each row,

    

    for k in range(l):

        label_col.append(label_name)

        

       

    # list needs to become array before we add it as column

    

    label_col=np.asarray(label_col)



    #Add as column to df with column name 'label'

    

    df['label']=label_col



    return df

def preprocess(train,test):



    #getting testing and training column list

    all_columns = train.columns

    predictor_col=test.columns

    

    # getting the y or ouptut column using "without" function

    

    ycol=without(all_columns,predictor_col)

    



    # seperating train into x and y values using .drop() method

    X_train=train.drop([ycol],axis=1)

    y=train[ycol]



    

    #Adding label column labelling data as test or train

    

    X_train_labelled=labelled(X_train,"train")

    X_test_labelled=labelled(test,"test")







    # combining all X data into one df

    

    X=pd.concat([X_train_labelled,X_test_labelled])

    

    # returning X and y in list, as we want them

    

    result=[X,y]



    return result





#applying to our data 



X,y=preprocess(train,test)



print("shape of  X df is",X.shape,",expected shape is (",len(train)+len(test),",",len(columns),")")



print("shape of y df is",y.shape,",expected shape is (",len(train),",)" )
def na_count(col_data):

    

    # Gets list of all na values in  array 



    na_list=pd.isna(col_data)

    

     # Gets proportion of NA by dividing by total lenght  of array

    

    result=sum(na_list)/len(col_data)

    

    #return proportion

    

    return result

    

    
def na_distribution(df,plot=False,cutoff=.25):



    result=[]

    nums=[]

    columns=df.columns



     # lopping through all columns, recoding columns name and proportion of na values

    

    for column in columns:

        col_data=df[column]



        num=na_count(col_data)

        nums.append(num)



        info=[column,na_count(col_data)]

        result.append(info)

        

    #sorting list 



    result.sort()

    nums.sort()



    

    # if given plot argument of true, plots na proportion per column, 

    #with line at cutoff value for reference in descending order

    

    if plot:

        plt.plot(nums)

        plt.axhline(y=cutoff,color='r')

        plt.show()

        

       



        

    # return list of na proportion per column

    

    return result



    

def na_remove(df,cutoff=.25):



    # first lets get proportions for each column using na_distribution

    

    bad_columns=[]

    na_dist=na_distribution(df)



    # loops through columns, getting list of "bad columns" whose porportion of na

    #is above acceptable threshold, default is .25

    

    for data in na_dist:

        if data[1]>cutoff:



            bad_columns.append(data[0])

        

    # loops through bad column list, dropping them from our inputed df



    for bad_column in bad_columns:



        df=df.drop([bad_column],axis=1)

        

    #returns this edited dataframe

    

    return df
#getting initial distribution, plotting na proportion.



initial_dist=na_distribution(X,plot=True)



#removing columns with proportiong greater than default .25 cutoff



X=na_remove(X)



# plotting again to see change



new_dist=na_distribution(X,plot=True)



def is_num(column,cutoff=7):



    alpha=False

    beta=False

    

    l=len(column)

    num_num=0



    # we are using numbers module to check if stuff in column is number

    

    Number = numbers.Number



    # looping over column keeping count of how many entries are numerical

    

    for item in column:



        if isinstance(item,Number):

            num_num+=1

            



    # get proportion of numerical items, if proportion is greater than .85 we call this a numerical column

    # reason we arent insisting on 100% is that some values may have been entered as string on accident like one hundred

    # insted of 100, this gives us flexibility to deal with this.

    

    num_prop=num_num/l



    if num_prop>.85: 

        alpha=True

        

    levels=set(column)

    num_levels=len(levels)

    

    if num_levels>cutoff:

        

        beta=True



    # return true if proportion of numerical items is greater than .85 and more than 7 levels

    

    if alpha and beta:

        

        return True



    else:

        return False


def numcat_split(df):

    

    #setting up empty data frames to be filled and getting column list



    numerical=pd.DataFrame()

    categorical=pd.DataFrame()

    columns=df.columns



    # lopping over column, adding numerical to numerical df, else to categorical

    

    for column in columns:



        col_data=df[column]



        if is_num(col_data):

            numerical[column]=col_data



        else:



            categorical[column]=col_data



    #return list containing 2 new dataframes



    return [numerical,categorical]

X_num,X_cat=numcat_split(X)



print("Shape of our numerical data is",X_num.shape, "meaning we have", len(X_num.columns),"numerical columns")



print(' ')



print("Shape of our categorical data is",X_cat.shape, "meaning we have", len(X_cat.columns),"categorical columns")

      

print(' ')



print('Expected total columns is',len(X.columns),",actual is",len(X_num.columns)+len(X_cat.columns))



def nan_to_mean(column):

    

    # get version of column with no na values and get its mean

    

    nona=column[~np.isnan(column)]

    mean=np.mean(nona)



    # loops through column, replacing nan with mean 

    

    result=[]

    for item in column:

        if np.isnan(item):

            result.append(mean)

        else:

            result.append(item)



    #return new column

    

    result=np.asarray(result)



    return result


def mean_df(df) :



    # creating blank df to build up with converted columns, and get column list

    

    result=pd.DataFrame()

    columns=df.columns



    #looping over columns converting with nan_to_mean and adding to our result df

    for column in columns:



        col_data=df[column]

        mean_data=nan_to_mean(col_data)



        result[column]=mean_data

        

    # return df of converted columns

    

    return result

        
def prop_0(column):

    

    l=len(column)

    

    zero_count=0

    

    for item in column:

        if item==0:

            

            zero_count+=1

    

    prop= zero_count/l

    return prop
def df_prop_0(df,plot=False):

    

    columns=df.columns

    

    result=[]

    props=[]

    

    for column in columns:

        

        col_data=df[column]

        

        prop=prop_0(col_data)

        report=[column,prop]

        

        props.append(prop)

        result.append(report)

    

    if plot:

        props.sort()

        plt.plot(props)

        plt.show()

    

    

    return result

        

        

    
# getting initial shape and NA counts



print("initial shape is ",X_num.shape)



initial_dist=na_distribution(X_num,plot=True,cutoff=.1)



# converting with mean_df



X_num=mean_df(X_num)



# getting new shape and NA counts



print("new shape is ",X_num.shape)



new_dist=na_distribution(X_num,plot=True,cutoff=.1)





zero_props=df_prop_0(X_num)

def cat_from_zero(column):

    

    cat_column=[]

    

    for item in column:

     

        if item !=0:

            cat_column.append(1)

            

        else:

            

            cat_column.append(item)

    

    cat_column=np.asarray(cat_column)

    return cat_column

        
def zero_to_cat(df,cutoff=.4):

    

    props=df_prop_0(df)

    add_cat=[]

    

    

    for item in props:

        if item[1]>cutoff:

            add_cat.append(item[0])

        

    for column in add_cat:

        

        col_name=column+"cat"

        

        col_data=df[column]

        col_data=cat_from_zero(col_data)

        

        df[col_name]=col_data

    

    return df
def clean_cat_col(column):



    clean_col=[]



    #loop throgh column change "nan" to "none" and leave other values unchanged

    

    for item in column:



        if str(item)=="nan":

            clean_col.append("none")

    



        else:

            clean_col.append(item)



    clean_col=np.asarray(clean_col)



    return clean_col

def clean_cat_df(df):



    #set up empty output df and get column list

    

    clean_df=pd.DataFrame()

    columns=df.columns



    # loop through columns, converting each with clean_cat_col

    

    for column in columns:



        col_data=df[column]

        clean_df[column]=clean_cat_col(col_data)



    return clean_df
# getting initial shape and NA counts



print("initial shape is ",X_cat.shape)



old_dist=na_distribution(X_cat,plot=True,cutoff=.1)



# converting with clean_cat_df



X_cat=clean_cat_df(X_cat)



# getting new shape and NA counts



print("new shape is ",X_cat.shape)



new_dist=na_distribution(X_cat,plot=True,cutoff=.1)


def column_namer(name,len):

    

    col_names=[]

    

    #loop through numer of dummies, gets name for each, name is just orignal column name +index 

    for i in range(len):



        col_name=name+str(i)

        col_names.append(col_name)

    

    #return list of column names

    

    return col_names
def get_dummydf(df):



    # first make sure our input df is clean of na's with clean_cat_df, and get column list

    clean_df = clean_cat_df(df)

    columns = clean_df.columns



    #create empty ouput df for us to fill with dummy vars

    

    dummy_df = pd.DataFrame()



    # loop through columns, getting dummy columns with pd.pd.get_dummies()

    for column in columns:



        col_data = clean_df[column]

        dummies = pd.get_dummies(col_data)

        columns = dummies.columns



        l=len(columns)

        

        #for each column get names for dummie columns with column_namer, set 0 as start index

        

        names=column_namer(column,l)

        index=0

        

        # give dummies correct names and add them to output df 

        for column in columns:



            data = dummies[column].values

            name=names[index]



            dummy_df[name] = data

            index=index+1



    

    #return output df made up of dummy columns

    

    return dummy_df



def df_proccesed(train,test,na_cut=.25,zcat=True):

    

    #getting X and y from train, test using preprocessed



    X, y = preprocess(train, test)



    

    #Removing na columns and splitting X into numerical and categorical

    

    X = na_remove(X,na_cut)

    

    X_num,X_cat = numcat_split(X)



    # cleaning numericals with mean_df  getting categorical dumm

    

    X_num = mean_df(X_num)

    

    if zcat:

        

        X_num=zero_to_cat(X_num)

        



    # getting categorical dummiesd with get_dummy_df

    

    X_cat = get_dummydf(X_cat)



    #combining 2 together into one df with all features

    

    result = pd.concat([X_num, X_cat], axis=1)





    #finding how many train values there are, using label column we created earlier with labelled function

    





    train=sum(result['label1'])



    # splitting X back into train and test rows usin

    X_train = result[:train]

    X_test = result[train:]





    #returning all data we need to feed model

    

    result = [X_train, X_test, y]



    return result
#getting processed data from train and test using df_processed



X_train, X_test, y= df_proccesed(train,test)





print(X_train.shape)



print(X_test.shape)



X_train.head()

def get_predictions0(data,model=LinearRegression(),ytran=True):

    



    # data is going to be output to our df_proccesed function, this outputs [X_train, X_test, y]

    

    X_train,X_test,y=data

    

    if ytran:

        y=np.log(y)



    #fitting model to our training data

    model.fit(X_train, y)

    

    #getting predictions for our test data 

    predictions = model.predict(X_test)

    

    if ytran:

        predictions=np.exp(predictions)

    

    #code below just gets predictions into acceptable format for submission

    start_index = len(X_train) + 1

    stop_index=len( X_train)+len(X_test) + 1

    Predictions = pd.DataFrame(predictions, columns=["SalePrice"], index=range(start_index, stop_index))

    Predictions.index.name = "Id"



    # return predictions in acceptable format

    

    return Predictions
#getting X_train, X_test, y from train, test

data=df_proccesed(train,test)



# getting predictions for our data and the default model, LinearRegression()



prediction0=get_predictions0(data)



# now on your computer save these to csv and submit just using 

prediction0.to_csv("prediction0.csv")


def k_fold_crossval(data,model=LinearRegression(),k=4,printout=True):

    

    #set X_train as our X data and y as y, 

    #X_test not used since cross vallidation requires y values for all entries 

    

    X,y=data[0],data[2]

    

    

    #Get array of scores from k fold cross validation, note this has strange error for k>4

    # it may be some kind of overflow error, I recommend just using the k=4 default arg

    

    scores = cross_val_score(model, X, y, cv=k)

    

    mean=scores.mean()

    std= scores.std()

    

    low=mean-(2*std)

    high=mean+(2*std)

    

    result=[low,mean,high]

    

    if printout:

        

        print("score is",mean,"+/-",(2*std),)

    

        print('range is',[low,high])

    

    

    return result



    

        
result=k_fold_crossval(data,k=4)
def feature_score(data,feature,model=LinearRegression(),k=4):

    

    X=(data[0])[feature]

    X=X.values.reshape(-1,1)

    

    y=data[2]

    y=y.values.reshape(-1,1)

    

    scores = cross_val_score(model, X, y, cv=k)

    mean=scores.mean()

    

    return mean

    

    
def rank_features(data,model=LinearRegression(),plot=True,plotrange=15):

    

    X,y=data[0],data[2]



    columns=X.columns

    

    result=[]

    scores=[]

    for feature in columns:

        

        score=feature_score(data,feature)

        

        scores.append(score)

        

        report=[score,feature]

        result.append(report)

        result.sort(reverse=True)

        

        

    if plot:

        

        scores=np.asarray(scores)

        indices = np.argsort(scores)[::-1][0:plotrange]

    

        

        plt.figure()

        plt.title("Feature importances")

        plt.barh(range(plotrange), scores[indices][::-1],

        color="r",  align="center")

        plt.yticks(range(plotrange), columns[indices][::-1])

    

        plt.xlim([0, 1])

        plt.show()

    

    

    return result

ranks=rank_features(data)
def drop_neg(data,model=LinearRegression()):

    

    X_train,X_test,y=data

    

    ranks=rank_features(data,model,plot=False)

    

    for item in ranks:

        if item[0]<0:

            column=item[1]

            X_train=X_train.drop([column],axis=1)

            X_test=X_test.drop([column],axis=1)

    

    data=[X_train, X_test, y]

    

    return data

print("For data:")



k_fold_crossval(data)



print(" ")



print("For data_new:")



data_new=drop_neg(data)



k_fold_crossval(data_new)



prediction1=get_predictions0(data_new)

prediction1.to_csv("prediction1.csv")

def data_copy(data):

    X_train, X_test, y=data

    

    X_train_copy=pd.DataFrame.copy(X_train,deep=True)

    X_test_copy=pd.DataFrame.copy(X_test,deep=True)

    

    return [X_train_copy,X_test_copy,y]
def log_feature(data,feature):

    

    data_temp=data_copy(data)

    

    X_train,X_test,y=data_temp

    

    min_train=min(X_train[feature])

    min_test=min(X_test[feature])

    

    Min=min([min_train,min_test])

    

    if Min>0:

        

        X_train[feature]=np.log(X_train[feature])

        X_test[feature]=np.log(X_test[feature])

    

    else:

        

        X_train[feature]=np.log(X_train[feature]+1)

        X_test[feature]=np.log(X_test[feature]+1)

        

    

    result=[X_train,X_test,y]

    

    return result
def get_prob_plot(data,feature,transform=False):

  

    

    if transform:

        data=log_feature(data,feature)

        X=pd.concat([data[0],data[1]])

        res = stats.probplot(X[feature], plot=plt)

        

    else:

        X=pd.concat([data[0],data[1]])

        res = stats.probplot(X[feature], plot=plt)

        

    
plot=stats.probplot(y, plot=plt)
def df_proccesed1(train,test,na_cut=.25,ytran=True,drop=True):

    

    #getting X and y from train, test using preprocessed



    X, y = preprocess(train, test)



    if ytran:

        

        y=np.log(y)

        

    

    #Removing na columns and splitting X into numerical and categorical

    

    X = na_remove(X,na_cut)

    

    X_num,X_cat = numcat_split(X)



    # cleaning numericals with mean_df  getting categorical dumm

    X_num = mean_df(X_num)



    # getting categorical dummiesd with get_dummy_df

    

    X_cat = get_dummydf(X_cat)



    #combining 2 together into one df with all features

    

    result = pd.concat([X_num, X_cat], axis=1)





    #finding how many train values there are, using label column we created earlier with labelled function

    

    train=sum(result['label1'])



    # splitting X back into train and test rows usin

    X_train = result[:train]

    X_test = result[train:]





    #returning all data we need to feed model

    

    result = [X_train, X_test, y]

    

    if drop:

        

        

        result=drop_neg(result)

        



    return result







def get_predictions1(data,model=LinearRegression(),ytran=True):

    

    

    # data is going to be output to our df_proccesed function, this outputs [X_train, X_test, y]

    

    X_train,X_test,y=data

    



    #fitting model to our training data

    model.fit(X_train, y)

    

    #getting predictions for our test data 

    predictions = model.predict(X_test)

    

    if ytran:

        

        predictions=np.exp(predictions)

    

    #code below just gets predictions into acceptable format for submission

    start_index = len(X_train) + 1

    stop_index=len( X_train)+len(X_test) + 1

    Predictions = pd.DataFrame(predictions, columns=["SalePrice"], index=range(start_index, stop_index))

    Predictions.index.name = "Id"



    # return predictions in acceptable format

    

    return Predictions
def transform_guesser(data,cut_val=0.005,early_cut=40):

    

    X_train, X_test, y = data



    features = X_train.columns

    to_transform = []

    

    #Getting our data and initializing list of columns to apply log transform



    for feature in features[0:early_cut]:

        

        

        data_new = log_feature(data, feature)



        val1 = k_fold_crossval(data, printout=False)

        

        val2 = k_fold_crossval(data_new, printout=False)

        

        diff=val2[1]-val1[1]

        

        # getting cross val score for data with and without a column being transferred for each column

       



        if (diff>cut_val) :

            

            to_transform.append(feature)

             

        # if score goes up by more than cutoff value, here .005, we apend said column's name

        #to list of columns to transform

             

             

             

    for feature in to_transform:

        data = log_feature(data, feature)

    

    #transforming all columns in the list

    

    return data



    # returning data with transformed columns
def poly_feature(data,feature1,feature2,centered=False):

    

    # this function adds polynomial feature made by multiplying 2 feature columns, feature 1 and feature 2, together

    data_temp=data[:]

    X_train, X_test, y = data_temp

    

    train_col1=X_train[feature1]

    train_col2=X_train[feature2]

    

    test_col1= X_test[feature1]

    test_col2=X_test[feature2]

    

    train_poly_col=np.multiply(train_col1,train_col2)

    

    test_poly_col=np.multiply(test_col1,test_col2)

    

    

    if centered:

        

        mean_train=train_poly_col.mean()

        train_poly_col=train_poly_col-mean_train

        

        mean_test=test_poly_col.mean()

        test_poly_col=test_poly_col-mean_test

        

    

    

    poly_name=feature1+'x'+feature2

    

    

    X_train[poly_name]=train_poly_col

    

    

    X_test[poly_name]=test_poly_col

    

   

    

    result=[X_train, X_test, y]

    

    

    return result
def best_poly_k(data,k=4,features_considered=8,reresult=False,centered=False,printout=True):

    

    # appends best k polynomial features to data, as measured by improvement in cross val score

    

    ranks=rank_features(data,plot=False)[0:features_considered]

    

    poly_added=[]

    result=[]

    

    for i in range(k):

        

        best_score=0

        

        for rank1 in ranks:

            feature1=rank1[1]

            

            for rank2 in ranks:

                

                feature2=rank2[1]

                

                features=[feature1,feature2]

                

                data_temp=data_copy(data)

                

                data_temp=poly_feature(data_temp,feature1,feature2,centered=centered)

                

                score=k_fold_crossval(data_temp,printout=False)[1]

                

                if (score>best_score) and (features not in poly_added) :

                    

                    best_score=score

                    bestf_1=feature1

                    bestf_2=feature2

                    

                    report=[best_score,[bestf_1,bestf_2]]

                    

                    

        data=poly_feature(data,bestf_1,bestf_2)    

        

        if printout:

            

            print("report:",report)

                     

        

        poly_added.append([bestf_1,bestf_2])

        poly_added.append([bestf_2,bestf_1])

        result.append(report)

        

    if reresult:

        

        return [data,result]

    

    else:

          return data



print("data0")

data0=df_proccesed(train,test)

val0=k_fold_crossval(data0)

print(' ')



print("data1")

data1=df_proccesed1(train,test,drop=False)

val1=k_fold_crossval(data1)

print(' ')



print("data2")

data2=df_proccesed1(train,test,drop=True)

val2=k_fold_crossval(data2)

print(' ')





print("data3")

data3=transform_guesser(data2)

val3=k_fold_crossval(data3)

print(' ')



print("data4")

data4=best_poly_k(data3,k=20,features_considered=20)

val4=k_fold_crossval(data4)



print(' ')





prediction4=get_predictions1(data4)

prediction4.to_csv('prediction4.csv')



gmodel=GradientBoostingRegressor

params={'n_estimators': 750,'max_depth': 3,'min_samples_split': 5, 'learning_rate': 0.1,'loss': 'ls'}

model=gmodel(**params)





predictions5=get_predictions1(data4,model=model)

predictions5.to_csv('prediction5.csv')



prediction6=(predictions5+prediction4)/2

prediction6.to_csv('prediction6.csv')