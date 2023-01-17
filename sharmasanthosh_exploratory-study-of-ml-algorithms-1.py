# Supress unnecessary warnings so that the presentation looks clean

import warnings

warnings.filterwarnings('ignore')



# Read raw data from the file



import pandas #provides data structures to quickly analyze data

import numpy

#Since this code runs on Kaggle server, data can be accessed directly in the 'input' folder

#Read the train dataset

dataset_train = pandas.read_csv("../input/train.csv") 

dataset_test = pandas.read_csv("../input/test.csv") 



#Print all rows and columns. Dont hide any

pandas.set_option('display.max_rows', None)

pandas.set_option('display.max_columns', None)



#Display the first five rows to get a feel of the data

print(dataset_train.head(5))



#Learning:

#Data has a mix of continuous, categorical, ordinal, date, and discrete attributes

#NaN means that the data is missing
# Size of the dataframe



print(dataset_train.shape)

print(dataset_test.shape)

# We can see that there are 1460 instances having 81 attributes in train and 1459 instances in test
print(dataset_train.skew())

#Attribute LotArea is highly skewed. It could be corrected using log transform
#Constants

MED = 0  #median

ENC = 1  #encoded

SIZE = [MED, ENC] #list of types

SIZE_STR = ['Med','Enc']

DUMMY_STR = 'Dummy'



#Stores both list of median cols and list of encoded cols

dataset_train_list = [pandas.DataFrame(),pandas.DataFrame()]

dataset_test_list = [pandas.DataFrame(),pandas.DataFrame()]
#import plotting libraries

import seaborn as sns

import matplotlib.pyplot as plt



#import for Imputing missing values

from sklearn.preprocessing import Imputer



#import for one hot encoding

from sklearn.feature_extraction import DictVectorizer



#Column name of target

target = 'SalePrice'



def analyse(var,cat):

    

    if (cat == "categorical"):        



        # Imputer

        # Convert NA to the word 'DUMMY_STR' so that the median on 'target' is used

        dataset_train[var].fillna(inplace=True,value=DUMMY_STR)

        dataset_test[var].fillna(inplace=True,value=DUMMY_STR)

        

        #Obtain the labels

        labels = numpy.array(dataset_train[var].dropna().unique())

        

        #Rotate alignment of labels if there are too many to prevent overlap

        if(len(labels) < 12):

            # Plot the attribute against target

            sns.boxplot(y=target, x=var, data=dataset_train)    

        

        else:

            fig, ax = plt.subplots()

            #plot

            sns.boxplot(y=target, x=var, data=dataset_train)    

            #Sort as the plot is already sorted

            labels = sorted(labels)

            #Rotate vertical if there are too many labels which result in overlap

            ax.set_xticklabels(labels,rotation='vertical')

            plt.show()



        # Hypothesis

        # Nominal values must be transformed to numerical values. We can transform the nominal value using 

        # one hot encoding or we can replace the nominal value with the median value of the target in the boxplot

        

        # Word of caution

        # Ideally, the median must be taken only for the 90% of the train  dataset as it may potentially 

        # cause some data leakage into the validation set. 

        # But to maintain simiplicity of the presentation, this is overlooked



        # Obtain the median value of target for each class

        medians = dataset_train.groupby(var)[target].median()

        #print("\n\nMedians for")

        #print(medians)



        # Add median column name

        var_median = var+"_median"



        # Map from label to median

        dataset_train_list[MED][var_median] = dataset_train[var].map(medians)

        dataset_test_list[MED][var_median] = dataset_test[var].map(medians)

        #print(dataset_train_list[MED][var_median])

        #print(dataset_test_list[MED][var_median])



        # One-hot encoding using DictVectorizer



        # Obtain column name, value pairs (covert numerical into string)

        vals_train = dataset_train[var].apply(lambda x : {var : "%s" % x} )

        vals_test = dataset_test[var].apply(lambda x : {var : "%s" % x} )



        # Create Dict Vectorizer class

        dv = DictVectorizer(sparse=False)

        # Concatenate is used to ensure all labels in both test and train are used

        dv.fit(numpy.concatenate((vals_train,vals_test),axis=0))



        # Perform one-hot encoding

        new_data_train = dv.transform(vals_train)

        new_data_test = dv.transform(vals_test)



        # Obtain column names

        new_cols = dv.get_feature_names()



        # Add new columns

        for i, col in enumerate(new_cols):

            dataset_train_list[ENC][col] = new_data_train[:,i]

            dataset_test_list[ENC][col] = new_data_test[:,i]

            #print(dataset_train_list[ENC][col])

            #print(dataset_test_list[ENC][col])

    

    else:

        if(cat == "target"):

            for s in SIZE:

                dataset_train_list[s][var] = dataset_train[var]

                #print(dataset_train_list[s][var])

            

            sns.violinplot(data=dataset_train, size=7, y=var)    

        else:    

            if(cat == "date"):

                var_orig = var

                var = var+"_time"

                # Convert year into time until 'YrSold' and fill NaN with median

                for s in SIZE:

                    dataset_train_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(dataset_train["YrSold"] - dataset_train[var_orig]).transpose()

                    dataset_test_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(dataset_test["YrSold"] - dataset_test[var_orig]).transpose()

                    #print(dataset_train_list[s][var])

                    #print(dataset_test_list[s][var])

                

            elif(cat == "skewed"):

                var_orig = var

                var = var+"_log"

                # Log transform to correct skew , if NaN fill with median

                for s in SIZE:

                    dataset_train_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(numpy.log1p(dataset_train[var_orig])).transpose()

                    dataset_test_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(numpy.log1p(dataset_test[var_orig])).transpose()

                    #print(dataset_train_list[s][var])

                    #print(dataset_test_list[s][var])

                

            elif(cat == "continuous"):

                #If NaN, fill with median

                for s in SIZE:

                    dataset_train_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(dataset_train[var]).transpose()

                    dataset_test_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(dataset_test[var]).transpose()

                    #print(dataset_train_list[s][var])

                    #print(dataset_test_list[s][var])

                

            #Obtain the correlation for the attribute against target

            col_list = [target,var]

            d_corr = dataset_train_list[MED][col_list].corr().iloc[0,1]

            #KDE plot if good correlation else scatter plot

            if(d_corr > 0.15):

                sns.jointplot(data=dataset_train_list[MED], size=7, x=var,y=target , kind="kde" )

            else:

                sns.jointplot(data=dataset_train_list[MED], size=7, x=var,y=target  )



    #print("Train : Median : ")

    #print(dataset_train_list[MED].shape)

    #print(dataset_train_list[MED].columns)

    #print("Train : Encoded : ")

    #print(dataset_train_list[ENC].shape)

    #print(dataset_train_list[ENC].columns)

    #print("Test : Median : ")

    #print(dataset_test_list[MED].shape)

    #print(dataset_test_list[MED].columns)

    #print("Test : Encoded : ")

    #print(dataset_test_list[ENC].shape)

    #print(dataset_test_list[ENC].columns)
#Id is ignored as it is not useful

analyse(var="SalePrice",cat="target")
analyse(var="MSSubClass",cat="categorical")
analyse(var="MSZoning",cat="categorical")
analyse(var="LotFrontage",cat="continuous")
analyse(var="LotArea",cat="skewed")
analyse(var="Street",cat="categorical")
analyse(var="Alley",cat="categorical")
analyse(var="LotShape",cat="categorical")
analyse(var="LandContour",cat="categorical")
analyse(var="Utilities",cat="categorical")
analyse(var="LotConfig",cat="categorical")
analyse(var="LandSlope",cat="categorical")
analyse(var="Neighborhood",cat="categorical")
analyse(var="Condition1",cat="categorical")
analyse(var="Condition2",cat="categorical")
analyse(var="BldgType",cat="categorical")
analyse(var="HouseStyle",cat="categorical")
analyse(var="OverallQual",cat="continuous")
analyse(var="OverallCond",cat="continuous")
analyse(var="YearBuilt",cat="date")
analyse(var="YearRemodAdd",cat="date")
analyse(var="RoofStyle",cat="categorical")
analyse(var="RoofMatl",cat="categorical")
analyse(var="Exterior1st",cat="categorical")
analyse(var="Exterior2nd",cat="categorical")
analyse(var="MasVnrType",cat="categorical")
analyse(var="MasVnrArea",cat="continuous")
analyse(var="ExterQual",cat="categorical")
analyse(var="ExterCond",cat="categorical")
analyse(var="Foundation",cat="categorical")
analyse(var="BsmtQual",cat="categorical")
analyse(var="BsmtCond",cat="categorical")
analyse(var="BsmtExposure",cat="categorical")
analyse(var="BsmtFinType1",cat="categorical")
analyse(var="BsmtFinSF1",cat="continuous")
analyse(var="BsmtFinType2",cat="categorical")
analyse(var="BsmtFinSF2",cat="continuous")
analyse(var="BsmtUnfSF",cat="continuous")
analyse(var="TotalBsmtSF",cat="continuous")
analyse(var="Heating",cat="categorical")
analyse(var="HeatingQC",cat="categorical")
analyse(var="CentralAir",cat="categorical")
analyse(var="Electrical",cat="categorical")
analyse(var="1stFlrSF",cat="continuous")
analyse(var="2ndFlrSF",cat="continuous")
analyse(var="LowQualFinSF",cat="continuous")
analyse(var="GrLivArea",cat="continuous")
analyse(var="BsmtFullBath",cat="continuous")
analyse(var="BsmtHalfBath",cat="continuous")
analyse(var="FullBath",cat="continuous")
analyse(var="HalfBath",cat="continuous")
analyse(var="BedroomAbvGr",cat="continuous")
analyse(var="KitchenAbvGr",cat="continuous")
analyse(var="KitchenQual",cat="categorical")
analyse(var="TotRmsAbvGrd",cat="continuous")
analyse(var="Functional",cat="categorical")
analyse(var="Fireplaces",cat="continuous")
analyse(var="FireplaceQu",cat="categorical")
analyse(var="GarageType",cat="categorical")
analyse(var="GarageYrBlt",cat="date")
analyse(var="GarageFinish",cat="categorical")
analyse(var="GarageCars",cat="continuous")
analyse(var="GarageArea",cat="continuous")
analyse(var="GarageQual",cat="categorical")
analyse(var="GarageCond",cat="categorical")
analyse(var="PavedDrive",cat="categorical")
analyse(var="WoodDeckSF",cat="continuous")
analyse(var="OpenPorchSF",cat="continuous")
analyse(var="EnclosedPorch",cat="continuous")
analyse(var="3SsnPorch",cat="continuous")
analyse(var="ScreenPorch",cat="continuous")
analyse(var="PoolArea",cat="continuous")
analyse(var="PoolQC",cat="categorical")
analyse(var="Fence",cat="categorical")
analyse(var="MiscFeature",cat="categorical")
analyse(var="MiscVal",cat="continuous")
analyse(var="MoSold",cat="categorical")
analyse(var="YrSold",cat="categorical")
analyse(var="SaleType",cat="categorical")
analyse(var="SaleCondition",cat="categorical")
# Correlation tells relation between two attributes.



# Calculates pearson co-efficient for all combinations

data_corr = dataset_train_list[MED].corr()



# Set the threshold to select only highly correlated attributes

threshold = 0.68



# List of pairs along with correlation above threshold

corr_list = []



#Length of the list

med_cols = dataset_train_list[MED].columns

size = len(med_cols)



#Search for the highly correlated pairs

for i in range(0,size): #for 'size' features

    for j in range(i+1,size): #avoid repetition

        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):

            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index



#Sort to show higher ones first            

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))



#Print correlations and column names

for v,i,j in s_corr_list:

    print ("%s and %s = %.2f" % (med_cols[i],med_cols[j],v))



# Strong correlation is observed between the following pairs
# Scatter plot of only the highly correlated pairs

for v,i,j in s_corr_list:

    sns.jointplot(data=dataset_train_list[MED], size=7, x=med_cols[i],y=med_cols[j] , kind="kde" )

    plt.show()
#Import libraries for data transformations

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import Normalizer

    

#All features

X_all = [[],[]]

X_all_add = [[],[]]



#List of combinations

comb = [[],[]]



#Split the data into two chunks

from sklearn import cross_validation

    

#Validation chunk size

val_size = 0.1



#Use a common seed in all experiments so that same chunk is used for validation

seed = 0



for s in SIZE:

    #Extract only the values 

    array = dataset_train_list[s].values



    #Shape of the dataset

    r , c = dataset_train_list[s].shape

    print(dataset_train_list[s].shape)

    

    #Y is the target column, X has the rest

    X = array[:,1:]

    Y = array[:,0]



    #Split X and Y

    X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)

    

    #create an array which has indexes of columns for X

    i_cols = []

    for i in range(0,c-1):

        i_cols.append(i)

    #print("i_cols")

    #print(i_cols)

    

    #create an array which has names of columns for X

    cols = dataset_train_list[s].columns[1:]    

    #print(cols)

    

    #Add the original version of X which is not transformed to the list

    n = 'All'

    i_bin =[]

    X_all[s].append(['Orig',n, X_train,X_val,Y_train,Y_val,cols,i_cols,i_cols,i_bin])



    #create separate array for binary and continuous columns for X

    i_cols_binary = []

    i_cols_conti = []

    cols_binary = []

    cols_conti = []



    for i,col in enumerate(cols):

        #Presence of '=' means that the column is binary

        if('=' in col):

            i_cols_binary.append(i)

            cols_binary.append(col)

        else:

            i_cols_conti.append(i)

            cols_conti.append(col)



    #print('i_cols_conti')

    #print(i_cols_conti)

    #print('cols_conti')

    #print(cols_conti)

    #print('i_cols_binary')

    #print(i_cols_binary)

    #print('cols_binary')

    #print(cols_binary)

        

    #Preprocessing list

    prep_list = [('StdSca',StandardScaler()),('MinMax',MinMaxScaler()),('Norm',Normalizer())]

    for name, prep in prep_list:

        #Prevent data leakage by applying the transforms separately for X_train and X_val

        #Apply transform only for non-categorical data

        X_temp = prep.fit_transform(X_train[:,i_cols_conti])

        X_val_temp = prep.fit_transform(X_val[:,i_cols_conti])

        #Concatenate non-categorical data and categorical

        X_con = numpy.concatenate((X_temp,X_train[:,i_cols_binary]),axis=1)

        X_val_con = numpy.concatenate((X_val_temp,X_val[:,i_cols_binary]),axis=1)

        #Column name location would have changed. Hence overwrite 

        cols = numpy.concatenate((cols_conti,cols_binary),axis=0)

        #pandas.DataFrame(data=X_con,columns=cols).to_csv("trans%sType%sTrain.csv" % (name,s))

        #pandas.DataFrame(data=X_val_con,columns=cols).to_csv("trans%sType%sVal.csv" % (name,s))

        #Add this version of X to the list 

        X_all[s].append([name,n, X_con,X_val_con,Y_train,Y_val,cols,i_cols,i_cols_conti,i_cols_binary])



#print(X_all)        
# % of features to select for median and enc

ratio_list = [[0.25],[]]
#Feature selection only for median

for s in range(1):

    #List of feature selection models

    feat = []



    #List of names of feature selection models

    feat_list =[]



    #Import the libraries

    from sklearn.ensemble import ExtraTreesClassifier



    #Add ExtraTreeClassifiers to the list

    n = 'ExTree'

    feat_list.append(n)

    for val in ratio_list[s]:

        feat.append([n,val,ExtraTreesClassifier(n_estimators=100,max_features=val,n_jobs=-1,random_state=seed)])      



    #For all transformations of X

    for trans,n, X, X_val,Y_train,Y_val, cols, i_cols, conti,binr in X_all[s]:

        #For all feature selection models

        for name,v, model in feat:

            #Train the model against Y

            model.fit(X,Y_train)

            #Combine importance and index of the column in the array joined

            joined = []

            for i, pred in enumerate(list(model.feature_importances_)):

                joined.append([i,cols[i],pred])

            #Sort in descending order    

            joined_sorted = sorted(joined, key=lambda x: -x[2])

            #Starting point of the columns to be dropped

            rem_start = int(v*(len(cols)))

            #List of names of columns selected

            cols_list = []

            #Indexes of columns selected

            i_cols_list = []

            #Ranking of all the columns

            rank_list =[]

            #Split the array. Store selected columns in cols_list and removed in rem_list

            for j, (i, col, x) in enumerate(list(joined_sorted)):

                #Store the rank

                rank_list.append([i,j])

                #Store selected columns in cols_list and indexes in i_cols_list

                if(j < rem_start):

                    cols_list.append(col)

                    i_cols_list.append(i)

            #Sort the rank_list and store only the ranks. Drop the index 

            #Append model name, array, columns selected to the additional list        

            X_all_add[s].append([trans,name,X,X_val,Y_train,Y_val,cols_list,[x[1] for x in sorted(rank_list,key=lambda x:x[0])],i_cols_list,cols])    



    #Set figure size

    plt.rc("figure", figsize=(12, 10))



    #Plot a graph for different feature selectors        

    for f_name in feat_list:

        #Array to store the list of combinations

        leg=[]

        fig, ax = plt.subplots()

        #Plot each combination

        for trans,name,X,X_val,Y,Y_val,cols_list,rank_list,i_cols_list,cols in X_all_add[s]:

            if(name==f_name):

                plt.plot(rank_list)

                leg.append(trans+"+"+name)

        #Set the tick names to names of columns

        ax.set_xticks(range(len(cols)))

        ax.set_xticklabels(cols,rotation='vertical')

        #Display the plot

        plt.legend(leg,loc='upper left')    

        #Plot the rankings of all the features for all combinations

        plt.show()
import math



#Dictionary to store the RMSE for all algorithms 

mse = [[],[]]



#Scoring parameter

from sklearn.metrics import mean_squared_error
#Evaluation of various combinations of LinearRegression



#Import the library

from sklearn.linear_model import Ridge



algo = "Ridge"



#Add the alpha value to the below list if you want to run the algo

a_list = numpy.array([0.1])





for s in SIZE:

    for alpha in a_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            if(name == 'MinMax'):

                continue

            print(name)

            #Set the base model

            model = Ridge(random_state=seed)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % alpha )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of LinearRegression



#Import the library

from sklearn.linear_model import Lasso



algo = "Lasso"



#Add the alpha value to the below list if you want to run the algo

a_list = numpy.array([0.1])



for s in SIZE:

    for alpha in a_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            if(name == 'MinMax'):

                continue

            print(name)

            #Set the base model

            model = Lasso(alpha=alpha,random_state=seed)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % alpha )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of ElasticNet



#Import the library

from sklearn.linear_model import ElasticNet



algo = "Elastic"



#Add the alpha value to the below list if you want to run the algo

a_list = numpy.array([0.01])



for s in SIZE:

    for alpha in a_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            if(name == 'MinMax'):

                continue

            print(name)

            #Set the base model

            model = ElasticNet(alpha=alpha,random_state=seed)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % alpha )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of KNN



#Import the library

from sklearn.neighbors import KNeighborsRegressor



algo = "KNN"



#Add the N value to the below list if you want to run the algo

n_list = numpy.array([9])



for s in SIZE:

    for n_neighbors in n_list:



        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            print(name)

            #Set the base model

            model = KNeighborsRegressor(n_neighbors=n_neighbors,n_jobs=-1)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % n_neighbors )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of CART



#Import the library

from sklearn.tree import DecisionTreeRegressor



algo = "CART"



#Add the max_depth value to the below list if you want to run the algo

d_list = numpy.array([13])



for s in SIZE:

    for max_depth in d_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            print(name)

            #Set the base model

            model = DecisionTreeRegressor(max_depth=max_depth,random_state=seed)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % max_depth )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of SVM



#Import the library

from sklearn.svm import SVR



algo = "SVM"



#Add the C value to the below list if you want to run the algo

c_list = numpy.array([10000])



for s in SIZE:

    for C in c_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            if(name == 'Orig' or name =='Norm'):  #very poor results, spoils the graph

                continue

            print(name)

            #Set the base model

            model = SVR(C=C)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % C )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of Bagged Decision Trees



#Import the library

from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor



algo = "Bag"



#Add the n_estimators value to the below list if you want to run the algo

n_list = numpy.array([200])



for s in SIZE:

    for n_estimators in n_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            print(name)

            #Set the base model

            model = BaggingRegressor(n_jobs=-1,n_estimators=n_estimators)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % n_estimators )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of RandomForest

    

#Import the library

from sklearn.ensemble import RandomForestRegressor



algo = "RF"



#Add the n_estimators value to the below list if you want to run the algo

n_list = numpy.array([300])



for s in SIZE:

    for n_estimators in n_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            print(name)

            #Set the base model

            model = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % n_estimators )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of ExtraTrees



#Import the library

from sklearn.ensemble import ExtraTreesRegressor



algo = "ET"



#Add the n_estimators value to the below list if you want to run the algo

n_list = numpy.array([300])



for s in SIZE:

    for n_estimators in n_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            print(name)

            #Set the base model

            model = ExtraTreesRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % n_estimators )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of ExtraTrees



#Import the library

from sklearn.ensemble import AdaBoostRegressor



algo = "Ada"



#Add the n_estimators value to the below list if you want to run the algo

n_list = numpy.array([300])



for s in SIZE:

    for n_estimators in n_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            print(name)

            #Set the base model

            model = AdaBoostRegressor(n_estimators=n_estimators,random_state=seed)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % n_estimators )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of SGB



#Import the library

from sklearn.ensemble import GradientBoostingRegressor



algo = "SGB"



#Add the n_estimators value to the below list if you want to run the algo

n_list = numpy.array([300])



for s in SIZE:

    for n_estimators in n_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            print(name)

            #Set the base model

            model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=seed)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % n_estimators )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of XGB



#Import the library

from xgboost import XGBRegressor



algo = "XGB"



#Add the n_estimators value to the below list if you want to run the algo

n_list = numpy.array([300])



for s in SIZE:

    for n_estimators in n_list:

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            print(name)

            #Set the base model

            model = XGBRegressor(n_estimators=n_estimators,seed=seed)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo + " %s" % n_estimators )

##Plot the MSE of all combinations for both types in the same figure

#fig, ax = plt.subplots()

#for s in SIZE:

#    plt.plot(mse[s])

##Set the tick names to names of combinations

#ax.set_xticks(range(len(comb[s])))

#ax.set_xticklabels(comb[s],rotation='vertical')

##Plot the accuracy for all combinations

#plt.legend(SIZE_STR,loc='best')    

#plt.show()    
#Evaluation of various combinations of multi-layer perceptrons



#Import libraries for deep learning

from keras.wrappers.scikit_learn import KerasRegressor

from keras.models import Sequential

from keras.layers import Dense



# define baseline model

def baseline(v):

     # create model

     model = Sequential()

     model.add(Dense(v, input_dim=v, init='normal', activation='relu'))

     model.add(Dense(1, init='normal'))

     # Compile model

     model.compile(loss='mean_squared_error', optimizer='adam')

     return model



# define smaller model

def smaller(v):

     # create model

     model = Sequential()

     model.add(Dense(v/2, input_dim=v, init='normal', activation='relu'))

     model.add(Dense(1, init='normal', activation='relu'))

     # Compile model

     model.compile(loss='mean_squared_error', optimizer='adam')

     return model



# define deeper model

def deeper(v):

 # create model

 model = Sequential()

 model.add(Dense(v, input_dim=v, init='normal', activation='relu'))

 model.add(Dense(v/2, init='normal', activation='relu'))

 model.add(Dense(1, init='normal', activation='relu'))

 # Compile model

 model.compile(loss='mean_squared_error', optimizer='adam')

 return model



# Optimize using dropout and decay

from keras.optimizers import SGD

from keras.layers import Dropout

from keras.constraints import maxnorm



def dropout(v):

    #create model

    model = Sequential()

    model.add(Dense(v, input_dim=v, init='normal', activation='relu',W_constraint=maxnorm(3)))

    model.add(Dropout(0.2))

    model.add(Dense(v/2, init='normal', activation='relu', W_constraint=maxnorm(3)))

    model.add(Dropout(0.2))

    model.add(Dense(1, init='normal', activation='relu'))

    # Compile model

    sgd = SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)

    model.compile(loss='mean_squared_error', optimizer=sgd)

    return model



# define decay model

def decay(v):

    # create model

    model = Sequential()

    model.add(Dense(v, input_dim=v, init='normal', activation='relu'))

    model.add(Dense(1, init='normal', activation='relu'))

    # Compile model

    sgd = SGD(lr=0.1,momentum=0.8,decay=0.01,nesterov=False)

    model.compile(loss='mean_squared_error', optimizer=sgd)

    return model



est_list = [('MLP',baseline),('smaller',smaller),('deeper',deeper),('dropout',dropout),('decay',decay)]



for s in SIZE:

    for mod, est in est_list:

        algo = mod

        #Accuracy of the model using all features

        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:

            if(name != 'Orig' or mod != 'MLP'):

               continue

            print(name+" "+mod)

            #Set the base model

            model = KerasRegressor(build_fn=est, v=len(cols), nb_epoch=10, verbose=0)

            model.fit(X_train,Y_train)

            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))

            mse[s].append(result)

            print(name + " %s" % result)

            comb[s].append(name+" "+algo )



#Plot the MSE of all combinations for both types in the same figure

fig, ax = plt.subplots()

for s in SIZE:

    plt.plot(mse[s])

#Set the tick names to names of combinations

ax.set_xticks(range(len(comb[s])))

ax.set_xticklabels(comb[s],rotation='vertical')

#Plot the accuracy for all combinations

plt.legend(SIZE_STR,loc='best')    

plt.show()    
# Make predictions using one-hot encoding of the Original version of the dataset along 

# with RandomForest algo (300 estimators) as it gave the best estimated performance        



#Extract only the values 

array = dataset_train_list[ENC].values

    

#Y is the target column, X has the rest

X = array[:,1:]

Y = array[:,0]



n_estimators = 300



#Best model definition

best_model = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)

best_model.fit(X,Y)



#Extract the ID for submission file

ID = dataset_test['Id']



#Use only values

X_test = dataset_test_list[ENC].values



#Make predictions using the best model

predictions = best_model.predict(X_test)

# Write submissions to output file in the correct format

with open("submission.csv", "w") as subfile:

    subfile.write("Id,SalePrice\n")

    #print("Id,SalePrice\n")

    for i, pred in enumerate(list(predictions)):

        #print("%s,%s\n"%(ID[i],pred))

        subfile.write("%s,%s\n"%(ID[i],pred))