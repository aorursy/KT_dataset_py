# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import math

import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error



from sklearn.cluster import KMeans





def rmsle(y, y_pred):

        assert len(y) == len(y_pred)

        terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

        return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

    



def fix_target(frame, key, target, new_target_name="target"):

    import numpy as np



    corrections = 0

    group_keys = frame[ key].values.tolist()

    target = frame[target].values.tolist()



    for i in range(1, len(group_keys) - 1):

        previous_group = group_keys[i - 1]

        current_group = group_keys[i]



        previous_value = target[i - 1]

        current_value = target[i]

        if current_group == previous_group:

                if current_value<previous_value:

                    current_value=previous_value

                    target[i] =current_value





        target[i] =max(0,target[i] )#correct negative values



    frame[new_target_name] = np.array(target)

    

    

def rate(frame, key, target, new_target_name="rate"):

    import numpy as np





    corrections = 0

    group_keys = frame[ key].values.tolist()

    target = frame[target].values.tolist()

    rate=[1.0 for k in range (len(target))]



    for i in range(1, len(group_keys) ):

        previous_group = group_keys[i - 1]

        current_group = group_keys[i]



        previous_value = target[i - 1]

        current_value = target[i]

         

        if current_group == previous_group:

                if previous_value!=0.0:

                     rate[i]=current_value/previous_value



                 

        rate[i] =max(1,rate[i] )#correct negative values



    frame[new_target_name] = np.array(rate)

    

def difference(frame, key, target, new_target_name="diff"):

    import numpy as np





    corrections = 0

    group_keys = frame[ key].values.tolist()

    target = frame[target].values.tolist()

    rate=[0 for k in range (len(target))]



    for i in range(1, len(group_keys) ):

        previous_group = group_keys[i - 1]

        current_group = group_keys[i]



        previous_value = target[i - 1]

        current_value = target[i]

         

        if current_group == previous_group:

                rate[i]=np.log1p(max(0,current_value-previous_value))#max(0,current_value-previous_value)#

          

        rate[i] =max(0,rate[i] )



    frame[new_target_name] = np.array(rate)    

    

    

def get_data_by_key(dataframe, key, key_value, fields=None):

    mini_frame=dataframe[dataframe[key]==key_value]

    if not fields is None:                

        mini_frame=mini_frame[fields].values

        

    return mini_frame



directory="/kaggle/input/covid19-global-forecasting-week-4/"

model_directory="model"

us_county="/kaggle/input/covid19-us-county-jhu-data-demographics/covid_us_county"

geo_dir=None

extra_stable_columns=None

group_by_columns=None



if not group_by_columns is None and "continent" in group_by_columns:

    assert not geo_dir is None



train=pd.read_csv(directory + "train.csv", parse_dates=["Date"] , engine="python")

test=pd.read_csv(directory + "test.csv", parse_dates=["Date"], engine="python")



train["key"]=train[["Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)

test["key"]=test[["Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)

if not geo_dir is None:

    region_metadata=pd.read_csv(geo_dir+"region_metadata.csv")

    km=KMeans(n_clusters=30, init='k-means++', n_init=10, max_iter=300, tol=0.0001,

              precompute_distances='auto', verbose=0, random_state=1234, copy_x=True, n_jobs=None, algorithm='auto')

    region_metadata["clusters"]=km.fit_predict(region_metadata[["lat","lon"]].values)    

    

    train=pd.merge(train,region_metadata, how="left", left_on=["Province_State","Country_Region"], right_on=["Province_State","Country_Region"] )

    test=pd.merge(test,region_metadata, how="left", left_on=["Province_State","Country_Region"], right_on=["Province_State","Country_Region"] )



    



group_names=None



if  not group_by_columns is None and len(group_by_columns)>0:

    group_names=[]

    for group in group_by_columns:

        groupss=train[["Date", group,target1,target2]]

        grp=groupss.groupby(["Date", group], as_index=False).sum()

        grp.columns=["Date", group,group +"_" + target1,group +"_" + target2 ]

        train=pd.merge(train,grp, how="left", left_on=["Date", group,group], right_on=["Date", group,group] )

        #group_names+=[group +"_" + target1,group +"_" + target2]

        for gr in [group +"_" + target1,group +"_" + target2]:

            rate(train, key, gr, new_target_name="rate_" +gr ) 

            group_names+=["rate_" +gr]

        train.to_csv(directory + "train_plus_groups.csv", index=False)

        

        

    

#last day in train

max_train_date=train["Date"].max()

max_test_date=test["Date"].max()

horizon=  (max_test_date-max_train_date).days

print ("horizon", int(horizon))





#test_new=pd.merge(test,train, how="left", left_on=["key","Date"], right_on=["key","Date"] )

#train.to_csv(directory + "transfomed.csv")



target1="ConfirmedCases"

target2="Fatalities"



key="key"





train_supplamanteary2=None

new_data_file2=[us_county]



if not new_data_file2 is None:

    

    trains=pd.read_csv(directory + "train.csv", parse_dates=["Date"] , engine="python")

    tests=pd.read_csv(directory + "test.csv", parse_dates=["Date"], engine="python")



    trains["key"]=train[["Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)

    tests["key"]=test[["Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)

    train_supplamanteary2=None

    for new_data_files in new_data_file2:

        sumplementary_data=pd.read_csv( new_data_files +".csv" , parse_dates=["date"] , engine="python")

        sumplementary_data.columns=["fips","Province_State","Country_Region","lat","long","Date",

                                    "ConfirmedCases","state_code","Fatalities"]

 

        sumplementary_data["key"]=sumplementary_data[["Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)              

        sumplementary_data=sumplementary_data[["Date","key","ConfirmedCases","Fatalities","Province_State","Country_Region"]]        

        sumplementary_data["Id"]=[k for k in range(999999999,sumplementary_data.shape[0]+999999999)]



        sumplementary_data=sumplementary_data[sumplementary_data.Date>= '2020-02-20']



        if train_supplamanteary2 is None:

            train_supplamanteary2=pd.concat([trains.reset_index(drop=True), sumplementary_data.reset_index(drop=True)], axis=0).reset_index(drop=True)

        else :

            train_supplamanteary2=pd.concat([train_supplamanteary2.reset_index(drop=True), sumplementary_data.reset_index(drop=True)], axis=0).reset_index(drop=True)

             

    print(train_supplamanteary2.head(5))

    train_supplamanteary2 = train_supplamanteary2.sort_values(["Country_Region","key","Id"], ascending = (True, True, True))



    fix_target(train_supplamanteary2, key, target1, new_target_name=target1)

    fix_target(train_supplamanteary2, key, target2, new_target_name=target2)  

    rate(train_supplamanteary2, key, target1, new_target_name="rate_" +target1 )

    rate(train_supplamanteary2, key, target2, new_target_name="rate_" +target2 ) 

    difference(train_supplamanteary2, key, target1, new_target_name="diff_" +target1 )

    difference(train_supplamanteary2, key, target2, new_target_name="diff_" +target2 )    

     

    unique_keys_sup2=train_supplamanteary2[key].unique()

    print(len(unique_keys_sup2))

    if not geo_dir is None:

        region_metadata=pd.read_csv(geo_dir+"region_metadata.csv")

        km=KMeans(n_clusters=30, init='k-means++', n_init=10, max_iter=300, tol=0.0001,

              precompute_distances='auto', verbose=0, random_state=1234, copy_x=True, n_jobs=None, algorithm='auto')

        region_metadata["clusters"]=km.fit_predict(region_metadata[["lat","lon"]].values)          

        train_supplamanteary2=pd.merge(train_supplamanteary2,region_metadata, how="left", left_on=["Province_State","Country_Region"], right_on=["Province_State","Country_Region"] )



    if  not group_by_columns is None and len(group_by_columns)>0:

        group_names=[]

        for group in group_by_columns:

            groupss=train_supplamanteary2[["Date", group,target1,target2]]

            grp=groupss.groupby(["Date", group], as_index=False).sum()

            grp.columns=["Date", group,group +"_" + target1,group +"_" + target2 ]

            train_supplamanteary2=pd.merge(train_supplamanteary2,grp, how="left", left_on=["Date", group,group], right_on=["Date", group,group] )

            #group_names+=[group +"_" + target1,group +"_" + target2]

            for gr in [group +"_" + target1,group +"_" + target2]:

                rate(train_supplamanteary2, key, gr, new_target_name="rate_" +gr ) 

                group_names+=["rate_" +gr]           
fix_target(train, key, target1, new_target_name=target1)

fix_target(train, key, target2, new_target_name=target2)



rate(train, key, target1, new_target_name="rate_" +target1 )

rate(train, key, target2, new_target_name="rate_" +target2 )



difference(train, key, target1, new_target_name="diff_" +target1 )

difference(train, key, target2, new_target_name="diff_" +target2 )





unique_keys=train[key].unique()

print(len(unique_keys))


def get_lags(rate_array, current_index, size=20):

    lag_confirmed_rate=[-1 for k in range(size)]

    for j in range (0, size):

        if current_index-j>=0:

            lag_confirmed_rate[j]=rate_array[current_index-j]

        else :

            break

    return lag_confirmed_rate



def days_ago_thresold_hit(full_array, indx, thresold):

        days_ago_confirmed_count_10=-1

        if full_array[indx]>thresold: # if currently the count of confirmed is more than 10

            for j in range (indx,-1,-1):

                entered=False

                if full_array[j]<=thresold:

                    days_ago_confirmed_count_10=abs(j-indx)

                    entered=True

                    break

                if entered==False:

                    days_ago_confirmed_count_10=100 #this value would we don;t know it cross 0      

        return days_ago_confirmed_count_10 

    

    

def ewma_vectorized(data, alpha):

    sums=sum([ (alpha**(k+1))*data[k] for  k in range(len(data)) ])

    counts=sum([ (alpha**(k+1)) for  k in range(len(data)) ])

    return sums/counts



def generate_ma_std_window(rate_array, current_index, size=20, window=3):

    ma_rate_confirmed=[-1 for k in range(size)]

    std_rate_confirmed=[-1 for k in range(size)] 

    

    for j in range (0, size):

        if current_index-j>=0:

            ma_rate_confirmed[j]=np.mean(rate_array[max(0,current_index-j-window+1 ):current_index-j+1])

            std_rate_confirmed[j]=np.std(rate_array[max(0,current_index-j-window+1 ):current_index-j+1])           

        else :

            break

    return ma_rate_confirmed, std_rate_confirmed



def generate_ewma_window(rate_array, current_index, size=20, window=3, alpha=0.05):

    ewma_rate_confirmed=[-1 for k in range(size)]



    

    for j in range (0, size):

        if current_index-j>=0:

            ewma_rate_confirmed[j]=ewma_vectorized(rate_array[max(0,current_index-j-window+1 ):current_index-j+1, ], alpha)           

        else :

            break

    

    #print(ewma_rate_confirmed)

    return ewma_rate_confirmed





def get_target(rate_col, indx, horizon=33, average=3, use_hard_rule=False):

    target_values=[-1 for k in range(horizon)]

    cou=0

    for j in range(indx+1, indx+1+horizon):

        if j<len(rate_col):

            if average==1:

                target_values[cou]=rate_col[j]

            else :

                if use_hard_rule and j +average <=len(rate_col) :

                     target_values[cou]=np.mean(rate_col[j:j +average])

                else :

                    target_values[cou]=np.mean(rate_col[j:min(len(rate_col),j +average)])

                   

            cou+=1

        else :

            break

    return target_values



def get_target_count(rate_col, indx, horizon=33, average=3, use_hard_rule=False):

    target_values=[-1 for k in range(horizon)]

    cou=0

    for j in range(indx+1, indx+1+horizon):

        if j<len(rate_col):

            if average==1:

                target_values[cou]=rate_col[j]

            else :

                if use_hard_rule and j +average <=len(rate_col) :

                     target_values[cou]=np.mean(rate_col[j:j +average])

                else :

                    target_values[cou]=np.mean(rate_col[j:min(len(rate_col),j +average)])

                   

            cou+=1

        else :

            break

    return target_values





def dereive_features(frame, confirmed, fatalities, rate_confirmed, rate_fatalities, count_confirmed, count_fatalities ,

                     horizon ,size=20, windows=[3,7], days_back_confimed=[1,10,100], days_back_fatalities=[1,2,10], 

                    extra_data=None, groups_data=None, windows_group=[3,7], size_group=20,

                    days_back_confimed_group=[1,10,100]):

    targets=[]

    if not extra_data is None:

        assert len(extra_stable_columns)==extra_data.shape[1]

        

    if not groups_data is None:

        assert len(group_names)==groups_data.shape[1]        

    names=[]    

    #names=["lag_confirmed_rate" + str(k+1) for k in range (size)]

    for day in days_back_confimed:

        names+=["days_ago_confirmed_count_" + str(day) ]

    for window in windows:        

        names+=["ma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]

        #names+=["std" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)] 

        #names+=["ewma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]         

        names+=["ma" + str(window) + "_count_confirmed" + str(k+1) for k in range (size)]        

        

    #names+=["lag_fatalities_rate" + str(k+1) for k in range (size)]

    for day in days_back_fatalities:

        names+=["days_ago_fatalitiescount_" + str(day) ]    

    for window in windows:        

        names+=["ma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]

        #names+=["std" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]  

        #names+=["ewma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)] 

        names+=["ma" + str(window) + "_count_fatalities" + str(k+1) for k in range (size)]

        

    names+=["confirmed_level"]

    names+=["fatalities_level"]   

    if not extra_data is None: 

        names+=[k for k in extra_stable_columns]

    if not groups_data is None:  

         for gg in range (groups_data.shape[1]):

             #names+=["lag_rate_group_"+ str(gg+1) + "_" + str(k+1) for k in range (size_group)]    

             for day in days_back_confimed_group:

                names+=["days_ago_grooupcount_" + str(gg+1) + "_" + str(day) ]             

             for window in windows_group:        

                names+=["ma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]

                #names+=["std_group_" + str(gg+1)+ "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]  

                #names+=["ewma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size)]  



            

    names+=["confirmed_plus" + str(k+1) for k in range (horizon)]    

    names+=["fatalities_plus" + str(k+1) for k in range (horizon)]  

    names+=["confirmed_count_plus" + str(k+1) for k in range (horizon)]    

    names+=["fatalities_count_plus" + str(k+1) for k in range (horizon)]      

    #names+=["current_confirmed"]

    #names+=["current_fatalities"]    

    

    features=[]

    for i in range (len(confirmed)):

        row_features=[]

        #####################lag_confirmed_rate       

        lag_confirmed_rate=get_lags(rate_confirmed, i, size=size)

        #row_features+=lag_confirmed_rate

        #####################days_ago_confirmed_count_10

        for day in days_back_confimed:

            days_ago_confirmed_count_10=days_ago_thresold_hit(confirmed, i, day)               

            row_features+=[days_ago_confirmed_count_10] 

        #####################ma_rate_confirmed       

        #####################std_rate_confirmed 

        for window in windows:

            ma3_rate_confirmed,std3_rate_confirmed= generate_ma_std_window(rate_confirmed, i, size=size, window=window)

            row_features+= ma3_rate_confirmed   

            #row_features+= std3_rate_confirmed          

            #ewma3_rate_confirmed=generate_ewma_window(rate_confirmed, i, size=size, window=window, alpha=0.05)

            #row_features+= ewma3_rate_confirmed 

            ma3_count_confirmed,std3_count_confirmed= generate_ma_std_window(count_confirmed, i, size=size, window=window)  

            row_features+= ma3_count_confirmed               

        #####################lag_fatalities_rate   

        lag_fatalities_rate=get_lags(rate_fatalities, i, size=size)

        #row_features+=lag_fatalities_rate

        #####################days_ago_confirmed_count_10

        for day in days_back_fatalities:

            days_ago_fatalitiescount_2=days_ago_thresold_hit(fatalities, i, day)               

            row_features+=[days_ago_fatalitiescount_2]     

        #####################ma_rate_fatalities       

        #####################std_rate_fatalities 

        for window in windows:        

            ma3_rate_fatalities,std3_rate_fatalities= generate_ma_std_window(rate_fatalities, i, size=size, window=window)

            row_features+= ma3_rate_fatalities             

            #row_features+= std3_rate_fatalities  

            #ewma3_rate_fatalities=generate_ewma_window(rate_fatalities, i, size=size, window=window, alpha=0.05)

            #row_features+= ewma3_rate_fatalities

            ma3_count_fatalities,std3_count_fatalities= generate_ma_std_window(count_fatalities, i, size=size, window=window)

            row_features+= ma3_count_fatalities               

            

        ##################confirmed_level

        confirmed_level=0

        

        """

        if confirmed[i]>0 and confirmed[i]<1000:

            confirmed_level= confirmed[i]

        else :

            confirmed_level=2000

        """   

        confirmed_level= confirmed[i]

        row_features+=[confirmed_level]

        ##################fatalities_is_level

        fatalities_is_level=0

        """

        if fatalities[i]>0 and fatalities[i]<100:

            fatalities_is_level= fatalities[i]

        else :

            fatalities_is_level=200            

        """

        fatalities_is_level= fatalities[i]

        

        row_features+=[fatalities_is_level] 

        

        if not extra_data is None:    

            row_features+=extra_data[i].tolist()

            

        if not groups_data is None:  

          for gg in range (groups_data.shape[1]): 

             ## lags per group

             this_group=groups_data[:,gg].tolist()

             lag_group_rate=get_lags(this_group, i, size=size_group)

             #row_features+=lag_group_rate           

             #####################days_ago_confirmed_count_10

             for day in days_back_confimed_group:

                days_ago_groupcount_2=days_ago_thresold_hit(this_group, i, day)               

                row_features+=[days_ago_groupcount_2]     

             #####################ma_rate_fatalities       

             #####################std_rate_fatalities 

             for window in windows_group:        

                ma3_rate_group,std3_rate_group= generate_ma_std_window(this_group, i, size=size_group, window=window)

                row_features+= ma3_rate_group   

                #row_features+= std3_rate_group             

            

            

        #######################confirmed_plus target

        confirmed_plus=get_target(rate_confirmed, i, horizon=horizon)

        row_features+= confirmed_plus          

        #######################fatalities_plus target

        fatalities_plus=get_target(rate_fatalities, i, horizon=horizon)

        row_features+= fatalities_plus 

            

        #######################confirmed_plus target count

        confirmed_plus=get_target(count_confirmed, i, horizon=horizon)

        row_features+= confirmed_plus          

        #######################fatalities_plus target

        fatalities_plus=get_target(count_fatalities, i, horizon=horizon)

        row_features+= fatalities_plus         

        

        

        ##################current_confirmed

        #row_features+=[confirmed[i]]

        ##################current_fatalities

        #row_features+=[fatalities[i]]        

        

          



        

        features.append(row_features)

        

    new_frame=pd.DataFrame(data=features, columns=names).reset_index(drop=True)

    frame=frame.reset_index(drop=True)

    frame=pd.concat([frame, new_frame], axis=1)

    #print(frame.shape)

    return frame

    

    

def feature_engineering_for_single_key(frame, group, key, horizon=33, size=20, windows=[3,7], 

                                       days_back_confimed=[1,10,100], days_back_fatalities=[1,2,10],

                                      extra_stable_=None, group_nams=None,windows_group=[3,7], 

                                       size_group=20, days_back_confimed_group=[1,10,100]):

    

    mini_frame=get_data_by_key(frame, group, key, fields=None)

    

    mini_frame_with_features=dereive_features(mini_frame, mini_frame["ConfirmedCases"].values,

                                              mini_frame["Fatalities"].values, mini_frame["rate_ConfirmedCases"].values, 

                                               mini_frame["rate_Fatalities"].values, mini_frame["diff_ConfirmedCases"].values, 

                                               mini_frame["diff_Fatalities"].values,

                                              

                                              

                                              

                                              horizon ,size=size, windows=windows,

                                              days_back_confimed=days_back_confimed, days_back_fatalities=days_back_fatalities,

                                              extra_data=mini_frame[extra_stable_].values if not extra_stable_ is None else None,

                                              groups_data=mini_frame[group_nams].values if not group_nams is None else None,

                                              windows_group=windows_group, size_group=size_group, 

                                              days_back_confimed_group=days_back_confimed_group)

    #print (mini_frame_with_features.shape[0])

    return mini_frame_with_features
from tqdm import tqdm

train_frame=[]

size=10

windows=[3]

days_back_confimed=[1,5,10,20,50,100,250,500,1000]

days_back_fatalities=[1,2,5,10,20,50]



size_group=10

windows_group=[3]

days_back_confimed_group=[1,10,100]





#print (len(train['key'].unique()))

for unique_k in tqdm(unique_keys):

    mini_frame=feature_engineering_for_single_key(train, key, unique_k, horizon=horizon, size=size, 

                                                  windows=windows, days_back_confimed=days_back_confimed,

                                                  days_back_fatalities=days_back_fatalities,

                                                  extra_stable_=extra_stable_columns if extra_stable_columns is not None and len(extra_stable_columns)>0 else None,

                                     group_nams=group_names,windows_group=windows_group, 

                                     size_group=size_group, days_back_confimed_group=days_back_confimed_group

                                                 ).reset_index(drop=True) 

    #print (mini_frame.shape[0])

    train_frame.append(mini_frame)

    

train_frame = pd.concat(train_frame, axis=0).reset_index(drop=True)

#train_frame.to_csv(directory +"all" + ".csv", index=False)

new_unique_keys=train_frame['key'].unique()

for kee in new_unique_keys:

    if kee not in unique_keys:

        print (kee , " is not there ")



print ("==================================================") #Add USA county data for the count model built later on           

train_frame_supplamanteary2=None

if not train_supplamanteary2 is None:

    train_frame_supplamanteary2=[]

    for unique_k in tqdm(unique_keys_sup2):

        mini_frame_sup=feature_engineering_for_single_key(train_supplamanteary2, key, unique_k, horizon=horizon, size=size, 

                                                      windows=windows, days_back_confimed=days_back_confimed,

                                                      days_back_fatalities=days_back_fatalities,

                                                      extra_stable_=extra_stable_columns if extra_stable_columns is not None and len(extra_stable_columns)>0 else None,

                                     group_nams=group_names,windows_group=windows_group, 

                                     size_group=size_group, days_back_confimed_group=days_back_confimed_group ).reset_index(drop=True) 

        

        train_frame_supplamanteary2.append(mini_frame_sup) 

    train_frame_supplamanteary2 = pd.concat(train_frame_supplamanteary2, axis=0).reset_index(drop=True)

    new_unique_keys_sup2=train_frame_supplamanteary2['key'].unique()

####the training model



import lightgbm as lgb

from sklearn.linear_model import Ridge

from sklearn.externals import joblib



def bagged_set_train(X_ts,y_cs,wts, seed, estimators,xtest, xt=None,yt=None, output_name=None):

   #print (type(yt))

   # create array object to hold predictions 

  

   baggedpred=np.array([ 0.0 for d in range(0, xtest.shape[0])]) 

   #print (y_cs[:10])

   #print (yt[:10])  



   #loop for as many times as we want bags

   for n in range (0, estimators):

       

       params = {'objective': 'rmse',

                'metric': 'rmse',

                'boosting': 'gbdt',

                'learning_rate': 0.005, #change here    

                'drop_rate':0.01,

                #'alpha': 0.99, 

                'skip_drop':0.6,

                'uniform_drop':True,               

                'verbose': -1,    

                'num_leaves': 40, # ~18    

                'bagging_fraction': 0.9,    

                'bagging_freq': 1,    

                'bagging_seed': seed + n,    

                'feature_fraction': 0.8,    

                'feature_fraction_seed': seed + n,    

                'min_data_in_leaf': 10, #30, #56, # 10-50    

                'max_bin': 100, # maybe useful with overfit problem    

                'max_depth':20,                   

                #'reg_lambda': 10,    

                'reg_alpha':1,    

                'lambda_l2': 10,

                #'categorical_feature':'2', # because training data is extremely unbalanced                     

                'num_threads':6

                }

       d_train = lgb.Dataset(X_ts,y_cs, weight=wts, free_raw_data=False)#np.log1p(

       if not type(yt) is type(None):           

           d_cv = lgb.Dataset(xt,yt, free_raw_data=False, reference=d_train)#, reference=d_train

           model = lgb.train(params,d_train,num_boost_round=500,

                             valid_sets=d_cv,



                             verbose_eval=50 ) #1000                        

           

       else :

           #d_cv = lgb.Dataset(xt, free_raw_data=False, categorical_feature="2")  

           model = lgb.train(params,d_train,num_boost_round=500) #1000                              

           #importances=model.feature_importance('gain')

           #print(importances)

       preds=model.predict(xtest)               

       # update bag's array

       baggedpred+=preds

       #np.savetxt("preds_lgb" + str(n)+ ".csv",baggedpred)   

       #if n%5==0:

           #print("completed: " + str(n)  )                 



   if not output_name is None:

        joblib.dump((model), output_name)

   """

   model=Ridge(normalize=True, alpha=0.1, random_state=1)

   model.fit(X_ts,y_cs)

   preds=model.predict(xtest)

   baggedpred+=preds

   """

   # divide with number of bags to create an average estimate  

   baggedpred/= estimators

     

   return baggedpred

#the inference part

import lightgbm as lgb

from sklearn.linear_model import Ridge

from sklearn.externals import joblib



def predict(xtest,input_name=None):

   #print (type(yt))

   # create array object to hold predictions 

  

   baggedpred=np.array([ 0.0 for d in range(0, xtest.shape[0])]) 

   model=  joblib.load( input_name) 

   preds=model.predict(xtest)               

   baggedpred+=preds



   return baggedpred
#features to use in the rate model



names=[]

#names=["lag_confirmed_rate" + str(k+1) for k in range (size)]

for day in days_back_confimed:

    names+=["days_ago_confirmed_count_" + str(day) ]

for window in windows:        

    names+=["ma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]

    #names+=["std" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)] 

    #names+=["ewma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]         

    #names+=["ma" + str(window) + "_count_confirmed" + str(k+1) for k in range (size)]  



#names+=["lag_fatalities_rate" + str(k+1) for k in range (size)]

for day in days_back_fatalities:

    names+=["days_ago_fatalitiescount_" + str(day) ]    

for window in windows:        

    names+=["ma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]

    #names+=["std" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]  

    #names+=["ewma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]    

    #names+=["ma" + str(window) + "_count_fatalities" + str(k+1) for k in range (size)]    

names+=["confirmed_level"]

names+=["fatalities_level"]   

if not extra_stable_columns is None and len(extra_stable_columns)>0: 

    names+=[k for k in extra_stable_columns]  

    

if not group_names is None:  

     for gg in range (len(group_names)):

         #names+=["lag_rate_group_"+ str(gg+1) + "_" + str(k+1) for k in range (size_group)]    

         for day in days_back_confimed_group:

            names+=["days_ago_grooupcount_" + str(gg+1) + "_" + str(day) ]             

         for window in windows_group:        

            names+=["ma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]

            #names+=["std_group_" + str(gg+1)+ "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]  

            #names+=["ewma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size)]      

      



###############################THIS IS THE TRAINING PART OF MY CODE - ALL FILES ARE SAVED IN the model directory - you need to run it offline ####################

#######################You need to uncomment this section to generate the lightgbm models one for each day+x rate for both confirmed and fatalities ##########################

#################Full model ######################



tr_frame=train_frame



target_confirmed=["confirmed_plus" + str(k+1) for k in range (horizon)]    

target_fatalities=["fatalities_plus" + str(k+1) for k in range (horizon)] 

seed=1412



target_confirmed_train=tr_frame[target_confirmed].values

print ("  original shape of train is {}  ".format( target_confirmed_train.shape) )



target_fatalities_train=tr_frame[target_fatalities].values

features_train=tr_frame[names].values   



standard_confirmed_train=tr_frame["ConfirmedCases"].values

standard_fatalities_train=tr_frame["Fatalities"].values

current_confirmed_train=tr_frame["ConfirmedCases"].values



     



features_cv=[]

name_cv=[]

standard_confirmed_cv=[]

standard_fatalities_cv=[]

names_=tr_frame["key"].values

training_horizon=int(features_train.shape[0]/len(unique_keys)) 

print("training horizon = ",training_horizon)

for dd in range(training_horizon-1,features_train.shape[0],training_horizon):

    features_cv.append(features_train[dd])

    name_cv.append(names_[dd])

    standard_confirmed_cv.append(standard_confirmed_train[dd])

    standard_fatalities_cv.append(standard_fatalities_train[dd])

    print (name_cv[-1], standard_confirmed_cv[-1], standard_fatalities_cv[-1])

    

 

    

current_confirmed_train=[k for k in range(len(current_confirmed_train)) if current_confirmed_train[k]>0]

target_confirmed_train=target_confirmed_train[current_confirmed_train]

target_fatalities_train=target_fatalities_train[current_confirmed_train]        

features_train=features_train[current_confirmed_train]         

standard_confirmed_train=standard_confirmed_train[current_confirmed_train]

standard_fatalities_train=standard_fatalities_train[current_confirmed_train]  

    

features_cv=np.array(features_cv)

preds_confirmed_cv=np.zeros((features_cv.shape[0],horizon))

preds_confirmed_standard_cv=np.zeros((features_cv.shape[0],horizon))



preds_fatalities_cv=np.zeros((features_cv.shape[0],horizon))

preds_fatalities_standard_cv=np.zeros((features_cv.shape[0],horizon))



overal_rmsle_metric_confirmed=0.0



for j in range (preds_confirmed_cv.shape[1]):

    this_target=target_confirmed_train[:,j]

    index_positive=[k for k in range(len(this_target)) if this_target[k]!=-1]

    this_features=features_train[index_positive]

    this_target=this_target[index_positive]

    this_weight=np.log(standard_confirmed_train[index_positive]+3.)

    #this_weight=[1. for k in range(len(this_weight))]

    this_features_cv=features_cv                          



    preds=bagged_set_train(this_features,this_target,this_weight, seed, 1,features_cv, xt=None,yt=None, output_name=model_directory +"confirmed"+ str(j))

    preds_confirmed_cv[:,j]=preds

    print (" modelling confirmed, case %d, original train %d, and after %d, original cv %d and after %d "%(

    j,target_confirmed_train.shape[0],this_target.shape[0],this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[]

for ii in range (preds_confirmed_cv.shape[0]):

    current_prediction=standard_confirmed_cv[ii]

    if current_prediction==0 :

        current_prediction=0.1    

    for j in range (preds_confirmed_cv.shape[1]):

                current_prediction*=max(1,preds_confirmed_cv[ii][j])

                preds_confirmed_standard_cv[ii][j]=current_prediction







for j in range (preds_confirmed_cv.shape[1]):

    this_target=target_fatalities_train[:,j]

    index_positive=[k for k in range(len(this_target)) if this_target[k]!=-1]

    this_features=features_train[index_positive]

    this_target=this_target[index_positive]

    this_weight=np.log(standard_confirmed_train[index_positive]+2.)

    #this_weight=[1. for k in range(len(this_weight))]

    

    this_features_cv=features_cv

                             

    preds=bagged_set_train(this_features,this_target,this_weight, seed, 1,features_cv, xt=None,yt=None, output_name=model_directory +"fatal"+ str(j))

    preds_fatalities_cv[:,j]=preds

    print (" modelling fatalities, case %d, original train %d, and after %d, original cv %d and after %d "%(

    j,target_confirmed_train.shape[0],this_target.shape[0],this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[]

for ii in range (preds_fatalities_cv.shape[0]):

    current_prediction=standard_fatalities_cv[ii]

    if current_prediction==0 and standard_confirmed_cv[ii]>400:

        current_prediction=0.1

    for j in range (preds_fatalities_cv.shape[1]):

                if current_prediction==0 and  preds_confirmed_standard_cv[ii][j]>400:

                    current_prediction=1.

                current_prediction*=max(1,preds_fatalities_cv[ii][j])

                preds_fatalities_standard_cv[ii][j]=current_prediction

                



########################################## THIS IS THE MOST IMPORTANT PART OF MY CODE - THESE RULES CONTROL TH DECAY OF THE RATHER BAD LIGHTGBM MODELS#########################

########################## The inference is also fone here ############################



def decay_4_first_10_then_1_f(array):

    arr=[1.0 for k in range(len(array))]

    for j in range(len(array)):

        if j<10:

            arr[j]=1. + (max(1,array[j])-1.)/4.

        else :

            arr[j]=1.

    return arr

	

def decay_16_first_10_then_1_f(array):

    arr=[1.0 for k in range(len(array))]

    for j in range(len(array)):

        if j<10:

            arr[j]=1. + (max(1,array[j])-1.)/16.

        else :

            arr[j]=1.

    return arr	

            

def decay_2_f(array):

    arr=[1.0 for k in range(len(array))]    

    for j in range(len(array)):

            arr[j]=1. + (max(1,array[j])-1.)/2.

    return arr 



def decay_4_f(array):

    arr=[1.0 for k in range(len(array))]    

    for j in range(len(array)):

            arr[j]=1. + (max(1,array[j])-1.)/4.

    return arr 	

	

def acceleratorx2_f(array):

    arr=[1.0 for k in range(len(array))]    

    for j in range(len(array)):

            arr[j]=1. + (max(1,array[j])-1.)*2.

    return arr 







def decay_1_5_f(array):

    arr=[1.0 for k in range(len(array))]    

    for j in range(len(array)):

            arr[j]=1. + (max(1,array[j])-1.)/1.5

    return arr            



def decay_1_2_f(array):

    arr=[1.0 for k in range(len(array))]    

    for j in range(len(array)):

            arr[j]=1. + (max(1,array[j])-1.)/1.2

    return arr            

  



def decay_1_1_f(array):

    arr=[1.0 for k in range(len(array))]    

    for j in range(len(array)):

            arr[j]=1. + (max(1,array[j])-1.)/1.1

    return arr            

      

    

    

    

         

def stay_same_f(array):

    arr=[1.0 for k in range(len(array))]      

    for j in range(len(array)):

        arr[j]=1.

    return arr   



def decay_2_last_12_linear_inter_f(array):

    arr=[1.0 for k in range(len(array))]

    for j in range(len(array)):

        arr[j]=1. + (max(1,array[j])-1.)/2.

    arr12= (max(1,arr[-12])-1.)/12. 



    for j in range(0, 12):

        arr[len(arr)-12 +j]= max(1, 1 + ( (arr12*12) - (j+1)*arr12 ))

    return arr



def decay_1_2_last_12_linear_inter_f(array):

    arr=[1.0 for k in range(len(array))]

    for j in range(len(array)):

        arr[j]=1. + (max(1,array[j])-1.)/1.2

    arr12= (max(1,arr[-12])-1.)/12. 



    for j in range(0, 12):

        arr[len(arr)-12 +j]= max(1, 1 + ( (arr12*12) - (j+1)*arr12 ))

    return arr



def decay_1_5_last_12_linear_inter_f(array):

    arr=[1.0 for k in range(len(array))]

    for j in range(len(array)):

        arr[j]=1. + (max(1,array[j])-1.)/1.5

    arr12= (max(1,arr[-12])-1.)/12. 



    for j in range(0, 12):

        arr[len(arr)-12 +j]= max(1, 1 + ( (arr12*12) - (j+1)*arr12 ))

    return arr







def decay_4_last_12_linear_inter_f(array):

    arr=[1.0 for k in range(len(array))]

    for j in range(len(array)):

        arr[j]=1. + (max(1,array[j])-1.)/4.

    arr12= (max(1,arr[-12])-1.)/12. 



    for j in range(0, 12):

        arr[len(arr)-12 +j]= max(1, 1 + ( (arr12*12) - (j+1)*arr12 ))

    return arr



def decay_8_last_12_linear_inter_f(array):

    arr=[1.0 for k in range(len(array))]

    for j in range(len(array)):

        arr[j]=1. + (max(1,array[j])-1.)/8.

    arr12= (max(1,arr[-12])-1.)/12. 



    for j in range(0, 12):

        arr[len(arr)-12 +j]= max(1, 1 + ( (arr12*12) - (j+1)*arr12 ))

    return arr







def linear_last_12_f(array):

    arr=[1.0 for k in range(len(array))]

    for j in range(len(array)):

        arr[j]=max(1,array[j])

    arr12= (max(1,arr[-12])-1.)/12. 

    

    for j in range(0, 12):

        arr[len(arr)-12 +j]= max(1, 1 + ( (arr12*12) - (j+1)*arr12 ))

    return arr





    

decay_4_first_10_then_1 =[ "Jilin_China","Tianjin_China"]#, "Hong Kong_China"

decay_4_first_10_then_1_fatality=[]



decay_16_first_10_then_1 =["Beijing_China","Fujian_China","Jiangsu_China"]

decay_16_first_10_then_1_fatality=[]



decay_4=["nan_Central African Republic","Shanghai_China","nan_Korea, South","nan_Zambia"]

decay_4_fatality=["Heilongjiang_China","Inner Mongolia_China"]



decay_2 =["nan_Burundi","Faroe Islands_Denmark","nan_Eritrea","nan_Italy","nan_Brunei""Heilongjiang_China","Inner Mongolia_China",

"Shanxi_China","nan_Congo (Kinshasa)","nan_Cyprus","Reunion_France","Sint Maarten_Netherlands","nan_Angola"]

decay_2_fatality=["nan_Iran"]



stay_same=["nan_Diamond Princess"]

stay_same_fatality=["Beijing_China","Fujian_China","Qinghai_China","Jiangsu_China","Jilin_China","Macau_China","Ningxia_China","Tianjin_China"

,"Faroe Islands_Denmark","Shanxi_China"]#



normal=[]

normal_fatality=["nan_Andorra","New South Wales_Australia","nan_Korea, South","New York_US","nan_Austria","Alberta_Canada","Nova Scotia_Canada",

"Saskatchewan_Canada","nan_Costa Rica","nan_Croatia","nan_Finland","Mayotte_France","nan_Hungary","nan_Iceland","nan_Latvia","nan_Czechia"

,"nan_Luxembourg","nan_Malta","nan_Montenegro","nan_New Zealand","nan_Norway","nan_South Africa","Hawaii_US","Ohio_US","Pennsylvania_US","nan_Uruguay","nan_Venezuela",

"nan_West Bank and Gaza"]



decay_4_last_12_linear_inter =[ "nan_Dominica","New Caledonia_France",

"Saint Barthelemy_France","nan_Gambia","nan_Grenada","nan_Holy See",

"nan_Mauritania","nan_Nicaragua","nan_Saint Lucia","nan_Saint Vincent and the Grenadines",

"nan_Seychelles","nan_Suriname","nan_Mongolia","nan_Cabo Verde",

"Montserrat_United Kingdom","Turks and Caicos Islands_United Kingdom", "Hong Kong_China","Northern Territory_Australia","Northwest Territories_Canada","Yukon_Canada",

"Guangdong_China","nan_MS Zaandam","nan_Maldives","nan_Saint Kitts and Nevis","nan_Sao Tome and Principe","nan_South Sudan","nan_Western Sahara"]

decay_4_last_12_linear_inter_fatality=["nan_Somalia"]



decay_2_last_12_linear_inter =[ "nan_Chad","French Polynesia_France","nan_Laos","nan_Nepal","nan_Sudan","Bermuda_United Kingdom","Cayman Islands_United Kingdom"

,"nan_Eswatini","nan_Guyana","nan_Liberia","nan_Malawi","nan_Mozambique","nan_Timor-Leste",

                               "Falkland Islands (Malvinas)_United Kingdom","nan_Equatorial Guinea"]

decay_2_last_12_linear_inter_fatality=[]



decay_1_5 =["nan_Cambodia","nan_Croatia","nan_Denmark",

"nan_Bahamas","nan_Bahrain","nan_Barbados" ,"Manitoba_Canada","New Brunswick_Canada",

"Saskatchewan_Canada","nan_France","nan_Libya","nan_Malta",

	"Aruba_Netherlands","nan_Niger","nan_Spain","Virgin Islands_US",

	"Channel Islands_United Kingdom","Gibraltar_United Kingdom","nan_United Kingdom",'nan_Burma',"nan_Congo (Brazzaville)",

	"French Guiana_France","Martinique_France","Mayotte_France","nan_Georgia","nan_Iran","nan_New Zealand",

	"nan_West Bank and Gaza"]

decay_1_5_fatality=["nan_Cuba","nan_Morocco","Ohio_US","Pennsylvania_US","Puerto Rico_US" ,"nan_Uzbekistan",

                    "nan_Cyprus","nan_France","nan_Niger","South Dakota_US","nan_Gabon"]



acceleratorx2=[]

acceleratorx2_fatality=["Newfoundland and Labrador_Canada"]



decay_1_5_last_12_linear_inter =["nan_Antigua and Barbuda","nan_Botswana","nan_Haiti","nan_Madagascar","nan_Somalia",

"nan_Zimbabwe"]

decay_1_5_last_12_linear_inter_fatality =["nan_Burma","nan_Cabo Verde","nan_Djibouti"]



decay_1_2 =["nan_Albania","nan_Andorra","New South Wales_Australia","nan_Austria","nan_Azerbaijan","nan_Bangladesh","nan_Belize","Alberta_Canada","nan_Mauritius","nan_Monaco","nan_Montenegro",

"Newfoundland and Labrador_Canada","nan_Costa Rica","nan_Czechia","nan_Dominican Republic","nan_Estonia","Guadeloupe_France","nan_Iceland","nan_Latvia","nan_Liechtenstein","nan_Luxembourg","nan_Norway",

"nan_Poland","nan_Rwanda","nan_South Africa","nan_Trinidad and Tobago","Arizona_US","California_US","Colorado_US","Connecticut_US","Georgia_US","Hawaii_US","Maine_US","New York_US","Oklahoma_US",

"Oregon_US","Pennsylvania_US","Tennessee_US","Washington_US","nan_Uruguay","nan_Venezuela","nan_Vietnam",

            "nan_Gabon","nan_Syria","nan_Djibouti"]



decay_1_2_fatality=["nan_Afghanistan","nan_Belarus","nan_Mali","nan_Mexico","nan_Peru","nan_Serbia","Kentucky_US","Puerto Rico_US","Texas_US","nan_Ukraine"]



decay_1_1 =["nan_Algeria","nan_Argentina","Tasmania_Australia","Nova Scotia_Canada","nan_Finland","nan_Hungary","nan_Iraq",

"nan_Jordan","nan_Kyrgyzstan","nan_Malaysia","Alabama_US","Iowa_US","Michigan_US","Missouri_US","Nebraska_US","North Dakota_US ","Ohio_US","South Carolina_US"]

decay_1_1_fatality=["nan_Ecuador","nan_Kenya","nan_Korea, South","nan_Kosovo","Indiana_US","New Mexico_US","Oregon_US","Rhode Island_US","Tennessee_US","nan_United Arab Emirates"]

 

decay_1_2_last_12_linear_inter =[]

decay_1_2_last_12_linear_inter_fatality=["nan_Angola" ]



linear_last_12=["nan_Afghanistan","nan_Ireland","nan_Benin","nan_Tanzania"]

linear_last_12_fatality=["nan_Guatemala","nan_Ireland"]



decay_8_last_12_linear_inter =[ "nan_Bhutan","Prince Edward Island_Canada","Greenland_Denmark","nan_Fiji","Saint Pierre and Miquelon_France","St Martin_France","nan_Namibia",

"Bonaire, Sint Eustatius and Saba_Netherlands","Curacao_Netherlands","nan_Papua New Guinea","Anguilla_United Kingdom","British Virgin Islands_United Kingdom",]



decay_8_last_12_linear_inter_fatality=[]



tr_frame=train_frame



features_train=tr_frame[names].values   



standard_confirmed_train=tr_frame["ConfirmedCases"].values

standard_fatalities_train=tr_frame["Fatalities"].values

current_confirmed_train=tr_frame["ConfirmedCases"].values



     



features_cv=[]

name_cv=[]

standard_confirmed_cv=[]

standard_fatalities_cv=[]

names_=tr_frame["key"].values

training_horizon=int(features_train.shape[0]/len(unique_keys)) 

print("training horizon = ",training_horizon)

for dd in range(training_horizon-1,features_train.shape[0],training_horizon):

    features_cv.append(features_train[dd])

    name_cv.append(names_[dd])

    standard_confirmed_cv.append(standard_confirmed_train[dd])

    standard_fatalities_cv.append(standard_fatalities_train[dd])

    print (name_cv[-1], standard_confirmed_cv[-1], standard_fatalities_cv[-1])

    

 



features_cv=np.array(features_cv)

preds_confirmed_cv=np.zeros((features_cv.shape[0],horizon))

preds_confirmed_standard_cv=np.zeros((features_cv.shape[0],horizon))



preds_fatalities_cv=np.zeros((features_cv.shape[0],horizon))

preds_fatalities_standard_cv=np.zeros((features_cv.shape[0],horizon))



overal_rmsle_metric_confirmed=0.0



for j in range (preds_confirmed_cv.shape[1]):



    this_features_cv=features_cv                          



    preds=predict(features_cv, input_name=model_directory +"confirmed"+ str(j))

    preds_confirmed_cv[:,j]=preds

    print (" modelling confirmed, case %d, , original cv %d and after %d "%(j,this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[] 

for ii in range (preds_confirmed_cv.shape[0]):

    current_prediction=standard_confirmed_cv[ii]

    if current_prediction==0 :

        current_prediction=0.1   

    this_preds=preds_confirmed_cv[ii].tolist()

    name=name_cv[ii]

    reserve=this_preds[0]

    #overrides





    if name in normal:

        this_preds=this_preds	

	

    elif name in decay_4_first_10_then_1:

        this_preds=decay_4_first_10_then_1_f(this_preds)

		

    elif name in decay_16_first_10_then_1:

        this_preds=decay_16_first_10_then_1_f(this_preds)		

		

    elif name in decay_4_last_12_linear_inter:

        this_preds=decay_4_last_12_linear_inter_f(this_preds)		  

        

    elif name in decay_8_last_12_linear_inter:

        this_preds=decay_8_last_12_linear_inter_f(this_preds)		          



          

    elif name in decay_4:

        this_preds=decay_4_f(this_preds)



    

    elif name in decay_2:

        this_preds=decay_2_f(this_preds)

        

    elif name in decay_2_last_12_linear_inter:

        this_preds=decay_2_last_12_linear_inter_f(this_preds)

        

    elif name in decay_1_5:

        this_preds=decay_1_5_f(this_preds)  



    elif name in decay_1_5_last_12_linear_inter:

        this_preds=decay_1_5_last_12_linear_inter_f(this_preds)  

                     

    elif name in decay_1_2:

        this_preds=decay_1_2_f(this_preds) 



    elif name in decay_1_2_last_12_linear_inter:

        this_preds=decay_1_2_last_12_linear_inter_f(this_preds)         

        

    elif name in decay_1_1:

        this_preds=decay_1_1_f(this_preds)            

        

    elif name in linear_last_12:

        this_preds=linear_last_12_f(this_preds)

        

    elif name in acceleratorx2:

        this_preds=acceleratorx2_f(this_preds)         



        

    elif name in stay_same or  "China" in name:

        this_preds=stay_same_f(this_preds)      



        

        

    for j in range (preds_confirmed_cv.shape[1]):

                

                if j==0 and "nan_Congo (Brazzaville)" in name:

                     current_prediction=117.

                        

                if j==0 and "nan_Cabo Verde" in name:

                      current_prediction=56.                   

        

                current_prediction*=max(1,this_preds[j])

                preds_confirmed_standard_cv[ii][j]=current_prediction





for j in range (preds_confirmed_cv.shape[1]):



    this_features_cv=features_cv

                             

    preds=predict(features_cv, input_name=model_directory +"fatal"+ str(j))

    preds_fatalities_cv[:,j]=preds

    print (" modelling fatalities, case %d, original cv %d and after %d "%( j,this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[]

for ii in range (preds_fatalities_cv.shape[0]):

    current_prediction=standard_fatalities_cv[ii]

        

    this_preds=preds_fatalities_cv[ii].tolist()

    name=name_cv[ii]

    reserve=this_preds[0]

    #overrides

   

    ####fatality special



    if name in normal_fatality:

        this_preds=this_preds	 	

	

    elif name in decay_4_first_10_then_1_fatality:

        this_preds=decay_4_first_10_then_1_f(this_preds) 

		

    elif name in decay_16_first_10_then_1_fatality:

        this_preds=decay_16_first_10_then_1_f(this_preds)

        

    elif name in decay_4_last_12_linear_inter_fatality:

        this_preds=decay_4_last_12_linear_inter_f(this_preds)		      

        

    elif name in decay_8_last_12_linear_inter_fatality:

        this_preds=decay_8_last_12_linear_inter_f(this_preds)		            



    elif name in decay_4_fatality:

        this_preds=decay_4_f(this_preds)		

		

    elif name in decay_2_fatality:

        this_preds=decay_2_f(this_preds) 

                



    elif name in decay_2_last_12_linear_inter_fatality:

        this_preds=decay_2_last_12_linear_inter_f(this_preds)

        

    elif name in decay_1_5_fatality:

        this_preds=decay_1_5_f(this_preds)   	

        

    elif name in decay_1_5_last_12_linear_inter_fatality:

        this_preds=decay_1_5_last_12_linear_inter_f(this_preds)         

        

    elif name in decay_1_2_fatality:

        this_preds=decay_1_2_f(this_preds) 

        

    elif name in decay_1_2_last_12_linear_inter_fatality:

        this_preds=decay_1_2_last_12_linear_inter_f(this_preds)             

        

    elif name in decay_1_1_fatality:

        this_preds=decay_1_1_f(this_preds)            

 

    elif name in linear_last_12_fatality:

        this_preds=linear_last_12_f(this_preds) 

         

    elif name in acceleratorx2_fatality:

        this_preds=acceleratorx2_f(this_preds)

        

    elif name in stay_same_fatality:     

        this_preds=stay_same_f(this_preds) 

        

     

    ####general   

    elif name in normal:

        this_preds=this_preds	   

	

    elif name in decay_4_first_10_then_1:

        this_preds=decay_4_first_10_then_1_f(this_preds)

		

    elif name in decay_16_first_10_then_1:

        this_preds=decay_16_first_10_then_1_f(this_preds)		

        

    elif name in decay_4_last_12_linear_inter:

        this_preds=decay_4_last_12_linear_inter_f(this_preds)		     

        

    elif name in decay_8_last_12_linear_inter:

        this_preds=decay_8_last_12_linear_inter_f(this_preds)		           



    elif name in decay_4:

        this_preds=decay_4_f(this_preds)        

		

    elif name in decay_2:

        this_preds=decay_2_f(this_preds)

        

    elif name in decay_2_last_12_linear_inter:

        this_preds=decay_2_last_12_linear_inter_f(this_preds)

        

    elif name in decay_1_5:

        this_preds=decay_1_5_f(this_preds) 

        

    elif name in decay_1_5_last_12_linear_inter:

        this_preds=decay_1_5_last_12_linear_inter_f(this_preds)         

         

    elif name in decay_1_2:

        this_preds=decay_1_2_f(this_preds) 

        

    elif name in decay_1_2_last_12_linear_inter:

        this_preds=decay_1_2_last_12_linear_inter_f(this_preds)             

        

    elif name in decay_1_1:

        this_preds=decay_1_1_f(this_preds)            

        

    elif name in linear_last_12:

        this_preds=linear_last_12_f(this_preds) 

        

    elif name in acceleratorx2:

        this_preds=acceleratorx2_f(this_preds)                 

        

    elif name in stay_same or  "China" in name:

        this_preds=stay_same_f(this_preds)         

        

  

    for j in range (preds_fatalities_cv.shape[1]):

                if current_prediction==0 and  preds_confirmed_standard_cv[ii][j]>400 and (name not in stay_same and name not in stay_same_fatality and "China" not in name ):

                    current_prediction=1.

                if j==0 and "Aruba_Netherlands" in name:

                     current_prediction=1.

                if j==0 and "nan_Monaco" in name:

                     current_prediction=3.

                if j==0 and 'nan_Costa Rica' in name or "Isle of Man_United Kingdom" in name:

                    current_prediction=4. 

                if j==0 and "nan_Somalia" in name:

                    current_prediction=5

                if j==0 and "Martinique_France" in name:

                    current_prediction=8. 

                  

                current_prediction*=max(1,this_preds[j])

                preds_fatalities_standard_cv[ii][j]=current_prediction     

        

# CREATE MAPPINGS 



key_to_confirmed_rate={}

key_to_fatality_rate={}

key_to_confirmed={}

key_to_fatality={}

print(len(features_cv), len(name_cv),len(standard_confirmed_cv),len(standard_fatalities_cv)) 

print(preds_confirmed_cv.shape,preds_confirmed_standard_cv.shape,preds_fatalities_cv.shape,preds_fatalities_standard_cv.shape) 



for j in range (len(name_cv)):

    

    key_to_confirmed_rate[name_cv[j]]=preds_confirmed_cv[j,:].tolist()

    #print(key_to_confirmed_rate[name_cv[j]])

    key_to_fatality_rate[name_cv[j]]=preds_fatalities_cv[j,:].tolist()

    key_to_confirmed[name_cv[j]]  =preds_confirmed_standard_cv[j,:].tolist()  

    key_to_fatality[name_cv[j]]=preds_fatalities_standard_cv[j,:].tolist()  

    
train_new=train[["Date","ConfirmedCases","Fatalities","key","rate_ConfirmedCases","rate_Fatalities"]]



test_new=pd.merge(test,train_new, how="left", left_on=["key","Date"], right_on=["key","Date"] ).reset_index(drop=True)

test_new
def fillin_columns(frame,key_column, original_name, training_horizon, test_horizon, unique_values, key_to_values):

    keys=frame[key_column].values

    original_values=frame[original_name].values.tolist()

    print(len(keys), len(original_values), training_horizon ,test_horizon,len(key_to_values))

    

    for j in range(unique_values):

        current_index=(j * (training_horizon +test_horizon )) +training_horizon 

        current_key=keys[current_index]

        values=key_to_values[current_key]

        co=0

        for g in range(current_index, current_index + test_horizon):

            original_values[g]=values[co]

            co+=1

    

    frame[original_name]=original_values

 



all_days=int(test_new.shape[0]/len(unique_keys))



tr_horizon=all_days-horizon

print(all_days,tr_horizon, horizon )



fillin_columns(test_new,"key", 'ConfirmedCases', tr_horizon, horizon, len(unique_keys), key_to_confirmed)    

fillin_columns(test_new,"key", 'Fatalities', tr_horizon, horizon, len(unique_keys), key_to_fatality)   

fillin_columns(test_new,"key", 'rate_ConfirmedCases', tr_horizon, horizon, len(unique_keys), key_to_confirmed_rate)   

fillin_columns(test_new,"key", 'rate_Fatalities', tr_horizon, horizon, len(unique_keys), key_to_fatality_rate) 
"""

def fillin_columns(frame,key_column, original_name, training_horizon, test_horizon, unique_values, key_to_values):

    keys=frame[key_column].values

    original_values=frame[original_name].values.tolist()

    print(len(keys), len(original_values), training_horizon ,test_horizon,len(key_to_values))

    

    for j in range(unique_values):

        current_index=(j * (training_horizon +test_horizon )) +training_horizon 

        current_key=keys[current_index]

        values=key_to_values[current_key]

        co=0

        for g in range(current_index, current_index + test_horizon):

            original_values[g]=values[co]

            co+=1

    

    frame[original_name]=original_values

 



all_days=int(test_new.shape[0]/len(unique_keys))



tr_horizon=all_days-horizon

print(all_days,tr_horizon, horizon )



fillin_columns(test_new,"key", 'ConfirmedCases', tr_horizon, horizon, len(unique_keys), key_to_confirmed)    

fillin_columns(test_new,"key", 'Fatalities', tr_horizon, horizon, len(unique_keys), key_to_fatality)   

submission=test_new[["ForecastId","ConfirmedCases","Fatalities"]]



submission.to_csv("submission.csv", index=False)

"""

############################### model to use to model (log) counts#############################



import lightgbm as lgb

from sklearn.linear_model import Ridge

from sklearn.externals import joblib



def bagged_set_trainc(X_ts,y_cs,wts, seed, estimators,xtest, xt=None,yt=None, output_name=None):

   #print (type(yt))

   # create array object to hold predictions 

  

   baggedpred=np.array([ 0.0 for d in range(0, xtest.shape[0])]) 

   #print (y_cs[:10])

   #print (yt[:10])  



   #loop for as many times as we want bags

   for n in range (0, estimators):

       

       params = {'objective': 'rmse',

                'metric': 'rmse',

                'boosting': 'gbdt',

                'learning_rate': 0.005, #change here    

                'drop_rate':0.01,

                #'alpha': 0.99, 

                'skip_drop':0.6,

                'uniform_drop':True,               

                'verbose': -1,    

                'num_leaves': 30, # ~18    

                'bagging_fraction': 0.9,    

                'bagging_freq': 1,    

                'bagging_seed': seed + n,    

                'feature_fraction': 0.8,    

                'feature_fraction_seed': seed + n,    

                'min_data_in_leaf': 10, #30, #56, # 10-50    

                'max_bin': 100, # maybe useful with overfit problem    

                'max_depth':20,                   

                #'reg_lambda': 10,    

                'reg_alpha':1,    

                'lambda_l2': 10,

                #'categorical_feature':'2', # because training data is extremely unbalanced                     

                'num_threads':6

                }

       d_train = lgb.Dataset(X_ts,y_cs, weight=wts, free_raw_data=False)#np.log1p(

       if not type(yt) is type(None):           

           d_cv = lgb.Dataset(xt,yt, free_raw_data=False, reference=d_train)#, reference=d_train

           model = lgb.train(params,d_train,num_boost_round=500,

                             valid_sets=d_cv,



                             verbose_eval=50 ) #1000                        

           

       else :

           #d_cv = lgb.Dataset(xt, free_raw_data=False, categorical_feature="2")  

           model = lgb.train(params,d_train,num_boost_round=500) #1000                              

           #importances=model.feature_importance('gain')

           #print(importances)

       preds=model.predict(xtest)               

       # update bag's array

       baggedpred+=preds

       #np.savetxt("preds_lgb" + str(n)+ ".csv",baggedpred)   

       #if n%5==0:

           #print("completed: " + str(n)  )                 



   if not output_name is None:

        joblib.dump((model), output_name)

   """

   model=Ridge(normalize=True, alpha=0.1, random_state=1)

   model.fit(X_ts,y_cs)

   preds=model.predict(xtest)

   baggedpred+=preds

   """

   # divide with number of bags to create an average estimate  

   baggedpred/= estimators

     

   return baggedpred



###########################features for the count model ###########################

names=[]

#names=["lag_confirmed_rate" + str(k+1) for k in range (size)]

for day in days_back_confimed:

    names+=["days_ago_confirmed_count_" + str(day) ]

for window in windows:        

    names+=["ma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]

    #names+=["std" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)] 

    #names+=["ewma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]         

    names+=["ma" + str(window) + "_count_confirmed" + str(k+1) for k in range (size)]  



#names+=["lag_fatalities_rate" + str(k+1) for k in range (size)]

for day in days_back_fatalities:

    names+=["days_ago_fatalitiescount_" + str(day) ]    

for window in windows:        

    names+=["ma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]

    #names+=["std" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]  

    #names+=["ewma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]    

    names+=["ma" + str(window) + "_count_fatalities" + str(k+1) for k in range (size)]    

names+=["confirmed_level"]

names+=["fatalities_level"]   

if not extra_stable_columns is None and len(extra_stable_columns)>0: 

    names+=[k for k in extra_stable_columns]  

    

if not group_names is None:  

     for gg in range (len(group_names)):

         #names+=["lag_rate_group_"+ str(gg+1) + "_" + str(k+1) for k in range (size_group)]    

         for day in days_back_confimed_group:

            names+=["days_ago_grooupcount_" + str(gg+1) + "_" + str(day) ]             

         for window in windows_group:        

            names+=["ma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]

            #names+=["std_group_" + str(gg+1)+ "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]  

            #names+=["ewma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size)]      

      

###############################THIS IS THE TRAINING PART OF MY CODE - ALL FILES ARE SAVED IN the model directory - you need to run it offline ####################

#######################You need to uncomment this section to generate the lightgbm models one for each day+x log_count difference for both confirmed and fatalities ##########################

#################Full model for counts ######################



tr_frame=train_frame

if not train_frame_supplamanteary2 is None:

    tr_frame=train_frame_supplamanteary2

target_confirmed=["confirmed_count_plus" + str(k+1) for k in range (horizon)]    

target_fatalities=["fatalities_count_plus" + str(k+1) for k in range (horizon)] 

seed=1412



target_confirmed_train=tr_frame[target_confirmed].values

print ("  original shape of train is {}  ".format( target_confirmed_train.shape) )



target_fatalities_train=tr_frame[target_fatalities].values

features_train=tr_frame[names].values   



standard_confirmed_train=tr_frame["ConfirmedCases"].values

standard_fatalities_train=tr_frame["Fatalities"].values

current_confirmed_train=tr_frame["ConfirmedCases"].values



     



features_cv=[]

name_cv=[]

standard_confirmed_cv=[]

standard_fatalities_cv=[]

names_=tr_frame["key"].values

training_horizon=int(features_train.shape[0]/len(unique_keys)) 

print("training horizon = ",training_horizon)

for dd in range(training_horizon-1,features_train.shape[0],training_horizon):

    features_cv.append(features_train[dd])

    name_cv.append(names_[dd])

    standard_confirmed_cv.append(standard_confirmed_train[dd])

    standard_fatalities_cv.append(standard_fatalities_train[dd])

    print (name_cv[-1], standard_confirmed_cv[-1], standard_fatalities_cv[-1])

    

 

    

current_confirmed_train=[k for k in range(len(current_confirmed_train)) if current_confirmed_train[k]>0]

target_confirmed_train=target_confirmed_train[current_confirmed_train]

target_fatalities_train=target_fatalities_train[current_confirmed_train]        

features_train=features_train[current_confirmed_train]         

standard_confirmed_train=standard_confirmed_train[current_confirmed_train]

standard_fatalities_train=standard_fatalities_train[current_confirmed_train]  

    

features_cv=np.array(features_cv)

preds_confirmed_cv=np.zeros((features_cv.shape[0],horizon))

preds_confirmed_standard_cv=np.zeros((features_cv.shape[0],horizon))



preds_fatalities_cv=np.zeros((features_cv.shape[0],horizon))

preds_fatalities_standard_cv=np.zeros((features_cv.shape[0],horizon))



overal_rmsle_metric_confirmed=0.0



for j in range (preds_confirmed_cv.shape[1]):

    this_target=target_confirmed_train[:,j]

    index_positive=[k for k in range(len(this_target)) if this_target[k]!=-1]

    this_features=features_train[index_positive]

    this_target=this_target[index_positive]

    this_weight=np.log(standard_confirmed_train[index_positive]+3.)

    this_weight=[1. for k in range(len(this_weight))]

    this_features_cv=features_cv                          



    preds=bagged_set_trainc(this_features,this_target,this_weight, seed, 1,features_cv, xt=None,yt=None, output_name=model_directory +"confirmedc"+ str(j))

    preds_confirmed_cv[:,j]=preds

    print (" modelling count confirmed, case %d, original train %d, and after %d, original cv %d and after %d "%(

    j,target_confirmed_train.shape[0],this_target.shape[0],this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[]

for ii in range (preds_confirmed_cv.shape[0]):

    current_prediction=standard_confirmed_cv[ii]  

    for j in range (preds_confirmed_cv.shape[1]):

                current_prediction+=np.expm1(max(0,preds_confirmed_cv[ii][j]))

                preds_confirmed_standard_cv[ii][j]=current_prediction







for j in range (preds_confirmed_cv.shape[1]):

    this_target=target_fatalities_train[:,j]

    index_positive=[k for k in range(len(this_target)) if this_target[k]!=-1]

    this_features=features_train[index_positive]

    this_target=this_target[index_positive]

    this_weight=np.log(standard_confirmed_train[index_positive]+2.)

    this_weight=[1. for k in range(len(this_weight))]

    

    this_features_cv=features_cv

                             

    preds=bagged_set_trainc(this_features,this_target,this_weight, seed, 1,features_cv, xt=None,yt=None, output_name=model_directory +"fatalc"+ str(j))

    preds_fatalities_cv[:,j]=preds

    print (" modelling count fatalities, case %d, original train %d, and after %d, original cv %d and after %d "%(

    j,target_confirmed_train.shape[0],this_target.shape[0],this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[]

for ii in range (preds_fatalities_cv.shape[0]):

    current_prediction=standard_fatalities_cv[ii]

    for j in range (preds_fatalities_cv.shape[1]):



                current_prediction+=np.expm1(max(0,preds_fatalities_cv[ii][j]))

                preds_fatalities_standard_cv[ii][j]=current_prediction

############################################## Once again, thees rules are the important bit ################################





def decay_4_first_10_then_1_f(array):

    arr=[k for k in array]

    for j in range(len(array)):

        if j<10:

            arr[j]*=1./4.

        else :

            arr[j]=0

    return arr



	

def decay_16_first_10_then_1_f(array):

    arr=[k for k in array]

    for j in range(len(array)):

        if j<10:

            arr[j]*=1./16.

        else :

            arr[j]=0

    return arr

            

def decay_2_f(array):

    arr=[k for k in array]    

    for j in range(len(array)):

            arr[j]*=1./2.

    return arr 



def decay_4_f(array):

    arr=[k for k in array]    

    for j in range(len(array)):

            arr[j]*=1./4.

    return arr 

	

def acceleratorx2_f(array):

    arr=[k for k in array]    

    for j in range(len(array)):

            arr[j]*=2.

    return arr 





def decay_1_5_f(array):

    arr=[k for k in array]    

    for j in range(len(array)):

            arr[j]*=1./1.5

    return arr          



def decay_1_2_f(array):

    arr=[k for k in array]    

    for j in range(len(array)):

            arr[j]*=1./1.2

    return arr   



def decay_1_1_f(array):

    arr=[k for k in array]    

    for j in range(len(array)):

            arr[j]*=1./1.1

    return arr   

         

def stay_same_f(array):

    arr=[0.0 for k in array]      

    return arr   



def decay_2_last_12_linear_inter_f(array):

    arr=[k for k in array]

    for j in range(len(array)):

        arr[j]*=1/2.

    arr12= arr[-12]



    for j in range(0, 12):

        arr[len(arr)-12 +j]= arr12/(j+1.)

    return arr



def decay_1_5_last_12_linear_inter_f(array):

    arr=[k for k in array]

    for j in range(len(array)):

        arr[j]*=1/(1.5)

    arr12= arr[-12]



    for j in range(0, 12):

        arr[len(arr)-12 +j]= arr12/(j+1.)

    return arr







def decay_1_2_last_12_linear_inter_f(array):

    arr=[k for k in array]

    for j in range(len(array)):

        arr[j]*=1./(1.2)

    arr12= arr[-12]



    for j in range(0, 12):

        arr[len(arr)-12 +j]= arr12/(j+1.)

    return arr







def decay_4_last_12_linear_inter_f(array):

    arr=[k for k in array]

    for j in range(len(array)):

        arr[j]*=1/4.

    arr12= arr[-12]



    for j in range(0, 12):

        arr[len(arr)-12 +j]= arr12/(j+1.)

    return arr





def decay_8_last_12_linear_inter_f(array):

    arr=[k for k in array]

    for j in range(len(array)):

        arr[j]*=1/8.

    arr12= arr[-12]



    for j in range(0, 12):

        arr[len(arr)-12 +j]= arr12/(j+1.)

    return arr











def linear_last_12_f(array):

    arr=[k for k in array]

    for j in range(len(array)):

        arr[j]=array[j]

    arr12=  arr[-12] 

    

    for j in range(0, 12):

        arr[len(arr)-12 +j]= arr12/(j+1.)

    return arr

    

decay_4_first_10_then_1 =[ "Jilin_China","Tianjin_China"]#, "Hong Kong_China"

decay_4_first_10_then_1_fatality=[]



decay_16_first_10_then_1 =["Beijing_China","Fujian_China","Jiangsu_China"]

decay_16_first_10_then_1_fatality=[]



decay_4=["nan_Central African Republic","Shanghai_China","nan_Equatorial Guinea","nan_Korea, South","nan_Zambia"]

decay_4_fatality=["Heilongjiang_China","Inner Mongolia_China"]



decay_2 =["nan_Burundi","Faroe Islands_Denmark","nan_Eritrea","nan_Gabon","nan_Italy","nan_Brunei""Heilongjiang_China","Inner Mongolia_China",

"Shanxi_China","nan_Congo (Kinshasa)","nan_Cyprus","Reunion_France","Sint Maarten_Netherlands","nan_Angola"]

decay_2_fatality=["nan_Iran"]



stay_same=["nan_Diamond Princess"]

stay_same_fatality=["Beijing_China","Fujian_China","Qinghai_China","Jiangsu_China","Jilin_China","Macau_China","Ningxia_China","Tianjin_China"

,"Faroe Islands_Denmark","Shanxi_China"]#



normal=[]

normal_fatality=["nan_Andorra","New South Wales_Australia","nan_Korea, South","New York_US","nan_Austria","Alberta_Canada","Nova Scotia_Canada",

"Saskatchewan_Canada","nan_Costa Rica","nan_Croatia","nan_Finland","Martinique_France","Mayotte_France","nan_Hungary","nan_Iceland","nan_Latvia","nan_Czechia"

,"nan_Luxembourg","nan_Malta","nan_Montenegro","nan_New Zealand","nan_Norway","nan_South Africa","Hawaii_US","Ohio_US","Pennsylvania_US","nan_Uruguay","nan_Venezuela",

"nan_West Bank and Gaza"]



decay_4_last_12_linear_inter =[ "nan_Dominica","New Caledonia_France",

"Saint Barthelemy_France","nan_Gambia","nan_Grenada","nan_Holy See",

"nan_Mauritania","nan_Nicaragua","nan_Saint Lucia","nan_Saint Vincent and the Grenadines",

"nan_Seychelles","nan_Suriname","nan_Mongolia",

"Montserrat_United Kingdom","Turks and Caicos Islands_United Kingdom", "Hong Kong_China","Northern Territory_Australia","Northwest Territories_Canada","Yukon_Canada"

,"Guangdong_China","nan_MS Zaandam","nan_Maldives","nan_Saint Kitts and Nevis","nan_Sao Tome and Principe","nan_South Sudan","nan_Western Sahara"]

decay_4_last_12_linear_inter_fatality=["nan_Somalia"]



decay_2_last_12_linear_inter =[ "nan_Chad","French Polynesia_France","nan_Laos","nan_Nepal","nan_Sudan","nan_Tanzania","Bermuda_United Kingdom","Cayman Islands_United Kingdom"

,"nan_Cabo Verde","nan_Eswatini","nan_Guyana","nan_Liberia","nan_Malawi","nan_Mozambique","nan_Somalia","nan_Timor-Leste","Falkland Islands (Malvinas)_United Kingdom",

"nan_Zimbabwe"]

decay_2_last_12_linear_inter_fatality=[]





decay_1_5 =["nan_Cambodia","nan_Croatia","nan_Denmark",

"nan_Bahamas","nan_Bahrain","nan_Barbados" ,"Manitoba_Canada","New Brunswick_Canada",

"Saskatchewan_Canada","nan_France","nan_Libya","nan_Malta",

	"Aruba_Netherlands","nan_Niger","nan_Spain","Virgin Islands_US",

	"Channel Islands_United Kingdom","Gibraltar_United Kingdom","nan_United Kingdom",'nan_Burma',"nan_Congo (Brazzaville)",

	"French Guiana_France","Martinique_France","Mayotte_France","nan_Georgia","nan_Iran","nan_New Zealand","nan_Syria",

	"nan_West Bank and Gaza"]

decay_1_5_fatality=["nan_Cuba","nan_Morocco","Ohio_US","Pennsylvania_US","Puerto Rico_US" ,"nan_Uzbekistan","nan_Cyprus","nan_France","nan_Niger","South Dakota_US"]



acceleratorx2=[]

acceleratorx2_fatality=["Newfoundland and Labrador_Canada"]



decay_1_5_last_12_linear_inter =["nan_Antigua and Barbuda","nan_Botswana","nan_Haiti","nan_Madagascar"]

decay_1_5_last_12_linear_inter_fatality =[]



decay_1_2 =["nan_Albania","nan_Andorra","New South Wales_Australia","nan_Austria","nan_Azerbaijan","nan_Bangladesh","nan_Belize","Alberta_Canada","nan_Mauritius","nan_Monaco","nan_Montenegro",

"Newfoundland and Labrador_Canada","nan_Costa Rica","nan_Czechia","nan_Dominican Republic","nan_Estonia","Guadeloupe_France","nan_Iceland","nan_Latvia","nan_Liechtenstein","nan_Luxembourg","nan_Norway",

"nan_Poland","nan_Rwanda","nan_South Africa","nan_Trinidad and Tobago","Arizona_US","California_US","Colorado_US","Connecticut_US","Georgia_US","Hawaii_US","Maine_US","New York_US","Oklahoma_US",

"Oregon_US","Pennsylvania_US","Tennessee_US","Washington_US","nan_Uruguay","nan_Venezuela","nan_Vietnam"]

decay_1_2_fatality=["nan_Afghanistan","nan_Belarus","nan_Djibouti","nan_Mali","nan_Mexico","nan_Peru","nan_Serbia","Kentucky_US","Puerto Rico_US","Texas_US","nan_Ukraine"]



decay_1_1 =["nan_Algeria","nan_Argentina","Tasmania_Australia","nan_Bolivia","Nova Scotia_Canada","nan_Finland","nan_Hungary","nan_Iraq",

"nan_Jordan","nan_Kyrgyzstan","nan_Malaysia","Alabama_US","Iowa_US","Michigan_US","Missouri_US","Nebraska_US","North Dakota_US ","Ohio_US","South Carolina_US"]

decay_1_1_fatality=["nan_Afghanistan","nan_Ecuador","nan_Kenya","nan_Korea, South","nan_Kosovo","Indiana_US","New Mexico_US","Oregon_US","Rhode Island_US","Tennessee_US","nan_United Arab Emirates"]

 

decay_1_2_last_12_linear_inter =[]

decay_1_2_last_12_linear_inter_fatality=["nan_Angola" ]



linear_last_12=["nan_Afghanistan","nan_Ireland","nan_Benin"]

linear_last_12_fatality=["nan_Guatemala","nan_Ireland"]



decay_8_last_12_linear_inter =[ "nan_Bhutan","Prince Edward Island_Canada","Greenland_Denmark","nan_Fiji","Saint Pierre and Miquelon_France","St Martin_France","nan_Namibia",

"Bonaire, Sint Eustatius and Saba_Netherlands","Curacao_Netherlands","nan_Papua New Guinea","Anguilla_United Kingdom","British Virgin Islands_United Kingdom",]



decay_8_last_12_linear_inter_fatality=[]





tr_frame=train_frame



features_train=tr_frame[names].values   



standard_confirmed_train=tr_frame["ConfirmedCases"].values

standard_fatalities_train=tr_frame["Fatalities"].values

current_confirmed_train=tr_frame["ConfirmedCases"].values



     



features_cv=[]

name_cv=[]

standard_confirmed_cv=[]

standard_fatalities_cv=[]

names_=tr_frame["key"].values

training_horizon=int(features_train.shape[0]/len(unique_keys)) 

print("training horizon = ",training_horizon)

for dd in range(training_horizon-1,features_train.shape[0],training_horizon):

    features_cv.append(features_train[dd])

    name_cv.append(names_[dd])

    standard_confirmed_cv.append(standard_confirmed_train[dd])

    standard_fatalities_cv.append(standard_fatalities_train[dd])

    print (name_cv[-1], standard_confirmed_cv[-1], standard_fatalities_cv[-1])

    

 



features_cv=np.array(features_cv)

preds_confirmed_cv=np.zeros((features_cv.shape[0],horizon))

preds_confirmed_standard_cv=np.zeros((features_cv.shape[0],horizon))



preds_fatalities_cv=np.zeros((features_cv.shape[0],horizon))

preds_fatalities_standard_cv=np.zeros((features_cv.shape[0],horizon))



overal_rmsle_metric_confirmed=0.0



for j in range (preds_confirmed_cv.shape[1]):



    this_features_cv=features_cv                          



    preds=predict(features_cv, input_name=model_directory +"confirmedc"+ str(j))

    preds_confirmed_cv[:,j]=preds

    print (" modelling confirmed, case %d, , original cv %d and after %d "%(j,this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[] 

for ii in range (preds_confirmed_cv.shape[0]):

    current_prediction=standard_confirmed_cv[ii]

    if current_prediction==0 :

        current_prediction=0.1   

    this_preds=preds_confirmed_cv[ii].tolist()

    name=name_cv[ii]

    reserve=this_preds[0]

    #overrides



	

    if name in normal:

        this_preds=this_preds	

	

    elif name in decay_4_first_10_then_1:

        this_preds=decay_4_first_10_then_1_f(this_preds)

		

    elif name in decay_16_first_10_then_1:

        this_preds=decay_16_first_10_then_1_f(this_preds)		

		

    elif name in decay_4_last_12_linear_inter:

        this_preds=decay_4_last_12_linear_inter_f(this_preds)		   

        

    elif name in decay_8_last_12_linear_inter:

        this_preds=decay_8_last_12_linear_inter_f(this_preds)		           

        

    elif name in decay_4:

        this_preds=decay_4_f(this_preds)

    

    elif name in decay_2:

        this_preds=decay_2_f(this_preds)

        

    elif name in decay_2_last_12_linear_inter:

        this_preds=decay_2_last_12_linear_inter_f(this_preds)

        

    elif name in decay_1_5:

        this_preds=decay_1_5_f(this_preds)   

        

    elif name in decay_1_5_last_12_linear_inter:

        this_preds=decay_1_5_last_12_linear_inter_f(this_preds)           

        

        

    elif name in decay_1_2:

        this_preds=decay_1_2_f(this_preds)  

        

    elif name in decay_1_2_last_12_linear_inter:

        this_preds=decay_1_2_last_12_linear_inter_f(this_preds)          

        

    elif name in decay_1_1:

        this_preds=decay_1_1_f(this_preds)  

        

    elif name in linear_last_12:

        this_preds=linear_last_12_f(this_preds)

        

    elif name in acceleratorx2:

        this_preds=acceleratorx2_f(this_preds)         



        

    elif name in stay_same or  "China" in name:

        this_preds=stay_same_f(this_preds)      





    for j in range (preds_confirmed_cv.shape[1]):

                current_prediction+=np.expm1(max(0,this_preds[j]))

                preds_confirmed_standard_cv[ii][j]=current_prediction





for j in range (preds_confirmed_cv.shape[1]):



    this_features_cv=features_cv

                             

    preds=predict(features_cv, input_name=model_directory +"fatalc"+ str(j))

    preds_fatalities_cv[:,j]=preds

    print (" modelling fatalities, case %d, original cv %d and after %d "%( j,this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[]

for ii in range (preds_fatalities_cv.shape[0]):

    current_prediction=standard_fatalities_cv[ii]

        

    this_preds=preds_fatalities_cv[ii].tolist()

    name=name_cv[ii]

    reserve=this_preds[0]

    #overrides

   

    ####fatality special

	

    if name in normal_fatality:

        this_preds=this_preds	 	

	

    elif name in decay_4_first_10_then_1_fatality:

        this_preds=decay_4_first_10_then_1_f(this_preds) 

		

    elif name in decay_16_first_10_then_1_fatality:

        this_preds=decay_16_first_10_then_1_f(this_preds)

        

    elif name in decay_4_last_12_linear_inter_fatality:

        this_preds=decay_4_last_12_linear_inter_f(this_preds)		   

        

    elif name in decay_8_last_12_linear_inter_fatality:

        this_preds=decay_8_last_12_linear_inter_f(this_preds)		             



    elif name in decay_4_fatality:

        this_preds=decay_4_f(this_preds)		

		

    elif name in decay_2_fatality:

        this_preds=decay_2_f(this_preds)        



    elif name in decay_2_last_12_linear_inter_fatality:

        this_preds=decay_2_last_12_linear_inter_f(this_preds)

        

    elif name in decay_1_5_fatality:

        this_preds=decay_1_5_f(this_preds)   	

        

    elif name in decay_1_5_last_12_linear_inter_fatality:

        this_preds=decay_1_5_last_12_linear_inter_f(this_preds)         

               

    elif name in decay_1_2_fatality:

        this_preds=decay_1_2_f(this_preds)   

        

    elif name in decay_1_2_last_12_linear_inter_fatality:

        this_preds=decay_1_2_last_12_linear_inter_f(this_preds)        

        

    elif name in decay_1_1_fatality:

        this_preds=decay_1_1_f(this_preds)         

 

    elif name in linear_last_12_fatality:

        this_preds=linear_last_12_f(this_preds) 

         

    elif name in acceleratorx2_fatality:

        this_preds=acceleratorx2_f(this_preds)

        

    elif name in stay_same_fatality:     

        this_preds=stay_same_f(this_preds) 

        

     

    ####general   

    elif name in normal:

        this_preds=this_preds	   

	

    elif name in decay_4_first_10_then_1:

        this_preds=decay_4_first_10_then_1_f(this_preds)

		

    elif name in decay_16_first_10_then_1:

        this_preds=decay_16_first_10_then_1_f(this_preds)		

        

    elif name in decay_4_last_12_linear_inter:

        this_preds=decay_4_last_12_linear_inter_f(this_preds)		       

        

    elif name in decay_8_last_12_linear_inter:

        this_preds=decay_8_last_12_linear_inter_f(this_preds)		          



    elif name in decay_4:

        this_preds=decay_4_f(this_preds)        

		

    elif name in decay_2:

        this_preds=decay_2_f(this_preds)

        

    elif name in decay_2_last_12_linear_inter:

        this_preds=decay_2_last_12_linear_inter_f(this_preds)

        

    elif name in decay_1_5:

        this_preds=decay_1_5_f(this_preds)  

        

    elif name in decay_1_5_last_12_linear_inter:

        this_preds=decay_1_5_last_12_linear_inter_f(this_preds)            

        

    elif name in decay_1_2:

        this_preds=decay_1_2_f(this_preds) 

        

    elif name in decay_1_2_last_12_linear_inter:

        this_preds=decay_1_2_last_12_linear_inter_f(this_preds)        

        

    elif name in decay_1_1:

        this_preds=decay_1_1_f(this_preds)            

        

    elif name in linear_last_12:

        this_preds=linear_last_12_f(this_preds) 

        

    elif name in acceleratorx2:

        this_preds=acceleratorx2_f(this_preds)                 

        

    elif name in stay_same or  "China" in name:

        this_preds=stay_same_f(this_preds)         

          

    

    for j in range (preds_fatalities_cv.shape[1]):

                    current_prediction+=np.expm1(max(0,this_preds[j]))

                    preds_fatalities_standard_cv[ii][j]=current_prediction
key_to_confirmed_rate={}

key_to_fatality_rate={}

key_to_confirmed={}

key_to_fatality={}

print(len(features_cv), len(name_cv),len(standard_confirmed_cv),len(standard_fatalities_cv)) 

print(preds_confirmed_cv.shape,preds_confirmed_standard_cv.shape,preds_fatalities_cv.shape,preds_fatalities_standard_cv.shape) 



for j in range (len(name_cv)):

    

    key_to_confirmed_rate[name_cv[j]]=preds_confirmed_cv[j,:].tolist()

    #print(key_to_confirmed_rate[name_cv[j]])

    key_to_fatality_rate[name_cv[j]]=preds_fatalities_cv[j,:].tolist()

    key_to_confirmed[name_cv[j]]  =preds_confirmed_standard_cv[j,:].tolist()  

    key_to_fatality[name_cv[j]]=preds_fatalities_standard_cv[j,:].tolist()  

    
train_new=train[["Date","ConfirmedCases","Fatalities","key","rate_ConfirmedCases","rate_Fatalities"]]



test_new_count=pd.merge(test,train_new, how="left", left_on=["key","Date"], right_on=["key","Date"] ).reset_index(drop=True)

test_new_count
def fillin_columns(frame,key_column, original_name, training_horizon, test_horizon, unique_values, key_to_values):

    keys=frame[key_column].values

    original_values=frame[original_name].values.tolist()

    print(len(keys), len(original_values), training_horizon ,test_horizon,len(key_to_values))

    

    for j in range(unique_values):

        current_index=(j * (training_horizon +test_horizon )) +training_horizon 

        current_key=keys[current_index]

        values=key_to_values[current_key]

        co=0

        for g in range(current_index, current_index + test_horizon):

            original_values[g]=values[co]

            co+=1

    

    frame[original_name]=original_values

 



all_days=int(test_new.shape[0]/len(unique_keys))



tr_horizon=all_days-horizon

print(all_days,tr_horizon, horizon )



fillin_columns(test_new_count,"key", 'ConfirmedCases', tr_horizon, horizon, len(unique_keys), key_to_confirmed)    

fillin_columns(test_new_count,"key", 'Fatalities', tr_horizon, horizon, len(unique_keys), key_to_fatality)   

fillin_columns(test_new_count,"key", 'rate_ConfirmedCases', tr_horizon, horizon, len(unique_keys), key_to_confirmed_rate)   

fillin_columns(test_new_count,"key", 'rate_Fatalities', tr_horizon, horizon, len(unique_keys), key_to_fatality_rate)   



######################HERE WE ADD THE COUNT MODEL, WHERE THE COUNTS IN THE RATE MODEL ARE ZERO ##################



test_dat=test_new['Fatalities'].values.tolist()

test_new_countfat=test_new_count['Fatalities'].values.tolist()

for j in range(len(test_dat)):

    if test_dat[j]<1:

        test_dat[j]=test_new_countfat[j]        



test_new2= test_new.copy()

test_new2['Fatalities']=test_dat

test_new2



submission=test_new2[["ForecastId","ConfirmedCases","Fatalities"]]



submission.to_csv("submission.csv", index=False)
