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
import math

import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.cluster import KMeans





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



  

def get_cumulative(frame, key, target, new_target_name="diff"):

    

    corrections = 0

    group_keys = frame[ key].values.tolist()

    target = frame[target].values.tolist()



    for i in range(1, len(group_keys) ):

        previous_group = group_keys[i - 1]

        current_group = group_keys[i]



        previous_value = target[i - 1]

        current_value = target[i]

         

        if current_group == previous_group:

                target[i]=previous_value + current_value

          

        target[i] =max(0,target[i] )



    frame[new_target_name] = np.array(target)    

    

        

def get_difference(frame, key, target, new_target_name="diff"):

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

                rate[i]=max(0,current_value-previous_value)

          

        rate[i] =max(0,rate[i] )



    frame[new_target_name] = np.array(rate)

   



def get_time(frame, key, new_target_name="time"):



    corrections = 0

    group_keys = frame[ key].values.tolist()

    rate=[1 for k in range (len(group_keys))]

    time_counter=1

    for i in range(1, len(group_keys) ):

        previous_group = group_keys[i - 1]

        current_group = group_keys[i]

         

        if current_group == previous_group:

                time_counter+=1

                rate[i]=time_counter

        else :

                time_counter=1

                rate[i]=time_counter                



    frame[new_target_name] = np.array(rate)



def get_x_day_min_max_avg(frame, key, targetcol,window=7, new_target_name="window"):



    corrections = 0

    group_keys = frame[ key].values.tolist()

    tar = frame[ targetcol].values.tolist()

    

    max_values=[0.0 for k in range (len(group_keys))]

    min_values=[0.0 for k in range (len(group_keys))]

    mean_values=[0.0 for k in range (len(group_keys))] 

    

    time_counter=1

    this_group_stats=[]    

    this_group_stats=[tar[0]]

        

    max_values[0]=np.max(this_group_stats[max(0,len(this_group_stats)-window) :len(this_group_stats)] )

    min_values[0]=np.min(this_group_stats[max(0,len(this_group_stats)-window) :len(this_group_stats)] )

    mean_values[0]=np.mean(this_group_stats[max(0,len(this_group_stats)-window) :len(this_group_stats)] )      

    

    for i in range(1, len(group_keys) ):

        previous_group = group_keys[i - 1]

        current_group = group_keys[i]

        this_target=tar[i]

         

        if current_group == previous_group:

                this_group_stats.append(this_target)

                time_counter+=1

                max_values[i]=np.max(this_group_stats[max(0,len(this_group_stats)-window) :len(this_group_stats)] )

                min_values[i]=np.min(this_group_stats[max(0,len(this_group_stats)-window) :len(this_group_stats)] )

                mean_values[i]=np.mean(this_group_stats[max(0,len(this_group_stats)-window) :len(this_group_stats)] )   

                

        else :

            

                this_group_stats=[]

                this_group_stats.append(this_target)                

                time_counter=1

                

                

                max_values[i]=np.max(this_group_stats[max(0,len(this_group_stats)-window) :len(this_group_stats)] )

                min_values[i]=np.min(this_group_stats[max(0,len(this_group_stats)-window) :len(this_group_stats)] )

                mean_values[i]=np.mean(this_group_stats[max(0,len(this_group_stats)-window) :len(this_group_stats)] )               



    frame[new_target_name+"_max" ] = np.array(max_values)    

    frame[new_target_name+"_min"  ] = np.array(min_values) 

    frame[new_target_name+"_mean"  ] = np.array(mean_values) 

    

def get_difference_special(frame, key, target, new_target_name="diff"):

    import numpy as np





    corrections = 0

    group_keys = frame[ key].values.tolist()

    target = frame[target].values.tolist()

    rate=[0 for k in range (len(target))]

    equal_counter=0

    for i in range(1, len(group_keys) ):

        previous_group = group_keys[i - 1]

        current_group = group_keys[i]



        previous_value = target[i - 1]

        current_value = target[i]

         

        if current_group == previous_group:

                rate[i]=max(0,current_value-previous_value)

                if equal_counter==0: 

                    rate[i-1]=rate[i]

                equal_counter+=1

        else :

            equal_counter=0

        rate[i] =max(0,rate[i] )



            

    frame[new_target_name] = np.array(rate) 



def get_data_by_key(dataframe, key, key_value, fields=None):

    mini_frame=dataframe[dataframe[key]==key_value]

    if not fields is None:                

        mini_frame=mini_frame[fields].values

        

    return mini_frame



#pinball loss with 3 different predictions for different quantiles

def pinball_loss_many_marios(ytrue,pred_05,pred_50,pred_95, weight):

    assert len(ytrue)==len(pred_05)==len(pred50)==len(pred95)==len(weight)

    total_pin_ball=0.0

    total_count=0.0

    for i in range(len(ytrue)):

        error_05=0.0

        error_50=0.0

        error_95=0.0

        total_count+=1.

        #####0.05 quantile

        if ytrue[i]>=pred_05[i]:

            error_05+=0.05*(ytrue[i]-pred_05[i])

        else :

            error_05+=(1-0.05)*(pred_05[i]-ytrue[i])

        #####0.50 quantile        

        if ytrue[i]>=pred_50[i]:

            error_50+=0.5*(ytrue[i]-pred_50[i])

        else :

            error_50+=(1-0.5)*(pred_50[i]-ytrue[i])  

        #####0.95 quantile             

        if ytrue[i]>=pred_95[i]:

            error_95+=0.95*(ytrue[i]-pred_95[i])

        else :

            error_95+=(1-0.95)*(pred_95[i]-ytrue[i]) 

        

        total_pin_ball+=weight[i]*(error_05 + error_50 + error_95)/3.

    return   total_pin_ball/ total_count 



#pinball loss assuming prediction is the same for all .05,.5,0.95    

def pinball_loss_single(ytrue,pred, weight):

    assert len(ytrue)==len(pred)==len(weight)

    total_pin_ball=0.0

    total_count=0.0

    for i in range(len(ytrue)):

        error_05=0.0

        error_50=0.0

        error_95=0.0

        total_count+=1.

        #####0.05 quantile

        if ytrue[i]>=pred[i]:

            error_05+=0.05*(ytrue[i]-pred[i])

        else :

            error_05+=(1-0.05)*(pred[i]-ytrue[i])

        #####0.50 quantile        

        if ytrue[i]>=pred[i]:

            error_50+=0.5*(ytrue[i]-pred[i])

        else :

            error_50+=(1-0.5)*(pred[i]-ytrue[i])  

        #####0.95 quantile             

        if ytrue[i]>=pred[i]:

            error_95+=0.95*(ytrue[i]-pred[i])

        else :

            error_95+=(1-0.95)*(pred[i]-ytrue[i]) 

        

        total_pin_ball+=weight[i]*(error_05 + error_50 + error_95)/3.

    return   total_pin_ball/ total_count  



#pinball loss assuming prediction is the same for all .05,.5,0.95    

def pinball_loss_single_with_t(ytrue,pred, weight, t=0.05):

    assert len(ytrue)==len(pred)==len(weight)

    total_pin_ball=0.0

    total_count=0.0

    for i in range(len(ytrue)):

        error=0.0

        total_count+=1.

        #####0.05 quantile

        if t==0.05:

            if ytrue[i]>=pred[i]:

                error+=0.05*(ytrue[i]-pred[i])

            else :

                error+=(1-0.05)*(pred[i]-ytrue[i])

        elif t==0.5:

            #####0.50 quantile        

            if ytrue[i]>=pred[i]:

                error+=0.5*(ytrue[i]-pred[i])

            else :

                error+=(1-0.5)*(pred[i]-ytrue[i])  

        elif t==0.95:

            #####0.95 quantile     

            if ytrue[i]>=pred[i]:

                error+=0.95*(ytrue[i]-pred[i])

            else :

                error+=(1-0.95)*(pred[i]-ytrue[i])



        total_pin_ball+=weight[i]*(error)

        

    return   total_pin_ball/ total_count  

    

def fix_target(frame, key, target, new_target_name="target"):

    import numpy as np



    corrections = 0

    group_keys = frame[ key].values.tolist()

    target = frame[target].values.tolist()



    for i in range(1, len(group_keys) - 1):

        previous_group = group_keys[i - 1]

        current_group = group_keys[i]



        target[i] =max(0,target[i] )#correct negative values



    frame[new_target_name] = np.array(target)

    



directory="/kaggle/input/covid19-global-forecasting-week-5/"

model_directory="/kaggle/input/model-v9-2/model"

train_file="train.csv"

test_file="test.csv"



bagging=1

minimum_count_for_rate_model=2000



extra_stable_columns=None

geo_dir=None

extra_stable_columns=None

group_by_columns=None

group_names=None

train_frame_supplamanteary=None





##################load data and bring them into previous competition format################



use_external=True

external_file="/kaggle/input/extra-day-11/extra_data_11_5_2020_v2.csv" #use external data from https://www.worldometers.info/coronavirus/#countries

#data obtained with the scraper: https://www.kaggle.com/philippsinger/covid-w5-worldometer-scraper



holdder_cumulative={}

holdder={}



train=pd.read_csv(directory + train_file, parse_dates=["Date"] , engine="python")

test=pd.read_csv(directory + test_file, parse_dates=["Date"], engine="python")



train.head()



train_fatalities=train[train.Target=="Fatalities"]

train_confirmed=train[train.Target=="ConfirmedCases"]



if use_external:

    extra_data=pd.read_csv(external_file)

    extra_data["key"]=extra_data[["County","Province_State","Country_Region"]].apply(

        lambda row: str(row[0]) + "_" + str(row[1])+ "_" + str(row[2]),axis=1)

    extra_train_fatalities=extra_data[extra_data.Target=="Fatalities"]

    extra_train_confirmed=extra_data[extra_data.Target=="ConfirmedCases"]

    print ("extra_train_fatalities shape", extra_train_fatalities.shape)

    print ("extra_train_confirmed shape", extra_train_confirmed.shape) 

    holdder={}

    for f in [extra_train_confirmed,extra_train_fatalities]:

        keysss=f["key"].values

        valuess=f["TargetValue"].values

        for g in range (len(keysss)):

            if keysss[g] in holdder:

                holdder[keysss[g]].append(valuess[g])

                assert len(holdder[keysss[g]])==2

            else :

                holdder[keysss[g]]=[valuess[g]]

        

            



train_fatalities.columns=["Id_Fatalities","County","Province_State","Country_Region",

                          "Population","Weight_Fatalities","Date","Target","diff_Fatalities"]



train_confirmed.columns=["Id_ConfirmedCases","County","Province_State","Country_Region",

                         "Population","Weight_ConfirmedCases","Date","Target","diff_ConfirmedCases"]



train_confirmed.drop("Population", inplace=True, axis=1)



train_fatalities.drop("Target", axis=1, inplace=True)

train_confirmed.drop("Target", axis=1, inplace=True)



train=pd.merge(train_confirmed, train_fatalities, how="left", left_on=["County","Province_State","Country_Region","Date"],

              right_on=["County","Province_State","Country_Region","Date"])



train=train[["Id_Fatalities","Id_ConfirmedCases","Date","County","Province_State","Country_Region",

                          "Population","Weight_ConfirmedCases","Weight_Fatalities","diff_ConfirmedCases","diff_Fatalities"]]





train["key"]=train[["County","Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1])+ "_" + str(row[2]),axis=1)





if use_external:

    for country_key,arra in holdder.items():

        print (" insterting key %s" % (country_key))

        confirmed_diff_hold,fatalities_diff_hold=holdder[country_key]

        mini_train=train[train.key==country_key]#.values.tolist()

        #print(mini_train.head(10))

        #fat_values=train[train.key==country_key,"diff_Fatalities"].values.tolist()  

        conf_values=mini_train["diff_ConfirmedCases"].values.tolist()         

        fat_values=mini_train["diff_Fatalities"].values.tolist()  

        

        for j in range (1, len(conf_values)):

            conf_values[j-1]= conf_values[j]

            fat_values[j-1]= fat_values[j]

            

        conf_values[-1]=confirmed_diff_hold    

        fat_values[-1]=fatalities_diff_hold  

        



        train.loc[train.key==country_key, 'diff_ConfirmedCases'] = np.array(conf_values)

        train.loc[train.key==country_key, 'diff_Fatalities'] =np.array(fat_values)



    

get_time(train, "key", new_target_name="time") #get time





fix_target(train, "key", "diff_ConfirmedCases", new_target_name="diff_ConfirmedCases")

fix_target(train, "key", "diff_Fatalities", new_target_name="diff_Fatalities")



get_cumulative(train, "key", "diff_ConfirmedCases", new_target_name="ConfirmedCases")

get_cumulative(train, "key", "diff_Fatalities", new_target_name="Fatalities")



if use_external:

    last_date=train[train["Date"]== train["Date"].max()]

    all_keys=last_date["key"].values.tolist()

    all_con=last_date["ConfirmedCases"].values.tolist()

    all_fat=last_date["Fatalities"].values.tolist()

    holdder_cumulative={}

    for jj in range (len(all_keys)):

        if all_keys[jj] in holdder:

            print (" adding %s to cumulative holder " %(all_keys[jj]))

            holdder_cumulative[all_keys[jj]]=[all_con[jj] ,all_fat[jj] ]

    print (" there are %d elements in the extra data " %(len(holdder_cumulative)))

    assert (len(holdder_cumulative))==len(holdder)



        



rate(train, "key", "ConfirmedCases", new_target_name="rate_" +"ConfirmedCases" )

rate(train, "key", "Fatalities", new_target_name="rate_" +"Fatalities" )



train["dow"]=train["Date"].dt.dayofweek

print(train["dow"].head(10))





#target1="ConfirmedCases"

#target2="Fatalities"

#key="key"



#train.to_csv(directory + "train_reshapedv2.csv", index=False)





max_train_date=train["Date"].max()

max_test_date=test["Date"].max()

min_test_date=test["Date"].min()

horizon=  (max_test_date-max_train_date).days

print ("horizon", int(horizon))

print ("max train date", max_train_date)

print ("max test date", max_test_date)

print ("min test date", min_test_date)



test_fatalities=test[test.Target=="Fatalities"]

test_confirmed=test[test.Target=="ConfirmedCases"]



test_fatalities.columns=["Id_Fatalities","County","Province_State","Country_Region",

                          "Population","Weight_Fatalities","Date","Target_Fatalities"]



test_confirmed.columns=["Id_ConfirmedCases","County","Province_State","Country_Region",

                         "Population","Weight_ConfirmedCases","Date","Target_ConfirmedCases"]



test_confirmed.drop("Population", inplace=True, axis=1)



test=pd.merge(test_confirmed, test_fatalities, how="left", left_on=["County","Province_State","Country_Region","Date"],

              right_on=["County","Province_State","Country_Region","Date"])





test=test[["Id_Fatalities","Id_ConfirmedCases","Date","County","Province_State","Country_Region",

                          "Population","Weight_ConfirmedCases","Weight_Fatalities","Target_ConfirmedCases","Target_Fatalities"]]                         



test["key"]=test[["County","Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1])+ "_" + str(row[2]),axis=1)





#test.to_csv(directory + "test_reshaped.csv", index=False)



key="key"

unique_keys=train[key].unique()
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

                     population,weight_confirmed,weight_fatalities,day_of_week,time,

                     horizon ,size=20, windows=[3,7], days_back_confimed=[1,10,100], days_back_fatalities=[1,2,10], 

                    extra_data=None, groups_data=None, windows_group=[3,7], size_group=20,

                    days_back_confimed_group=[1,10,100]):

    targets=[]

    if not extra_data is None:

        assert len(extra_stable_columns)==extra_data.shape[1]

        

    if not groups_data is None:

        assert len(group_names)==groups_data.shape[1]        

    names=[]    

    names=["lag_confirmed_rate" + str(k+1) for k in range (28)]

    names+=["lag_confirmed_count" + str(k+1) for k in range (28)]    

    for day in days_back_confimed:

        names+=["days_ago_confirmed_count_" + str(day) ]

    for window in windows:        

        names+=["ma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]

        #names+=["std" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)] 

        #names+=["ewma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]         

        names+=["ma" + str(window) + "_count_confirmed" + str(k+1) for k in range (size)]        

        

    names+=["lag_fatalities_rate" + str(k+1) for k in range (28)]

    names+=["lag_fatalities_count" + str(k+1) for k in range (28)]    

    

    for day in days_back_fatalities:

        names+=["days_ago_fatalitiescount_" + str(day) ]    

    for window in windows:        

        names+=["ma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]

        #names+=["std" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]  

        #names+=["ewma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)] 

        names+=["ma" + str(window) + "_count_fatalities" + str(k+1) for k in range (size)]

        

    names+=["confirmed_level"]

    names+=["fatalities_level"]  

    

    names+=["weight_confirmed"]

    names+=["weight_fatalities"]     

    names+=["population"]  

    names+=["dowthis"]      

    names+=["timethis"]          



    

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

        lag_confirmed_rate=get_lags(rate_confirmed, i, size=28)

        lag_confirmed_count=get_lags(count_confirmed, i, size=28)        

        row_features+=lag_confirmed_rate

        row_features+=lag_confirmed_count        

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

        lag_fatalities_rate=get_lags(rate_fatalities, i, size=28)

        lag_fatalities_count=get_lags(count_fatalities, i, size=28)        

        row_features+=lag_fatalities_rate

        row_features+=lag_fatalities_count        

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

        

        conf_weight=  weight_confirmed[i]

        fat_weight=  weight_fatalities[i]        

        current_population=population[i]

        doweek=day_of_week[i]

        tim=time[i]

        

        

                

        row_features+=[conf_weight]

        row_features+=[fat_weight]

        row_features+=[current_population]

        row_features+=[doweek]

        row_features+=[tim]        

        

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

                                               mini_frame["diff_Fatalities"].values,mini_frame["Population"].values

                                              ,mini_frame["Weight_ConfirmedCases"].values,mini_frame["Weight_Fatalities"].values,

                                              mini_frame["dow"].values,mini_frame["time"].values, horizon ,size=size, windows=windows,

                                              days_back_confimed=days_back_confimed, days_back_fatalities=days_back_fatalities,

                                              extra_data=mini_frame[extra_stable_].values if not extra_stable_ is None else None,

                                              groups_data=mini_frame[group_nams].values if not group_nams is None else None,

                                              windows_group=windows_group, size_group=size_group, 

                                              days_back_confimed_group=days_back_confimed_group)

    #print (mini_frame_with_features.shape[0])

    return mini_frame_with_features



size=10

windows=[3]

days_back_confimed=[1,5,10,20,50,100,500]

days_back_fatalities=[1,2,5,10,20,50,200]



size_group=10

windows_group=[3,5]

days_back_confimed_group=[1,10,100]





from tqdm import tqdm

train_frame=[]

size=10

windows=[3]

days_back_confimed=[1,5,10,20,50,100,250,500,1000]

days_back_fatalities=[1,2,5,10,20,50]



size_group=10

windows_group=[3]

days_back_confimed_group=[1,10,100]





print ("total unique keys ", len(train['key'].unique()))

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

print ("total unique new keys" , len(new_unique_keys))
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

       

       params = {'objective': 'mae',

                'metric': 'mae',

                'boosting': 'gbdt',

                'learning_rate': 0.005, #change here    

                'drop_rate':0.005,

                'alpha': 0.95, 

                'skip_drop':0.6,

                'max_drop':2,                

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

                'num_threads':38

                }

       d_train = lgb.Dataset(X_ts,y_cs, weight=wts, free_raw_data=False)#np.log1p(

       if not type(yt) is type(None):           

           d_cv = lgb.Dataset(xt,yt, free_raw_data=False, reference=d_train)#, reference=d_train

           model = lgb.train(params,d_train,num_boost_round=500,

                             valid_sets=d_cv,



                             verbose_eval=500 ) #1000                        

           

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

            joblib.dump((model), output_name+ "_" +str(n))



   # divide with number of bags to create an average estimate  

   baggedpred/= estimators

     

   return baggedpred





def predict(xtest, estimators ,input_name=None):

   #print (type(yt))

   # create array object to hold predictions 

  

   baggedpred=np.array([ 0.0 for d in range(0, xtest.shape[0])]) 

   for n in range (0, estimators):    

       print("loading model %s"% (input_name+ "_" +str(n)))

       model=  joblib.load( input_name+ "_" +str(n)) 

       preds=model.predict(xtest)               

       baggedpred+=preds

   baggedpred/= estimators



   return baggedpred
names=[]

names+=["lag_confirmed_rate" + str(k+1) for k in [6,13,20]]

names+=["lag_confirmed_count" + str(k+1) for k in [6,13,20]]

for day in days_back_confimed:

    names+=["days_ago_confirmed_count_" + str(day) ]

for window in windows:        

    names+=["ma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]

    #names+=["std" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)] 

    #names+=["ewma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]         

    names+=["ma" + str(window) + "_count_confirmed" + str(k+1) for k in range (size)]  



names+=["lag_fatalities_rate" + str(k+1) for k in [6,13,20]]

names+=["lag_fatalities_count" + str(k+1) for k in [6,13,20]]

for day in days_back_fatalities:

    names+=["days_ago_fatalitiescount_" + str(day) ]    

for window in windows:        

    names+=["ma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]

    #names+=["std" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]  

    #names+=["ewma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]    

    names+=["ma" + str(window) + "_count_fatalities" + str(k+1) for k in range (size)]  

      

#names+=["confirmed_level"]

#names+=["fatalities_level"]  

names+=["dowthis"]  

#names+=["timethis"]

#names+=["weight_confirmed"]

#names+=["weight_fatalities"]     

#names+=["population"]  



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
##############################################################################################################

############################ THIS PART IS THE TRAINING CODE - COMMENT IT OUT TO MODEL IT. ####################

############################ HERE I USE COUNT ON DIFFERENCES MODEL TO ESTIMATE CONFIRMED AND FAT.#############

############################ BEAR IN MIND , PAST EXPERIENCE HAS SHOWN THAT RESULTS MIGHT BE ##################

############################ SLIGHTLY DIFFERENT THAN MY CURRENT UPLOADED ONE (HOPEFULLY NOT BY MUCH)  ########

##############################################################################################################



"""



#################Full model



tr_frame=train_frame



    

target_confirmed=["confirmed_count_plus" + str(k+1) for k in range (horizon)]    

target_fatalities=["fatalities_count_plus" + str(k+1) for k in range (horizon)] 

weight_confirmed="timethis"    

weight_fatalities="timethis"

seed=1412



target_confirmed_train=tr_frame[target_confirmed].values

print ("  original shape of train is {}  ".format( target_confirmed_train.shape) )



weight_confirmed_train=tr_frame[weight_confirmed].values

print ("  original shape of weight confirmed for train is {}  ".format( weight_confirmed_train.shape) )



target_fatalities_train=tr_frame[target_fatalities].values

print ("  original shape of train fatalities is {}  ".format( target_fatalities_train.shape) )



weight_fatalities_train=tr_frame[weight_fatalities].values

print ("  original shape of weight fatalities for train is {}  ".format( weight_fatalities_train.shape) )



features_train=tr_frame[names].values   

current_fatalities_train=tr_frame["Fatalities"].values

current_confirmed_train=tr_frame["ConfirmedCases"].values

 

print("features_train.shape", features_train.shape)    



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

    standard_confirmed_cv.append(current_confirmed_train[dd])

    standard_fatalities_cv.append(current_fatalities_train[dd])

    print (name_cv[-1], standard_confirmed_cv[-1], standard_fatalities_cv[-1])

    



current_confirmed_train_index=[k for k in range(len(current_confirmed_train)) if current_confirmed_train[k]>0]

target_confirmed_train=target_confirmed_train[current_confirmed_train_index]

target_fatalities_train=target_fatalities_train[current_confirmed_train_index] 



weight_confirmed_train=weight_confirmed_train[current_confirmed_train_index]

weight_fatalities_train=weight_fatalities_train[current_confirmed_train_index] 



features_train=features_train[current_confirmed_train_index]         

current_confirmed_train=current_confirmed_train[current_confirmed_train_index]

current_fatalities_train=current_fatalities_train[current_confirmed_train_index]  

    

features_cv=np.array(features_cv)



overal_rmsle_metric_confirmed=0.0



for j in range (horizon):

    this_target=target_confirmed_train[:,j]

    index_positive=[k for k in range(len(this_target)) if this_target[k]!=-1]

    this_features=features_train[index_positive]

    this_target=this_target[index_positive]

    this_weight=np.log(weight_confirmed_train[index_positive] +2.)    

    #this_weight=weight_confirmed_train[index_positive]

    #this_weight=np.log(standard_confirmed_train[index_positive]+2.)

    #this_weight=[1. for k in range(len(this_weight))]

    this_features_cv=features_cv                          



    preds=bagged_set_trainc(this_features,this_target,this_weight, seed, bagging,features_cv, xt=None,yt=None, output_name=model_directory +"confirmedc"+ str(j))

    print (" modelling count confirmed, case %d, original train %d, and after %d, original cv %d and after %d "%(

    j,target_confirmed_train.shape[0],this_target.shape[0],this_features_cv.shape[0],this_features_cv.shape[0])) 





for j in range (horizon):

    this_target=target_fatalities_train[:,j]

    index_positive=[k for k in range(len(this_target)) if this_target[k]!=-1]

    this_features=features_train[index_positive]

    this_target=this_target[index_positive]

    

    this_weight=np.log(weight_fatalities_train[index_positive] +2.)  

    

    #this_weight=np.log(standard_confirmed_train[index_positive]+2.)

    #this_weight=[1. for k in range(len(this_weight))]

    

    this_features_cv=features_cv

                             

    preds=bagged_set_trainc(this_features,this_target,this_weight, seed, bagging,features_cv, xt=None,yt=None, output_name=model_directory +"fatalc"+ str(j))

    print (" modelling count fatalities, case %d, original train %d, and after %d, original cv %d and after %d "%(

    j,target_confirmed_train.shape[0],this_target.shape[0],this_features_cv.shape[0],this_features_cv.shape[0]))

    

"""
from scipy.stats import pearsonr

cut_off_count=1.2

###prediction part ##############



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

    

def add_3_contsants_f(array,constants):

    arr=[k for k in array]    

    for j in range(0,min(10,len(array))):

        arr[j]=array[j]+ constants[0]

    for j in range(10,min(20,len(array))):

        arr[j]=array[j]+ constants[1]    

    for j in range(20,len(array)):

        arr[j]=array[j]+ constants[2]  

    return arr



def revert_preds(array):

    arr=[k for k in array] 

    for jjj in range (1,len(array)):

        arr[jjj]=array[jjj-1]

    return arr

    

stay_same=["nan_nan_Belize",

"nan_Falkland Islands (Malvinas)_United Kingdom",

"nan_Greenland_Denmark",

"nan_nan_Suriname",

"nan_nan_Papua New Guinea",

"nan_Saint Barthelemy_France",

"nan_Anguilla_United Kingdom",

"nan_Faroe Islands_Denmark",

"nan_nan_Diamond Princess",

"nan_Saint Pierre and Miquelon_France",

"nan_nan_MS Zaandam"]



not_china=["nan_Hong Kong_China",

           "nan_Jilin_China",

           "nan_Shanghai_China",

           "nan_nan_China"]

    

tr_frame=train_frame



features_train=tr_frame[names].values   



standard_confirmed_train=tr_frame["ConfirmedCases"].values

standard_fatalities_train=tr_frame["Fatalities"].values

current_confirmed_train=tr_frame["ConfirmedCases"].values

 



#based on differences between aggraged level of state level and county level

US_County_decays=[1.12928721917076,

1.11319717445063,

1.09019080003495,

1.09054639211022,

1.07359162437952,

1.07758690695603,

1.09228026420265,

1.08177697041925,

1.03976897078037,

1.02818087417532,

1.03918061105423,

1.03200772948294,

1.02445390740797,

1.02074697729396,

1.03411094916523,

1.0062638939838,

0.994894354473368,

0.962558969500773,

0.972769268084967,

0.963989662013557,

0.994878533190241,

0.973785778384885,

0.955063495068311,

0.937809259916995,

0.938370164891967,

0.894556518582016,

0.889929316241106,

0.87077864868053,

0.857970890863051,

0.828321911237346,

0.801852381304044,

]    

    

#obtained from average per count plus constant    

US_OVERRIDE=[[

23207.3722652086,

23889.0306567728,

24430.0431046741,

24466.7567050266,

23713.2400041044,

22608.4892577587,

21367.8534988397,

20769.6129680705,

21228.1211938592,

21495.4428971699,

22035.5166697628,

21125.9867035126,

20181.3684696248,

18494.0407094816,

18475.5718884067,

19215.0940461046,

19333.1028894944,

19522.3401300248,

19166.0663535153,

18096.9669398416,

16937.4222363175,

16947.6467406821,

16357.855664555,

15976.4969540291,

16624.4012014047,

16262.3505500094,

15364.4351409369,

15028.6796673928,

15579.9384603638,

15834.0289720947,

16576.2953662852,



], [1521.00845305084,

1566.97981581732,

1590.42795692813,

1431.97778497973,

1281.31188429921,

1151.84665265908,

1198.1294341396,

1311.54799108804,

1356.57819142736,

1271.28355785146,

1227.7373880933,

1099.79166367917,

1033.79403187646,

958.252997644407,

1046.47668605323,

1085.99235244191,

1081.7845211432,

994.467577598198,

897.908986384629,

852.023641179723,

879.298706602438,

883.484565921627,

972.507881529548,

922.139598187947,

893.864912358786,

901.67568945017,

783.837418771163,

793.314828070173,

896.614556007508,

859.957640953242,

770.734101867327,

] ]





#ghana reports 0 every 2 days

GHANA_OVERRIDE=[

    [437.34,

160.085898221233,

423.982956408698,

197.208012040989,

438.1867611832,

160.685861259004,

383.667040128953,

175.753495639198,

400.74307732034,

188.782063124033,

479.921308802172,

205.837541082642,

462.821368862156,

344.595696311912,

461.519828860741,

397.104823526226,

521.899614517621,

502.391633158483,

481.702595518469,

467.449979112643,

560.96931390999,

455.537034350064,

482.244071286786,

506.221918800896,

510.67633991002,

462.067712758548,

383.930145751492,

363.267022670611,

453.059337065173,

324.008740298506,

299.563588083208,



] , [1.17583524490721,

0,

1.26096966594506,

0,

1.45773722679996,

0,

1.31769888587172,

0,

1.90002768724863,

0,

1.97527592437864,

1.75370174412068,

1.72406661083427,

1.65081144481032,

1.8676119135745,

2.12479552032021,

2.8060755390374,

2.38864420453819,

1.90622435354147,

1.93737239215752,

1.82606349042584,

2.51189161572323,

2.64862312509722,

2.93841977514869,

2.33613023947002,

1.73078769935274,

1.63014931330317,

1.19478903202637,

1.92652108876484,

1.89717476536816,

1.04840010811724,

] ]

    

add_3_constants={}

#manual adjustements

add_3_constants["nan_nan_Belgium"]=[ [100,150,70] , [10,15,10]   ]

add_3_constants["nan_Ontario_Canada"]=[ [100,70,50] , [20,15,10]   ]

add_3_constants["nan_nan_Canada"]=[ [0,100,100] , [0,0,0]   ]

add_3_constants["nan_nan_Germany"]=[ [0,0,0] , [0,0,0]   ] 

add_3_constants["nan_Quebec_Canada"]=[ [100,200,150] , [20,20,20]   ] 

add_3_constants["nan_nan_Indonesia"]=[ [50,50,50] , [0,0,0]   ]

add_3_constants["nan_nan_Iran"]=[ [100,200,300] , [0,1,10]   ]

add_3_constants["nan_nan_Italy"]=[ [100,200,100] , [0,50,0]   ]

add_3_constants["nan_nan_Romania"]=[ [40,70,70] , [5,5,5] ] 

add_3_constants["nan_nan_Spain"]=[ [500,400,300] , [50,30,0] ] 

add_3_constants["nan_nan_Sweden"]=[ [100,70,50] , [0,5,5] ] 

add_3_constants["nan_nan_Turkey"]=[ [100,50,150] , [0,20,10] ]

add_3_constants["nan_nan_Portugal"]=[ [0,0,0] , [0,0,0] ]

add_3_constants["nan_nan_Kenya"]=[ [0,0,0] , [0,0,0]   ]

add_3_constants["nan_nan_United Kingdom"]=[ [1200,1200,1000] , [0,10,10]   ]

add_3_constants["nan_nan_Denmark"]=[ [10,10,20] , [2,1,1]   ] 

add_3_constants["nan_nan_Netherlands"]=[ [50,70,40] , [0,5,5]   ]

add_3_constants["nan_nan_Poland"]=[ [70,70,70] , [0,0,0]   ] 

add_3_constants["nan_nan_Ukraine"]=[ [50,50,20] , [0,0,0] ] 

add_3_constants["nan_Arizona_US"]=[ [50,50,100] , [5,2,1] ] 

add_3_constants["Los Angeles_California_US"]=[ [50,100,100] , [0,0,5] ] 

add_3_constants["Santa Barbara_California_US"]=[ [0,0,0] , [0,0,0] ] 

add_3_constants["nan_California_US"]=[ [200,200,200] , [0,0,0]   ]

add_3_constants["nan_Colorado_US"]=[ [70,100,100] , [0,0,0]   ]

add_3_constants["nan_Florida_US"]=[ [100,100,100] , [0,0,0]   ]

add_3_constants["nan_Georgia_US"]=[ [150,150,110] , [0,0,0]   ]

add_3_constants["Cook_Illinois_US"]=[ [200,200,400] , [0,0,10]   ]

add_3_constants["nan_Illinois_US"]=[ [0,100,200] , [0,0,10]   ]

add_3_constants["nan_Indiana_US"]=[ [50,100,70] , [0,0,10]   ]

add_3_constants["nan_Louisiana_US"]=[ [50,100,70] , [0,0,10]   ]

add_3_constants["Prince George's_Maryland_US"]=[ [0,0,50] , [0,0,1]   ]

add_3_constants["nan_Maryland_US"]=[ [100,150,200] , [0,0,0]   ]

add_3_constants["nan_Massachusetts_US"]=[ [0,100,200] , [5,10,10] ] 

add_3_constants["nan_Michigan_US"]=[ [0,100,50] , [0,0,0] ] 

add_3_constants["nan_Minnesota_US"]=[ [100,120,110] , [0,0,0] ] 

add_3_constants["nan_New Jersey_US"]=[ [100,200,150] , [0,0,0] ] 

add_3_constants["New York_New York_US"]=[ [100,100,100] , [0,0,0] ] 

add_3_constants["nan_New York_US"]=[ [300,500,500] , [0,0,50] ] 

add_3_constants["nan_North Carolina_US"]=[ [100,250,250] , [0,0,50] ] 

add_3_constants["nan_Ohio_US"]=[ [100,150,100] , [5,10,5]   ] 

add_3_constants["nan_Pennsylvania_US"]=[ [200,200,200] , [0,0,0] ] 

add_3_constants["Philadelphia_Pennsylvania_US"]=[ [100,100,100] , [0,0,0] ] 

add_3_constants["nan_Tennessee_US"]=[ [50,50,70] , [0,0,0] ] 

add_3_constants["nan_Texas_US"]=[ [300,200,500] , [0,0,0]   ]

add_3_constants["nan_Virginia_US"]=[ [100,200,300] , [0,0,0]   ]

add_3_constants["nan_Wisconsin_US"]=[ [30,50,50] , [0,0,0]   ]

add_3_constants["nan_Iowa_US"]=[ [0,50,50] , [0,0,0]   ]

add_3_constants["nan_Kansas_US"]=[ [0,0,50] , [0,0,0]   ]

add_3_constants["Middlesex_Massachusetts_US"]=[ [50,50,50] , [0,0,0]   ]

add_3_constants["Hudson_New Jersey_US"]=[ [0,20,20] , [0,0,0]   ]

add_3_constants["nan_Washington_US"]=[ [0,20,40] , [0,0,0]   ]

add_3_constants["nan_Nebraska_US"]=[ [0,0,0] , [0,0,0]   ]

add_3_constants["Suffolk_New York_US"]=[ [0,0,50] , [0,0,0]   ]

add_3_constants["nan_Rhode Island_US"]=[ [10,20,30] , [0,0,0] ] 

add_3_constants["nan_nan_Serbia"]=[ [20,30,20] , [0,0,0] ] 

add_3_constants["nan_nan_Egypt"]=[ [100,150,170] , [0,0,0] ] 





for cn,vall in add_3_constants.items():

    if cn not in new_unique_keys:

        raise Exception("%s not in unique keys"%(cn))

        



###################### HERE #################

#add_3_constants={}



print(" len add_3_constants", len(add_3_constants))





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

preds_confirmed_non_cumulative_cv=np.zeros((features_cv.shape[0],horizon))



preds_fatalities_cv=np.zeros((features_cv.shape[0],horizon))

preds_fatalities_standard_cv=np.zeros((features_cv.shape[0],horizon))

preds_fatalities_non_cumulative_cv=np.zeros((features_cv.shape[0],horizon))



overal_rmsle_metric_confirmed=0.0



for j in range (preds_confirmed_cv.shape[1]):



    this_features_cv=features_cv                          

    preds=predict(features_cv,bagging, input_name=model_directory +"confirmedc"+ str(j))

    preds_confirmed_cv[:,j]=preds

    print (" modelling confirmed, case %d, , original cv %d and after %d "%(j,this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[] 

for ii in range (preds_confirmed_cv.shape[0]):

    current_prediction=standard_confirmed_cv[ii]

    #if current_prediction==0 :

        #current_prediction=0.1   

    this_preds=preds_confirmed_cv[ii].tolist()

    name=name_cv[ii]

    reserve=this_preds[0]

    #overrides

    

    if use_external and name in holdder :

        print(" name %s is for confirmed" %(name))

        this_preds=revert_preds(this_preds)

    

    

    

    if name in add_3_constants:

         print("name " , name , " is in 3 constants")

        

         this_preds=add_3_contsants_f(this_preds,add_3_constants[name][0])        

        

    if name in stay_same or ("China" in name and name not in not_china):



         this_preds=stay_same_f(this_preds)



    if name=="nan_nan_US":

        print (" ======= overriding USA Conf ========== ")

        this_preds=US_OVERRIDE[0]

        

    if name=="nan_nan_Ghana":

        print (" ======= overriding Ghana Fat ========== ")

        this_preds=GHANA_OVERRIDE[0]  

        

    

    for j in range (preds_confirmed_cv.shape[1]):

                this_pr=max(0,this_preds[j])

                

                if j>18:

                   previous_pr= max(0,this_preds[j-1])

                   if  previous_pr>0.1:

                        if this_pr/previous_pr>cut_off_count:

                            this_pr=np.mean(this_preds[j-8:j+1])

                            this_preds[j]=this_pr

           

                #usa counties overrides            

                if "_US" in name and "nan_" not in name and name not in add_3_constants and j>23:

                     this_preds[j]= this_preds[j]*US_County_decays[j]

                   

                if use_external and j==0 and name in holdder :    

                    current_prediction+=max(0,holdder[name][0])

                    preds_confirmed_standard_cv[ii][j]=current_prediction

                    preds_confirmed_non_cumulative_cv[ii][j]=max(0,holdder[name][0])                

                else:

                    current_prediction+=max(0,this_preds[j])

                    preds_confirmed_standard_cv[ii][j]=current_prediction

                    preds_confirmed_non_cumulative_cv[ii][j]=max(0,this_preds[j])



for j in range (preds_confirmed_cv.shape[1]):



    this_features_cv=features_cv

                             

    preds=predict(features_cv,bagging, input_name=model_directory +"fatalc"+ str(j))

    preds_fatalities_cv[:,j]=preds

    print (" modelling fatalities, case %d, original cv %d and after %d "%( j,this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[]

for ii in range (preds_fatalities_cv.shape[0]):

    current_prediction=standard_fatalities_cv[ii]

        

    this_preds=preds_fatalities_cv[ii].tolist()

    name=name_cv[ii]

    reserve=this_preds[0]

    #overrides

   

    

    if use_external and name in holdder :

        print(" name %s is for fatalities" %(name))        

        this_preds=revert_preds(this_preds)



    ####fatality special

    if name in add_3_constants:

         print("name " , name , " is in 3 constants FATALITIES")        

         this_preds=add_3_contsants_f(this_preds,add_3_constants[name][1])             

        

    if name in stay_same or ("China" in name and name not in not_china):

         this_preds=stay_same_f(this_preds)

           

    if name=="nan_nan_US":

        print (" ======= overriding USA Fat ========== ")

        this_preds=US_OVERRIDE[1] 

        

    if name=="nan_nan_Ghana":

        print (" ======= overriding Ghana Fat ========== ")

        this_preds=GHANA_OVERRIDE[1]         

       

        

    for j in range (preds_fatalities_cv.shape[1]):

                    this_pr=max(0,this_preds[j])

                    if j>18:

                       previous_pr= max(0,this_preds[j-1])

                       if  previous_pr>0.1:

                            if this_pr/previous_pr>cut_off_count:

                                this_pr=np.mean(this_preds[j-8:j+1])

                                this_preds[j]=this_pr    

                                

                    if use_external and j==0 and name in holdder :    

                        current_prediction+=max(0,holdder[name][1])

                        preds_confirmed_standard_cv[ii][j]=current_prediction

                        preds_confirmed_non_cumulative_cv[ii][j]=max(0,holdder[name][1])                                  

                    else :

                        current_prediction+=max(0,this_preds[j])

                        preds_fatalities_standard_cv[ii][j]=current_prediction

                        preds_fatalities_non_cumulative_cv[ii][j]=max(0,this_preds[j])   
#### rate modelling



import lightgbm as lgb

from sklearn.linear_model import Ridge

from sklearn.externals import joblib



def bagged_set_train_rate(X_ts,y_cs,wts, seed, estimators,xtest, xt=None,yt=None, output_name=None):

   #print (type(yt))

   # create array object to hold predictions 

  

   baggedpred=np.array([ 0.0 for d in range(0, xtest.shape[0])]) 

   #print (y_cs[:10])

   #print (yt[:10])  



   #loop for as many times as we want bags

   for n in range (0, estimators):

       ###

       params = {'objective': 'rmse',

                'metric': 'rmse',

                'boosting': 'gbdt',

                'learning_rate': 0.02, #change here    

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

                'num_threads':38

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

            joblib.dump((model), output_name+ "_" +str(n))



   # divide with number of bags to create an average estimate  

   baggedpred/= estimators

     

   return baggedpred
names_rate=[]

names_rate+=["lag_confirmed_rate" + str(k+1) for k in [6,13,20]]

names_rate+=["lag_confirmed_count" + str(k+1) for k in [6,13,20]]



names_rate+=["lag_fatalities_rate" + str(k+1) for k in [6,13,20]]

names_rate+=["lag_fatalities_count" + str(k+1) for k in [6,13,20]]



for day in days_back_confimed:

    names_rate+=["days_ago_confirmed_count_" + str(day) ]

for window in windows:        

    names_rate+=["ma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]

    #names_rate+=["std" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)] 

    #names_rate+=["ewma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]         

    #names_rate+=["ma" + str(window) + "_count_confirmed" + str(k+1) for k in range (size)]  



#names_rate+=["lag_fatalities_rate" + str(k+1) for k in range (size)]

for day in days_back_fatalities:

    names_rate+=["days_ago_fatalitiescount_" + str(day) ]    

for window in windows:        

    names_rate+=["ma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]

    #names_rate+=["std" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]  

    #names_rate+=["ewma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]    

    #names_rate+=["ma" + str(window) + "_count_fatalities" + str(k+1) for k in range (size)]    

#names_rate+=["confirmed_level"]

#names_rate+=["fatalities_level"]  

names_rate+=["dowthis"] 

#names_rate+=["timethis"]



#names_rate+=["weight_confirmed"]

#names_rate+=["weight_fatalities"]     

#names_rate+=["population"]     

if not extra_stable_columns is None and len(extra_stable_columns)>0: 

    names_rate+=[k for k in extra_stable_columns]  

    

if not group_names is None:  

     for gg in range (len(group_names)):

         #names_rate+=["lag_rate_group_"+ str(gg+1) + "_" + str(k+1) for k in range (size_group)]    

         for day in days_back_confimed_group:

            names_rate+=["days_ago_grooupcount_" + str(gg+1) + "_" + str(day) ]             

         for window in windows_group:        

            names_rate+=["ma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]

            #names_rate+=["std_group_" + str(gg+1)+ "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]  

            #names_rate+=["ewma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size)]   
##############################################################################################################

############################ THIS PART IS THE TRAINING CODE - COMMENT IT OUT TO MODEL IT. ####################

############################ HERE I USE to MODEL GROWTH RATE OF CONFIRMED AND FAT. ###########################

############################ BEAR IN MIND , PAST EXPERIENCE HAS SHOWN THAT RESULTS MIGHT BE ##################

############################ SLIGHTLY DIFFERENT THAN MY CURRENT UPLOADED ONE (HOPEFULLY NOT BY MUCH)  ########

##############################################################################################################



"""

tr_frame=train_frame



	

target_confirmed_rate=["confirmed_plus" + str(k+1) for k in range (horizon)]    

target_fatalities_rate=["fatalities_plus" + str(k+1) for k in range (horizon)] 

weight_confirmed="timethis"    

weight_fatalities="timethis"

seed=1412



target_confirmed_train=tr_frame[target_confirmed_rate].values

print ("  original shape of train is {}  ".format( target_confirmed_train.shape) )



weight_confirmed_train=tr_frame[weight_confirmed].values

print ("  original shape of weight confirmed for train is {}  ".format( weight_confirmed_train.shape) )



target_fatalities_train=tr_frame[target_fatalities_rate].values

print ("  original shape of train fatalities is {}  ".format( target_fatalities_train.shape) )



weight_fatalities_train=tr_frame[weight_fatalities].values

print ("  original shape of weight fatalities for train is {}  ".format( weight_fatalities_train.shape) )





standard_confirmed_train=tr_frame["ConfirmedCases"].values

standard_fatalities_train=tr_frame["Fatalities"].values

current_confirmed_train=tr_frame["ConfirmedCases"].values



features_train=tr_frame[names_rate].values   

print("features_train.shape", features_train.shape)





features_cv=[]

name_cv=[]

standard_confirmed_cv=[]

standard_fatalities_cv=[]

names_rate_=tr_frame["key"].values

training_horizon=int(features_train.shape[0]/len(unique_keys)) 

print("training horizon = ",training_horizon)

for dd in range(training_horizon-1,features_train.shape[0],training_horizon):

    features_cv.append(features_train[dd])

    name_cv.append(names_rate_[dd])

    standard_confirmed_cv.append(standard_confirmed_train[dd])

    standard_fatalities_cv.append(standard_fatalities_train[dd])

    print (name_cv[-1], standard_confirmed_cv[-1], standard_fatalities_cv[-1])

    

 

    

current_confirmed_train=[k for k in range(len(current_confirmed_train)) if current_confirmed_train[k]>0]

target_confirmed_train=target_confirmed_train[current_confirmed_train]

target_fatalities_train=target_fatalities_train[current_confirmed_train] 



weight_confirmed_train=weight_confirmed_train[current_confirmed_train_index]

weight_fatalities_train=weight_fatalities_train[current_confirmed_train_index] 



features_train=features_train[current_confirmed_train]         

standard_confirmed_train=standard_confirmed_train[current_confirmed_train]

standard_fatalities_train=standard_fatalities_train[current_confirmed_train]  

    

features_cv=np.array(features_cv)





overal_rmsle_metric_confirmed=0.0



for j in range (horizon):

    this_target=target_confirmed_train[:,j]

    index_positive=[k for k in range(len(this_target)) if this_target[k]!=-1]

    this_features=features_train[index_positive]

    this_target=this_target[index_positive]

    

    this_weight=np.log(weight_confirmed_train[index_positive] +2.)     

    

    #this_weight=weight_confirmed_train[index_positive]    

    #this_weight=np.log(standard_confirmed_train[index_positive]+2.)

    #this_weight=[1. for k in range(len(this_weight))]

    this_features_cv=features_cv                          



    preds=bagged_set_train_rate(this_features,this_target,this_weight, seed, bagging,features_cv, xt=None,yt=None, output_name=model_directory +"confirmed"+ str(j))

    print (" modelling confirmed, case %d, original train %d, and after %d, original cv %d and after %d "%(

    j,target_confirmed_train.shape[0],this_target.shape[0],this_features_cv.shape[0],this_features_cv.shape[0])) 







for j in range (horizon):

    this_target=target_fatalities_train[:,j]

    index_positive=[k for k in range(len(this_target)) if this_target[k]!=-1]

    this_features=features_train[index_positive]

    this_target=this_target[index_positive]

    this_weight=np.log(weight_fatalities_train[index_positive] +2.)      

    #this_weight=weight_fatalities_train[index_positive]        

    #this_weight=np.log(standard_confirmed_train[index_positive]+2.)

    #this_weight=[1. for k in range(len(this_weight))]

    

    this_features_cv=features_cv

                             

    preds=bagged_set_train_rate(this_features,this_target,this_weight, seed, bagging,features_cv, xt=None,yt=None, output_name=model_directory +"fatal"+ str(j))

    print (" modelling fatalities, case %d, original train %d, and after %d, original cv %d and after %d "%(

    j,target_confirmed_train.shape[0],this_target.shape[0],this_features_cv.shape[0],this_features_cv.shape[0])) 

    

"""
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





def add_rate(array, rate):

    arr=[k for k in array]    

    for j in range(len(array)):

            arr[j]+=rate

    return arr 



def revert_preds(array):

    arr=[k for k in array] 

    for jjj in range (1,len(array)):

        arr[jjj]=array[jjj-1]

    return arr





tr_frame=train_frame



features_train=tr_frame[names_rate].values   



standard_confirmed_train=tr_frame["ConfirmedCases"].values

standard_fatalities_train=tr_frame["Fatalities"].values

current_confirmed_train=tr_frame["ConfirmedCases"].values



 

add_constant_rate={}    

add_constant_rate["nan_nan_Brazil"]=[0.003,0.004]

add_constant_rate["nan_nan_Chile"]=[0.002,0.0001]

add_constant_rate["nan_nan_Colombia"]=[0.002,0.0001]

add_constant_rate["nan_nan_Dominican Republic"]=[0.006,0.0001]

add_constant_rate["nan_nan_India"]=[0.015,0.001]

add_constant_rate["nan_nan_El Salvador"]=[0.012,0.001]

add_constant_rate["nan_nan_Kuwait"]=[0.01,0.01]

add_constant_rate["nan_nan_Mexico"]=[0.01,0.002]

add_constant_rate["nan_nan_Pakistan"]=[0.01,0.002]

add_constant_rate["nan_nan_Peru"]=[0.0025,0.001]

add_constant_rate["nan_nan_Qatar"]=[0.0055,0.001]

add_constant_rate["nan_nan_Russia"]=[0.012,0.003]

add_constant_rate["nan_nan_Saudi Arabia"]=[0.003,0.001]

add_constant_rate["nan_nan_Singapore"]=[0.005,0.001]

add_constant_rate["nan_nan_South Africa"]=[0.007,0.002]

add_constant_rate["nan_nan_Nigeria"]=[0.005,0.0001]

add_constant_rate["nan_nan_United Arab Emirates"]=[0.01,0.02]

add_constant_rate["nan_Alabama_US"]=[0.001,0.001]

add_constant_rate["nan_Connecticut_US"]=[0.001,0.001]

add_constant_rate["nan_nan_Kazakhstan"]=[0.001,0.001]

add_constant_rate["nan_nan_Ecuador"]=[0.0001,0.0001]

add_constant_rate["nan_nan_Honduras"]=[0.005,0.0001]

add_constant_rate["nan_nan_Guinea-Bissau"]=[0.0001,0.0001]





for cn,vall in add_constant_rate.items():

    if cn not in new_unique_keys:

        raise Exception("%s not in unique keys"%(cn))



####### HERE UNC #######

#add_constant_rate={}  



print ("len(add_constant_rate)", len(add_constant_rate))

        

    

    

features_cv=[]

name_cv=[]

standard_confirmed_cv=[]

standard_fatalities_cv=[]

names_rate_=tr_frame["key"].values

training_horizon=int(features_train.shape[0]/len(unique_keys)) 

print("training horizon = ",training_horizon)

for dd in range(training_horizon-1,features_train.shape[0],training_horizon):

    features_cv.append(features_train[dd])

    name_cv.append(names_rate_[dd])

    standard_confirmed_cv.append(standard_confirmed_train[dd])

    standard_fatalities_cv.append(standard_fatalities_train[dd])

    print (name_cv[-1], standard_confirmed_cv[-1], standard_fatalities_cv[-1])

    

 

features_cv=np.array(features_cv)

preds_confirmed_rate_cv=np.zeros((features_cv.shape[0],horizon))

preds_confirmed_cumulative_rate_cv=np.zeros((features_cv.shape[0],horizon))



preds_fatalities_rate_cv=np.zeros((features_cv.shape[0],horizon))

preds_fatalities_cumulative_rate_cv=np.zeros((features_cv.shape[0],horizon))



overal_rmsle_metric_confirmed=0.0



for j in range (preds_confirmed_rate_cv.shape[1]):



    this_features_cv=features_cv                          



    preds=predict(features_cv,bagging, input_name=model_directory +"confirmed"+ str(j))

    preds_confirmed_rate_cv[:,j]=preds

    print (" modelling confirmed, case %d, , original cv %d and after %d "%(j,this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[] 

for ii in range (preds_confirmed_rate_cv.shape[0]):

    current_prediction=standard_confirmed_cv[ii]

    if current_prediction==0 :

        current_prediction=0.1   

    this_preds=preds_confirmed_rate_cv[ii].tolist()

    name=name_cv[ii]

    reserve=this_preds[0]

    #overrides     

    if use_external and name in holdder_cumulative :

        print(" name %s is for confirmed rate" %(name))

        this_preds=revert_preds(this_preds)    

    

    if name in add_constant_rate:

        this_preds=add_rate(this_preds, add_constant_rate[name][0])

        

    for j in range (preds_confirmed_rate_cv.shape[1]):

        

                if use_external and j==0 and name in holdder_cumulative:

                    current_prediction=holdder_cumulative[name][0]

                    preds_confirmed_cumulative_rate_cv[ii][j]=current_prediction                                     

                else:

                    current_prediction*=max(1,this_preds[j])

                    preds_confirmed_cumulative_rate_cv[ii][j]=current_prediction





for j in range (preds_confirmed_rate_cv.shape[1]):



    this_features_cv=features_cv

                             

    preds=predict(features_cv,bagging, input_name=model_directory +"fatal"+ str(j))

    preds_fatalities_rate_cv[:,j]=preds

    print (" modelling fatalities, case %d, original cv %d and after %d "%( j,this_features_cv.shape[0],this_features_cv.shape[0])) 



    

predictions=[]

for ii in range (preds_fatalities_rate_cv.shape[0]):

    current_prediction=standard_fatalities_cv[ii]

        

    this_preds=preds_fatalities_rate_cv[ii].tolist()

    name=name_cv[ii]

    reserve=this_preds[0]

    #overrides

    if use_external and name in holdder_cumulative :

        print(" name %s is for confirmed rate" %(name))

        this_preds=revert_preds(this_preds)    

    

    if name in add_constant_rate:

        this_preds=add_rate(this_preds, add_constant_rate[name][1])       

  

    for j in range (preds_fatalities_rate_cv.shape[1]):

                if current_prediction==0 and  preds_confirmed_cumulative_rate_cv[ii][j]>400 :#(preds_confirmed_cumulative_rate_cv[ii][j]>400 or "Malta" in name or "Somalia" in name):

                    current_prediction=1.

                    

                if use_external and j==0 and name in holdder_cumulative:

                    current_prediction=holdder_cumulative[name][1]

                    preds_fatalities_cumulative_rate_cv[ii][j]=current_prediction      

                else:    

                    current_prediction*=max(1,this_preds[j])

                    preds_fatalities_cumulative_rate_cv[ii][j]=current_prediction


key_to_confirmed={}

key_to_fatality={}

key_to_confirmed_count={}

key_to_fatality_count={}



key_to_confirmed_cumulative_rate={}

key_to_fatality_cumulative_rate={}

key_to_confirmed_rate={}

key_to_fatality_rate={}



print(len(features_cv), len(name_cv),len(standard_confirmed_cv),len(standard_fatalities_cv)) 

print(preds_confirmed_cv.shape,

      preds_confirmed_standard_cv.shape,

      preds_fatalities_cv.shape,

      preds_fatalities_standard_cv.shape,

     

      preds_confirmed_rate_cv.shape,

      preds_confirmed_cumulative_rate_cv.shape,

      preds_fatalities_rate_cv.shape,

      preds_fatalities_cumulative_rate_cv.shape     

     

     ) 

for j in range (len(name_cv)):

    

    key_to_confirmed_count[name_cv[j]]=preds_confirmed_non_cumulative_cv[j,:].tolist()

    key_to_fatality_count[name_cv[j]]=preds_fatalities_non_cumulative_cv[j,:].tolist()

    key_to_confirmed[name_cv[j]]  =preds_confirmed_standard_cv[j,:].tolist()  

    key_to_fatality[name_cv[j]]=preds_fatalities_standard_cv[j,:].tolist()  

    

    key_to_confirmed_cumulative_rate[name_cv[j]]=preds_confirmed_cumulative_rate_cv[j,:].tolist()

    key_to_fatality_cumulative_rate[name_cv[j]]=preds_fatalities_cumulative_rate_cv[j,:].tolist()

    key_to_confirmed_rate[name_cv[j]] =preds_confirmed_rate_cv[j,:].tolist()  

    key_to_fatality_rate[name_cv[j]]=preds_fatalities_rate_cv[j,:].tolist()  
train_new=train[["Date","key","rate_ConfirmedCases","rate_Fatalities","ConfirmedCases","Fatalities","diff_ConfirmedCases","diff_Fatalities"]]



train_new["ConfirmedCasescumrate"]=train_new["ConfirmedCases"].values

train_new["Fatalitiescumrate"]=train_new["Fatalities"].values



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

 



all_days=int(test_new_count.shape[0]/len(unique_keys))



tr_horizon=all_days-horizon

print(all_days,tr_horizon, horizon )



fillin_columns(test_new_count,"key", 'ConfirmedCases', tr_horizon, horizon, len(unique_keys), key_to_confirmed)    

fillin_columns(test_new_count,"key", 'Fatalities', tr_horizon, horizon, len(unique_keys), key_to_fatality)   

fillin_columns(test_new_count,"key", 'diff_ConfirmedCases', tr_horizon, horizon, len(unique_keys), key_to_confirmed_count)   

fillin_columns(test_new_count,"key", 'diff_Fatalities', tr_horizon, horizon, len(unique_keys), key_to_fatality_count)   



fillin_columns(test_new_count,"key", 'ConfirmedCasescumrate', tr_horizon, horizon, len(unique_keys), key_to_confirmed_cumulative_rate)    

fillin_columns(test_new_count,"key", 'Fatalitiescumrate', tr_horizon, horizon, len(unique_keys), key_to_fatality_cumulative_rate)   

fillin_columns(test_new_count,"key", 'rate_ConfirmedCases', tr_horizon, horizon, len(unique_keys), key_to_confirmed_rate)   

fillin_columns(test_new_count,"key", 'rate_Fatalities', tr_horizon, horizon, len(unique_keys), key_to_fatality_rate)   





test_new_count
#####create count_difference from rate model



#get_difference(test_new_count, "key", "ConfirmedCasescumrate", new_target_name="diff_ConfirmedCasescumrate" )

#get_difference(test_new_count, "key", 'Fatalitiescumrate', new_target_name="diff_Fatalitiescumrate" )

######################## HERE COMM ##################

get_difference(test_new_count, "key", "ConfirmedCasescumrate", new_target_name="diff_ConfirmedCasescumrate" )#_special

get_difference(test_new_count, "key", 'Fatalitiescumrate', new_target_name="diff_Fatalitiescumrate" )#_special





#test_new_count['final_diff_ConfirmedCases']=test_new_count['diff_ConfirmedCases'].values *0.5 + test_new_count['diff_ConfirmedCasescumrate'].values *0.5

#test_new_count['final_diff_Fatalities']=test_new_count['diff_Fatalities'].values *0.5 + test_new_count['diff_Fatalitiescumrate'].values *0.5



#test_new_count['final_cumulative_ConfirmedCases']=test_new_count['ConfirmedCases'].values*0.5+test_new_count['ConfirmedCasescumrate'].values*0.5

#test_new_count['final_cumulative_Fatalities']=test_new_count['Fatalities'].values*0.5+test_new_count['Fatalitiescumrate'].values*0.5



test_new_count['final_diff_ConfirmedCases']=test_new_count['diff_ConfirmedCases'].values 

test_new_count['final_diff_Fatalities']=test_new_count['diff_Fatalities'].values 



test_new_count['final_cumulative_ConfirmedCases']=test_new_count['ConfirmedCases'].values

test_new_count['final_cumulative_Fatalities']=test_new_count['Fatalities'].values





#test_new_count['final_diff_ConfirmedCases']=test_new_count['diff_ConfirmedCasescumrate'].values

#test_new_count['final_diff_Fatalities']=test_new_count['diff_Fatalitiescumrate'].values



#test_new_count['final_cumulative_ConfirmedCases']=test_new_count['ConfirmedCasescumrate'].values

#test_new_count['final_cumulative_Fatalities']=test_new_count['Fatalitiescumrate'].values

###find all_countries that in the last day have count more than minimum

train_again=pd.read_csv(directory + "train.csv", parse_dates=["Date"] , engine="python")

train_again["key"]=train_again[["County","Province_State","Country_Region"]].apply(

    lambda row: str(row[0]) + "_" + str(row[1])+ "_" + str(row[2]),axis=1)



max_train_new=train_again["Date"].max()

print("last date in the training data is %s" %(max_train_new))

train_again_mini=train_again[train_again.Date == max_train_new].copy()

print (" shape of frame using only date {} is {}".format(max_train_new,train_again_mini.shape))

#flitered framed based on counts of confirmed and 



train_again_mini_filtered=train_again_mini[train_again_mini.Target=="ConfirmedCases"]

print (" shape of frame using only date {}  and confirmed cases is {}".format(max_train_new,train_again_mini_filtered.shape))

train_again_mini_filtered=train_again_mini_filtered[train_again_mini_filtered.TargetValue>minimum_count_for_rate_model]

print (" shape of frame using only date {}  and confirmed cases higher than {} is {}".format(

    max_train_new,minimum_count_for_rate_model,train_again_mini_filtered.shape))

unique_countrie_for_rate_model=train_again_mini_filtered["key"].unique()

################## HERE COM #################### 

unique_countrie_for_rate_model=[cn for  cn,vall in add_constant_rate.items()]

 

print (" there are {} unique countries that on the last training date {} had a count of more than {}".format(

    len(unique_countrie_for_rate_model),max_train_new, minimum_count_for_rate_model))

print(unique_countrie_for_rate_model)



test_new_count.loc[test_new_count.key.isin(unique_countrie_for_rate_model), 'final_diff_ConfirmedCases'] = test_new_count.loc[test_new_count.key.isin(unique_countrie_for_rate_model), 'diff_ConfirmedCasescumrate']

test_new_count.loc[test_new_count.key.isin(unique_countrie_for_rate_model), 'final_diff_Fatalities'] = test_new_count.loc[test_new_count.key.isin(unique_countrie_for_rate_model), 'diff_Fatalitiescumrate']



test_new_count.loc[test_new_count.key.isin(unique_countrie_for_rate_model), 'final_cumulative_ConfirmedCases'] = test_new_count.loc[test_new_count.key.isin(unique_countrie_for_rate_model), 'ConfirmedCasescumrate']

test_new_count.loc[test_new_count.key.isin(unique_countrie_for_rate_model), 'final_cumulative_Fatalities'] = test_new_count.loc[test_new_count.key.isin(unique_countrie_for_rate_model), 'Fatalitiescumrate']



#create time variable

get_time(test_new_count, "key", new_target_name="t") #get time

time=test_new_count["t"].values.tolist()

print (time[:40])

for jjj in range(len(time)):

    if time[jjj]<15:

        time[jjj]=1

    else :

        time[jjj]=time[jjj]-14

print (time[:40])

test_new_count["t"]=np.array(time)



get_x_day_min_max_avg(test_new_count,  "key", 'final_diff_ConfirmedCases',window=7, new_target_name="final_diff_ConfirmedCases_window")

get_x_day_min_max_avg(test_new_count,  "key", 'final_diff_Fatalities',window=7, new_target_name="final_diff_Fatalities_window")







test_new_count['final_diff_ConfirmedCases_05']=test_new_count['final_diff_ConfirmedCases_window_min'].values*(

    ((test_new_count['t'].values*10.85600625)*0.00001) +0.207181222)



test_new_count['final_diff_ConfirmedCases_95']=test_new_count['final_diff_ConfirmedCases_window_max'].values*(

    ((test_new_count['t'].values*0.832347133)*0.042631428) +1.325626509)







test_new_count['final_diff_Fatalities_05']=test_new_count['final_diff_Fatalities_window_min'].values*(

    ((test_new_count['t'].values*11.17489827)*0.00001) +0.271907924)



test_new_count['final_diff_Fatalities_95']=test_new_count['final_diff_Fatalities_window_max'].values*(

    ((test_new_count['t'].values*0.00001)*0.001111873) +1.771248526)



#test_new_count['final_diff_ConfirmedCases']=test_new_count['final_diff_ConfirmedCases'].values * 0.9

#test_new_count['final_diff_Fatalities']=test_new_count['final_diff_Fatalities'].values *0.9





test_new_count.loc[test_new_count['final_diff_ConfirmedCases']<0.5, "final_diff_ConfirmedCases"]=0.0

test_new_count.loc[test_new_count['final_diff_Fatalities']<0.5,'final_diff_Fatalities']=0.0



test_new_count.loc[test_new_count['final_diff_ConfirmedCases_05']<0.5, "final_diff_ConfirmedCases_05"]=0.0

test_new_count.loc[test_new_count['final_diff_Fatalities_05']<0.5,'final_diff_Fatalities_05']=0.0







test_new_count

test_new_count['final_diff_ConfirmedCases'].head(10)




taus=["0.05","0.5","0.95"]

taus+=taus



_preds=['final_diff_ConfirmedCases_05','final_diff_ConfirmedCases','final_diff_ConfirmedCases_95'] +['final_diff_Fatalities_05','final_diff_Fatalities','final_diff_Fatalities_95']

assert(len(taus))==6==len(_preds)



submission=[]



for j in range(len(taus)):

    ids=""

    if "Fatalities" in _preds[j]:

         ids="Id_Fatalities"

    elif"ConfirmedCases" in _preds[j]:

         ids="Id_ConfirmedCases"

    else :

        raise Exception(" name not identified in title")

    print (" adding %s with pred name %s for tau %s"%(ids,_preds[j],taus[j]))

        

    mini_frame=test_new_count[[ids,_preds[j]]]

    mini_frame["id"]=mini_frame[ids].values

    mini_frame[ids]=mini_frame[ids].apply(lambda x: str(x) +"_" + taus[j])

    mini_frame=mini_frame.reset_index(drop=True)

    mini_frame.columns=["ForecastId_Quantile","TargetValue","id"]

    submission.append(mini_frame)

    

submission=pd.concat(submission)

submission = submission.sort_values(["id","ForecastId_Quantile"], ascending = (True, True))



submission.drop("id", inplace=True, axis=1)



submission.to_csv( "submission.csv", index=False)



submission