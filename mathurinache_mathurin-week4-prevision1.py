#Kazanova

#!/usr/bin/env python

# coding: utf-8



# In[1]:





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





# In[2]:





import math

import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error



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



    for i in range(1, len(group_keys) - 1):

        previous_group = group_keys[i - 1]

        current_group = group_keys[i]



        previous_value = target[i - 1]

        current_value = target[i]

         

        if current_group == previous_group:

                if previous_value!=0.0:

                     rate[i]=current_value/previous_value



                 

        rate[i] =max(1,rate[i] )#correct negative values



    frame[new_target_name] = np.array(rate)

    

def get_data_by_key(dataframe, key, key_value, fields=None):

    mini_frame=dataframe[dataframe[key]==key_value]

    if not fields is None:                

        mini_frame=mini_frame[fields].values

        

    return mini_frame



directory="/kaggle/input/covid19-global-forecasting-week-4/"

model_directory="/kaggle/input/model-dir/model"



train=pd.read_csv(directory + "train.csv", parse_dates=["Date"] , engine="python")

test=pd.read_csv(directory + "test.csv", parse_dates=["Date"], engine="python")



train["key"]=train[["Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)

test["key"]=test[["Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)



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





# In[3]:





fix_target(train, key, target1, new_target_name=target1)

#fix_target(train, key, target2, new_target_name=target2)



rate(train, key, target1, new_target_name="rate_" +target1 )

rate(train, key, target2, new_target_name="rate_" +target2 )

unique_keys=train[key].unique()

print(len(unique_keys))





train





# In[4]:





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





def dereive_features(frame, confirmed, fatalities, rate_confirmed, rate_fatalities, 

                     horizon ,size=20, windows=[3,7], days_back_confimed=[1,10,100], days_back_fatalities=[1,2,10]):

    targets=[]

    

    names=["lag_confirmed_rate" + str(k+1) for k in range (size)]

    for day in days_back_confimed:

        names+=["days_ago_confirmed_count_" + str(day) ]

    for window in windows:        

        names+=["ma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]

        names+=["std" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)] 

        names+=["ewma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]         

        

        

    names+=["lag_fatalities_rate" + str(k+1) for k in range (size)]

    for day in days_back_fatalities:

        names+=["days_ago_fatalitiescount_" + str(day) ]    

    for window in windows:        

        names+=["ma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]

        names+=["std" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]  

        names+=["ewma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]        

    names+=["confirmed_level"]

    names+=["fatalities_level"]    

    

    names+=["confirmed_plus" + str(k+1) for k in range (horizon)]    

    names+=["fatalities_plus" + str(k+1) for k in range (horizon)]  

    

    #names+=["current_confirmed"]

    #names+=["current_fatalities"]    

    

    features=[]

    for i in range (len(confirmed)):

        row_features=[]

        #####################lag_confirmed_rate       

        lag_confirmed_rate=get_lags(rate_confirmed, i, size=size)

        row_features+=lag_confirmed_rate

        #####################days_ago_confirmed_count_10

        for day in days_back_confimed:

            days_ago_confirmed_count_10=days_ago_thresold_hit(confirmed, i, day)               

            row_features+=[days_ago_confirmed_count_10] 

        #####################ma_rate_confirmed       

        #####################std_rate_confirmed 

        for window in windows:

            ma3_rate_confirmed,std3_rate_confirmed= generate_ma_std_window(rate_confirmed, i, size=size, window=window)

            row_features+= ma3_rate_confirmed   

            row_features+= std3_rate_confirmed          

            ewma3_rate_confirmed=generate_ewma_window(rate_confirmed, i, size=size, window=window, alpha=0.05)

            row_features+= ewma3_rate_confirmed              

        #####################lag_fatalities_rate   

        lag_fatalities_rate=get_lags(rate_fatalities, i, size=size)

        row_features+=lag_fatalities_rate

        #####################days_ago_confirmed_count_10

        for day in days_back_fatalities:

            days_ago_fatalitiescount_2=days_ago_thresold_hit(fatalities, i, day)               

            row_features+=[days_ago_fatalitiescount_2]     

        #####################ma_rate_fatalities       

        #####################std_rate_fatalities 

        for window in windows:        

            ma3_rate_fatalities,std3_rate_fatalities= generate_ma_std_window(rate_fatalities, i, size=size, window=window)

            row_features+= ma3_rate_fatalities   

            row_features+= std3_rate_fatalities  

            ewma3_rate_fatalities=generate_ewma_window(rate_fatalities, i, size=size, window=window, alpha=0.05)

            row_features+= ewma3_rate_fatalities                  

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

            

        #######################confirmed_plus target

        confirmed_plus=get_target(rate_confirmed, i, horizon=horizon)

        row_features+= confirmed_plus          

        #######################fatalities_plus target

        fatalities_plus=get_target(rate_fatalities, i, horizon=horizon)

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

                                       days_back_confimed=[1,10,100], days_back_fatalities=[1,2,10]):

    mini_frame=get_data_by_key(frame, group, key, fields=None)

    

    mini_frame_with_features=dereive_features(mini_frame, mini_frame["ConfirmedCases"].values,

                                              mini_frame["Fatalities"].values, mini_frame["rate_ConfirmedCases"].values, 

                                               mini_frame["rate_Fatalities"].values, horizon ,size=size, windows=windows,

                                              days_back_confimed=days_back_confimed, days_back_fatalities=days_back_fatalities)

    #print (mini_frame_with_features.shape[0])

    return mini_frame_with_features





# In[5]:





from tqdm import tqdm

train_frame=[]

size=20

windows=[3,5,7]

days_back_confimed=[1,10,100]

days_back_fatalities=[1,2,10]

#print (len(train['key'].unique()))

for unique_k in tqdm(unique_keys):

    mini_frame=feature_engineering_for_single_key(train, key, unique_k, horizon=horizon, size=size, 

                                                  windows=windows, days_back_confimed=days_back_confimed,

                                                  days_back_fatalities=days_back_fatalities).reset_index(drop=True) 

    #print (mini_frame.shape[0])

    train_frame.append(mini_frame)

    

train_frame = pd.concat(train_frame, axis=0).reset_index(drop=True)

#train_frame.to_csv(directory +"all" + ".csv", index=False)

new_unique_keys=train_frame['key'].unique()

for kee in new_unique_keys:

    if kee not in unique_keys:

        print (kee , " is not there ")





# In[6]:





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





# In[7]:





names=["lag_confirmed_rate" + str(k+1) for k in range (size)]

for day in days_back_confimed:

    names+=["days_ago_confirmed_count_" + str(day) ]

for window in windows:        

    names+=["ma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]

    names+=["std" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)] 

    names+=["ewma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]         





names+=["lag_fatalities_rate" + str(k+1) for k in range (size)]

for day in days_back_fatalities:

    names+=["days_ago_fatalitiescount_" + str(day) ]    

for window in windows:        

    names+=["ma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]

    names+=["std" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]  

    names+=["ewma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]        

names+=["confirmed_level"]

names+=["fatalities_level"]      





# In[8]:





#### scoring 

def decay_4_first_10_then_1_f(array):

    arr=[1.0 for k in range(len(array))]

    for j in range(len(array)):

        if j<10:

            arr[j]=1. + (max(1,array[j])-1.)/4.

        else :

            arr[j]=1.

    return arr

            

def decay_2_f(array):

    arr=[1.0 for k in range(len(array))]    

    for j in range(len(array)):

            arr[j]=1. + (max(1,array[j])-1.)/2.

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



def linear_last_12_f(array):

    arr=[1.0 for k in range(len(array))]

    for j in range(len(array)):

        arr[j]=max(1,array[j])

    arr12= (max(1,arr[-12])-1.)/12. 

    

    for j in range(0, 12):

        arr[len(arr)-12 +j]= max(1, 1 + ( (arr12*12) - (j+1)*arr12 ))

    return arr

    

decay_4_first_10_then_1 =["Beijing_China","Fujian_China","Guangdong_China", "Hong Kong_China",

"Inner Mongolia_China","Jiangsu_China","Liaoning_China","Macau_China","Shandong_China","Tianjin_China",

"Yunnan_China","Zhejiang_China","Northern Territory_Australia",

"nan_Belize","nan_Benin","nan_Bhutan","nan_Seychelles","nan_Cabo Verde"]



decay_2 =["Shanghai_China" , "nan_Afghanistan","nan_Andorra","Australian Capital Territory_Australia",

"South Australia_Australia","Tasmania_Australia","nan_Bahrain","nan_Belarus"

"nan_Belgium","nan_Bolivia","Manitoba_Canada","New Brunswick_Canada","Newfoundland and Labrador_Canada",

"Saskatchewan_Canada","nan_Central African Republic","nan_Congo (Kinshasa)","nan_Cote d'Ivoire","Mayotte_France","nan_Ukraine"]



stay_same=["China", "nan_Antigua and Barbuda","nan_Diamond Princess","nan_Saint Vincent and the Grenadines",

           "nan_Timor-Leste","Montserrat_United Kingdom"]



decay_2_last_12_linear_inter =["nan_Angola" , "nan_Barbados" ,"Prince Edward Island_Canada","nan_Chad",

"nan_Congo (Brazzaville)","Greenland_Denmark","nan_Djibouti","nan_Dominica","nan_El Salvador",

"nan_Eritrea","nan_Eswatini","nan_Fiji","French Guiana_France","French Polynesia_France","New Caledonia_France",

"Saint Barthelemy_France","St Martin_France","nan_Gabon","nan_Gambia","nan_Grenada","nan_Guinea-Bissau",

"nan_Guyana","nan_Haiti","nan_Holy See","nan_Kyrgyzstan","nan_Laos","nan_Libya","nan_Madagascar",

"nan_Maldives","nan_Mali","nan_Mauritania","nan_Mauritius","nan_Mozambique","nan_Nepal",

"Aruba_Netherlands","Curacao_Netherlands","Sint Maarten_Netherlands","nan_Nicaragua","nan_Niger","nan_Papua New Guinea",

"nan_Saint Kitts and Nevis","nan_Saint Lucia","nan_Somalia","nan_Sudan","nan_Suriname","nan_Syria","nan_Tanzania",

"nan_Togo","Virgin Islands_US","Bermuda_United Kingdom","Cayman Islands_United Kingdom","Channel Islands_United Kingdom",

"Gibraltar_United Kingdom","Isle of Man_United Kingdom","nan_Zimbabwe","nan_Bahamas","nan_Zambia"]



acceleratorx2=["nan_Kenya","nan_Moldova"]



decay_1_5 =["nan_Kazakhstan","nan_Tunisia", "Alabama_US", "Alaska_US",

	"Arizona_US","Colorado_US","Florida_US","Montana_US","Nebraska_US","Nevada_US","New Hampshire_US","New Mexico_US",

	"Puerto Rico_US","nan_Uzbekistan","nan_Azerbaijan","nan_Bangladesh","nan_Bosnia and Herzegovina",

	"nan_Cameroon","nan_Cuba","nan_Guatemala","nan_Jamaica","nan_Morocco","nan_New Zealand","nan_Philippines","nan_Romania",

	"nan_Trinidad and Tobago"]



linear_last_12=["nan_Uganda","nan_Equatorial Guinea","nan_Guinea","nan_Honduras","nan_Liberia","nan_Mongolia","nan_Namibia"]



stay_same=[ "nan_Antigua and Barbuda","nan_Diamond Princess","nan_Saint Vincent and the Grenadines","nan_Timor-Leste","Montserrat_United Kingdom"]



#"China",



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

print("preds_confirmed_cv.shape[1]",preds_confirmed_cv.shape[1])

for j in range (27):



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

    #overrides

    if name in decay_4_first_10_then_1:

        this_preds=decay_4_first_10_then_1_f(this_preds)

        

    elif name in decay_2:

        this_preds=decay_2_f(this_preds)

        

    elif name in decay_2_last_12_linear_inter:

        this_preds=decay_2_last_12_linear_inter_f(this_preds)

        

    elif name in decay_1_5:

        this_preds=decay_1_5_f(this_preds)        

        

    elif name in linear_last_12:

        this_preds=linear_last_12_f(this_preds)

        

    elif name in acceleratorx2:

        this_preds=acceleratorx2_f(this_preds)         



        

    elif name in stay_same or  "China" in name:

        this_preds=stay_same_f(this_preds)      



    for j in range (preds_confirmed_cv.shape[1]):

                current_prediction*=max(1,this_preds[j])

                preds_confirmed_standard_cv[ii][j]=current_prediction





for j in range (27):



    this_features_cv=features_cv

                             

    preds=predict(features_cv, input_name=model_directory +"fatal"+ str(j))

    preds_fatalities_cv[:,j]=preds

    print (" modelling fatalities, case %d, original cv %d and after %d "%( j,this_features_cv.shape[0],this_features_cv.shape[0])) 



predictions=[]

for ii in range (preds_fatalities_cv.shape[0]):

    current_prediction=standard_fatalities_cv[ii]

    if current_prediction==0 and standard_confirmed_cv[ii]>400:

        current_prediction=0.1

        

    this_preds=preds_fatalities_cv[ii].tolist()

    name=name_cv[ii]

    #overrides

    if name in decay_4_first_10_then_1:

        this_preds=decay_4_first_10_then_1_f(this_preds)

        

    elif name in decay_2:

        this_preds=decay_2_f(this_preds)

        

    elif name in decay_2_last_12_linear_inter:

        this_preds=decay_2_last_12_linear_inter_f(this_preds)

        

    elif name in decay_1_5:

        this_preds=decay_1_5_f(this_preds)        

        

    elif name in linear_last_12:

        this_preds=linear_last_12_f(this_preds) 

        

    elif name in acceleratorx2:

        this_preds=acceleratorx2_f(this_preds)                 

        

    elif name in stay_same or  "China" in name:

        this_preds=stay_same_f(this_preds)         

        

    for j in range (preds_fatalities_cv.shape[1]):

                if current_prediction==0 and  preds_confirmed_standard_cv[ii][j]>400:

                    current_prediction=1.

                current_prediction*=max(1,this_preds[j])

                preds_fatalities_standard_cv[ii][j]=current_prediction























# In[9]:





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

    





# In[10]:





train_new=train[["Date","ConfirmedCases","Fatalities","key","rate_ConfirmedCases","rate_Fatalities"]]



test_new=pd.merge(test,train_new, how="left", left_on=["key","Date"], right_on=["key","Date"] ).reset_index(drop=True)

test_new





# In[11]:





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



submission.to_csv( "sub1.csv", index=False)

sub1=submission.copy()

#CPMP

#!/usr/bin/env python

# coding: utf-8



# In[1]:





import pandas as pd

import numpy as np

from scipy.optimize import curve_fit



get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.metrics import log_loss

from sklearn.preprocessing import OneHotEncoder



import xgboost as xgb



from tensorflow.keras.optimizers import Nadam

from sklearn.metrics import mean_squared_error

import tensorflow as tf

import tensorflow.keras.layers as KL

from datetime import timedelta

import numpy as np

import pandas as pd





import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge



import datetime

import gc

from tqdm import tqdm





# In[2]:





def get_cpmp_sub(save_oof=False, save_public_test=False):

    train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

    train['Province_State'].fillna('', inplace=True)

    train['Date'] = pd.to_datetime(train['Date'])

    train['day'] = train.Date.dt.dayofyear

    #train = train[train.day <= 85]

    train['geo'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]

    train



    test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

    test['Province_State'].fillna('', inplace=True)

    test['Date'] = pd.to_datetime(test['Date'])

    test['day'] = test.Date.dt.dayofyear

    test['geo'] = ['_'.join(x) for x in zip(test['Country_Region'], test['Province_State'])]

    test



    day_min = train['day'].min()

    train['day'] -= day_min

    test['day'] -= day_min



    min_test_val_day = test.day.min()

    max_test_val_day = train.day.max()

    max_test_day = test.day.max()

    num_days = max_test_day + 1



    min_test_val_day, max_test_val_day, num_days



    train['ForecastId'] = -1

    test['Id'] = -1

    test['ConfirmedCases'] = 0

    test['Fatalities'] = 0



    debug = False



    data = pd.concat([train,

                      test[test.day > max_test_val_day][train.columns]

                     ]).reset_index(drop=True)

    if debug:

        data = data[data['geo'] >= 'France_'].reset_index(drop=True)

    #del train, test

    gc.collect()



    dates = data[data['geo'] == 'France_'].Date.values



    if 0:

        gr = data.groupby('geo')

        data['ConfirmedCases'] = gr.ConfirmedCases.transform('cummax')

        data['Fatalities'] = gr.Fatalities.transform('cummax')



    geo_data = data.pivot(index='geo', columns='day', values='ForecastId')

    num_geo = geo_data.shape[0]

    geo_data



    geo_id = {}

    for i,g in enumerate(geo_data.index):

        geo_id[g] = i





    ConfirmedCases = data.pivot(index='geo', columns='day', values='ConfirmedCases')

    Fatalities = data.pivot(index='geo', columns='day', values='Fatalities')



    if debug:

        cases = ConfirmedCases.values

        deaths = Fatalities.values

    else:

        cases = np.log1p(ConfirmedCases.values)

        deaths = np.log1p(Fatalities.values)





    def get_dataset(start_pred, num_train, lag_period):

        days = np.arange( start_pred - num_train + 1, start_pred + 1)

        lag_cases = np.vstack([cases[:, d - lag_period : d] for d in days])

        lag_deaths = np.vstack([deaths[:, d - lag_period : d] for d in days])

        target_cases = np.vstack([cases[:, d : d + 1] for d in days])

        target_deaths = np.vstack([deaths[:, d : d + 1] for d in days])

        geo_ids = np.vstack([geo_ids_base for d in days])

        country_ids = np.vstack([country_ids_base for d in days])

        return lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days



    def update_valid_dataset(data, pred_death, pred_case):

        lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days = data

        day = days[-1] + 1

        new_lag_cases = np.hstack([lag_cases[:, 1:], pred_case])

        new_lag_deaths = np.hstack([lag_deaths[:, 1:], pred_death]) 

        new_target_cases = cases[:, day:day+1]

        new_target_deaths = deaths[:, day:day+1] 

        new_geo_ids = geo_ids  

        new_country_ids = country_ids  

        new_days = 1 + days

        return new_lag_cases, new_lag_deaths, new_target_cases, new_target_deaths, new_geo_ids, new_country_ids, new_days



    def fit_eval(lr_death, lr_case, data, start_lag_death, end_lag_death, num_lag_case, fit, score):

        lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days = data



        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], country_ids])

        X_death = np.hstack([lag_deaths[:, -num_lag_case:], country_ids])

        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], lag_deaths[:, -num_lag_case:], country_ids])

        y_death = target_deaths

        y_death_prev = lag_deaths[:, -1:]

        if fit:

            if 0:

                keep = (y_death > 0).ravel()

                X_death = X_death[keep]

                y_death = y_death[keep]

                y_death_prev = y_death_prev[keep]

            lr_death.fit(X_death, y_death)

        y_pred_death = lr_death.predict(X_death)

        y_pred_death = np.maximum(y_pred_death, y_death_prev)



        X_case = np.hstack([lag_cases[:, -num_lag_case:], geo_ids])

        X_case = lag_cases[:, -num_lag_case:]

        y_case = target_cases

        y_case_prev = lag_cases[:, -1:]

        if fit:

            lr_case.fit(X_case, y_case)

        y_pred_case = lr_case.predict(X_case)

        y_pred_case = np.maximum(y_pred_case, y_case_prev)



        if score:

            death_score = val_score(y_death, y_pred_death)

            case_score = val_score(y_case, y_pred_case)

        else:

            death_score = 0

            case_score = 0



        return death_score, case_score, y_pred_death, y_pred_case



    def train_model(train, valid, start_lag_death, end_lag_death, num_lag_case, num_val, score=True):

        alpha = 2#3

        lr_death = Ridge(alpha=alpha, fit_intercept=False)

        lr_case = Ridge(alpha=alpha, fit_intercept=True)



        (train_death_score, train_case_score, train_pred_death, train_pred_case,

        ) = fit_eval(lr_death, lr_case, train, start_lag_death, end_lag_death, num_lag_case, fit=True, score=score)



        death_scores = []

        case_scores = []



        death_pred = []

        case_pred = []



        for i in range(num_val):



            (valid_death_score, valid_case_score, valid_pred_death, valid_pred_case,

            ) = fit_eval(lr_death, lr_case, valid, start_lag_death, end_lag_death, num_lag_case, fit=False, score=score)



            death_scores.append(valid_death_score)

            case_scores.append(valid_case_score)

            death_pred.append(valid_pred_death)

            case_pred.append(valid_pred_case)



            if 0:

                print('val death: %0.3f' %  valid_death_score,

                      'val case: %0.3f' %  valid_case_score,

                      'val : %0.3f' %  np.mean([valid_death_score, valid_case_score]),

                      flush=True)

            valid = update_valid_dataset(valid, valid_pred_death, valid_pred_case)



        if score:

            death_scores = np.sqrt(np.mean([s**2 for s in death_scores]))

            case_scores = np.sqrt(np.mean([s**2 for s in case_scores]))

            if 0:

                print('train death: %0.3f' %  train_death_score,

                      'train case: %0.3f' %  train_case_score,

                      'val death: %0.3f' %  death_scores,

                      'val case: %0.3f' %  case_scores,

                      'val : %0.3f' % ( (death_scores + case_scores) / 2),

                      flush=True)

            else:

                print('%0.4f' %  case_scores,

                      ', %0.4f' %  death_scores,

                      '= %0.4f' % ( (death_scores + case_scores) / 2),

                      flush=True)

        death_pred = np.hstack(death_pred)

        case_pred = np.hstack(case_pred)

        return death_scores, case_scores, death_pred, case_pred



    countries = [g.split('_')[0] for g in geo_data.index]

    countries = pd.factorize(countries)[0]



    country_ids_base = countries.reshape((-1, 1))

    ohe = OneHotEncoder(sparse=False)

    country_ids_base = 0.2 * ohe.fit_transform(country_ids_base)

    country_ids_base.shape



    geo_ids_base = np.arange(num_geo).reshape((-1, 1))

    ohe = OneHotEncoder(sparse=False)

    geo_ids_base = 0.1 * ohe.fit_transform(geo_ids_base)

    geo_ids_base.shape



    def val_score(true, pred):

        pred = np.log1p(np.round(np.expm1(pred) - 0.2))

        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))



    def val_score(true, pred):

        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))







    start_lag_death, end_lag_death = 14, 6,

    num_train = 10#5

    num_lag_case = 28#14

    lag_period = max(start_lag_death, num_lag_case)



    def get_oof(start_val_delta=0):   

        start_val = min_test_val_day + start_val_delta

        last_train = start_val - 1

        num_val = max_test_val_day - start_val + 1

        print(dates[start_val], start_val, num_val)

        train_data = get_dataset(last_train, num_train, lag_period)

        valid_data = get_dataset(start_val, 1, lag_period)

        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 

                                                            start_lag_death, end_lag_death, num_lag_case, num_val)



        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()

        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)

        pred_deaths = pred_deaths.stack().reset_index()

        pred_deaths.columns = ['geo', 'day', 'Fatalities']

        pred_deaths



        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()

        pred_cases.iloc[:, :] = np.expm1(val_case_preds)

        pred_cases = pred_cases.stack().reset_index()

        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']

        pred_cases



        sub = train[['Date', 'Id', 'geo', 'day']]

        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])

        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])

        #sub = sub.fillna(0)

        sub = sub[sub.day >= start_val]

        sub = sub[['Id', 'ConfirmedCases', 'Fatalities']].copy()

        return sub





    if save_oof:

        for start_val_delta, date in zip(range(3, -8, -3),

                                  ['2020-03-22', '2020-03-19', '2020-03-16', '2020-03-13']):

            print(date, end=' ')

            oof = get_oof(start_val_delta)

            oof.to_csv('../submissions/cpmp-%s.csv' % date, index=None)



    def get_sub(start_val_delta=0):   

        start_val = min_test_val_day + start_val_delta

        last_train = start_val - 1

        num_val = max_test_val_day - start_val + 1

        print(dates[last_train], start_val, num_val)

        num_lag_case = 28#14

        train_data = get_dataset(last_train, num_train, lag_period)

        valid_data = get_dataset(start_val, 1, lag_period)

        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 

                                                            start_lag_death, end_lag_death, num_lag_case, num_val)



        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()

        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)

        pred_deaths = pred_deaths.stack().reset_index()

        pred_deaths.columns = ['geo', 'day', 'Fatalities']

        pred_deaths



        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()

        pred_cases.iloc[:, :] = np.expm1(val_case_preds)

        pred_cases = pred_cases.stack().reset_index()

        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']

        pred_cases



        sub = test[['Date', 'ForecastId', 'geo', 'day']]

        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])

        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])

        sub = sub.fillna(0)

        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]

        return sub

        return sub





    known_test = train[['geo', 'day', 'ConfirmedCases', 'Fatalities']

              ].merge(test[['geo', 'day', 'ForecastId']], how='left', on=['geo', 'day'])

    known_test = known_test[['ForecastId', 'ConfirmedCases', 'Fatalities']][known_test.ForecastId.notnull()].copy()

    known_test



    unknow_test = test[test.day > max_test_val_day]

    unknow_test



    def get_final_sub():   

        start_val = max_test_val_day + 1

        last_train = start_val - 1

        num_val = max_test_day - start_val + 1

        print(dates[last_train], start_val, num_val)

        num_lag_case = num_val + 3

        train_data = get_dataset(last_train, num_train, lag_period)

        valid_data = get_dataset(start_val, 1, lag_period)

        (_, _, val_death_preds, val_case_preds

        ) = train_model(train_data, valid_data, start_lag_death, end_lag_death, num_lag_case, num_val, score=False)



        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()

        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)

        pred_deaths = pred_deaths.stack().reset_index()

        pred_deaths.columns = ['geo', 'day', 'Fatalities']

        pred_deaths



        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()

        pred_cases.iloc[:, :] = np.expm1(val_case_preds)

        pred_cases = pred_cases.stack().reset_index()

        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']

        pred_cases

        print(unknow_test.shape, pred_deaths.shape, pred_cases.shape)



        sub = unknow_test[['Date', 'ForecastId', 'geo', 'day']]

        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])

        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])

        #sub = sub.fillna(0)

        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]

        sub = pd.concat([known_test, sub])

        return sub



    if save_public_test:

        sub = get_sub()

    else:

        sub = get_final_sub()

    return sub





# In[3]:





def get_nn_sub():

    df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

    sub_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")



    coo_df = pd.read_csv("../input/covidweek1/train.csv").rename(columns={"Country/Region": "Country_Region"})

    coo_df = coo_df.groupby("Country_Region")[["Lat", "Long"]].mean().reset_index()

    coo_df = coo_df[coo_df["Country_Region"].notnull()]



    loc_group = ["Province_State", "Country_Region"]





    def preprocess(df):

        df["Date"] = df["Date"].astype("datetime64[ms]")

        df["days"] = (df["Date"] - pd.to_datetime("2020-01-01")).dt.days

        df["weekend"] = df["Date"].dt.dayofweek//5



        df = df.merge(coo_df, how="left", on="Country_Region")

        df["Lat"] = (df["Lat"] // 30).astype(np.float32).fillna(0)

        df["Long"] = (df["Long"] // 60).astype(np.float32).fillna(0)



        for col in loc_group:

            df[col].fillna("none", inplace=True)

        return df



    df = preprocess(df)

    sub_df = preprocess(sub_df)



    print(df.shape)



    TARGETS = ["ConfirmedCases", "Fatalities"]



    for col in TARGETS:

        df[col] = np.log1p(df[col])



    NUM_SHIFT = 10#5



    features = ["Lat", "Long"]



    for s in range(1, NUM_SHIFT+1):

        for col in TARGETS:

            df["prev_{}_{}".format(col, s)] = df.groupby(loc_group)[col].shift(s)

            features.append("prev_{}_{}".format(col, s))



    df = df[df["Date"] >= df["Date"].min() + timedelta(days=NUM_SHIFT)].copy()



    TEST_FIRST = sub_df["Date"].min() # pd.to_datetime("2020-03-13") #

    TEST_DAYS = (df["Date"].max() - TEST_FIRST).days + 1



    dev_df, test_df = df[df["Date"] < TEST_FIRST].copy(), df[df["Date"] >= TEST_FIRST].copy()



    def nn_block(input_layer, size, dropout_rate, activation):

        out_layer = KL.Dense(size, activation=None)(input_layer)

        #out_layer = KL.BatchNormalization()(out_layer)

        out_layer = KL.Activation(activation)(out_layer)

        out_layer = KL.Dropout(dropout_rate)(out_layer)

        return out_layer





    def get_model():

        inp = KL.Input(shape=(len(features),))



        hidden_layer = nn_block(inp, 128, 0.0, "relu")

        gate_layer = nn_block(hidden_layer, 64, 0.0, "sigmoid")

        hidden_layer = nn_block(hidden_layer, 64, 0.0, "relu")

        hidden_layer = KL.multiply([hidden_layer, gate_layer])



        out = KL.Dense(len(TARGETS), activation="linear")(hidden_layer)



        model = tf.keras.models.Model(inputs=[inp], outputs=out)

        return model



    get_model().summary()



    def get_input(df):

        return [df[features]]



    NUM_MODELS = 10





    def train_models(df, save=False):

        models = []

        for i in range(NUM_MODELS):

            model = get_model()

            model.compile(loss="mean_squared_error", optimizer=Nadam(lr=1e-4))

            hist = model.fit(get_input(df), df[TARGETS],

                             batch_size=2048, epochs=500, verbose=0, shuffle=True)

            if save:

                model.save_weights("model{}.h5".format(i))

            models.append(model)

        return models



    models = train_models(dev_df)





    prev_targets = ['prev_ConfirmedCases_1', 'prev_Fatalities_1']



    def predict_one(df, models):

        pred = np.zeros((df.shape[0], 2))

        for model in models:

            pred += model.predict(get_input(df))/len(models)

        pred = np.maximum(pred, df[prev_targets].values)

        pred[:, 0] = np.log1p(np.expm1(pred[:, 0]) + 0.1)

        pred[:, 1] = np.log1p(np.expm1(pred[:, 1]) + 0.01)

        return np.clip(pred, None, 15)



    print([mean_squared_error(dev_df[TARGETS[i]], predict_one(dev_df, models)[:, i]) for i in range(len(TARGETS))])





    def rmse(y_true, y_pred):

        return np.sqrt(mean_squared_error(y_true, y_pred))



    def evaluate(df):

        error = 0

        for col in TARGETS:

            error += rmse(df[col].values, df["pred_{}".format(col)].values)

        return np.round(error/len(TARGETS), 5)





    def predict(test_df, first_day, num_days, models, val=False):

        temp_df = test_df.loc[test_df["Date"] == first_day].copy()

        y_pred = predict_one(temp_df, models)



        for i, col in enumerate(TARGETS):

            test_df["pred_{}".format(col)] = 0

            test_df.loc[test_df["Date"] == first_day, "pred_{}".format(col)] = y_pred[:, i]



        print(first_day, np.isnan(y_pred).sum(), y_pred.min(), y_pred.max())

        if val:

            print(evaluate(test_df[test_df["Date"] == first_day]))





        y_prevs = [None]*NUM_SHIFT



        for i in range(1, NUM_SHIFT):

            y_prevs[i] = temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]].values



        for d in range(1, num_days):

            date = first_day + timedelta(days=d)

            print(date, np.isnan(y_pred).sum(), y_pred.min(), y_pred.max())



            temp_df = test_df.loc[test_df["Date"] == date].copy()

            temp_df[prev_targets] = y_pred

            for i in range(2, NUM_SHIFT+1):

                temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]] = y_prevs[i-1]



            y_pred, y_prevs = predict_one(temp_df, models), [None, y_pred] + y_prevs[1:-1]





            for i, col in enumerate(TARGETS):

                test_df.loc[test_df["Date"] == date, "pred_{}".format(col)] = y_pred[:, i]



            if val:

                print(evaluate(test_df[test_df["Date"] == date]))



        return test_df



    test_df = predict(test_df, TEST_FIRST, TEST_DAYS, models, val=True)

    print(evaluate(test_df))



    for col in TARGETS:

        test_df[col] = np.expm1(test_df[col])

        test_df["pred_{}".format(col)] = np.expm1(test_df["pred_{}".format(col)])



    models = train_models(df, save=True)



    sub_df_public = sub_df[sub_df["Date"] <= df["Date"].max()].copy()

    sub_df_private = sub_df[sub_df["Date"] > df["Date"].max()].copy()



    pred_cols = ["pred_{}".format(col) for col in TARGETS]

    #sub_df_public = sub_df_public.merge(test_df[["Date"] + loc_group + pred_cols].rename(columns={col: col[5:] for col in pred_cols}), 

    #                                    how="left", on=["Date"] + loc_group)

    sub_df_public = sub_df_public.merge(test_df[["Date"] + loc_group + TARGETS], how="left", on=["Date"] + loc_group)



    SUB_FIRST = sub_df_private["Date"].min()

    SUB_DAYS = (sub_df_private["Date"].max() - sub_df_private["Date"].min()).days + 1



    sub_df_private = df.append(sub_df_private, sort=False)



    for s in range(1, NUM_SHIFT+1):

        for col in TARGETS:

            sub_df_private["prev_{}_{}".format(col, s)] = sub_df_private.groupby(loc_group)[col].shift(s)



    sub_df_private = sub_df_private[sub_df_private["Date"] >= SUB_FIRST].copy()



    sub_df_private = predict(sub_df_private, SUB_FIRST, SUB_DAYS, models)



    for col in TARGETS:

        sub_df_private[col] = np.expm1(sub_df_private["pred_{}".format(col)])



    sub_df = sub_df_public.append(sub_df_private, sort=False)

    sub_df["ForecastId"] = sub_df["ForecastId"].astype(np.int16)



    return sub_df[["ForecastId"] + TARGETS]





# In[4]:





sub1 = get_cpmp_sub()

sub1['ForecastId'] = sub1['ForecastId'].astype('int')





# In[5]:





sub2 = get_nn_sub()





# In[6]:





sub1.sort_values("ForecastId", inplace=True)

sub2.sort_values("ForecastId", inplace=True)





# In[7]:





sub1.to_csv("sub1.csv",index=False)

sub2.to_csv("sub2.csv",index=False)





# In[8]:





from sklearn.metrics import mean_squared_error



TARGETS = ["ConfirmedCases", "Fatalities"]



[np.sqrt(mean_squared_error(np.log1p(sub1[t].values), np.log1p(sub2[t].values))) for t in TARGETS]





# In[9]:





sub_df = sub1.copy()

for t in TARGETS:

    sub_df[t] = np.expm1(np.log1p(sub1[t].values)*0.5 + np.log1p(sub2[t].values)*0.5)

    

sub_df.to_csv("sub2.csv", index=False)

sub2=sub_df.copy()
#ROSS

#!/usr/bin/env python

# coding: utf-8



# In[1]:





import numpy as np

import pandas as pd

import os, gc

import lightgbm as lgb

import xgboost as xgb

import catboost as ctb

import matplotlib.pyplot as plt

import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import date, datetime, timedelta

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from scipy.optimize import nnls

pd.set_option('display.max_columns', 10)

pd.set_option('display.max_rows', 10)

np.set_printoptions(precision=6, suppress=True)





# In[2]:





# note: all data update 2020-04-01, china rule, smooth weights

mname = 'gbt1u'

path = '/kaggle/input/gbt1n-external/'

pathk = '/kaggle/input/covid19-global-forecasting-week-4/'

nhorizon = 30

skip = 0

# kv = [3]

# kv = [6]

kv = [6,11]

# kv = [13]

train_full = True

save_data = False



# booster = ['lgb','xgb']

booster = ['lgb','xgb','ctb']

# booster = ['cas']



# if using updated daily data, also update time-varying external data

# in COVID-19 and covid-19-data, git pull origin master 

# ecdc wget https://opendata.ecdc.europa.eu/covid19/casedistribution/csv

# weather: https://www.kaggle.com/davidbnn92/weather-data/output?scriptVersionId=31103959

# google trends: pytrends0b.ipynb





# In[3]:





train = pd.read_csv(pathk+'train.csv')

test = pd.read_csv(pathk+'test.csv')

ss = pd.read_csv(pathk+'submission.csv')





# In[4]:





train





# In[5]:





# tmax and dmax are the last day of training

tmax = train.Date.max()

dmax = datetime.strptime(tmax,'%Y-%m-%d').date()

print(tmax, dmax)





# In[6]:





# ddate is the last day of validation training

ddate = dmax - timedelta(days=nhorizon)

ddate





# In[7]:





test





# In[8]:





fmax = test.Date.max()

fdate = datetime.strptime(fmax,'%Y-%m-%d').date()

fdate





# In[9]:





tmin = train.Date.min()

fmin = test.Date.min()

tmin, fmin





# In[10]:





dmin = datetime.strptime(tmin,'%Y-%m-%d').date()

print(dmin)





# In[11]:





# train['ForecastId'] = train.Id - train.Id.max()

cp = ['Country_Region','Province_State']

cpd = cp + ['Date']

train = train.merge(test[cpd+['ForecastId']], how='left', on=cpd)

train['ForecastId'] = train['ForecastId'].fillna(0).astype(int)

train['y0_pred'] = np.nan

train['y1_pred'] = np.nan



test['Id'] = test.ForecastId + train.Id.max()

test['ConfirmedCases'] = np.nan

test['Fatalities'] = np.nan

# use zeros here instead of nans so monotonic adjustment fills final dates if necessary

test['y0_pred'] = 0.0

test['y1_pred'] = 0.0





# In[12]:





# concat non-overlapping part of test to train for feature engineering

d = pd.concat((train,test[test.Date > train.Date.max()])).reset_index(drop=True)

d





# In[13]:





(dmin + timedelta(30)).isoformat()





# In[14]:





d['Date'].value_counts().std()





# In[15]:





# fill missing province with blank, must also do this with external data before merging

d[cp] = d[cp].fillna('')



# create single location variable

d['Loc'] = d['Country_Region'] + ' ' + d['Province_State']

d['Loc'] = d['Loc'].str.strip()

d['Loc'].value_counts()





# In[16]:





# sort by location then date

d = d.sort_values(['Loc','Date']).reset_index(drop=True)





# In[17]:





d['Country_Region'].value_counts(dropna=False)





# In[18]:





d['Province_State'].value_counts(dropna=False)





# In[19]:





gt = pd.read_csv(path+'google_trends.csv')

gt[cp] = gt[cp].fillna('')

gt





# In[20]:





# since trends data lags behind a day or two, shift the date to make it contemporaneous

gmax = gt.Date.max()

gmax = datetime.strptime(gmax,'%Y-%m-%d').date()

goff = (dmax - gmax).days

print(dmax, gmax, goff)

gt['Date'] = (pd.to_datetime(gt.Date) + timedelta(goff)).dt.strftime('%Y-%m-%d')

gt['google_covid'] = gt['coronavirus'] + gt['covid-19'] + gt['covid19']

gt.drop(['coronavirus','covid-19','covid19'], axis=1, inplace=True)

google = ['google_covid']

gt





# In[21]:





d = d.merge(gt, how='left', on=['Country_Region','Province_State','Date'])

d





# In[22]:





d['google_covid'].describe()





# In[23]:





# merge country info

country = pd.read_csv(path+'covid19countryinfo1.csv')

# country["pop"] = country["pop"].str.replace(",","").astype(float)

country





# In[24]:





country.columns





# In[25]:





d.shape





# In[26]:





# first merge by country

d = d.merge(country.loc[country.medianage.notnull(),['country','pop','testpop','medianage']],

            how='left', left_on='Country_Region', right_on='country')

d





# In[27]:





# then merge by province

c1 = country.loc[country.medianage.isnull(),['country','pop','testpop']]

print(c1.shape)

c1.columns = ['Province_State','pop1','testpop1']

# d.update(c1)

d = d.merge(c1,how='left',on='Province_State')

d.loc[d.pop1.notnull(),'pop'] = d.loc[d.pop1.notnull(),'pop1']

d.loc[d.testpop1.notnull(),'testpop'] = d.loc[d.testpop1.notnull(),'testpop1']

d.drop(['pop1','testpop1'], axis=1, inplace=True)

print(d.shape)

print(d.loc[(d.Date=='2020-03-25') & (d['Province_State']=='New York')])





# In[28]:





# testing data time series, us states only, would love to have this for all countries

ct = pd.read_csv(path+'states_daily_4pm_et.csv')

si = pd.read_csv(path+'states_info.csv')

si = si.rename(columns={'name':'Province_State'})

ct = ct.merge(si[['state','Province_State']], how='left', on='state')

ct['Date'] = ct['date'].apply(str).transform(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))

ct.loc[ct.Province_State=='US Virgin Islands','Province_State'] = 'Virgin Islands'

ct.loc[ct.Province_State=='District Of Columbia','Province_State'] = 'District of Columbia'

pd.set_option('display.max_rows', 20)

ct

# ct = ct['Date','state','total']





# In[29]:





ckeep = ['positive','negative','totalTestResults']

for c in ckeep: ct[c] = np.log1p(ct[c])





# In[30]:





d = d.merge(ct[['Province_State','Date']+ckeep], how='left',

            on=['Province_State','Date'])

d





# In[31]:





w = pd.read_csv(path+'training_data_with_weather_info_week_2.csv')

w.drop(['Id','Id.1','ConfirmedCases','Fatalities','country+province','day_from_jan_first'], axis=1, inplace=True)

w[cp] = w[cp].fillna('')

wf = list(w.columns[5:])

w





# In[32]:





# since weather data lags behind a day or two, adjust the date to make it contemporaneous

wmax = w.Date.max()

wmax = datetime.strptime(wmax,'%Y-%m-%d').date()

woff = (dmax - wmax).days

print(dmax, wmax, woff)

w['Date'] = (pd.to_datetime(w.Date) + timedelta(woff)).dt.strftime('%Y-%m-%d')

w





# In[33]:





# merge Lat and Long for all times and the time-varying weather data based on date

d = d.merge(w[cp+['Lat','Long']].drop_duplicates(), how='left', on=cp)

w.drop(['Lat','Long'],axis=1,inplace=True)

d = d.merge(w, how='left', on=cpd)

d





# In[34]:





# combine ecdc and nytimes data

ecdc = pd.read_csv(path+'ecdc.csv', encoding = 'latin')

ecdc





# In[35]:





# combine ecdc and nytimes data as extra y0 and y1

# https://opendata.ecdc.europa.eu/covid19/casedistribution/csv

ecdc['Date'] = pd.to_datetime(ecdc[['year','month','day']]).dt.strftime('%Y-%m-%d')

ecdc = ecdc.rename(mapper={'countriesAndTerritories':'Country_Region'}, axis=1)

ecdc['Country_Region'] = ecdc['Country_Region'].replace('_',' ',regex=True)

ecdc['Province_State'] = ''

ecdc['cc'] = ecdc.groupby(cp)['cases'].cummax()

ecdc['extra_y0'] = np.log1p(ecdc.cc)

ecdc['cd'] = ecdc.groupby(cp)['deaths'].cummax()

ecdc['extra_y1'] = np.log1p(ecdc.cd)

ecdc = ecdc[cpd + ['extra_y0','extra_y1']]

ecdc[::63]





# In[36]:





ecdc = ecdc[ecdc.Date >= '2020-01-22']

ecdc





# In[37]:





# https://github.com/nytimes/covid-19-data

nyt = pd.read_csv(path+'us-states.csv')

nyt['extra_y0'] = np.log1p(nyt.cases)

nyt['extra_y1'] = np.log1p(nyt.deaths)

nyt['Country_Region'] = 'US'

nyt = nyt.rename(mapper={'date':'Date','state':'Province_State'},axis=1)

nyt.drop(['fips','cases','deaths'],axis=1,inplace=True)

nyt





# In[38]:





extra = pd.concat([ecdc,nyt], sort=True)

extra





# In[39]:





d = d.merge(extra, how='left', on=cpd)

d





# In[40]:





d[['extra_y0','extra_y1']].describe()





# In[41]:





# recovered data from hopkins, https://github.com/CSSEGISandData/COVID-19

recovered = pd.read_csv(path+'time_series_covid19_recovered_global.csv')

recovered = recovered.rename(mapper={'Country/Region':'Country_Region','Province/State':'Province_State'}, axis=1)

recovered[cp] = recovered[cp].fillna('')

recovered = recovered.drop(['Lat','Long'], axis=1)

recovered





# In[42]:





# replace US row with identical rows for every US state

usp = d.loc[d.Country_Region=='US','Province_State'].unique()

print(usp, len(usp))

rus = recovered[recovered.Country_Region=='US']

rus





# In[43]:





rus = rus.reindex(np.repeat(rus.index.values,len(usp)))

rus.loc[:,'Province_State'] = usp

rus





# In[44]:





recovered =  recovered[recovered.Country_Region!='US']

recovered = pd.concat([recovered,rus]).reset_index(drop=True)

recovered





# In[45]:





# melt and merge

rm = pd.melt(recovered, id_vars=cp, var_name='d', value_name='recov')

rm





# In[46]:





rm['Date'] = pd.to_datetime(rm.d)

rm.drop('d',axis=1,inplace=True)

rm['Date'] = rm['Date'].dt.strftime('%Y-%m-%d')

rm





# In[47]:





d = d.merge(rm, how='left', on=['Country_Region','Province_State','Date'])

d





# In[48]:





d['recov'].describe()





# In[49]:





# approximate US state recovery via proportion of confirmed cases

d['ccsum'] = d.groupby(['Country_Region','Date'])['ConfirmedCases'].transform(lambda x: x.sum())

d.loc[d.Country_Region=='US','recov'] = d.loc[d.Country_Region=='US','recov'] *                                         d.loc[d.Country_Region=='US','ConfirmedCases'] /                                         (d.loc[d.Country_Region=='US','ccsum'] + 1)





# In[50]:





d.loc[:,'recov'] = np.log1p(d.recov)

# d.loc[:,'recov'] = d['recov'].fillna(0)





# In[51]:





d.loc[d.Province_State=='North Carolina','recov'][45:55]





# In[52]:





# log1p transform both targets

ynames = ['ConfirmedCases','Fatalities']

ny = len(ynames)

yv = []

for i in range(ny):

    v = 'y'+str(i)

    d[v] = np.log1p(d[ynames[i]])

    yv.append(v)

print(d[yv].describe())





# In[53]:





d['rate0'] = d.y0 - np.log(d['pop'])

d['rate1'] = d.y1 - np.log(d['pop'])





# In[54]:





d = d.sort_values(['Loc','Date']).reset_index(drop=True)

d.shape





# In[55]:





# compute nearest neighbors

regions = d[['Loc','Lat','Long']].drop_duplicates('Loc').reset_index(drop=True)

regions





# In[56]:





# regions.to_csv('regions.csv', index=False)





# In[57]:





# knn max features

k = kv[0]

nn = NearestNeighbors(k)

regions[['Lat','Long']]=regions[['Lat','Long']].fillna(-1)

nn.fit(regions[['Lat','Long']])





# In[58]:





# first matrix is distances, second indices to nearest neighbors including self

# note two cruise ships are replicated and have identical lat, long values

knn = nn.kneighbors(regions[['Lat','Long']])

knn





# In[59]:





ns = d['Loc'].nunique()





# In[60]:





# time series matrix

ky = d['y0'].values.reshape(ns,-1)

print(ky.shape)



print(ky[0])



# use knn indices to create neighbors

knny = ky[knn[1]]

print(knny.shape)



knny = knny.transpose((0,2,1)).reshape(-1,k)

print(knny.shape)





# In[61]:





# knn max features

nk = len(kv)

kp = []

kd = []

ns = regions.shape[0]

for k in kv:

    nn = NearestNeighbors(k)

    nn.fit(regions[['Lat','Long']])

    knn = nn.kneighbors(regions[['Lat','Long']])

    kp.append('knn'+str(k)+'_')

    kd.append('kd'+str(k)+'_')

    for i in range(ny):

        yi = 'y'+str(i)

        kc = kp[-1]+yi

        # time series matrix

        ky = d[yi].values.reshape(ns,-1)

        # use knn indices to create neighbor matrix

        km = ky[knn[1]].transpose((0,2,1)).reshape(-1,k)

        

        # take maximum value over all neighbors to approximate spreading

        d[kc] = np.amax(km, axis=1)

        print(d[kc].describe())

        print()

        

        # distance to max

        kc = kd[-1]+yi

        ki = np.argmax(km, axis=1).reshape(ns,-1)

        kw = np.zeros_like(ki).astype(float)

        # inefficient indexing, surely some way to do it faster

        for j in range(ns): 

            kw[j] = knn[0][j,ki[j]]

        d[kc] = kw.flatten()

        print(d[kc].describe())

        print()





# In[62]:





ki[j]





# In[63]:





# range of dates for training

# dates = d[~d.y0.isnull()]['Date'].drop_duplicates()

dates = d[d.y0.notnull()]['Date'].drop_duplicates()

dates





# In[64]:





# correlations

cols = []

for i in range(ny):

    yi = yv[i]

    cols.append(yi)

    for k in kp:

        cols.append(k+yi)

d.loc[:,cols].corr()





# In[65]:





d['Date'] = pd.to_datetime(d['Date'])

d['Date'].describe()





# In[66]:





# days since beginning

# basedate = train['Date'].min()

# train['dint'] = train.apply(lambda x: (x.name.to_datetime() - basedate).days, axis=1)

d['dint'] = (d['Date'] - d['Date'].min()).dt.days

d['dint'].describe()





# In[67]:





d.shape





# In[68]:





# reference days since exp(j)th occurrence

for i in range(ny):

    

    for j in range(3):



        ij = str(i)+'_'+str(j)

        

        cut = 2**j if i==0 else j

        

        qd1 = (d[yv[i]] > cut) & (d[yv[i]].notnull())

        d1 = d.loc[qd1,['Loc','dint']]

        # d1.shape

        # d1.head()



        # get min for each location

        d1['dmin'] = d1.groupby('Loc')['dint'].transform(lambda x: x.min())

        # dintmax = d1['dint'].max()

        # print(i,j,'dintmax',dintmax)

        # d1.head()



        d1.drop('dint',axis=1,inplace=True)

        d1 = d1.drop_duplicates()

        d = d.merge(d1,how='left',on=['Loc'])

 

        # if dmin is missing then the series had no occurrences in the training set

        # go ahead and assume there will be one at the beginning of the test period

        # the average time between first occurrence and first death is 14 days

        # if j==0: d[dmi] = d[dmi].fillna(dintmax + 1 + i*14)



        # ref day is days since dmin, must clip at zero to avoid leakage

        d['ref_day'+ij] = np.clip(d.dint - d.dmin, 0, 100000)

        d.drop('dmin',axis=1,inplace=True)



        # asymptotic curve may bin differently

        d['recip_day'+ij] = 1 / (1 + (1 + d['ref_day'+ij])**(-1.0))

    



gc.collect()





# In[69]:





d['dint'].value_counts().std()





# In[70]:





# diffs and rolling means

e = 1

r = 5

for i in range(ny):

    yi = 'y'+str(i)

    dd = '_d'+str(e)

    rr = '_r'+str(r)

    

    d[yi+dd] = d.groupby('Loc')[yi].transform(lambda x: x.diff(e))

    d[yi+rr] = d.groupby('Loc')[yi].transform(lambda x: x.rolling(r).mean())

    d['rate'+str(i)+dd] = d.groupby('Loc')['rate'+str(i)].transform(lambda x: x.diff(e))

    d['rate'+str(i)+rr] = d.groupby('Loc')['rate'+str(i)].transform(lambda x: x.rolling(r).mean())

    d['extra_y'+str(i)+dd] = d.groupby('Loc')['extra_y'+str(i)].transform(lambda x: x.diff(e))

    d['extra_y'+str(i)+rr] = d.groupby('Loc')['extra_y'+str(i)].transform(lambda x: x.rolling(r).mean())



    for k in kp:

        d[k+yi+dd] = d.groupby('Loc')[k+yi].transform(lambda x: x.diff(e))

        d[k+yi+rr] = d.groupby('Loc')[k+yi].transform(lambda x: x.rolling(r).mean())



    for k in kd:

        d[k+yi+dd] = d.groupby('Loc')[k+yi].transform(lambda x: x.diff(e))

        d[k+yi+rr] = d.groupby('Loc')[k+yi].transform(lambda x: x.rolling(r).mean())

        

laglist = ['recov'] + google + wf



for v in laglist:

    d[v+dd] = d.groupby('Loc')[v].transform(lambda x: x.diff(e))

    d[v+rr] = d.groupby('Loc')[v].transform(lambda x: x.rolling(r).mean())





# In[71]:





# final sort before training

d = d.sort_values(['Loc','dint']).reset_index(drop=True)

d.shape





# In[72]:





# initial continuous and categorical features

dogs = []

for i in range(ny):

    for j in range(3):

        dogs.append('ref_day'+str(i)+'_'+str(j))

cats = ['Loc']

print(dogs, len(dogs))

print(cats, len(cats))





# In[73]:





# one-hot encode categorical features

ohef = []

for i,c in enumerate(cats):

    print(c, d[c].nunique())

    ohe = pd.get_dummies(d[c], prefix=c)

    ohec = [f.translate({ord(c): "_" for c in " !@#$%^&*()[]{};:,./<>?\|`~-=_+"}) for f in list(ohe.columns)]

    ohe.columns = ohec

    d = pd.concat([d,ohe],axis=1)

    ohef = ohef + ohec





# In[74]:





d['Loc_US_North_Carolina'].describe()





# In[75]:





d['Loc_US_Colorado'].describe()





# In[76]:





# boosting hyperparameters

params = {}



params[('lgb','y0')] = {'lambda_l2': 1.9079933811271934, 'max_depth': 5}

params[('lgb','y1')] = {'lambda_l2': 1.690407455211948, 'max_depth': 3}

params[('xgb','y0')] = {'lambda_l2': 1.9079933811271934, 'max_depth': 5}

params[('xgb','y1')] = {'lambda_l2': 1.690407455211948, 'max_depth': 3}

params[('ctb','y0')] = {'l2_leaf_reg': 1.9079933811271934, 'max_depth': 5}

params[('ctb','y1')] = {'l2_leaf_reg': 1.690407455211948, 'max_depth': 3}





# In[77]:





# must start cas server before running this cell

if 'cas' in booster:

    from swat import *

    s = CAS('server', 1)





# In[78]:





# single horizon validation using one day at a time for 28 days

nb = len(booster)

nls = np.zeros((nhorizon,ny,nb))

rallv = np.zeros((nhorizon,ny,nb))

iallv = np.zeros((nhorizon,ny,nb)).astype(int)

yallv = []

pallv = []

imps = []

 

# loop over horizons

for horizon in range(1+skip,nhorizon+1):

# for horizon in range(4,5):

    

    print()

#     print('*'*20)

#     print(f'horizon {horizon}')

#     print('*'*20)

    

    gc.collect()

    

    hs = str(horizon)

    if horizon < 10: hs = '0' + hs

    

    # build lists of features

    lags = []

    diffs = []

    for i in range(ny):

        yi = 'y'+str(i)

        lags.append(yi)

        lags.append('extra_'+yi)

        lags.append('rate'+str(i))

        lags.append(yi+dd)

        lags.append('extra_'+yi+dd)

        lags.append('rate'+str(i)+dd)

        lags.append(yi+rr)

        lags.append('extra_'+yi+rr)

        lags.append('rate'+str(i)+rr)

        for k in kp:

            lags.append(k+yi)

            lags.append(k+yi+dd)

            lags.append(k+yi+rr)

        for k in kd:

            lags.append(k+yi)

            lags.append(k+yi+dd)

            lags.append(k+yi+rr)

       

    lags.append('recov')

    

    lags = lags + google + wf + ckeep

    

#     cinfo = ['pop', 'tests', 'testpop', 'density', 'medianage',

#        'urbanpop', 'hospibed', 'smokers']

    cinfo0 = ['testpop']

    cinfo1 = ['testpop','medianage']

    

    f0 = dogs + lags + cinfo0 + ohef

    f1 = dogs + lags + cinfo1 + ohef

    

    # remove some features based on validation experiments

    f0 = [f for f in f0 if not f.startswith('knn11') and not f.startswith('kd')          and not f.startswith('rate') and not f.endswith(dd) and not f.endswith(rr)]

    f1 = [f for f in f1 if not f.startswith('knn6') and not f.startswith('kd6')]

    

    # remove any duplicates

    # f0 = list(set(f0))

    # f1 = list(set(f1))

    

    features = []

    features.append(f0)

    features.append(f1)

    

    nf = []

    for i in range(ny):

        nf.append(len(features[i]))

        # print(nf[i], features[i][:10])

        

    qtrain = d['Date'] <= ddate.isoformat()



    vdate = ddate + timedelta(days=horizon)

    qval = d['Date'] == vdate.isoformat()

    qvallag = d['Date'] == ddate.isoformat()

    

    x_train = d[qtrain].copy()

    # make y training data monotonic nondecreasing

    y_train = []

    for i in range(ny):

        y_train.append(pd.Series(d.loc[qtrain,['Loc',yv[i]]].groupby('Loc')[yv[i]].cummax()))



    x_val = d[qval].copy()

    y_val = [d.loc[qval,'y0'].copy(), d.loc[qval,'y1'].copy()]

    yallv.append(y_val)

    

    # lag features

    x_train.loc[:,lags] = x_train.groupby('Loc')[lags].transform(lambda x: x.shift(horizon))

    x_val.loc[:,lags] = d.loc[qvallag,lags].values



    print()

    print(horizon, 'x_train', x_train.shape)

    print(horizon, 'x_val', x_val.shape)

    

    if train_full:

        

        qfull = (d['Date'] <= tmax)

        

        tdate = dmax + timedelta(days=horizon)

        qtest = d['Date'] == tdate.isoformat()

        qtestlag = d['Date'] == dmax.isoformat()

    

        x_full = d[qfull].copy()

        

        # make y training data monotonic nondecreasing

        y_full = []

        for i in range(ny):

            y_full.append(pd.Series(d.loc[qfull,['Loc',yv[i]]].groupby('Loc')[yv[i]].cummax()))

        

        x_test = d[qtest].copy()

        

        # lag features

        x_full.loc[:,lags] = x_full.groupby('Loc')[lags].transform(lambda x: x.shift(horizon))

        x_test.loc[:,lags] = d.loc[qtestlag,lags].values



        print(horizon, 'x_full', x_full.shape)

        print(horizon, 'x_test', x_test.shape)



    train_set = []

    val_set = []

    ny = len(y_train)



#     for i in range(ny):

#         train_set.append(xgb.DMatrix(x_train[features[i]], y_train[i]))

#         val_set.append(xgb.DMatrix(x_val[features[i]], y_val[i]))



    gc.collect()



    # loop over multiple targets

    mod = []

    pred = []

    rez = []

    iters = []

    

    for i in range(ny):

#     for i in range(1):

        print()

        print('*'*40)

        print(f'horizon {horizon} {yv[i]} {ynames[i]} {vdate}')

        print('*'*40)

        

        # use catboost only for y1

        # nb = 2 if i==0 else 3

       

        # matrices to store predictions

        vpm = np.zeros((x_val.shape[0],nb))

        tpm = np.zeros((x_test.shape[0],nb))

        

        for b in range(nb):

            

            if booster[b] == 'cas':

                

                x_train['Partition'] = 1

                x_val['Partition'] = 0

                x_cas_all = pd.concat([x_train, x_val], axis=0)

                # make copy of target since it is also used for lags

                x_cas_all['target'] = pd.concat([y_train[i], y_val[i]], axis=0).values

                s.upload(x_cas_all, casout="x_cas_val")



                target = 'target'

                inputs = features[i]

                inputs.append(target)



                s.loadactionset("autotune")

                res=s.autotune.tuneGradientBoostTree (

                    trainOptions = {

                        "table":{"name":'x_cas_val',"where":"Partition=1"},

                        "target":target,

                        "inputs":inputs,

                        "casOut":{"name":"model", "replace":True}

                    },

                    scoreOptions = {

                        "table":{"name":'x_cas_val', "where":"Partition=0"},

                        "model":{"name":'model'},

                        "casout":{"name":"x_valid_preds","replace":True},

                        "copyvars": ['Id','Loc','Date']

                    },

                    tunerOptions = {

                        "seed":54321,  

                        "objective":"RASE", 

                        "userDefinedPartition":True 

                    }

                )

                print()

                print(res.TunerSummary)

                print()

                print(res.BestConfiguration)        



                TunerSummary=pd.DataFrame(res['TunerSummary'])

                TunerSummary["Value"]=pd.to_numeric(TunerSummary["Value"])

                BestConf=pd.DataFrame(res['BestConfiguration'])

                BestConf["Value"]=pd.to_numeric(BestConf["Value"])

                vpt = s.CASTable("x_valid_preds").to_frame()

                #FG: resort the CAS predictions by Id

                vpt = vpt.sort_values(['Loc','Date']).reset_index(drop=True)

                vp = vpt['P_target'].values



                s.dropTable("x_cas_val")

                s.dropTable("x_valid_preds")

                

            else:

                # scikit interface automatically uses best model for predictions

                params[(booster[b],yv[i])]['n_estimators'] = 5000

                if booster[b]=='lgb':

                    model = lgb.LGBMRegressor(**params[(booster[b],yv[i])]) 

                elif booster[b]=='xgb':

                    model = xgb.XGBRegressor(**params[(booster[b],yv[i])])

                else:

                    # hack for categorical features, ctb must be last in booster list

                    features[i] = features[i][:-294] + ['Loc']

                    params[(booster[b],yv[i])]['cat_features'] = ['Loc']

                    model = ctb.CatBoostRegressor(**params[(booster[b],yv[i])])

                    

                model.fit(x_train[features[i]], y_train[i],

                                  eval_set=[(x_train[features[i]], y_train[i]),

                                            (x_val[features[i]], y_val[i])],

                                  early_stopping_rounds=30,

                                  verbose=False)



                vp = model.predict(x_val[features[i]])

                

                iallv[horizon-1,i,b] = model._best_iteration if booster[b]=='lgb' else                                        model.best_iteration if booster[b]=='xgb' else                                        model.best_iteration_



                gain = model.feature_importances_

        #         gain = model.get_score(importance_type='gain')

        #         split = model.get_score(importance_type='weight')   

            #     gain = model.feature_importance(importance_type='gain')

            #     split = model.feature_importance(importance_type='split').astype(float)  

            #     imp = pd.DataFrame({'feature':features,'gain':gain,'split':split})

                imp = pd.DataFrame({'feature':features[i],'gain':gain})

        #         imp = pd.DataFrame({'feature':features[i]})

        #         imp['gain'] = imp['feature'].map(gain)

        #         imp['split'] = imp['feature'].map(split)



                imp.set_index(['feature'],inplace=True)



                imp.gain /= np.sum(imp.gain)

        #         imp.split /= np.sum(imp.split)



                imp.sort_values(['gain'], ascending=False, inplace=True)



                print()

                print(imp.head(n=10))

                # print(imp.shape)



                imp.reset_index(inplace=True)

                imp['horizon'] = horizon

                imp['target'] = yv[i]

                imp['set'] = 'valid'

                imp['booster'] = booster[b]



                mod.append(model)

                imps.append(imp)

                

            # china rule, last observation carried forward, set to zero here

            qcv = (x_val['Country_Region'] == 'China') &                   (x_val['Province_State'] != 'Hong Kong') &                   (x_val['Province_State'] != 'Macau')

            vp[qcv] = 0.0



            # make sure horizon 1 prediction is not smaller than first lag

            # because we know series is monotonic

            # if horizon==1+skip:

            if True:

                a = np.zeros((len(vp),2))

                a[:,0] = vp

                a[:,1] = x_val[yv[i]].values

                vp = np.nanmax(a,axis=1)

            

            val_score = np.sqrt(mean_squared_error(vp, y_val[i]))

            vpm[:,b] = vp

            

            print()

            print(f'{booster[b]} validation rmse {val_score:.6f}')

            rallv[horizon-1,i,b] = val_score



            gc.collect()

    

#             break



            if train_full:

                

                print()

                print(f'{booster[b]} training with full data and predicting', tdate.isoformat())

                    

                if booster[b] == 'cas':

                    

                    x_full['target'] = y_full[i].values

                    s.upload(x_full, casout="x_full")

                    # use hyperparameters from validation fit

                    s.loadactionset("decisionTree")

                    result = s.gbtreetrain(

                        table={"name":'x_full'},

                        target=target,

                        inputs= inputs,

                        varimp=True,

                        ntree=BestConf.iat[0,2], 

                        m=BestConf.iat[1,2],

                        learningRate=BestConf.iat[2,2],

                        subSampleRate=BestConf.iat[3,2],

                        lasso=BestConf.iat[4,2],

                        ridge=BestConf.iat[5,2],

                        nbins=BestConf.iat[6,2],

                        maxLevel=BestConf.iat[7,2],

                        #quantileBin=True,

                        seed=326146718,

                        #savestate={"name":"aStore","replace":True}

                        casOut={"name":'fullmodel', "replace":True}

                        ) 



                    s.upload(x_test, casout="x_test_cas")



                    s.decisionTree.gbtreeScore(

                        modelTable={"name":"fullmodel"},        

                        table={"name":"x_test_cas"},

                        casout={"name":"x_test_preds","replace":True},

                        copyvars= ['Loc','Date']

                        ) 

                    # save test predictions back into main table

                    forecast = s.CASTable("x_test_preds").to_frame()

                    forecast = forecast.sort_values(['Loc','Date']).reset_index(drop=True)

                    tp = forecast['_GBT_PredMean_'].values

                    

                    s.dropTable("x_full")

                    s.dropTable("x_test_cas")

                     

                else:

                

                    # use number of iterations from validation fit

                    params[(booster[b],yv[i])]['n_estimators'] = iallv[horizon-1,i,b]

                    if booster[b]=='lgb':

                        model = lgb.LGBMRegressor(**params[(booster[b],yv[i])])

                    elif booster[b]=='xgb':

                        model = xgb.XGBRegressor(**params[(booster[b],yv[i])])

                    else:

                        model = ctb.CatBoostRegressor(**params[(booster[b],yv[i])])

                    

                    model.fit(x_full[features[i]], y_full[i], verbose=False)

                    

                    params[(booster[b],yv[i])]['n_estimators'] = 5000



                    tp = model.predict(x_test[features[i]])

                

                    gain = model.feature_importances_

            #         gain = model.get_score(importance_type='gain')

            #         split = model.get_score(importance_type='weight')   

                #     gain = model.feature_importance(importance_type='gain')

                #     split = model.feature_importance(importance_type='split').astype(float)  

                #     imp = pd.DataFrame({'feature':features,'gain':gain,'split':split})

                    imp = pd.DataFrame({'feature':features[i],'gain':gain})

            #         imp = pd.DataFrame({'feature':features[i]})

            #         imp['gain'] = imp['feature'].map(gain)

            #         imp['split'] = imp['feature'].map(split)



                    imp.set_index(['feature'],inplace=True)



                    imp.gain /= np.sum(imp.gain)

            #         imp.split /= np.sum(imp.split)



                    imp.sort_values(['gain'], ascending=False, inplace=True)



                    print()

                    print(imp.head(n=10))

                    # print(imp.shape)



                    imp.reset_index(inplace=True)

                    imp['horizon'] = horizon

                    imp['target'] = yv[i]

                    imp['set'] = 'full'

                    imp['booster'] = booster[b]



                    imps.append(imp)



                # china rule, last observation carried forward, set to zero here

                qct = (x_test['Country_Region'] == 'China') &                       (x_test['Province_State'] != 'Hong Kong') &                       (x_test['Province_State'] != 'Macau')

                tp[qct] = 0.0



                # make sure first horizon prediction is not smaller than first lag

                # because we know series is monotonic

                # if horizon==1+skip:

                if True:

                    a = np.zeros((len(tp),2))

                    a[:,0] = tp

                    a[:,1] = x_test[yv[i]].values

                    tp = np.nanmax(a,axis=1)



                tpm[:,b] = tp

                

                gc.collect()

                

        # nonnegative least squares to estimate ensemble weights

        x, rnorm = nnls(vpm, y_val[i])

        

        # smooth weights by shrinking towards all equal

        # x = (x + np.ones(3)/3.)/2



        # smooth weights with rolling mean, ewma

        # if horizon+skip > 1: x = (x + nls[horizon+skip-2,i])/2

        alpha = 0.1

        if horizon+skip > 1: x = alpha * x + (1 - alpha) * nls[horizon+skip-2,i]



        nls[horizon+skip-1,i] = x

        

        val_pred = np.matmul(vpm, x)

        test_pred = np.matmul(tpm, x)

        

        # china rule in case weights do not sum to 1

        # val_pred[qcv] = vpm[:,0][qcv]

        # test_pred[qcv] = tpm[:,0][qct]

        

        # save validation and test predictions back into main table

        d.loc[qval,yv[i]+'_pred'] = val_pred

        d.loc[qtest,yv[i]+'_pred'] = test_pred



        # ensemble validation score

        # val_score = np.sqrt(rnorm/vpm.shape[0])

        val_score = np.sqrt(mean_squared_error(val_pred, y_val[i]))

        

        rez.append(val_score)

        pred.append(val_pred)



    pallv.append(pred)

    

    # nnls weights

    w0 = ''

    w1 = ''

    for b in range(nb):

        w0 = w0 + f' {nls[horizon-1,0,b]:.2f}'

        w1 = w1 + f' {nls[horizon-1,1,b]:.2f}'

        

    print()

    print('Validation RMSLE')

    print(f'{ynames[0]} \t {rez[0]:.6f}  ' + w0)

    print(f'{ynames[1]} \t {rez[1]:.6f}  ' + w1)

    print(f'Mean \t \t {np.mean(rez):.6f}')



#     # break down RMSLE by day

#     rp = np.zeros((2,7))

#     for i in range(ny):

#         for di in range(50,57):

#             j = di - 50

#             qf = x_val.dint == di

#             rp[i,j] = np.sqrt(mean_squared_error(pred[i][qf], y_val[i][qf]))

#             print(i,di,f'{rp[i,j]:.6f}')

#         print(i,f'{np.mean(rp[i,:]):.6f}')

#         plt.plot(rp[i])

#         plt.title(ynames[i] + ' RMSLE')

#         plt.show()

        

    # plot actual vs predicted

    plt.figure(figsize=(10, 5))

    for i in range(ny):

        plt.subplot(1,2,i+1)

        plt.plot([0, 12], [0, 12], 'black')

        plt.plot(pred[i], y_val[i], '.')

        plt.xlabel('Predicted')

        plt.ylabel('Actual')

        plt.title(ynames[i])

        plt.grid()

    plt.show()

    

# save one big table of importances

impall = pd.concat(imps)



# remove number suffixes from lag names to aid in analysis

# impall['feature1'] = impall['feature'].replace(to_replace='lag..', value='lag', regex=True)



os.makedirs('imp', exist_ok=True)

fname = 'imp/' + mname + '_imp.csv'

impall.to_csv(fname, index=False)

print()

print(fname, impall.shape)



# save scores and weights

os.makedirs('rez', exist_ok=True)

fname = 'rez/' + mname+'_rallv.npy'

np.save(fname, rallv)

print(fname, rallv.shape)



fname = 'rez/' + mname+'_nnls.npy'

np.save(fname, nls)

print(fname, nls.shape)





# In[79]:





if 'cas' in booster: s.shutdown()





# In[80]:





tdate.isoformat()





# In[81]:





rf = [f for f in features[0] if f.startswith('ref')]

d[rf].describe()





# In[82]:





np.mean(iallv, axis=0)





# In[83]:





plt.figure(figsize=(10, 8))

for i in range(ny):

    plt.subplot(2,2,1+i)

    plt.plot(rallv[:,i])

    plt.title(ynames[i] + ' RMSLE vs Horizon')

    plt.grid()

    

    plt.subplot(2,2,3+i)

    plt.plot(nls[:,i])

    plt.title(ynames[i] + ' Ensemble Weights')

    plt.grid()

plt.show()





# In[84]:





# compute validation rmsle

m = 0

locs = d.loc[:,['Loc','Country_Region','Province_State']].drop_duplicates().reset_index(drop=True)

# locs = x_val.copy().reset_index(drop=True)

# print(locs.shape)

y_truea = []

y_preda = []



print(f'# {mname}')

for i in range(ny):

    y_true = []

    y_pred = []

    for j in range(nhorizon-skip):

        y_true.append(yallv[j][i])

        y_pred.append(pallv[j][i])

    y_true = np.stack(y_true)

    y_pred = np.stack(y_pred)

    # print(y_pred.shape)

    # make each series monotonic increasing

    for j in range(y_pred.shape[1]): 

        y_pred[:,j] = np.maximum.accumulate(y_pred[:,j])

    # copy updated predictions into main table

    for horizon in range(1+skip,nhorizon+1):

        vdate = ddate + timedelta(days=horizon)

        qval = d['Date'] == vdate.isoformat()

        d.loc[qval,yv[i]+'_pred'] = y_pred[horizon-1-skip]

    rmse = np.sqrt(mean_squared_error(y_pred, y_true))

    print(f'# {rmse:.6f}')

    m += rmse/2

    locs['rmse'+str(i)] = np.sqrt(np.mean((y_true-y_pred)**2, axis=0))

    y_truea.append(y_true)

    y_preda.append(y_pred)

print(f'# {m:.6f}')





# In[85]:





# gbt1u

# 0.626124

# 0.404578

# 0.515351





# In[86]:





# sort to find worst predictions of y0

locs = locs.sort_values('rmse0', ascending=False)

locs[:10]





# In[87]:





# plot worst fits of y0

for i in range(5):

    li = locs.index[i]

    plt.plot(y_truea[0][:,li])

    plt.plot(y_preda[0][:,li])

    plt.title(locs.loc[li,'Loc'])

    plt.show()





# In[88]:





# plt.plot(d.loc[d.Loc=='Belgium','y0'][39:])





# In[89]:





# sort to find worst predictions of y1

locs = locs.sort_values('rmse1', ascending=False)

locs[:10]





# In[90]:





# plot worst fits of y1

for i in range(5):

    li = locs.index[i]

    plt.plot(y_truea[1][:,li])

    plt.plot(y_preda[1][:,li])

    plt.title(locs.loc[li,'Loc'])

    plt.show()





# In[91]:





tmax





# In[92]:





# enforce monotonicity of forecasts in test set after last date in training

loc = d['Loc'].unique()

for l in loc:

    # q = (d.Loc==l) & (d.ForecastId > 0)

    q = (d.Loc==l) & (d.Date > tmax)

    for yi in yv:

        yp = yi+'_pred'

        d.loc[q,yp] = np.maximum.accumulate(d.loc[q,yp])





# In[93]:





# plot actual and predicted curves over time for specific locations

locs = ['China Tibet','China Xinjiang','China Hong Kong', 'China Macau',

        'Spain','Italy','India',

        'US Washington','US New York','US California',

        'US North Carolina','US Ohio']

xlab = ['03-12','03-18','03-25','04-01','04-08','04-15','04-22']

for loc in locs:

    plt.figure(figsize=(14,2))

    

    # fig, ax = plt.subplots()

    # fig.autofmt_xdate()

    

    for i in range(ny):

    

        plt.subplot(1,2,i+1)

        plt.plot(d.loc[d.Loc==loc,[yv[i],'Date']].set_index('Date'))

        plt.plot(d.loc[d.Loc==loc,[yv[i]+'_pred','Date']].set_index('Date'))

        # plt.plot(d.loc[d.Loc==loc,[yv[i]]])

        # plt.plot(d.loc[d.Loc==loc,[yv[i]+'_pred']])

        # plt.xticks(np.arange(len(xlab)), xlab, rotation=-45)

        # plt.xticks(np.arange(12), calendar.month_name[3:5], rotation=20)

        # plt.xticks(rotation=-45)

        plt.xticks([])

        plt.title(loc + ' ' + ynames[i])

       

    plt.show()





# In[94]:





pd.set_option('display.max_rows', 100)

loc = 'China Xinjiang'

d.loc[d.Loc==loc,['Date',yv[0],yv[0]+'_pred']]





# In[95]:





tmax





# In[96]:





fmin





# In[97]:





# compute public lb score

q = (d.Date >= fmin) & (d.Date <= tmax)

print(f'# {tmax} {sum(q)/ns} {mname}')

s0 = np.sqrt(mean_squared_error(d.loc[q,'y0'],d.loc[q,'y0_pred']))

s1 = np.sqrt(mean_squared_error(d.loc[q,'y1'],d.loc[q,'y1_pred']))

print(f'# CC \t {s0:.6f}')

print(f'# Fa \t {s1:.6f}')

print(f'# Mean \t {(s0+s1)/2:.6f}')





# In[98]:





# 2020-03-31 13.0 gbt1u

# CC 	 0.744772

# Fa 	 0.567829

# Mean 	 0.656301





# In[99]:





# create submission

sub = d.loc[d.ForecastId > 0, ['ForecastId','y0_pred','y1_pred']]

sub['ConfirmedCases'] = np.expm1(sub['y0_pred'])

sub['Fatalities'] = np.expm1(sub['y1_pred'])

sub.drop(['y0_pred','y1_pred'],axis=1,inplace=True)

os.makedirs('sub',exist_ok=True)

# fname = 'sub/' + mname + '.csv'

fname = 'sub3.csv'

sub.to_csv(fname, index=False)

print(fname, sub.shape)

sub3=sub.copy()





sub1.reset_index(drop=True,inplace=True)

sub2.reset_index(inplace=True)

sub3.reset_index(inplace=True)
sub1.shape,sub2.shape,sub3.shape
sub1=sub1[["ForecastId","ConfirmedCases","Fatalities"]]

sub2=sub2[["ForecastId","ConfirmedCases","Fatalities"]]

sub3=sub3[["ForecastId","ConfirmedCases","Fatalities"]]

sub1.head()
sub1.sort_values(['ForecastId'],inplace=True)

sub2.sort_values(['ForecastId'],inplace=True)

sub3.sort_values(['ForecastId'],inplace=True)

sub=sub1.copy()

sub['ConfirmedCases']=sub1['ConfirmedCases']*0.33+sub2['ConfirmedCases']*0.34+sub3['ConfirmedCases']*0.33

sub['Fatalities']=sub1['Fatalities']*0.33+sub2['Fatalities']*0.34+sub3['Fatalities']*0.33

sub.head()
sub.to_csv("submission.csv", index=False)