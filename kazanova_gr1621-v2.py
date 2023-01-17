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



directory="/kaggle/input/covid19-global-forecasting-week-2/"

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
fix_target(train, key, target1, new_target_name=target1)

#fix_target(train, key, target2, new_target_name=target2)



rate(train, key, target1, new_target_name="rate_" +target1 )

rate(train, key, target2, new_target_name="rate_" +target2 )

unique_keys=train[key].unique()

print(len(unique_keys))





train
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





for j in range (preds_confirmed_cv.shape[1]):



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

submission=test_new[["ForecastId","ConfirmedCases","Fatalities"]]



submission.to_csv( "submission.csv", index=False)



submission