import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import numpy as np

from tqdm import tqdm
np.random.seed(3)
raw_train_data = pd.read_csv ('../input/train.csv', nrows=100000000,dtype={'acoustic_data': np.int8, 'time_to_failure': np.float32})

#raw_train_data = pd.read_csv ('../input/train.csv',dtype={'acoustic_data': np.int8, 'time_to_failure': np.float32})
def generate_data_set (input_list,set_size,max_overlap=0,histogram=[],exclusions=[]):

# input_list - numpy array(?,2) of train data

# set_size - number of samples in the data set

# overlap - define max overlap in rows (150000=100%)

# histogram - target (y-)histogram for the data set (in 1 second steps). If [], random distribution

# exclusions - array of index of samples to be excluded (eg. to generate more mutually exclusive validation sets) 

#              Use with care! There is a limited number of possibilites to create validation sets with specific 

#              histogram. There is no protection against endless loop built in!





    i=0

    m=input_list.shape[0]

    index=[]



    if histogram!=[]:

        data_size=np.sum(histogram[0])

        set_hist=np.round(histogram[0]/data_size*set_size)

    else:

        data_size=17        

        set_hist=np.zeros((17,))

        set_hist[:]=set_size



#    print ("Set buckets shape "+str(set_hist.shape))

#    print ("Expected buckets\n"+str(set_hist))

    if np.sum(set_hist)!=set_size:             # Correction for rounding errors

        set_hist[0]=set_hist[0]-(np.sum(set_hist)-set_size)

    set_bucket=np.zeros(set_hist.shape[0])

    set_full=np.zeros(set_hist.shape[0])         

    set_full[set_hist==0]=1.                    # When histogram bucket is empty, it is actually already full

    for j in tqdm(range (0,set_size)):

        while True:

            start = np.int(np.random.rand()*(m-150000)) #reduced length by 150000 to avoid to hit prematurelly end

#            print ("Start "+str(start))

            overlap=False                               #test for overlap > max_overlap between samples

            for i in index:

                if (150000-np.abs(start-i)) > max_overlap:

#                    print ("overlap of "+str(150000-np.abs(start-i))+" detected at "+str(i)+" with "+str(start))

                    overlap=True

                    break

            if overlap:

                continue 

            exclude=False

            for i in exclusions:

                if (150000-np.abs(start-i))>0:

#                    print ("In exclusion list "+str(i)+" with "+str(start))

                    overlap=True

                    break

            if overlap:

                continue 

            x=input_list[start:start+150000]

            if np.max(x[1:,1]-x[:-1,1]) > 1:           #testing for earthquake within the sample

#                print ("earthquake within sample")

                continue  

            y=np.average(x[:,1])   

            y_int=int(np.floor(y))

            

            if set_full[y_int]:                 #testing if we need this y - does it fit to histogram?

#                print ("Bucket "+str(y_int)+" full")

                continue

            # If we reached it here, we should have a valid sample

#            print ("valid sample "+str(start)+" with y "+str(y))

            set_bucket[y_int]=set_bucket[y_int]+1

#            print (set_bucket)

            if set_bucket[y_int]==set_hist[y_int]:

#                print ("Bucket "+str(y_int)+" full")

                set_full[y_int]=1

            break

        index.append(start)

#    print ("Final buckets\n"+str(set_bucket))

    return index
train_data_hist=np.histogram(raw_train_data['time_to_failure'].values,

                             bins=range(int(np.floor(raw_train_data['time_to_failure'].values.min())),

                                        int(np.ceil(raw_train_data['time_to_failure'].values.max()))+1))

print (train_data_hist)
validation_set=np.zeros((3,64),dtype=int)
validation_set[0,:] = generate_data_set(raw_train_data.values,64)
validation_set[1,:] = generate_data_set(raw_train_data.values,64,histogram=train_data_hist)
validation_set[2,:] = generate_data_set(raw_train_data.values,64,histogram=train_data_hist,

                                              exclusions=validation_set[1,:])
train_data = generate_data_set(raw_train_data.values,500,histogram=train_data_hist,exclusions=validation_set[1,:],

                              max_overlap=75000)
plt.hist(raw_train_data['time_to_failure'].values,

         bins=range(int(np.floor(raw_train_data['time_to_failure'].values.min())),

                    int(np.ceil(raw_train_data['time_to_failure'].values.max()))+1))
plt.hist(raw_train_data['time_to_failure'].values[train_data],

         bins=range(int(np.floor(raw_train_data['time_to_failure'].values.min())),

                    int(np.ceil(raw_train_data['time_to_failure'].values.max()))+1))
plt.hist(raw_train_data['time_to_failure'].values[validation_set[0,:]],

         bins=range(int(np.floor(raw_train_data['time_to_failure'].values.min())),

                    int(np.ceil(raw_train_data['time_to_failure'].values.max()))+1))
plt.hist(raw_train_data['time_to_failure'].values[validation_set[1,:]],

         bins=range(int(np.floor(raw_train_data['time_to_failure'].values.min())),

                    int(np.ceil(raw_train_data['time_to_failure'].values.max()))+1))
plt.hist(raw_train_data['time_to_failure'].values[validation_set[2,:]],

         bins=range(int(np.floor(raw_train_data['time_to_failure'].values.min())),

                    int(np.ceil(raw_train_data['time_to_failure'].values.max()))+1))