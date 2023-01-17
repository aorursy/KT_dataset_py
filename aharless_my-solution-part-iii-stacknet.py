ACTUALLY_RUN_MAKEDATA = False

ACTUALLY_RUN_MAKESUB = False
# make_stacknet_data17a.py



# Based on this kaggle script : https://www.kaggle.com/danieleewww/xgboost-lightgbm-and-olsv107-w-month-features/code



import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix





directory="input/" # hodls the data



## converts arrayo to sparse svmlight format

def fromsparsetofile(filename, array, deli1=" ", deli2=":",ytarget=None):    

    zsparse=csr_matrix(array)

    indptr = zsparse.indptr

    indices = zsparse.indices

    data = zsparse.data

    print(" data lenth %d" % (len(data)))

    print(" indices lenth %d" % (len(indices)))    

    print(" indptr lenth %d" % (len(indptr)))

    

    f=open(filename,"w")

    counter_row=0

    for b in range(0,len(indptr)-1):

        #if there is a target, print it else , print nothing

        if type(ytarget)!=type(None):

             f.write(str(ytarget[b]) + deli1)     

             

        for k in range(indptr[b],indptr[b+1]):

            if (k==indptr[b]):

                if np.isnan(data[k]):

                    f.write("%d%s%f" % (indices[k],deli2,-1))

                else :

                    f.write("%d%s%f" % (indices[k],deli2,data[k]))                    

            else :

                if np.isnan(data[k]):

                     f.write("%s%d%s%f" % (deli1,indices[k],deli2,-1))  

                else :

                    f.write("%s%d%s%f" % (deli1,indices[k],deli2,data[k]))

        f.write("\n")

        counter_row+=1

        if counter_row%10000==0:    

            print(" row : %d " % (counter_row))    

    f.close()  

    

#creates the main dataset abd prints 2 files to dataset2_train.txt and  dataset2_test.txt



def dataset2():



    ##### RE-READ PROPERTIES FILE

    

    print( "\nRe-reading properties file ...")

    properties = pd.read_csv(directory +'properties_2016.csv')



    train = pd.read_csv(directory +"train_2016_v2.csv")      



    properties2016_raw = pd.read_csv('../input/properties_2016.csv', low_memory = False)

    properties2017 = pd.read_csv('../input/properties_2017.csv', low_memory = False)

    taxvars = ['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount']

    tax2016 = properties2016_raw[['parcelid']+taxvars]

    properties2016 = properties2017.drop(taxvars,axis=1).merge(tax2016, 

                 how='left', on='parcelid').reindex_axis(properties2017.columns, axis=1)

    for c in properties2016.columns:

        properties2016[c]=properties2016[c].fillna(-1)

        if properties2016[c].dtype == 'object':

            lbl = LabelEncoder()

            lbl.fit(list(properties2016[c].values))

            properties2016[c] = lbl.transform(list(properties2016[c].values))

    for c in properties2017.columns:

        properties2017[c]=properties2017[c].fillna(-1)

        if properties2017[c].dtype == 'object':

            lbl = LabelEncoder()

            lbl.fit(list(properties2017[c].values))

            properties2017[c] = lbl.transform(list(properties2017[c].values))

    train2016 = pd.read_csv('../input/train_2016_v2.csv')

    train2017 = pd.read_csv('../input/train_2017.csv')



    sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False)

    train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')

    train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')

    train_df = pd.concat([train2016, train2017], axis = 0)



    test = pd.merge(sample_submission[['ParcelId']], properties2017.rename(columns = {'parcelid': 'ParcelId'}), 

                how = 'left', on = 'ParcelId')



    ##### PROCESS DATA FOR XGBOOST

        

    train_df["transactiondate"] = pd.to_datetime(train_df["transactiondate"])

    train_df["Month"] = train_df["transactiondate"].dt.month

    

    x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)

    x_test = test.drop(['ParcelId'], axis=1)

    

    x_test["transactiondate"] = '2016-07-01'

    x_test["transactiondate"] = pd.to_datetime(x_test["transactiondate"])

    x_test["Month"] = x_test["transactiondate"].dt.month #should use the most common training date 2016-10-01

    x_test = x_test.drop(['transactiondate'], axis=1)

    

    # shape        

    print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

    

    # drop out ouliers

    train_df=train_df[ train_df.logerror > -0.4 ]

    train_df=train_df[ train_df.logerror < 0.419 ]

    x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)

    y_train = train_df["logerror"].values.astype(np.float32)

    x_train = x_train.values.astype(np.float32, copy=False)

    x_test = x_test.values.astype(np.float32, copy=False)  

  

    print('After removing outliers:')     

    print (" shapes of dataset 2 ", x_train.shape, y_train.shape, x_test.shape)

    

    print (" printing %s " % ("dataset2_train.txt") )

    fromsparsetofile("dataset2_train.txt", x_train, deli1=" ", deli2=":",ytarget=y_train)     

    print (" printing %s " % ("dataset2_test.txt") )    

    fromsparsetofile("dataset2_test.txt", x_test, deli1=" ", deli2=":",ytarget=None)         

    print (" finished with daatset2 " )      

    return

 





def main():

    

    if ACTUALLY_RUN_MAKEDATA:

        dataset2()



    

        print( "\nFinished ...")

    

    

    



if __name__ == '__main__':

   main()
# create_submission.py



if ACTUALLY_RUN_MAKESUB:



    # -*- coding: utf-8 -*-



    #generates submission based on 1-column prediction csv



    sample="input/sample_submission.csv" # name of sample sybmission

    prediction="pred2.csv"# prediction file

    output="output_dataset2017a.csv"# output submission



    #the predictions are copied 6 times



    ff=open(sample, "r")

    ff_pred=open(prediction, "r")

    fs=open(output,"w")

    fs.write(ff.readline())

    s=0

    for line in ff: #read sample submission file

        splits=line.split(",")

        ids=splits[0] # get id

        pre_line=ff_pred.readline().replace("\n","") # parse prediction file and get prediction for the row

        fs.write(ids) # write id

        for j in range(6): # copy the prediction 6 times

            fs.write( "," +pre_line )

        fs.write("\n")

        s+=1

    ff.close() 

    ff_pred.close()

    fs.close()       

    print ("done")