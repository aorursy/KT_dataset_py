# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#!/usr/bin/python

# -*- coding: utf-8 -*-



from numpy import *  

import csv  

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier



  

def preprocess(array):  

    array=mat(array)  

    m,n=shape(array)  

    temp=zeros((m,n))  

    for i in xrange(m):  

        for j in xrange(n):  

                temp[i,j]=int(array[i,j])  

    return temp  



def nomalizing(array):  

    m,n=shape(array)  

    for i in xrange(m):  

        for j in xrange(n):  

            if array[i,j]!=0:  

                array[i,j]=1  

    return array  

      

def load_train_data():  

    l=[]  

    with open('train.csv') as file:  

         lines=csv.reader(file)  

         for line in lines:  

             l.append(line) 

    l.remove(l[0])  

    l=array(l)  

    label=l[:,0]  

    data=l[:,1:]  

    return nomalizing(preprocess(data)),preprocess(label)



def load_test_data():  

    l=[]  

    with open('test.csv') as file:  

         lines=csv.reader(file)  

         for line in lines:  

             l.append(line)

    l.remove(l[0])  

    data=array(l)  

    return nomalizing(preprocess(data)) 



def load_test_result():  

    l=[]  

    with open('rf_benchmark.csv') as file:  

         lines=csv.reader(file)  

         for line in lines:  

             l.append(line)

    l.remove(l[0])  

    label=array(l)  

    return preprocess(label[:,1])



def save_result(results,file):  

    this_file=open(file,'w')

    this_file.write("ImageId,Label\n")

    for i,r in enumerate(results):

        this_file.write(str(i+1)+","+str(r)+"\n")



    this_file.close()



    # with open(file,'wb') as this_file:  



    #     the_writer=csv.writer(this_file) 

    #     the_writer.writerow("ImageId,Label\n")  

    #     for i in result: 

    #         image_id=str(i+1)

    #         images=[]

    #         images.append(image_id)

    #         tmp=[]  

    #         tmp.append(i)  

    #         the_writer.writerow(image_id+','+ str(tmp) )



def kann_classify(train_data,train_label,test_data):  

      

    knnClf=KNeighborsClassifier(n_neighbors=5)

    knnClf.fit(train_data,ravel(train_label))  

    test_label=knnClf.predict(test_data)  

    save_result(test_label,'sklearn_knn_Result.csv')  

    return test_label  



def random_forest_regressor(train_data,train_label,test_data):

    model = RandomForestRegressor(n_estimators=100,n_jobs=2, min_samples_leaf=2)

    model.fit(train_data, ravel(train_label))

    test_label=model.predict(test_data)

    save_result(test_label,'sklearn_random_forest_regressor_Result.csv')  

    return test_label  



def random_forest_classify(train_data,train_label,test_data):

    rf = RandomForestClassifier(n_estimators=100)

    rf.fit(train_data, ravel(train_label))

    test_label=rf.predict(test_data)

    

    save_result(test_label,'sklearn_random_forest_classify_Result.csv')  

    return test_label 



def recognize_digit(model='rfc'):  

    train_data,train_label=load_train_data()  

    test_data=load_test_data()  

    

    if model=='kc':

        result=kann_classify(train_data,train_label,test_data)

    elif model=='rfr':

        result=random_forest_regressor(train_data,train_label,test_data)

    elif model=='rfc':

        result=random_forest_classify(train_data,train_label,test_data)

   

    result_given=load_test_result()  

    m,n=shape(test_data)  



    different=0      

    for i in xrange(m):



        if result[i]!=result_given[0,i]:



            different+=1

           

    print(different)











if __name__ == '__main__':



    recognize_digit()
