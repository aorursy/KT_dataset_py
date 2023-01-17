

import numpy as np

import pandas as pd

from Package.ixFileP import fileP

fp = fileP()

#Agrithm for preception linear regression

#substract





#introduction





#method





#result





#concolution

#1. discussion for program convergency speed by 

#    1.1 increase data

#    1.2 distance in decreasing or increasing punish value 

#    1.3 

#2. 
fdExamplesSource = 'Z:\\out\\tmp\\dataset\\source\\'

fdExamplesFinal = 'Z:\\out\\tmp\\dataset\\Final\\'

fn = 'hw1_18_train.dat (1).txt'

fn1 = 'hw1_18_test.dat (1).txt'



def read_to_convert(fname):   

    fdin = fdExamplesSource

    fdout = fdExamplesFinal

    text_file = open(fdin+fname, "r")

    data = text_file.read()

    rData = data.replace('\t' , ' ').split('\n')

    rDataFin = '\n'.join(rData)

    fp.save_dict_to_file(fdout+fname , rDataFin)
#read all daily stock file

dSstAll = fp.getAllLocalFile(fdExamplesSource)

dSstAll01 = fp.getAllLocalFile(fdExamplesFinal)

dSstAll01
#convert \t

for fn in dSstAll:

    try:

        read_to_convert(fn)

    except:

        print('error fn',fn)
#verify

for fn in dSstAll:

    try:

        df = pd.read_csv(fdExamplesSource+fn, sep=" ", header=None)

        df1 = pd.read_csv(fdExamplesFinal+fn, sep=" ", header=None)

        len(df),len(df1),len(df.columns),len(df1.columns) 

        if(len(df) != len(df1)): print('error len of data' , fn , 's len' , len(df)  , 'd len' , len(df1))

        if(len(df.columns) != len(df1.columns)): print('error len of columns' , fn , 's len' , len(df.columns)  , 'd len' , len(df1.columns))

    except:

        print('error fn',fn)
#show data info

for fn in dSstAll:

    try:

        df = pd.read_csv(fdExamplesFinal+fn, sep=" ", header=None)

        colNames = ['X' + str(f) for f in df.columns]

        df.columns = colNames

        print('fname',fn,'columns',colNames,'data shape',df.shape)

        #df = df.set_index(0)

        #df.to_csv(fdExamplesFinal+fn, sep=" ")

    except :

        print('error fn',fn)