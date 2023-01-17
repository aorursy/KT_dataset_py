# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import csv



# Input data files are available in the "../input/" directory.



## initializing paths

inputFolderPath = '/kaggle/input/data-series-summarization-project-v3/'

outputFolderPath = '/kaggle/working/'



filename = 'synthetic_size50k_len256_znorm.bin'



inputFilePath = inputFolderPath + filename
def sum32(inputFilePath):

    """ Summurizes 50k 256*float32 time series read from a binary file to 50k 32-bytes summaries 

    

    Parameters: 

    inputFilePath (string): the path of the binary file containing the 50k time series to summarize

  

    Returns: 

    string: the path of the binary file of the 50k summaries



    """



    summary_filepath =  outputFolderPath + filename + '_sum32'    

    ##############################################################################

    ##change the following code for 32-bytes summarization 

    

    #read binary file

    time_series50k = np.fromfile(inputFilePath, dtype=np.float32).reshape(-1, 256)

        

    summary50k = []

    #dummy summarization(picking the first value to summarize the hole series)

    for time_series in time_series50k:

        summary50k.append(time_series[0])

    

    #write the result in a binary file, then return the output file path

    summary50knp = np.array(summary50k,dtype=np.float32)

    summary50knp.tofile(summary_filepath)    



    ##############################################################################

    return summary_filepath



def rec32(summary_filepath):

    """ reconstructs 50k 256*float32 time series based on 50k 32-bytes summaries read from binary file

    

    Parameters: 

    summary_filepath (string): the path of the binary file containing the 50k 32-bytes summaries

  

    Returns: 

    string: the path of the binary file of the 50k reconstructed time series

    """

    reconstructed_filepath = summary_filepath + '_rec32'



    ##############################################################################

    ##change the following code for reconstruction from 32-bytes summaries 

    

    #read binary file

    summary50k = np.fromfile(summary_filepath, dtype=np.float32)



    reconstructed50k = []

    #dummy reconstruction (duplicating single float32 value 256 times)

    for summary in summary50k:

        reconstructed50k.append([summary]*256)

    

    #write the result in a binary file, then return the output file path

    reconstructed50knp = np.array(reconstructed50k,dtype=np.float32)

    reconstructed50knp.tofile(reconstructed_filepath) 



    ##############################################################################

    return reconstructed_filepath

    



def sum64(inputFilePath):

    """ Summurizes 50k 256*float32 time series read from a binary file to 50k 64-bytes summaries 

    

    Parameters: 

    inputFilePath (string): the path of the binary file containing the 50k time series to summarize

  

    Returns: 

    string: the path of the binary file of the 50k summaries



    """



    summary_filepath =  outputFolderPath + filename + '_sum64'    

    ##############################################################################

    ##change the following code for 64-bytes summarization 

    

    #read binary file

    time_series50k = np.fromfile(inputFilePath, dtype=np.float32).reshape(-1, 256)

        

    summary50k = []

    #dummy summarization(picking the first value to summarize the hole series)

    for time_series in time_series50k:

        summary50k.append(time_series[0])

        summary50k.append(time_series[0])

    

    #write the result in a binary file, then return the output file path

    summary50knp = np.array(summary50k,dtype=np.float32)

    summary50knp.tofile(summary_filepath)    



    ##############################################################################

    return summary_filepath



def rec64(summary_filepath):

    """ reconstructs 50k 256*float32 time series based on 50k 64-bytes summaries read from binary file

    

    Parameters: 

    summary_filepath (string): the path of the binary file containing the 50k 64-bytes summaries

  

    Returns: 

    string: the path of the binary file of the 50k reconstructed time series

    """

    reconstructed_filepath = summary_filepath + '_rec64'



    ##############################################################################

    ##change the following code for reconstruction from 64-bytes summaries 

    

    #read binary file

    summary50k = np.fromfile(summary_filepath, dtype=np.float32).reshape(-1, 2)



    reconstructed50k = []

    #dummy reconstruction (duplicating the first single float32 value 256 times)

    for summary in summary50k:

        reconstructed50k.append([summary[0]]*256)

    

    #write the result in a binary file, then return the output file path

    reconstructed50knp = np.array(reconstructed50k,dtype=np.float32)

    reconstructed50knp.tofile(reconstructed_filepath) 



    ##############################################################################

    return reconstructed_filepath

    



def sum128(inputFilePath):

    """ Summurizes 50k 256*float32 time series read from a binary file to 50k 128-bytes summaries 

    

    Parameters: 

    inputFilePath (string): the path of the binary file containing the 50k time series to summarize

  

    Returns: 

    string: the path of the binary file of the 50k summaries



    """



    summary_filepath =  outputFolderPath + filename + '_sum128'    

    ##############################################################################

    ##change the following code for 128-bytes summarization 

    

    #read binary file

    time_series50k = np.fromfile(inputFilePath, dtype=np.float32).reshape(-1, 256)

        

    summary50k = []

    #dummy summarization(picking the first value to summarize the hole series)

    for time_series in time_series50k:

        summary50k.append(time_series[0])

        summary50k.append(time_series[0])

        summary50k.append(time_series[0])

        summary50k.append(time_series[0])

    

    #write the result in a binary file, then return the output file path

    summary50knp = np.array(summary50k,dtype=np.float32)

    summary50knp.tofile(summary_filepath)    



    ##############################################################################

    return summary_filepath



def rec128(summary_filepath):

    """ reconstructs 50k 256*float32 time series based on 50k 128-bytes summaries read from binary file

    

    Parameters: 

    summary_filepath (string): the path of the binary file containing the 50k 128-bytes summaries

  

    Returns: 

    string: the path of the binary file of the 50k reconstructed time series

    """

    reconstructed_filepath = summary_filepath + '_rec128'



    ##############################################################################

    ##change the following code for reconstruction from 128-bytes summaries 

    

    #read binary file

    summary50k = np.fromfile(summary_filepath, dtype=np.float32).reshape(-1, 4)



    reconstructed50k = []

    #dummy reconstruction (duplicating the first single float32 value 256 times)

    for summary in summary50k:

        reconstructed50k.append([summary[0]]*256)

    

    #write the result in a binary file, then return the output file path

    reconstructed50knp = np.array(reconstructed50k,dtype=np.float32)

    reconstructed50knp.tofile(reconstructed_filepath) 



    ##############################################################################

    return reconstructed_filepath

    

########################## Submission ################################################

############ follow this templateand do not modify this cell code ####################

s32= sum32(inputFilePath)

r32 = rec32(s32)

pred32=np.fromfile(r32, dtype=np.float32)



s64= sum64(inputFilePath)

r64 = rec64(s64)

pred64=np.fromfile(r64, dtype=np.float32)



s128= sum128(inputFilePath)

r128 = rec128(s128)

pred128=np.fromfile(r128, dtype=np.float32)



#preparing submission 

output = []

globalCsvIndex = 0 



#append reconstruction from 32-bytes summaries (pred32) to csv

for i in range (len(pred32)) :

    output.append([globalCsvIndex,pred32[i]])

    globalCsvIndex = globalCsvIndex+1



#append reconstruction from 64-bytes summaries (pred64) to csv

for i in range (len(pred64)) :

    output.append([globalCsvIndex,pred64[i]])

    globalCsvIndex = globalCsvIndex+1



#append reconstruction from 128-bytes summaries (pred128) to csv

for i in range (len(pred128)) :

    output.append([globalCsvIndex,pred128[i]])

    globalCsvIndex = globalCsvIndex+1



with open('submission.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(['id','expected'])

    writer.writerows(output)