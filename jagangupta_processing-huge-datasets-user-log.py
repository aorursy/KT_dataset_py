#Load the required packages

import numpy as np

import pandas as pd

import multiprocessing as mp

import time



from subprocess import check_output

import subprocess

files=check_output(["ls", "../input"]).decode("utf8")

print(files)

#check the number of cores

num_cores = mp.cpu_count()

print("This kernel has :",num_cores,"cores")

#Check the number of row of each file

for file in files.split("\n"):

    path='../input/'+file

    popenobj=subprocess.Popen(['wc', '-l', path], stdout=subprocess.PIPE, 

                                              stderr=subprocess.PIPE)

    result,error= popenobj.communicate()

    #print(result,error)

    print("The file :",file,"has :",result.strip().split()[0],"rows")

# testing out the funciton for one iteration

import time

size=1e6   # 1 million

user_log_chunks = pd.read_csv('../input/user_logs.csv',chunksize=size,index_col=['msno'])

start_time = time.time()

for i,user_log_chunk in enumerate(user_log_chunks):

    #print(user_log_chunk.head())

    grouped_object=user_log_chunk.groupby(user_log_chunk.index,sort=False) # not sorting results in a minor speedup

    func = {'date':['min','max'], 

           'num_25':['sum'],'num_50':['sum'],

           'num_75':['sum'],'num_985':['sum'],

           'num_100':['sum'],'num_unq':['sum'],'total_secs':['sum']}

    if(i==0):  # for the fiest run create the result 

        result=grouped_object.agg(func)

        print(len(result))

        print("Round ",i,"took %s seconds" % (time.time() - start_time))

    else:

        result=result.append(grouped_object.agg(func))

        print(len(result))

        print("Round ",i,"took %s seconds" % (time.time() - start_time))

        break

    #print(print("Round ",i,"took %s seconds" % (time.time() - start_time)))

    
#writing as a function

def process_user_log(chunk):

    grouped_object=chunk.groupby(chunk.index,sort=False) # not sorting results in a minor speedup

    func = {'date':['min','max','count'], 

           'num_25':['sum'],'num_50':['sum'],

           'num_75':['sum'],'num_985':['sum'],

           'num_100':['sum'],'num_unq':['sum'],'total_secs':['sum']}

    answer=grouped_object.agg(func)

    return answer

print("done")
size=4e7 # 40 million

reader = pd.read_csv('../input/user_logs.csv',chunksize=size,index_col=['msno'])

start_time = time.time()

for i in range(10):

    user_log_chunk=next(reader)

    if(i==0):

        result=process_user_log(user_log_chunk)

        print("Loop ",i,"took %s seconds" % (time.time() - start_time))

    else:

        result=result.append(process_user_log(user_log_chunk))

        print("Loop ",i,"took %s seconds" % (time.time() - start_time))

    del(user_log_chunk)    

  
print(len(result))

check=result.index.unique()

print(len(check))

#result.columns.values

result.columns = ['_'.join(col).strip() for col in result.columns.values]

result.columns
result.head()

func = {'date_min':['min'],'date_max':['max'],'date_count':['count'] ,

           'num_25_sum':['sum'],'num_50_sum':['sum'],

           'num_75_sum':['sum'],'num_985_sum':['sum'],

           'num_100_sum':['sum'],'num_unq_sum':['sum'],'total_secs_sum':['sum']}

processed_user_log=result.groupby(result.index).agg(func)

print(len(processed_user_log))

processed_user_log.columns = processed_user_log.columns.get_level_values(0)

print(processed_user_log.head())
processed_user_log.head()
# this function did not work because apparently only lists can be passed back from a function through the 

# multiprocessing package 

# If we use multiple cores the process can be 16 times faster



# create as many processes as there are CPUs on your machine

# if __name__ == '__main__':

#     size=1000000

#     parser = pd.read_csv('../input/user_logs.csv',chunksize=size,index_col=['msno'])

#     num_processes = mp.cpu_count()

#     pool=mp.Pool(processes=num_processes)

#     start_time = time.time()

#     list_of_ans=[]

#     def collect_results(result):

#         list_of_ans.extend(result)

#         print(done)

#     for i,df in enumerate(parser):

#         print("starting at",i," %s seconds" % (time.time() - start_time))

#         pool.map(process_user_log,df,callback=collect_results) 

#         

#     print("action ",i,"took %s seconds" % (time.time() - start_time))

#     pool.close()

#     pool.join()

        

processed_user_log.to_csv("processed_user_log.csv")