
# -*- coding: utf-8 -*-

"""
Created on Mon May 28 09:53:54 2018
By INNOVATIVE DESIGNS Team of
Prof. S.Balachandran,K.V.Suresh and P.V.Subramaniam

@author: user
"""

"""
Function name: set_environment()

Purpose: To print the OS / Language / packages used with version number

Description: This program prints the OS / Language / packages used with version number

Input: 1) None

Output: prints the OS / Language / packages used with version number

User defined Functions called: Nil

Global parameters defined: Nil

"""

def set_environment():
    
    import platform
    import pandas as pd
    import numpy as np
    import matplotlib
    
    print('Operating system version....', platform.platform())
    print("Python version is........... %s.%s.%s" % sys.version_info[:3])
    print('pandas version is...........', pd.__version__)
    print('numpy version is............', np.__version__)
    print('matplotlib version is.......', matplotlib.__version__)



###                         ###
### End of function         ###
###                         ###
    
    
"""
Function name: Timer()

Purpose: Define a TIme Class to computer total execution time

Description: This program calculates the total execution time and prints

Input: 1) None

Output: prints the total execution time 

User defined Functions called: Nil

Global parameters defined: Nil

"""
import time

class Timer:

    def __init__(self):
        self.start = time.time()
    
    def restart(self):
        self.start = time.time()
    
    def get_time(self):
        end = time.time()
    
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
    
        return time_str
   
###                         ###
### End of function         ###
###                         ###
   
"""
       Repeat Donations	Probability
       
         1	                0.5	
         2	                0.53	
         3		           0.55	
         4		           0.57	
         5&6		      0.59	
         7&8		      0.60	
         9 to 15		 0.61	
         16 to 87		 0.62	
         88 and Above		 0.63	
"""
def get_prob_for_opt(donations_repeat_count):
    
    probs = [('Count', ['0-1','2-2','3-3','4-4','5-6','7-8','9-15','16-87','88-100000']),
            ('prob',  [0.5, 0.53, 0.55, 0.57, 0.59, 0.60, 0.61, 0.62, 0.63])]
            
    opt_prob = pd.DataFrame.from_items(probs)
   
    for ii in range(0, 10):
        
        str = opt_prob['Count'][ii] 
        lts = re.findall('\\b\\d+\\b', str)
        prb = opt_prob['prob'][ii] 
   
        lv = int(lts[0])
        uv = int(lts[1]) + 1
        
        if donations_repeat_count in range(lv, uv):
            return prb   
 
"""
### Function name: logs()
###
### Purpose: Define a function to log the activities performed
###
### Description: Logs successful events FYI info or unsuccessful events as debug info at various stages into the log file
###
### Input: Nil
###
### Output: Log file with name myapp.log
###         True or False
###
### User defined Functions called: Nil
###
### Global parameters defined: logger1, logger2 to indicate the stages of when actions or events took place.
###
###
####
"""
def logs():

    print("#### logs ####")


    import sys, getopt, os, logging, timeit, getpass

    ### set up logging to a file

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name) -12s %(levelname) -8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='predict_donor.log',
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    #set  a format which is simpler for console use

    formatter = logging.Formatter('%(name) - 12s: %(levelname) -8s %(message)s')

    # tell the handler to use this format

    console.setFormatter(formatter)

    # add the handler to the root Logger

    logging.getLogger('').addHandler(console)

    return True

###                         ###
### End of function         ###
###                         ###

### Function for main()

def main():

   import pandas as pd

#  df   =  pd.read_csv('C:/Users/kvsur/Desktop/Kaggle1/kaggle2/donor_params.csv',header = 0,
   df   =  pd.read_csv('../input/inoovative130618/donor_params.csv',header = 0,                    
                     nrows = 1,    
                     usecols = ['project_id','project_state', 'project_city', 'cutoff'],
                     low_memory = False)  
   
   print("\n Donor parameter file shape ","\n","-----------------------\n")              
   print(df.shape)     
   
   project_id        = df['project_id']    
   project_state     = df['project_state']
   project_city      = df['project_city']

   cutoff            = df['cutoff']
   
   print("\n project_id: ", project_id)      
   print("\n project_state: ", project_state)
   print("\n project_city: ", project_city)
   print("\n cutoff: ", cutoff)   

   return df
if __name__ == "__main__":

   import sys
   import pandas as pd
   import numpy as np
   import re, logging, timeit, getpass
   import datetime
   
                           
   #begin time

   my_timer = Timer() # At the beginning of the logic


   df1 = main()

   logs()

   logger1 = logging.getLogger('General Information')
   logger2 = logging.getLogger('Data')
   logger3 = logging.getLogger('Output')

   # Get the Python version
    
   set_environment()  

   user = ""

   user        = " User: " + getpass.getuser() 
   logger1.info(user)   
   
   user_input1 = "Project ID: " + df1['project_id']
   logger1.info(user_input1)
   
   user_input2 = "project_state: " + df1['project_state']
   logger1.info(user_input2)   
   
   user_input3 = "project_city: " + df1['project_city']
   logger1.info(user_input3)
    
   user_input4 = "cutoff: " + str(df1['cutoff'])
   logger1.info(user_input4) 
    

   donors =  pd.read_csv('../input/io/Donors.csv',header = 0,
#                    nrows = 5000,    
                     usecols=['Donor ID', 'Donor City', 'Donor State', 'Donor Zip','Donor Is Teacher'], low_memory=False)  
   
   logger2.info("Donors file shape")              

   logger2.info(donors.shape)                  
   
   logger2.info(list(donors.columns))   
   
   donations =  pd.read_csv('../input/io/Donations.csv',header = 0,   
                     usecols=['Donor ID'],low_memory=False)

   donors_repeat_donations   = donations['Donor ID'].value_counts()
   
   logger2.info("Repeat_Donations")

   donors_repeat_donations = pd.DataFrame({'Donor ID':donors_repeat_donations.index, 'Repeat Donations Count':donors_repeat_donations.values})
   logger2.info("donors_repeat_donations file shape ") 
   logger2.info(donors_repeat_donations.shape)
   
   logger2.info("\n ------------------------------------------------ ")                  

   ### Merge two data frames - Left outer join

   data =   pd.merge(donors, donors_repeat_donations, on= 'Donor ID')
   
   logger2.info("Merged file shape")
   logger2.info(data.shape)
   logger2.info(list(data.columns))
   
   n                  = data.shape[0]
   OptProb1           = [0] * n 
   Addprob1           = [0] * n 
   TotalProbability1  = [0] * n
   logger2.info(print("Loop count ", n))  
    
   for k in range(0, n):
       ky = data['Repeat Donations Count'][k]
       val = get_prob_for_opt(ky)       
       OptProb1[k] = val
       
       if data['Donor Is Teacher'][k] == 'Yes':
          if (data['Donor State'][k] == df1.iloc[0]['project_state']):
              if (data['Donor City'][k] != df1.iloc[0]['project_city']):
                  Addprob1[k] = 0.163392
              else: # 'Donor City' == 'project_city'
                  Addprob1[k] = 0.182336   
          else: # Donor State != project_state
              if (data['Donor City'][k] != df1.iloc[0]['project_city']):
                  Addprob1[k] = 0.054464
       else:  # 'donor_teacher' == 'No'         
          if (data['Donor State'][k] == df1.iloc[0]['project_state']):
              if (data['Donor City'][k] != df1.iloc[0]['project_city']):
                  Addprob1[k] = 0.091908
              else: # Donor City == 'project_city'
                  Addprob1[k] = 0.102564
          else: # Donor State != project_state
              if (data['Donor City'][k] != df1.iloc[0]['project_city']):
                  Addprob1[k] = 0.030636 
                  
       TotalProbability1[k] = (OptProb1[k] + Addprob1[k])
   
   data['TotalProbability']    = pd.Series(TotalProbability1)
   
   data  = data.sort_values(by = "TotalProbability", ascending =0)  
                                             
   data_filtered = data[(data['TotalProbability'] >=  float(df1.iloc[0]['cutoff']) )]
   
   logger3.info(print(data_filtered['TotalProbability'].describe()))
   

   ### We need only two columns donor id and total prob 

   donorid          = data_filtered['Donor ID']
   totalprobability = data_filtered['TotalProbability']

   print(df1['project_id'].shape)

   ### Create a CSV file
   output_file_name = df1['project_id'][0] + "_at_cut_off" + str(df1['cutoff'][0])+".csv"
   print(output_file_name)



   labels = ['Donor Id' ,'Total Probability'] 
   df2 = pd.DataFrame(list(zip(donorid, totalprobability)),columns = labels)
   
   logger3.info(print(df2.tail()))
    
   ### Write to a csv file
   df2.to_csv(output_file_name, index = False)
   
   msg = "Number of selected donors who are likely to repeat donation: "
   logger3.info(msg)
   logger3.info(data_filtered.shape[0])
   
   logger3.info(output_file_name + " file is created")
   
  
   
   ### Closing time
   
   elapsed = my_timer.get_time()

   msg = "Total compute time was hh:mm:ss: " + str(elapsed)
   print(msg)    
   logger3.info(msg) # Towards end of the program

###
###  End
###    