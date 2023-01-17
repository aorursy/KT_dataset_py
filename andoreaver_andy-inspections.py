import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys
import numpy

# list file into array
path = "../input"
dirs = os.listdir( path )

# only use csv files and process inspections and violations separately
for file in dirs:
    if file.endswith("inspections.csv"):
       print ("Charting Inspections File")
    
       inspections=pd.read_csv(path + "/" + file)
        
       print (inspections)
        
       df = pd.DataFrame(inspections, columns = ['facility_id', 'facility_zip'])
       df['facility_zip'] = df['facility_zip'].str[:5]
       df = df.drop_duplicates(keep=False)
       countbyfacility = df.groupby('facility_zip').count()
    
       countbyfacility.rename(columns={'facility_id' : 'RestaurantCount'}, inplace=True)
                     
       print (countbyfacility)
    
       countbyfacility.plot(kind='barh', title='Count of Inspected Restaurants by Zip')

       plt.show()

    if file.endswith("violations.csv"):
       print ("Charting Violations File")

       violations=pd.read_csv(path + "/" + file)
        
       status = pd.DataFrame(violations, columns = ['score', 'program_status'])
       active = status.loc[status['program_status'] == 'ACTIVE']
       inactive = status.loc[status['program_status'] == 'INACTIVE']
    
       active = active['score'].apply(pd.to_numeric)
       inactive = inactive['score'].apply(pd.to_numeric)
          
       print ("Plotting Active vs Inactives in Violations File")
            
       plt.hist(active, alpha=0.5,label='active')
       plt.hist(inactive, alpha=0.5, label='inactive')
       plt.title("Score Comparison of Actives vs Inactives")
       plt.legend(loc='upper left')
       plt.show()
   

