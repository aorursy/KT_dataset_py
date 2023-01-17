import os

import glob



import pandas as pd
# Filename variables shortcuts

data_path = r'Datasets\Processed'

csv_name = '\Detailed_Statistics_Arrivals.csv'

dest_name = '\Las Vegas Arrivals'

filename =  data_path + dest_name + csv_name

filename
# Check data

df = pd.read_csv(filename)

df.info()
# To combine multiple datasets in a folder: 

# https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/



# Set the working directory

os.chdir(data_path + dest_name)
# Find '.csv' externsions

extension = 'csv'

all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# Check 'all_filenames'

all_filenames
# Combine all files and export!

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])



data_path = r'..\..\Combined'

dest_name = '\Las Vegas Arrivals'

csv_combined = 'LAS_Arrivals_2019.csv'



# Change the working directory

os.chdir(data_path + dest_name)







save_path = csv_combined  



# Export 

combined_csv.to_csv(csv_combined, index=False)
combined_csv