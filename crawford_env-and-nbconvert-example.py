!touch file1.csv

!touch file2.csv

!touch file3.csv
import os

import subprocess
files = ['file1.csv', 'file2.csv', 'file3.csv']



for file in files:

    print("Processing", file)

    

    # Step 1) For each iteration the dataset path will be stored in an environment variable so the notebook can read it

    os.environ['FILE_PATH'] = file 



    # Step 2) Call the same notebook each iteration, the notebook will get the dataset from the environment variable that was just set

    subprocess.call(['jupyter', 'nbconvert', '--execute', 'notebook1.ipynb', '--to html']) 

    

    # Step 3) Rename the output file to something else

    os.rename('notebook1.ipynb', file+'.html')
'''In your processing notebook,  modify it to access the dataset like this:'''

# this variable will be be different in each iteration of the for loop

filepath = os.environ.get('FILE_PATH') 

df = pd.dataframe(filepath)
