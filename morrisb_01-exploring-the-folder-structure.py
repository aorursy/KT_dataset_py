# To search directories

import os



# To store the data

import pandas as pd
# Variable to store all files

files = []



# Iterate over all files and subfolders

for dirname, _, filenames in os.walk('/kaggle/input'):

    

    # Iterate all filenames

    for filename in filenames:

        

        # Store the filename for later inspection

        files.append(os.path.join(dirname, filename))



print('{} files are in the directories.'.format(len(files)))





# Split the filenames into subfolders and filename

# Remove the first three folders (home, kaggle and input) since they do not add new information

files_split = [file.split('/')[3:] for file in files]





# Store the split files as DataFrame to get aggregated summaries 

df = pd.DataFrame(files_split, columns=['Folder_Depth_0', 'Folder_Depth_1', 'Folder_Depth_2', 'Folder_Depth_3', 'Folder_Depth_4'])



print('\nThese are some sampled entries:')

df.sample(3)
# Group the folders and count the number of entries

print('Here you can see the main files and the count of the articles in the subfolders:')

df.groupby(['Folder_Depth_0', 'Folder_Depth_1', 'Folder_Depth_2']).size()