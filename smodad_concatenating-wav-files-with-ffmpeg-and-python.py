import os

from pathlib import Path

import subprocess #module for launching process from within Python
# os.listdir(directory) makes a list of all subdirectories in 'directory' (specified as the main argument)

# Here, I placed all of the birdcall folders (aldfly, ... , yetvir) in the directory 'call'

dir_list = os.listdir('call')





for directory in dir_list:

    filename_list = os.listdir('./call/' + directory) # Make a list of filenames for each sub-directory in 'call'

    file_object = open(directory + '.txt', 'w') # Create a .txt file named according to the species

    with open('./' + directory + '.txt', 'w') as filehandle:

        for filename in filename_list:

            # This creates a .txt file where each filename is listed per line for ffmpeg to process

            filehandle.write("file 'D:\\Kaggle\\birdcall\\call\\"  + directory + "\\" + "%s" % filename + "'\n")

    file_object.close()
for directory in dir_list:

    cmdargs = "ffmpeg -f concat -safe 0 -i " + directory + ".txt -c copy ./call/" + directory + "/" + directory + ".wav"

    # subprocess.run() will allow the process to end before creating a new one

    subprocess.run(cmdargs, check=True, shell=True)