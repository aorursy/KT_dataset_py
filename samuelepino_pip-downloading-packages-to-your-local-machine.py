!pip download facenet-pytorch -d ./facenet_pytorch/

# add other pip downloads if needed
import os

from zipfile import ZipFile



dirName = "./"

zipName = "packages.zip"



# Create a ZipFile Object

with ZipFile(zipName, 'w') as zipObj:

    # Iterate over all the files in directory

    for folderName, subfolders, filenames in os.walk(dirName):

        for filename in filenames:

            if (filename != zipName):

                # create complete filepath of file in directory

                filePath = os.path.join(folderName, filename)

                # Add file to zip

                zipObj.write(filePath)