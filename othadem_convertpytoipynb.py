##Convert py file to ipynb files to open and work in google colab##

# In this we use two libraries one is nbformat to create ipynb files and another one is import_ipynb to import one ipynb in other ipynb file

# So install both packages using below commands

#!pip install nbformat
#!pip install import_ipynb

import os

def convertPyToIpynbFile(dirPath) :
  print("in convertPyToIpynb dirPath ", dirPath)
  import os
  import nbformat
  
  if not dirPath.endswith("/") :
    dirPath = dirPath + "/"
  
  files = os.listdir(dirPath)
  
  for file in files :
    newFile = dirPath + file.split(".")[0]+".ipynb"
    if newFile not in files and file.endswith(".py") :
      with open(dirPath + file) as source :
        code = source.read()
        
      nb = nbformat.v4.new_notebook()
      nb.cells.append(nbformat.v4.new_code_cell(code))
      nbformat.write(nb, newFile)
      
      os.rename(dirPath + file, dirPath + file.split(".py")[0]+".py_backup")
   

#To convert py files existing in a particular folder then provide 
#the folder path below, uncomment the code and run

#dirPath = '/path/to/the/folder/'
#convertPyToIpynbFile(dirPath)

#after conversion,in order to import ipynb files use below line of code
#import import_ipynb
