# dependecies
import os
import os.path 
import pandas as pd
import re
# creating a list containing all folder names of the dataset
folders = ["business","entertainment","politics","sport","tech"]
# Changing the current working dirctory to the dataset directory

os.chdir("/kaggle/input/bbc-full-text-document-classification/bbc") 

# Verifying whether the current directory changed to the desier directory or not
print(os.getcwd())
# generating  a list call paths containg all the detailed paths for accessing the folders of the dataset 
paths = []
for i in folders:
    paths.append(os.getcwd()+'/'+i)
# printing the path just for verification
print(paths)
# creating two list texts and labels to store the detailed content and
# and the category of the news
texts = []
labels = []
# getting the path of a folders one by one from the list called paths[]
for path in paths:
#     getting filenames one by one from the list of all files under a directory called path
    for filename in os.listdir(path):
#         opening the file in read only mode with encoding latin1
        with open(path+"/"+filename,"r", encoding = "latin") as file:
#             after reading the file contents are stored in a variable called data
            data = file.read()
#             after removing all newline characters and carriage return from the  data,
#             it is being added at the end of the list texts
            data = data.replace("\n"," ").replace('\r','')
            texts.append(data)
            file.close() # file is being closed
#         setting the category of the news as the folder name in which it reside  
        labels.append(os.path.basename(path)) # from the whole path basename only returns the folder name
       
# creating a pandas dataframe from two lists texts and labels as two columns
df = pd.DataFrame({'texts':texts,'labels':labels})
# Printing the data frame for verification purpose
df
# adding dependencies to encode the categories of the news  with integer
from sklearn.preprocessing import LabelEncoder
# Encoding the categories of the news  with integer
df["labels_to_category"] = LabelEncoder().fit_transform(df["labels"])
# verfing by printing last 11 data from the dataframe
df.tail(11)
# Converting the dataframe into csv file
df.to_csv('/kaggle/working/BBC_news.csv')
# Verifing whether the csv file is generated properly or not by reading data from it
df2 = pd.read_csv('/kaggle/working/BBC_news.csv', encoding = "latin")
df2