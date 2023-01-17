import os  
from PIL import Image 
import matplotlib.pyplot as plt
import zipfile
sunny_dir = '/kaggle/input/twoclass-weather-classification/weather_database/sunny/' #path where the sunny images are
sunny_files = os.listdir(sunny_dir) #saves all the names of the files contained in the cloudy folder into a list
print(len(sunny_files))

cloudy_dir = '/kaggle/input/twoclass-weather-classification/weather_database/cloudy/' #path where the sunny images are
cloudy_files = os.listdir(cloudy_dir) #saves all the names of the files contained in the cloudy folder into a list
print(len(cloudy_files))
os.mkdir('./resized_wdb') #creates a folder, it will be inside output/kaggle/working
os.mkdir('./resized_wdb/cloudy') #creates folder cloudy
os.mkdir('./resized_wdb/sunny') #creates folder sunny
my_zipfile = zipfile.ZipFile("./weather_db2.zip", mode='w', compression=zipfile.ZIP_DEFLATED) 
#creates an empty zip file. It will be saved inside the folder resized_wdb
for i in range(5000):
    img = Image.open(cloudy_dir+cloudy_files[i]) #opens an image from the original dataset
    img_resize = img.resize((200,200)) #resize the image --> 200x200 pixels
    img_resize.save('./resized_wdb/cloudy/'+'c'+cloudy_files[i])
    #saves the new image with the name c0457.jpg for example (c for cloudy + the number of the image + format=jpg)
for i in range(5000):
    img = Image.open(sunny_dir+sunny_files[i])
    img_resize = img.resize((200,200))
    img_resize.save('./resized_wdb/sunny/'+'s'+sunny_files[i])
# Writes all the images contained in the 'sunny' folder into  the zip file
for i in range(5000):
    my_zipfile.write("./resized_wdb/sunny/"+'s'+sunny_files[i])
# Writes all the images contained in the 'cloudy' folder into  the zip file
for i in range(5000):
    my_zipfile.write("./resized_wdb/cloudy/"+'c'+cloudy_files[i])
my_zipfile.close() #close zipfile
#shows an image from the resized_wdb 
example = Image.open('./resized_wdb/cloudy/c0056.jpg')
plt.imshow(example)