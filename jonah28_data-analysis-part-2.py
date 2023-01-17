import os
import pandas as pd
from shutil import copyfile

df = pd.read_csv("myauto_ge_cars_data.csv")
df.head()
ids = list(df["ID"])
manufacturers = list(df["Manufacturer"])
manufactureres = ['TOYOTA',
                 'MERCEDES-BENZ',
                 'HYUNDAI',
                 'LEXUS',
                 'FORD',
                 'HONDA',
                 'BMW',
                 'VOLKSWAGEN',
                 'NISSAN',
                 'CHEVROLET']
rootdir = os.getcwd() + "/First_Cars"

counter = 1
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        car_id = int(subdir.split("/")[-1])
        
        m = manufacturers[ids.index(car_id)]
        if car_id in ids and m in manufactureres:            
            try:
                os.mkdir(os.getcwd() + "/training_set/" + m.lower())
            except:
                pass
            
            copyfile(path, os.getcwd() + "/training_set/" + m.lower() + "/" + str(counter) + ".jpg")
            counter += 1


files = folders = 0

for _, dirnames, filenames in os.walk(os.getcwd() + "/training_set"):
    files += len(filenames)
    folders += len(dirnames)

print("{:,} files, {:,} folders".format(files, folders))
# ახლა გადავხედოთ ფოლდერებს და წავშალოთ აუთლაიერები, ან ცუდი სურათები