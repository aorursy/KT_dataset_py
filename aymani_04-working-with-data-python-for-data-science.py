# Check current working dirctory

!pwd
!mkdir data
!wget -O /kaggle/working/data/Example1.txt https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/labs/example1.txt
example1_path = '/kaggle/working/data/Example1.txt'
file1 = open(example1_path, "r")

print(f'file1 object = {file1}')

print(f'Type of file1 object = {type(file1)}')
file1.name
file1.mode
file1.close()
file1.closed
with open(example1_path,'r') as file1:

    file_contents = file1.read()

    print(f'file_contents \n{file_contents}')

print(file1.closed)    

print(f'file_contents \n{file_contents}')
with open(example1_path,'r') as file1:

    file_contents = file1.readlines()

    print(f'file_contents \n{file_contents}')

print(file1.closed)    

print(f'file_contents \n{file_contents}')
with open(example1_path,'r') as file1:

    file_contents = file1.readline()

    print(f'file_contents \n{file_contents}')

print(file1.closed)    

print(f'file_contents \n{file_contents}')
with open(example1_path,'r') as file1:

    for i,line in enumerate(file1):

        print(f'Line {i+1} contains {line}')
example2_path ='/kaggle/working/data/example2.txt'
with open(example2_path,'w') as file2:

    file2.write('This is line A')
with open(example2_path,'r') as file2:

        print(file2.read())
with open(example2_path,'w') as file2:

    file2.write("This is line A\n")

    file2.write("This is line B\n")

    file2.write("This is line C\n")
lines = ["This is line D\n",

         "This is line E\n",

         "This is line F\n"]

lines
with open(example2_path,'a') as file2:

    for line in lines:

        file2.write(line)
with open(example2_path,'r') as file2:

    print(file2.read())
example3_path ='/kaggle/working/data/Example3.txt'
with open(example2_path,'r') as readfile:

    with open(example3_path,'w') as writefile:

        for line in readfile:

            writefile.write(line)
with open(example3_path,'r') as testfile:    

    print(testfile.read())
testfile.name
import pandas as pd
csv_url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%204/Datasets/TopSellingAlbums.csv'
df = pd.read_csv(csv_url)
df.head()
!wget -O ./data/TopSellingAlbums.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%204/Datasets/TopSellingAlbums.csv
csv_path = '/kaggle/working/data/TopSellingAlbums.csv'
df = pd.read_csv(csv_path)
df.head()
df.iloc[0,0]
df.loc[0,'Artist']
df.loc[1,'Artist']
df.iloc[0:2,0:3]
df.loc[0:2,'Artist':'Released']
df['Released']
len(df['Released'])
df['Released'].unique()
len(df['Released'].unique())
df['Released']>=1980
new_songs = df[df['Released']>=1980]

new_songs
new_songs.to_csv('/kaggle/working/data/new_songs.csv')
with open('/kaggle/working/data/new_songs.csv','r') as songsfile:    

    print(songsfile.read())