import os
os.getcwd()
os.path.relpath('/kaggle/working','/kaggle')
os.path.isabs('/kaggle/working')
os.path.join('kaggle','input')
os.chdir('/kaggle')
os.getcwd()
os.listdir()
os.path.abspath('Temp.txt')
print(os.path.isabs(r'../input'))
print(os.path.isabs(r'/kaggle/input'))
print(os.path.dirname(r'/kaggle/input/Temp.txt'))
print(os.path.basename(r'/kaggle/input/Temp.txt'))
print(os.path.exists(r'/kaggle/input'))
print(os.path.exists(r'/kaggle/output'))
print(os.path.exists(r'/kaggle/input/Temp.txt'))
print(os.path.isfile(r'/kaggle/input'))
print(os.path.isdir(r'/kaggle/input'))
print(os.path.isfile(r'/kaggle/input/Temp.txt'))
print(os.path.getsize('/kaggle/input/Temp.txt'))
print(os.path.getsize('/kaggle/input'))
TotalFileSizeinFolder = 0 
Ppath = r'/kaggle/input'
for filename in os.listdir(Ppath):
    if not os.path.isfile(os.path.join(Ppath,filename)):
        continue
    TotalFileSizeinFolder = TotalFileSizeinFolder + os.path.getsize(os.path.join(Ppath,filename))

print(TotalFileSizeinFolder)
os.mkdir(r'/kaggle/input/ABCD')
os.listdir(r'/kaggle/input')   
