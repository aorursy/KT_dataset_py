import os;
files = os.listdir('/kaggle/input/tomato-cultivars/')
print('Total files -> ', len(files))
files_dict_count = {'01' : 0, '02' : 0,'03' : 0,'04' : 0,'05' : 0,'06' : 0,'07' : 0,'08' : 0,'09' : 0,'10' : 0,'11' : 0, '12' : 0}
for file in files:

    if '01_' in file:

        files_dict_count['01'] = files_dict_count['01'] + 1

    elif '02_' in file:

        files_dict_count['02'] = files_dict_count['02'] + 1

    elif '03_' in file:

        files_dict_count['03'] = files_dict_count['03'] + 1

    elif '04_' in file:

        files_dict_count['04'] = files_dict_count['04'] + 1

    elif '05_' in file:

        files_dict_count['05'] = files_dict_count['05'] + 1

    elif '06_' in file:

        files_dict_count['06'] = files_dict_count['06'] + 1

    elif '07_' in file:

        files_dict_count['07'] = files_dict_count['07'] + 1

    elif '08_' in file:

        files_dict_count['08'] = files_dict_count['08'] + 1

    elif '09_' in file:

        files_dict_count['09'] = files_dict_count['09'] + 1

    elif '10_' in file:

        files_dict_count['10'] = files_dict_count['10'] + 1

    elif '11_' in file:

        files_dict_count['11'] = files_dict_count['11'] + 1

    elif '12_' in file:

        files_dict_count['12'] = files_dict_count['12'] + 1
files_dict_count
import matplotlib.pyplot as plt; import seaborn as sns; sns.set(rc = {'figure.figsize' : (11.7, 8.27)});
sns.set_style('whitegrid')

sns.barplot(x = list(files_dict_count.keys()), y = list(files_dict_count.values()))

plt.show()
files_dict = {'01' : [], '02' : [], '03' : [], '04' : [], '05' : [], '06' : [], '07' : [], '08' : [], '09' : [], '10' : [], '11' : [], '12' : [], '13' : [], '14' : [], '15' : []}
for file in files:

    if '01_' in file:

        files_dict['01'].append(file)

    if '02_' in file:

        files_dict['02'].append(file)

    if '03_' in file:

        files_dict['03'].append(file)

    if '04_' in file:

        files_dict['04'].append(file)

    if '05_' in file:

        files_dict['05'].append(file)

    if '06_' in file:

        files_dict['06'].append(file)

    if '07_' in file:

        files_dict['07'].append(file)

    if '08_' in file:

        files_dict['08'].append(file)

    if '09_' in file:

        files_dict['09'].append(file)

    if '10_' in file:

        files_dict['10'].append(file)

    if '11_' in file:

        files_dict['11'].append(file)

    if '12_' in file:

        files_dict['12'].append(file)

    if '13_' in file:

        files_dict['13'].append(file)

    if '14_' in file:

        files_dict['14'].append(file)

    if '15_' in file:

        files_dict['15'].append(file)

    

    
# Making a data folder in /kaggle/working

os.mkdir('/kaggle/working/data')
# changing the current working directory to the newly created folder

os.chdir('/kaggle/working/data')
os.mkdir('01');os.mkdir('02');os.mkdir('03');os.mkdir('04');os.mkdir('05');os.mkdir('06');os.mkdir('07');os.mkdir('08');os.mkdir('09');os.mkdir('10');os.mkdir('11');os.mkdir('12');os.mkdir('13'); os.mkdir('14'); os.mkdir('15')
DATA_DIR = '/kaggle/input/tomato-cultivars/'
import shutil
OUTPUT_DIR = '/kaggle/working/data/'

for key, values in files_dict.items():

    for value in values:

        

        shutil.copy(DATA_DIR + value, OUTPUT_DIR + key + '/' + value)

    
lst = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15' ]; sum_files = 0

for i in lst:

    sum_files += len(os.listdir('/kaggle/working/data/' + i))

print(sum_files)
!zip -r /kaggle/working/data_tomato.zip /kaggle/working/data