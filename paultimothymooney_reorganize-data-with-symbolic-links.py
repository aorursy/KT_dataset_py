import os
print(os.listdir("../input"))
print(os.listdir("../input/train"))
from sklearn.model_selection import train_test_split
PATH = "../input/"
root_prefix = PATH
train_filenames = os.listdir('%s/train/' % (root_prefix))
print("Sample of Training Data:", train_filenames[0:10])
test_filenames  = os.listdir('%s/test/'  % (root_prefix))
print("\nSample of Testing Data:", test_filenames[0:10])
my_train = train_filenames
my_train, my_cv = train_test_split(train_filenames, test_size=0.1, random_state=0)
print("Number of Training Images:",len(my_train))
print("Number of Testing Images:", len(my_cv))
import shutil
from pathlib import Path
# Make symlinks
!cp -as "$(pwd)/../input/" "$(pwd)/COPY"
root_prefix = 'COPY'

def remove_and_create_class(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    os.mkdir(dirname+'/cat')
    os.mkdir(dirname+'/dog')

remove_and_create_class('%s/train' % (root_prefix))
remove_and_create_class('%s/valid' % (root_prefix))

for filename in filter(lambda x: x.split(".")[0] == "cat", my_train):
    os.symlink('%s/train/' % (root_prefix)+filename, '%s/train/cat/' % (root_prefix)+filename)
for filename in filter(lambda x: x.split(".")[0] == "dog", my_train):
    os.symlink('%s/train/' % (root_prefix)+filename, '%s/train/dog/' % (root_prefix)+filename)
for filename in filter(lambda x: x.split(".")[0] == "cat", my_cv):
    os.symlink('%s/train/' % (root_prefix)+filename, '%s/valid/cat/' % (root_prefix)+filename)
for filename in filter(lambda x: x.split(".")[0] == "dog", my_cv):
    os.symlink('%s/train/' % (root_prefix)+filename, '%s/valid/dog/' % (root_prefix)+filename)
PATH = 'COPY'
print(os.listdir('COPY/train'))
print(os.listdir('COPY/valid'))
print(os.listdir('COPY/valid/cat'))
# Remove symlinks before committing
!rm -rf "$(pwd)/COPY"