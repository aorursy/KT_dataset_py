import pandas as pd
import numpy as np
import os
import shutil

dtypes = {
        'id': 'str',
        'url': 'str',
        'landmark_id': 'uint32',
}
train_df = pd.read_csv("../input/train.csv", dtype = dtypes)

'''
Create the subfolders with the given labels from the data frame
'''
def create_folders():
    categories = np.unique(train_df.landmark_id.values)
    for i in categories:
        folder = r"../input/train/" + str(i)
        if not os.path.exists(folder):
            print('created folder', i)
            os.mkdir(folder)
        else:
            print( str(i), ' exists!')
'''
function to move the images to their corresponding folder
'''
def move_files():
    failed = 0
    for i, row in train_df.iterrows():
        filename = r"../input/train/{}/{}.jpg".format(row.landmark_id, 
                                      row.id )
        oldfile = r"../input/train/{}.jpg".format(row.id )
        if not os.path.exists(filename):
            try:
                os.rename(oldfile, filename)
                print('moved {}.jpg to {}'.format(row.id, row.landmark_id))
            except:
                failed +=1
        else:
            print('{}.jpg is in {}'.format(row.id, row.landmark_id))
    
    print('failed on {} files'.format(failed))

'''
function to create the training validation folder
'''
def create_test_folder():
    failed = 0
    val_size = int(len(train_df)*0.1)
    val_folder = r"../input/train_val"
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)
        
    for i, row in train_df.iloc[:val_size].iterrows():
        filename = r"../input/train/{}/{}.jpg".format(row.landmark_id, 
                                      row.id )
        newFile = r"../input/train_val/{}/{}.jpg".format(row.landmark_id, 
                                      row.id )
        folder = r"../input/train_val/{}".format(row.landmark_id )
        print('testing {}.jpg in {}'.format(row.id, row.landmark_id))
        if not os.path.exists(newFile):
            if not os.path.exists(folder):
                os.mkdir(folder)
                print('created folder', folder)
            try:
                shutil.copy2(filename, newFile)
                print('copied {}.jpg to train_val/{}'.format(row.id, row.landmark_id))
            except:
                failed +=1
        else:
            print('{}.jpg is in {}'.format(row.id, row.landmark_id))
    print('failed on {} files'.format(failed))
'''
For the test folder, we need to create a dummy subfolder. In this case, we created 0
'''
def create_test():
    count = 0
    folder = r"../input/test"    
    
    if not os.path.exists(folder + r"/0"):
            os.mkdir(folder + r"/0")
    
    for root, dirs, files in os.walk(folder):
        if '0' in dirs:
            for f in files:
                oldfile  = os.path.join(root, f)    
                newfile = '{}/0/{}'.format(root,f)
                os.rename(oldfile,newfile)
                count +=1
                if count % 20 == 0:
                    print('moved {} to {}'.format(oldfile,newfile))