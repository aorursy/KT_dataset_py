!ls ../input
import os
import numpy as np
import json


"""

    VERSION 2

"""


def createSubmission(filename,coords,classes,test_directory = '../input/test/test/',
                     multi_submisison_file="multi_test_xy.json"):
    """
        @brief Function for creating a submission    
        @param filename filename for the competition, without ending
        @param coords predicted x,y  coorinates
        @param classes predicted classes in categorical encoding
        @param multi_submission_file file holding ground truth to data where, 
        are several traffic signs
        @param test_directory directory where the test images are located
    """
    
    # check the data
    if coords.shape != (225,2):
        raise ValueError('coords must have shape (225,2)')
 
    if classes.shape != (225,31):
        raise ValueError('classes must have shape (225,31)') 
    
    
    with open("../input/" + multi_submisison_file,"r") as f:
        gt = json.load(f)
        
    multi_files = []
    for entry in gt:
        multi_files.append(entry['filename'])
    
    
    files = os.listdir(test_directory)
    
    with open(filename + '.csv','w') as f:
        
        f.write('Nr,X,Y,' + ",".join(['C' + str(i) for i in range(0,31)])+'\n')
    
        for n,coords,id_ in zip(classes,coords,files):
            
            
            if id_ in multi_files:
                print("this is a file already located....")
                
                # get indices 
                idx = multi_files.index(id_)
                
                
                
                x_coords = float(gt[idx]['x_centroid']) / 1360
                y_coords = float(gt[idx]['y_centroid']) / 1080
                label = int(gt[idx]["class"])
            
                d = np.zeros(shape=(31,))
                d[label] = 1.0
                
                class_ = str(d.tolist()).replace('[','').replace(']','').replace(' ','')
                string = str(id_) + ',' + str(x_coords) + ',' + str(y_coords) + ',' + str(class_) + '\n'
                
                f.write(string)
        
            else:
            
            
            
            # set the maxium class to 1 
                d = np.zeros(shape=(31,))
                d[np.argmax(n[0])] = 1.0
    
                class_ = str(d.tolist()).replace('[','').replace(']','').replace(' ','')
                x_coords = coords[0] 
                y_coords = coords[1]
                
                string = str(id_) + ',' + str(x_coords) + ',' + str(y_coords) + ',' + str(class_) + '\n'
               
                    
                f.write(string) 
                
            
            
# random prediction
classes =np.zeros(shape=(225,31))

createSubmission("MULTI_BEFORE_submission.csv",np.random.rand(225,2),classes)