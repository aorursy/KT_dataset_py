!pip install /kaggle/input/20-faces-demo/mtcnn/mtcnn-0.1.0-py3-none-any.whl
from mtcnn import MTCNN

import cv2

import os

import numpy as np

from tqdm import tqdm



def check_dir(dir, kaggle):

    # checks is a directory exists. If it does not it is created

    # if it exists it is checked to see if it contains files

    # it is contains files gives you the option to just continue, or to Quit or to delete the files

    if os.path.isdir(dir)==False:

        #print('the  {0} does not exist - creating it for you'.format(dir))

        os.makedirs(dir)

        return True    

    else:

        dir_list=os.listdir(dir)

        if len(dir_list)==0:

            return True

        if kaggle==True:

            for f in dir_list:

                f_path=os.path.join (dir, f)

                if os.path.isfile(f_path):

                    os.remove(f_path)

            return True

        else:

            try_again=True

            while try_again:

                msg='directory {0} contains files. Enter C to contine, enter Q to Quit, or enter D to delete files'

                ans=input(msg.format(dir)) 

                if ans=='C' or ans=='c':

                    return True                

                elif ans=='D' or ans=='d':

                    count=0

                    for f in dir_list:

                        f_path=os.path.join(dir,f)

                        if os.path.isfile(f_path):

                            os.remove(f_path)

                            count=count +1

                    print('{0} files were removed from {1}'.format(count, dir), flush=True)                    

                    return True

                elif ans=='Q' or ans=='q':

                    print('**************Program has Terminate************')

                    return False

                else:

                    print('your entry {0} was not a C, or a D or a Q enter your response again'.format(ans))

                

            

def check_for_files(dir):

    dir_list=os.listdir(dir)

    if len(dir_list)>0:

        return True

    else:

        print('directory {0} is empty. program wil terminate'.format(dir))

        return False

    

def check_extension(ext, ext_list):

    #checks if the extension of a file is in a list of extensions

    if ext.lower() not in ext_list:

        msg=' the extension you entered {0} is not in {1} program will terminate'

        print(msf.format(ext, ext_list))

        return False

    

def get_paths(dir, ext_list):

    # checks a directory to see if it contains files

    # if it contains files it extracts the files extension and determines if it is in an ext_list

    # if the extension for a file is in the list the file is appended to the f_paths list

    # if  the extension is not in the list the file name is appended to the bad list

    f_paths=[]

    bad=[]

    dir_list=os.listdir(dir)

    for f in dir_list:

        f_path=os.path.join(dir,f)        

        if os.path.isfile(f_path):

            # get the file extension

            index=f.rfind('.')

            ext=f[index+1:]                            

            if ext in ext_list:

                f_paths.append(f_path)

            else:

                bad.append(f_path)                

    if len(f_paths)==0:

        print('there are no files in {0} that can be processed.Program will terminate'.format(dir))

        return (False, None, bad)

    else:

        return ( True,f_paths, bad)

    

def rotate_img(f):

    # used as part of the crop image process it reads in an image, detects the bounding boxes for the

    #faces detected in the image and then selects the face with the largest number of pixels.

    #  for the largest faces the eye centers are detected and the angle of the eyes with respect to

    # the horizontal axis is determined. It then provides this angle to the rotate_bound function

    # the rotate_bound function the rotates the image so the eyes are parallel to the horizontal axis

    detector = MTCNN()

    img=cv2.imread(f)

    data=detector.detect_faces(img)

    #y=box[1] h=box[3] x=box[0] w=box[2]

    biggest=0

    if data !=[]:

        for faces in data:

            box=faces['box']            

            # calculate the area in the image

            area = box[3]  * box[2]

            if area>biggest:

                biggest=area

                bbox=box                

                keypoints=faces['keypoints']

                left_eye=keypoints['left_eye']

                right_eye=keypoints['right_eye']                 

        lx,ly=left_eye        

        rx,ry=right_eye

        dx=rx-lx

        dy=ry-ly

        tan=dy/dx

        theta=np.arctan(tan)

        theta=np.degrees(theta)    

        img=rotate_bound(img, theta)        

        return (True,img)

    else:

        return (False, None)

    

    

def crop_image(f,code):

    detector = MTCNN()

    if code:

        img=cv2.imread(f)

    else:

        img=f                

    data=detector.detect_faces(img)

    #y=box[1] h=box[3] x=box[0] w=box[2]

    biggest=0

    if data !=[]:

        for faces in data:

            box=faces['box']            

            # calculate the area in the image

            area = box[3]  * box[2]

            if area>biggest:

                biggest=area

                bbox=box 

        bbox[0]= 0 if bbox[0]<0 else bbox[0]

        bbox[1]= 0 if bbox[1]<0 else bbox[1]

        img=img[bbox[1]: bbox[1]+bbox[3],bbox[0]: bbox[0]+ bbox[2]]

        return (True, img) 

    else:

        return (False, None)

    

def rotate_bound(image, angle):

    #rotates an image by the degree angle

    # grab the dimensions of the image and then determine the center

    (h, w) = image.shape[:2]

    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the angle to rotate clockwise), then grab the sine and cosine

    # (i.e., the rotation components of the matrix)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    cos = np.abs(M[0, 0])

    sin = np.abs(M[0, 1]) 

    # compute the new bounding dimensions of the image

    nW = int((h * sin) + (w * cos))

    nH = int((h * cos) + (w * sin)) 

    # adjust the rotation matrix to take into account translation

    M[0, 2] += (nW / 2) - cX

    M[1, 2] += (nH / 2) - cY 

    # perform the actual rotation and return the image

    return cv2.warpAffine(image, M, (nW, nH))  

    

def resize_image(img, height, width, pres_aspect):

    # resizes an image

    if pres_aspect:

        img_shape=img.shape

        scale=height/img_shape[0]

        new_width=int(img_shape[1]* scale)

        size=(new_width, height)        

        img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

    else:

        size=(width, height)

        img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

    return img



def save_img(img,f_path,output_dir,output_ext):

    # saves an image file to the output_dir labele as file name.output_ext    

    file_name=os.path.basename(f_path)

    index=file_name.rfind('.')

    file_name=file_name[:index]

    file_name=file_name + '.' + output_ext    

    out_path=os.path.join(output_dir,file_name)    

    cv2.imwrite(out_path,img)

    

def find_duplicate(dir,distance=12,hash_length=128): 

    # searches the files within the dir directory and determines is any of the images are

    # duplicates of each other. Returns a dictionary of duplicate images

    hash_dict={}

    f_list=os.listdir(dir)

    for f in f_list:

        f_path=os.path.join(dir,f)

        if os.path.isfile(f_path):     # make sure f is a file not a directory

            hash=get_hash(f_path,hash_length)      # get the 64 bit hash of the file

            hash_dict[f]=hash     # store the hash in a dictionary with key-filename and value=hash

        # now go the dictionary and create a new dictionary of duplicate files

    dup_dict=find_duplicates(hash_dict, distance)        

    return dup_dict

        

def get_hash(fpath, hash_length):

    # generates a difference hash of a specified length of a file

    r_str=''      

    img=cv2.imread(fpath,0)        # read image as gray scale image

    img = cv2.resize(img, (hash_length+1, 1), interpolation = cv2.INTER_AREA)    

    # now compare adjacent horizontal values in a row if pixel to the left>pixel toright result=1 else 0

    for col in range (0,hash_length):

        if(img[0][col]>img[0][col+1]):

            value=str(1)

        else:

            value=str(0)

        r_str=r_str + value

    return r_str



def find_duplicates(hash_dict, distance):

    # runs through a dictionay of file names:hash and determines if an of the files have a hash value

    #that differ from each other by a value less than or equal to distance

    # used to measure if files are duplicates of each other returns a dictionary of the form

    # filename:[list of filenames of duplicate files]

    dup_dict={}

    key_list=[]

    for key,value in hash_dict.items():

        dup_list=[]

        for key2,value2 in hash_dict.items():

            if key != key2 and key not in key_list and key2 not in key_list:

                if match(value,value2,distance):

                    dup_list.append(key2)

                    dup_dict[key]=dup_list

                    if key2 not in key_list:

                        key_list.append(key2)

        if key not in key_list:

            key_list.append(key)

        #print('key= {0}  key2= {1} match= {2} key list = {3}'.format(key,key2,m,key_list))

    return dup_dict



def match(value1, value2, distance):

    # determines is two hash values are within a distance of each other

    length=len(value1)

    mismatch=0

    for i in range(0, length):

        if value1[i] !=value2[i]:

            mismatch = mismatch  + 1

            if mismatch>distance:

                return False

    return True

      



def crop_align(source_dir,output_dir,output_ext,resize=False,align=False,

               pres_aspect=False, height=None, width=None, kaggle=False):

    ext_list=['jpg', 'jpe', 'jpeg', 'png', 'bmp', 'tiff', 'wepb', 'pxm', 'exr']

    f_paths=[]

    no_face_list=[]

    c_count=0

    p_count=0

    status=check_dir(output_dir, kaggle)

    if status==False:

        return (False, None, None)

    status=check_for_files(source_dir)

    if status==False:  

        return (False, None,None)

    status=check_extension(output_ext, ext_list)

    if status==False:

        return (False, None,None)

    status,f_paths, bad_ext_list=get_paths(source_dir, ext_list)

    if status==False:

        return (False, bad_ext_list,None)

    for f in tqdm(f_paths, leave=True, unit='File'):

        p_count=p_count+1

        if align ==False:

            status, img =crop_image(f,True)

        else:

            status, img =rotate_img(f)

            if status==True:

                status, img =crop_image(img,False)        

        if resize==True and status==True:

            img=resize_image(img, height, width, pres_aspect)                    

        if status==True:

            c_count=c_count + 1

            save_img(img,f,output_dir, output_ext)

        else: # no face was detected

            no_face_list.append(f)

    if c_count>0:

        dup_dict=find_duplicate(output_dir,distance=12,hash_length=128)

    else:

        dup_dict={}

    if align:

        if resize:

             msg='{0} files were processed, {1} files were aligned,cropped,resized and saved to {2}'

        else:       

            msg='{0} files were processed, {1} files were aligned then cropped and saved to {2}'

    else:

        if resize:

             msg='{0} files were processed, {1} files were cropped, resized and saved to {2}'

        else:

            msg='{0} files were processed, {1} files were cropped and saved to {2}'

    print(msg.format(p_count, c_count, output_dir))

    

    if c_count==0:

        return(False, bad_ext_list,no_face_list,None)

    else:

        if len(dup_dict)==0:

            print('\nNo duplicates were found in the cropped images')

        else:

            print('\nduplicate images were detected in the cropped images, see list below:')

            print('{0}{1:^20s}{0:3s}{2:}'.format(' ','IMAGE', 'DUPLICATE IMAGES'))

            for key,value in dup_dict.items():

                print('{0}{1:^20s}{0:3s}{2}'.format(' ', key, value))

    if len(bad_ext_list)==0:

        print('\nThere were no improper file extension found')

    else:

        print('\nThe following files had invalid file extensions and were not processed:')

        for file in bad_ext_list:

            print('   ', file)

                  

    return (True, bad_ext_list, no_face_list, dup_dict )
"""

Created on Mon Jan 27 13:27:57 2020



@author: Gerry Piosenka

This function operates on image files. I you download images files from an internet search

you will notice the files often have very long and un-weildy names often with strange characters

in the name. Also these files may come in a multitude of image formats as denotd by their

extensions. After I download theseimages I find it convenient to rename the files and generally to convert them all to a common image format. This function achieves that purpose. It will

nename the files in a numerical order starting with a number you specificy. If you specify an extension call files will be converted to the image format specified by the extension.

On another point the naming of the files to numerical file names includes what is called "zeros"

padding. Unlike windows, files in most python functions read files alphanumerically, For example is you have files in a directory with names 1.jpg, 2.jpg, ... 9.jpg, 10,jpg the order in which

the will be processed is not the numeric orderbut the alphabetic order. Thus they will be 

processed in the order 1.jpg, 10.jpg, 2.jpg etc. The use of "zeros' padding will prevent this.

With zero padding the files are renamed 01.jpg, 02.jpg .. 09.jpg, 10.jpg so the are processed in

the order you would expect.

Use:

    1- place the images you want processed in a single directory. I call it the source_dir.

       place ONLY image files in the directory. NOTE this function use the cv2 module so

       file types are limited to having extensions 'jpg', 'jpe', 'jpeg', 'png', 'bmp', 'tiff','wepb', 'pxm', 'exr', 'jfif'.

    2 create a python environment with the module cv2 (OpenCV) installed

    3 run  rename (dir, snum=1, new_ext=None) where:

        dir is a string denoting the full path to the directory containing the image files

        snum is an integer (default value=1) that defines the starting number for the

            numerical renaming of the files

        new_ext is a string (default-None)denoting the file format all image files will be

            converted to. If not specifies the files will be renumber sequentially but each

            file will retain its original extension.

            

Processing:

        1 The function first checks to see if the dir exists and that it contains files.

          If not an error message is printed and the program terminates

        2 Next the files in dir are checked to verify the file extensions are in the list

          noted above. If files are found with invalid extensions, a sub directory called

          "bad_ext" is created and the file with the improper extension is removed from

          dir and stored in the sub directory. A bad_list is created and the filenames

          having improper extensions are stored in that list and provided as part of the

          functions return tupple.

        3 Next each of the remaining files in dir are processed sequentially. CV2 is used

          to read in the image file. If an error is encounter the image file is corrupted

          and can not be processed.The file name is added to the bad_list and the file is

          removed from dir and placed into the "bad_ext subdirectory

        4 Now the dir directory only  contains image files that can be processed. These

          files are then renamed sequentially start from snum.

        5 If you specified n_ext, are files are then read in by cv2 then written back into

          the dir directory with the new image formatspecified by n_ext.

          

returns: The function returns a tupple of the form (status, bad_list) where:

           status is a boolean. It is set to true if at least one image file was

               processed to completetion. Otherwise it is set to False

           bad_list is a list of file names. If files in dir were found to have 

           extensions not in the list above, or if an image file could not be read by cv2

           the file name is included in the list. These files will be stored in the

           sub directory dir\bad ext.

"""



import os

import shutil 

import cv2

def check_directory(dir, ext_list, n_ext):

    f_list=[]

    bad_list=[]

    one_good=False

    save_path=os.path.join(dir,'bad ext')

    if os.path.isdir(dir)==False:

        print('\n The directory {0} does not exist, program  is terminating')

        return (False, None, None)

    if n_ext not in ext_list and n_ext !=None:

        msg='\nthe new extension you specified {0} is not in the list of acceptable extensions - program terminating'

        print(msg.format(n_ext))

        return (False, None, None)

    else:

        dir_list=os.listdir(dir)

        if len(dir_list)==0:

            print('\ndirectory {0} is empty, program is terminating'.format(dir))

            return (False,None,None)

        else:

            for d in dir_list:

                f_path=os.path.join(dir,d)

                if os.path.isfile(f_path):

                    index=f_path.rfind('.')                        

                    ext=f_path[index+1:]

                    if ext.lower() in ext_list:

                        # extensions are valid but are they really image files?

                        # try to read in each file to test it

                        try:

                            img= cv2.imread(f_path)

                            cv2.imwrite( f_path, img)

                            f_list.append(f_path)

                            one_good=True                                                       

                        except (cv2.error ):

                            # file is not a valid image file

                            msg='\nfile {0} is not a valid image file, it wil be removed and saved to {1}'

                            print(msg.format(f_path, save_path ))

                            bad_list.append(f_path) 

                    else:

                        bad_list.append(f_path)

        if one_good==False:

            print('There are no files in {0} that have the proper image extensions'.format(dir))

            return (False, None, bad_list)

        else:

            if len(bad_list)==0:

                return (True,f_list, None)

            else:                

                msg='\n {0} files in {1} have invalid extensions or are not valid image files and will be moved to {2}'

                print(msg.format(len(bad_list), dir, save_path))

                if os.path.isdir(save_path)==False:

                    os.mkdir(save_path)

                for f in bad_list:

                    # extract the file name

                    fsplit=f.split('\\')                    

                    fname=os.path.basename(f)                  

                    test_path=os.path.join(save_path,fname)                    

                    if os.path.isfile(test_path)==True:

                        os.remove(f_path)

                    else:

                        dest = shutil.move(f, save_path)                    

                msg='\n{0} files have been removed from {1} and stored in {2}'

                print(msg.format(len(bad_list),dir, save_path ))

                return (True, f_list, bad_list)

                    





def rename (dir, snum=1, new_ext=None, kaggle=False):

    f_dict={}

    if new_ext !=None:

        new_ext=new_ext.lower()

    ext_list=['jpg', 'jpe', 'jpeg', 'png', 'bmp', 'tiff', 'wepb', 'pxm', 'exr', 'jfif']

    status, f_list, bad_list=check_directory(dir, ext_list, new_ext)

    if status==False:

        return (status, bad_list)

    else:

        if kaggle==True:

            char='Y'

        else:

            char=input ('Warning this function will rename files in {0}\n enter Y to proceed or  N to abort'.format(dir)) 

    if char !="y" and char !='Y':

        print('User terminated execution')

        return (False,None,None)

    else:  #determine the amount of preceeding zeros to pad the file name with to preserve order

        pad=0

        mod = 10

        fc=len(f_list)        

        for i in range(1, fc + 1): # skip i=0 because 0 modulo anything is 0 and we don't want to increment pad

            if i % mod == 0:

                pad=pad+1                    

                mod =mod * 10

        #  make all file names unique by adding a numeric extension in place of real extension

        for i in range(0,fc):

            f=os.path.basename(f_list[i])            

            f_path=os.path.join(dir,f) #full path to the file

            index=f.rfind('.')

            fname=f[:index]

            ext=f[index:]   # this includes the period (.)

            if ext=='.jfif':  # jfif is the same as a jpg file but cv2 and other modules don't process jfif files

                ext='.jpg'

            #to avoid trying to rename a file with a name that is in the directory

            # give each file a new unique numerical extension

            #sore the original extension in a dictionay where the key is the file name

            fnew=fname +'.' + str(i+1)

            new_path=os.path.join(dir, fnew)

            os.rename(f_path, new_path)            

            f_dict[fname]=ext  #key is new file name, value is original extension

            #now all full filenames are unique with unique extensions

        f_list=os.listdir (dir)

        i=0

        for d in f_list:

            d_path=os.path.join (dir,d)

            if os.path.isdir(d_path)==False:

                    full_name=os.path.basename(d_path) 

                    index=full_name.rfind('.')

                    fname=full_name[:index]

                    ext=f_dict[fname]                

                    fname=str(i + snum).zfill(pad+1) + ext

                    i=i+1

                    new_path=os.path.join(dir, fname)

                    shutil.copy(d_path,new_path)                

                    os.remove(d_path)                

                    if new_ext !=None:                                        

                        index=new_path.rfind('.')

                        old_ext=new_path[index+1:]

                        if old_ext !=new_ext:

                            fname=new_path[:index]

                            new_name=fname + '.' + new_ext

                            img= cv2.imread(new_path)                        

                            cv2.imwrite( new_name, img)                        

                            os.remove(new_path)

        if new_ext==None:

            print('{0} files were processed, renamed and saved'.format(len(f_list)))

        else:

             print('{0} files were processed, renamed and saved in {1} format'.format(i, new_ext))

        return (True, bad_list)
import os

d=r'/kaggle/input'

d_list = os.listdir(d)

d1_path=os.path.join(d, d_list[0])

d2_list=os.listdir(d1_path)

input=os.path.join(d1_path, d2_list[0])

d3_path=os.path.join(d1_path, d2_list[1])

d3_list=os.listdir(d3_path)

mtcnn_path=os.path.join(d3_path, d3_list[0])





source_dir=input

output_dir=r'/kaggle/working/results'

output_ext='jpg'

resize=True

align=True

kaggle=True

pres_aspect=True

height=224

width=224

snum=100 # set an arbitray number for start value of numeric file renaming





status, bad_ext_list, no_face_list, dup_dict=crop_align(source_dir,output_dir,output_ext,resize=resize,align=align,

               pres_aspect=pres_aspect, height=height, width=width,kaggle=kaggle)

if status==True:

    # if you do not want to rename the files that were cropped eliminate the line below.

    status, bad_list=rename (output_dir, snum=snum, new_ext=None, kaggle=kaggle)

    



"""import os

import shutil

d=r'/kaggle/working'

d_list=os.listdir(d)

print('d list is ', d_list)

for f in d_list:

    index=f.rfind('.')

    ext=f[index+1:]

    if ext=='jpg':

        f_path=os.path.join(d, f)

        os.remove(f_path) 

r_path=os.path.join (d, "results")

r_list=os.listdir(r_path)

for f in r_list:

    f_path=os.path.join(r_path, f)

    if os.path.isfile(f_path):

        os.remove(f_path)



shutil.rmtree(r'/kaggle/working/results/bad ext')"""