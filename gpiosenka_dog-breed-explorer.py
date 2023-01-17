import os

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg

import cv2
dir=r'/kaggle/input/70-dog-breedsimage-data-set/dog_classes'

dir_list=os.listdir(dir)

print (dir_list)
def print_in_color(txt_msg,fore_tupple,back_tupple,):

    #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple 

    #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)

    rf,gf,bf=fore_tupple

    rb,gb,bb=back_tupple

    msg='{0}' + txt_msg

    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m' 

    print(msg .format(mat))




train_dir=r'/kaggle/input/70-dog-breedsimage-data-set/dog_classes/train'

test_dir=r'/kaggle/input/70-dog-breedsimage-data-set/dog_classes/test'

valid_dir=r'/kaggle/input/70-dog-breedsimage-data-set/dog_classes/valid'

dir_list=os.listdir(train_dir)

print(len(dir_list))

msg='{0:8s}{1:4s}{2:^35s}{1:4s}{3:11s}{1:3s}{4:10s}{1:3s}{5:11s}{6}'

msg=msg.format('Class Id', ' ', 'Bird Species', 'Train Files','Test Files', 'Valid Files','\n')

print_in_color(msg, (255,0,0), (255,255,255))

species_list= sorted(os.listdir(train_dir))

for i, specie in enumerate (species_list):

    file_path=os.path.join(train_dir,specie)

    train_files_list=os.listdir(file_path)

    train_file_count=str(len(train_files_list))

    test_path=os.path.join(test_dir, specie)

    test_files=os.listdir(test_path)

    test_file_count=str(len(test_files))

    #valid_path=os.path.join(valid_dir, specie)

    #valid_files=os.listdir(valid_path)

    #valid_file_count=str(len(valid_files))    

    msg='{0:^8s}{1:4s}{2:^35s}{1:4s}{3:^11s}{1:3s}{4:^10s}{1:3s}{5:^11s}'

    msg=msg.format(str(i), ' ',specie, train_file_count,'10', '10')

    toggle=i% 2   

    if toggle==0:

        back_color=(255,255,255)

    else:

        back_color=(191, 239, 242)

    print_in_color(msg, (0,0,0), back_color)

#print('\33[0m')
test_dir=r'/kaggle/input/70-dog-breedsimage-data-set/dog_classes/test'

classes=len(os.listdir(test_dir))

fig = plt.figure(figsize=(20,100))

if classes % 5==0:

    rows=int(classes/5)

else:

    rows=int(classes/5) +1

for row in range(rows):

    for column in range(5):

        i= row * 5 + column 

        if i>classes:

            break            

        specie=species_list[i]

        species_path=os.path.join(test_dir, specie)

        f_path=os.path.join(species_path, '01.jpg')        

        img = mpimg.imread(f_path)

        img=cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)  

        a = fig.add_subplot(rows, 5, i+1)

        imgplot=plt.imshow(img)

        a.axis("off")

        a.set_title(specie)	