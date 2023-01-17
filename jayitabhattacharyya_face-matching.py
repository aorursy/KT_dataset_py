!pip install face_recognition
import fnmatch
import face_recognition as fr
import os
from matplotlib import pyplot as plt
%matplotlib inline
face = fr.load_image_file('../input/face-match/trainset/0001/0001_0000255/0000001.jpg')
plt.imshow(face)
plt.show()
script = fr.load_image_file('../input/face-match/trainset/0001/0001_0000255/0001_0000255_script.jpg')
plt.imshow(script)
plt.show()
faceLoc = fr.face_locations(face)    #locate where face is in picture
encodes = fr.face_encodings(face, faceLoc)  #apply face encoding 

encode = fr.face_encodings(script)[0]    #apply encoding to test image

matches = fr.compare_faces(encodes, encode) #match the two images and check same person or not

print(matches)
encodesCurFrame = []
encode = []
matches = []
i = 0

for root,_,files in os.walk('../input/face-match/trainset/'):
    t=0
    f=0
    for filename in files: 
        matches = []
        file = os.path.join(root,filename)
        if fnmatch.fnmatch(file,'*script*'):
            label = file
            #print("label=",label)
            test = fr.load_image_file(label)        
            encode = fr.face_encodings(test)[0]
                    
        else:
            image = file
            #print("image=",image)
            img = fr.load_image_file(image)
            facesCurFrame = fr.face_locations(img)
            encodesCurFrame = fr.face_encodings(img, facesCurFrame)
        
        #print(file)
    matches = fr.compare_faces(encodesCurFrame, encode)  
    if matches == []:
        continue
    else:      
        i+=1
        print(matches,i)
        for m in matches:
            if m==True:
                t+=1
            else:
                f+=1

        print("acc =", t/(t+f))
       
     

j=0
for root,_,files in os.walk('../input/face-match/trainset/'):   
    for filename in it:
        print(os.path.join(root,filename))
    j=j+1
print(j)        
            
for root,_,files in os.walk('../input/face-match/trainset/'):
    it = iter(files)
    next(it, None)  # skip first item.
    next(it, None)
    for filename in it:
        print(os.path.join(root,filename))
            

