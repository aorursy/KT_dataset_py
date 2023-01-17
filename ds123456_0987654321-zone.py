# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

file = []

file = os.listdir('/kaggle/input/deepfake-detection-challenge/train_sample_videos/')

#file = os.listdir('/kaggle/input/deepfake-detection-challenge/test_videos/')

#file.remove["metadata.json"]

similar = []

frame1 = []

dissimilar = []

for i in file:

    if (i =='metadata.json'):

              file.remove('metadata.json')

#print(file)

for i in file:

        file2 = []

        file1 = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'+i;

        #file1 = '/kaggle/input/deepfake-detection-challenge/test_videos/'+i;

        file2.append(file1)

#         /*print(file2)*/ 

        import cv2

        j=0;

        for j in file2:

                    cap = cv2.VideoCapture(j)

                    #print(cap)

                    #frame=[]

                    _, frame = cap.read()

                    frame1.append(frame)

                    

#print(frame1)

#                     
#print(file)
import pandas as pd



data1 = pd.DataFrame(frame1)
#print(data1)
data1.columns = ['col']
#print(data1)
#data1
rowvalues =[]

similar=[]

dissimilar=[]

x =-1

y=1

for y in range(400):

    #if x<400:

    #print(data1)

                        x =x+1;

                        y = y+1;

                        #print(x,y)

                        z = data1.col[x:y]

                        #dupes = [a for n, a in enumerate(z) if a in z[:n]]

                        #no_dupes = [a for n, a in enumerate(z) if a not in z[:n]]

                        #print(z)

                        for i in z:

                            #print(k)

                            #dupes = [a for n, a in enumerate(i) if a in i[:n]]

                            #no_dupes = [a for n, a in enumerate(i) if a not in i[:n]]

                            if (i == i+1).any():

                            

                                        #print(i)

                                        similar.append(i)

                                        #print("Similar"+i)

                                        #print(i)

                                        #print(similar)

                                        #print(similar.append(i))

                                        #print(similar.append(i), file=open('/Users/debopriyosanyal/Desktop/16071982/op.log', 'w'))

                                        #print(similar)

                            else:

                                    

                                        dissimilar.append(i)

                                        #print("dissimilar"+i,file=open('/Users/debopriyosanyal/Desktop/16071982/op1.log', 'w'))

                                        #print("dissimilar")

                                        #fileOut.write()

                                        #print(dissimilar.append(i))

                                        #print(dissimilar.append(i), file=open('/Users/debopriyosanyal/Desktop/16071982/op1.log', 'w'))

                                        #print(dissimilar)

                                    

                                    

                            #print(similar)

                            #print(dissimilar)

                        if similar:

                                rowvalues.append(x,y)

                        #x = x+1

                        #print(x)

                        #y = y+2

                        #print(y)
print(rowvalues)
duplicate = []

if not rowvalues:

    df = pd.DataFrame({'filename':file})

    #print (df)

    df['label'] = df.apply(lambda x: 0, axis=1)

    print (df)

    df = df.sort_values('filename')

    print(df)

    df = df[df.filename != 'metadata.json']

    print(df)

    df.to_csv("submission.csv", encoding='utf-8', index=False)

    

else:



    df1 = pd.DataFrame(rowvalues)

    df1.columns = ['X','Y']

    A= df1.col['Y']

    for i in A:

       G= df.iloc[i,'filename']

       duplicate.append(G)

    dffake = pd.DataFrame(G)

    dffake.columns = ['filename']

    dffake['label'] = dffake.apply(lambda x: 1, axis=1)

    dffake = dffake.sort_values('filename')

    dffake.to_csv("submission.csv", encoding='utf-8', index =False)

    