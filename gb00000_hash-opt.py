# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
samp_sub = pd.read_csv('/kaggle/input/hashcode-photo-slideshow/sample_submission.txt')
samp_sub
pet_pics = pd.read_csv('/kaggle/input/hashcode-photo-slideshow/d_pet_pictures.txt')
#pet_pics
#This creates a list of indices of all horizontal pics
hs = []
for i in pet_pics.index:
    if pet_pics.iloc[i].values[0].startswith('H'):
        hs.append(i)
len(hs)
#hs
#This creates a list of all vertical pics indices
vs = []
for i in pet_pics.index:
    if pet_pics.iloc[i].values[0].startswith('V'):
        vs.append(i)
len(vs)
#vs
#This pairs up vertical pics in the order they came in, giving a list of indices
vs2 = []
for i, v in enumerate(vs):
    if i == (len(vs)-1):
        break
    else:
        d = str(vs[i]) + ' ' + str(vs[i+1])
    if i % 2 == 0:
        vs2.append(d)
    else:
        continue
#vs2

len(vs2)
#This puts the two lists together (horizontal and vertical in the order they came in)
#This gets results of less than 200000, maybe shuffle will improve it by chance sometimes
hsvs = np.array(hs + vs2)
hsvs.tolist()[:5]
hsvs.tolist()[-5:]
#np.random.shuffle(pet_pics.index.values)
#pet_pics.index.values
#np.random.shuffle(hsvs)  
#first horizonatal array
pet_pics.values[hs[0]]
#first array pic tags, split into a list
np.array(pet_pics.values[hs[0]][0].split()[2:])
#second horizontal
np.array(pet_pics.values[hs[1]][0].split()[2:])
#tags that are in both the first and second array
np.intersect1d(np.array(pet_pics.values[hs[0]][0].split()[2:]),(np.array(pet_pics.values[hs[1]][0].split()[2:]) ))
#FIRST MINIMUM REQUIREMENT
# number of tags that are in both the first and second array
len(np.intersect1d(np.array(pet_pics.values[hs[0]][0].split()[2:]),(np.array(pet_pics.values[hs[1]][0].split()[2:]) )))
#shows which array is shorter, the first or second horizontal
np.min([len(np.array(pet_pics.values[hs[0]][0].split()[2:])),len(np.array(pet_pics.values[hs[1]][0].split()[2:]))])
# SECOND AND THIRD MINIMUM REQUIREMENTS
# The minimum of the number of tags in one but not the other (for the first and second horizontal tag arrays)
np.min([len(np.array(pet_pics.values[hs[0]][0].split()[2:])),len(np.array(pet_pics.values[hs[1]][0].split()[2:]))]) - len(np.intersect1d(np.array(pet_pics.values[hs[0]][0].split()[2:]),(np.array(pet_pics.values[hs[1]][0].split()[2:]))))
#function giving the number of common tags in two tag arrays, indices are parameters required (single not pairs)
def common(s1,s2):
    return len(np.intersect1d(np.array(pet_pics.values[s1][0].split()[2:]),(np.array(pet_pics.values[s2][0].split()[2:]) )))
#number of common tags in the first two horizonal tag arrays (selecting the first two indices)
common(hs[0],hs[1])
#function that gives the minimum of the number of tags in one but not the other, indices are required parameters (single, not pairs)
def mindif(s1,s2):
    return np.min([len(np.array(pet_pics.values[s1][0].split()[2:])),len(np.array(pet_pics.values[s2][0].split()[2:]))]) - len(np.intersect1d(np.array(pet_pics.values[s1][0].split()[2:]),(np.array(pet_pics.values[s2][0].split()[2:]))))
mindif(hs[0],hs[1])
#ALL THREE REQUIREMENTS MINIMUM
#This gives the minimum of common tags or the number of tags in one but not the other, single indices required parameters
def mincomdif(s1,s2):
    return min(common(s1,s2),mindif(s1,s2))
mincomdif(hs[0],hs[1])
#this would find the maximum minimum number match for a single index in the horizontal arrays and return the index for that match
def findmaxmatch(s1):
    strt = 0
    maxid = 0
    for i in range(len(hs)-1):
        mcd = mincomdif(hs[s1],hs[i])
        if mcd > strt:
            maxid = i
            strt = mcd
        else:
            continue
    return maxid
#find the best match for the first horizontal array
findmaxmatch(hs[0])
#the minimum of the number of common tags or tags not both in the first horizontal and index 555
mincomdif(hs[0],hs[555])
#first horizontal array
np.array(pet_pics.values[hs[0]][0].split()[2:])
#horizontal array index 555
np.array(pet_pics.values[hs[555]][0].split()[2:])
#This took too long to run
# This would have found the best match in the horizontal index, with a list showing the best match index and the index of what it matched with
'''matchlist = []
bestmatch = 0
for i in range(len(hs)-1):
    maxmatch = findmaxmatch(i)
    if maxmatch > bestmatch:
        bestmatch = maxmatch
        matchwith = i
        
matchlist.append([bestmatch, matchwith])'''
#matchlist
#mincomdif(hs[bestmatch],hs[matchwith])
# make a list for the best match for each horizontal term 
#This took too long to run
'''best_match = []
for i in range(len(hs)-1):
    best_match.append([hs[i],findmaxmatch(hs[i])])'''
#best_match
















sub = pd.DataFrame(hsvs, columns = [str(len(hsvs))] )
sub
sub.to_csv('submission.txt', index=False)



