import librosa

import matplotlib.pyplot as plt

# from dtw import dtw



#Loading audio files

y1, sr1 = librosa.load('/kaggle/input/sound-sakura/Sakura-Ikimono-Gakari.wav', offset=30, duration=5) 

y2, sr2 = librosa.load('/kaggle/input/sound-sakura/Sakura-Ikimono-Gakari.wav', offset=40, duration=10) 







#Showing multiple plots using subplot

plt.subplot(1, 2, 1) 

mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values

plt.imshow(mfcc1)



plt.subplot(1, 2, 2)

mfcc2 = librosa.feature.mfcc(y2, sr2)

plt.imshow(mfcc2)



# dist, cost, path = dtw(mfcc1.T, mfcc2.T)

# print("The normalized distance between the two : ",dist)   # 0 for similar audios 



# plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')

# plt.plot(path[0], path[1], 'w')   #creating plot for DTW



# plt.show()  #To display the plots graphically
len(y2) / mfcc2.shape[1]
from dtw import dtw