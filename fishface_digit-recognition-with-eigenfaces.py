import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

numClasses = 10
numEig = 28*28
picSize = 28*28
#read input data
trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")
trainData.sort_values(by=['label'], inplace=True)
trainY = trainData.iloc[:,0].values
trainX = trainData.iloc[:,1:].values
testX = testData.iloc[:,:].values
#generate eigenface
trainMean = np.mean(trainX, axis=0)
trainX = trainX - trainMean
cov = np.cov(trainX.T)
w, v = np.linalg.eig(cov)
ws = np.sort(w)
ws = ws[::-1]
for i in range(0,numEig):
    v[:,i] = v[:,np.where(w==ws[i])[0][0]]

v = v[:,:numEig].real
#free memory
del trainData, testData, cov, w, ws


#generate weights
omega = np.zeros((numClasses, numEig ,picSize))
for i in range(0,numClasses):
    trainDigit = trainX[np.where(trainY==i)]
    print("calculating weights for digit %d, samples %d" % (i,len(trainDigit)))
    for k in range(0,len(trainDigit)):
        tmp = v.T * trainDigit[k]
        omega[i] += tmp
    omega[i] /= len(trainDigit)
#generate weights
orig = testX[np.random.randint(0, len(testX))]
omega_m = v.T * (orig - trainMean)

#find best match
dist = np.zeros((numClasses))
for i in range(0,numClasses):
    dist[i] = np.linalg.norm(omega[i] - omega_m)
i = dist.argmin()

#reconstruct
recon = v.T * omega_m
recon = np.sum(recon,axis=0) + trainMean
match = v.T * omega[i]
match = np.sum(match, axis=0) + trainMean

#show result
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(orig.reshape(28,28), cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('testX', fontsize=10)

ax2.imshow(recon.reshape(28,28), cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('reconstruct', fontsize=10)

ax3.imshow(match.reshape(28,28), cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('match', fontsize=10)

plt.show()
plt.close()
#make a submission
out_file = open("submission_dr_eigenface.csv", "w")
out_file.write("ImageId,Label\n")

for k in range(0,len(testX)):
    #generate weights
    omega_m = v.T * (testX[k] - trainMean)
    #find best match
    dist = np.zeros((numClasses))
    for i in range(0,numClasses):
        dist[i] = np.linalg.norm(omega[i] - omega_m)
    out_file.write(str(k+1) + "," + str(int(dist.argmin())) + "\n")    
   
out_file.close()