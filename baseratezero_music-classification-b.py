from os import listdir
from numpy import array, mean, cov, append, linalg, dot, transpose, zeros, argmin
from random import shuffle
dataDir = "../input/music_data/"
genres = ['classical', 'country', 'jazz', 'pop', 'rock', 'techno']

# Step 1: load data into a list of list representation - each song is a dictionary
print("Loading Data...")

data = list()
for g in range(len(genres)):
    genreDir = dataDir + "/" + genres[g]
    data.append(list())
    sFiles = listdir(genreDir)
    for s in range(len(sFiles)):

        sFile = genreDir + "/" + sFiles[s]
        f = open(sFile)
        lines = f.readlines()
        meta = lines[0].replace("#", "").split("-")
        songDict = {'file': sFiles[s], 'song': meta[0].strip(), 'artist': meta[1].strip()}

        # read in matrix of values starting from second line in data file
        mat = list()
        for i in range(1, len(lines)):
            vec = lines[i].split(",")
            # cast to floats
            for j in range(len(vec)):
                vec[j] = float(vec[j])
            mat.append(vec)

        songDict['featureMat'] = array(mat)
        data[g].append(songDict)
# get the first genre
classical = data[1]
# list of dicts, each dict is one song
for song in classical[0:3]:
    print(song['song'] + ' by ' + song['artist'] + '. Features: ' + str(song['featureMat'].shape))
# Split data into training and test sets
dSize = len(data[0])
rIdx = list(range(dSize))
# shuffles the contents of rIdx
shuffle(rIdx)
trainIdx = rIdx[0:int(dSize * 0.8) + 1]
testIdx = rIdx[int(dSize * 0.8):]
print('Train : ' + str(trainIdx))
print('Test : ' + str(testIdx))
# Step 2 - Learning Gaussian Model for each Genre
print('Learning Gaussian Model for each Genre')
gModel = list()
# One Gaussian for genre
for g in range(len(genres)):
    # Training data for this genre
    trainMat = data[g][trainIdx[0]]['featureMat']
    for i in range(1, len(trainIdx)):
        trainMat = append(trainMat, data[g][trainIdx[i]]['featureMat'], axis=0)
    gModel.append({'mean': mean(trainMat, 0), 'cov': cov(trainMat, rowvar=False),
                   'icov': linalg.inv(cov(trainMat, rowvar=False))})
first_model = gModel[0]

print('Mean vector: ' + str(first_model['mean']))
print('Covariance : ' + str(first_model['cov']))
# Step 3: Calculating Average Unnormalized Likelihood for each test song and genre model
print('Calculating UNLL')
meanUNLL = zeros((len(genres), len(testIdx), len(genres)))
# where we'll store our final guess based on the prediction of the models
guess = zeros((len(genres), len(testIdx)))
# go over all the genres
for gs in range(len(genres)):
    # get the test songs for this genre
    for t in range(len(testIdx)):
        # the song in question
        ts = testIdx[t]
        # get the data for this song
        x = data[gs][ts]['featureMat']
        # the length r varies between songs (different number of segments)
        [r, c] = x.shape
        # compute the likelihood of this song occurring given each of the genres / Gaussians
        for m in range(len(genres)):
            unll = zeros((r, 1))
            # for each of the segments, compute UNLL
            # un-normalized negative log likelihood (UNLL) is:
            # UNLL = (x - mean_genre) * inverse(cov_genre) * transpose(x - mean_genre)
            for v in range(r):
                # x - mean_genre
                diff = (x[v] - gModel[m]['mean'])
                # times inverse covariance
                res = dot(diff, gModel[m]['icov'])
                # times transposed difference
                res = dot(res, transpose(diff))
                unll[v] = res
            # compute the mean over segments for this genre
            meanUNLL[gs][t][m] = mean(unll)
        # our final guess: the genre that minimizes the average UNLL
        guess[gs][t] = argmin(meanUNLL[gs][t])
print('The guess for the first song: ' + str(int(guess[0][0])))
if guess[0][0] == 0.0:
    print('This is correct.')
else:
    print('This guess was false')
# Step 4 Evaluate Results
print('Evaluating')
[rows, cols] = guess.shape
# Confusion matrix 
confMat = zeros((rows, rows))
# number of correct guesses
nCorr = 0
for r in range(rows):
    for c in range(cols):
        # cast to int here because predictions are floats
        value = int(guess[r][c])
        confMat[value][r] += 1
        if guess[r][c] == r:
            nCorr += 1
# compute the accuracy
acc = float(nCorr) / (rows * cols)

print('Confusion matrix :')
print(str(confMat))
print('Accuracy :' + str(acc))