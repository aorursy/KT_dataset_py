from sklearn import svm
from sklearn import decomposition
import numpy as np
def readFile(url, type='train'):
    with open(url, 'r') as reader:
        reader.readline()
        if type=='train':
            trainLabel = []
            trainData = []
            for line in reader.readlines():
                data = list(map(int, line.rstrip().split(',')))
                trainLabel.append(data[0])
                trainData.append(data[1:])
            return trainData, trainLabel
        elif type=='test':
            returnData = []
            for line in reader.readlines():
                data = list(map(int, line.rstrip().split(',')))
                returnData.append(data)
            return returnData
        else:
            return []
trainData = '../input/train.csv'
testData = '../input/test.csv'
trainData, trainLabel = readFile(trainData, type='train')
trainLabel = np.array(trainLabel)
trainData = np.array(trainData)
pca = decomposition.PCA(n_components=100, whiten=True)
pca.fit(trainData)
trainData = pca.transform(trainData)
svc = svm.SVC()
svc.fit(trainData, trainLabel)
testData = readFile(testData, type='test')
testData = np.array(testData)
testData = pca.transform(testData)
predict = svc.predict(testData)
with open('predict.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0 
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')
print('Done!')
