import pandas as pd

sdss_df = pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
sdss_df.head()
sdss_df.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(sdss_df['class'])
sdss_df.drop(['class'], axis=1, inplace=True)

y_encoded = y_encoded.reshape(-1, 1)

enc = OneHotEncoder()
enc.fit(y_encoded)
y_encoded = enc.transform(y_encoded)
y_encoded = y_encoded.toarray().tolist()
scaler = MinMaxScaler(feature_range=(-1, 1))
sdss = scaler.fit_transform(sdss_df).tolist()
import numpy as np

def stitch(inputs, target):
    series_inputs = pd.Series(inputs)
    series_target = pd.Series(target)
    df = pd.DataFrame({'inputs': series_inputs, 'target': series_target})
    return df.values.tolist()

def unstitch(data):
    df = pd.DataFrame(data)
    return df[0].values.tolist(), df[1].values.tolist()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, dropout=False, weightDecay=0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(11, 11) # 2 Input noses, 50 in middle layers
        self.do1 = nn.Dropout(p=0.2)
        self.rl1 = nn.Sigmoid()
        self.fc2 = nn.Linear(11, 3)
        self.do2 = nn.Dropout(p=0.2)
        self.smout = nn.Softmax(dim=1)      
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0)
        self.dropout=dropout
        
        self.cuda()
        
        self.trainOverTimeAccuracy = []
        self.trainOverTimeLoss = []
        self.testOverTimeAccuracy = []
        self.testOverTimeLoss = []
    
    def forward(self, x):
        x = self.fc1(x)
        if self.dropout:
            x = self.do1(x)
        x = self.rl1(x)
        x = self.fc2(x)
        if self.dropout:
            x = self.do1(x)
        x = self.smout(x)
        return x
    
    def log(self, epoch, train, test):
        if epoch % REPORT_RATE == 0:
            self.trainOverTimeAccuracy.append(self.accuracy(train))
            #self.trainOverTimeLoss.append(self.loss(train))
            self.testOverTimeAccuracy.append(self.accuracy(test))
            #self.testOverTimeLoss.append(self.loss(test))
            
    def epochTrain(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self(inputs)
        self.loss = self.criterion(outputs, torch.max(labels, 1)[1])
        self.loss.backward()    
        self.optimizer.step()

        
    def train(self, numEpochs, train_set, train, test):
        inputs, labels = unstitch(train_set)
        
        inputs = Variable(torch.FloatTensor(inputs).cuda())
        labels = Variable(torch.FloatTensor(labels).cuda())

        for epoch in range(numEpochs):
            self.epochTrain(inputs, labels)
            self.log(epoch, train, test)
                
    def activetrain(self, activeUpdateRate, batchSize, numEpochs, train, test):
        trainings = int(numEpochs/activeUpdateRate)
        currentEpoch = 0

        for i in range(trainings):
            activecriterion = nn.CrossEntropyLoss(reduction='none')
            
            inputs, labels = unstitch(train)
            
            inputs = Variable(torch.FloatTensor(inputs).cuda())
            labels = Variable(torch.FloatTensor(labels).cuda())
            
            outputs = self(inputs)

            loss = activecriterion(outputs, torch.max(labels, 1)[1]).detach().cpu().numpy()
            sortIndexs = np.argsort(loss)[::-1]
            dynamicTrainSet = np.array(np.copy(train))[sortIndexs]
            dynamicTrainSet = dynamicTrainSet[0:batchSize]

            inputs, labels = unstitch(dynamicTrainSet)
        
            inputs = Variable(torch.FloatTensor(inputs).cuda())
            labels = Variable(torch.FloatTensor(labels).cuda())
            
            for epoch in range(activeUpdateRate):
                currentEpoch += 1
                self.epochTrain(inputs, labels)
                self.log(currentEpoch, train, test)
            
    def randomtrain(self, batchSize, numEpochs, train, test):
        shuffledTrainSet = np.array(np.copy(train))
        for epoch in range(numEpochs):
            np.random.shuffle(shuffledTrainSet)

            inputs, labels = unstitch(shuffledTrainSet[0:batchSize])

            inputs = Variable(torch.FloatTensor(inputs).cuda())
            labels = Variable(torch.FloatTensor(labels).cuda())
            
            self.epochTrain(inputs, labels)
            self.log(epoch, train, test)

            
    def loss(self, test_set):
        inputs, labels = unstitch(test_set)
        inputs = Variable(torch.FloatTensor(inputs).cuda())
        labels = Variable(torch.FloatTensor(labels).cuda())
        result = self(inputs)
        loss = self.criterion(result, torch.max(labels, 1)[1])

        return loss.item()

    def accuracy(self, test_set):
        inputs, labels = unstitch(test_set)
        result = self(Variable(torch.FloatTensor(inputs)).cuda())
        inputs_max = np.argmax(result.detach().cpu().numpy(), axis=1)
        labels_max = np.argmax(np.array(labels), axis=1)
        correct = np.sum(inputs_max == labels_max)

        return correct/len(test_set)
print(len(sdss), len(y_encoded))
data = stitch(sdss, y_encoded)
from sklearn.model_selection import train_test_split, ShuffleSplit

testSize = 0.2
valSize = 0.2
trainSize = 1.0 - (testSize + valSize)
subPercentage = 0.25

SAMPLES = 30
NUM_EPOCHS = 2000
BATCH_SIZE = 200
ACTIVE_UPDATE_RATE = 1
REPORT_RATE = 10

print(subPercentage)

train, test = train_test_split(data, test_size=testSize, shuffle=True)
train, val = train_test_split(train, test_size=subPercentage , shuffle=True)

print(len(data))
print(len(train), len(test), len(val))
%%time
regularisationBatchNets = []
regularisationMiniBatchNets = []
regularisationActiveNets = []

regularisationSchemes = [[False, 0, 'no regularisation'], [True, 0, 'dropout'], [False, 0.01, 'weight decay'], [False, 0.01, 'dropout & weight decay']]

for scheme in regularisationSchemes:
    print(scheme[2])
    batchNets = []
    miniBatchNets = []
    activeNets = []
    for i in range(SAMPLES):
        batchNet = Net(scheme[0], scheme[1])
        batchNet.train(NUM_EPOCHS, train, train, test)
        batchNets.append(batchNet)

        miniBatchNet = Net(scheme[0], scheme[1])
        miniBatchNet.randomtrain(BATCH_SIZE, NUM_EPOCHS, train, test)
        miniBatchNets.append(miniBatchNet)

        activeNet = Net(scheme[0], scheme[1])
        activeNet.activetrain(ACTIVE_UPDATE_RATE, BATCH_SIZE, NUM_EPOCHS, train, test)
        activeNets.append(activeNet)
        
    regularisationBatchNets.append([batchNets, scheme[2]])
    regularisationMiniBatchNets.append([miniBatchNets, scheme[2]])
    regularisationActiveNets.append([activeNets, scheme[2]])
def buildDataFrame():
    meanTrainAccList = []
    meanTestAccList = []
    meanValAccList = []
    descriptionList = []
    
    padding = ['', '', '', '', '', '', '']
        
    typeList = []
    typeList.extend(['batch'])
    typeList.extend(padding)
    typeList.extend(['random mini batch'])
    typeList.extend(padding)
    typeList.extend(['selective learning'])
    typeList.extend(padding)
    for scheme in regularisationBatchNets:
        trainAccuracys = np.array(list(map(lambda x: x.accuracy(train), scheme[0])))
        meanTrainAccList.append(1.0 - np.mean(trainAccuracys))
        meanTrainAccList.append(np.std(trainAccuracys))
        
        testAccuracys = np.array(list(map(lambda x: x.accuracy(test), scheme[0])))
        meanTestAccList.append(1.0 - np.mean(testAccuracys))
        meanTestAccList.append(np.std(testAccuracys))
        
        valAccuracys = np.array(list(map(lambda x: x.accuracy(val), scheme[0])))
        meanValAccList.append(np.mean(testAccuracys / trainAccuracys))
        meanValAccList.append(np.std(testAccuracys / trainAccuracys))
        
        
        descriptionList.append(scheme[1])
        descriptionList.append('')
        
    for scheme in regularisationMiniBatchNets:
        trainAccuracys = np.array(list(map(lambda x: x.accuracy(train), scheme[0])))
        meanTrainAccList.append(1.0 - np.mean(trainAccuracys))
        meanTrainAccList.append(np.std(trainAccuracys))
        
        testAccuracys = np.array(list(map(lambda x: x.accuracy(test), scheme[0])))
        meanTestAccList.append(1.0 - np.mean(testAccuracys))
        meanTestAccList.append(np.std(testAccuracys))
        
        valAccuracys = np.array(list(map(lambda x: x.accuracy(val), scheme[0])))
        meanValAccList.append(np.mean(testAccuracys / trainAccuracys))
        meanValAccList.append(np.std(testAccuracys / trainAccuracys))
        
        
        descriptionList.append(scheme[1])
        descriptionList.append('')
        
    for scheme in regularisationActiveNets:
        trainAccuracys = np.array(list(map(lambda x: x.accuracy(train), scheme[0])))
        meanTrainAccList.append(1.0 - np.mean(trainAccuracys))
        meanTrainAccList.append(np.std(trainAccuracys))
        
        testAccuracys = np.array(list(map(lambda x: x.accuracy(test), scheme[0])))
        meanTestAccList.append(1.0 - np.mean(testAccuracys))
        meanTestAccList.append(np.std(testAccuracys))
        
        valAccuracys = np.array(list(map(lambda x: x.accuracy(val), scheme[0])))
        meanValAccList.append(np.mean(testAccuracys / trainAccuracys))
        meanValAccList.append(np.std(testAccuracys / trainAccuracys))
        
        
        descriptionList.append(scheme[1])
        descriptionList.append('')
        
    df = pd.DataFrame({'training': typeList, 'regularisation': descriptionList, '$trainError$': meanTrainAccList, '$testError$': meanTestAccList, '$genFactor$': meanValAccList})
    
    return df

table = buildDataFrame()
table_result = table[['training', 'regularisation', '$trainError$', '$testError$', '$genFactor$']]
table_result
#print(table_result.to_latex(index=False, bold_rows=True, na_rep=''))
with open('./resulttable.txt', 'w') as f:
    print(table_result.to_latex(index=False, bold_rows=True, na_rep=''), file=f)
import matplotlib.pyplot as plt

# SMALL_SIZE = 10
# MEDIUM_SIZE = 12

# plt.rc('font', size=SMALL_SIZE)
# plt.rc('axes', titlesize=MEDIUM_SIZE)
# plt.rc('axes', labelsize=MEDIUM_SIZE)
# plt.rcParams['figure.dpi']=150

generations = np.arange(0, NUM_EPOCHS, REPORT_RATE)

plotColors = [
    'b--',
    'r--',
    'g--',
    'k--',
    'g^',
    'k'
]


graphs = [[regularisationBatchNets, 'batch'], [regularisationMiniBatchNets, 'miniBatch'], [regularisationActiveNets, 'active']]
for graph in graphs:
    fig = plt.figure()
    plt.grid(1)
    plt.xlim([0, NUM_EPOCHS])
    plt.ion()
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plots = []
    descriptions = []
    for x, result in enumerate(graph[0]):
        overTimeAccuracy = np.array(list(map(lambda x: x.trainOverTimeAccuracy, result[0])))
        meanOverTimeAccuracy = 1 - np.mean(overTimeAccuracy, axis=0)
        plots.append(plt.plot(generations, meanOverTimeAccuracy, plotColors[x%len(plotColors)] , linewidth=1, markersize=1)[0])
        descriptions.append(result[1])

    plt.legend(plots, descriptions)
    fig.savefig('./' + graph[1] + 'Traning.png')
    plt.show(5)

    plt.close()
fig = plt.figure()
plt.grid(1)
plt.xlim([0, NUM_EPOCHS])
plt.ion()
plt.xlabel('Generations')
plt.ylabel('Fitness')
plots = []
descriptions = []

things = [[regularisationBatchNets, 'batch'], [regularisationMiniBatchNets, 'miniBatch'], [regularisationActiveNets, 'active']]
for x, graph in enumerate(things):
    
    overTimeAccuracy = np.array(list(map(lambda x: x.trainOverTimeAccuracy, graph[0][0][0])))
    meanOverTimeAccuracy = 1 - np.mean(overTimeAccuracy, axis=0)
    plots.append(plt.plot(generations, meanOverTimeAccuracy, plotColors[x%len(plotColors)] , linewidth=1, markersize=3)[0])
    descriptions.append(graph[1])

plt.legend(plots, descriptions)
fig.savefig('./none.png')
plt.show(5)

plt.close()