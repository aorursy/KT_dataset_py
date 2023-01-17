from scipy.io import loadmat
from subprocess import check_output

fileList = check_output(["ls", "../input/train_1"]).decode("utf8")
fileList = fileList.split('.mat\n')
#for item in fileList:

item = fileList[0]

fileName = item

[patient,fileNum,ictalState] = item.split('_')   
mat = loadmat('../input/train_1/'+ fileName +'.mat')
mdata = mat['dataStruct']
mtype = mdata.dtype
ndata = {n: mdata[n][0,0] for n in mtype.names}
data_headline = ndata['channelIndices']

print(data_headline)
csv_headline = 'patient;sampleId;ictalState;ch1;ch2;ch3;ch4;ch5;ch6;ch7;ch8;ch9;ch10;ch11;ch12;ch13;ch14;ch15;ch16\n'
data_raw = ndata['data']

len(data_raw)
f = open(fileName + '.csv','w')
f.write(csv_headline)
i=1

#fn = int(fileNum)

for sample in data_raw:

    line = [str(n) for n in sample]

    line = patient + ';' + str(i) + ';' + ictalState + ';' + ';'.join(line) + '\n'

    f.write(line)

    i = i + 1
f.close()