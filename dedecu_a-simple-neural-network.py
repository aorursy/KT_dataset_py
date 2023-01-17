import pandas

train = pandas.read_csv('../input/train.csv')
imgs = train.iloc[:,1:]

i = 0 # image 0

img0 = imgs.iloc[i].values.reshape(28,28)



labels = train.iloc[:,:1]



import matplotlib.pyplot as plot

plot.figure(1, figsize=(5, 5))

plot.imshow(img0, cmap=plot.cm.gray_r, interpolation='nearest')

plot.title(labels.iloc[i,0])

plot.show()
training = train.iloc[:,1:] 
from sklearn.neural_network import MLPClassifier

c = MLPClassifier(solver='lbfgs',activation='logistic', learning_rate_init=0.1, verbose=True)
# training ...

c.fit(training,labels.iloc[:,0])
test = pandas.read_csv('../input/test.csv')



r = c.predict(test)

import pandas as pd

df = pd.DataFrame(r)

df.columns=['Label']

df.index+=1

df.index.name = 'Imageid'

df.to_csv('submission.csv', header=True)