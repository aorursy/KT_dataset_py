import torch
torch.cuda.device(0)
torch.cuda.get_device_name(0)
torch.cuda.is_available()
# final
import pandas as pd
from sklearn import svm

train_data = pd.read_csv("../input/train.csv")
images = train_data.iloc[:, 1:]
labels = train_data.iloc[:, 0]
images[images>0]=1

clf = svm.SVC(kernel='rbf')
clf.fit(images, labels)

# test_images =  train_data.iloc[-2000:, 1:]
# test_labels = train_data.iloc[-2000:, 0]
# test_images[test_images>0]=1
# clf.score(test_images, test_labels)
from joblib import dump, load
dump(clf, 'svm_rbf.joblib') 
clf = load('svm_rbf.joblib') 
# test_images =  train_data.iloc[-2000:, 1:]
# test_labels = train_data.iloc[-2000:, 0]
# test_images[test_images>0]=1
# clf.score(test_images, test_labels)
test_images = pd.read_csv("../input/test.csv")


test_images[test_images>0]=1
pred_labels = clf.predict(test_images)

# print(test_images.index.values)
df = pd.DataFrame(pred_labels)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)

