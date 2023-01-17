import os
os.listdir("../input")
# import library
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as implt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
%matplotlib inline
# import data and split label and image
labeled_image = pd.read_csv("../input/train.csv")
label = labeled_image.iloc[0:5000,:1]
image = labeled_image.iloc[0:5000,1:]
# split data into 2 subdata like training dataset and testing dataset
label_train, label_test, image_train, image_test = train_test_split(label, image, test_size=0.2, random_state=0)
# we use support vector machine to classifier
# Note: return `label_train` to 1-D array by ravel() function
clf = svm.SVC()
clf.fit(image_train,label_train.values.ravel())
# defaults parameters of estimator "SVC"
clf.get_params()
pred = clf.predict(image_test)
test = label_test.values.ravel()
print("prediction:", pred[:10])
print("real label:", test[:10])
print("mean accuracy: ", clf.score(image_test, label_test))
# show 9 image:
for i in range(9):
    plt.subplot(191 + i)
    img = image_test.iloc[i].values.reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.title(label_test.iloc[i,0])
def show_detail_digit(data_test, results_of_model, number_of_digit):
    index_of_digit = [results_of_model == number_of_digit][0]
    f, axes = plt.subplots(9,9, sharex=True, sharey=True, figsize=(16,8))
    for r in range(9):
        for c in range(9):
            img = data_test.iloc[index_of_digit].iloc[r*9+c].values.reshape((28,28))
            axes[r,c].imshow(img, cmap='binary')
# show image black white
image_train.iloc[image_train>0] = 1
image_test.iloc[image_test>0] = 1

#show_detail_digit(image_train,label_train.values.ravel(), 3)

for i in range(9):
    plt.subplot(191 + i)
    img = image_train.iloc[i].values.reshape((28,28))
    plt.imshow(img, cmap='binary')
    plt.title(label_train.iloc[i,0])
# We repeat the classification like above to see how's the performance going:
#clf = svm.SVC()
clf.fit(image_train,label_train.values.ravel())

pred = clf.predict(image_test)
test = label_test.values.ravel()

print("prediction:", pred[:10])
print("real label:", test[:10])

print("mean accuracy: ", clf.score(image_test, label_test))
# the image that was be failed to predict

for i in range(9):
    plt.subplot(191 + i)
    img = image_test.iloc[i].values.reshape((28,28))
    plt.imshow(img, cmap='binary')
    plt.title(pred[i], color='red')

estimator = svm.SVC(kernel='rbf')
print("defaults parameters: ", estimator.get_params())
label_train_1D_array = label_train.values.ravel()
param_grid = {
    'C':[1, 5 , 7, 10],
    'gamma': [0.001, 0.01, 0,1]
}
svc_params_selection = GridSearchCV(estimator, param_grid)
svc_params_selection.fit(image_train,label_train_1D_array)
print(svc_params_selection.best_params_)
print(svc_params_selection.best_score_)
print(svc_params_selection.best_estimator_)
pred = svc_params_selection.predict(image_test)
print("prediction:", pred[:10])
print("real label:", test[:10])

print("mean accuracy: ", svc_params_selection.score(image_test, label_test))
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=svc_params_selection.predict(test_data[0:5000])
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('sample_submission.csv', header=True)
# show 10 image results:
f, axes = plt.subplots(9,9, sharex=True, sharey=True, figsize=(16,16))
for i in range(9):
    for j in range(9):
        img = test_data.iloc[i*9+j].values.reshape((28,28))
        axes[i,j].imshow(img, cmap='binary')
        axes[i,j].set_title(results[i*9+j], color='red')
        

# show all of digit number:
show_detail_digit(test_data, results, 9)