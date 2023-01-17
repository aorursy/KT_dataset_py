import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
#Draw plot from digits 5x5 
def draw_digits(array):
    row = 5
    column = 5
    for i in range(0, row * column):
        plt.subplot(row,column,i + 1)
        mean_digit = array[i]    
        mean_digit = mean_digit.reshape(28,28)
        plt.imshow(mean_digit, cmap="gray")
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.show()

#Normilize images to bin representation. Zero is white, one is black 
def normilizeData(digits):    
    newDigits = np.zeros(digits.shape, dtype=np.int64)
    for i in range(0, digits.shape[0]):
        for j in range(0, digits.shape[1]):
            if(digits[i, j] > 0):
                newDigits[i, j] = 1
    return newDigits

#Evaluate model
def evaluate_model(estimator, test_datas, test_labels):
    prediction = estimator.predict(test_datas)
    accuracy = estimator.score(test_datas, test_labels)
    #print("Accuracy: " + str(accuracy))
    f1_micro = f1_score(test_labels, prediction, average="micro")
    #print("F1 score micro: " + str(f1_micro))
    f1_macro = f1_score(test_labels, prediction, average="macro")
    #print("F1 score macro: " + str(f1_macro))
    return (accuracy, f1_micro, f1_macro)

#Function for creating submission
def competition(estimator, pca):
    data = pd.read_csv("../input/test.csv")
    data = normilizeData(data.as_matrix())
    data = pca.transform(data)
    prediction = estimator.predict(data)
    d = {"ImageId": range(1,data.shape[0]+1), "Label": prediction}
    data_frame = pd.DataFrame(data=d)
    data_frame.to_csv("../output/result.csv", index=False)
data = pd.read_csv("../input/train.csv")
for i in range(0, 10):
    plt.subplot(5,2,i + 1)
    specific_digit = data[data.label == i].iloc[:,1:]
    mean_digit = specific_digit.mean()
    mean_digit = mean_digit.as_matrix().reshape(28,28)
    plt.imshow(mean_digit, cmap="gray")
plt.show()
d = data.iloc[:,1:].as_matrix()
pixels = np.zeros((1,256), dtype=np.int64)
for i in range(0, d.shape[0]):
    for j in range(0, d.shape[1]):
        pixels[0, d[i,j]] += 1
plt.hist(range(0,256), weights=pixels.flatten())
plt.xlabel("Brightness of pixels")
plt.ylabel("Amoun of pixels")
plt.show()
draw_digits(data.iloc[:,1:].as_matrix())
matrix = data.iloc[:,1:].as_matrix()
newDigits = np.zeros(matrix.shape, dtype=np.int64)
for i in range(0, matrix.shape[0]):
    for j in range(0, matrix.shape[1]):
        if(matrix[i, j] > 0):
            newDigits[i, j] = 1
draw_digits(newDigits)
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:].as_matrix(),data.iloc[:,0].as_matrix())
x_train = normilizeData(x_train)
x_test = normilizeData(x_test)
pca_bin = PCA()
pca_bin.fit(x_train)

X = pca_bin.explained_variance_ratio_
Y = np.zeros(X.shape[0])
for i in range(0, X.shape[0]):
    if i==0:
        Y[i] = X[i]
    else:
        Y[i]=Y[i-1] + X[i]
plt.plot(range(0,Y.shape[0]),Y)
plt.show()

#Find the bound after each we save 95% information
indexBound95 = 0
for i in range(0, Y.shape[0]):
    if Y[i] > 0.95:
        indexBound95 = i
        break
print ("Number of PC: " + str(indexBound95 + 1))
print ("Amount of saving information: " + str(Y[indexBound95]))
array = []
for number_of_component in range(1,indexBound95):
    pca_bin = PCA(n_components=number_of_component)
    pca_bin.fit(x_train)
    x_train_transform = pca_bin.transform(x_train)
    svm = SVC()
    svm.fit(x_train_transform, y_train)
    x_test_transform = pca_bin.transform(x_test)
    res = evaluate_model(svm, x_test_transform, y_test)
    array.append(res)
array
accuracy = np.zeros(len(array))
f1_micro = np.zeros(len(array))
f1_macro = np.zeros(len(array))
for i in range(0, len(array)):
    accuracy[i] = array[i][0]
    f1_micro[i] = array[i][1]
    f1_macro[i] = array[i][2]
plt.plot(range(1,indexBound95), accuracy, label="Accuracy")
plt.plot(range(1,indexBound95), f1_micro, label="F1_micro")
plt.plot(range(1,indexBound95), f1_macro, label="F1_macro")
plt.legend()
plt.show()
maxIndex = 0
for i in range(1, 100):
    if f1_micro[i] > f1_micro[maxIndex]:
        maxIndex = i
print("Max f1_micro have " + str(maxIndex + 1) + " PCs")

maxIndex = 0
for i in range(1, 100):
    if f1_macro[i] > f1_macro[maxIndex]:
        maxIndex = i
print("Max f1_macro have " + str(maxIndex + 1) + " PCs")
pca_bin = PCA(n_components=54)
pca_bin.fit(x_train)
x_train_transform = pca_bin.transform(x_train)
svm = SVC()
svm.fit(x_train_transform, y_train)
x_test_transform = pca_bin.transform(x_test)
evaluate_model(svm, x_test_transform, y_test)
competition(svm,pca_bin)