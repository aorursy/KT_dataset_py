import matplotlib.pyplot as plt

from sklearn import datasets, metrics

from sklearn.linear_model import LogisticRegression
digits = datasets.load_digits()
#打印样例数字

samples = list(zip(digits.images, digits.target))

for id, (img, label) in enumerate(samples[:4]):

    plt.subplot(2, 4, id + 1)

    plt.axis('off')

    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title('Label: %i' % label)

plt.show()    
#有n个图像

n = len(digits.images)

#将二维图像变成一维向量（nx8x8->nx64），方便处理

data = digits.images.reshape(n, -1)
model = LogisticRegression(C = 1e5)
#用前一半数据作为训练数据

model.fit(data[:n//2], digits.target[:n//2])
answer = digits.target[n//2:]

pred = model.predict(data[n//2:])
metrics.confusion_matrix(answer, pred)
samples = list(zip(digits.images[n//2:], pred))

for id, (img, label) in enumerate(samples[:4]):

    plt.subplot(2, 4, id + 1)

    plt.axis('off')

    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title('Predict: %i' % label)

plt.show()    