import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库



# separate train set & validation set

from sklearn.model_selection import train_test_split

# import model: KNN

from sklearn.neighbors import KNeighborsClassifier

# review result: confusion matrix, accuracy, classification report

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

train_data.head()



# separate Y_data:labels and X_data:feature data

train_row = min(5000, train_data.shape[0])

X_data = train_data.values[:train_row, 1:]

Y_data = train_data.values[:train_row, 0]

X_test = test_data.values[:]
# Verify image content

def show_img(pixels, label):

    print(label)

    plt.imshow(pixels.reshape((28, 28)))

    plt.show()



show_img(X_data[15], Y_data[15])
x_train, x_valid, y_train, y_valid = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 0)
x_train, x_valid, y_train, y_valid = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 0)



krange = range(1,15)

scores = []

max_score, best_k = 0, 1

for k in krange:

    knn_model = KNeighborsClassifier(n_neighbors = k)

    knn_model.fit(x_train, y_train)

    predictions = knn_model.predict(x_valid)

    

    score = accuracy_score(y_valid, predictions)

    scores.append(score)

    print("K=" + str(k) + ": accuracy:" + str(score), "(best K=" + str(best_k) + ", accuracy_rate=" + str(max_score) + ")")

    # update K and show info when getting better result

    if score > max_score:

        max_score, best_k = score, k

        print(confusion_matrix(y_valid, predictions))

        print(classification_report(y_valid, predictions))

    

    
plt.plot(krange, scores)

plt.ylabel("Accuracy")

plt.xlabel("K")

plt.show()

print("Best K:", best_k)
k_model = KNeighborsClassifier(n_neighbors = best_k)

k_model.fit(X_data, Y_data)



predictions = k_model.predict(X_test)

image_show_shape = (4, 10)

random_idx = np.random.randint(0, len(predictions), image_show_shape)



for row in range(image_show_shape[0]):

    for col in range(image_show_shape[1]):

        img_idx = random_idx[row][col]

        plt.subplot(image_show_shape[0], image_show_shape[1], image_show_shape[1] * row + col + 1)

        plt.subplots_adjust(wspace = 0.2, hspace = 1)

        plt.title(predictions[img_idx])

        plt.axis("off")

        plt.imshow(X_test[img_idx].reshape(28, 28))
pd_result = pd.DataFrame({"ImageID": list(range(1, len(predictions) + 1)), "Label": predictions})

pd_result.to_csv("Digit_Recongizer_Result.csv", index=False, header=True)