import numpy as np

import pandas as pd

from sklearn.datasets import load_breast_cancer



cancer = load_breast_cancer()



#print(cancer.DESCR) # Print the data set description
cancer.keys()
# You should write your whole answer within the function provided. The autograder will call

# this function and compare the return value against the correct solution value

def answer_zero():

    # This function returns the number of features of the breast cancer dataset, which is an integer. 

    # The assignment question description will tell you the general format the autograder is expecting

    return len(cancer['feature_names'])



# You can examine what your function returns by calling it in the cell. If you have questions

# about the assignment formats, check out the discussion forums for any FAQs

answer_zero() 
def answer_one():

    

    df = np.c_[cancer.data, cancer.target]

    columns = np.append(cancer.feature_names, ['target'] )

    

    return pd.DataFrame(df, columns=columns)





answer_one()
def answer_two():

    cancerdf = answer_one()

    count = cancerdf.target.value_counts(ascending=True)

    count.index = 'malignant benign'.split()

    return count





answer_two()
def answer_three():

    cancerdf = answer_one()

    X = X = cancerdf.iloc[:, 0:-1]

    y = cancerdf.iloc[:,-1]

    # Your code here

    

    return X, y

answer_three()
from sklearn.model_selection import train_test_split



def answer_four():

    X, y = answer_three()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Your code here

    

    return X_train, X_test, y_train, y_test

answer_four()
from sklearn.neighbors import KNeighborsClassifier



def answer_five():

    X_train, X_test, y_train, y_test = answer_four()

    

    knn = KNeighborsClassifier(n_neighbors = 1)

    train = knn.fit(X_train, y_train)

    

    return knn# Return your answer

answer_five()
def answer_six():

    knn = answer_five()

    cancerdf = answer_one()

    means = cancerdf.mean()[:-1].values.reshape(1, -1)

    class_predict = knn.predict(means)

    # Your code here

    

    return class_predict# Return your answer

answer_six()
def answer_seven():

    X_train, X_test, y_train, y_test = answer_four()

    knn = answer_five()

    class_predict = knn.predict(X_test)

    class_predict.shape

    class_predict

    # Your code here

    

    return class_predict# Return your answer

answer_seven()
def answer_eight():

    X_train, X_test, y_train, y_test = answer_four()

    knn = answer_five()

    accuracy = knn.score(X_test, y_test)

    # Your code here

    

    return accuracy# Return your answer

answer_eight()
def accuracy_plot():

    import matplotlib.pyplot as plt



    %matplotlib notebook



    X_train, X_test, y_train, y_test = answer_four()



    # Find the training and testing accuracies by target value (i.e. malignant, benign)

    mal_train_X = X_train[y_train==0]

    mal_train_y = y_train[y_train==0]

    ben_train_X = X_train[y_train==1]

    ben_train_y = y_train[y_train==1]



    mal_test_X = X_test[y_test==0]

    mal_test_y = y_test[y_test==0]

    ben_test_X = X_test[y_test==1]

    ben_test_y = y_test[y_test==1]



    knn = answer_five()



    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 

              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]





    plt.figure()



    # Plot the scores as a bar chart

    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])



    # directly label the score onto the bars

    for bar in bars:

        height = bar.get_height()

        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 

                     ha='center', color='w', fontsize=11)



    # remove all the ticks (both axes), and tick labels on the Y axis

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')



    # remove the frame of the chart

    for spine in plt.gca().spines.values():

        spine.set_visible(False)



    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);

    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
accuracy_plot()