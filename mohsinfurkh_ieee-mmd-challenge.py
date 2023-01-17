import pandas as pd
import numpy as np
comments=pd.read_csv('../input/metoommd-dataset/Tweet_cmt.csv')
df=pd.read_csv('../input/metoommd-dataset/MeTooMMD_train.csv')
MeToo = df[df['TweetId'].isin(comments['Id'])]
MeToo.head()
comments.shape
comments.head()
MeToo_labels = MeToo[["Text_Only_Informative", "Image_Only_Informative", "Directed_Hate",\
                         "Generalized_Hate", "Sarcasm", "Allegation", "Justification", "Refutation", \
                        "Support", "Oppose"]]
MeToo_labels.shape
class_names = ["Text_Only_Informative", "Image_Only_Informative", "Directed_Hate",\
                         "Generalized_Hate", "Sarcasm", "Allegation", "Justification", "Refutation", \
                        "Support", "Oppose"]
import re
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

X = []
sentences = list(comments["Comments"])
for sen in sentences:
    X.append(preprocess_text(sen))
y = MeToo_labels.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.29, random_state=42)
np.shape(X_train)
from zeugma.embeddings import EmbeddingTransformer
gloves = EmbeddingTransformer('glove')
X_train_glove = gloves.transform(X_train)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
neighbors = list(range(1, 50, 2))
cv_scores = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_glove, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
import matplotlib.pyplot as plt
# changing to misclassification error
mse = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal number of neighbors is {}".format(optimal_k))

# plot misclassification error vs k
plt.plot(neighbors, mse)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Misclassification Error")
plt.show()
target_output=pd.read_csv('../input/metoommd-dataset/target.csv')
from sklearn.neighbors import KNeighborsRegressor
X_test_glove = gloves.transform(X_test)
score=[]
for class_name in class_names:
    train_target=target_output[class_name]
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X_train_glove, train_target)
    print('Training KNeighborRegressor for {} is complete!!'.format(class_name))
    submission=model.predict(X_test_glove)
    
    for pred in submission:
        if pred>=0.5:
            score.append(1)
            #out_put[class_name]=1
            
        else:
            score.append(0)
            #out_put[class_name]=0
newarr=np.reshape(score,(1992,10))
np.shape(newarr)
y = MeToo.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.29, random_state=42)
np.shape(y_test)
predictions = pd.DataFrame(newarr, columns = [["Text_Only_Informative", "Image_Only_Informative", "Directed_Hate",\
                         "Generalized_Hate", "Sarcasm", "Allegation", "Justification", "Refutation", \
                        "Support", "Oppose"]])
predictions.info()

tweetIds=[]
tweetIds=y_test[:,0]
tweetIds
tweetIds = tweetIds.astype('Int64')
predictions.insert(0, "TweetId", tweetIds , True)
import csv

with open('predictions.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
predictions.to_csv('../predict.csv', index=False)
predictions.info()
predictions.head()