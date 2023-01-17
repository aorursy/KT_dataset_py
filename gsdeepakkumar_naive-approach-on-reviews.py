

# Loading the required libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import itertools
warnings.filterwarnings('ignore')
review =pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")
review.head()
text = review[['Review Text','Rating']]
text.shape
text['Review Text'][0]
text[text['Review Text']==""]=np.NaN
text['Review Text'].fillna("No Review",inplace=True)
# Split into train and test data:
split = np.random.randn(len(text)) <0.8
train = text[split]
test = text[~split]
print("Total rows in train:",len(train),"and test:",len(test))
ytrain=train['Rating']
ytest=test['Rating']
lens=train['Review Text'].str.len()
print("Mean Length:",lens.mean(),"Standard Deviation",lens.std(),"Maximum Length",lens.max())


lens.hist()
plt.figure(figsize=(8,8))
text['Length']=lens
fx=sns.boxplot(x='Rating',y='Length',data=text)
plt.title("Distribution of length with respect to rating")
plt.xlabel("Rating")
plt.ylabel("Length")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss,confusion_matrix,classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re

count_vect = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english',max_features=5000)
count_vect.fit(list(train['Review Text'].values.astype('U'))+list(test['Review Text'].values.astype('U')))
xtrain=count_vect.transform(train['Review Text'].values.astype('U'))
xtest=count_vect.transform(test['Review Text'].values.astype('U'))

## Applying naive bayes:

model = MultinomialNB()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

### Lets check the accuracy score.
print(accuracy_score(ytest, predictions))
conf_matrix=confusion_matrix(ytest,predictions)
### Print confusion matrix:
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure(figsize=(8,8))
plot_confusion_matrix(conf_matrix, classes=['1', '2','3','4','5'],
                      title='Confusion matrix')
plt.show()