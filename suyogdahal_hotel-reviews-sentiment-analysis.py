import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import re

import string



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from sklearn.utils.multiclass import unique_labels
train_df = pd.read_csv("../input/hotel-review/train.csv")

test_df = pd.read_csv("../input/hotel-review/test.csv")
#print five rows of the training data

train_df.head()
#print datatype of columns

train_df.info()
#display count, uniqiue count and the most frequent value in each column

train_df.describe().transpose()
#Display percentage of distribution of data between the two target classes



happy_percent = train_df['Is_Response'].value_counts()['happy']/train_df['Is_Response'].count()

not_happy_percent = train_df['Is_Response'].value_counts()['not happy']/train_df['Is_Response'].count()

print(f'Happy: {happy_percent*100}%\nNot Happy: {not_happy_percent*100}%')



sns.countplot(train_df['Is_Response'])
train_df.drop(columns=['User_ID', 'Browser_Used', 'Device_Used'], inplace=True)
def text_clean(text):

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\w*\d\w*', '', text)

    text = re.sub('[""''_]', '', text)

    text = re.sub('\n', '', text)

    return text
def decontract_text(text):

    """

    Decontract text

    """

    # specific

    text = re.sub(r"won\'t", "will not", text)

    text = re.sub(r"can\'t", "can not", text)

    text = re.sub(r"won\’t", "will not", text)

    text = re.sub(r"can\’t", "can not", text)

    text = re.sub(r"\'t've", " not have", text)

    text = re.sub(r"\'d've", " would have", text)

    text = re.sub(r"\'clock", "f the clock", text)

    text = re.sub(r"\'cause", " because", text)



    # general

    text = re.sub(r"n\'t", " not", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'s", " is", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'t", " not", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'m", " am", text)



    text = re.sub(r"n\’t", " not", text)

    text = re.sub(r"\’re", " are", text)

    text = re.sub(r"\’s", " is", text)

    text = re.sub(r"\’d", " would", text)

    text = re.sub(r"\’ll", " will", text)

    text = re.sub(r"\’t", " not", text)

    text = re.sub(r"\’ve", " have", text)

    text = re.sub(r"\’m", " am", text)

    

    return text
train_df['cleaned_description'] = train_df['Description'].apply(lambda x: decontract_text(x))

train_df['cleaned_description'] = train_df['cleaned_description'].apply(lambda x: text_clean(x))
print('Original Description:\n', train_df['Description'][0])

print('\n\nCleaned Description:\n', train_df['cleaned_description'][0])
x, y = train_df['cleaned_description'], train_df['Is_Response']



x_train, x_test, y_train, y_test = train_test_split(x, y,

                                                    test_size=0.1,

                                                    random_state=42)



print(f'x_train: {len(x_train)}')

print(f'x_test: {len(x_test)}')

print(f'y_train: {len(y_train)}')

print(f'y_test: {len(y_test)}')
tvec = TfidfVectorizer()

clf = LogisticRegression(solver='lbfgs', max_iter=1000)



model = Pipeline([('vectorizer', tvec), ('classifier', clf)])
model.fit(x_train, y_train)
y_pred = model.predict(x_test)



print(f'Accurcy: {accuracy_score(y_pred, y_test)}')

print(f'Precision: {precision_score(y_pred, y_test, average="weighted")}')

print(f'Recall: {recall_score(y_pred, y_test, average="weighted")}')
def print_confusion_matrix(confusion_matrix, class_names, figsize = (8,4), fontsize=12, model='clf'):

    """

    Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix,

    as a seaborn heatmap. 

    """

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    heatmap = sns.heatmap(df_cm, annot=True, ax=ax, fmt="d", cmap=plt.cm.Oranges)   

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

    plt.show()
conf_mat = confusion_matrix(y_test, y_pred)

uniq_labels = unique_labels(y_test, y_pred)



print_confusion_matrix(conf_mat, uniq_labels)