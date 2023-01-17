import numpy as np

import pandas as pd

from pathlib import Path



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score



from matplotlib import pyplot as plt

%config InlineBackend.figure_format = 'retina'
PATH_TO_DATA = Path('../input/hierarchical-text-classification/')
train_df = pd.read_csv(PATH_TO_DATA / 'train_40k.csv').fillna(' ')

valid_df = pd.read_csv(PATH_TO_DATA / 'val_10k.csv').fillna(' ')
train_df.head()
train_df.info()
train_df.loc[0, 'Text']
train_df.loc[0, 'Cat1'], train_df.loc[0, 'Cat2'], train_df.loc[0, 'Cat3']
train_df['Cat1'].value_counts()
train_df['Cat1_Cat2'] = train_df['Cat1'] + '/' + train_df['Cat2']

valid_df['Cat1_Cat2'] = valid_df['Cat1'] + '/' + valid_df['Cat2']
train_df['Cat1_Cat2'].nunique()
train_df['Cat1_Cat2'].value_counts().head()
# put a limit on maximal number of features and minimal word frequency

tf_idf = TfidfVectorizer(max_features=50000, min_df=2)

# multinomial logistic regression a.k.a softmax classifier

logit = LogisticRegression(C=1e2, n_jobs=4, solver='lbfgs', 

                           random_state=17, verbose=0, 

                           multi_class='multinomial',

                           fit_intercept=True)

# sklearn's pipeline

tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf), 

                                 ('logit', logit)])
%%time

tfidf_logit_pipeline.fit(train_df['Title'], train_df['Cat1_Cat2'])
%%time

valid_pred_level_2 = tfidf_logit_pipeline.predict(valid_df['Title'])
valid_pred_level_1 = [el.split('/')[0] for el in valid_pred_level_2]
print("Level 1:\n\tF1 micro (=accuracy): {}\n\tF1 weighted:\t      {}".format(

    f1_score(y_true=valid_df['Cat1'], y_pred=valid_pred_level_1, average='micro').round(3),

    f1_score(y_true=valid_df['Cat1'], y_pred=valid_pred_level_1, average='weighted').round(3)

    )

)
print("Level 2:\n\tF1 micro (=accuracy): {}\n\tF1 weighted:\t      {}".format(

    f1_score(y_true=valid_df['Cat1_Cat2'], y_pred=valid_pred_level_2, average='micro').round(3),

    f1_score(y_true=valid_df['Cat1_Cat2'], y_pred=valid_pred_level_2, average='weighted').round(3)

    )

)
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title='Confusion matrix', figsize=(7,7),

                          cmap=plt.cm.Blues, path_to_save_fig=None):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    import itertools

    cm = confusion_matrix(y_true, y_pred).T

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    plt.figure(figsize=figsize)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('Predicted label')

    plt.xlabel('True label')

    

    if path_to_save_fig:

        plt.savefig(path_to_save_fig, dpi=300, bbox_inches='tight')
plot_confusion_matrix(

    y_true=valid_df['Cat1'],

    y_pred=valid_pred_level_1, 

    classes=sorted(train_df['Cat1'].unique()),

    figsize=(8, 8)

)
%%capture

import eli5
eli5.show_weights(

    estimator=tfidf_logit_pipeline.named_steps['logit'],

    vec=tfidf_logit_pipeline.named_steps['tf_idf'])