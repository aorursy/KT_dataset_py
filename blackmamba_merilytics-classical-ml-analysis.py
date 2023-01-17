import pandas as pd

import numpy as np
df = pd.read_csv("/kaggle/input/multiclass-text-classification/Merilytics_Clean.csv")
df.head(10)
def reduce_mem_usage(df: pd.DataFrame,

                     verbose: bool = True) -> pd.DataFrame:

    

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2



    for col in df.columns:

        col_type = df[col].dtypes



        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()



            if str(col_type)[:3] == 'int':



                if (c_min > np.iinfo(np.int32).min

                      and c_max < np.iinfo(np.int32).max):

                    df[col] = df[col].astype(np.int32)

                elif (c_min > np.iinfo(np.int64).min

                      and c_max < np.iinfo(np.int64).max):

                    df[col] = df[col].astype(np.int64)

            else:

                if (c_min > np.finfo(np.float16).min

                        and c_max < np.finfo(np.float16).max):

                    df[col] = df[col].astype(np.float16)

                elif (c_min > np.finfo(np.float32).min

                      and c_max < np.finfo(np.float32).max):

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    reduction = (start_mem - end_mem) / start_mem



    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'

    if verbose:

        print(msg)



    return df



df_reduced = reduce_mem_usage(df)
print("\n {} rows of data available\n\n".format(len(df_reduced)))

print("\n {} No of unique reviwers \n\n".format(len(df_reduced['review_id'].unique())))

print("\n No of null's in all columns \n\n",df_reduced.iloc[:,1:].isnull().sum())

print("\n Data per class of review rating \n\n",df_reduced['stars'].value_counts())
#Data Preperation

df_clean = df_reduced[df_reduced['review_id'].notnull() & df_reduced['text'].notnull()]
#Data looks clean enough for now

df_clean.isnull().sum()
df = (df_clean.groupby('stars')['text']

    .apply(lambda x: np.mean(x.str.len()))

    .reset_index(name='mean_len_text'))

df
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

import seaborn as sns



def get_pipeline(model):

    get_numeric_data = FunctionTransformer(lambda x: x[['cool','funny','useful']], validate=False)

    get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)



    pipeline = Pipeline([

        ('features', FeatureUnion([

            ('numeric_features', get_numeric_data),

             ('word_features', Pipeline([

                ('selector', get_text_data),

                ('vectorizer',TfidfVectorizer(ngram_range=(1,3),min_df=0.2,max_df=0.7))

            ]))

         ])),

        ('clf', model)

     ])

    

    return pipeline



def run_models(models,train_X,train_y,test_X,test_y):

    plt.figure(figsize=(6,6))

    folds = 3

    CV = StratifiedKFold(folds)

    cv_df = pd.DataFrame(index=range(folds * len(models)))

    entries = []

    #model_no = 1

    for model in models:

        pipeline = get_pipeline(model)

        model_name = model.__class__.__name__

        accuracies = cross_val_score(pipeline, X=train_X, y=train_y, scoring='accuracy', cv=CV.split(train_X, train_y))

        #print(accuracies)

        for fold_idx, accuracy in enumerate(accuracies):

            entries.append((model_name, fold_idx, accuracy))

        pipeline.fit(train_X,train_y)

        pred_y = pipeline.predict(test_X)

        print("------"*5)

        print("\nClassification Report on Test Data\n",classification_report(test_y,pred_y))

        print("Done with {}".format(model_name))

        print("******"*5)

        #model_no+=1

    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    sns.boxplot(x='model_name', y='accuracy', data=cv_df)

    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 

              size=8, jitter=True, edgecolor="gray", linewidth=2)

    plt.xticks(rotation=30)

    plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import LinearSVC

models = [

    RandomForestClassifier(random_state=0),

    LogisticRegression(random_state=0),

    MultinomialNB()

]
import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



# build train and test datasets

target = df_clean.stars.values

label_encoder = LabelEncoder()

target = label_encoder.fit_transform(target)



features = df_clean[['useful','funny','cool','text']].copy()

train_X,test_X,train_y,test_y = train_test_split(features,target,stratify=target)

print(f'Train Size : {len(train_X)} Test Size : {len(test_X)}')
import warnings

warnings.filterwarnings("ignore")

run_models(models,train_X,train_y,test_X,test_y)
from collections import Counter

def balance_classes(df,target):

    """Undersample to balance classes."""

    freqs = Counter(df[target])

    # the least common class is the maximum number we want for all classes

    max_allowable = freqs.most_common()[-1][1]

    print(f'frequencies {freqs} \nMax Alowable is {max_allowable}')

    balanced_df = pd.DataFrame()

    for i in freqs.keys():

        balanced_df = balanced_df.append(df[df['stars'] == i].iloc[:max_allowable,:])

    return balanced_df



balanced_df = balance_classes(df_clean, 'stars')

print(f'Balanced Dataset size {len(balanced_df)}')
target = balanced_df.stars.values

label_encoder = LabelEncoder()

target = label_encoder.fit_transform(target)

features = balanced_df[['useful','funny','cool','text']].copy()

train_X,test_X,train_y,test_y = train_test_split(features,target,stratify=target)

print(f'Train Size : {len(train_X)} Test Size : {len(test_X)}')
run_models(models,train_X,train_y,test_X,test_y)