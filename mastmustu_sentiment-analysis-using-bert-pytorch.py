# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#import seaborn as sns

from sklearn.model_selection import train_test_split

from wordcloud import WordCloud

df =pd.read_csv('/kaggle/input/sentiment-analysis-for-financial-news/all-data.csv' , header=None , encoding='ISO-8859-1')

df.columns =['sentiment' ,'statement']

df.head()
df.drop_duplicates(subset=['statement'],keep='first',inplace=True)

df.info()
text = " ".join([x for x in df.statement])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
## for neutral



text = " ".join([x for x in df.statement[df.sentiment=='neutral']])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
## for positive



text = " ".join([x for x in df.statement[df.sentiment=='positive']])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
## for negative



text = " ".join([x for x in df.statement[df.sentiment=='negative']])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
train_df,eval_df = train_test_split(df,test_size = 0.2)
!pip install simpletransformers
from simpletransformers.classification import ClassificationModel





# Create a TransformerModel

model = ClassificationModel('bert', 'bert-base-cased', num_labels=3, args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)
def making_label(st):

    if(st=='positive'):

        return 0

    elif(st=='neutral'):

        return 2

    else:

        return 1

    

train_df['label'] = train_df['sentiment'].apply(making_label)

eval_df['label'] = eval_df['sentiment'].apply(making_label)

print(train_df.head())
train_df2 = pd.DataFrame({

    'text': train_df['statement'].replace(r'\n', ' ', regex=True),

    'label': train_df['label']

})



eval_df2 = pd.DataFrame({

    'text': eval_df['statement'].replace(r'\n', ' ', regex=True),

    'label': eval_df['label']

})
model.train_model(train_df2)
result, model_outputs, wrong_predictions = model.eval_model(eval_df2)
print(result)

print(model_outputs)

#print(wrong_predictions)
lst = []

for arr in model_outputs:

    lst.append(np.argmax(arr))
true = eval_df2['label'].tolist()

predicted = lst
import sklearn

mat = sklearn.metrics.confusion_matrix(true , predicted)

mat
print(sklearn.metrics.classification_report(true,predicted,target_names=['positive','neutral','negative']))
sklearn.metrics.accuracy_score(true,predicted)
def get_result(statement):

    result = model.predict([statement])

    pos = np.where(result[1][0] == np.amax(result[1][0]))

    pos = int(pos[0])

    sentiment_dict = {0:'positive',1:'negative',2:'neutral'}

    print(sentiment_dict[pos])

    return
## neutral statement

get_result("According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .")
## positive statement

get_result("According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales .")
## negative statement

get_result('Sales in Finland decreased by 2.0 % , and international sales decreased by 9.3 % in terms of euros , and by 15.1 % in terms of local currencies .')
statement = "TCS records 15% YoY growth"

get_result(statement)