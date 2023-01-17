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
df = pd.read_csv(r'/kaggle/input/banking-complaints-data/Complaint_data.csv')
df.head(5)
df.groupby(['Product', 'Product_Encoding'])['Product_Encoding'].count()
df_comp = df[['Complaint','Product_Encoding']]

df_comp.columns  = ['text','label']



df_comp.head(5)
df_comp.drop_duplicates(subset=['text'],keep='first',inplace=True)

df_comp.info()
text = " ".join([x for x in df_comp.text if 'X' not in x ])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
#product 0 - Bank Account



text = " ".join([x for x in df_comp.text[df_comp.label ==0] if 'X' not in x ])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()



#product 1 - Credit Card



text = " ".join([x for x in df_comp.text[df_comp.label ==1] if 'X' not in x ])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()



#product 2  - Credit Reporting



text = " ".join([x for x in df_comp.text[df_comp.label ==2] if 'X' not in x ])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()



#product 3 - Mortgage



text = " ".join([x for x in df_comp.text[df_comp.label ==3] if 'X' not in x ])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()



#product 4 Student Loan



text = " ".join([x for x in df_comp.text[df_comp.label ==4] if 'X' not in x ])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()



df_comp_orig = df_comp.copy()




df_comp = df_comp_orig.sample(frac= 0.2)

df_comp.label.value_counts()
df_comp.shape
train_df,eval_df = train_test_split(df_comp,test_size = 0.3)
!pip install simpletransformers==0.32.3
from simpletransformers.classification import ClassificationModel





# Create a TransformerModel

model = ClassificationModel('bert', 'bert-base-cased', num_labels=5, args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)

#model = ClassificationModel('bert', 'bert-large-cased', num_labels=2, args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)
train_df2 = pd.DataFrame({

    'text': train_df['text'].replace(r'\n', ' ', regex=True),

    'label': train_df['label']

})



eval_df2 = pd.DataFrame({

    'text': eval_df['text'].replace(r'\n', ' ', regex=True),

    'label': eval_df['label']

})
%%time

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
product_list = ['Bank account or service','Credit card','Credit reporting','Mortgage','Student loan']
print(sklearn.metrics.classification_report(true,predicted,target_names=product_list))
sklearn.metrics.accuracy_score(true,predicted)
def get_result(statement):

    result = model.predict([statement])

    pos = np.where(result[1][0] == np.amax(result[1][0]))

    pos = int(pos[0])

    sentiment_dict = {0:'Bank account or service',1:'Credit card',2:'Credit reporting' ,3:'Mortgage' ,4: 'Student loan'}

    print(sentiment_dict[pos])

    return
get_result("Planning to get some loan for my Harvard law degree with low interest")