import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
print(os.listdir("../input/workbook-1-text-pre-processing-for-beginners/"))
train_data=pd.read_csv('../input/workbook-1-text-pre-processing-for-beginners/train_data.csv')
test_data=pd.read_csv('../input/workbook-1-text-pre-processing-for-beginners/test_data.csv')
print("Preprocessed Training Data: \n",train_data.head())
print("\n Preprocessed Test Data: \n",test_data.head())
print("Empty Comment Cells In Train: ",train_data['comment_text'].isna().sum())
print("Empty Comment Cells In Test: ",test_data['comment_text'].isna().sum())
train_drop_list=train_data[train_data.iloc[:,0].isna()]
train_drop_list_idx=train_drop_list.index
test_drop_list=test_data[test_data.iloc[:,0].isna()]
test_drop_list_idx=test_drop_list.index
print("Index of Empty Comment Cells in Train: \n",train_drop_list_idx)
print("Index of Empty Comment Cells in Test : \n",test_drop_list_idx)
train_data_new=train_data.drop(train_drop_list_idx,axis=0)
#test_data_new=test_data.drop(test_drop_list_idx,axis=0)
test_data_new=test_data
print("Verifying - Empty Comments After Removal: ")
print("Train: ",train_data_new['comment_text'].isna().sum())
print("Test: ",test_data_new['comment_text'].isna().sum())
print("Train Data shape --  Before drop: ",train_data.shape, "After Drop: ",train_data_new.shape )
print("Test Data shape --  Before drop: ",test_data.shape, "After Drop: ",test_data_new.shape )
#train, test = train_test_split(data_new, test_size=0.2,random_state=42)
train, test = train_data_new, test_data_new
test_comments=test.iloc[:,0]
train_comments=train.iloc[:,0]
test_labels=test.iloc[:,1:]
train_labels=train.iloc[:,1:]
print("Train Comments Shape : ",train_comments.shape)
print("Train Labels Shape :",train_labels.shape)
print("Test Comments Shape :",test_comments.shape)
#print("Test Labels Shape :",test_labels.shape)
vectorizer = CountVectorizer(analyzer = 'word',stop_words='english',max_features=10000)
train_comments_count=vectorizer.fit(train_comments).transform(train_comments)
print("Term Frequency Matrix(TF): \n",train_comments_count.toarray())
print("Verifying that TF is not empty by checking the sum ",train_comments_count.toarray().sum() )
tf_transformer = TfidfTransformer()
tf_transformer.fit(train_comments_count)
train_tfidf = tf_transformer.transform(train_comments_count)
print("Train TF-IDF Matrix Shape: ",train_tfidf.shape)
test_comments_count = vectorizer.transform(test_comments)
test_tfidf = tf_transformer.transform(test_comments_count)
print("Test TF-IDF Matrix Shape: ",test_tfidf.shape)
from scipy import sparse

train.to_csv('train.csv', index = False)
test.to_csv('test.csv', index = False)
sparse.save_npz("train_tfidf.npz", train_tfidf)
sparse.save_npz("test_tfidf.npz", test_tfidf)
